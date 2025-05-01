import torch
import torch.nn as nn
from torch.distributed import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.profiler import record_function

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.moe import MoE


class CUDATimer:
    def __init__(self) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._stop_events: list[torch.cuda.Event] = []

    def __enter__(self) -> "CUDATimer":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events.append(start)
        self._stop_events.append(stop)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._stop_events[-1].record()

    def __len__(self) -> int:
        return len(self._start_events)

    def get_time_list_s(self) -> list[float]:
        if not self._start_events:
            return [0.0]
        torch.cuda.synchronize()
        time_list_s = [
            start.elapsed_time(stop) / 1e3
            for start, stop in zip(self._start_events, self._stop_events)
        ]
        return time_list_s

    def get_total_time_s(self) -> float:
        return sum(self.get_time_list_s())

    def get_mean_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return sum(time_list_s) / len(time_list_s)

    def get_std_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return torch.tensor(time_list_s).std().item()

    def reset(self) -> None:
        self._start_events.clear()
        self._stop_events.clear()


def get_ep_mesh(ep_degree: int, world_size: int) -> DeviceMesh:
    # Cases:
    # 1. ep_degree = 1: full replication, no ep_mesh
    # 2. ep_degree = world_size: ep_mesh is the world
    # 3. world_size > ep_degree > world_size: 2D mesh, experts distributed along slice .
    if ep_degree == 1:
        ep_mesh = None
    elif ep_degree == world_size:
        ep_mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("inner",),
        )
    else:
        ep_mesh = init_device_mesh(
            "cuda",
            (world_size // ep_degree, ep_degree),
            mesh_dim_names=("outer", "inner"),
        )
    return ep_mesh


class SequentialProfileModule(nn.Module):
    def __init__(self, modules_list) -> None:
        super().__init__()
        self.layers = nn.ModuleList(modules_list)

    def forward(self, inputs):
        outputs = inputs
        for idx, module in enumerate(self.layers):
            with record_function(f"fwd_{idx}"):
                outputs = module(outputs)
        return outputs


def shard_sequential_model(
    seq_model: SequentialProfileModule,
    ep_degree: int,
    world_size: int,
    ep_mesh: DeviceMesh,
    fsdp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
) -> None:
    for moe in seq_model.layers:
        # Cases:
        # 1. ep_degree = 1: full replication, fully shard with the fsdp_mesh
        # 2. ep_degree = world_size: no expert replication at all. Ignore experts in fully_shard
        # 3. world_size > ep_degree > world_size: world_size // ep_degree expert replicas. Need
        #    to individually wrap experts using the ep_mesh because ModuleDict doesn't have a
        #    forward method.

        # The ignored_params arg requires torch nightly (> 2.6.0)
        ignored_params = set()
        if ep_degree == 1:
            pass
        elif ep_degree == world_size:
            # No replication in this case.
            ignored_params.add(moe.experts.parameters())
        else:
            # Don't reshard due to comms costs
            assert ep_mesh is not None
            fully_shard(
                moe.experts,
                mesh=ep_mesh["outer"],
                mp_policy=mp_policy,
                reshard_after_forward=False,
            )
            moe.experts.set_reshard_after_backward(False)
        fully_shard(
            moe,
            mesh=fsdp_mesh,
            ignored_params=ignored_params,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )
    # The root unit doesn't own any params, so manually force the first layer to not shard after
    # bwd
    seq_model.layers[0].set_reshard_after_backward(False)
    fully_shard(seq_model, mesh=fsdp_mesh, mp_policy=mp_policy)


def shard_full_model(
    model: MambaLMHeadModel,
    ep_degree: int,
    world_size: int,
    ep_mesh: DeviceMesh,
    fsdp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
) -> None:
    fully_shard(model.lm_head, mesh=fsdp_mesh, mp_policy=mp_policy)
    fully_shard(model.backbone.embedding, mesh=fsdp_mesh, mp_policy=mp_policy)
    # NOTE: @goon - model.backbone.layers is a module_dict on the MoE branch
    for idx, block in model.backbone.layers.items():
        # Cases:
        # 1. ep_degree = 1: full replication, fully shard with the fsdp_mesh
        # 2. ep_degree = world_size: no expert replication at all. Ignore experts in fully_shard
        # 3. world_size > ep_degree > world_size: world_size // ep_degree expert replicas.

        # The ignored_params arg requires torch nightly (> 2.6.0)
        ignored_params = set()
        if isinstance(block.mlp, MoE):
            if ep_degree == 1:
                pass
            elif ep_degree == world_size:
                # No replication in this case.
                ignored_params.add(block.mlp.experts.parameters())
            else:
                # Don't reshard due to comms costs
                fully_shard(
                    block.mlp.experts,
                    mesh=ep_mesh["outer"],
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                )
                block.mlp.experts.set_reshard_after_backward(False)
        is_not_last_block = int(idx) < len(model.backbone.layers) - 1
        fully_shard(
            block,
            mesh=fsdp_mesh,
            ignored_params=ignored_params,
            mp_policy=mp_policy,
            reshard_after_forward=is_not_last_block,
        )
    fully_shard(
        model,
        mesh=fsdp_mesh,
        reshard_after_forward=False,
        mp_policy=mp_policy,
    )
