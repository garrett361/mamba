from dtest import DTest
import torch
import torch.distributed as dist
from mamba_ssm.modules.mamba2_cp import RingCommsFn


class TestCausalRingCommsFn(DTest):
    def test_fwd(self):
        torch.manual_seed(42)
        dim = 16
        t_send = torch.randn(
            self.world_size,
            dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        group = dist.group.WORLD
        t_recv = RingCommsFn.apply(t_send[self.rank], group)

        torch.testing.assert_close(t_recv, t_send[(self.rank - 1) % self.world_size])

    def test_bwd(self):
        torch.manual_seed(42)
        dim = 16
        t_send = torch.randn(
            self.world_size,
            dim,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        group = dist.group.WORLD
        t_recv = RingCommsFn.apply(t_send[self.rank], group)
        t_recv.pow(2).div(2).sum().backward()

        grad = t_send.grad[self.rank]
        torch.testing.assert_close(grad, t_send[self.rank])

        other_idxs = torch.arange(self.world_size, device=self.device) != self.rank
        zero_grads = t_send.grad[other_idxs]
        torch.testing.assert_close(zero_grads, torch.zeros_like(zero_grads))
