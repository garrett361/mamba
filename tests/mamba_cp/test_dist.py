from dtest import DTest
import torch
import torch.distributed as dist
from mamba_ssm.modules.mamba2_cp import RingCommsFn


class TestCausalRingCommsFn(DTest):
    def test_fwd(self):
        torch.manual_seed(42)
        dim = 16
        t_send = torch.randn(
            self.get_world_size(),
            dim,
            device=self.get_device(),
            dtype=torch.bfloat16,
        )
        group = dist.group.WORLD
        t_recv = RingCommsFn.apply(t_send[self.get_rank()], group)

        torch.testing.assert_close(
            t_recv, t_send[(self.get_rank() - 1) % self.get_world_size()]
        )

    def test_bwd(self):
        torch.manual_seed(42)
        dim = 16
        t_send = torch.randn(
            self.get_world_size(),
            dim,
            device=self.get_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        group = dist.group.WORLD
        t_recv = RingCommsFn.apply(t_send[self.get_rank()], group)
        t_recv.pow(2).div(2).sum().backward()

        grad = t_send.grad[self.get_rank()]
        torch.testing.assert_close(grad, t_send[self.get_rank()])

        other_idxs = (
            torch.arange(self.get_world_size(), device=self.get_device())
            != self.get_rank()
        )
        zero_grads = t_send.grad[other_idxs]
        torch.testing.assert_close(zero_grads, torch.zeros_like(zero_grads))
