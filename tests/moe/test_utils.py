import pytest
import torch
import torch.nn.functional as F

from mamba_ssm.modules.moe import (
    RoutedExpertsNoEPGroupedMM,
    RoutedExpertsNoEPGroupedMMTriton,
    RoutedExpertsTorchEPGroupedMM,
    RoutedExpertsTorchEPGroupedMMTriton,
)

H100_CLASSES = (
    RoutedExpertsNoEPGroupedMM,
    RoutedExpertsNoEPGroupedMMTriton,
    RoutedExpertsTorchEPGroupedMM,
    RoutedExpertsTorchEPGroupedMMTriton,
)


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def skip_if_no_h100s() -> None:
    if not has_cuda_capability(9, 0):
        pytest.skip("Requires H100s")


def skip_moe_impl_if_no_h100s(moe_impl: str) -> None:
    if moe_impl in H100_CLASSES:
        skip_if_no_h100s()


def mean_loss_fn(input: torch.Tensor) -> torch.Tensor:
    """
    Dummy loss function which does a mean over the batch dim and tries to make the grads not too big
    or small.
    """
    return input.mean(0).sum()


def flattened_cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(input.view(-1, input.size(-1)), target.view(-1).long())


@torch.no_grad
def assert_close(
    actual: torch.Tensor, expected: torch.Tensor, tol: float = 1e-1
) -> None:
    """
    Check that actual and expected are close relative to the typical variation in expected.
    """
    expected_variation = expected.abs().mean()
    diff_variation = (actual - expected).abs().mean()
    assert diff_variation < tol * expected_variation, (
        f"{diff_variation=} is not {tol=} smaller than {expected_variation=}"
    )
