import pytest
import torch

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


def mean_loss_fn(tensor: torch.Tensor) -> torch.Tensor:
    """
    Dummy loss function which does a mean over the batch dim and tries to make the grads not too big
    or small.
    """
    return tensor.pow(2).mean()
