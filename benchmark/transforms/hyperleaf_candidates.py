"""Deterministic, predeclared transforms for HyperLeaf candidate training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import Tensor
from torch.nn import functional as F


class GswLossKind(str, Enum):
    MSE = "mse"
    HUBER = "huber"


class GswTargetTransform(str, Enum):
    STANDARDIZED = "standardized"
    ASINH_005 = "asinh-scale-0.05"


@dataclass(frozen=True)
class HyperLeafStyleConfig:
    """One immutable style augmentation setting declared before screening."""

    key: str
    application_probability: float
    rgb_gain: float
    rgb_bias: float
    rgb_noise: float
    spectrum_gain: float
    spectrum_noise: float
    spectrum_drop_probability: float
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("style key must be non-empty")
        probability_fields = (
            self.application_probability,
            self.spectrum_drop_probability,
            self.horizontal_flip_probability,
            self.vertical_flip_probability,
        )
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probability_fields):
            raise ValueError("style probabilities must be finite values in [0, 1]")
        magnitude_fields = (
            self.rgb_gain,
            self.rgb_bias,
            self.rgb_noise,
            self.spectrum_gain,
            self.spectrum_noise,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in magnitude_fields):
            raise ValueError("style magnitudes must be finite and non-negative")


NO_STYLE = HyperLeafStyleConfig("none", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
MILD_STYLE = HyperLeafStyleConfig("mild", 1.0, 0.10, 0.05, 0.015, 0.05, 0.02, 0.00)
STRONG_STYLE = HyperLeafStyleConfig("strong", 1.0, 0.25, 0.12, 0.030, 0.12, 0.04, 0.05)
MIXED_STYLE_P01 = HyperLeafStyleConfig("mixed-p01", 0.10, 0.25, 0.12, 0.030, 0.12, 0.04, 0.05)
MIXED_STYLE_P03 = HyperLeafStyleConfig("mixed-p03", 0.30, 0.25, 0.12, 0.030, 0.12, 0.04, 0.05)

HYPERLEAF_STYLE_CONFIGS = {
    item.key: item
    for item in (NO_STYLE, MILD_STYLE, STRONG_STYLE, MIXED_STYLE_P01, MIXED_STYLE_P03)
}


def apply_hyperleaf_style(
    visible_rgb: Tensor,
    global_spectrum: Tensor,
    *,
    config: HyperLeafStyleConfig,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create a deterministic style view without changing source tensors."""

    if visible_rgb.ndim != 4 or visible_rgb.shape[1] != 3:
        raise ValueError("visible_rgb must have shape (batch, 3, height, width)")
    if global_spectrum.ndim != 2:
        raise ValueError("global_spectrum must have shape (batch, bands)")
    if visible_rgb.shape[0] != global_spectrum.shape[0]:
        raise ValueError("RGB and spectrum batches must have equal length")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not torch.is_floating_point(visible_rgb) or not torch.is_floating_point(global_spectrum):
        raise TypeError("style augmentation requires floating-point tensors")
    if not torch.isfinite(visible_rgb).all() or not torch.isfinite(global_spectrum).all():
        raise ValueError("style augmentation inputs must be finite")

    if config.application_probability == 0.0:
        return visible_rgb.clone(), global_spectrum.clone()

    generator = torch.Generator(device=visible_rgb.device)
    generator.manual_seed(seed)
    batch = visible_rgb.shape[0]
    apply_mask = torch.rand(batch, generator=generator, device=visible_rgb.device) < config.application_probability

    rgb_gain = 1.0 + (torch.rand(batch, 1, 1, 1, generator=generator, device=visible_rgb.device) * 2.0 - 1.0) * config.rgb_gain
    rgb_bias = (torch.rand(batch, 3, 1, 1, generator=generator, device=visible_rgb.device) * 2.0 - 1.0) * config.rgb_bias
    rgb_noise = torch.randn(visible_rgb.shape, generator=generator, device=visible_rgb.device, dtype=visible_rgb.dtype) * config.rgb_noise
    styled_rgb = (visible_rgb * rgb_gain + rgb_bias + rgb_noise).clamp(0.0, 1.0)

    horizontal = torch.rand(batch, generator=generator, device=visible_rgb.device) < config.horizontal_flip_probability
    vertical = torch.rand(batch, generator=generator, device=visible_rgb.device) < config.vertical_flip_probability
    styled_rgb = torch.where(horizontal[:, None, None, None], styled_rgb.flip(-1), styled_rgb)
    styled_rgb = torch.where(vertical[:, None, None, None], styled_rgb.flip(-2), styled_rgb)

    spectrum_generator = torch.Generator(device=global_spectrum.device)
    spectrum_generator.manual_seed(seed + 1)
    spectrum_gain = 1.0 + (torch.rand(batch, 1, generator=spectrum_generator, device=global_spectrum.device) * 2.0 - 1.0) * config.spectrum_gain
    spectrum_noise = torch.randn(global_spectrum.shape, generator=spectrum_generator, device=global_spectrum.device, dtype=global_spectrum.dtype) * config.spectrum_noise
    styled_spectrum = global_spectrum * spectrum_gain + spectrum_noise
    drops = torch.rand(global_spectrum.shape, generator=spectrum_generator, device=global_spectrum.device) < config.spectrum_drop_probability
    styled_spectrum = styled_spectrum.masked_fill(drops, 0.0)

    rgb_result = torch.where(apply_mask[:, None, None, None], styled_rgb, visible_rgb)
    spectrum_result = torch.where(apply_mask.to(global_spectrum.device)[:, None], styled_spectrum, global_spectrum)
    return rgb_result, spectrum_result


def transform_gsw_target(values: Tensor, transform: GswTargetTransform) -> Tensor:
    """Apply one frozen Gsw target transform to normalized target values."""

    if transform is GswTargetTransform.STANDARDIZED:
        return values
    if transform is GswTargetTransform.ASINH_005:
        return torch.asinh(values / 0.05)
    raise ValueError(f"unsupported Gsw target transform: {transform!r}")


def inverse_gsw_target(values: Tensor, transform: GswTargetTransform) -> Tensor:
    """Invert a frozen Gsw transform before metric computation."""

    if transform is GswTargetTransform.STANDARDIZED:
        return values
    if transform is GswTargetTransform.ASINH_005:
        return torch.sinh(values) * 0.05
    raise ValueError(f"unsupported Gsw target transform: {transform!r}")


def gsw_regression_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_kind: GswLossKind,
    huber_delta: float = 1.0,
) -> Tensor:
    """Evaluate one predeclared Gsw loss; target transformation is separate."""

    if prediction.shape != target.shape:
        raise ValueError("Gsw prediction and target shapes must match")
    if loss_kind is GswLossKind.MSE:
        return F.mse_loss(prediction, target)
    if loss_kind is GswLossKind.HUBER:
        if not math.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        return F.huber_loss(prediction, target, delta=huber_delta)
    raise ValueError(f"unsupported Gsw loss: {loss_kind!r}")


__all__ = [
    "GswLossKind",
    "GswTargetTransform",
    "HYPERLEAF_STYLE_CONFIGS",
    "HyperLeafStyleConfig",
    "MILD_STYLE",
    "MIXED_STYLE_P01",
    "MIXED_STYLE_P03",
    "NO_STYLE",
    "STRONG_STYLE",
    "apply_hyperleaf_style",
    "gsw_regression_loss",
    "inverse_gsw_target",
    "transform_gsw_target",
]
