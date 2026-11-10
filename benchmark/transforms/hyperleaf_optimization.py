"""Fold-scoped normalization and regression losses for task 24.3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor

from benchmark.schema.canonical import hash_object


TARGET_NAMES = ("GrainWeight", "Gsw", "PhiPS2")
STANDARD_SPECTRAL_NORMALIZATION = "per-band-zscore-inner-train.v1"
VARIETY_ROBUST_SPECTRAL_NORMALIZATION = (
    "variety-balanced-median-mad-spectrum-inner-train.v1"
)
MSE_LOSS = "target-standardized-mse.v1"
HETEROSCEDASTIC_LOSS = "target-standardized-heteroscedastic-gaussian-nll.v1"
TARGET_BALANCED_LOSS = "inner-train-inverse-variance-target-balanced-mse.v1"


@dataclass(frozen=True)
class VarietyRobustSpectralState:
    """Immutable robust spectrum state fitted to one exact permitted boundary."""

    dataset_hash: str
    protocol_hash: str
    fold_id: str
    fit_sample_ids: tuple[str, ...]
    fit_variety_ids: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    epsilon: float
    config_hash: str
    state_hash: str

    def transform(self, spectra: Tensor) -> Tensor:
        if spectra.ndim != 2 or spectra.shape[1] != len(self.center):
            raise ValueError("spectra must have shape (samples, fitted_bands)")
        if not torch.is_floating_point(spectra) or not torch.isfinite(spectra).all():
            raise ValueError("spectra must be finite floating-point values")
        center = spectra.new_tensor(self.center)
        scale = spectra.new_tensor(self.scale)
        return (spectra - center) / scale


def _require_sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return normalized


def fit_variety_robust_spectral_normalization(
    spectra: Tensor,
    *,
    sample_ids: Sequence[str],
    variety_ids: Sequence[str],
    allowed_sample_ids: Iterable[str],
    dataset_hash: str,
    protocol_hash: str,
    fold_id: str,
    epsilon: float = 1.0e-6,
) -> VarietyRobustSpectralState:
    """Fit equal-variety median/MAD statistics without leaving the train boundary."""

    ids = tuple(str(value) for value in sample_ids)
    groups = tuple(str(value) for value in variety_ids)
    allowed = {str(value) for value in allowed_sample_ids}
    if spectra.ndim != 2 or spectra.shape[0] != len(ids) or len(ids) != len(groups):
        raise ValueError("spectra, sample_ids, and variety_ids must have equal sample counts")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("fit_sample_ids must be non-empty and unique")
    if set(ids) - allowed:
        raise ValueError("fit_sample_ids must remain inside the permitted training boundary")
    if any(not group for group in groups):
        raise ValueError("every fitted sample must have a variety ID")
    if not torch.is_floating_point(spectra) or not torch.isfinite(spectra).all():
        raise ValueError("spectra must be finite floating-point values")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    varieties = tuple(sorted(set(groups), key=lambda value: value.encode("utf-8")))
    per_variety_centers = torch.stack(
        tuple(
            spectra[
                torch.tensor(
                    [group == variety for group in groups],
                    dtype=torch.bool,
                    device=spectra.device,
                )
            ].median(dim=0).values
            for variety in varieties
        )
    )
    center = per_variety_centers.median(dim=0).values
    mad = (per_variety_centers - center).abs().median(dim=0).values * 1.4826
    pooled_mad = (spectra - spectra.median(dim=0).values).abs().median(dim=0).values * 1.4826
    scale = torch.where(mad > epsilon, mad, pooled_mad)
    scale = torch.where(scale > epsilon, scale, torch.ones_like(scale))

    config = {
        "normalization_key": VARIETY_ROBUST_SPECTRAL_NORMALIZATION,
        "estimator": "median_of_per_variety_medians_and_mad",
        "variety_weighting": "equal",
        "zero_scale_fallback": "pooled_mad_then_one",
        "epsilon": float(epsilon),
    }
    payload = {
        "dataset_hash": _require_sha256(dataset_hash, "dataset_hash"),
        "protocol_hash": _require_sha256(protocol_hash, "protocol_hash"),
        "fold_id": str(fold_id),
        "fit_sample_ids": tuple(sorted(ids, key=lambda value: value.encode("utf-8"))),
        "fit_variety_ids": varieties,
        "center": tuple(float(value) for value in center.detach().cpu().tolist()),
        "scale": tuple(float(value) for value in scale.detach().cpu().tolist()),
        "epsilon": float(epsilon),
        "config_hash": hash_object(config),
    }
    return VarietyRobustSpectralState(**payload, state_hash=hash_object(payload))


def heteroscedastic_regression_loss(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
    *,
    minimum_log_variance: float = -8.0,
    maximum_log_variance: float = 8.0,
) -> Tensor:
    """Gaussian negative log likelihood with predeclared variance bounds."""

    if mean.shape != target.shape or log_variance.shape != target.shape:
        raise ValueError("mean, log_variance, and target shapes must match")
    if minimum_log_variance >= maximum_log_variance:
        raise ValueError("minimum_log_variance must be below maximum_log_variance")
    if not all(torch.isfinite(value).all() for value in (mean, log_variance, target)):
        raise ValueError("heteroscedastic loss inputs must be finite")
    bounded = log_variance.clamp(minimum_log_variance, maximum_log_variance)
    return 0.5 * (torch.exp(-bounded) * (target - mean).square() + bounded).mean()


def fit_target_balance_weights(targets: Tensor, *, epsilon: float = 1.0e-6) -> Tensor:
    """Fit normalized inverse-variance target weights from inner-train targets."""

    if targets.ndim != 2 or targets.shape[1] != len(TARGET_NAMES) or targets.shape[0] < 2:
        raise ValueError("targets must have shape (at_least_two_samples, 3)")
    if not torch.is_floating_point(targets) or not torch.isfinite(targets).all():
        raise ValueError("targets must be finite floating-point values")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    variance = targets.var(dim=0, unbiased=False).clamp_min(epsilon)
    weights = variance.reciprocal()
    return weights / weights.mean()


def target_balanced_regression_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    target_weights: Tensor,
) -> Tensor:
    """Average per-target MSE under fixed positive inner-train weights."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching 2D shapes")
    if target_weights.shape != (prediction.shape[1],):
        raise ValueError("target_weights must contain one value per target")
    if not all(torch.isfinite(value).all() for value in (prediction, target, target_weights)):
        raise ValueError("target-balanced loss inputs must be finite")
    if torch.any(target_weights <= 0):
        raise ValueError("target_weights must be positive")
    normalized_weights = target_weights / target_weights.mean()
    per_target_mse = (prediction - target).square().mean(dim=0)
    return (per_target_mse * normalized_weights).mean()


__all__ = [
    "HETEROSCEDASTIC_LOSS",
    "MSE_LOSS",
    "STANDARD_SPECTRAL_NORMALIZATION",
    "TARGET_BALANCED_LOSS",
    "TARGET_NAMES",
    "VARIETY_ROBUST_SPECTRAL_NORMALIZATION",
    "VarietyRobustSpectralState",
    "fit_target_balance_weights",
    "fit_variety_robust_spectral_normalization",
    "heteroscedastic_regression_loss",
    "target_balanced_regression_loss",
]
