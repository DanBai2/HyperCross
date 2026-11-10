"""Fold-scoped preprocessing and feature cache interfaces."""

from benchmark.transforms.feature_cache import FeatureCache, FeatureCacheValidationError
from benchmark.transforms.hyperleaf_candidates import (
    GswLossKind,
    GswTargetTransform,
    HYPERLEAF_STYLE_CONFIGS,
    HyperLeafStyleConfig,
    MILD_STYLE,
    MIXED_STYLE_P01,
    MIXED_STYLE_P03,
    NO_STYLE,
    STRONG_STYLE,
    apply_hyperleaf_style,
    gsw_regression_loss,
    inverse_gsw_target,
    transform_gsw_target,
)
from benchmark.transforms.preprocessing import (
    PreprocessingManager,
    PreprocessingValidationError,
)

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
    "FeatureCache",
    "FeatureCacheValidationError",
    "PreprocessingManager",
    "PreprocessingValidationError",
]
