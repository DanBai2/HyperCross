"""Strict schemas for prediction-first metric computation."""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import Field, StrictInt, field_validator, model_validator

from .base import ContentHash, GroupId, Identifier, TaskKind
from .canonical import canonical_serialize, normalize_id
from .environment import FrozenSchemaModel


UNDEFINED_SINGLE_CLASS_AUROC = "single_class_fold"
UNDEFINED_ZERO_VARIANCE_R2 = "zero_variance_target"


def _validate_label(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{field_name} must be a scalar JSON label")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _label_identity(value: Any) -> bytes:
    """Return a typed identity so bool, integer, and string labels do not collide."""

    return canonical_serialize(value)


class MetricSchema(FrozenSchemaModel):
    """Frozen metric configuration with an explicit numeric tolerance."""

    schema_version: Identifier = "1.0"
    task_kind: TaskKind
    numeric_tolerance: float = Field(gt=0.0)
    target_names: Tuple[Identifier, ...] = ()
    negative_label: Any = 0
    positive_label: Any = 1
    probability_class_order: Tuple[Any, ...] = ()
    class_labels: Tuple[Any, ...] = ()
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ece_bins: StrictInt = Field(default=10, gt=0)

    @field_validator("target_names", mode="before")
    @classmethod
    def normalize_target_names(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("target_names must be a collection")
        return tuple(normalize_id(item) for item in value)

    @field_validator("negative_label", "positive_label")
    @classmethod
    def validate_binary_labels(cls, value: Any, info: Any) -> Any:
        return _validate_label(value, field_name=info.field_name)

    @field_validator("probability_class_order", "class_labels")
    @classmethod
    def validate_label_collections(cls, value: Tuple[Any, ...], info: Any) -> Tuple[Any, ...]:
        checked = tuple(
            _validate_label(item, field_name=info.field_name) for item in value
        )
        identities = tuple(_label_identity(item) for item in checked)
        if len(set(identities)) != len(identities):
            raise ValueError(f"{info.field_name} must not contain duplicate labels")
        return checked

    @model_validator(mode="after")
    def validate_task_configuration(self) -> "MetricSchema":
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target_names must not contain duplicates")

        if self.task_kind is TaskKind.MULTI_TARGET_REGRESSION:
            if not self.target_names:
                raise ValueError("regression metric schema requires target_names")
        elif self.task_kind is TaskKind.BINARY_CLASSIFICATION:
            if _label_identity(self.negative_label) == _label_identity(self.positive_label):
                raise ValueError("negative_label and positive_label must differ")
            if self.probability_class_order:
                expected = {
                    _label_identity(self.negative_label),
                    _label_identity(self.positive_label),
                }
                observed = {_label_identity(item) for item in self.probability_class_order}
                if len(self.probability_class_order) != 2 or observed != expected:
                    raise ValueError(
                        "probability_class_order must contain the negative and positive labels"
                    )
        elif self.task_kind is TaskKind.REMOTE_CLASSIFICATION:
            if len(self.class_labels) < 2:
                raise ValueError("remote metric schema requires at least two class_labels")
        return self


class MetricValue(FrozenSchemaModel):
    """One scalar metric, including a reason when its value is undefined."""

    value: Optional[float] = None
    reason: Optional[Identifier] = None

    @model_validator(mode="after")
    def validate_defined_or_reasoned(self) -> "MetricValue":
        if (self.value is None) == (self.reason is None):
            raise ValueError("metric must contain exactly one of value or reason")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        return self


class MetricSet(FrozenSchemaModel):
    """Metrics derived from one immutable prediction artifact."""

    metric_schema: MetricSchema
    source_prediction_hash: ContentHash
    metrics: Dict[Identifier, MetricValue]
    confusion_matrices: Dict[Identifier, Tuple[Tuple[StrictInt, ...], ...]] = Field(
        default_factory=dict
    )

    @field_validator("metrics")
    @classmethod
    def require_metrics(cls, value: Dict[str, MetricValue]) -> Dict[str, MetricValue]:
        if not value:
            raise ValueError("metrics must not be empty")
        return dict(sorted(value.items(), key=lambda item: item[0].encode("utf-8")))

    @field_validator("confusion_matrices")
    @classmethod
    def validate_confusion_matrices(
        cls, value: Dict[str, Tuple[Tuple[int, ...], ...]]
    ) -> Dict[str, Tuple[Tuple[int, ...], ...]]:
        ordered: Dict[str, Tuple[Tuple[int, ...], ...]] = {}
        for name, matrix in sorted(value.items(), key=lambda item: item[0].encode("utf-8")):
            if not matrix or any(not row for row in matrix):
                raise ValueError("confusion matrices must not be empty")
            width = len(matrix[0])
            if len(matrix) != width or any(len(row) != width for row in matrix):
                raise ValueError("confusion matrices must be square")
            if any(cell < 0 for row in matrix for cell in row):
                raise ValueError("confusion matrix counts must be non-negative")
            ordered[name] = matrix
        return ordered

    def metric(self, name: str) -> MetricValue:
        """Return a named scalar metric."""

        return self.metrics[name]


class ConfidenceInterval(FrozenSchemaModel):
    """Auditable fixed-policy patient bootstrap confidence interval."""

    metric_name: Identifier
    source_prediction_hash: ContentHash
    point_estimate: float
    lower_bound: float
    upper_bound: float
    resampling_unit: Literal["patient_group_id"] = "patient_group_id"
    resamples: Literal[10000] = 10000
    confidence_level: Literal[0.95] = 0.95
    sidedness: Literal["two_sided"] = "two_sided"
    seed: StrictInt

    @model_validator(mode="after")
    def validate_interval(self) -> "ConfidenceInterval":
        if not self.lower_bound <= self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        return self


class StatisticalComparison(FrozenSchemaModel):
    """Auditable paired candidate-minus-control patient comparison."""

    metric_name: Identifier
    candidate_prediction_hash: ContentHash
    control_prediction_hash: ContentHash
    candidate_estimate: float
    control_estimate: float
    estimated_difference: float
    lower_bound: float
    upper_bound: float
    common_group_ids: Tuple[GroupId, ...] = Field(min_length=1)
    resampling_unit: Literal["patient_group_id"] = "patient_group_id"
    resamples: Literal[10000] = 10000
    confidence_level: Literal[0.95] = 0.95
    sidedness: Literal["two_sided"] = "two_sided"
    seed: StrictInt

    @field_validator("common_group_ids", mode="before")
    @classmethod
    def canonicalize_common_groups(
        cls, value: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        normalized = tuple(normalize_id(item) for item in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("common_group_ids must not contain duplicates")
        return tuple(sorted(normalized, key=lambda item: item.encode("utf-8")))

    @model_validator(mode="after")
    def validate_comparison(self) -> "StatisticalComparison":
        if not self.lower_bound <= self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        return self


__all__ = [
    "ConfidenceInterval",
    "MetricSchema",
    "MetricSet",
    "MetricValue",
    "StatisticalComparison",
    "UNDEFINED_SINGLE_CLASS_AUROC",
    "UNDEFINED_ZERO_VARIANCE_R2",
]
