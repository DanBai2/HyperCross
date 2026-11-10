"""Frozen aggregation policy used for prediction-first aggregate records."""

from __future__ import annotations

from typing import Any, Literal, Tuple

from pydantic import Field, StrictInt, field_validator

from .base import FoldId, Identifier
from .canonical import normalize_id
from .environment import FrozenSchemaModel
from .metrics import MetricSchema


class AggregationPolicy(FrozenSchemaModel):
    """All choices that affect aggregate point estimates and variances.

    The declared seed/fold Cartesian product is also the required aggregate
    membership. Population variance is explicit so later metric engines cannot
    silently change the estimator while reusing the same policy identity.
    """

    policy_key: Identifier
    metric_schema: MetricSchema
    variance_metric: Identifier
    seed_set: Tuple[StrictInt, ...] = Field(min_length=1)
    fold_ids: Tuple[FoldId, ...] = Field(min_length=1)
    variance_estimator: Literal["population"] = "population"

    @field_validator("seed_set")
    @classmethod
    def canonicalize_seed_set(cls, value: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("seed_set must not contain duplicates")
        return tuple(sorted(value))

    @field_validator("fold_ids", mode="before")
    @classmethod
    def canonicalize_fold_ids(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("fold_ids must be a collection of Fold_IDs")
        folds = tuple(normalize_id(fold_id) for fold_id in value)
        if len(set(folds)) != len(folds):
            raise ValueError("fold_ids must not contain duplicates")
        return tuple(sorted(folds, key=lambda fold_id: fold_id.encode("utf-8")))


__all__ = ["AggregationPolicy"]
