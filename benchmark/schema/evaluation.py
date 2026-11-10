"""Immutable capabilities and outputs for one-shot outer evaluation."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

from pydantic import Field, StrictInt, field_validator, model_validator
from typing_extensions import Annotated

from .base import ContentHash, FoldId, Identifier, ProtocolKey, SampleId, UtcTimestamp
from .canonical import normalize_id
from .environment import FrozenSchemaModel
from .preprocessing import canonical_sample_ids
from .records import ArtifactReference


PositiveStrictInt = Annotated[StrictInt, Field(gt=0)]
PredictionScalar = Union[int, float, str, bool]
PredictionValue = Union[PredictionScalar, Tuple[PredictionScalar, ...]]
ProbabilityValue = Union[float, Tuple[float, ...]]


def _prediction_value(value: Any, *, field_name: str) -> PredictionValue:
    if isinstance(value, list):
        value = tuple(value)
    values = value if isinstance(value, tuple) else (value,)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    checked = []
    for item in values:
        if not isinstance(item, (str, bool, int, float)):
            raise ValueError(f"{field_name} must contain scalar JSON values")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{field_name} must contain only finite numbers")
        checked.append(item)
    return tuple(checked) if isinstance(value, tuple) else checked[0]


class FrozenOuterCheckpoint(FrozenSchemaModel):
    """Immutable descriptor for an outer-train checkpoint ready for inference."""

    artifact: ArtifactReference
    fold_id: FoldId
    seed: StrictInt
    inference_branch: Identifier
    selection_decision_hash: ContentHash
    frozen_utc: UtcTimestamp

    @field_validator("fold_id", "inference_branch", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> Any:
        return normalize_id(value) if isinstance(value, str) else value


class EvaluationLease(FrozenSchemaModel):
    """Opaque reservation capability issued by one OuterEvaluationGate instance.

    Constructing an equal value does not grant label access. The issuing gate
    also checks object identity and its private lease ledger before every use.
    """

    lease_id: Identifier
    protocol_key: ProtocolKey
    protocol_hash: ContentHash
    requested_count: PositiveStrictInt
    reserved_utc: UtcTimestamp

    @field_validator("lease_id", "protocol_key", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> Any:
        return normalize_id(value) if isinstance(value, str) else value


class OuterPrediction(FrozenSchemaModel):
    """A label-free model output produced from one frozen checkpoint."""

    y_pred: Any
    probability: Optional[Any] = None
    target_name: Optional[Identifier] = None

    @field_validator("y_pred")
    @classmethod
    def validate_prediction(cls, value: Any) -> PredictionValue:
        return _prediction_value(value, field_name="y_pred")

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, value: Any) -> Optional[ProbabilityValue]:
        if value is None:
            return None
        if isinstance(value, list):
            value = tuple(value)
        values = value if isinstance(value, tuple) else (value,)
        if not values:
            raise ValueError("probability must not be empty")
        checked = []
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("probability must contain numeric values")
            probability = float(item)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("probability values must be finite and within [0, 1]")
            checked.append(probability)
        return tuple(checked) if isinstance(value, tuple) else checked[0]

    @model_validator(mode="after")
    def validate_probability_shape(self) -> "OuterPrediction":
        if isinstance(self.probability, tuple) and isinstance(self.y_pred, tuple):
            if len(self.probability) != len(self.y_pred):
                raise ValueError(
                    "probability and vector prediction must have equal length"
                )
        return self


__all__ = ["EvaluationLease", "FrozenOuterCheckpoint", "OuterPrediction"]
