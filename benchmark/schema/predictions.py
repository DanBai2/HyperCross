"""Immutable, hash-bound schemas for per-sample prediction artifacts."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Tuple, Union

from pydantic import Field, StrictInt, field_validator, model_validator

from .base import ArtifactId, ContentHash, FoldId, Identifier, RunId, SampleId
from .canonical import hash_object, normalize_id
from .environment import FrozenSchemaModel


PredictionScalar = Union[int, float, str, bool]
PredictionValue = Union[PredictionScalar, Tuple[PredictionScalar, ...]]
ProbabilityValue = Union[float, Tuple[float, ...]]


def _validate_scalar(value: Any, *, field_name: str) -> PredictionScalar:
    if not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{field_name} must contain scalar JSON values")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must contain only finite numbers")
    return value


def _validate_prediction_value(value: Any, *, field_name: str) -> PredictionValue:
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple):
        if not value:
            raise ValueError(f"{field_name} vector must not be empty")
        return tuple(
            _validate_scalar(item, field_name=field_name) for item in value
        )
    return _validate_scalar(value, field_name=field_name)


def _validate_probability(value: Any) -> Optional[ProbabilityValue]:
    if value is None:
        return None
    if isinstance(value, list):
        value = tuple(value)
    values = value if isinstance(value, tuple) else (value,)
    if not values:
        raise ValueError("probability vector must not be empty")
    checked = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("probability must contain only numeric values")
        probability = float(item)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability values must be finite and within [0, 1]")
        checked.append(probability)
    return tuple(checked) if isinstance(value, tuple) else checked[0]


def canonical_prediction_key(key: "PredictionKey") -> tuple[bytes, bytes, int, bytes]:
    """Return the stable ordering key used by shards and complete artifacts."""

    return (
        normalize_id(key.sample_id).encode("utf-8"),
        normalize_id(key.fold_id).encode("utf-8"),
        key.seed,
        normalize_id(key.inference_branch).encode("utf-8"),
    )


def prediction_key_text(key: "PredictionKey") -> str:
    """Return an unambiguous machine-readable duplicate-key description."""

    return (
        f"sample_id={key.sample_id!r},fold_id={key.fold_id!r},"
        f"seed={key.seed},inference_branch={key.inference_branch!r}"
    )


class PredictionKey(FrozenSchemaModel):
    """The four-part identity of one outer-evaluation prediction."""

    sample_id: SampleId
    fold_id: FoldId
    seed: StrictInt
    inference_branch: Identifier

    @field_validator("sample_id", "fold_id", "inference_branch", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_id(value)
        return value


class PredictionRecord(FrozenSchemaModel):
    """Truth and model output for exactly one prediction key.

    Scalar values support classification and one-target regression. Non-empty
    tuples support multi-target regression and multiclass probability vectors.
    """

    key: PredictionKey
    target_name: Optional[Identifier] = None
    y_true: Any
    y_pred: Any
    probability: Optional[Any] = None

    @field_validator("target_name", mode="before")
    @classmethod
    def normalize_target_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_id(value)
        return value

    @field_validator("y_true", "y_pred")
    @classmethod
    def validate_values(cls, value: Any, info: Any) -> PredictionValue:
        return _validate_prediction_value(value, field_name=info.field_name)

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, value: Any) -> Optional[ProbabilityValue]:
        return _validate_probability(value)

    @model_validator(mode="after")
    def validate_value_shapes(self) -> "PredictionRecord":
        true_is_vector = isinstance(self.y_true, tuple)
        pred_is_vector = isinstance(self.y_pred, tuple)
        if true_is_vector != pred_is_vector:
            raise ValueError("y_true and y_pred must both be scalar or both be vectors")
        if true_is_vector and len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred vectors must have equal length")
        return self


class PredictionShard(FrozenSchemaModel):
    """One bounded batch of prediction records supplied to the publisher."""

    shard_id: Identifier
    records: Tuple[PredictionRecord, ...] = Field(min_length=1)

    @field_validator("records")
    @classmethod
    def canonicalize_records(
        cls, records: Tuple[PredictionRecord, ...]
    ) -> Tuple[PredictionRecord, ...]:
        return tuple(sorted(records, key=lambda record: canonical_prediction_key(record.key)))


class PredictionArtifact(FrozenSchemaModel):
    """Canonical complete per-sample predictions for one evaluation context."""

    schema_version: Identifier = "1.0"
    artifact_id: ArtifactId
    run_id: RunId
    records: Tuple[PredictionRecord, ...] = Field(min_length=1)
    expected_sample_ids: Tuple[SampleId, ...] = Field(min_length=1)
    content_hash: ContentHash
    complete: bool

    @field_validator("expected_sample_ids", mode="before")
    @classmethod
    def canonicalize_expected_ids(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("expected_sample_ids must be a collection of Sample_IDs")
        try:
            normalized = tuple(normalize_id(item) for item in value)
        except TypeError as exc:
            raise ValueError(
                "expected_sample_ids must be a collection of Sample_IDs"
            ) from exc
        if not normalized:
            raise ValueError("expected_sample_ids must not be empty")
        duplicates = tuple(
            sorted(
                (item for item, count in Counter(normalized).items() if count > 1),
                key=lambda item: item.encode("utf-8"),
            )
        )
        if duplicates:
            raise ValueError(
                "expected_sample_ids contains duplicates after normalization: "
                + ", ".join(duplicates)
            )
        return tuple(sorted(normalized, key=lambda item: item.encode("utf-8")))

    @field_validator("records")
    @classmethod
    def canonicalize_records(
        cls, records: Tuple[PredictionRecord, ...]
    ) -> Tuple[PredictionRecord, ...]:
        return tuple(sorted(records, key=lambda record: canonical_prediction_key(record.key)))

    @classmethod
    def create(cls, **values: Any) -> "PredictionArtifact":
        """Create an artifact whose hash covers all fields except the self hash."""

        payload = dict(values)
        payload.pop("content_hash", None)
        unhashed = cls.model_validate(
            {**payload, "content_hash": "0" * 64},
            context={
                "allow_unbound_prediction_hash": True,
                "allow_incomplete_prediction_validation": True,
            },
        )
        hash_payload = unhashed.hash_payload()
        hash_payload["content_hash"] = hash_object(hash_payload)
        return cls.model_validate(hash_payload)

    def hash_payload(self) -> dict[str, Any]:
        """Return all scientific fields except the self-referential hash."""

        return self.model_dump(mode="python", exclude={"content_hash"})

    @model_validator(mode="after")
    def validate_hash_and_completion(self, info: Any) -> "PredictionArtifact":
        context = info.context or {}
        if not context.get("allow_unbound_prediction_hash"):
            if self.content_hash != hash_object(self.hash_payload()):
                raise ValueError("content_hash does not match the prediction artifact")
        if self.complete and not context.get("allow_incomplete_prediction_validation"):
            keys = [record.key for record in self.records]
            if len(set(keys)) != len(keys):
                raise ValueError("complete prediction artifact contains duplicate Prediction_Keys")
            sample_ids = [key.sample_id for key in keys]
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError("complete prediction artifact contains duplicate Sample_IDs")
            if set(sample_ids) != set(self.expected_sample_ids):
                raise ValueError(
                    "complete prediction artifact sample coverage does not match expected_sample_ids"
                )
            contexts = {
                (key.fold_id, key.seed, key.inference_branch) for key in keys
            }
            if len(contexts) != 1:
                raise ValueError(
                    "complete prediction artifact must use one fold, seed, and inference branch"
                )
        return self


__all__ = [
    "PredictionArtifact",
    "PredictionKey",
    "PredictionRecord",
    "PredictionShard",
    "PredictionValue",
    "ProbabilityValue",
    "canonical_prediction_key",
    "prediction_key_text",
]
