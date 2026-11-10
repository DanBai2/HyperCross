"""Immutable protocol, fold, selection-policy, and evaluation-budget schemas."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

from pydantic import Field, StrictInt, ValidationInfo, field_validator, model_validator

from .base import (
    ContentHash,
    FoldId,
    Identifier,
    MetricDirection,
    ProtocolKey,
    SampleId,
)
from .canonical import hash_object, normalize_id
from .environment import FrozenSchemaModel


def _canonical_identifiers(
    values: Iterable[str], *, field_name: str, allow_empty: bool = False
) -> Tuple[str, ...]:
    """Normalize and sort an identifier set by UTF-8 bytes."""

    normalized = tuple(normalize_id(value) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    duplicates = sorted(
        {value for value in normalized if normalized.count(value) > 1},
        key=lambda value: value.encode("utf-8"),
    )
    if duplicates:
        raise ValueError(
            f"{field_name} contains duplicate identifiers after normalization: "
            + ", ".join(duplicates)
        )
    return tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))


class SelectionPolicy(FrozenSchemaModel):
    """Predeclared rule used to select a candidate from permitted evidence."""

    policy_key: Identifier
    objective_metric: Identifier
    metric_direction: MetricDirection
    tie_break_rule: Identifier


class EvaluationBudget(FrozenSchemaModel):
    """Successful outer-evaluation allowance and committed consumption."""

    maximum_successful_evaluations: StrictInt = Field(gt=0)
    consumed_successful_evaluations: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_consumption(self) -> "EvaluationBudget":
        if self.consumed_successful_evaluations > self.maximum_successful_evaluations:
            raise ValueError(
                "consumed_successful_evaluations must not exceed "
                "maximum_successful_evaluations"
            )
        return self

    @property
    def remaining_successful_evaluations(self) -> int:
        """Return the number of successful evaluations still available."""

        return (
            self.maximum_successful_evaluations
            - self.consumed_successful_evaluations
        )


class InnerFold(FrozenSchemaModel):
    """Canonical inner-training and validation sample memberships."""

    fold_id: FoldId
    train_ids: Tuple[SampleId, ...]
    validation_ids: Tuple[SampleId, ...]

    @field_validator("train_ids", "validation_ids", mode="before")
    @classmethod
    def canonicalize_sample_ids(cls, value: Any, info: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{info.field_name} must be a collection of Sample_IDs")
        try:
            return _canonical_identifiers(value, field_name=info.field_name)
        except TypeError as exc:
            raise ValueError(
                f"{info.field_name} must be a collection of Sample_IDs"
            ) from exc


class OuterFold(FrozenSchemaModel):
    """Canonical outer-training/test memberships and nested folds."""

    fold_id: FoldId
    outer_train_ids: Tuple[SampleId, ...]
    outer_test_ids: Tuple[SampleId, ...]
    inner_folds: Tuple[InnerFold, ...] = ()

    @field_validator("outer_train_ids", "outer_test_ids", mode="before")
    @classmethod
    def canonicalize_sample_ids(cls, value: Any, info: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{info.field_name} must be a collection of Sample_IDs")
        try:
            return _canonical_identifiers(value, field_name=info.field_name)
        except TypeError as exc:
            raise ValueError(
                f"{info.field_name} must be a collection of Sample_IDs"
            ) from exc

    @field_validator("inner_folds")
    @classmethod
    def canonicalize_inner_folds(
        cls, value: Tuple[InnerFold, ...]
    ) -> Tuple[InnerFold, ...]:
        identifiers = [normalize_id(fold.fold_id) for fold in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("inner_folds contains duplicate Fold_IDs")
        return tuple(
            sorted(value, key=lambda fold: normalize_id(fold.fold_id).encode("utf-8"))
        )


class ProtocolManifest(FrozenSchemaModel):
    """Hash-bound protocol identity with canonical folds and seed set."""

    schema_version: Identifier
    protocol_key: ProtocolKey
    dataset_hash: ContentHash
    folds: Tuple[OuterFold, ...] = Field(min_length=1)
    selection_policy: SelectionPolicy
    seed_set: Tuple[StrictInt, ...] = Field(min_length=1)
    evaluation_budget: EvaluationBudget
    protocol_hash: ContentHash

    @field_validator("seed_set", mode="before")
    @classmethod
    def canonicalize_seed_set(cls, value: Any) -> Tuple[int, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("seed_set must be a collection of integers")
        try:
            seeds = tuple(value)
        except TypeError as exc:
            raise ValueError("seed_set must be a collection of integers") from exc
        if not seeds:
            raise ValueError("seed_set must not be empty")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError("seed_set must contain only strict integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("seed_set contains duplicate seeds")
        return tuple(sorted(seeds))

    @field_validator("folds")
    @classmethod
    def canonicalize_outer_folds(
        cls, value: Tuple[OuterFold, ...]
    ) -> Tuple[OuterFold, ...]:
        identifiers = [normalize_id(fold.fold_id) for fold in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("folds contains duplicate Fold_IDs")
        return tuple(
            sorted(value, key=lambda fold: normalize_id(fold.fold_id).encode("utf-8"))
        )

    @classmethod
    def create(cls, **values: Any) -> "ProtocolManifest":
        """Validate, canonicalize, and bind a manifest to its protocol hash."""

        payload = dict(values)
        payload.pop("protocol_hash", None)
        unhashed = cls.model_validate(
            {**payload, "protocol_hash": "0" * 64},
            context={"allow_unbound_protocol_hash": True},
        )
        payload = unhashed.hash_payload()
        payload["protocol_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    def hash_payload(self) -> dict[str, Any]:
        """Return every protocol identity field except the self hash."""

        return self.model_dump(mode="python", exclude={"protocol_hash"})

    @model_validator(mode="after")
    def validate_protocol_hash(self, info: ValidationInfo) -> "ProtocolManifest":
        if info.context and info.context.get("allow_unbound_protocol_hash"):
            return self
        if self.protocol_hash != hash_object(self.hash_payload()):
            raise ValueError("protocol_hash does not match the protocol manifest")
        return self


__all__ = [
    "EvaluationBudget",
    "InnerFold",
    "OuterFold",
    "ProtocolManifest",
    "SelectionPolicy",
]
