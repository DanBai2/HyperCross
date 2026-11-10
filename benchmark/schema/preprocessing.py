"""Immutable schemas for fold-scoped fitted preprocessing artifacts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from pydantic import Field, StrictInt, field_validator, model_validator

from .base import (
    ArtifactId,
    ContentHash,
    FittedArtifactKind,
    FoldId,
    Identifier,
    SampleId,
    ensure_machine_value,
)
from .canonical import hash_object, normalize_id
from .environment import FrozenSchemaModel


class FitBoundaryScope(str, Enum):
    """Permitted training boundary for one fitted artifact."""

    INNER_TRAIN = "inner_train"
    OUTER_TRAIN = "outer_train"


class FitDataRole(str, Enum):
    """Semantic source used while fitting learned preprocessing state."""

    TRAINING_VALUES = "training_values"
    INNER_OOF_PREDICTIONS = "inner_oof_predictions"


def canonical_sample_ids(
    values: Iterable[str], *, field_name: str, allow_empty: bool = False
) -> Tuple[str, ...]:
    """Normalize a Sample_ID set and return canonical UTF-8 byte order."""

    normalized = tuple(normalize_id(value) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} contains duplicate Sample_IDs")
    return tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))


class FitBoundary(FrozenSchemaModel):
    """Protocol-derived boundary checked before fitting or applying state."""

    fold_id: FoldId
    scope: FitBoundaryScope
    allowed_fit_sample_ids: Tuple[SampleId, ...] = Field(min_length=1)
    outer_test_sample_ids: Tuple[SampleId, ...] = ()
    inner_fold_id: Optional[FoldId] = None

    @field_validator("allowed_fit_sample_ids", "outer_test_sample_ids", mode="before")
    @classmethod
    def canonicalize_ids(cls, value: Any, info: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{info.field_name} must be a collection of Sample_IDs")
        try:
            return canonical_sample_ids(
                value,
                field_name=info.field_name,
                allow_empty=info.field_name == "outer_test_sample_ids",
            )
        except TypeError as exc:
            raise ValueError(
                f"{info.field_name} must be a collection of Sample_IDs"
            ) from exc

    @model_validator(mode="after")
    def validate_scope(self) -> "FitBoundary":
        overlap = set(self.allowed_fit_sample_ids) & set(self.outer_test_sample_ids)
        if overlap:
            raise ValueError(
                "allowed_fit_sample_ids must not intersect outer_test_sample_ids"
            )
        if self.scope is FitBoundaryScope.INNER_TRAIN and self.inner_fold_id is None:
            raise ValueError("inner_train boundary requires inner_fold_id")
        if self.scope is FitBoundaryScope.OUTER_TRAIN and self.inner_fold_id is not None:
            raise ValueError("outer_train boundary must not declare inner_fold_id")
        return self


class FittedArtifact(FrozenSchemaModel):
    """Learned preprocessing state bound to exact samples and one fold."""

    artifact_id: ArtifactId
    kind: FittedArtifactKind
    dataset_hash: ContentHash
    protocol_hash: ContentHash
    fold_id: FoldId
    boundary_scope: FitBoundaryScope
    inner_fold_id: Optional[FoldId] = None
    fit_data_role: FitDataRole
    fit_sample_ids: Tuple[SampleId, ...] = Field(min_length=1)
    config: Dict[Identifier, Any]
    config_hash: ContentHash
    input_hash: ContentHash
    state: Dict[Identifier, Any]
    state_hash: ContentHash
    output_shape: Tuple[StrictInt, ...]

    @field_validator("fit_sample_ids", mode="before")
    @classmethod
    def canonicalize_fit_ids(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("fit_sample_ids must be a collection of Sample_IDs")
        try:
            return canonical_sample_ids(value, field_name="fit_sample_ids")
        except TypeError as exc:
            raise ValueError(
                "fit_sample_ids must be a collection of Sample_IDs"
            ) from exc

    @field_validator("config", "state")
    @classmethod
    def validate_machine_mapping(cls, value: Dict[str, Any], info: Any) -> Dict[str, Any]:
        ensure_machine_value(value, field_name=info.field_name)
        return value

    @field_validator("output_shape")
    @classmethod
    def validate_output_shape(cls, value: Tuple[int, ...]) -> Tuple[int, ...]:
        if any(isinstance(item, bool) or item < 0 for item in value):
            raise ValueError("output_shape dimensions must be non-negative integers")
        return value

    @model_validator(mode="after")
    def validate_bound_hashes_and_scope(self) -> "FittedArtifact":
        if self.config_hash != hash_object(self.config):
            raise ValueError("config_hash does not match config")
        if self.state_hash != hash_object(self.state):
            raise ValueError("state_hash does not match state")
        if self.boundary_scope is FitBoundaryScope.INNER_TRAIN and self.inner_fold_id is None:
            raise ValueError("inner_train artifact requires inner_fold_id")
        if self.boundary_scope is FitBoundaryScope.OUTER_TRAIN and self.inner_fold_id is not None:
            raise ValueError("outer_train artifact must not declare inner_fold_id")
        if self.kind in {FittedArtifactKind.THRESHOLD, FittedArtifactKind.CALIBRATION}:
            if self.fit_data_role is not FitDataRole.INNER_OOF_PREDICTIONS:
                raise ValueError(
                    "threshold and calibration artifacts require inner OOF predictions"
                )
        return self


class PCAIdentity(FrozenSchemaModel):
    """Complete identity of one fold-scoped PCA basis and representation."""

    dataset_hash: ContentHash
    protocol_hash: ContentHash
    fold_id: FoldId
    representation_key: Identifier
    fit_sample_ids: Tuple[SampleId, ...] = Field(min_length=1)
    fit_sample_ids_hash: ContentHash
    config_hash: ContentHash
    transform_state_hash: ContentHash

    @classmethod
    def from_fitted_artifact(
        cls,
        artifact: FittedArtifact,
        *,
        representation_key: str,
    ) -> "PCAIdentity":
        """Bind a representation identity to an already validated PCA artifact."""

        if artifact.kind is not FittedArtifactKind.PCA:
            raise ValueError("PCA identity requires a PCA fitted artifact")
        return cls(
            dataset_hash=artifact.dataset_hash,
            protocol_hash=artifact.protocol_hash,
            fold_id=artifact.fold_id,
            representation_key=representation_key,
            fit_sample_ids=artifact.fit_sample_ids,
            fit_sample_ids_hash=hash_object(artifact.fit_sample_ids),
            config_hash=artifact.config_hash,
            transform_state_hash=artifact.state_hash,
        )

    @field_validator("fit_sample_ids", mode="before")
    @classmethod
    def canonicalize_identity_fit_ids(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("fit_sample_ids must be a collection of Sample_IDs")
        return canonical_sample_ids(value, field_name="fit_sample_ids")

    @model_validator(mode="after")
    def validate_fit_sample_ids_hash(self) -> "PCAIdentity":
        if self.fit_sample_ids_hash != hash_object(self.fit_sample_ids):
            raise ValueError("fit_sample_ids_hash does not match fit_sample_ids")
        return self


class FeatureCacheKey(FrozenSchemaModel):
    """Identity of one immutable, fold-scoped cached feature tensor."""

    dataset_hash: ContentHash
    protocol_hash: ContentHash
    fold_id: FoldId
    representation_key: Identifier
    transform_state_hash: ContentHash

    @classmethod
    def from_pca_identity(cls, identity: PCAIdentity) -> "FeatureCacheKey":
        return cls(
            dataset_hash=identity.dataset_hash,
            protocol_hash=identity.protocol_hash,
            fold_id=identity.fold_id,
            representation_key=identity.representation_key,
            transform_state_hash=identity.transform_state_hash,
        )

    @property
    def identity_hash(self) -> str:
        """Return the stable filename-safe hash of every cache-key field."""

        return hash_object(self)


class FeatureLineage(FrozenSchemaModel):
    """Verified sources and transform state that produced a cached tensor."""

    source_hashes: Dict[Identifier, ContentHash] = Field(min_length=1)
    transform_config: Dict[Identifier, Any]
    config_hash: ContentHash
    fit_sample_ids: Tuple[SampleId, ...] = Field(min_length=1)
    fit_sample_ids_hash: ContentHash
    transform_hash: ContentHash
    output_shape: Tuple[StrictInt, ...]

    @classmethod
    def create(
        cls,
        *,
        source_hashes: Mapping[str, str],
        transform_config: Mapping[str, Any],
        fit_sample_ids: Iterable[str],
        transform_hash: str,
        output_shape: Iterable[int],
    ) -> "FeatureLineage":
        """Create lineage with hashes bound to exact config and fit membership."""

        canonical_fit_ids = canonical_sample_ids(
            fit_sample_ids, field_name="fit_sample_ids"
        )
        config = dict(transform_config)
        return cls(
            source_hashes=dict(source_hashes),
            transform_config=config,
            config_hash=hash_object(config),
            fit_sample_ids=canonical_fit_ids,
            fit_sample_ids_hash=hash_object(canonical_fit_ids),
            transform_hash=transform_hash,
            output_shape=tuple(output_shape),
        )

    @classmethod
    def from_fitted_artifact(
        cls,
        artifact: FittedArtifact,
        *,
        source_hashes: Mapping[str, str],
        output_shape: Iterable[int],
    ) -> "FeatureLineage":
        """Build complete lineage from a fitted transform and current sources."""

        return cls.create(
            source_hashes=source_hashes,
            transform_config=artifact.config,
            fit_sample_ids=artifact.fit_sample_ids,
            transform_hash=artifact.state_hash,
            output_shape=output_shape,
        )

    @field_validator("fit_sample_ids", mode="before")
    @classmethod
    def canonicalize_lineage_fit_ids(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("fit_sample_ids must be a collection of Sample_IDs")
        return canonical_sample_ids(value, field_name="fit_sample_ids")

    @field_validator("transform_config")
    @classmethod
    def validate_transform_config(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        ensure_machine_value(value, field_name="transform_config")
        return value

    @field_validator("output_shape")
    @classmethod
    def validate_cache_output_shape(cls, value: Tuple[int, ...]) -> Tuple[int, ...]:
        if any(isinstance(item, bool) or item < 0 for item in value):
            raise ValueError("output_shape dimensions must be non-negative integers")
        return value

    @model_validator(mode="after")
    def validate_lineage_hashes(self) -> "FeatureLineage":
        if self.config_hash != hash_object(self.transform_config):
            raise ValueError("config_hash does not match transform_config")
        if self.fit_sample_ids_hash != hash_object(self.fit_sample_ids):
            raise ValueError("fit_sample_ids_hash does not match fit_sample_ids")
        return self


__all__ = [
    "FeatureCacheKey",
    "FeatureLineage",
    "FitBoundary",
    "FitBoundaryScope",
    "FitDataRole",
    "FittedArtifact",
    "PCAIdentity",
    "canonical_sample_ids",
]
