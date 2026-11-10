"""Stable model identity, input contract, and run metadata schemas."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator
from typing_extensions import Annotated

from .base import (
    ContentHash,
    Identifier,
    ImplementationKind,
    ModelKey,
    StrictSchemaModel,
    UtcTimestamp,
)


NonNegativeStrictInteger = Annotated[StrictInt, Field(ge=0)]
PositiveShapeDimension = Annotated[StrictInt, Field(gt=0)]
TensorShape = Tuple[PositiveShapeDimension, ...]


class FrozenModelSchema(StrictSchemaModel):
    """Immutable base for hash- and registry-bound model declarations."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
        frozen=True,
    )


class ModelIdentity(FrozenModelSchema):
    """Identity that a stable ``Model_Key`` promises to construct."""

    model_key: ModelKey
    display_name: str = Field(min_length=1)
    class_fully_qualified_name: str = Field(min_length=3)
    implementation_kind: ImplementationKind
    parameter_count: NonNegativeStrictInteger
    source_revision: str = Field(min_length=1)

    @field_validator("display_name", "class_fully_qualified_name", "source_revision")
    @classmethod
    def reject_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("value must not have leading or trailing whitespace")
        return value

    @field_validator("class_fully_qualified_name")
    @classmethod
    def require_qualified_class_name(cls, value: str) -> str:
        module_name, separator, class_name = value.rpartition(".")
        if not separator or not module_name or not class_name:
            raise ValueError("class_fully_qualified_name must include module and class")
        return value


class InputContract(FrozenModelSchema):
    """One model's declared representation and tensor input requirements.

    Full supplied-input equality checks are implemented by task 6.2. This
    schema freezes the single contract returned by registry resolution.
    """

    representation_key: Identifier
    modalities: Tuple[Identifier, ...] = Field(min_length=1)
    tensor_shapes: Dict[str, TensorShape]
    dtypes: Dict[str, Identifier]
    channel_or_band_order: Dict[str, Tuple[Identifier, ...]]
    preprocessing_key: Identifier

    @field_validator("modalities")
    @classmethod
    def require_unique_modalities(
        cls, value: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("modalities must be unique")
        return value

    @model_validator(mode="after")
    def require_one_declaration_per_modality(self) -> "InputContract":
        expected = set(self.modalities)
        for field_name in ("tensor_shapes", "dtypes", "channel_or_band_order"):
            observed = set(getattr(self, field_name))
            if observed != expected:
                raise ValueError(
                    f"{field_name} keys must exactly equal modalities; "
                    f"missing={sorted(expected - observed)}, "
                    f"unexpected={sorted(observed - expected)}"
                )
        return self


class UpstreamReproductionStatus(str, Enum):
    """Audited outcome of reproducing the pinned upstream implementation."""

    NOT_RUN = "not_run"
    FAILED = "failed"
    PASSED = "passed"


class UpstreamReproductionRecord(FrozenModelSchema):
    """Hash-bound evidence produced by an upstream reproduction audit."""

    model_key: ModelKey
    upstream_repository: str = Field(min_length=1)
    upstream_commit: Identifier
    license_reference: str = Field(min_length=1)
    source_hash: ContentHash
    dependency_environment_hash: ContentHash
    result_artifact_hash: ContentHash
    status: UpstreamReproductionStatus
    completed_utc: UtcTimestamp

    @field_validator("upstream_repository")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        return _validate_absolute_url(value, field_name="upstream_repository")

    @field_validator("license_reference")
    @classmethod
    def reject_blank_license_reference(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("license_reference must not have surrounding whitespace")
        return value


class ModelProvenance(FrozenModelSchema):
    """Pinned source and adaptation provenance for an external model.

    ``adaptation_notes`` records each input adaptation. An empty
    ``ignored_modalities`` tuple explicitly means that the adaptation ignores
    no upstream modality.
    """

    upstream_repository: str = Field(min_length=1)
    upstream_commit: Identifier
    license_reference: str = Field(min_length=1)
    downloaded_utc: UtcTimestamp
    source_hash: ContentHash
    local_patch_hash: ContentHash
    dependency_environment_hash: ContentHash
    upstream_reproduction_status: UpstreamReproductionStatus
    upstream_reproduction_record: Optional[UpstreamReproductionRecord] = None
    adaptation_notes: Tuple[str, ...] = ()
    ignored_modalities: Tuple[Identifier, ...] = ()

    @field_validator("upstream_repository")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        return _validate_absolute_url(value, field_name="upstream_repository")

    @field_validator("license_reference")
    @classmethod
    def reject_blank_license_reference(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("license_reference must not have surrounding whitespace")
        return value

    @field_validator("adaptation_notes")
    @classmethod
    def validate_adaptation_notes(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if any(not item or item != item.strip() for item in value):
            raise ValueError("adaptation_notes must contain non-blank trimmed strings")
        if len(value) != len(set(value)):
            raise ValueError("adaptation_notes must not contain duplicates")
        return value

    @field_validator("ignored_modalities")
    @classmethod
    def canonicalize_ignored_modalities(
        cls, value: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("ignored_modalities must not contain duplicates")
        canonical = tuple(sorted(value, key=lambda item: item.encode("utf-8")))
        if value != canonical:
            raise ValueError("ignored_modalities must be in canonical bytewise order")
        return value


def _validate_absolute_url(value: str, *, field_name: str) -> str:
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain credentials")
    return value


class ModelRunMetadata(FrozenModelSchema):
    """Registry-derived identity fields recorded when a model run starts."""

    model_key: ModelKey
    source_revision: str = Field(min_length=1)
    config_hash: ContentHash
    parameter_count: NonNegativeStrictInteger
    representation_key: Identifier
    preprocessing_key: Identifier
