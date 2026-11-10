"""Dataset catalog records and immutable sample-manifest schemas.

This module defines identities and metadata only. Dataset discovery, file access,
manifest publication, and version allocation belong to the catalog
materialization layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import ConfigDict, Field, StrictBool, StrictInt, TypeAdapter, field_validator, model_validator
from typing_extensions import Annotated

from .base import ContentHash, GroupId, Identifier, SampleId, StrictSchemaModel, TaskKind, UtcTimestamp
from .canonical import normalize_id


PositiveStrictInteger = Annotated[StrictInt, Field(gt=0)]
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class FrozenDatasetModel(StrictSchemaModel):
    """Strict, assignment-immutable base for dataset identity records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class LabelValueKind(str, Enum):
    """Supported semantic label encodings."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"


class LabelSchema(FrozenDatasetModel):
    """Versioned label shape and semantic metadata for one dataset."""

    schema_key: Identifier
    value_kind: LabelValueKind
    dimension: PositiveStrictInteger
    target_names: Tuple[Identifier, ...] = Field(min_length=1)
    class_names: Tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_shape_and_classes(self) -> "LabelSchema":
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target_names must be unique")
        if len(self.target_names) != self.dimension:
            raise ValueError("target_names length must equal label dimension")

        if self.value_kind == LabelValueKind.CONTINUOUS:
            if self.class_names:
                raise ValueError("continuous labels must not declare class_names")
        elif self.value_kind == LabelValueKind.BINARY:
            if self.dimension != 1 or len(self.class_names) != 2:
                raise ValueError("binary labels require dimension 1 and exactly two classes")
        elif self.dimension != 1 or len(self.class_names) < 2:
            raise ValueError("categorical labels require dimension 1 and at least two classes")

        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        return self


class DatasetSpec(FrozenDatasetModel):
    """Registered dataset semantics, independent of any machine-local path."""

    dataset_key: Identifier
    task_kind: TaskKind
    expected_sample_count: PositiveStrictInteger
    label_schema: LabelSchema
    grouped: StrictBool
    license_reference: Optional[str] = None

    @field_validator("license_reference")
    @classmethod
    def validate_license_reference(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("license_reference must be non-empty without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_task_label_compatibility(self) -> "DatasetSpec":
        expected_kind = {
            TaskKind.MULTI_TARGET_REGRESSION: LabelValueKind.CONTINUOUS,
            TaskKind.BINARY_CLASSIFICATION: LabelValueKind.BINARY,
            TaskKind.REMOTE_CLASSIFICATION: LabelValueKind.CATEGORICAL,
        }[self.task_kind]
        if self.label_schema.value_kind != expected_kind:
            raise ValueError(
                f"task_kind {self.task_kind.value!r} requires label value_kind {expected_kind!r}"
            )
        return self


class SampleRecord(FrozenDatasetModel):
    """One sample's stable identity, label metadata, and required input hashes."""

    sample_id: SampleId
    group_id: Optional[GroupId] = None
    label_reference: str
    input_references: Dict[Identifier, str] = Field(min_length=1)
    input_content_hashes: Dict[Identifier, ContentHash] = Field(min_length=1)
    label_schema_metadata: Dict[Identifier, Any] = Field(min_length=1)

    @field_validator("label_reference")
    @classmethod
    def validate_label_reference(cls, value: str) -> str:
        if not value or not value.strip() or value != value.strip():
            raise ValueError("label_reference must be non-empty without surrounding whitespace")
        return value

    @field_validator("input_references")
    @classmethod
    def validate_input_references(cls, value: Dict[str, str]) -> Dict[str, str]:
        for name, reference in value.items():
            _IDENTIFIER_ADAPTER.validate_python(name)
            if not reference or not reference.strip() or reference != reference.strip():
                raise ValueError(f"input reference for {name!r} must be non-empty without surrounding whitespace")
        return value

    @field_validator("label_schema_metadata")
    @classmethod
    def validate_label_schema_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        for key, item in value.items():
            _IDENTIFIER_ADAPTER.validate_python(key)
            if item is not None and not isinstance(item, (str, bool, int, float)):
                raise ValueError("label_schema_metadata values must be scalar")
        return value

    @model_validator(mode="after")
    def validate_input_hash_coverage(self) -> "SampleRecord":
        reference_names = set(self.input_references)
        hash_names = set(self.input_content_hashes)
        if reference_names != hash_names:
            missing = sorted(reference_names - hash_names)
            unexpected = sorted(hash_names - reference_names)
            raise ValueError(
                "input_content_hashes must exactly cover input_references; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return self


def canonical_sample_order(sample_ids: Tuple[str, ...]) -> Tuple[str, ...]:
    """Return normalized sample IDs sorted by their UTF-8 byte sequence."""

    normalized = tuple(normalize_id(sample_id) for sample_id in sample_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError("sample IDs must be unique after canonical normalization")
    return tuple(sorted(normalized, key=lambda sample_id: sample_id.encode("utf-8")))


class SampleManifest(FrozenDatasetModel):
    """Hash-bearing, immutable declaration of a dataset's complete sample set."""

    schema_version: Identifier
    dataset_key: Identifier
    records: Tuple[SampleRecord, ...] = Field(min_length=1)
    canonical_order: Tuple[SampleId, ...] = Field(min_length=1)
    dataset_hash: ContentHash
    created_utc: UtcTimestamp

    @field_validator("canonical_order")
    @classmethod
    def normalize_canonical_order(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return tuple(normalize_id(sample_id) for sample_id in value)

    @model_validator(mode="after")
    def validate_canonical_order(self) -> "SampleManifest":
        expected = canonical_sample_order(tuple(record.sample_id for record in self.records))
        if self.canonical_order != expected:
            raise ValueError(
                "canonical_order must contain the complete normalized Sample_ID set "
                "sorted by UTF-8 bytes"
            )
        return self


class PublicSampleRecord(FrozenDatasetModel):
    """A publication-safe sample projection with no source paths or label reference.

    Sample and group identifiers are pseudonymous. Only explicitly approved,
    non-raw input hashes and label metadata may be copied into this record.
    """

    sample_id: SampleId
    group_id: GroupId
    input_content_hashes: Dict[Identifier, ContentHash] = Field(default_factory=dict)
    approved_label_metadata: Dict[Identifier, Any] = Field(default_factory=dict)

    @field_validator("approved_label_metadata")
    @classmethod
    def validate_approved_label_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        for key, item in value.items():
            _IDENTIFIER_ADAPTER.validate_python(key)
            if item is not None and not isinstance(item, (str, bool, int, float)):
                raise ValueError("approved label metadata values must be scalar")
        return value


class PublicSampleManifest(FrozenDatasetModel):
    """Immutable, de-identified projection suitable for public release."""

    schema_version: Identifier
    dataset_key: Identifier
    source_dataset_hash: ContentHash
    records: Tuple[PublicSampleRecord, ...] = Field(min_length=1)
    canonical_order: Tuple[SampleId, ...] = Field(min_length=1)
    deidentification_scheme: Identifier
    deidentification_key_id: Identifier
    export_policy_hash: ContentHash
    public_manifest_hash: ContentHash
    created_utc: UtcTimestamp

    @field_validator("canonical_order")
    @classmethod
    def normalize_public_order(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        return tuple(normalize_id(sample_id) for sample_id in value)

    @model_validator(mode="after")
    def validate_public_order(self) -> "PublicSampleManifest":
        expected = canonical_sample_order(tuple(record.sample_id for record in self.records))
        if self.canonical_order != expected:
            raise ValueError(
                "canonical_order must contain the complete pseudonymous Sample_ID set "
                "sorted by UTF-8 bytes"
            )
        return self
