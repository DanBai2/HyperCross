"""Immutable records persisted by the append-only benchmark registry."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import Field, StrictInt, field_validator, model_validator
from typing_extensions import Annotated

from benchmark.schema.aggregation import AggregationPolicy
from benchmark.schema.base import (
    ClaimStatus,
    ContentHash,
    FoldId,
    Identifier,
    ModelKey,
    RunId,
    RunStatus,
    UtcTimestamp,
    ensure_machine_value,
)
from benchmark.schema.environment import FrozenSchemaModel
from benchmark.schema.canonical import hash_object, normalize_id
from benchmark.schema.models import InputContract, ModelIdentity
from benchmark.schema.records import ArtifactReference, ResourceBudget
from benchmark.schema.resources import RuntimeMeasurements


class FrozenRunIdentity(FrozenSchemaModel):
    """Identity persisted before any training starts."""

    dataset_hash: ContentHash
    protocol_hash: ContentHash
    model_key: ModelKey
    config_hash: ContentHash
    fold_id: FoldId
    seed: StrictInt
    environment_lock_hash: ContentHash
    environment_compatibility_policy: Identifier


class RecoveryArtifact(FrozenSchemaModel):
    """A complete checkpoint-boundary recovery bundle registration."""

    schema_version: str = Field(default="1.0", min_length=1)
    recovery_id: Identifier
    run_id: RunId
    frozen_identity: FrozenRunIdentity
    artifact: ArtifactReference
    model_state_hash: ContentHash
    optimizer_state_hash: ContentHash
    scheduler_state_hash: ContentHash
    stage_state_hash: ContentHash
    last_completed_stage: Identifier
    completed_checkpoint_boundary: Annotated[StrictInt, Field(ge=0)]
    created_utc: UtcTimestamp

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> "RecoveryArtifact":
        if self.artifact.artifact_id != self.recovery_id:
            raise ValueError("recovery artifact_id must equal recovery_id")
        return self


class Provenance(FrozenSchemaModel):
    """Code, environment, data, and external-source lineage for a run."""

    code_revision: Identifier
    upstream_repository: Optional[str] = None
    upstream_commit: Optional[Identifier] = None
    license_reference: Optional[str] = None
    downloaded_utc: Optional[UtcTimestamp] = None
    source_hash: ContentHash
    local_patch_hash: Optional[ContentHash] = None
    environment_hash: ContentHash
    config_hash: ContentHash
    dataset_hash: ContentHash
    artifact_hashes: Dict[str, ContentHash] = Field(default_factory=dict)
    upstream_reproduction_status: Optional[Identifier] = None
    adaptation_notes: Tuple[str, ...] = ()
    ignored_modalities: Tuple[Identifier, ...] = ()


class ClaimCeiling(FrozenSchemaModel):
    """One applicable evidence ceiling and its auditable cause."""

    status: ClaimStatus
    reason_code: Identifier
    evidence_reference: Identifier


class ClaimDecision(FrozenSchemaModel):
    """Claim status and its machine-readable evidence."""

    status: ClaimStatus
    supporting_reasons: Tuple[Identifier, ...] = ()
    blocking_reasons: Tuple[Identifier, ...] = ()
    applied_ceilings: Tuple[ClaimCeiling, ...] = ()
    prerequisite_checks: Dict[str, bool] = Field(default_factory=dict)


class LegacySourceReference(FrozenSchemaModel):
    """Read-only runner or checkpoint reference discovered in legacy evidence."""

    reference_kind: Identifier
    source_value: str = Field(min_length=1)
    resolved_path: Optional[str] = None
    content_hash: Optional[ContentHash] = None


class RunRecord(FrozenSchemaModel):
    """One append-only snapshot in a run's status history."""

    schema_version: str = Field(default="1.0", min_length=1)
    run_id: RunId
    frozen_identity: FrozenRunIdentity
    model_identity: ModelIdentity
    input_contract: InputContract
    provenance: Provenance
    transform_artifact_refs: Tuple[ArtifactReference, ...] = ()
    checkpoint_ref: Optional[ArtifactReference] = None
    recovery_artifact_ref: Optional[ArtifactReference] = None
    prediction_artifact_ref: Optional[ArtifactReference] = None
    last_completed_stage: Optional[Identifier] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    resource_budget: ResourceBudget
    runtime_measurements: Optional[RuntimeMeasurements] = None
    status: RunStatus
    claim_decision: ClaimDecision
    source_path: Optional[str] = None
    source_content_hash: Optional[ContentHash] = None
    legacy_source_format: Optional[Identifier] = None
    protocol_descriptions: Tuple[str, ...] = ()
    known_caveats: Tuple[str, ...] = ()
    runner_metadata: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_references: Tuple[LegacySourceReference, ...] = ()
    missing_provenance_fields: Tuple[Identifier, ...] = ()
    started_utc: UtcTimestamp
    completed_utc: Optional[UtcTimestamp] = None

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return ensure_machine_value(value, field_name="metrics")

    @field_validator("runner_metadata")
    @classmethod
    def validate_runner_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return ensure_machine_value(value, field_name="runner_metadata")

    @field_validator("protocol_descriptions", "known_caveats")
    @classmethod
    def validate_legacy_text(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError("legacy text entries must be non-empty strings")
        return value

    @field_validator("missing_provenance_fields")
    @classmethod
    def validate_missing_provenance_fields(
        cls, value: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("missing_provenance_fields must be unique")
        if value != tuple(sorted(value, key=lambda item: item.encode("utf-8"))):
            raise ValueError("missing_provenance_fields must be canonically sorted")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> "RunRecord":
        if self.model_identity.model_key != self.frozen_identity.model_key:
            raise ValueError("model identity must match frozen model_key")
        if self.provenance.config_hash != self.frozen_identity.config_hash:
            raise ValueError("provenance config_hash must match frozen identity")
        if self.provenance.dataset_hash != self.frozen_identity.dataset_hash:
            raise ValueError("provenance dataset_hash must match frozen identity")
        if self.provenance.environment_hash != self.frozen_identity.environment_lock_hash:
            raise ValueError("provenance environment_hash must match frozen identity")
        if self.completed_utc is not None and self.completed_utc < self.started_utc:
            raise ValueError("completed_utc must not precede started_utc")

        if self.status in {RunStatus.PLANNED, RunStatus.RUNNING}:
            if self.completed_utc is not None:
                raise ValueError("active run records must not have completed_utc")
        elif self.completed_utc is None:
            raise ValueError("terminal run records must have completed_utc")

        if self.status in {RunStatus.FAILED, RunStatus.ABORTED}:
            if self.last_completed_stage is None:
                raise ValueError("interrupted run must record last_completed_stage")
        elif self.last_completed_stage is not None:
            raise ValueError(
                "last_completed_stage is only valid for failed or aborted runs"
            )

        if self.status is RunStatus.COMPLETE:
            if self.runtime_measurements is None:
                raise ValueError("complete run must include runtime_measurements")
            if self.checkpoint_ref is None:
                raise ValueError("complete run must include checkpoint_ref")
            if self.prediction_artifact_ref is None:
                raise ValueError("complete run must include prediction_artifact_ref")

        legacy_values_present = any(
            (
                self.source_path is not None,
                self.source_content_hash is not None,
                self.legacy_source_format is not None,
                bool(self.protocol_descriptions),
                bool(self.known_caveats),
                bool(self.runner_metadata),
                bool(self.checkpoint_references),
                bool(self.missing_provenance_fields),
            )
        )
        if self.status is RunStatus.IMPORTED:
            if self.source_path is None or self.source_content_hash is None:
                raise ValueError(
                    "imported run must preserve source_path and source_content_hash"
                )
            if self.legacy_source_format is None:
                raise ValueError("imported run must record legacy_source_format")
        elif legacy_values_present:
            raise ValueError("legacy evidence fields are only valid for imported runs")
        return self


class ArtifactRecord(FrozenSchemaModel):
    """One immutable artifact registration."""

    schema_version: str = Field(default="1.0", min_length=1)
    artifact: ArtifactReference
    artifact_kind: Identifier
    run_id: Optional[RunId] = None
    created_utc: UtcTimestamp


class AggregateRecord(FrozenSchemaModel):
    """One immutable aggregate and its exact run/artifact membership."""

    schema_version: str = Field(default="1.0", min_length=1)
    aggregate_id: Identifier
    member_run_ids: Tuple[RunId, ...] = Field(min_length=1)
    prediction_hashes: Dict[str, ContentHash]
    dataset_hash: ContentHash
    protocol_hash: ContentHash
    model_key: ModelKey
    representation_key: Identifier
    aggregation_policy: Optional[AggregationPolicy] = None
    aggregation_policy_hash: Optional[ContentHash] = None
    metrics: Dict[str, Any]
    artifact_refs: Tuple[ArtifactReference, ...] = ()
    claim_decision: ClaimDecision
    provenance: Provenance
    created_utc: UtcTimestamp

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return ensure_machine_value(value, field_name="metrics")

    @model_validator(mode="after")
    def validate_membership(self) -> "AggregateRecord":
        canonical_members = tuple(
            sorted(
                self.member_run_ids,
                key=lambda run_id: normalize_id(run_id).encode("utf-8"),
            )
        )
        if canonical_members != self.member_run_ids:
            raise ValueError("member_run_ids must be in canonical sorted order")
        if len(set(self.member_run_ids)) != len(self.member_run_ids):
            raise ValueError("member_run_ids must be unique")
        if set(self.prediction_hashes) != set(self.member_run_ids):
            raise ValueError("prediction_hashes keys must exactly equal member_run_ids")
        if (self.aggregation_policy is None) != (self.aggregation_policy_hash is None):
            raise ValueError(
                "aggregation_policy and aggregation_policy_hash must be recorded together"
            )
        if (
            self.aggregation_policy is not None
            and self.aggregation_policy_hash != hash_object(self.aggregation_policy)
        ):
            raise ValueError("aggregation_policy_hash must match aggregation_policy")
        if self.provenance.dataset_hash != self.dataset_hash:
            raise ValueError("aggregate provenance dataset_hash must match aggregate")
        return self


class NegativeEvidenceReason(str, Enum):
    """Predeclared candidate outcomes retained even when promotion stops."""

    UNDERPERFORMANCE = "underperformance"
    SEED_INSTABILITY = "seed_instability"
    REPRESENTATION_ABLATION_FAILURE = "representation_ablation_failure"


class NegativeEvidenceRecord(FrozenSchemaModel):
    """Append-only candidate result with its exact tested configuration."""

    schema_version: str = Field(default="1.0", min_length=1)
    evidence_id: Identifier
    reason: NegativeEvidenceReason
    candidate_aggregate_id: Identifier
    candidate_run_ids: Tuple[RunId, ...] = Field(min_length=1)
    control_aggregate_id: Optional[Identifier] = None
    control_run_ids: Tuple[RunId, ...] = ()
    tested_configuration: Dict[str, Any]
    candidate_metrics: Dict[str, Any]
    control_metrics: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    created_utc: UtcTimestamp

    @field_validator(
        "tested_configuration", "candidate_metrics", "control_metrics", "details"
    )
    @classmethod
    def validate_machine_payload(cls, value: Any, info: Any) -> Any:
        if value is None and info.field_name == "control_metrics":
            return value
        return ensure_machine_value(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_membership(self) -> "NegativeEvidenceRecord":
        for field_name in ("candidate_run_ids", "control_run_ids"):
            values = getattr(self, field_name)
            canonical = tuple(
                sorted(values, key=lambda run_id: normalize_id(run_id).encode("utf-8"))
            )
            if values != canonical or len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique and canonically sorted")
        control_fields_present = (
            self.control_aggregate_id is not None,
            bool(self.control_run_ids),
            self.control_metrics is not None,
        )
        if any(control_fields_present) and not all(control_fields_present):
            raise ValueError("control aggregate, runs, and metrics must be recorded together")
        return self


class SupersedingRecord(FrozenSchemaModel):
    """Append-only link from prior evidence to its correcting record."""

    schema_version: str = Field(default="1.0", min_length=1)
    original_record_id: Identifier
    superseding_record_id: Identifier
    reason: str = Field(min_length=1)
    created_utc: UtcTimestamp

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        if value != value.strip():
            raise ValueError("reason must not have leading or trailing whitespace")
        return value

    @model_validator(mode="after")
    def validate_distinct_records(self) -> "SupersedingRecord":
        if self.original_record_id == self.superseding_record_id:
            raise ValueError("a record cannot supersede itself")
        return self


__all__ = [
    "AggregateRecord",
    "ArtifactRecord",
    "ClaimCeiling",
    "ClaimDecision",
    "FrozenRunIdentity",
    "InputContract",
    "LegacySourceReference",
    "ModelIdentity",
    "NegativeEvidenceReason",
    "NegativeEvidenceRecord",
    "Provenance",
    "RecoveryArtifact",
    "RunRecord",
    "SupersedingRecord",
]
