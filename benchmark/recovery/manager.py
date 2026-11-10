"""Checkpoint-boundary recovery and exact missing-job scheduling."""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from pydantic import Field, StrictInt, field_validator, model_validator

from benchmark.registry.error_journal import ErrorJournal
from benchmark.registry.records import (
    ArtifactRecord,
    FrozenRunIdentity,
    RecoveryArtifact,
    RunRecord,
)
from benchmark.registry.run_registry import RegistryInvariantError, RunRegistry
from benchmark.schema.base import ContentHash, FoldId, ModelKey, RunStatus
from benchmark.schema.canonical import (
    CanonicalEncodingError,
    canonical_deserialize,
    canonical_serialize,
)
from benchmark.schema.environment import FrozenSchemaModel
from benchmark.schema.records import ErrorRecord

from .artifact_store import FailureAtomicArtifactStore


_STATE_NAMES = ("model", "optimizer", "scheduler", "stage")
StateRestorer = Callable[[bytes], None]
EnvironmentCompatibility = Callable[[str, str, str], bool]


class RecoveryError(RuntimeError):
    """Recovery failure carrying a machine-readable error record."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        super().__init__(
            f"{error_record.error_code}: recovery failed at "
            f"{error_record.failed_stage} ({error_record.failing_field})"
        )


class ResumeRejectedError(RecoveryError):
    """An in-place resume rejection that requires a newly allocated run."""

    requires_new_run_id = True


class JobKey(FrozenSchemaModel):
    """The dimensions that uniquely identify one declared aggregate job."""

    fold_id: FoldId
    model_key: ModelKey
    seed: StrictInt
    config_hash: ContentHash


class AggregateJobSpec(FrozenSchemaModel):
    """A frozen Cartesian job space under one dataset/protocol/environment."""

    dataset_hash: ContentHash
    protocol_hash: ContentHash
    environment_lock_hash: ContentHash
    environment_compatibility_policy: str = Field(min_length=1)
    fold_ids: Tuple[FoldId, ...] = Field(min_length=1)
    model_keys: Tuple[ModelKey, ...] = Field(min_length=1)
    seeds: Tuple[StrictInt, ...] = Field(min_length=1)
    config_hashes: Tuple[ContentHash, ...] = Field(min_length=1)

    @field_validator("environment_compatibility_policy")
    @classmethod
    def validate_policy(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("environment compatibility policy must be trimmed")
        return value

    @model_validator(mode="after")
    def validate_canonical_dimensions(self) -> "AggregateJobSpec":
        for field_name in ("fold_ids", "model_keys", "seeds", "config_hashes"):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique values")
            if values != tuple(sorted(values)):
                raise ValueError(f"{field_name} must use canonical sorted order")
        return self

    def declared_jobs(self) -> Tuple[JobKey, ...]:
        """Expand the exact, deterministic Cartesian product."""

        return tuple(
            JobKey(
                fold_id=fold_id,
                model_key=model_key,
                seed=seed,
                config_hash=config_hash,
            )
            for fold_id, model_key, seed, config_hash in product(
                self.fold_ids,
                self.model_keys,
                self.seeds,
                self.config_hashes,
            )
        )

    def frozen_identity(self, job: JobKey) -> FrozenRunIdentity:
        """Bind a job key to the shared identity fields in this declaration."""

        return FrozenRunIdentity(
            dataset_hash=self.dataset_hash,
            protocol_hash=self.protocol_hash,
            model_key=job.model_key,
            config_hash=job.config_hash,
            fold_id=job.fold_id,
            seed=job.seed,
            environment_lock_hash=self.environment_lock_hash,
            environment_compatibility_policy=self.environment_compatibility_policy,
        )


class _RecoveryBundle(FrozenSchemaModel):
    """Canonical on-disk payload for four opaque state byte streams."""

    schema_version: str = Field(default="1.0", min_length=1)
    run_id: str = Field(min_length=1)
    frozen_identity: FrozenRunIdentity
    last_completed_stage: str = Field(min_length=1)
    completed_checkpoint_boundary: int = Field(strict=True, ge=0)
    state_hashes: Dict[str, ContentHash]
    state_payloads_base64: Dict[str, str]

    @model_validator(mode="after")
    def validate_states(self) -> "_RecoveryBundle":
        expected = set(_STATE_NAMES)
        if set(self.state_hashes) != expected:
            raise ValueError("state_hashes must contain all recovery state names")
        if set(self.state_payloads_base64) != expected:
            raise ValueError("state payloads must contain all recovery state names")
        for name in _STATE_NAMES:
            try:
                decoded = base64.b64decode(
                    self.state_payloads_base64[name].encode("ascii"),
                    validate=True,
                )
            except (UnicodeEncodeError, binascii.Error) as exc:
                raise ValueError(f"{name} state is not valid base64") from exc
            observed = hashlib.sha256(decoded).hexdigest()
            if observed != self.state_hashes[name]:
                raise ValueError(f"{name} state hash does not match payload")
        return self

    def decoded_states(self) -> Dict[str, bytes]:
        return {
            name: base64.b64decode(self.state_payloads_base64[name], validate=True)
            for name in _STATE_NAMES
        }


@dataclass(frozen=True)
class RestoredRecoveryState:
    """Verified opaque state bytes ready for framework-specific restoration."""

    recovery: RecoveryArtifact
    model_state: bytes
    optimizer_state: bytes
    scheduler_state: bytes
    stage_state: bytes


class RecoveryManager:
    """Persist complete recovery boundaries and reject unsafe in-place resume."""

    def __init__(
        self,
        registry: RunRegistry,
        artifact_store: FailureAtomicArtifactStore,
        *,
        error_journal: Optional[ErrorJournal] = None,
        environment_compatibility: Optional[EnvironmentCompatibility] = None,
    ) -> None:
        self.registry = registry
        self.artifact_store = artifact_store
        self.error_journal = error_journal or artifact_store.error_journal
        self.environment_compatibility = (
            environment_compatibility or self._exact_environment_compatibility
        )

    def save_checkpoint_boundary(
        self,
        *,
        run_id: str,
        recovery_id: str,
        relative_path: str,
        model_state: bytes,
        optimizer_state: bytes,
        scheduler_state: bytes,
        stage_state: bytes,
        last_completed_stage: str,
        completed_checkpoint_boundary: int,
        created_utc: Optional[datetime] = None,
    ) -> RecoveryArtifact:
        """Atomically save all four states as one complete recovery boundary."""

        latest = self.registry.latest_run(run_id)
        if latest.status not in {RunStatus.PLANNED, RunStatus.RUNNING}:
            raise RegistryInvariantError(
                "recovery boundaries can only be saved for active runs"
            )
        try:
            prior = self.registry.latest_recovery(run_id)
        except KeyError:
            prior = None
        if (
            prior is not None
            and completed_checkpoint_boundary
            <= prior.completed_checkpoint_boundary
        ):
            raise RegistryInvariantError(
                "recovery checkpoint boundary must increase monotonically"
            )

        states = {
            "model": self._require_bytes("model_state", model_state),
            "optimizer": self._require_bytes("optimizer_state", optimizer_state),
            "scheduler": self._require_bytes("scheduler_state", scheduler_state),
            "stage": self._require_bytes("stage_state", stage_state),
        }
        state_hashes = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in states.items()
        }
        bundle = _RecoveryBundle(
            run_id=run_id,
            frozen_identity=latest.frozen_identity,
            last_completed_stage=last_completed_stage,
            completed_checkpoint_boundary=completed_checkpoint_boundary,
            state_hashes=state_hashes,
            state_payloads_base64={
                name: base64.b64encode(payload).decode("ascii")
                for name, payload in states.items()
            },
        )
        content = canonical_serialize(bundle)
        reference = self.artifact_store.publish_bytes(
            artifact_id=recovery_id,
            relative_path=relative_path,
            content=content,
            expected_hash=hashlib.sha256(content).hexdigest(),
            media_type="application/vnd.hypercross.recovery+json",
        )
        timestamp = created_utc or datetime.now(timezone.utc)
        self.registry.append_artifact(
            ArtifactRecord(
                artifact=reference,
                artifact_kind="recovery",
                run_id=run_id,
                created_utc=timestamp,
            )
        )
        recovery = RecoveryArtifact(
            recovery_id=recovery_id,
            run_id=run_id,
            frozen_identity=latest.frozen_identity,
            artifact=reference,
            model_state_hash=state_hashes["model"],
            optimizer_state_hash=state_hashes["optimizer"],
            scheduler_state_hash=state_hashes["scheduler"],
            stage_state_hash=state_hashes["stage"],
            last_completed_stage=last_completed_stage,
            completed_checkpoint_boundary=completed_checkpoint_boundary,
            created_utc=timestamp,
        )
        self.registry.append_recovery(recovery)
        return recovery

    def record_interruption(
        self,
        run_id: str,
        *,
        status: RunStatus,
        last_completed_stage: str,
        completed_utc: Optional[datetime] = None,
    ) -> RunRecord:
        """Append a failed/aborted snapshot with the last resumable boundary."""

        if status not in {RunStatus.FAILED, RunStatus.ABORTED}:
            raise ValueError("interruption status must be failed or aborted")
        current = self.registry.latest_run(run_id)
        try:
            recovery_ref = self.registry.latest_recovery(run_id).artifact
        except KeyError:
            recovery_ref = None
        interrupted = RunRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "status": status,
                "recovery_artifact_ref": recovery_ref,
                "last_completed_stage": last_completed_stage,
                "completed_utc": completed_utc or datetime.now(timezone.utc),
            }
        )
        self.registry.append_run(interrupted)
        return interrupted

    def resume(
        self,
        run_id: str,
        observed_identity: FrozenRunIdentity,
        *,
        restore_model: Optional[StateRestorer] = None,
        restore_optimizer: Optional[StateRestorer] = None,
        restore_scheduler: Optional[StateRestorer] = None,
        restore_stage: Optional[StateRestorer] = None,
    ) -> RestoredRecoveryState:
        """Validate all frozen fields before reading or restoring any state."""

        if not isinstance(observed_identity, FrozenRunIdentity):
            raise TypeError("observed_identity must be a FrozenRunIdentity")
        expected_identity = self.registry.run_history(run_id)[0].frozen_identity
        mismatches = self._identity_mismatches(expected_identity, observed_identity)
        if mismatches:
            expected_values = {
                field: expected for field, (expected, _) in mismatches.items()
            }
            observed_values = {
                field: observed for field, (_, observed) in mismatches.items()
            }
            self._reject(
                ResumeRejectedError,
                code="RESUME_FROZEN_IDENTITY_MISMATCH",
                run_id=run_id,
                field="frozen_identity",
                expected=expected_values,
                observed=observed_values,
                mismatched_fields=tuple(mismatches),
                invariants={
                    "state_restoration_started": False,
                    "completed_artifacts_unchanged": True,
                    "requires_new_run_id": True,
                },
            )

        latest = self.registry.latest_run(run_id)
        if latest.status not in {RunStatus.FAILED, RunStatus.ABORTED}:
            self._reject(
                RecoveryError,
                code="RUN_NOT_INTERRUPTED",
                run_id=run_id,
                field="status",
                expected=[RunStatus.FAILED.value, RunStatus.ABORTED.value],
                observed=latest.status.value,
                invariants={
                    "state_restoration_started": False,
                    "completed_artifacts_unchanged": True,
                },
            )

        restorers = (
            restore_model,
            restore_optimizer,
            restore_scheduler,
            restore_stage,
        )
        if any(restorer is not None for restorer in restorers) and not all(
            callable(restorer) for restorer in restorers
        ):
            raise ValueError("provide either all four state restorers or none")

        recovery = self.registry.latest_recovery(run_id)
        states = self._load_verified_bundle(recovery)
        restored = RestoredRecoveryState(
            recovery=recovery,
            model_state=states["model"],
            optimizer_state=states["optimizer"],
            scheduler_state=states["scheduler"],
            stage_state=states["stage"],
        )
        if all(callable(restorer) for restorer in restorers):
            restore_model(restored.model_state)  # type: ignore[misc]
            restore_optimizer(restored.optimizer_state)  # type: ignore[misc]
            restore_scheduler(restored.scheduler_state)  # type: ignore[misc]
            restore_stage(restored.stage_state)  # type: ignore[misc]
        return restored

    def missing_jobs(self, aggregate_spec: AggregateJobSpec) -> Tuple[JobKey, ...]:
        """Return declared jobs with no canonical completed registry record."""

        if not isinstance(aggregate_spec, AggregateJobSpec):
            raise TypeError("aggregate_spec must be an AggregateJobSpec")
        return tuple(
            job
            for job in aggregate_spec.declared_jobs()
            if self.registry.canonical_completed(
                aggregate_spec.frozen_identity(job)
            )
            is None
        )

    def _load_verified_bundle(
        self,
        recovery: RecoveryArtifact,
    ) -> Dict[str, bytes]:
        target = self._artifact_path(recovery.artifact.relative_path)
        try:
            content = target.read_bytes()
        except OSError as exc:
            self._reject(
                RecoveryError,
                code="RECOVERY_ARTIFACT_READ_FAILED",
                run_id=recovery.run_id,
                field="recovery_artifact",
                expected="readable complete recovery artifact",
                observed=repr(exc),
                invariants={"state_restoration_started": False},
            )
        observed_hash = hashlib.sha256(content).hexdigest()
        if observed_hash != recovery.artifact.content_hash:
            self._reject(
                RecoveryError,
                code="RECOVERY_ARTIFACT_HASH_MISMATCH",
                run_id=recovery.run_id,
                field="content_hash",
                expected=recovery.artifact.content_hash,
                observed=observed_hash,
                invariants={"state_restoration_started": False},
            )
        try:
            decoded = canonical_deserialize(content)
            if canonical_serialize(decoded) != content:
                raise CanonicalEncodingError("recovery bundle is not canonical")
            bundle = _RecoveryBundle.model_validate(decoded)
        except Exception as exc:
            self._reject(
                RecoveryError,
                code="RECOVERY_ARTIFACT_INVALID",
                run_id=recovery.run_id,
                field="recovery_artifact",
                expected="valid canonical recovery bundle",
                observed=repr(exc),
                invariants={"state_restoration_started": False},
            )
        if (
            bundle.run_id != recovery.run_id
            or bundle.frozen_identity != recovery.frozen_identity
            or bundle.last_completed_stage != recovery.last_completed_stage
            or bundle.completed_checkpoint_boundary
            != recovery.completed_checkpoint_boundary
        ):
            self._reject(
                RecoveryError,
                code="RECOVERY_METADATA_MISMATCH",
                run_id=recovery.run_id,
                field="recovery_metadata",
                expected={
                    "run_id": recovery.run_id,
                    "frozen_identity": recovery.frozen_identity.model_dump(mode="json"),
                    "last_completed_stage": recovery.last_completed_stage,
                    "completed_checkpoint_boundary": (
                        recovery.completed_checkpoint_boundary
                    ),
                },
                observed={
                    "run_id": bundle.run_id,
                    "frozen_identity": bundle.frozen_identity.model_dump(mode="json"),
                    "last_completed_stage": bundle.last_completed_stage,
                    "completed_checkpoint_boundary": (
                        bundle.completed_checkpoint_boundary
                    ),
                },
                invariants={"state_restoration_started": False},
            )
        manifest_hashes = {
            "model": recovery.model_state_hash,
            "optimizer": recovery.optimizer_state_hash,
            "scheduler": recovery.scheduler_state_hash,
            "stage": recovery.stage_state_hash,
        }
        if bundle.state_hashes != manifest_hashes:
            self._reject(
                RecoveryError,
                code="RECOVERY_STATE_HASH_MANIFEST_MISMATCH",
                run_id=recovery.run_id,
                field="state_hashes",
                expected=manifest_hashes,
                observed=bundle.state_hashes,
                invariants={"state_restoration_started": False},
            )
        return bundle.decoded_states()

    def _artifact_path(self, relative_path: str) -> Path:
        target = (self.artifact_store.root / Path(relative_path)).resolve(strict=False)
        try:
            target.relative_to(self.artifact_store.root)
        except ValueError as exc:
            raise ValueError("recovery artifact path escapes artifact root") from exc
        return target

    def _identity_mismatches(
        self,
        expected: FrozenRunIdentity,
        observed: FrozenRunIdentity,
    ) -> Dict[str, Tuple[object, object]]:
        mismatches: Dict[str, Tuple[object, object]] = {}
        for field in (
            "dataset_hash",
            "protocol_hash",
            "model_key",
            "config_hash",
            "fold_id",
            "seed",
        ):
            expected_value = getattr(expected, field)
            observed_value = getattr(observed, field)
            if expected_value != observed_value:
                mismatches[field] = (expected_value, observed_value)
        if (
            expected.environment_compatibility_policy
            != observed.environment_compatibility_policy
        ):
            mismatches["environment_compatibility_policy"] = (
                expected.environment_compatibility_policy,
                observed.environment_compatibility_policy,
            )
        compatible = self.environment_compatibility(
            expected.environment_compatibility_policy,
            expected.environment_lock_hash,
            observed.environment_lock_hash,
        )
        if not compatible:
            mismatches["environment_lock_hash"] = (
                expected.environment_lock_hash,
                observed.environment_lock_hash,
            )
        return mismatches

    @staticmethod
    def _exact_environment_compatibility(
        policy: str,
        expected_hash: str,
        observed_hash: str,
    ) -> bool:
        return policy == "exact" and expected_hash == observed_hash

    @staticmethod
    def _require_bytes(field: str, value: bytes) -> bytes:
        if not isinstance(value, bytes):
            raise TypeError(f"{field} must be bytes")
        return value

    def _reject(
        self,
        error_type: type[RecoveryError],
        *,
        code: str,
        run_id: str,
        field: str,
        expected: object,
        observed: object,
        mismatched_fields: Tuple[str, ...] = (),
        invariants: Optional[Dict[str, bool]] = None,
    ) -> None:
        affected = {"run_id": run_id}
        if mismatched_fields:
            affected["mismatched_fields"] = ",".join(mismatched_fields)
        record = ErrorRecord(
            error_code=code,
            failed_stage="resume_validation",
            affected_identifiers=affected,
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            invariant_checks=invariants or {},
        )
        failure = error_type(record)
        try:
            self.error_journal.append(record)
        except OSError as journal_error:
            failure.add_note(f"error journal append also failed: {journal_error!r}")
        raise failure


__all__ = [
    "AggregateJobSpec",
    "JobKey",
    "RecoveryError",
    "RecoveryManager",
    "RestoredRecoveryState",
    "ResumeRejectedError",
]
