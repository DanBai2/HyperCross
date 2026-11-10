"""Durable append-only storage for runs, artifacts, and aggregates."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from uuid import uuid4

from pydantic import ValidationError

from benchmark.schema.base import RunStatus
from benchmark.schema.canonical import normalize_id

from .records import (
    AggregateRecord,
    ArtifactRecord,
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    NegativeEvidenceRecord,
    Provenance,
    RecoveryArtifact,
    RunRecord,
    SupersedingRecord,
)
from benchmark.schema.records import ResourceBudget


class RegistryError(RuntimeError):
    """Base error for append-only registry operations."""


class RegistryInvariantError(RegistryError):
    """Raised before append when a registry invariant would be violated."""


class RegistryCorruptionError(RegistryError):
    """Raised when a persisted line cannot be validated."""


class RegistryWriteError(RegistryError):
    """Raised when a durable append cannot be completed."""


_ALLOWED_TRANSITIONS = {
    RunStatus.PLANNED: {
        RunStatus.RUNNING,
        RunStatus.COMPLETE,
        RunStatus.FAILED,
        RunStatus.ABORTED,
    },
    RunStatus.RUNNING: {
        RunStatus.RUNNING,
        RunStatus.COMPLETE,
        RunStatus.FAILED,
        RunStatus.ABORTED,
    },
    RunStatus.FAILED: {RunStatus.RUNNING},
    RunStatus.ABORTED: {RunStatus.RUNNING},
    RunStatus.COMPLETE: set(),
    RunStatus.IMPORTED: set(),
}


class AppendOnlyRunRegistry:
    """Persist registry entries as durable JSON Lines without rewrite APIs.

    Every status change appends a complete run snapshot. Run allocation appends
    the initial ``PLANNED`` snapshot before returning its identifier. Completed
    runs, artifact identifiers, and aggregate identifiers are immutable.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        # Validate existing bytes immediately so callers never append behind a
        # malformed or semantically inconsistent registry history.
        with self._lock:
            self._load_state()

    @property
    def path(self) -> Path:
        return self._path

    def allocate_run(
        self,
        *,
        frozen_identity: FrozenRunIdentity,
        model_identity: ModelIdentity,
        input_contract: InputContract,
        provenance: Provenance,
        resource_budget: ResourceBudget,
        claim_decision: ClaimDecision,
        started_utc: Optional[datetime] = None,
    ) -> RunRecord:
        """Allocate and durably persist a unique planned run before training."""

        timestamp = started_utc or datetime.now(timezone.utc)
        with self._lock:
            state = self._load_state()
            while True:
                run_id = str(uuid4())
                if run_id not in state.runs:
                    break
            record = RunRecord(
                run_id=run_id,
                frozen_identity=frozen_identity,
                model_identity=model_identity,
                input_contract=input_contract,
                provenance=provenance,
                resource_budget=resource_budget,
                status=RunStatus.PLANNED,
                claim_decision=claim_decision,
                started_utc=timestamp,
            )
            self._append_entry("run", record)
            return record

    def append_imported(self, record: RunRecord) -> None:
        """Append one fully validated legacy run without a planned allocation.

        Callers must validate and snapshot the source before constructing the
        record. This method never assigns identifiers and accepts only a first,
        terminal ``IMPORTED`` snapshot.
        """

        if not isinstance(record, RunRecord):
            raise TypeError("record must be a RunRecord")
        record = RunRecord.model_validate(record.model_dump(mode="python"))
        if record.status is not RunStatus.IMPORTED:
            raise RegistryInvariantError("legacy append requires IMPORTED status")
        with self._lock:
            state = self._load_state()
            if record.run_id in state.runs:
                raise RegistryInvariantError(
                    f"run {record.run_id!r} is already registered"
                )
            self._append_entry("run", record)

    def append_run(self, record: RunRecord) -> None:
        """Append a valid status snapshot; never alter an earlier snapshot."""

        if not isinstance(record, RunRecord):
            raise TypeError("record must be a RunRecord")
        record = RunRecord.model_validate(record.model_dump(mode="python"))
        with self._lock:
            state = self._load_state()
            history = state.runs.get(record.run_id)
            if not history:
                raise RegistryInvariantError(
                    "run must be allocated before status snapshots are appended"
                )
            previous = history[-1]
            self._validate_run_transition(previous, record)
            if record.status is RunStatus.COMPLETE:
                self._validate_completed_artifacts(record, state.artifacts)
            self._append_entry("run", record)

    def append_artifact(self, record: ArtifactRecord) -> None:
        """Register one immutable artifact identifier exactly once."""

        if not isinstance(record, ArtifactRecord):
            raise TypeError("record must be an ArtifactRecord")
        record = ArtifactRecord.model_validate(record.model_dump(mode="python"))
        with self._lock:
            state = self._load_state()
            artifact_id = record.artifact.artifact_id
            if artifact_id in state.artifacts:
                raise RegistryInvariantError(
                    f"artifact {artifact_id!r} is already registered and cannot be overwritten"
                )
            if record.run_id is not None and record.run_id not in state.runs:
                raise RegistryInvariantError(
                    f"artifact references unknown run {record.run_id!r}"
                )
            self._append_entry("artifact", record)

    def append_recovery(self, record: RecoveryArtifact) -> None:
        """Register one complete checkpoint-boundary recovery bundle."""

        if not isinstance(record, RecoveryArtifact):
            raise TypeError("record must be a RecoveryArtifact")
        record = RecoveryArtifact.model_validate(record.model_dump(mode="python"))
        with self._lock:
            state = self._load_state()
            self._validate_recovery(record, state)
            self._append_entry("recovery", record)

    def recovery_artifacts(self, run_id: str) -> Tuple[RecoveryArtifact, ...]:
        """Return complete recovery bundles in checkpoint-boundary order."""

        with self._lock:
            state = self._load_state()
            if run_id not in state.runs:
                raise KeyError(run_id)
            records = state.recoveries.get(run_id, [])
            return tuple(records)

    def latest_recovery(self, run_id: str) -> RecoveryArtifact:
        """Return the last complete recovery bundle for an allocated run."""

        records = self.recovery_artifacts(run_id)
        if not records:
            raise KeyError(run_id)
        return records[-1]

    @staticmethod
    def _validate_recovery(
        record: RecoveryArtifact,
        state: "_RegistryState",
    ) -> None:
        history = state.runs.get(record.run_id)
        if not history:
            raise RegistryInvariantError(
                f"recovery artifact references unknown run {record.run_id!r}"
            )
        if history[0].frozen_identity != record.frozen_identity:
            raise RegistryInvariantError(
                "recovery artifact frozen identity differs from allocated run"
            )
        if record.recovery_id in state.recovery_ids:
            raise RegistryInvariantError(
                f"recovery artifact {record.recovery_id!r} is already registered"
            )
        registered = state.artifacts.get(record.artifact.artifact_id)
        if registered is None or registered.artifact != record.artifact:
            raise RegistryInvariantError(
                "recovery bundle must reference an identically registered artifact"
            )
        if registered.artifact_kind != "recovery":
            raise RegistryInvariantError(
                "recovery bundle artifact must have artifact_kind 'recovery'"
            )
        if registered.run_id != record.run_id:
            raise RegistryInvariantError("recovery bundle belongs to another run")
        prior = state.recoveries.get(record.run_id, [])
        if prior and (
            record.completed_checkpoint_boundary
            <= prior[-1].completed_checkpoint_boundary
        ):
            raise RegistryInvariantError(
                "recovery checkpoint boundary must increase monotonically"
            )

    def append_aggregate(self, record: AggregateRecord) -> None:
        """Register one immutable aggregate backed by completed run members."""

        if not isinstance(record, AggregateRecord):
            raise TypeError("record must be an AggregateRecord")
        record = AggregateRecord.model_validate(record.model_dump(mode="python"))
        with self._lock:
            state = self._load_state()
            if record.aggregate_id in state.aggregates:
                raise RegistryInvariantError(
                    f"aggregate {record.aggregate_id!r} is already registered"
                )
            for run_id in record.member_run_ids:
                history = state.runs.get(run_id)
                if not history or history[-1].status is not RunStatus.COMPLETE:
                    raise RegistryInvariantError(
                        f"aggregate member {run_id!r} is not a completed run"
                    )
                prediction = history[-1].prediction_artifact_ref
                assert prediction is not None  # guaranteed by RunRecord validation
                if record.prediction_hashes[run_id] != prediction.content_hash:
                    raise RegistryInvariantError(
                        f"aggregate prediction hash does not match run {run_id!r}"
                    )
            self._append_entry("aggregate", record)

    def append_negative_evidence(
        self, record: NegativeEvidenceRecord
    ) -> None:
        """Append one tested candidate outcome without rewriting prior evidence."""

        if not isinstance(record, NegativeEvidenceRecord):
            raise TypeError("record must be a NegativeEvidenceRecord")
        record = NegativeEvidenceRecord.model_validate(record.model_dump(mode="python"))
        with self._lock:
            state = self._load_state()
            if record.evidence_id in state.negative_evidence:
                raise RegistryInvariantError(
                    f"negative evidence {record.evidence_id!r} is already registered"
                )
            candidate = state.aggregates.get(record.candidate_aggregate_id)
            if candidate is None:
                raise RegistryInvariantError("candidate aggregate is not registered")
            if candidate.member_run_ids != record.candidate_run_ids:
                raise RegistryInvariantError(
                    "negative evidence candidate runs differ from aggregate membership"
                )
            if record.control_aggregate_id is not None:
                control = state.aggregates.get(record.control_aggregate_id)
                if control is None:
                    raise RegistryInvariantError("control aggregate is not registered")
                if control.member_run_ids != record.control_run_ids:
                    raise RegistryInvariantError(
                        "negative evidence control runs differ from aggregate membership"
                    )
            self._append_entry("negative_evidence", record)

    def negative_evidence(
        self, *, candidate_aggregate_id: Optional[str] = None
    ) -> Tuple[NegativeEvidenceRecord, ...]:
        """Read negative evidence in append order, optionally for one candidate."""

        with self._lock:
            records = tuple(self._load_state().negative_evidence.values())
        if candidate_aggregate_id is None:
            return records
        return tuple(
            record
            for record in records
            if record.candidate_aggregate_id == candidate_aggregate_id
        )

    def supersede(
        self,
        original_record_id: str,
        superseding_record_id: str,
        reason: str,
        *,
        created_utc: Optional[datetime] = None,
    ) -> SupersedingRecord:
        """Append a correction link without rewriting either evidence record.

        Run records may supersede run records and aggregate records may
        supersede aggregate records. An original record can have only one
        direct replacement; later corrections form a traceable chain.
        """

        record = SupersedingRecord(
            original_record_id=original_record_id,
            superseding_record_id=superseding_record_id,
            reason=reason,
            created_utc=created_utc or datetime.now(timezone.utc),
        )
        with self._lock:
            state = self._load_state()
            self._validate_supersession(record, state)
            self._append_entry("supersession", record)
        return record

    def get_supersession(self, original_record_id: str) -> SupersedingRecord:
        """Return the direct correction link for an original record."""

        with self._lock:
            try:
                return self._load_state().supersessions[original_record_id]
            except KeyError:
                raise KeyError(original_record_id) from None

    def completed_runs(
        self,
        frozen_identity: FrozenRunIdentity,
    ) -> Tuple[RunRecord, ...]:
        """Return every completed duplicate for an exact frozen identity.

        Results use the same deterministic order as canonical selection while
        retaining all duplicate records for audit.
        """

        if not isinstance(frozen_identity, FrozenRunIdentity):
            raise TypeError("frozen_identity must be a FrozenRunIdentity")
        with self._lock:
            state = self._load_state()
            completed = [
                history[-1]
                for history in state.runs.values()
                if history[-1].status is RunStatus.COMPLETE
                and history[-1].frozen_identity == frozen_identity
            ]
        return tuple(sorted(completed, key=self._completed_sort_key))

    def canonical_completed(
        self,
        frozen_identity: FrozenRunIdentity,
    ) -> Optional[RunRecord]:
        """Select at most one completed record for an exact frozen identity."""

        completed = self.completed_runs(frozen_identity)
        return completed[0] if completed else None

    @staticmethod
    def _completed_sort_key(record: RunRecord) -> tuple[datetime, bytes]:
        assert record.completed_utc is not None
        return (
            record.completed_utc,
            normalize_id(record.run_id).encode("utf-8"),
        )

    @classmethod
    def _validate_supersession(
        cls,
        record: SupersedingRecord,
        state: "_RegistryState",
    ) -> None:
        if record.original_record_id in state.supersessions:
            raise RegistryInvariantError(
                f"record {record.original_record_id!r} is already superseded"
            )
        original_kind = cls._evidence_record_kind(state, record.original_record_id)
        superseding_kind = cls._evidence_record_kind(
            state, record.superseding_record_id
        )
        if original_kind is None:
            raise RegistryInvariantError(
                f"original record {record.original_record_id!r} is not registered"
            )
        if superseding_kind is None:
            raise RegistryInvariantError(
                f"superseding record {record.superseding_record_id!r} is not registered"
            )
        if original_kind == "ambiguous" or superseding_kind == "ambiguous":
            raise RegistryInvariantError("evidence record identifier is ambiguous")
        if original_kind != superseding_kind:
            raise RegistryInvariantError(
                "original and superseding records must have the same record kind"
            )

    @staticmethod
    def _evidence_record_kind(state: "_RegistryState", record_id: str) -> Optional[str]:
        kinds = []
        if record_id in state.runs:
            kinds.append("run")
        if record_id in state.aggregates:
            kinds.append("aggregate")
        if len(kinds) > 1:
            return "ambiguous"
        return kinds[0] if kinds else None

    def run_history(self, run_id: str) -> Tuple[RunRecord, ...]:
        with self._lock:
            history = self._load_state().runs.get(run_id)
            if history is None:
                raise KeyError(run_id)
            return tuple(history)

    def latest_run(self, run_id: str) -> RunRecord:
        return self.run_history(run_id)[-1]

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self._lock:
            try:
                return self._load_state().artifacts[artifact_id]
            except KeyError:
                raise KeyError(artifact_id) from None

    def get_aggregate(self, aggregate_id: str) -> AggregateRecord:
        with self._lock:
            try:
                return self._load_state().aggregates[aggregate_id]
            except KeyError:
                raise KeyError(aggregate_id) from None

    @staticmethod
    def _validate_run_transition(previous: RunRecord, current: RunRecord) -> None:
        if current.status not in _ALLOWED_TRANSITIONS[previous.status]:
            raise RegistryInvariantError(
                f"run status transition {previous.status.value!r} -> "
                f"{current.status.value!r} is not allowed"
            )
        stable_fields = (
            "schema_version",
            "frozen_identity",
            "model_identity",
            "input_contract",
            "provenance",
            "resource_budget",
            "started_utc",
        )
        changed = [
            field
            for field in stable_fields
            if getattr(previous, field) != getattr(current, field)
        ]
        if changed:
            raise RegistryInvariantError(
                "run snapshot changed frozen fields: " + ", ".join(changed)
            )

    @staticmethod
    def _validate_completed_artifacts(
        record: RunRecord,
        artifacts: Dict[str, ArtifactRecord],
    ) -> None:
        references = (
            *record.transform_artifact_refs,
            record.checkpoint_ref,
            record.prediction_artifact_ref,
        )
        for reference in references:
            if reference is None:
                continue
            registered = artifacts.get(reference.artifact_id)
            if registered is None:
                raise RegistryInvariantError(
                    f"complete run references unregistered artifact {reference.artifact_id!r}"
                )
            if registered.artifact != reference:
                raise RegistryInvariantError(
                    f"complete run artifact reference differs from registry for "
                    f"{reference.artifact_id!r}"
                )
            if registered.run_id not in {None, record.run_id}:
                raise RegistryInvariantError(
                    f"artifact {reference.artifact_id!r} belongs to another run"
                )

    def _append_entry(self, kind: str, record: Any) -> None:
        payload = json.dumps(
            {"entry_kind": kind, "record": record.model_dump(mode="json")},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self._path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("registry append made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise RegistryWriteError(f"could not append registry {self._path}") from exc

    def _load_state(self) -> "_RegistryState":
        state = _RegistryState()
        if not self._path.exists():
            return state
        try:
            with self._path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith("\n"):
                        raise RegistryCorruptionError(
                            f"registry line {line_number} is not a complete record"
                        )
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RegistryCorruptionError(
                            f"registry line {line_number} is invalid JSON"
                        ) from exc
                    self._apply_loaded_entry(state, entry, line_number)
        except OSError as exc:
            raise RegistryCorruptionError(f"could not read registry {self._path}") from exc
        return state

    def _apply_loaded_entry(
        self,
        state: "_RegistryState",
        entry: Any,
        line_number: int,
    ) -> None:
        if not isinstance(entry, dict) or set(entry) != {"entry_kind", "record"}:
            raise RegistryCorruptionError(
                f"registry line {line_number} has an invalid envelope"
            )
        kind = entry["entry_kind"]
        model = {
            "run": RunRecord,
            "artifact": ArtifactRecord,
            "recovery": RecoveryArtifact,
            "aggregate": AggregateRecord,
            "negative_evidence": NegativeEvidenceRecord,
            "supersession": SupersedingRecord,
        }.get(kind)
        if model is None:
            raise RegistryCorruptionError(
                f"registry line {line_number} has unknown entry kind {kind!r}"
            )
        try:
            record = model.model_validate(entry["record"])
        except ValidationError as exc:
            raise RegistryCorruptionError(
                f"registry line {line_number} does not match the {kind} schema"
            ) from exc

        if kind == "run":
            history = state.runs.setdefault(record.run_id, [])
            if not history:
                if record.status not in {RunStatus.PLANNED, RunStatus.IMPORTED}:
                    raise RegistryCorruptionError(
                        f"registry line {line_number} starts a run without allocation"
                    )
            else:
                try:
                    self._validate_run_transition(history[-1], record)
                except RegistryInvariantError as exc:
                    raise RegistryCorruptionError(
                        f"registry line {line_number} violates run history: {exc}"
                    ) from exc
            if record.status is RunStatus.COMPLETE:
                try:
                    self._validate_completed_artifacts(record, state.artifacts)
                except RegistryInvariantError as exc:
                    raise RegistryCorruptionError(
                        f"registry line {line_number} violates artifact membership: {exc}"
                    ) from exc
            history.append(record)
            return

        if kind == "artifact":
            artifact_id = record.artifact.artifact_id
            if artifact_id in state.artifacts:
                raise RegistryCorruptionError(
                    f"registry line {line_number} duplicates artifact {artifact_id!r}"
                )
            state.artifacts[artifact_id] = record
            return

        if kind == "recovery":
            try:
                self._validate_recovery(record, state)
            except RegistryInvariantError as exc:
                raise RegistryCorruptionError(
                    f"registry line {line_number} violates recovery history: {exc}"
                ) from exc
            state.recoveries.setdefault(record.run_id, []).append(record)
            state.recovery_ids.add(record.recovery_id)
            return

        if kind == "aggregate":
            if record.aggregate_id in state.aggregates:
                raise RegistryCorruptionError(
                    f"registry line {line_number} duplicates aggregate {record.aggregate_id!r}"
                )
            for run_id in record.member_run_ids:
                history = state.runs.get(run_id)
                if not history or history[-1].status is not RunStatus.COMPLETE:
                    raise RegistryCorruptionError(
                        f"registry line {line_number} references incomplete run {run_id!r}"
                    )
                prediction = history[-1].prediction_artifact_ref
                assert prediction is not None
                if record.prediction_hashes[run_id] != prediction.content_hash:
                    raise RegistryCorruptionError(
                        f"registry line {line_number} has a mismatched prediction hash"
                    )
            state.aggregates[record.aggregate_id] = record
            return

        if kind == "negative_evidence":
            if record.evidence_id in state.negative_evidence:
                raise RegistryCorruptionError(
                    f"registry line {line_number} duplicates negative evidence "
                    f"{record.evidence_id!r}"
                )
            candidate = state.aggregates.get(record.candidate_aggregate_id)
            if candidate is None or candidate.member_run_ids != record.candidate_run_ids:
                raise RegistryCorruptionError(
                    f"registry line {line_number} has invalid candidate evidence membership"
                )
            if record.control_aggregate_id is not None:
                control = state.aggregates.get(record.control_aggregate_id)
                if control is None or control.member_run_ids != record.control_run_ids:
                    raise RegistryCorruptionError(
                        f"registry line {line_number} has invalid control evidence membership"
                    )
            state.negative_evidence[record.evidence_id] = record
            return

        try:
            self._validate_supersession(record, state)
        except RegistryInvariantError as exc:
            raise RegistryCorruptionError(
                f"registry line {line_number} violates supersession history: {exc}"
            ) from exc
        state.supersessions[record.original_record_id] = record


class _RegistryState:
    def __init__(self) -> None:
        self.runs: Dict[str, list[RunRecord]] = {}
        self.artifacts: Dict[str, ArtifactRecord] = {}
        self.recoveries: Dict[str, list[RecoveryArtifact]] = {}
        self.recovery_ids: set[str] = set()
        self.aggregates: Dict[str, AggregateRecord] = {}
        self.negative_evidence: Dict[str, NegativeEvidenceRecord] = {}
        self.supersessions: Dict[str, SupersedingRecord] = {}


# Keep the concise design name as the public default.
RunRegistry = AppendOnlyRunRegistry


__all__ = [
    "AppendOnlyRunRegistry",
    "RegistryCorruptionError",
    "RegistryError",
    "RegistryInvariantError",
    "RegistryWriteError",
    "RunRegistry",
]
