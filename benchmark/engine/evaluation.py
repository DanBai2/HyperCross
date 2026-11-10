"""Capability-gated, budgeted, one-shot outer evaluation."""

from __future__ import annotations

import importlib
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Tuple, Union
from uuid import uuid4

from pydantic import ValidationError

from benchmark.metrics.predictions import (
    PredictionArtifactWriter,
    PublishedPredictionArtifact,
)
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import ContentHash
from benchmark.schema.canonical import hash_object, normalize_id
from benchmark.schema.evaluation import (
    EvaluationLease,
    FrozenOuterCheckpoint,
    OuterPrediction,
)
from benchmark.schema.predictions import PredictionKey, PredictionRecord
from benchmark.schema.preprocessing import FittedArtifact
from benchmark.schema.protocol import EvaluationBudget, OuterFold, ProtocolManifest
from benchmark.schema.records import ErrorRecord
from benchmark.schema.selection import SelectionDecision


OuterPredictionInput = Union[OuterPrediction, Mapping[str, Any]]
OuterPredictor = Callable[[FrozenOuterCheckpoint, str], OuterPredictionInput]


class OuterLabelVault(Protocol):
    """Capability holder that reveals labels only to the outer gate."""

    def unlock(
        self,
        lease: EvaluationLease,
        sample_ids: Tuple[str, ...],
    ) -> Mapping[str, Any]:
        ...


class _LeaseStatus(str, Enum):
    RESERVED = "reserved"
    EVALUATING = "evaluating"
    READY_TO_COMMIT = "ready_to_commit"
    COMMITTED = "committed"
    CANCELED = "canceled"


@dataclass
class _LeaseEntry:
    lease: EvaluationLease
    status: _LeaseStatus = _LeaseStatus.RESERVED
    prediction_hash: Optional[ContentHash] = None


class OuterEvaluationError(RuntimeError):
    """Outer evaluation rejection with a machine-readable error record."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        super().__init__(
            f"outer evaluation rejected: {error_record.error_code} "
            f"({error_record.failing_field})"
        )


class EvaluationBudgetExceeded(OuterEvaluationError):
    """A reservation would exceed the successful-evaluation budget."""


class EvaluationLeaseError(OuterEvaluationError):
    """A lease is foreign, stale, canceled, or used in the wrong state."""


class OuterEvaluationGate:
    """Own the sole outer-label capability for one immutable protocol.

    Callers can submit only a frozen checkpoint descriptor, immutable fitted
    artifacts, and a label-free prediction callable. The callable runs under
    PyTorch inference mode when PyTorch is installed and never receives labels
    or fitted artifacts. The gate validates fitted-state hashes before and after
    prediction and label access. A successful prediction publication consumes
    the reserved budget once; every failed attempt cancels without consumption.
    """

    def __init__(
        self,
        protocol: ProtocolManifest,
        label_vault: OuterLabelVault,
        prediction_writer: PredictionArtifactWriter,
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        if not isinstance(protocol, ProtocolManifest):
            raise TypeError("protocol must be a ProtocolManifest")
        if not callable(getattr(label_vault, "unlock", None)):
            raise TypeError("label_vault must implement unlock")
        if not isinstance(prediction_writer, PredictionArtifactWriter):
            raise TypeError("prediction_writer must be a PredictionArtifactWriter")
        self._protocol = protocol
        self._label_vault = label_vault
        self._prediction_writer = prediction_writer
        self._error_journal = error_journal or prediction_writer.error_journal
        self._lock = threading.RLock()
        self._leases: dict[str, _LeaseEntry] = {}
        self._consumed = protocol.evaluation_budget.consumed_successful_evaluations

    @property
    def protocol(self) -> ProtocolManifest:
        """Return the immutable protocol bound to this capability gate."""

        return self._protocol

    @property
    def artifact_store(self) -> Any:
        """Return the failure-atomic store used for prediction publication."""

        return self._prediction_writer.artifact_store

    @property
    def budget(self) -> EvaluationBudget:
        """Return an immutable snapshot of configured and committed usage."""

        with self._lock:
            return EvaluationBudget(
                maximum_successful_evaluations=(
                    self._protocol.evaluation_budget.maximum_successful_evaluations
                ),
                consumed_successful_evaluations=self._consumed,
            )

    @property
    def reserved_evaluations(self) -> int:
        """Return capacity held by active, not-yet-committed leases."""

        with self._lock:
            return sum(
                entry.lease.requested_count
                for entry in self._leases.values()
                if entry.status
                in {
                    _LeaseStatus.RESERVED,
                    _LeaseStatus.EVALUATING,
                    _LeaseStatus.READY_TO_COMMIT,
                }
            )

    def reserve(
        self,
        protocol_key: str,
        requested_count: int = 1,
    ) -> EvaluationLease:
        """Reserve successful-evaluation capacity without opening the vault."""

        normalized_key = normalize_id(protocol_key)
        if normalized_key != self._protocol.protocol_key:
            self._reject(
                EvaluationLeaseError,
                code="EVALUATION_PROTOCOL_KEY_MISMATCH",
                field="protocol_key",
                expected=self._protocol.protocol_key,
                observed=normalized_key,
            )
        if isinstance(requested_count, bool) or not isinstance(requested_count, int):
            raise TypeError("requested_count must be a strict integer")
        if requested_count <= 0:
            raise ValueError("requested_count must be positive")
        self._require_protocol_integrity()

        with self._lock:
            maximum = self._protocol.evaluation_budget.maximum_successful_evaluations
            reserved = self.reserved_evaluations
            if self._consumed + reserved + requested_count > maximum:
                self._reject(
                    EvaluationBudgetExceeded,
                    code="EVALUATION_BUDGET_EXCEEDED",
                    field="evaluation_budget",
                    expected={
                        "protocol_key": self._protocol.protocol_key,
                        "maximum_successful_evaluations": maximum,
                    },
                    observed={
                        "consumed_successful_evaluations": self._consumed,
                        "reserved_evaluations": reserved,
                        "requested_count": requested_count,
                    },
                )
            lease = EvaluationLease(
                lease_id=f"evaluation-{uuid4()}",
                protocol_key=self._protocol.protocol_key,
                protocol_hash=self._protocol.protocol_hash,
                requested_count=requested_count,
                reserved_utc=datetime.now(timezone.utc),
            )
            self._leases[lease.lease_id] = _LeaseEntry(lease=lease)
            return lease

    def evaluate(
        self,
        lease: EvaluationLease,
        *,
        outer_fold: OuterFold,
        checkpoint: FrozenOuterCheckpoint,
        selection_decision: SelectionDecision,
        fitted_artifacts: Iterable[FittedArtifact],
        predictor: OuterPredictor,
        artifact_id: str,
        run_id: str,
        schema_version: str = "1.0",
        relative_path: Optional[str] = None,
    ) -> PublishedPredictionArtifact:
        """Infer without labels, unlock truth, publish, then consume once."""

        entry = self._begin_evaluation(lease)
        artifacts = tuple(fitted_artifacts)
        snapshots: Tuple[str, ...] = ()
        try:
            self._validate_evaluation_request(
                outer_fold=outer_fold,
                checkpoint=checkpoint,
                decision=selection_decision,
                fitted_artifacts=artifacts,
            )
            snapshots = self._fitted_snapshots(artifacts)
            outputs = self._predict_without_labels(
                checkpoint,
                outer_fold.outer_test_ids,
                predictor,
            )
            self._require_fitted_unchanged(artifacts, snapshots)

            labels = self._unlock_labels(lease, outer_fold.outer_test_ids)
            self._require_fitted_unchanged(artifacts, snapshots)
            records = tuple(
                PredictionRecord(
                    key=PredictionKey(
                        sample_id=sample_id,
                        fold_id=outer_fold.fold_id,
                        seed=checkpoint.seed,
                        inference_branch=checkpoint.inference_branch,
                    ),
                    target_name=outputs[sample_id].target_name,
                    y_true=labels[sample_id],
                    y_pred=outputs[sample_id].y_pred,
                    probability=outputs[sample_id].probability,
                )
                for sample_id in outer_fold.outer_test_ids
            )
            published = self._prediction_writer.publish_shards(
                artifact_id=artifact_id,
                run_id=run_id,
                expected_sample_ids=outer_fold.outer_test_ids,
                shards=(records,),
                schema_version=schema_version,
                relative_path=relative_path,
            )
            self._mark_ready(entry, published.artifact.content_hash)
            self.commit_successful_evaluation(lease)
            return published
        except Exception:
            self.cancel(lease)
            raise

    def commit_successful_evaluation(self, lease: EvaluationLease) -> EvaluationBudget:
        """Commit a gate-published result once; duplicate commits are idempotent."""

        with self._lock:
            entry = self._require_owned_lease(lease)
            if entry.status is _LeaseStatus.COMMITTED:
                return self.budget
            if entry.status is not _LeaseStatus.READY_TO_COMMIT or not entry.prediction_hash:
                self._reject(
                    EvaluationLeaseError,
                    code="EVALUATION_LEASE_NOT_SUCCESSFUL",
                    field="lease.status",
                    expected=_LeaseStatus.READY_TO_COMMIT.value,
                    observed=entry.status.value,
                    lease=lease,
                )
            maximum = self._protocol.evaluation_budget.maximum_successful_evaluations
            if self._consumed + lease.requested_count > maximum:
                self._reject(
                    EvaluationBudgetExceeded,
                    code="EVALUATION_BUDGET_COMMIT_EXCEEDED",
                    field="evaluation_budget",
                    expected=maximum,
                    observed=self._consumed + lease.requested_count,
                    lease=lease,
                )
            self._consumed += lease.requested_count
            entry.status = _LeaseStatus.COMMITTED
            return self.budget

    def cancel(self, lease: EvaluationLease) -> None:
        """Release an unsuccessful reservation without consuming the budget."""

        with self._lock:
            entry = self._require_owned_lease(lease)
            if entry.status is _LeaseStatus.COMMITTED:
                return
            entry.status = _LeaseStatus.CANCELED
            entry.prediction_hash = None

    def _begin_evaluation(self, lease: EvaluationLease) -> _LeaseEntry:
        with self._lock:
            entry = self._require_owned_lease(lease)
            if lease.requested_count != 1:
                self._reject(
                    EvaluationLeaseError,
                    code="ONE_SHOT_LEASE_REQUIRED",
                    field="requested_count",
                    expected=1,
                    observed=lease.requested_count,
                    lease=lease,
                )
            if entry.status is not _LeaseStatus.RESERVED:
                self._reject(
                    EvaluationLeaseError,
                    code="EVALUATION_LEASE_NOT_RESERVED",
                    field="lease.status",
                    expected=_LeaseStatus.RESERVED.value,
                    observed=entry.status.value,
                    lease=lease,
                )
            entry.status = _LeaseStatus.EVALUATING
            return entry

    def _require_owned_lease(self, lease: EvaluationLease) -> _LeaseEntry:
        if not isinstance(lease, EvaluationLease):
            raise TypeError("lease must be an EvaluationLease")
        entry = self._leases.get(lease.lease_id)
        if entry is None or entry.lease is not lease:
            self._reject(
                EvaluationLeaseError,
                code="EVALUATION_LEASE_NOT_OWNED",
                field="lease_id",
                expected="an active capability issued by this gate",
                observed=lease.lease_id,
                lease=lease,
            )
        assert entry is not None
        return entry

    def _validate_evaluation_request(
        self,
        *,
        outer_fold: OuterFold,
        checkpoint: FrozenOuterCheckpoint,
        decision: SelectionDecision,
        fitted_artifacts: Tuple[FittedArtifact, ...],
    ) -> None:
        self._require_protocol_integrity()
        declared_fold = next(
            (fold for fold in self._protocol.folds if fold.fold_id == outer_fold.fold_id),
            None,
        )
        if declared_fold is None or declared_fold != outer_fold:
            self._reject(
                OuterEvaluationError,
                code="OUTER_FOLD_NOT_IN_PROTOCOL",
                field="outer_fold",
                expected="an exact fold from the frozen Protocol_Manifest",
                observed=outer_fold.fold_id,
            )
        try:
            decision.require_integrity()
        except ValueError as exc:
            self._reject(
                OuterEvaluationError,
                code="SELECTION_DECISION_NOT_FROZEN",
                field="selection_decision.decision_hash",
                expected="an intact frozen Selection_Decision",
                observed=str(exc),
            )
        if decision.fold_id != outer_fold.fold_id:
            self._reject(
                OuterEvaluationError,
                code="EVALUATION_SELECTION_FOLD_MISMATCH",
                field="selection_decision.fold_id",
                expected=outer_fold.fold_id,
                observed=decision.fold_id,
            )
        forbidden = sorted(
            set(decision.source_sample_ids) & set(outer_fold.outer_test_ids),
            key=lambda item: item.encode("utf-8"),
        )
        if forbidden:
            self._reject(
                OuterEvaluationError,
                code="EVALUATION_SELECTION_OUTER_EVIDENCE",
                field="selection_decision.source_sample_ids",
                expected=[],
                observed=forbidden,
                unexpected_ids=tuple(forbidden),
            )
        if checkpoint.fold_id != outer_fold.fold_id:
            self._reject(
                OuterEvaluationError,
                code="EVALUATION_CHECKPOINT_FOLD_MISMATCH",
                field="checkpoint.fold_id",
                expected=outer_fold.fold_id,
                observed=checkpoint.fold_id,
            )
        if checkpoint.selection_decision_hash != decision.decision_hash:
            self._reject(
                OuterEvaluationError,
                code="EVALUATION_CHECKPOINT_SELECTION_MISMATCH",
                field="checkpoint.selection_decision_hash",
                expected=decision.decision_hash,
                observed=checkpoint.selection_decision_hash,
            )

        outer_train = set(outer_fold.outer_train_ids)
        outer_test = set(outer_fold.outer_test_ids)
        for artifact in fitted_artifacts:
            if artifact.protocol_hash != self._protocol.protocol_hash:
                self._reject(
                    OuterEvaluationError,
                    code="EVALUATION_FITTED_PROTOCOL_MISMATCH",
                    field="fitted_artifact.protocol_hash",
                    expected=self._protocol.protocol_hash,
                    observed=artifact.protocol_hash,
                    artifact_id=artifact.artifact_id,
                )
            if artifact.fold_id != outer_fold.fold_id:
                self._reject(
                    OuterEvaluationError,
                    code="EVALUATION_FITTED_FOLD_MISMATCH",
                    field="fitted_artifact.fold_id",
                    expected=outer_fold.fold_id,
                    observed=artifact.fold_id,
                    artifact_id=artifact.artifact_id,
                )
            fit_ids = set(artifact.fit_sample_ids)
            invalid = sorted(
                (fit_ids - outer_train) | (fit_ids & outer_test),
                key=lambda item: item.encode("utf-8"),
            )
            if invalid:
                self._reject(
                    OuterEvaluationError,
                    code="EVALUATION_FITTED_STATE_LEAKAGE",
                    field="fitted_artifact.fit_sample_ids",
                    expected=list(outer_fold.outer_train_ids),
                    observed=invalid,
                    unexpected_ids=tuple(invalid),
                    artifact_id=artifact.artifact_id,
                )

    def _predict_without_labels(
        self,
        checkpoint: FrozenOuterCheckpoint,
        sample_ids: Tuple[str, ...],
        predictor: OuterPredictor,
    ) -> dict[str, OuterPrediction]:
        if not callable(predictor):
            raise TypeError("predictor must be callable")
        outputs: dict[str, OuterPrediction] = {}
        with self._gradient_context():
            for sample_id in sample_ids:
                raw = predictor(checkpoint, sample_id)
                try:
                    output = (
                        raw
                        if isinstance(raw, OuterPrediction)
                        else OuterPrediction.model_validate(raw)
                    )
                except (TypeError, ValidationError) as exc:
                    raise ValueError(
                        f"predictor returned invalid output for {sample_id!r}"
                    ) from exc
                outputs[sample_id] = output
        return outputs

    @staticmethod
    def _gradient_context() -> Any:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return nullcontext()
        return torch.inference_mode()

    def _unlock_labels(
        self,
        lease: EvaluationLease,
        sample_ids: Tuple[str, ...],
    ) -> dict[str, Any]:
        raw = self._label_vault.unlock(lease, sample_ids)
        if not isinstance(raw, Mapping):
            raise TypeError("outer label vault must return a mapping")
        labels: dict[str, Any] = {}
        for sample_id, value in raw.items():
            if not isinstance(sample_id, str):
                raise TypeError("outer label keys must be Sample_ID strings")
            normalized = normalize_id(sample_id)
            if normalized in labels:
                raise ValueError("outer label IDs collide after normalization")
            labels[normalized] = value
        expected = set(sample_ids)
        if set(labels) != expected:
            missing = sorted(expected - set(labels), key=lambda item: item.encode("utf-8"))
            unexpected = sorted(set(labels) - expected, key=lambda item: item.encode("utf-8"))
            raise ValueError(
                "outer label coverage mismatch; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        return labels

    @staticmethod
    def _fitted_snapshots(artifacts: Tuple[FittedArtifact, ...]) -> Tuple[str, ...]:
        snapshots = []
        for artifact in artifacts:
            if hash_object(artifact.config) != artifact.config_hash:
                raise ValueError("fitted artifact config changed after validation")
            if hash_object(artifact.state) != artifact.state_hash:
                raise ValueError("fitted artifact state changed after validation")
            snapshots.append(hash_object(artifact))
        return tuple(snapshots)

    @staticmethod
    def _require_fitted_unchanged(
        artifacts: Tuple[FittedArtifact, ...],
        snapshots: Tuple[str, ...],
    ) -> None:
        if OuterEvaluationGate._fitted_snapshots(artifacts) != snapshots:
            raise ValueError("fitted state changed during outer evaluation")

    def _mark_ready(self, entry: _LeaseEntry, prediction_hash: str) -> None:
        with self._lock:
            if entry.status is not _LeaseStatus.EVALUATING:
                raise RuntimeError("evaluation lease changed state during publication")
            entry.prediction_hash = prediction_hash
            entry.status = _LeaseStatus.READY_TO_COMMIT

    def _require_protocol_integrity(self) -> None:
        if self._protocol.protocol_hash != hash_object(self._protocol.hash_payload()):
            self._reject(
                OuterEvaluationError,
                code="EVALUATION_PROTOCOL_MUTATED",
                field="protocol_hash",
                expected=self._protocol.protocol_hash,
                observed=hash_object(self._protocol.hash_payload()),
            )

    def _reject(
        self,
        error_type: type[OuterEvaluationError],
        *,
        code: str,
        field: str,
        expected: Any,
        observed: Any,
        lease: Optional[EvaluationLease] = None,
        unexpected_ids: Tuple[str, ...] = (),
        artifact_id: Optional[str] = None,
    ) -> None:
        identifiers = {"protocol_key": self._protocol.protocol_key}
        if lease is not None:
            identifiers["evaluation_lease_id"] = lease.lease_id
        if artifact_id is not None:
            identifiers["fitted_artifact_id"] = artifact_id
        record = ErrorRecord(
            error_code=code,
            failed_stage="outer_evaluation_gate",
            affected_identifiers=identifiers,
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            unexpected_ids=unexpected_ids,
            invariant_checks={
                "outer_labels_not_loaded_before_budget_reservation": True,
                "successful_budget_not_consumed": True,
                "completed_prediction_artifacts_unchanged": True,
            },
        )
        self._error_journal.append(record)
        raise error_type(record)


__all__ = [
    "EvaluationBudgetExceeded",
    "EvaluationLeaseError",
    "OuterEvaluationError",
    "OuterEvaluationGate",
    "OuterLabelVault",
    "OuterPredictor",
]
