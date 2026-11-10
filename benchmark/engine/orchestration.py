"""Unified, recoverable orchestration of one frozen outer-fold run."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Tuple

from pydantic import Field, field_validator, model_validator

from benchmark.engine.evaluation import OuterEvaluationGate
from benchmark.engine.selection import SelectionController
from benchmark.engine.training import (
    InnerTrainingArtifact,
    InnerTrainingBackend,
    InnerTrainingStatus,
    TrainingEngine,
    TrainingStateSnapshot,
)
from benchmark.metrics.predictions import PublishedPredictionArtifact
from benchmark.protocols import PublishedProtocol
from benchmark.recovery import RecoveryManager, RestoredRecoveryState
from benchmark.registry import ArtifactRecord, FrozenRunIdentity, RunRecord, RunRegistry
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema import (
    FitBoundary,
    FitBoundaryScope,
    FittedArtifact,
    FrozenOuterCheckpoint,
    FrozenRunConfig,
    OuterFold,
    OuterRefitRequest,
    ProtocolManifest,
    RunStatus,
    SelectionCandidate,
    SelectionDecision,
    canonical_deserialize,
    canonical_serialize,
)
from benchmark.schema.environment import FrozenSchemaModel
from benchmark.schema.predictions import PredictionArtifact
from benchmark.schema.records import ArtifactReference, ErrorRecord
from benchmark.transforms import PreprocessingManager


class OrchestrationStage(str, Enum):
    """Durable major-stage boundaries for one outer-fold run."""

    INNER_PREPROCESSING = "inner_preprocessing"
    INNER_TRAINING = "inner_training"
    SELECTION_FROZEN = "selection_frozen"
    OUTER_PREPROCESSING = "outer_preprocessing"
    OUTER_REFIT = "outer_refit"
    OUTER_EVALUATION = "outer_evaluation"


_STAGE_ORDER = {stage: index for index, stage in enumerate(OrchestrationStage)}


class SelectionEvidence(FrozenSchemaModel):
    """Complete label-free inner evidence needed to freeze a decision."""

    candidates: Tuple[SelectionCandidate, ...] = Field(min_length=1)
    candidate_scores: Mapping[str, float]
    source_sample_ids: Tuple[str, ...] = Field(min_length=1)
    source_prediction_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OuterRefitResult(FrozenSchemaModel):
    """Frozen outer checkpoint and restorable framework state after refit."""

    checkpoint: FrozenOuterCheckpoint
    state_snapshot: TrainingStateSnapshot


class OrchestrationRequest(FrozenSchemaModel):
    """Immutable inputs for one already allocated run."""

    run_id: str = Field(min_length=1)
    published_protocol: PublishedProtocol
    outer_fold_id: str = Field(min_length=1)
    frozen_config: FrozenRunConfig
    prediction_artifact_id: str = Field(min_length=1)
    prediction_relative_path: Optional[str] = None

    @field_validator("published_protocol", mode="before")
    @classmethod
    def require_published_protocol(cls, value: Any) -> Any:
        if not isinstance(value, PublishedProtocol):
            raise TypeError("published_protocol must be a PublishedProtocol")
        return value

    @model_validator(mode="after")
    def validate_protocol_key(self) -> "OrchestrationRequest":
        if self.frozen_config.protocol_key != self.published_protocol.manifest.protocol_key:
            raise ValueError("frozen config protocol_key must match published protocol")
        return self


class OrchestrationCallbacks(Protocol):
    """Track adapter callbacks; none receives an outer-label capability."""

    def fit_inner(
        self,
        preprocessing: PreprocessingManager,
        protocol: ProtocolManifest,
        outer_fold: OuterFold,
    ) -> Iterable[FittedArtifact]:
        ...

    def training_backend(
        self,
        outer_fold: OuterFold,
        fitted_artifacts: Tuple[FittedArtifact, ...],
    ) -> InnerTrainingBackend:
        ...

    def selection_evidence(
        self,
        outer_fold: OuterFold,
        training: InnerTrainingArtifact,
    ) -> SelectionEvidence:
        ...

    def fit_outer(
        self,
        preprocessing: PreprocessingManager,
        protocol: ProtocolManifest,
        outer_fold: OuterFold,
        decision: SelectionDecision,
    ) -> Iterable[FittedArtifact]:
        ...

    def refit(
        self,
        request: OuterRefitRequest,
        training: InnerTrainingArtifact,
        fitted_artifacts: Tuple[FittedArtifact, ...],
    ) -> OuterRefitResult:
        ...

    def predict(self, checkpoint: FrozenOuterCheckpoint, sample_id: str) -> Any:
        ...

    def restore_state(self, restored: RestoredRecoveryState) -> None:
        ...


class OrchestrationResult(FrozenSchemaModel):
    """Terminal evidence returned after the COMPLETE snapshot is appended."""

    run_record: RunRecord
    training: InnerTrainingArtifact
    selection_decision: SelectionDecision
    outer_checkpoint: FrozenOuterCheckpoint
    prediction: PredictionArtifact


class OrchestrationAborted(RuntimeError):
    """A controlled stop that produces an ABORTED run snapshot."""


class OrchestrationResumeError(RuntimeError):
    """A recovery artifact does not contain a valid orchestration boundary."""


class _StageEnvelope(FrozenSchemaModel):
    """Canonical stage output persisted inside RecoveryArtifact.stage_state."""

    schema_version: str = "1.0"
    run_id: str
    protocol_hash: str
    fold_id: str
    completed_stage: Optional[OrchestrationStage] = None
    inner_fitted_artifacts: Tuple[FittedArtifact, ...] = ()
    training: Optional[InnerTrainingArtifact] = None
    selection_decision: Optional[SelectionDecision] = None
    outer_fitted_artifacts: Tuple[FittedArtifact, ...] = ()
    outer_checkpoint: Optional[FrozenOuterCheckpoint] = None
    prediction: Optional[PredictionArtifact] = None
    prediction_reference: Optional[ArtifactReference] = None


class TrainingOrchestrator:
    """Connect protocol, preprocessing, training, selection, refit, and evaluation.

    Every major completed stage is stored as a canonical recovery envelope. A
    resumed invocation validates the frozen identity through ``RecoveryManager``,
    restores framework state through the track callback, and skips only stages
    whose immutable outputs are present in the envelope.
    """

    def __init__(
        self,
        *,
        registry: RunRegistry,
        training_engine: TrainingEngine,
        selection_controller: SelectionController,
        preprocessing_manager: PreprocessingManager,
        evaluation_gate: OuterEvaluationGate,
        recovery_manager: RecoveryManager,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        if training_engine.registry is not registry:
            raise ValueError("training engine and orchestrator must share a registry")
        if recovery_manager.registry is not registry:
            raise ValueError("recovery manager and orchestrator must share a registry")
        if training_engine.recovery_manager is not recovery_manager:
            raise ValueError("training engine and orchestrator must share a recovery manager")
        if evaluation_gate.artifact_store is not recovery_manager.artifact_store:
            raise ValueError("evaluation gate and orchestrator must share an artifact store")
        self.registry = registry
        self.training_engine = training_engine
        self.selection_controller = selection_controller
        self.preprocessing_manager = preprocessing_manager
        self.evaluation_gate = evaluation_gate
        self.recovery_manager = recovery_manager
        self.error_journal = error_journal or recovery_manager.error_journal

    def execute(
        self,
        request: OrchestrationRequest,
        callbacks: OrchestrationCallbacks,
    ) -> OrchestrationResult:
        """Execute a fresh planned run and append COMPLETE or interrupted state."""

        request = self._validated_request(request)
        current = self.registry.latest_run(request.run_id)
        if current.status not in {RunStatus.PLANNED, RunStatus.RUNNING}:
            raise ValueError("fresh execution requires a planned or running run")
        state = _StageEnvelope(
            run_id=request.run_id,
            protocol_hash=request.published_protocol.manifest.protocol_hash,
            fold_id=request.outer_fold_id,
        )
        return self._execute_guarded(request, callbacks, state, None)

    def resume(
        self,
        request: OrchestrationRequest,
        callbacks: OrchestrationCallbacks,
        *,
        observed_identity: FrozenRunIdentity,
    ) -> OrchestrationResult:
        """Restore the last complete stage after frozen-identity validation."""

        request = self._validated_request(request)
        restored = self.recovery_manager.resume(request.run_id, observed_identity)
        try:
            decoded = canonical_deserialize(restored.stage_state)
            state = _StageEnvelope.model_validate(decoded)
            if canonical_serialize(state) != restored.stage_state:
                raise ValueError("stage envelope is not canonical")
            if (
                state.run_id != request.run_id
                or state.protocol_hash != request.published_protocol.manifest.protocol_hash
                or state.fold_id != request.outer_fold_id
            ):
                raise ValueError("stage envelope identity differs from request")
        except Exception as exc:
            record = self._error_record(
                request,
                code="ORCHESTRATION_RECOVERY_STATE_INVALID",
                failed_stage="resume_validation",
                last_completed_stage="run_allocated",
                error=exc,
            )
            self.error_journal.append(record)
            raise OrchestrationResumeError(str(exc)) from exc

        restore = getattr(callbacks, "restore_state", None)
        if not callable(restore):
            raise TypeError("resume callbacks must implement restore_state")
        restore(restored)
        failed = self.registry.latest_run(request.run_id)
        running = RunRecord.model_validate(
            {
                **failed.model_dump(mode="python"),
                "status": RunStatus.RUNNING,
                "last_completed_stage": None,
                "completed_utc": None,
            }
        )
        self.registry.append_run(running)
        restored_snapshot = TrainingStateSnapshot(
            model_state=restored.model_state,
            optimizer_state=restored.optimizer_state,
            scheduler_state=restored.scheduler_state,
            stage_state=restored.stage_state,
        )
        return self._execute_guarded(request, callbacks, state, restored_snapshot)

    def _execute_guarded(
        self,
        request: OrchestrationRequest,
        callbacks: OrchestrationCallbacks,
        state: _StageEnvelope,
        restored_snapshot: Optional[TrainingStateSnapshot],
    ) -> OrchestrationResult:
        attempted_stage = "protocol_verification"
        try:
            protocol, outer_fold = self._require_published_protocol(request)

            if not self._completed(state, OrchestrationStage.INNER_PREPROCESSING):
                attempted_stage = OrchestrationStage.INNER_PREPROCESSING.value
                inner_artifacts = tuple(
                    callbacks.fit_inner(self.preprocessing_manager, protocol, outer_fold)
                )
                self._validate_inner_artifacts(protocol, outer_fold, inner_artifacts)
                state = self._advance(
                    state,
                    OrchestrationStage.INNER_PREPROCESSING,
                    inner_fitted_artifacts=inner_artifacts,
                )
                state = self._persist_stage(state, TrainingStateSnapshot(
                    model_state=b"", optimizer_state=b"", scheduler_state=b"", stage_state=b""
                ))

            if not self._completed(state, OrchestrationStage.INNER_TRAINING):
                attempted_stage = OrchestrationStage.INNER_TRAINING.value
                backend = callbacks.training_backend(
                    outer_fold, state.inner_fitted_artifacts
                )
                training = self.training_engine.train_inner(
                    run_id=request.run_id,
                    frozen_config=request.frozen_config,
                    backend=backend,
                )
                snapshot = backend.snapshot(
                    training.measurements.completed_epochs,
                    training.measurements.completed_training_steps,
                )
                if not isinstance(snapshot, TrainingStateSnapshot):
                    snapshot = TrainingStateSnapshot.model_validate(snapshot)
                state = self._advance(
                    state, OrchestrationStage.INNER_TRAINING, training=training
                )
                state = self._persist_stage(state, snapshot)
                if training.status is InnerTrainingStatus.HARD_LIMIT_STOPPED:
                    raise OrchestrationAborted("inner training stopped at a hard limit")
            else:
                if restored_snapshot is None:
                    raise OrchestrationResumeError(
                        "completed training stage requires restored framework state"
                    )
                snapshot = restored_snapshot

            assert state.training is not None
            if not self._completed(state, OrchestrationStage.SELECTION_FROZEN):
                attempted_stage = OrchestrationStage.SELECTION_FROZEN.value
                evidence = callbacks.selection_evidence(outer_fold, state.training)
                if not isinstance(evidence, SelectionEvidence):
                    evidence = SelectionEvidence.model_validate(evidence)
                decision = self.selection_controller.freeze_selection(
                    outer_fold,
                    candidates=evidence.candidates,
                    candidate_scores=evidence.candidate_scores,
                    source_sample_ids=evidence.source_sample_ids,
                    source_prediction_hash=evidence.source_prediction_hash,
                    selection_policy=protocol.selection_policy,
                )
                state = self._advance(
                    state,
                    OrchestrationStage.SELECTION_FROZEN,
                    selection_decision=decision,
                )
                state = self._persist_stage(state, snapshot)

            assert state.selection_decision is not None
            if not self._completed(state, OrchestrationStage.OUTER_PREPROCESSING):
                attempted_stage = OrchestrationStage.OUTER_PREPROCESSING.value
                outer_artifacts = tuple(
                    callbacks.fit_outer(
                        self.preprocessing_manager,
                        protocol,
                        outer_fold,
                        state.selection_decision,
                    )
                )
                self._validate_outer_artifacts(protocol, outer_fold, outer_artifacts)
                state = self._advance(
                    state,
                    OrchestrationStage.OUTER_PREPROCESSING,
                    outer_fitted_artifacts=outer_artifacts,
                )
                state = self._persist_stage(state, snapshot)

            if not self._completed(state, OrchestrationStage.OUTER_REFIT):
                attempted_stage = OrchestrationStage.OUTER_REFIT.value
                outcome = self.selection_controller.apply_refit(
                    outer_fold,
                    state.selection_decision,
                    lambda refit_request: callbacks.refit(
                        refit_request,
                        state.training,
                        state.outer_fitted_artifacts,
                    ),
                )
                if not isinstance(outcome, OuterRefitResult):
                    outcome = OuterRefitResult.model_validate(outcome)
                self._validate_checkpoint(
                    request, outer_fold, state.selection_decision, outcome.checkpoint
                )
                self._register_artifact(
                    outcome.checkpoint.artifact,
                    artifact_kind="outer_checkpoint",
                    run_id=request.run_id,
                )
                snapshot = outcome.state_snapshot
                state = self._advance(
                    state,
                    OrchestrationStage.OUTER_REFIT,
                    outer_checkpoint=outcome.checkpoint,
                )
                state = self._persist_stage(state, snapshot)

            assert state.outer_checkpoint is not None
            if not self._completed(state, OrchestrationStage.OUTER_EVALUATION):
                attempted_stage = OrchestrationStage.OUTER_EVALUATION.value
                lease = self.evaluation_gate.reserve(protocol.protocol_key, 1)
                published = self.evaluation_gate.evaluate(
                    lease,
                    outer_fold=outer_fold,
                    checkpoint=state.outer_checkpoint,
                    selection_decision=state.selection_decision,
                    fitted_artifacts=state.outer_fitted_artifacts,
                    predictor=callbacks.predict,
                    artifact_id=request.prediction_artifact_id,
                    run_id=request.run_id,
                    relative_path=request.prediction_relative_path,
                )
                self._register_prediction(request.run_id, published)
                state = self._advance(
                    state,
                    OrchestrationStage.OUTER_EVALUATION,
                    prediction=published.artifact,
                    prediction_reference=published.reference,
                )
                state = self._persist_stage(state, snapshot)

            return self._complete_run(request, state)
        except BaseException as exc:
            self._record_interruption(request, state, attempted_stage, exc)
            raise

    def _validated_request(self, request: OrchestrationRequest) -> OrchestrationRequest:
        if not isinstance(request, OrchestrationRequest):
            request = OrchestrationRequest.model_validate(request)
        return request

    def _require_published_protocol(
        self, request: OrchestrationRequest
    ) -> tuple[ProtocolManifest, OuterFold]:
        protocol = request.published_protocol.manifest
        reference = request.published_protocol.artifact
        self._require_artifact_bytes(reference, expected_bytes=canonical_serialize(protocol))
        if self.evaluation_gate.protocol != protocol:
            raise ValueError("evaluation gate protocol differs from published protocol")
        try:
            outer_fold = next(
                fold for fold in protocol.folds if fold.fold_id == request.outer_fold_id
            )
        except StopIteration as exc:
            raise ValueError("outer_fold_id is not declared by the protocol") from exc

        run = self.registry.latest_run(request.run_id)
        identity = run.frozen_identity
        expected = {
            "dataset_hash": protocol.dataset_hash,
            "protocol_hash": protocol.protocol_hash,
            "fold_id": outer_fold.fold_id,
            "seed": identity.seed,
            "model_key": request.frozen_config.model_key,
            "config_hash": request.frozen_config.config_hash,
        }
        observed = {
            "dataset_hash": identity.dataset_hash,
            "protocol_hash": identity.protocol_hash,
            "fold_id": identity.fold_id,
            "seed": identity.seed,
            "model_key": identity.model_key,
            "config_hash": identity.config_hash,
        }
        if expected != observed:
            raise ValueError(f"run identity does not match request: expected={expected}, observed={observed}")
        if identity.seed not in protocol.seed_set:
            raise ValueError("run seed is not declared by the protocol")
        return protocol, outer_fold

    def _validate_inner_artifacts(
        self,
        protocol: ProtocolManifest,
        outer_fold: OuterFold,
        artifacts: Tuple[FittedArtifact, ...],
    ) -> None:
        inner_by_id = {fold.fold_id: fold for fold in outer_fold.inner_folds}
        for artifact in artifacts:
            if artifact.dataset_hash != protocol.dataset_hash or artifact.protocol_hash != protocol.protocol_hash:
                raise ValueError("inner fitted artifact identity differs from protocol")
            if artifact.boundary_scope is not FitBoundaryScope.INNER_TRAIN:
                raise ValueError("inner fitted artifact must use inner_train scope")
            inner_fold = inner_by_id.get(artifact.inner_fold_id or "")
            if inner_fold is None:
                raise ValueError("inner fitted artifact references an unknown inner fold")
            self.preprocessing_manager.require_boundary(
                artifact,
                FitBoundary(
                    fold_id=outer_fold.fold_id,
                    scope=FitBoundaryScope.INNER_TRAIN,
                    inner_fold_id=inner_fold.fold_id,
                    allowed_fit_sample_ids=inner_fold.train_ids,
                    outer_test_sample_ids=outer_fold.outer_test_ids,
                ),
            )

    def _validate_outer_artifacts(
        self,
        protocol: ProtocolManifest,
        outer_fold: OuterFold,
        artifacts: Tuple[FittedArtifact, ...],
    ) -> None:
        boundary = FitBoundary(
            fold_id=outer_fold.fold_id,
            scope=FitBoundaryScope.OUTER_TRAIN,
            allowed_fit_sample_ids=outer_fold.outer_train_ids,
            outer_test_sample_ids=outer_fold.outer_test_ids,
        )
        for artifact in artifacts:
            if artifact.dataset_hash != protocol.dataset_hash or artifact.protocol_hash != protocol.protocol_hash:
                raise ValueError("outer fitted artifact identity differs from protocol")
            self.preprocessing_manager.require_boundary(artifact, boundary)

    def _validate_checkpoint(
        self,
        request: OrchestrationRequest,
        outer_fold: OuterFold,
        decision: SelectionDecision,
        checkpoint: FrozenOuterCheckpoint,
    ) -> None:
        identity = self.registry.latest_run(request.run_id).frozen_identity
        if checkpoint.fold_id != outer_fold.fold_id:
            raise ValueError("outer checkpoint fold differs from run fold")
        if checkpoint.seed != identity.seed:
            raise ValueError("outer checkpoint seed differs from run seed")
        if checkpoint.selection_decision_hash != decision.decision_hash:
            raise ValueError("outer checkpoint does not bind the frozen selection decision")
        self._require_artifact_bytes(checkpoint.artifact)

    def _require_artifact_bytes(
        self,
        reference: ArtifactReference,
        *,
        expected_bytes: Optional[bytes] = None,
    ) -> bytes:
        root = self.recovery_manager.artifact_store.root
        target = (root / Path(reference.relative_path)).resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("artifact reference escapes artifact root") from exc
        content = target.read_bytes()
        if hashlib.sha256(content).hexdigest() != reference.content_hash:
            raise ValueError("artifact content hash does not match reference")
        if expected_bytes is not None and content != expected_bytes:
            raise ValueError("published protocol bytes differ from manifest")
        return content

    def _register_artifact(
        self,
        reference: ArtifactReference,
        *,
        artifact_kind: str,
        run_id: str,
    ) -> None:
        self.registry.append_artifact(
            ArtifactRecord(
                artifact=reference,
                artifact_kind=artifact_kind,
                run_id=run_id,
                created_utc=datetime.now(timezone.utc),
            )
        )

    def _register_prediction(
        self, run_id: str, published: PublishedPredictionArtifact
    ) -> None:
        self._register_artifact(
            published.reference,
            artifact_kind="prediction",
            run_id=run_id,
        )

    def _persist_stage(
        self,
        state: _StageEnvelope,
        snapshot: TrainingStateSnapshot,
    ) -> _StageEnvelope:
        try:
            latest = self.registry.latest_recovery(state.run_id)
            boundary = latest.completed_checkpoint_boundary + 1
        except KeyError:
            boundary = 0
        digest = hashlib.sha256(state.run_id.encode("utf-8")).hexdigest()
        self.recovery_manager.save_checkpoint_boundary(
            run_id=state.run_id,
            recovery_id=f"orchestration-{digest}-boundary-{boundary}",
            relative_path=f"orchestration/{digest}/boundary-{boundary}.json",
            model_state=snapshot.model_state,
            optimizer_state=snapshot.optimizer_state,
            scheduler_state=snapshot.scheduler_state,
            stage_state=canonical_serialize(state),
            last_completed_stage=state.completed_stage.value if state.completed_stage else "run_allocated",
            completed_checkpoint_boundary=boundary,
        )
        return state

    @staticmethod
    def _completed(state: _StageEnvelope, stage: OrchestrationStage) -> bool:
        return (
            state.completed_stage is not None
            and _STAGE_ORDER[state.completed_stage] >= _STAGE_ORDER[stage]
        )

    @staticmethod
    def _advance(
        state: _StageEnvelope,
        stage: OrchestrationStage,
        **updates: Any,
    ) -> _StageEnvelope:
        if state.completed_stage is not None and _STAGE_ORDER[stage] <= _STAGE_ORDER[state.completed_stage]:
            raise ValueError("orchestration stages must advance monotonically")
        values = state.model_dump(mode="python")
        values.update(updates)
        values["completed_stage"] = stage
        return _StageEnvelope.model_validate(values)

    def _complete_run(
        self,
        request: OrchestrationRequest,
        state: _StageEnvelope,
    ) -> OrchestrationResult:
        if state.completed_stage is not OrchestrationStage.OUTER_EVALUATION:
            raise ValueError("run cannot complete before outer evaluation")
        assert state.training is not None
        assert state.selection_decision is not None
        assert state.outer_checkpoint is not None
        assert state.prediction is not None
        assert state.prediction_reference is not None
        current = self.registry.latest_run(request.run_id)
        complete = RunRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "checkpoint_ref": state.outer_checkpoint.artifact,
                "prediction_artifact_ref": state.prediction_reference,
                "runtime_measurements": state.training.measurements,
                "status": RunStatus.COMPLETE,
                "completed_utc": datetime.now(timezone.utc),
                "last_completed_stage": None,
            }
        )
        self.registry.append_run(complete)
        return OrchestrationResult(
            run_record=complete,
            training=state.training,
            selection_decision=state.selection_decision,
            outer_checkpoint=state.outer_checkpoint,
            prediction=state.prediction,
        )

    def _record_interruption(
        self,
        request: OrchestrationRequest,
        state: _StageEnvelope,
        attempted_stage: str,
        error: BaseException,
    ) -> None:
        try:
            current = self.registry.latest_run(request.run_id)
        except Exception as registry_error:
            error.add_note(f"could not load run for interruption record: {registry_error!r}")
            return
        if current.status in {RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.ABORTED}:
            return
        last_completed = (
            state.completed_stage.value if state.completed_stage else "run_allocated"
        )
        record = self._error_record(
            request,
            code="ORCHESTRATION_STAGE_FAILED",
            failed_stage=attempted_stage,
            last_completed_stage=last_completed,
            error=error,
        )
        try:
            self.error_journal.append(record)
        except Exception as journal_error:
            error.add_note(f"error journal append failed: {journal_error!r}")
        status = (
            RunStatus.ABORTED
            if isinstance(error, (KeyboardInterrupt, SystemExit, OrchestrationAborted))
            else RunStatus.FAILED
        )
        try:
            self.recovery_manager.record_interruption(
                request.run_id,
                status=status,
                last_completed_stage=last_completed,
            )
        except Exception as registry_error:
            error.add_note(f"interruption status append failed: {registry_error!r}")

    @staticmethod
    def _error_record(
        request: OrchestrationRequest,
        *,
        code: str,
        failed_stage: str,
        last_completed_stage: str,
        error: BaseException,
    ) -> ErrorRecord:
        return ErrorRecord(
            error_code=code,
            failed_stage=failed_stage,
            affected_identifiers={
                "run_id": request.run_id,
                "protocol_key": request.published_protocol.manifest.protocol_key,
                "fold_id": request.outer_fold_id,
            },
            failing_field="stage",
            expected_value={"outcome": "completed", "last_completed_stage": last_completed_stage},
            observed_value={"exception_type": type(error).__name__, "message": str(error)},
            invariant_checks={
                "completed_artifacts_unchanged": True,
                "completed_run_records_unchanged": True,
                "outer_labels_capability_remained_in_evaluator": True,
            },
        )


__all__ = [
    "OrchestrationAborted",
    "OrchestrationCallbacks",
    "OrchestrationRequest",
    "OrchestrationResult",
    "OrchestrationResumeError",
    "OrchestrationStage",
    "OuterRefitResult",
    "SelectionEvidence",
    "TrainingOrchestrator",
]
