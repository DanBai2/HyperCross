"""Synthetic tests for the unified recoverable training orchestrator."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.engine import (
    EpochTrainingResult,
    OrchestrationRequest,
    OuterEvaluationGate,
    OuterRefitResult,
    SelectionController,
    SelectionEvidence,
    TrainingEngine,
    TrainingOrchestrator,
    TrainingStateSnapshot,
)
from benchmark.metrics import PredictionArtifactWriter
from benchmark.protocols import PublishedProtocol
from benchmark.recovery import FailureAtomicArtifactStore, RecoveryManager
from benchmark.registry import (
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    Provenance,
    RunRegistry,
)
from benchmark.schema import (
    CheckpointRetentionPolicy,
    ClaimStatus,
    EvaluationBudget,
    ExecutionMode,
    FitBoundary,
    FitBoundaryScope,
    FrozenOuterCheckpoint,
    FrozenRunConfig,
    ImplementationKind,
    InnerFold,
    LimitEnforcement,
    MetricDirection,
    NumericPrecision,
    OuterFold,
    ProtocolManifest,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    SelectionCandidate,
    SelectionPolicy,
    canonical_serialize,
)
from benchmark.transforms import PreprocessingManager


NOW = datetime(2027, 3, 4, 5, 6, tzinfo=timezone.utc)
SOURCE_HASH = "4" * 64
ENVIRONMENT_HASH = "5" * 64


def _fold() -> OuterFold:
    return OuterFold(
        fold_id="outer-synthetic",
        outer_train_ids=("train-a", "train-b", "validation-a"),
        outer_test_ids=("test-a", "test-b"),
        inner_folds=(
            InnerFold(
                fold_id="inner-synthetic",
                train_ids=("train-a", "train-b"),
                validation_ids=("validation-a",),
            ),
        ),
    )


def _protocol() -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1",
        protocol_key="synthetic.nested.v1",
        dataset_hash="1" * 64,
        folds=(_fold(),),
        selection_policy=SelectionPolicy(
            policy_key="synthetic-selection-v1",
            objective_metric="score",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule="candidate_id_utf8_ascending",
        ),
        seed_set=(7,),
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=1),
    )


def _config() -> FrozenRunConfig:
    return FrozenRunConfig.create(
        model_key="toy.orchestration.v1",
        protocol_key="synthetic.nested.v1",
        execution_mode=ExecutionMode.DETERMINISTIC,
        numeric_precision=NumericPrecision.FLOAT32,
        input_volume_bytes=20,
        resource_budget=ResourceBudget(
            limits=(
                ResourceLimit(
                    field="completed_training_steps",
                    value=10,
                    unit="count",
                    enforcement=LimitEnforcement.HARD,
                ),
            )
        ),
        checkpoint_retention=CheckpointRetentionPolicy(
            preserve_selected=True,
            preserve_last=True,
            preserve_recovery_state=True,
        ),
        parameters={
            "epochs": 1,
            "selection_metric_direction": "maximize",
        },
    )


class _Backend:
    device_identity = "cpu:synthetic"
    parameter_count = 2

    def prepare(self, config: FrozenRunConfig, seed: int) -> None:
        assert seed == 7

    def train_epoch(self, epoch: int) -> EpochTrainingResult:
        return EpochTrainingResult(training_steps=1, selection_score=0.75)

    def snapshot(self, epoch: int, completed_training_steps: int) -> TrainingStateSnapshot:
        suffix = f"{epoch}:{completed_training_steps}".encode("ascii")
        return TrainingStateSnapshot(
            model_state=b"model:" + suffix,
            optimizer_state=b"optimizer:" + suffix,
            scheduler_state=b"scheduler:" + suffix,
            stage_state=b"backend:" + suffix,
        )

    def peak_device_memory_bytes(self) -> int:
        return 0


class _Vault:
    def __init__(self) -> None:
        self.calls = 0

    def unlock(self, lease, sample_ids):
        self.calls += 1
        return {"test-a": 1, "test-b": 0}


class _Callbacks:
    def __init__(self, store: FailureAtomicArtifactStore, *, fail_refit_once: bool = False) -> None:
        self.store = store
        self.fail_refit_once = fail_refit_once
        self.calls: list[str] = []
        self.restore_calls = 0

    def fit_inner(self, preprocessing, protocol, outer_fold):
        self.calls.append("fit_inner")
        inner = outer_fold.inner_folds[0]
        boundary = FitBoundary(
            fold_id=outer_fold.fold_id,
            scope=FitBoundaryScope.INNER_TRAIN,
            inner_fold_id=inner.fold_id,
            allowed_fit_sample_ids=inner.train_ids,
            outer_test_sample_ids=outer_fold.outer_test_ids,
        )
        return (
            preprocessing.fit_band_normalization(
                [[1.0], [3.0]],
                inner.train_ids,
                boundary,
                dataset_hash=protocol.dataset_hash,
                protocol_hash=protocol.protocol_hash,
            ),
        )

    def training_backend(self, outer_fold, fitted_artifacts):
        self.calls.append("training_backend")
        assert fitted_artifacts[0].fit_sample_ids == ("train-a", "train-b")
        return _Backend()

    def selection_evidence(self, outer_fold, training):
        self.calls.append("selection_evidence")
        return SelectionEvidence(
            candidates=(
                SelectionCandidate(candidate_id="candidate-a", configuration={"epoch": 1}),
            ),
            candidate_scores={"candidate-a": 0.75},
            source_sample_ids=("validation-a",),
            source_prediction_hash="2" * 64,
        )

    def fit_outer(self, preprocessing, protocol, outer_fold, decision):
        self.calls.append("fit_outer")
        boundary = FitBoundary(
            fold_id=outer_fold.fold_id,
            scope=FitBoundaryScope.OUTER_TRAIN,
            allowed_fit_sample_ids=outer_fold.outer_train_ids,
            outer_test_sample_ids=outer_fold.outer_test_ids,
        )
        return (
            preprocessing.fit_band_normalization(
                [[1.0], [3.0], [2.0]],
                outer_fold.outer_train_ids,
                boundary,
                dataset_hash=protocol.dataset_hash,
                protocol_hash=protocol.protocol_hash,
            ),
        )

    def refit(self, request, training, fitted_artifacts):
        self.calls.append("refit")
        if self.fail_refit_once:
            self.fail_refit_once = False
            raise RuntimeError("synthetic refit interruption")
        content = b"synthetic frozen outer checkpoint"
        reference = self.store.publish_bytes(
            artifact_id="outer-checkpoint-synthetic",
            relative_path="outer/checkpoint.bin",
            content=content,
            expected_hash=hashlib.sha256(content).hexdigest(),
            media_type="application/octet-stream",
        )
        return OuterRefitResult(
            checkpoint=FrozenOuterCheckpoint(
                artifact=reference,
                fold_id=request.fold_id,
                seed=7,
                inference_branch="primary",
                selection_decision_hash=request.selection_decision.decision_hash,
                frozen_utc=NOW,
            ),
            state_snapshot=TrainingStateSnapshot(
                model_state=b"refit-model",
                optimizer_state=b"refit-optimizer",
                scheduler_state=b"refit-scheduler",
                stage_state=b"refit-stage",
            ),
        )

    def predict(self, checkpoint, sample_id: str) -> dict[str, Any]:
        self.calls.append(f"predict:{sample_id}")
        return {
            "y_pred": int(sample_id == "test-a"),
            "probability": 0.8 if sample_id == "test-a" else 0.2,
        }

    def restore_state(self, restored) -> None:
        self.restore_calls += 1
        assert restored.model_state.startswith(b"model:")


def _system(tmp_path: Path, *, fail_refit_once: bool = False):
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    recovery = RecoveryManager(registry, store)
    training = TrainingEngine(registry, store, recovery)
    protocol = _protocol()
    protocol_bytes = canonical_serialize(protocol)
    protocol_ref = store.publish_bytes(
        artifact_id=f"protocol-{protocol.protocol_hash}",
        relative_path=f"protocols/{protocol.protocol_hash}.json",
        content=protocol_bytes,
        expected_hash=hashlib.sha256(protocol_bytes).hexdigest(),
        media_type="application/json",
    )
    published = PublishedProtocol(protocol, protocol_ref)
    config = _config()
    identity = FrozenRunIdentity(
        dataset_hash=protocol.dataset_hash,
        protocol_hash=protocol.protocol_hash,
        model_key=config.model_key,
        config_hash=config.config_hash,
        fold_id=_fold().fold_id,
        seed=7,
        environment_lock_hash=ENVIRONMENT_HASH,
        environment_compatibility_policy="exact",
    )
    planned = registry.allocate_run(
        frozen_identity=identity,
        model_identity=ModelIdentity(
            model_key=config.model_key,
            display_name="Synthetic orchestration model",
            class_fully_qualified_name="synthetic.Model",
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=2,
            source_revision="test-revision",
        ),
        input_contract=InputContract(
            representation_key="synthetic.features.v1",
            modalities=("features",),
            tensor_shapes={"features": (1,)},
            dtypes={"features": "float32"},
            channel_or_band_order={"features": ("value",)},
            preprocessing_key="band-normalization.v1",
        ),
        provenance=Provenance(
            code_revision="test-revision",
            source_hash=SOURCE_HASH,
            environment_hash=ENVIRONMENT_HASH,
            config_hash=config.config_hash,
            dataset_hash=protocol.dataset_hash,
        ),
        resource_budget=config.resource_budget,
        claim_decision=ClaimDecision(status=ClaimStatus.STRICT_CANDIDATE),
    )
    vault = _Vault()
    gate = OuterEvaluationGate(protocol, vault, PredictionArtifactWriter(store))
    orchestrator = TrainingOrchestrator(
        registry=registry,
        training_engine=training,
        selection_controller=SelectionController(error_journal=store.error_journal),
        preprocessing_manager=PreprocessingManager(error_journal=store.error_journal),
        evaluation_gate=gate,
        recovery_manager=recovery,
    )
    request = OrchestrationRequest(
        run_id=planned.run_id,
        published_protocol=published,
        outer_fold_id=_fold().fold_id,
        frozen_config=config,
        prediction_artifact_id="prediction-synthetic",
        prediction_relative_path="predictions/synthetic.json",
    )
    return orchestrator, request, _Callbacks(store, fail_refit_once=fail_refit_once), registry, vault


def test_synthetic_orchestration_connects_all_stages_and_completes(tmp_path: Path) -> None:
    """Validates Requirements 3.1, 4.8-4.13, 5.13-5.24, 13.1-13.19."""

    orchestrator, request, callbacks, registry, vault = _system(tmp_path)

    result = orchestrator.execute(request, callbacks)

    assert result.run_record.status is RunStatus.COMPLETE
    assert result.run_record.frozen_identity.protocol_hash == _protocol().protocol_hash
    assert result.selection_decision.source_sample_ids == ("validation-a",)
    assert result.outer_checkpoint.selection_decision_hash == result.selection_decision.decision_hash
    assert result.prediction.expected_sample_ids == ("test-a", "test-b")
    assert vault.calls == 1
    assert callbacks.calls == [
        "fit_inner",
        "training_backend",
        "selection_evidence",
        "fit_outer",
        "refit",
        "predict:test-a",
        "predict:test-b",
    ]
    assert len(registry.recovery_artifacts(request.run_id)) >= 6
    assert registry.get_artifact(result.run_record.checkpoint_ref.artifact_id).run_id == request.run_id
    assert registry.get_artifact(result.run_record.prediction_artifact_ref.artifact_id).run_id == request.run_id


def test_refit_failure_is_append_only_and_resume_skips_completed_stages(tmp_path: Path) -> None:
    """A failed stage records FAILED plus Error_Record, then resumes its boundary."""

    orchestrator, request, callbacks, registry, vault = _system(
        tmp_path, fail_refit_once=True
    )

    try:
        orchestrator.execute(request, callbacks)
    except RuntimeError as exc:
        assert str(exc) == "synthetic refit interruption"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("synthetic interruption was not raised")

    failed = registry.latest_run(request.run_id)
    assert failed.status is RunStatus.FAILED
    assert failed.last_completed_stage == "outer_preprocessing"
    errors = orchestrator.error_journal.path.read_text(encoding="utf-8")
    assert "ORCHESTRATION_STAGE_FAILED" in errors
    assert "outer_refit" in errors
    completed_calls = tuple(callbacks.calls)
    assert vault.calls == 0

    result = orchestrator.resume(
        request,
        callbacks,
        observed_identity=failed.frozen_identity,
    )

    assert result.run_record.status is RunStatus.COMPLETE
    assert callbacks.restore_calls == 1
    assert tuple(callbacks.calls[: len(completed_calls)]) == completed_calls
    assert callbacks.calls.count("fit_inner") == 1
    assert callbacks.calls.count("training_backend") == 1
    assert callbacks.calls.count("selection_evidence") == 1
    assert callbacks.calls.count("fit_outer") == 1
    assert callbacks.calls.count("refit") == 2
    assert vault.calls == 1
    assert [record.status for record in registry.run_history(request.run_id)] == [
        RunStatus.PLANNED,
        RunStatus.RUNNING,
        RunStatus.FAILED,
        RunStatus.RUNNING,
        RunStatus.COMPLETE,
    ]
