"""Frozen-config inner training and checkpoint lifecycle management."""

from __future__ import annotations

import base64
import hashlib
import importlib
import math
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator
from typing_extensions import Annotated

from benchmark.recovery import FailureAtomicArtifactStore, RecoveryManager
from benchmark.registry import ArtifactRecord, RunRecord, RunRegistry
from benchmark.schema import (
    ArtifactReference,
    BudgetAssessment,
    ExecutionMode,
    FrameworkDeterminismSettings,
    FrozenRunConfig,
    MetricDirection,
    RunStatus,
    RuntimeMeasurements,
    assess_resource_budget,
    canonical_deserialize,
    canonical_serialize,
)
from benchmark.schema.environment import FrozenSchemaModel


NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]
PositiveStrictInt = Annotated[StrictInt, Field(gt=0)]


class TrainingCheckpointKind(str, Enum):
    """Durable model checkpoint roles retained by inner training."""

    SELECTED = "selected"
    LAST = "last"


class InnerTrainingStatus(str, Enum):
    """Terminal status of one bounded inner-training invocation."""

    COMPLETED = "completed"
    HARD_LIMIT_STOPPED = "hard_limit_stopped"


class EpochTrainingResult(FrozenSchemaModel):
    """Outcome returned by a backend after one complete epoch boundary."""

    training_steps: NonNegativeStrictInt
    selection_score: StrictFloat

    @field_validator("selection_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("selection_score must be finite")
        return value


class TrainingStateSnapshot(FrozenSchemaModel):
    """Opaque serializations required for checkpoints and exact recovery."""

    model_state: bytes
    optimizer_state: bytes
    scheduler_state: bytes
    stage_state: bytes


class RetainedTrainingCheckpoint(FrozenSchemaModel):
    """One atomically published selected or last model checkpoint."""

    kind: TrainingCheckpointKind
    epoch: PositiveStrictInt
    completed_training_steps: NonNegativeStrictInt
    selection_score: StrictFloat
    artifact: ArtifactReference


class InnerTrainingArtifact(FrozenSchemaModel):
    """Complete, immutable evidence emitted by bounded inner training."""

    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: StrictInt
    execution_mode: ExecutionMode
    framework_settings: FrameworkDeterminismSettings
    status: InnerTrainingStatus
    selected_epoch: PositiveStrictInt
    measurements: RuntimeMeasurements
    budget_assessment: BudgetAssessment
    retained_checkpoints: Tuple[RetainedTrainingCheckpoint, ...] = ()
    recovery_artifacts: Tuple[ArtifactReference, ...] = ()

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "InnerTrainingArtifact":
        kinds = [checkpoint.kind for checkpoint in self.retained_checkpoints]
        if len(kinds) != len(set(kinds)):
            raise ValueError("retained checkpoint kinds must be unique")
        if self.selected_epoch > self.measurements.completed_epochs:
            raise ValueError("selected_epoch cannot exceed completed epochs")
        stopped = self.status is InnerTrainingStatus.HARD_LIMIT_STOPPED
        if stopped != self.budget_assessment.stop_at_next_checkpoint_boundary:
            raise ValueError("training status must match the hard-limit assessment")
        return self


@runtime_checkable
class InnerTrainingBackend(Protocol):
    """Framework adapter used by :class:`TrainingEngine`.

    ``prepare`` is called only after the engine applies the frozen seed and
    framework settings. Backends should create or reset seed-sensitive model
    state there when model initialization must be covered by the run seed.
    """

    @property
    def device_identity(self) -> str:
        ...

    @property
    def parameter_count(self) -> int:
        ...

    def prepare(self, config: FrozenRunConfig, seed: int) -> None:
        ...

    def train_epoch(self, epoch: int) -> EpochTrainingResult:
        ...

    def snapshot(self, epoch: int, completed_training_steps: int) -> TrainingStateSnapshot:
        ...

    def peak_device_memory_bytes(self) -> int:
        ...


class _CheckpointBundle(FrozenSchemaModel):
    """Canonical checkpoint payload containing one opaque model state."""

    schema_version: str = "1.0"
    run_id: str
    config_hash: str
    seed: StrictInt
    kind: TrainingCheckpointKind
    epoch: PositiveStrictInt
    completed_training_steps: NonNegativeStrictInt
    selection_score: StrictFloat
    model_state_hash: str
    model_state_base64: str

    @model_validator(mode="after")
    def validate_model_state(self) -> "_CheckpointBundle":
        payload = base64.b64decode(self.model_state_base64.encode("ascii"), validate=True)
        if hashlib.sha256(payload).hexdigest() != self.model_state_hash:
            raise ValueError("model state hash does not match checkpoint payload")
        return self


class TrainingEngine:
    """Run inner epochs from a frozen identity and retain bounded checkpoints."""

    def __init__(
        self,
        registry: RunRegistry,
        artifact_store: FailureAtomicArtifactStore,
        recovery_manager: RecoveryManager,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if recovery_manager.registry is not registry:
            raise ValueError("recovery manager and training engine must share a registry")
        if recovery_manager.artifact_store is not artifact_store:
            raise ValueError("recovery manager and training engine must share an artifact store")
        self.registry = registry
        self.artifact_store = artifact_store
        self.recovery_manager = recovery_manager
        self.monotonic_ns = monotonic_ns

    def train_inner(
        self,
        *,
        run_id: str,
        frozen_config: FrozenRunConfig,
        backend: InnerTrainingBackend,
    ) -> InnerTrainingArtifact:
        """Execute bounded epochs and stop only at a complete checkpoint boundary."""

        if not isinstance(frozen_config, FrozenRunConfig):
            raise TypeError("frozen_config must be a FrozenRunConfig")
        if not isinstance(backend, InnerTrainingBackend):
            raise TypeError("backend does not implement InnerTrainingBackend")

        current = self.registry.latest_run(run_id)
        self._validate_frozen_identity(current, frozen_config)
        maximum_epochs, direction = self._training_parameters(frozen_config)
        self._validate_backend_metadata(current, frozen_config, backend)

        if current.status is RunStatus.PLANNED:
            running = RunRecord.model_validate(
                {**current.model_dump(mode="python"), "status": RunStatus.RUNNING}
            )
            self.registry.append_run(running)
        elif current.status is not RunStatus.RUNNING:
            raise ValueError("inner training requires a planned or running run")

        seed = current.frozen_identity.seed
        self._apply_frozen_execution(seed, frozen_config)
        backend.prepare(frozen_config, seed)

        start_ns = self.monotonic_ns()
        completed_steps = 0
        best_epoch = 0
        best_completed_steps = 0
        best_score: Optional[float] = None
        best_state: Optional[TrainingStateSnapshot] = None
        last_state: Optional[TrainingStateSnapshot] = None
        last_score = 0.0
        recovery_refs: list[ArtifactReference] = []
        assessment: Optional[BudgetAssessment] = None
        measurements: Optional[RuntimeMeasurements] = None

        for epoch in range(1, maximum_epochs + 1):
            outcome = backend.train_epoch(epoch)
            if not isinstance(outcome, EpochTrainingResult):
                outcome = EpochTrainingResult.model_validate(outcome)
            completed_steps += outcome.training_steps
            snapshot = backend.snapshot(epoch, completed_steps)
            if not isinstance(snapshot, TrainingStateSnapshot):
                snapshot = TrainingStateSnapshot.model_validate(snapshot)
            last_state = snapshot
            last_score = outcome.selection_score

            if best_score is None or self._is_better(
                outcome.selection_score, best_score, direction
            ):
                best_epoch = epoch
                best_completed_steps = completed_steps
                best_score = outcome.selection_score
                best_state = snapshot

            measurements = self._measurements(
                frozen_config=frozen_config,
                backend=backend,
                completed_epochs=epoch,
                completed_training_steps=completed_steps,
                start_ns=start_ns,
            )
            assessment = assess_resource_budget(
                frozen_config.resource_budget,
                measurements,
                checkpoint_boundary=epoch,
            )

            if frozen_config.checkpoint_retention.preserve_recovery_state:
                recovery = self.recovery_manager.save_checkpoint_boundary(
                    run_id=run_id,
                    recovery_id=self._artifact_id(run_id, "inner-recovery", epoch),
                    relative_path=self._relative_path(
                        run_id, "recovery", f"epoch-{epoch}.json"
                    ),
                    model_state=snapshot.model_state,
                    optimizer_state=snapshot.optimizer_state,
                    scheduler_state=snapshot.scheduler_state,
                    stage_state=snapshot.stage_state,
                    last_completed_stage="inner_training",
                    completed_checkpoint_boundary=epoch,
                )
                recovery_refs.append(recovery.artifact)

            if assessment.stop_at_next_checkpoint_boundary:
                break

        assert measurements is not None
        assert assessment is not None
        assert best_state is not None and best_score is not None and best_epoch > 0
        assert last_state is not None

        retained: list[RetainedTrainingCheckpoint] = []
        policy = frozen_config.checkpoint_retention
        if policy.preserve_selected:
            retained.append(
                self._publish_checkpoint(
                    run_id=run_id,
                    config=frozen_config,
                    seed=seed,
                    kind=TrainingCheckpointKind.SELECTED,
                    epoch=best_epoch,
                    completed_training_steps=best_completed_steps,
                    selection_score=best_score,
                    state=best_state,
                )
            )
        if policy.preserve_last:
            retained.append(
                self._publish_checkpoint(
                    run_id=run_id,
                    config=frozen_config,
                    seed=seed,
                    kind=TrainingCheckpointKind.LAST,
                    epoch=measurements.completed_epochs,
                    completed_training_steps=completed_steps,
                    selection_score=last_score,
                    state=last_state,
                )
            )

        return InnerTrainingArtifact(
            run_id=run_id,
            config_hash=frozen_config.config_hash,
            seed=seed,
            execution_mode=frozen_config.execution_mode,
            framework_settings=frozen_config.framework_settings,
            status=(
                InnerTrainingStatus.HARD_LIMIT_STOPPED
                if assessment.stop_at_next_checkpoint_boundary
                else InnerTrainingStatus.COMPLETED
            ),
            selected_epoch=best_epoch,
            measurements=measurements,
            budget_assessment=assessment,
            retained_checkpoints=tuple(retained),
            recovery_artifacts=tuple(recovery_refs),
        )

    @staticmethod
    def _training_parameters(
        config: FrozenRunConfig,
    ) -> tuple[int, MetricDirection]:
        parameters = canonical_deserialize(
            config.parameters_canonical_serialization.encode("utf-8")
        )
        if not isinstance(parameters, dict):
            raise ValueError("frozen training parameters must be an object")
        epochs = parameters.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("frozen parameter 'epochs' must be a positive integer")
        try:
            direction = MetricDirection(parameters["selection_metric_direction"])
        except KeyError as exc:
            raise ValueError(
                "frozen parameter 'selection_metric_direction' is required"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                "frozen parameter 'selection_metric_direction' is invalid"
            ) from exc
        return epochs, direction

    @staticmethod
    def _validate_frozen_identity(
        run: RunRecord,
        config: FrozenRunConfig,
    ) -> None:
        identity = run.frozen_identity
        mismatches = []
        if identity.config_hash != config.config_hash:
            mismatches.append("config_hash")
        if identity.model_key != config.model_key:
            mismatches.append("model_key")
        if run.resource_budget != config.resource_budget:
            mismatches.append("resource_budget")
        if mismatches:
            raise ValueError(
                "frozen config differs from allocated run: " + ", ".join(mismatches)
            )

    @staticmethod
    def _validate_backend_metadata(
        run: RunRecord,
        config: FrozenRunConfig,
        backend: InnerTrainingBackend,
    ) -> None:
        if isinstance(backend.parameter_count, bool) or not isinstance(
            backend.parameter_count, int
        ):
            raise TypeError("backend parameter_count must be an integer")
        if backend.parameter_count < 0:
            raise ValueError("backend parameter_count must be non-negative")
        if backend.parameter_count != run.model_identity.parameter_count:
            raise ValueError("backend parameter_count differs from frozen model identity")
        if not isinstance(backend.device_identity, str) or not backend.device_identity.strip():
            raise ValueError("backend device_identity must be non-empty")

    @staticmethod
    def _apply_frozen_execution(seed: int, config: FrozenRunConfig) -> None:
        random.seed(seed)
        try:
            numpy = importlib.import_module("numpy")
        except ImportError:
            numpy = None
        if numpy is not None:
            numpy.random.seed(seed % (2**32))

        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        settings = config.framework_settings
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(settings.deterministic_algorithms)
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.deterministic = settings.cudnn_deterministic
            cudnn.benchmark = settings.cudnn_benchmark
        cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        if cuda_matmul is not None:
            cuda_matmul.allow_tf32 = settings.allow_tf32

    def _measurements(
        self,
        *,
        frozen_config: FrozenRunConfig,
        backend: InnerTrainingBackend,
        completed_epochs: int,
        completed_training_steps: int,
        start_ns: int,
    ) -> RuntimeMeasurements:
        elapsed_ns = max(0, self.monotonic_ns() - start_ns)
        peak_memory = backend.peak_device_memory_bytes()
        if isinstance(peak_memory, bool) or not isinstance(peak_memory, int):
            raise TypeError("backend peak device memory must be an integer")
        return RuntimeMeasurements(
            completed_epochs=completed_epochs,
            completed_training_steps=completed_training_steps,
            wall_clock_duration_ms=elapsed_ns // 1_000_000,
            peak_device_memory_bytes=peak_memory,
            device_identity=backend.device_identity,
            numeric_precision=frozen_config.numeric_precision.value,
            parameter_count=backend.parameter_count,
            input_volume_bytes=frozen_config.input_volume_bytes,
        )

    @staticmethod
    def _is_better(
        candidate: float,
        incumbent: float,
        direction: MetricDirection,
    ) -> bool:
        if direction is MetricDirection.MAXIMIZE:
            return candidate > incumbent
        return candidate < incumbent

    def _publish_checkpoint(
        self,
        *,
        run_id: str,
        config: FrozenRunConfig,
        seed: int,
        kind: TrainingCheckpointKind,
        epoch: int,
        completed_training_steps: int,
        selection_score: float,
        state: TrainingStateSnapshot,
    ) -> RetainedTrainingCheckpoint:
        bundle = _CheckpointBundle(
            run_id=run_id,
            config_hash=config.config_hash,
            seed=seed,
            kind=kind,
            epoch=epoch,
            completed_training_steps=completed_training_steps,
            selection_score=selection_score,
            model_state_hash=hashlib.sha256(state.model_state).hexdigest(),
            model_state_base64=base64.b64encode(state.model_state).decode("ascii"),
        )
        content = canonical_serialize(bundle)
        artifact = self.artifact_store.publish_bytes(
            artifact_id=self._artifact_id(run_id, f"inner-{kind.value}", epoch),
            relative_path=self._relative_path(
                run_id, "checkpoints", f"{kind.value}-epoch-{epoch}.json"
            ),
            content=content,
            expected_hash=hashlib.sha256(content).hexdigest(),
            media_type="application/vnd.hypercross.training-checkpoint+json",
        )
        self.registry.append_artifact(
            ArtifactRecord(
                artifact=artifact,
                artifact_kind=f"inner_{kind.value}_checkpoint",
                run_id=run_id,
                created_utc=self._utc_now(),
            )
        )
        return RetainedTrainingCheckpoint(
            kind=kind,
            epoch=epoch,
            completed_training_steps=completed_training_steps,
            selection_score=selection_score,
            artifact=artifact,
        )

    @staticmethod
    def _artifact_id(run_id: str, role: str, epoch: int) -> str:
        run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return f"{role}-{run_digest}-epoch-{epoch}"

    @staticmethod
    def _relative_path(run_id: str, category: str, filename: str) -> str:
        run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return f"training/{run_digest}/{category}/{filename}"

    @staticmethod
    def _utc_now() -> Any:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)


__all__ = [
    "EpochTrainingResult",
    "InnerTrainingArtifact",
    "InnerTrainingBackend",
    "InnerTrainingStatus",
    "RetainedTrainingCheckpoint",
    "TrainingCheckpointKind",
    "TrainingEngine",
    "TrainingStateSnapshot",
]
