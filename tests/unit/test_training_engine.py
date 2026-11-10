"""Small bounded tests for frozen inner training and checkpoint retention."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from benchmark.engine import (
    EpochTrainingResult,
    InnerTrainingStatus,
    TrainingCheckpointKind,
    TrainingEngine,
    TrainingStateSnapshot,
)
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
    ExecutionMode,
    FrozenRunConfig,
    ImplementationKind,
    LimitEnforcement,
    NumericPrecision,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
)


HASHES = {name: f"{index:064x}" for index, name in enumerate(
    ("dataset", "protocol", "environment", "source"), start=1
)}


def _budget(
    *,
    step_limit: int = 100,
    memory_limit: int | None = None,
) -> ResourceBudget:
    limits = [
        ResourceLimit(
            field="completed_training_steps",
            value=step_limit,
            unit="count",
            enforcement=LimitEnforcement.HARD,
        )
    ]
    if memory_limit is not None:
        limits.append(
            ResourceLimit(
                field="peak_device_memory_bytes",
                value=memory_limit,
                unit="bytes",
                enforcement=LimitEnforcement.ADVISORY,
            )
        )
    return ResourceBudget(limits=tuple(limits))


def _config(
    budget: ResourceBudget,
    *,
    epochs: int = 2,
    retention: CheckpointRetentionPolicy | None = None,
) -> FrozenRunConfig:
    return FrozenRunConfig.create(
        model_key="toy.linear.v1",
        protocol_key="toy.inner.v1",
        execution_mode=ExecutionMode.DETERMINISTIC,
        numeric_precision=NumericPrecision.FLOAT32,
        input_volume_bytes=24,
        resource_budget=budget,
        checkpoint_retention=retention,
        parameters={
            "epochs": epochs,
            "learning_rate": 0.05,
            "selection_metric_direction": "maximize",
        },
    )


def _allocate(registry: RunRegistry, config: FrozenRunConfig, *, seed: int = 17) -> str:
    identity = FrozenRunIdentity(
        dataset_hash=HASHES["dataset"],
        protocol_hash=HASHES["protocol"],
        model_key=config.model_key,
        config_hash=config.config_hash,
        fold_id="inner-fold-1",
        seed=seed,
        environment_lock_hash=HASHES["environment"],
        environment_compatibility_policy="exact",
    )
    planned = registry.allocate_run(
        frozen_identity=identity,
        model_identity=ModelIdentity(
            model_key=config.model_key,
            display_name="Toy linear",
            class_fully_qualified_name="torch.nn.modules.linear.Linear",
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=2,
            source_revision="test-revision",
        ),
        input_contract=InputContract(
            representation_key="toy.scalar.v1",
            modalities=("x",),
            tensor_shapes={"x": (1,)},
            dtypes={"x": "float32"},
            channel_or_band_order={"x": ("value",)},
            preprocessing_key="none.v1",
        ),
        provenance=Provenance(
            code_revision="test-revision",
            source_hash=HASHES["source"],
            environment_hash=HASHES["environment"],
            config_hash=config.config_hash,
            dataset_hash=HASHES["dataset"],
        ),
        resource_budget=config.resource_budget,
        claim_decision=ClaimDecision(status=ClaimStatus.STRICT_CANDIDATE),
    )
    return planned.run_id


def _engine(tmp_path: Path) -> tuple[TrainingEngine, RunRegistry, FailureAtomicArtifactStore]:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    recovery = RecoveryManager(registry, store)
    return TrainingEngine(registry, store, recovery), registry, store


class _TorchToyBackend:
    """Real two-batch CPU training backend used for the bounded TCFM test."""

    def __init__(self) -> None:
        torch = pytest.importorskip("torch")
        self.torch = torch
        self.model = torch.nn.Linear(1, 1)
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.y = 2.0 * self.x + 1.0

    @property
    def device_identity(self) -> str:
        return "cpu"

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def prepare(self, config: FrozenRunConfig, seed: int) -> None:
        assert config.numeric_precision is NumericPrecision.FLOAT32
        assert seed == 17
        self.model.reset_parameters()
        self.optimizer = self.torch.optim.SGD(self.model.parameters(), lr=0.05)
        self.scheduler = self.torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9
        )

    def train_epoch(self, epoch: int) -> EpochTrainingResult:
        for start in (0, 2):
            prediction = self.model(self.x[start : start + 2])
            loss = self.torch.nn.functional.mse_loss(
                prediction, self.y[start : start + 2]
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        with self.torch.no_grad():
            score = -float(
                self.torch.nn.functional.mse_loss(self.model(self.x), self.y).item()
            )
        return EpochTrainingResult(training_steps=2, selection_score=score)

    def snapshot(self, epoch: int, completed_training_steps: int) -> TrainingStateSnapshot:
        def serialize(value: Any) -> bytes:
            stream = io.BytesIO()
            self.torch.save(value, stream)
            return stream.getvalue()

        return TrainingStateSnapshot(
            model_state=serialize(self.model.state_dict()),
            optimizer_state=serialize(self.optimizer.state_dict()),
            scheduler_state=serialize(self.scheduler.state_dict()),
            stage_state=(
                f'{{"epoch":{epoch},"steps":{completed_training_steps}}}'
            ).encode("utf-8"),
        )

    def peak_device_memory_bytes(self) -> int:
        return 0


class _BoundedBackend:
    def __init__(self, *, peak_memory: int = 0) -> None:
        self.peak_memory = peak_memory
        self.prepared = False

    @property
    def device_identity(self) -> str:
        return "cpu:test"

    @property
    def parameter_count(self) -> int:
        return 2

    def prepare(self, config: FrozenRunConfig, seed: int) -> None:
        self.prepared = True

    def train_epoch(self, epoch: int) -> EpochTrainingResult:
        assert self.prepared
        return EpochTrainingResult(training_steps=2, selection_score=float(epoch))

    def snapshot(self, epoch: int, completed_training_steps: int) -> TrainingStateSnapshot:
        suffix = f"{epoch}:{completed_training_steps}".encode("ascii")
        return TrainingStateSnapshot(
            model_state=b"model:" + suffix,
            optimizer_state=b"optimizer:" + suffix,
            scheduler_state=b"scheduler:" + suffix,
            stage_state=b"stage:" + suffix,
        )

    def peak_device_memory_bytes(self) -> int:
        return self.peak_memory


def test_small_torch_training_records_measurements_and_retains_checkpoint_roles(
    tmp_path: Path,
) -> None:
    engine, registry, store = _engine(tmp_path)
    config = _config(_budget())
    run_id = _allocate(registry, config)

    result = engine.train_inner(
        run_id=run_id,
        frozen_config=config,
        backend=_TorchToyBackend(),
    )

    assert result.status is InnerTrainingStatus.COMPLETED
    assert result.execution_mode is ExecutionMode.DETERMINISTIC
    assert result.framework_settings.deterministic_algorithms is True
    assert result.framework_settings.cudnn_benchmark is False
    assert result.measurements.completed_epochs == 2
    assert result.measurements.completed_training_steps == 4
    assert result.measurements.device_identity == "cpu"
    assert result.measurements.numeric_precision == "float32"
    assert result.measurements.parameter_count == 2
    assert result.measurements.input_volume_bytes == 24
    assert result.measurements.wall_clock_duration_ms >= 0
    assert result.measurements.peak_device_memory_bytes == 0
    assert {checkpoint.kind for checkpoint in result.retained_checkpoints} == {
        TrainingCheckpointKind.SELECTED,
        TrainingCheckpointKind.LAST,
    }
    assert result.selected_epoch == 2
    assert len(result.recovery_artifacts) == 2
    assert len(registry.recovery_artifacts(run_id)) == 2
    assert registry.latest_run(run_id).status is RunStatus.RUNNING
    for checkpoint in result.retained_checkpoints:
        assert (store.root / checkpoint.artifact.relative_path).is_file()
        assert registry.get_artifact(checkpoint.artifact.artifact_id).run_id == run_id


def test_hard_limit_stops_at_first_complete_boundary_and_records_advisory_warning(
    tmp_path: Path,
) -> None:
    engine, registry, _ = _engine(tmp_path)
    config = _config(_budget(step_limit=1, memory_limit=1), epochs=5)
    run_id = _allocate(registry, config)

    result = engine.train_inner(
        run_id=run_id,
        frozen_config=config,
        backend=_BoundedBackend(peak_memory=2),
    )

    assert result.status is InnerTrainingStatus.HARD_LIMIT_STOPPED
    assert result.measurements.completed_epochs == 1
    assert result.measurements.completed_training_steps == 2
    assert result.budget_assessment.stopping_checkpoint == 1
    assert [item.field.value for item in result.budget_assessment.hard_exceeded] == [
        "completed_training_steps"
    ]
    assert [
        item.field.value for item in result.budget_assessment.advisory_exceeded
    ] == ["peak_device_memory_bytes"]
    assert len(result.recovery_artifacts) == 1
    assert len(registry.recovery_artifacts(run_id)) == 1


def test_retention_policy_can_disable_all_checkpoint_publication(tmp_path: Path) -> None:
    engine, registry, store = _engine(tmp_path)
    config = _config(
        _budget(),
        epochs=1,
        retention=CheckpointRetentionPolicy(
            preserve_selected=False,
            preserve_last=False,
            preserve_recovery_state=False,
        ),
    )
    run_id = _allocate(registry, config)

    result = engine.train_inner(
        run_id=run_id,
        frozen_config=config,
        backend=_BoundedBackend(),
    )

    assert result.status is InnerTrainingStatus.COMPLETED
    assert result.retained_checkpoints == ()
    assert result.recovery_artifacts == ()
    assert not (store.root / "training").exists()


def test_frozen_config_mismatch_is_rejected_before_backend_preparation(
    tmp_path: Path,
) -> None:
    engine, registry, store = _engine(tmp_path)
    allocated_config = _config(_budget(), epochs=1)
    run_id = _allocate(registry, allocated_config)
    different_config = _config(_budget(), epochs=2)
    backend = _BoundedBackend()

    with pytest.raises(ValueError, match="config_hash"):
        engine.train_inner(
            run_id=run_id,
            frozen_config=different_config,
            backend=backend,
        )

    assert backend.prepared is False
    assert registry.latest_run(run_id).status is RunStatus.PLANNED
    assert not (store.root / "training").exists()
