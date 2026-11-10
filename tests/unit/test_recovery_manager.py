"""Targeted tests for checkpoint recovery and missing-job scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from benchmark.recovery import (
    AggregateJobSpec,
    FailureAtomicArtifactStore,
    RecoveryManager,
    ResumeRejectedError,
)
from benchmark.registry import (
    ArtifactRecord,
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.schema import (
    ArtifactReference,
    ClaimStatus,
    ImplementationKind,
    LimitEnforcement,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    RuntimeMeasurements,
)


NOW = datetime(2027, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
HASHES = {
    name: f"{index:064x}"
    for index, name in enumerate(
        (
            "dataset",
            "protocol",
            "config_a",
            "config_b",
            "environment",
            "source",
            "checkpoint",
            "prediction",
        ),
        start=1,
    )
}


def _identity(
    *,
    fold_id: str = "fold-1",
    model_key: str = "classic.rf.v1",
    seed: int = 1,
    config_hash: str = HASHES["config_a"],
) -> FrozenRunIdentity:
    return FrozenRunIdentity(
        dataset_hash=HASHES["dataset"],
        protocol_hash=HASHES["protocol"],
        model_key=model_key,
        config_hash=config_hash,
        fold_id=fold_id,
        seed=seed,
        environment_lock_hash=HASHES["environment"],
        environment_compatibility_policy="exact",
    )


def _allocate(registry: RunRegistry, identity: FrozenRunIdentity) -> RunRecord:
    return registry.allocate_run(
        frozen_identity=identity,
        model_identity=ModelIdentity(
            model_key=identity.model_key,
            display_name=identity.model_key,
            class_fully_qualified_name="example.Model",
            implementation_kind=ImplementationKind.CLASSIC,
            parameter_count=2,
            source_revision="revision-1",
        ),
        input_contract=InputContract(
            representation_key="spectrum.v1",
            modalities=("spectrum",),
            tensor_shapes={"spectrum": (2,)},
            dtypes={"spectrum": "float32"},
            channel_or_band_order={"spectrum": ("ascending",)},
            preprocessing_key="none.v1",
        ),
        provenance=Provenance(
            code_revision="revision-1",
            source_hash=HASHES["source"],
            environment_hash=identity.environment_lock_hash,
            config_hash=identity.config_hash,
            dataset_hash=identity.dataset_hash,
        ),
        resource_budget=ResourceBudget(
            limits=(
                ResourceLimit(
                    field="completed_epochs",
                    value=1,
                    unit="count",
                    enforcement=LimitEnforcement.HARD,
                ),
            )
        ),
        claim_decision=ClaimDecision(status=ClaimStatus.STRICT_CANDIDATE),
        started_utc=NOW,
    )


def _append_running(registry: RunRegistry, planned: RunRecord) -> RunRecord:
    running = RunRecord.model_validate(
        {**planned.model_dump(mode="python"), "status": RunStatus.RUNNING}
    )
    registry.append_run(running)
    return running


def _complete(registry: RunRegistry, planned: RunRecord, suffix: str) -> RunRecord:
    checkpoint = ArtifactReference(
        artifact_id=f"checkpoint-{suffix}",
        relative_path=f"checkpoints/{suffix}.bin",
        content_hash=HASHES["checkpoint"],
    )
    prediction = ArtifactReference(
        artifact_id=f"prediction-{suffix}",
        relative_path=f"predictions/{suffix}.json",
        content_hash=HASHES["prediction"],
    )
    for reference, kind in ((checkpoint, "checkpoint"), (prediction, "prediction")):
        registry.append_artifact(
            ArtifactRecord(
                artifact=reference,
                artifact_kind=kind,
                run_id=planned.run_id,
                created_utc=NOW + timedelta(minutes=1),
            )
        )
    complete = RunRecord.model_validate(
        {
            **planned.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "checkpoint_ref": checkpoint,
            "prediction_artifact_ref": prediction,
            "runtime_measurements": RuntimeMeasurements(
                completed_epochs=1,
                completed_training_steps=1,
                wall_clock_duration_ms=1,
                peak_device_memory_bytes=1,
                device_identity="cpu",
                numeric_precision="float32",
                parameter_count=2,
                input_volume_bytes=8,
            ),
            "completed_utc": NOW + timedelta(minutes=2),
        }
    )
    registry.append_run(complete)
    return complete


def _manager(tmp_path: Path) -> tuple[RecoveryManager, RunRegistry]:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    return RecoveryManager(registry, store), registry


@pytest.mark.parametrize("status", (RunStatus.FAILED, RunStatus.ABORTED))
def test_interruption_appends_terminal_status_and_last_stage_without_touching_artifacts(
    tmp_path: Path,
    status: RunStatus,
) -> None:
    manager, registry = _manager(tmp_path)
    running = _append_running(registry, _allocate(registry, _identity()))
    completed = manager.artifact_store.publish_bytes(
        artifact_id="completed-before-interruption",
        relative_path="completed/existing.bin",
        content=b"immutable-completed-artifact",
    )
    artifact_path = tmp_path / "artifacts" / completed.relative_path
    artifact_before = artifact_path.read_bytes()

    interrupted = manager.record_interruption(
        running.run_id,
        status=status,
        last_completed_stage="outer_refit",
        completed_utc=NOW + timedelta(minutes=1),
    )

    assert interrupted.status is status
    assert interrupted.last_completed_stage == "outer_refit"
    assert interrupted.recovery_artifact_ref is None
    assert [record.status for record in registry.run_history(running.run_id)] == [
        RunStatus.PLANNED,
        RunStatus.RUNNING,
        status,
    ]
    assert RunRegistry(registry.path).latest_run(running.run_id) == interrupted
    assert artifact_path.read_bytes() == artifact_before


def test_checkpoint_boundary_round_trip_restores_all_four_states(
    tmp_path: Path,
) -> None:
    manager, registry = _manager(tmp_path)
    running = _append_running(registry, _allocate(registry, _identity()))
    expected_states = {
        "model": b"model-state\x00",
        "optimizer": b"optimizer-state\x01",
        "scheduler": b"scheduler-state\x02",
        "stage": b'{"epoch":3,"step":17}',
    }
    recovery = manager.save_checkpoint_boundary(
        run_id=running.run_id,
        recovery_id="recovery-boundary-3",
        relative_path="recovery/recovery-boundary-3.json",
        model_state=expected_states["model"],
        optimizer_state=expected_states["optimizer"],
        scheduler_state=expected_states["scheduler"],
        stage_state=expected_states["stage"],
        last_completed_stage="inner_training",
        completed_checkpoint_boundary=3,
        created_utc=NOW + timedelta(minutes=1),
    )
    artifact_bytes = (
        tmp_path / "artifacts" / recovery.artifact.relative_path
    ).read_bytes()
    interrupted = manager.record_interruption(
        running.run_id,
        status=RunStatus.ABORTED,
        last_completed_stage="inner_training",
        completed_utc=NOW + timedelta(minutes=2),
    )
    restored: dict[str, bytes] = {}
    state = manager.resume(
        running.run_id,
        _identity(),
        restore_model=lambda value: restored.__setitem__("model", value),
        restore_optimizer=lambda value: restored.__setitem__("optimizer", value),
        restore_scheduler=lambda value: restored.__setitem__("scheduler", value),
        restore_stage=lambda value: restored.__setitem__("stage", value),
    )

    assert restored == expected_states
    assert state.recovery == recovery
    assert interrupted.last_completed_stage == "inner_training"
    assert interrupted.recovery_artifact_ref == recovery.artifact
    assert registry.latest_recovery(running.run_id) == recovery
    assert (tmp_path / "artifacts" / recovery.artifact.relative_path).read_bytes() == artifact_bytes


def test_resume_rejects_every_mismatch_before_state_restoration(
    tmp_path: Path,
) -> None:
    manager, registry = _manager(tmp_path)
    running = _append_running(registry, _allocate(registry, _identity()))
    recovery = manager.save_checkpoint_boundary(
        run_id=running.run_id,
        recovery_id="recovery-boundary-1",
        relative_path="recovery/recovery-boundary-1.json",
        model_state=b"model",
        optimizer_state=b"optimizer",
        scheduler_state=b"scheduler",
        stage_state=b"stage",
        last_completed_stage="selection",
        completed_checkpoint_boundary=1,
        created_utc=NOW + timedelta(minutes=1),
    )
    manager.record_interruption(
        running.run_id,
        status=RunStatus.FAILED,
        last_completed_stage="selection",
        completed_utc=NOW + timedelta(minutes=2),
    )
    observed = _identity().model_copy(
        update={
            "dataset_hash": f"{20:064x}",
            "protocol_hash": f"{21:064x}",
            "model_key": "classic.svm.v1",
            "config_hash": f"{22:064x}",
            "fold_id": "fold-2",
            "seed": 99,
            "environment_lock_hash": f"{23:064x}",
            "environment_compatibility_policy": "compatible-minor",
        }
    )
    registry_before = registry.path.read_bytes()
    artifact_path = tmp_path / "artifacts" / recovery.artifact.relative_path
    artifact_before = artifact_path.read_bytes()
    restored: list[bytes] = []

    with pytest.raises(ResumeRejectedError) as caught:
        manager.resume(
            running.run_id,
            observed,
            restore_model=restored.append,
            restore_optimizer=restored.append,
            restore_scheduler=restored.append,
            restore_stage=restored.append,
        )

    error = caught.value.error_record
    expected_fields = {
        "dataset_hash",
        "protocol_hash",
        "model_key",
        "config_hash",
        "fold_id",
        "seed",
        "environment_lock_hash",
        "environment_compatibility_policy",
    }
    assert caught.value.requires_new_run_id is True
    assert set(error.expected_value) == expected_fields
    assert set(error.observed_value) == expected_fields
    assert error.invariant_checks["state_restoration_started"] is False
    assert error.invariant_checks["completed_artifacts_unchanged"] is True
    assert restored == []
    assert registry.path.read_bytes() == registry_before
    assert artifact_path.read_bytes() == artifact_before

    replacement = _allocate(registry, observed)
    assert replacement.run_id != running.run_id
    assert replacement.frozen_identity == observed
    assert registry.latest_run(running.run_id).status is RunStatus.FAILED
    assert artifact_path.read_bytes() == artifact_before


def test_missing_jobs_is_exact_cartesian_difference_and_keeps_duplicates(
    tmp_path: Path,
) -> None:
    manager, registry = _manager(tmp_path)
    spec = AggregateJobSpec(
        dataset_hash=HASHES["dataset"],
        protocol_hash=HASHES["protocol"],
        environment_lock_hash=HASHES["environment"],
        environment_compatibility_policy="exact",
        fold_ids=("fold-1", "fold-2"),
        model_keys=("classic.rf.v1", "classic.svm.v1"),
        seeds=(1, 2),
        config_hashes=(HASHES["config_a"], HASHES["config_b"]),
    )
    completed_job = spec.declared_jobs()[5]
    identity = spec.frozen_identity(completed_job)
    first = _complete(registry, _allocate(registry, identity), "first")
    second = _complete(registry, _allocate(registry, identity), "second")
    planned_job = spec.declared_jobs()[0]
    running_job = spec.declared_jobs()[10]
    _allocate(registry, spec.frozen_identity(planned_job))
    _append_running(
        registry,
        _allocate(registry, spec.frozen_identity(running_job)),
    )

    missing = manager.missing_jobs(spec)

    assert len(spec.declared_jobs()) == 16
    assert missing == tuple(job for job in spec.declared_jobs() if job != completed_job)
    assert planned_job in missing
    assert running_job in missing
    assert completed_job not in missing
    assert registry.canonical_completed(identity) in (first, second)
    assert {record.run_id for record in registry.completed_runs(identity)} == {
        first.run_id,
        second.run_id,
    }
