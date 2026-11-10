"""Focused unit coverage for task 7 registry storage and selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.registry import (
    AggregateRecord,
    ArtifactRecord,
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    Provenance,
    RegistryCorruptionError,
    RegistryInvariantError,
    RunRecord,
    RunRegistry,
)
from benchmark.schema import (
    ArtifactReference,
    ClaimStatus,
    ImplementationKind,
    InputContract as SchemaInputContract,
    LimitEnforcement,
    ModelIdentity as SchemaModelIdentity,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    RuntimeMeasurements,
)


NOW = datetime(2027, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
HASHES = {name: f"{index:064x}" for index, name in enumerate(
    ("dataset", "protocol", "config", "environment", "source", "checkpoint", "prediction", "aggregate"),
    start=1,
)}


def _identity() -> FrozenRunIdentity:
    return FrozenRunIdentity(
        dataset_hash=HASHES["dataset"],
        protocol_hash=HASHES["protocol"],
        model_key="classic.rf.v1",
        config_hash=HASHES["config"],
        fold_id="outer-01",
        seed=42,
        environment_lock_hash=HASHES["environment"],
        environment_compatibility_policy="exact",
    )


def _model() -> ModelIdentity:
    return ModelIdentity(
        model_key="classic.rf.v1",
        display_name="Random Forest",
        class_fully_qualified_name="sklearn.ensemble.RandomForestRegressor",
        implementation_kind=ImplementationKind.CLASSIC,
        parameter_count=123,
        source_revision="revision-1",
    )


def _contract() -> InputContract:
    return InputContract(
        representation_key="global-spectrum.v1",
        modalities=("spectrum",),
        tensor_shapes={"spectrum": (224,)},
        dtypes={"spectrum": "float32"},
        channel_or_band_order={"spectrum": ("wavelength_ascending",)},
        preprocessing_key="standardize.v1",
    )


def _provenance() -> Provenance:
    return Provenance(
        code_revision="revision-1",
        source_hash=HASHES["source"],
        environment_hash=HASHES["environment"],
        config_hash=HASHES["config"],
        dataset_hash=HASHES["dataset"],
    )


def _claim() -> ClaimDecision:
    return ClaimDecision(
        status=ClaimStatus.STRICT_CANDIDATE,
        supporting_reasons=("protocol_valid",),
        blocking_reasons=("seed_set_incomplete",),
        prerequisite_checks={"protocol": True, "seed_set": False},
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        limits=(
            ResourceLimit(
                field="completed_epochs",
                value=10,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )


def _allocate(registry: RunRegistry) -> RunRecord:
    return registry.allocate_run(
        frozen_identity=_identity(),
        model_identity=_model(),
        input_contract=_contract(),
        provenance=_provenance(),
        resource_budget=_budget(),
        claim_decision=_claim(),
        started_utc=NOW,
    )


def _reference(artifact_id: str, kind: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        relative_path=f"{kind}/{artifact_id}.bin",
        content_hash=HASHES[kind],
        media_type="application/octet-stream",
    )


def test_registry_reexports_canonical_model_identity_and_input_contract(
    tmp_path: Path,
) -> None:
    assert ModelIdentity is SchemaModelIdentity
    assert InputContract is SchemaInputContract

    registry = RunRegistry(tmp_path / "registry.jsonl")
    planned = registry.allocate_run(
        frozen_identity=_identity(),
        model_identity=SchemaModelIdentity.model_validate(
            _model().model_dump(mode="python")
        ),
        input_contract=SchemaInputContract.model_validate(
            _contract().model_dump(mode="python")
        ),
        provenance=_provenance(),
        resource_budget=_budget(),
        claim_decision=_claim(),
        started_utc=NOW,
    )

    assert planned.model_identity is not None
    assert isinstance(planned.model_identity, SchemaModelIdentity)
    assert isinstance(planned.input_contract, SchemaInputContract)


def _complete_run(registry: RunRegistry, planned: RunRecord) -> RunRecord:
    checkpoint = _reference("checkpoint-1", "checkpoint")
    prediction = _reference("prediction-1", "prediction")
    registry.append_artifact(
        ArtifactRecord(
            artifact=checkpoint,
            artifact_kind="checkpoint",
            run_id=planned.run_id,
            created_utc=NOW + timedelta(minutes=1),
        )
    )
    registry.append_artifact(
        ArtifactRecord(
            artifact=prediction,
            artifact_kind="prediction",
            run_id=planned.run_id,
            created_utc=NOW + timedelta(minutes=2),
        )
    )
    running = RunRecord.model_validate(
        {**planned.model_dump(mode="python"), "status": RunStatus.RUNNING}
    )
    registry.append_run(running)
    complete = RunRecord.model_validate(
        {
            **running.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "checkpoint_ref": checkpoint,
            "prediction_artifact_ref": prediction,
            "metrics": {"mean_r2": 0.75, "undefined_metric": None},
            "runtime_measurements": RuntimeMeasurements(
                completed_epochs=10,
                completed_training_steps=100,
                wall_clock_duration_ms=2000,
                peak_device_memory_bytes=4096,
                device_identity="cuda:0:RTX 5080",
                numeric_precision="float32",
                parameter_count=123,
                input_volume_bytes=896,
            ),
            "completed_utc": NOW + timedelta(minutes=3),
        }
    )
    registry.append_run(complete)
    return complete


def _complete_duplicate_run(
    registry: RunRegistry,
    planned: RunRecord,
    *,
    suffix: str,
    completed_utc: datetime,
) -> RunRecord:
    checkpoint = _reference(f"checkpoint-{suffix}", "checkpoint")
    prediction = _reference(f"prediction-{suffix}", "prediction")
    registry.append_artifact(
        ArtifactRecord(
            artifact=checkpoint,
            artifact_kind="checkpoint",
            run_id=planned.run_id,
            created_utc=completed_utc - timedelta(minutes=2),
        )
    )
    registry.append_artifact(
        ArtifactRecord(
            artifact=prediction,
            artifact_kind="prediction",
            run_id=planned.run_id,
            created_utc=completed_utc - timedelta(minutes=1),
        )
    )
    complete = RunRecord.model_validate(
        {
            **planned.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "checkpoint_ref": checkpoint,
            "prediction_artifact_ref": prediction,
            "metrics": {"mean_r2": 0.75},
            "runtime_measurements": RuntimeMeasurements(
                completed_epochs=1,
                completed_training_steps=1,
                wall_clock_duration_ms=1,
                peak_device_memory_bytes=1,
                device_identity="cpu",
                numeric_precision="float32",
                parameter_count=123,
                input_volume_bytes=1,
            ),
            "completed_utc": completed_utc,
        }
    )
    registry.append_run(complete)
    return complete


def test_supersession_appends_trace_without_changing_original_record(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    original = _complete_duplicate_run(
        registry,
        _allocate(registry),
        suffix="original",
        completed_utc=NOW + timedelta(minutes=3),
    )
    replacement = _complete_duplicate_run(
        registry,
        _allocate(registry),
        suffix="replacement",
        completed_utc=NOW + timedelta(minutes=4),
    )
    original_snapshot = registry.latest_run(original.run_id)
    before = registry.path.read_bytes()

    supersession = registry.supersede(
        original.run_id,
        replacement.run_id,
        "Corrected the conclusion after provenance review",
        created_utc=NOW + timedelta(minutes=5),
    )

    assert registry.path.read_bytes().startswith(before)
    assert registry.latest_run(original.run_id) == original_snapshot
    assert supersession.original_record_id == original.run_id
    assert supersession.superseding_record_id == replacement.run_id
    assert RunRegistry(registry.path).get_supersession(original.run_id) == supersession


def test_supersession_rejects_unknown_or_second_replacement_without_append(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    original = _allocate(registry)
    replacement = _allocate(registry)
    registry.supersede(original.run_id, replacement.run_id, "First correction")
    before = registry.path.read_bytes()

    with pytest.raises(RegistryInvariantError, match="already superseded"):
        registry.supersede(original.run_id, "unknown-run", "Second correction")

    assert registry.path.read_bytes() == before


def test_canonical_completed_uses_completion_time_then_run_id_and_keeps_duplicates(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    first_planned = _allocate(registry)
    second_planned = _allocate(registry)
    completed_at = NOW + timedelta(minutes=3)
    first = _complete_duplicate_run(
        registry,
        first_planned,
        suffix="first",
        completed_utc=completed_at,
    )
    second = _complete_duplicate_run(
        registry,
        second_planned,
        suffix="second",
        completed_utc=completed_at,
    )

    duplicates = registry.completed_runs(_identity())
    expected = min((first, second), key=lambda item: item.run_id.encode("utf-8"))

    assert duplicates == tuple(sorted((first, second), key=lambda item: item.run_id.encode("utf-8")))
    assert registry.canonical_completed(_identity()) == expected
    assert {item.run_id for item in duplicates} == {first.run_id, second.run_id}
    assert len(registry.run_history(first.run_id)) == 2
    assert len(registry.run_history(second.run_id)) == 2


def test_canonical_completed_prefers_earlier_completion_and_exact_identity(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    later = _complete_duplicate_run(
        registry,
        _allocate(registry),
        suffix="later",
        completed_utc=NOW + timedelta(minutes=5),
    )
    earlier = _complete_duplicate_run(
        registry,
        _allocate(registry),
        suffix="earlier",
        completed_utc=NOW + timedelta(minutes=4),
    )
    other_identity = _identity().model_copy(update={"seed": 7})

    assert registry.canonical_completed(_identity()) == earlier
    assert registry.canonical_completed(other_identity) is None
    assert later in registry.completed_runs(_identity())


def test_allocate_persists_frozen_identity_and_complete_run_metadata(tmp_path: Path) -> None:
    path = tmp_path / "registry with spaces" / "records.jsonl"
    registry = RunRegistry(path)
    planned = _allocate(registry)

    assert planned.status is RunStatus.PLANNED
    assert path.exists()
    assert registry.latest_run(planned.run_id).frozen_identity == _identity()

    complete = _complete_run(registry, planned)
    reloaded = RunRegistry(path).latest_run(planned.run_id)

    assert reloaded == complete
    assert reloaded.provenance == _provenance()
    assert reloaded.input_contract == _contract()
    assert reloaded.metrics == {"mean_r2": 0.75, "undefined_metric": None}
    assert reloaded.claim_decision.status is ClaimStatus.STRICT_CANDIDATE
    assert reloaded.started_utc == NOW
    assert reloaded.completed_utc == NOW + timedelta(minutes=3)
    assert [item.status for item in registry.run_history(planned.run_id)] == [
        RunStatus.PLANNED,
        RunStatus.RUNNING,
        RunStatus.COMPLETE,
    ]


def test_completed_run_cannot_be_updated_and_existing_bytes_are_unchanged(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    complete = _complete_run(registry, _allocate(registry))
    before = registry.path.read_bytes()
    replacement = RunRecord.model_validate(
        {
            **complete.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "metrics": {"mean_r2": 0.99},
            "completed_utc": complete.completed_utc + timedelta(seconds=1),
        }
    )

    with pytest.raises(RegistryInvariantError, match="not allowed"):
        registry.append_run(replacement)

    assert registry.path.read_bytes() == before
    assert registry.latest_run(complete.run_id).metrics["mean_r2"] == 0.75


def test_frozen_run_fields_cannot_drift_between_appended_snapshots(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    planned = _allocate(registry)
    before = registry.path.read_bytes()
    changed_identity = planned.frozen_identity.model_copy(update={"seed": 7})
    changed = RunRecord.model_validate(
        {
            **planned.model_dump(mode="python"),
            "status": RunStatus.RUNNING,
            "frozen_identity": changed_identity,
        }
    )

    with pytest.raises(RegistryInvariantError, match="frozen_identity"):
        registry.append_run(changed)

    assert registry.path.read_bytes() == before
    assert registry.latest_run(planned.run_id).frozen_identity.seed == 42


def test_artifact_ids_are_append_only_and_survive_reload(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    planned = _allocate(registry)
    artifact = ArtifactRecord(
        artifact=_reference("prediction-1", "prediction"),
        artifact_kind="prediction",
        run_id=planned.run_id,
        created_utc=NOW,
    )
    registry.append_artifact(artifact)
    before = registry.path.read_bytes()

    with pytest.raises(RegistryInvariantError, match="cannot be overwritten"):
        registry.append_artifact(artifact)

    assert registry.path.read_bytes() == before
    assert RunRegistry(registry.path).get_artifact("prediction-1") == artifact


def test_aggregate_persists_exact_completed_membership_claim_and_provenance(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    complete = _complete_run(registry, _allocate(registry))
    aggregate_artifact = _reference("aggregate-1", "aggregate")
    registry.append_artifact(
        ArtifactRecord(
            artifact=aggregate_artifact,
            artifact_kind="aggregate",
            created_utc=NOW + timedelta(minutes=4),
        )
    )
    aggregate = AggregateRecord(
        aggregate_id="aggregate-1",
        member_run_ids=(complete.run_id,),
        prediction_hashes={complete.run_id: HASHES["prediction"]},
        dataset_hash=HASHES["dataset"],
        protocol_hash=HASHES["protocol"],
        model_key="classic.rf.v1",
        representation_key="global-spectrum.v1",
        metrics={"mean_r2": 0.75},
        artifact_refs=(aggregate_artifact,),
        claim_decision=_claim(),
        provenance=_provenance(),
        created_utc=NOW + timedelta(minutes=5),
    )
    registry.append_aggregate(aggregate)
    before = registry.path.read_bytes()

    with pytest.raises(RegistryInvariantError, match="already registered"):
        registry.append_aggregate(aggregate)

    assert registry.path.read_bytes() == before
    assert RunRegistry(registry.path).get_aggregate("aggregate-1") == aggregate


def test_complete_run_requires_registered_checkpoint_and_prediction(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    planned = _allocate(registry)
    complete = RunRecord.model_validate(
        {
            **planned.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "checkpoint_ref": _reference("checkpoint-1", "checkpoint"),
            "prediction_artifact_ref": _reference("prediction-1", "prediction"),
            "runtime_measurements": RuntimeMeasurements(
                completed_epochs=1,
                completed_training_steps=1,
                wall_clock_duration_ms=1,
                peak_device_memory_bytes=1,
                device_identity="cpu",
                numeric_precision="float32",
                parameter_count=123,
                input_volume_bytes=1,
            ),
            "completed_utc": NOW + timedelta(seconds=1),
        }
    )
    before = registry.path.read_bytes()

    with pytest.raises(RegistryInvariantError, match="unregistered artifact"):
        registry.append_run(complete)

    assert registry.path.read_bytes() == before


def test_registry_rejects_incomplete_or_malformed_persisted_lines(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_bytes(b'{"entry_kind":"run","record":')

    with pytest.raises(RegistryCorruptionError, match="complete record"):
        RunRegistry(path)


def test_record_schemas_reject_nonfinite_metrics_and_mutation() -> None:
    registry_values = {
        "run_id": "run-1",
        "frozen_identity": _identity(),
        "model_identity": _model(),
        "input_contract": _contract(),
        "provenance": _provenance(),
        "resource_budget": _budget(),
        "status": RunStatus.PLANNED,
        "claim_decision": _claim(),
        "started_utc": NOW,
    }
    with pytest.raises(ValidationError):
        RunRecord(**registry_values, metrics={"loss": float("nan")})

    record = RunRecord(**registry_values)
    with pytest.raises(ValidationError):
        record.status = RunStatus.RUNNING
