"""Property 31: exact missing jobs and canonical duplicate selection.

**Validates: Requirements 13.30-13.32**
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import given, settings, strategies as st

from benchmark.recovery import AggregateJobSpec, FailureAtomicArtifactStore, RecoveryManager
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


_NOW = datetime(2027, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
_HASHES = {
    name: hashlib.sha256(name.encode("utf-8")).hexdigest()
    for name in (
        "dataset",
        "protocol",
        "environment",
        "source",
        "checkpoint",
        "prediction",
    )
}


def _config_hash(index: int) -> str:
    return hashlib.sha256(f"config-{index}".encode("utf-8")).hexdigest()


def _allocate(registry: RunRegistry, identity: FrozenRunIdentity) -> RunRecord:
    return registry.allocate_run(
        frozen_identity=identity,
        model_identity=ModelIdentity(
            model_key=identity.model_key,
            display_name=identity.model_key,
            class_fully_qualified_name="property31.Model",
            implementation_kind=ImplementationKind.CLASSIC,
            parameter_count=2,
            source_revision="property31-revision",
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
            code_revision="property31-revision",
            source_hash=_HASHES["source"],
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
        started_utc=_NOW,
    )


def _complete(
    registry: RunRegistry,
    identity: FrozenRunIdentity,
    *,
    suffix: str,
    completed_utc: datetime,
) -> RunRecord:
    planned = _allocate(registry, identity)
    checkpoint = ArtifactReference(
        artifact_id=f"checkpoint-{suffix}",
        relative_path=f"checkpoints/{suffix}.bin",
        content_hash=_HASHES["checkpoint"],
    )
    prediction = ArtifactReference(
        artifact_id=f"prediction-{suffix}",
        relative_path=f"predictions/{suffix}.json",
        content_hash=_HASHES["prediction"],
    )
    for reference, kind in ((checkpoint, "checkpoint"), (prediction, "prediction")):
        registry.append_artifact(
            ArtifactRecord(
                artifact=reference,
                artifact_kind=kind,
                run_id=planned.run_id,
                created_utc=completed_utc,
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
            "completed_utc": completed_utc,
        }
    )
    registry.append_run(complete)
    return complete


@st.composite
def _declared_completed_cases(
    draw: st.DrawFn,
) -> tuple[AggregateJobSpec, set[int], int, tuple[int, ...]]:
    fold_count = draw(st.integers(min_value=1, max_value=2))
    model_count = draw(st.integers(min_value=1, max_value=2))
    seed_count = draw(st.integers(min_value=1, max_value=2))
    config_count = draw(st.integers(min_value=1, max_value=2))
    spec = AggregateJobSpec(
        dataset_hash=_HASHES["dataset"],
        protocol_hash=_HASHES["protocol"],
        environment_lock_hash=_HASHES["environment"],
        environment_compatibility_policy="exact",
        fold_ids=tuple(f"fold-{index}" for index in range(fold_count)),
        model_keys=tuple(f"classic.model-{index}.v1" for index in range(model_count)),
        seeds=tuple(range(seed_count)),
        config_hashes=tuple(sorted(_config_hash(index) for index in range(config_count))),
    )
    job_count = len(spec.declared_jobs())
    completed_indices = draw(
        st.sets(
            st.integers(min_value=0, max_value=job_count - 1),
            min_size=1,
            max_size=min(job_count, 4),
        )
    )
    duplicate_index = draw(st.sampled_from(sorted(completed_indices)))
    duplicate_count = draw(st.integers(min_value=2, max_value=3))
    completion_offsets = tuple(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=5),
                min_size=duplicate_count,
                max_size=duplicate_count,
            )
        )
    )
    return spec, completed_indices, duplicate_index, completion_offsets


@settings(max_examples=100, deadline=None)
@given(case=_declared_completed_cases())
def test_declared_minus_completed_is_exact_and_duplicates_have_one_canonical_member(
    case: tuple[AggregateJobSpec, set[int], int, tuple[int, ...]],
) -> None:
    """Only missing identities are scheduled while all completed duplicates remain auditable."""

    spec, completed_indices, duplicate_index, completion_offsets = case
    declared_jobs = spec.declared_jobs()

    with tempfile.TemporaryDirectory(prefix="property31-") as directory:
        root = Path(directory)
        registry_path = root / "registry.jsonl"
        registry = RunRegistry(registry_path)
        records_by_index: dict[int, list[RunRecord]] = {}

        for index in sorted(completed_indices):
            offsets = completion_offsets if index == duplicate_index else (index % 6,)
            records_by_index[index] = [
                _complete(
                    registry,
                    spec.frozen_identity(declared_jobs[index]),
                    suffix=f"job-{index}-duplicate-{duplicate_number}",
                    completed_utc=_NOW + timedelta(minutes=offset),
                )
                for duplicate_number, offset in enumerate(offsets)
            ]

        reloaded_registry = RunRegistry(registry_path)
        manager = RecoveryManager(
            reloaded_registry,
            FailureAtomicArtifactStore(root / "artifacts"),
        )
        expected_missing = tuple(
            job
            for index, job in enumerate(declared_jobs)
            if index not in completed_indices
        )

        assert manager.missing_jobs(spec) == expected_missing

        for index in sorted(completed_indices):
            identity = spec.frozen_identity(declared_jobs[index])
            expected_records = records_by_index[index]
            audited_records = reloaded_registry.completed_runs(identity)
            expected_canonical = min(
                expected_records,
                key=lambda record: (
                    record.completed_utc,
                    record.run_id.encode("utf-8"),
                ),
            )

            assert {record.run_id for record in audited_records} == {
                record.run_id for record in expected_records
            }
            assert len(audited_records) == len(expected_records)
            assert reloaded_registry.canonical_completed(identity) == expected_canonical
            assert declared_jobs[index] not in manager.missing_jobs(spec)

        duplicate_identity = spec.frozen_identity(declared_jobs[duplicate_index])
        duplicate_audit = reloaded_registry.completed_runs(duplicate_identity)
        assert len(duplicate_audit) == len(completion_offsets)
        assert len({record.run_id for record in duplicate_audit}) == len(completion_offsets)
        assert reloaded_registry.canonical_completed(duplicate_identity) in duplicate_audit
