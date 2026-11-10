"""Property 29: Frozen-identity resume safety.

**Validates: Requirements 13.1, 13.2, 13.9-13.29**
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.recovery import (
    FailureAtomicArtifactStore,
    RecoveryManager,
    ResumeRejectedError,
)
from benchmark.registry import (
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.schema import (
    ClaimStatus,
    ImplementationKind,
    LimitEnforcement,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
)


_NOW = datetime(2027, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
_HASHES = {
    name: f"{index:064x}"
    for index, name in enumerate(
        ("dataset", "protocol", "config", "environment", "source"), start=1
    )
}
_FROZEN_FIELDS = (
    "dataset_hash",
    "protocol_hash",
    "model_key",
    "config_hash",
    "fold_id",
    "seed",
    "environment_lock_hash",
    "environment_compatibility_policy",
)
_STATE_BUNDLES = st.fixed_dictionaries(
    {
        "model": st.binary(max_size=64),
        "optimizer": st.binary(max_size=64),
        "scheduler": st.binary(max_size=64),
        "stage": st.binary(max_size=64),
    }
)
_INTERRUPTION_STATUSES = st.sampled_from((RunStatus.FAILED, RunStatus.ABORTED))


def _identity() -> FrozenRunIdentity:
    return FrozenRunIdentity(
        dataset_hash=_HASHES["dataset"],
        protocol_hash=_HASHES["protocol"],
        model_key="classic.rf.v1",
        config_hash=_HASHES["config"],
        fold_id="fold-1",
        seed=17,
        environment_lock_hash=_HASHES["environment"],
        environment_compatibility_policy="exact",
    )


def _allocate(registry: RunRegistry, identity: FrozenRunIdentity) -> RunRecord:
    return registry.allocate_run(
        frozen_identity=identity,
        model_identity=ModelIdentity(
            model_key=identity.model_key,
            display_name=identity.model_key,
            class_fully_qualified_name="property29.Model",
            implementation_kind=ImplementationKind.CLASSIC,
            parameter_count=1,
            source_revision="property29-revision",
        ),
        input_contract=InputContract(
            representation_key="property29.spectrum.v1",
            modalities=("spectrum",),
            tensor_shapes={"spectrum": (1,)},
            dtypes={"spectrum": "float32"},
            channel_or_band_order={"spectrum": ("ascending",)},
            preprocessing_key="property29.none.v1",
        ),
        provenance=Provenance(
            code_revision="property29-revision",
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


def _prepare_interrupted_run(
    root: Path,
    states: dict[str, bytes],
    status: RunStatus,
) -> tuple[RecoveryManager, RunRegistry, RunRecord, Path]:
    registry = RunRegistry(root / "registry" / "runs.jsonl")
    manager = RecoveryManager(
        registry,
        FailureAtomicArtifactStore(root / "artifacts with spaces"),
    )
    planned = _allocate(registry, _identity())

    # Allocation is durable before any running/training snapshot is appended.
    persisted = RunRegistry(registry.path).latest_run(planned.run_id)
    assert persisted.status is RunStatus.PLANNED
    assert persisted.frozen_identity == _identity()

    running = RunRecord.model_validate(
        {**planned.model_dump(mode="python"), "status": RunStatus.RUNNING}
    )
    registry.append_run(running)
    recovery = manager.save_checkpoint_boundary(
        run_id=planned.run_id,
        recovery_id="property29-recovery-1",
        relative_path="recovery/property29-boundary-1.json",
        model_state=states["model"],
        optimizer_state=states["optimizer"],
        scheduler_state=states["scheduler"],
        stage_state=states["stage"],
        last_completed_stage="inner_training",
        completed_checkpoint_boundary=1,
        created_utc=_NOW + timedelta(minutes=1),
    )
    manager.record_interruption(
        planned.run_id,
        status=status,
        last_completed_stage="inner_training",
        completed_utc=_NOW + timedelta(minutes=2),
    )
    artifact_path = manager.artifact_store.root / recovery.artifact.relative_path
    return manager, registry, planned, artifact_path


def _changed_identity(changed_fields: set[str]) -> FrozenRunIdentity:
    values = _identity().model_dump(mode="python")
    changed_hashes = {
        "dataset_hash": f"{101:064x}",
        "protocol_hash": f"{102:064x}",
        "config_hash": f"{103:064x}",
        "environment_lock_hash": f"{104:064x}",
    }
    for field in changed_fields:
        if field in changed_hashes:
            values[field] = changed_hashes[field]
        elif field == "model_key":
            values[field] = "classic.svm.v1"
        elif field == "fold_id":
            values[field] = "fold-2"
        elif field == "seed":
            values[field] = values[field] + 1
        elif field == "environment_compatibility_policy":
            values[field] = "compatible-minor"
        else:  # pragma: no cover - the strategy only generates declared fields
            raise AssertionError(f"unsupported frozen field: {field}")
    return FrozenRunIdentity.model_validate(values)


@settings(max_examples=100, deadline=None)
@given(states=_STATE_BUNDLES, status=_INTERRUPTION_STATUSES)
def test_fully_compatible_identity_restores_all_four_states(
    states: dict[str, bytes],
    status: RunStatus,
) -> None:
    """A fully compatible resume restores each saved state exactly once."""

    with TemporaryDirectory(prefix="property29-compatible-") as directory:
        manager, registry, planned, artifact_path = _prepare_interrupted_run(
            Path(directory), states, status
        )
        artifact_before = artifact_path.read_bytes()
        restored: list[tuple[str, bytes]] = []

        result = manager.resume(
            planned.run_id,
            _identity(),
            restore_model=lambda value: restored.append(("model", value)),
            restore_optimizer=lambda value: restored.append(("optimizer", value)),
            restore_scheduler=lambda value: restored.append(("scheduler", value)),
            restore_stage=lambda value: restored.append(("stage", value)),
        )

        assert restored == [
            ("model", states["model"]),
            ("optimizer", states["optimizer"]),
            ("scheduler", states["scheduler"]),
            ("stage", states["stage"]),
        ]
        assert result.model_state == states["model"]
        assert result.optimizer_state == states["optimizer"]
        assert result.scheduler_state == states["scheduler"]
        assert result.stage_state == states["stage"]
        assert artifact_path.read_bytes() == artifact_before

        another_run = _allocate(registry, _identity())
        assert another_run.run_id != planned.run_id
        assert RunRegistry(registry.path).latest_run(another_run.run_id) == another_run


@settings(max_examples=100, deadline=None)
@given(
    states=_STATE_BUNDLES,
    status=_INTERRUPTION_STATUSES,
    changed_fields=st.sets(
        st.sampled_from(_FROZEN_FIELDS),
        min_size=1,
        max_size=len(_FROZEN_FIELDS),
    ),
)
def test_any_frozen_identity_mismatch_rejects_before_restoration_and_requires_new_run(
    states: dict[str, bytes],
    status: RunStatus,
    changed_fields: set[str],
) -> None:
    """Any generated frozen-field drift rejects in-place restoration atomically."""

    with TemporaryDirectory(prefix="property29-mismatch-") as directory:
        manager, registry, planned, artifact_path = _prepare_interrupted_run(
            Path(directory), states, status
        )
        observed = _changed_identity(changed_fields)
        registry_before = registry.path.read_bytes()
        artifact_before = artifact_path.read_bytes()
        restoration_calls: list[bytes] = []

        with pytest.raises(ResumeRejectedError) as caught:
            manager.resume(
                planned.run_id,
                observed,
                restore_model=restoration_calls.append,
                restore_optimizer=restoration_calls.append,
                restore_scheduler=restoration_calls.append,
                restore_stage=restoration_calls.append,
            )

        error = caught.value.error_record
        assert caught.value.requires_new_run_id is True
        assert set(error.expected_value) == changed_fields
        assert set(error.observed_value) == changed_fields
        assert set(error.affected_identifiers["mismatched_fields"].split(",")) == changed_fields
        assert error.invariant_checks == {
            "state_restoration_started": False,
            "completed_artifacts_unchanged": True,
            "requires_new_run_id": True,
        }
        assert restoration_calls == []
        assert registry.path.read_bytes() == registry_before
        assert artifact_path.read_bytes() == artifact_before

        replacement = _allocate(registry, observed)
        assert replacement.run_id != planned.run_id
        assert replacement.frozen_identity == observed
        assert registry.latest_run(planned.run_id).status is status
        assert artifact_path.read_bytes() == artifact_before
