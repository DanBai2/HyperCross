"""Property 39: Complete reproducibility record.

**Validates: Requirements 17.1-17.33, 17.35-17.38**
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from hypothesis import given, settings, strategies as st

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
    EnvironmentLock,
    ImplementationKind,
    LimitEnforcement,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    RuntimeMeasurements,
    ValidationReport,
    validate_schema,
)


_BASE_TIME = datetime(2027, 1, 1, tzinfo=timezone.utc)
_REQUIRED_OMISSIONS = st.sampled_from(
    (
        "checkpoint_ref",
        "prediction_artifact_ref",
        "frozen_identity.dataset_hash",
        "frozen_identity.protocol_hash",
        "frozen_identity.config_hash",
        "frozen_identity.environment_lock_hash",
        "provenance.source_hash",
        "provenance.config_hash",
        "provenance.dataset_hash",
        "provenance.environment_hash",
    )
)


def _hash(namespace: str, token: int) -> str:
    return hashlib.sha256(f"property39:{namespace}:{token}".encode("utf-8")).hexdigest()


def _reference(kind: str, index: int, token: int) -> ArtifactReference:
    artifact_id = f"property39-{kind}-{index}-{token}"
    return ArtifactReference(
        artifact_id=artifact_id,
        relative_path=f"{kind}/{artifact_id}.bin",
        content_hash=_hash(f"{kind}-{index}", token),
        media_type="application/octet-stream",
    )


def _complete_record(
    *,
    token: int,
    seed: int,
    bands: int,
    parameter_count: int,
    score: float,
    transform_count: int,
    implementation_kind: ImplementationKind,
    start_offset_seconds: int,
) -> tuple[RunRecord, EnvironmentLock, tuple[ArtifactReference, ...]]:
    started_utc = _BASE_TIME + timedelta(seconds=start_offset_seconds)
    completed_utc = started_utc + timedelta(seconds=token % 3600 + 1)
    downloaded_utc = started_utc - timedelta(days=1)
    device_identity = f"cuda:0:property39-device-{token % 17}"
    environment = EnvironmentLock.create(
        python_version=f"3.{9 + token % 5}.{token % 10}",
        pytorch_version=f"2.{token % 9}.{token % 10}",
        numpy_version=f"1.{20 + token % 7}.{token % 10}",
        pandas_version=f"2.{token % 4}.{token % 10}",
        pillow_version=f"10.{token % 5}.{token % 10}",
        tifffile_version=f"2027.{1 + token % 12}.{1 + token % 28}",
        sklearn_version=f"1.{token % 8}.{token % 10}",
        scipy_version=f"1.{10 + token % 7}.{token % 10}",
        cuda_version=f"12.{token % 10}",
        cuda_available=True,
        cuda_device_count=1,
        device_identity=device_identity,
    )

    dataset_hash = _hash("dataset", token)
    protocol_hash = _hash("protocol", token)
    config_hash = _hash("config", token)
    source_hash = _hash("source", token)
    patch_hash = _hash("patch", token)
    model_key = f"property39.external-model-{token}.v1"
    code_revision = f"commit-{token}"
    frozen_identity = FrozenRunIdentity(
        dataset_hash=dataset_hash,
        protocol_hash=protocol_hash,
        model_key=model_key,
        config_hash=config_hash,
        fold_id=f"outer-{token}",
        seed=seed,
        environment_lock_hash=environment.lock_hash,
        environment_compatibility_policy="exact",
    )
    model_identity = ModelIdentity(
        model_key=model_key,
        display_name=f"Property 39 external model {token}",
        class_fully_qualified_name="external.package.Property39Model",
        implementation_kind=implementation_kind,
        parameter_count=parameter_count,
        source_revision=code_revision,
    )
    input_contract = InputContract(
        representation_key=f"property39.full-cube-{bands}.v1",
        modalities=("cube",),
        tensor_shapes={"cube": (16, 16, bands)},
        dtypes={"cube": "float32"},
        channel_or_band_order={"cube": ("wavelength-ascending",)},
        preprocessing_key=f"property39.fold-scoped-{token}.v1",
    )

    transforms = tuple(
        _reference("transform", index, token) for index in range(transform_count)
    )
    checkpoint = _reference("checkpoint", 0, token)
    prediction = _reference("prediction", 0, token)
    artifacts = (*transforms, checkpoint, prediction)
    reproduction_result_hash = _hash("upstream-reproduction-result", token)
    artifact_hashes = {
        reference.artifact_id: reference.content_hash for reference in artifacts
    }
    artifact_hashes["upstream-reproduction-result"] = reproduction_result_hash

    provenance = Provenance(
        code_revision=code_revision,
        upstream_repository=f"https://example.org/property39/model-{token}",
        upstream_commit=code_revision,
        license_reference=f"https://example.org/property39/model-{token}/LICENSE",
        downloaded_utc=downloaded_utc,
        source_hash=source_hash,
        local_patch_hash=patch_hash,
        environment_hash=environment.lock_hash,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        artifact_hashes=artifact_hashes,
        upstream_reproduction_status="passed",
        adaptation_notes=(
            f"adapt bands to the declared {bands}-band input contract",
        ),
        ignored_modalities=(),
    )
    claim_decision = ClaimDecision(
        status=ClaimStatus.PRIMARY_STRICT,
        supporting_reasons=(
            "artifacts_complete",
            "hashes_complete",
            "external_provenance_complete",
        ),
        blocking_reasons=(),
        prerequisite_checks={
            "artifacts_complete": True,
            "environment_lock_complete": True,
            "external_provenance_complete": True,
            "hashes_complete": True,
        },
    )
    record = RunRecord(
        run_id=f"property39-run-{token}",
        frozen_identity=frozen_identity,
        model_identity=model_identity,
        input_contract=input_contract,
        provenance=provenance,
        transform_artifact_refs=transforms,
        checkpoint_ref=checkpoint,
        prediction_artifact_ref=prediction,
        metrics={"primary_score": score},
        resource_budget=ResourceBudget(
            limits=(
                ResourceLimit(
                    field="completed_epochs",
                    value=1 + token % 100,
                    unit="count",
                    enforcement=LimitEnforcement.HARD,
                ),
            )
        ),
        runtime_measurements=RuntimeMeasurements(
            completed_epochs=token % 100,
            completed_training_steps=token % 10_000,
            wall_clock_duration_ms=1 + token % 1_000_000,
            peak_device_memory_bytes=1 + token % 10_000_000,
            device_identity=device_identity,
            numeric_precision="float32",
            parameter_count=parameter_count,
            input_volume_bytes=16 * 16 * bands * 4,
        ),
        status=RunStatus.COMPLETE,
        claim_decision=claim_decision,
        started_utc=started_utc,
        completed_utc=completed_utc,
    )
    return record, environment, artifacts


def _case_strategy() -> st.SearchStrategy[dict[str, Any]]:
    return st.fixed_dictionaries(
        {
            "token": st.integers(min_value=1, max_value=2**31 - 1),
            "seed": st.integers(min_value=0, max_value=2**31 - 1),
            "bands": st.integers(min_value=1, max_value=512),
            "parameter_count": st.integers(min_value=0, max_value=10**9),
            "score": st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            "transform_count": st.integers(min_value=1, max_value=4),
            "implementation_kind": st.sampled_from(
                (ImplementationKind.OFFICIAL, ImplementationKind.ADAPTED_OFFICIAL)
            ),
            "start_offset_seconds": st.integers(
                min_value=0,
                max_value=365 * 24 * 60 * 60,
            ),
        }
    )


@settings(max_examples=100, deadline=None)
@given(case=_case_strategy())
def test_complete_formal_external_run_round_trips_all_reproducibility_fields(
    case: dict[str, Any],
) -> None:
    """A completed formal run retains every reproducibility field after reload."""

    record, environment, artifacts = _complete_record(**case)
    with TemporaryDirectory(prefix="property39-") as directory:
        registry_path = Path(directory) / "registry with spaces" / "runs.jsonl"
        registry = RunRegistry(registry_path)
        planned = registry.allocate_run(
            frozen_identity=record.frozen_identity,
            model_identity=record.model_identity,
            input_contract=record.input_contract,
            provenance=record.provenance,
            resource_budget=record.resource_budget,
            claim_decision=record.claim_decision,
            started_utc=record.started_utc,
        )
        persisted_record = record.model_copy(update={"run_id": planned.run_id})
        for index, reference in enumerate(artifacts):
            registry.append_artifact(
                ArtifactRecord(
                    artifact=reference,
                    artifact_kind=(
                        "transform"
                        if reference in persisted_record.transform_artifact_refs
                        else "checkpoint"
                        if reference == persisted_record.checkpoint_ref
                        else "prediction"
                    ),
                    run_id=planned.run_id,
                    created_utc=record.started_utc + timedelta(microseconds=index + 1),
                )
            )
        registry.append_run(persisted_record)

        stored = RunRegistry(registry_path).latest_run(planned.run_id)

    assert stored.status is RunStatus.COMPLETE
    assert stored.frozen_identity.dataset_hash == record.frozen_identity.dataset_hash
    assert stored.frozen_identity.protocol_hash == record.frozen_identity.protocol_hash
    assert stored.frozen_identity.config_hash == record.frozen_identity.config_hash
    assert stored.frozen_identity.environment_lock_hash == environment.lock_hash
    assert stored.frozen_identity.seed == case["seed"]
    assert stored.model_identity == record.model_identity
    assert stored.input_contract == record.input_contract
    assert stored.provenance.code_revision == record.model_identity.source_revision
    assert stored.transform_artifact_refs == record.transform_artifact_refs
    assert stored.checkpoint_ref == record.checkpoint_ref
    assert stored.prediction_artifact_ref == record.prediction_artifact_ref
    assert stored.metrics == record.metrics
    assert stored.started_utc == record.started_utc
    assert stored.completed_utc == record.completed_utc
    assert stored.claim_decision.status is ClaimStatus.PRIMARY_STRICT
    assert stored.claim_decision.blocking_reasons == ()
    assert stored.claim_decision.supporting_reasons
    assert stored.claim_decision.prerequisite_checks
    assert all(stored.claim_decision.prerequisite_checks.values())

    environment_values = environment.model_dump(mode="python")
    for field in (
        "python_version",
        "pytorch_version",
        "numpy_version",
        "pandas_version",
        "pillow_version",
        "tifffile_version",
        "sklearn_version",
        "scipy_version",
        "cuda_version",
        "device_identity",
        "lock_hash",
    ):
        assert environment_values[field]
    assert stored.runtime_measurements is not None
    assert stored.runtime_measurements.device_identity == environment.device_identity

    provenance = stored.provenance
    assert provenance.upstream_repository
    assert provenance.upstream_commit
    assert provenance.license_reference
    assert provenance.downloaded_utc is not None
    assert provenance.source_hash
    assert provenance.local_patch_hash
    assert provenance.environment_hash == environment.lock_hash
    assert provenance.upstream_reproduction_status == "passed"
    assert provenance.adaptation_notes
    assert provenance.artifact_hashes["upstream-reproduction-result"]
    for reference in (
        *stored.transform_artifact_refs,
        stored.checkpoint_ref,
        stored.prediction_artifact_ref,
    ):
        assert reference is not None
        assert provenance.artifact_hashes[reference.artifact_id] == reference.content_hash


@settings(max_examples=100, deadline=None)
@given(case=_case_strategy(), omitted_field=_REQUIRED_OMISSIONS)
def test_required_artifact_or_hash_omission_blocks_primary_strict_record(
    case: dict[str, Any],
    omitted_field: str,
) -> None:
    """Required artifact/hash omissions are rejected with machine-readable details."""

    record, _, _ = _complete_record(**case)
    candidate = record.model_dump(mode="python")
    owner: dict[str, Any] = candidate
    path = omitted_field.split(".")
    for part in path[:-1]:
        owner = owner[part]
    owner[path[-1]] = None

    validated, report = validate_schema(RunRecord, candidate)

    assert validated is None
    assert isinstance(report, ValidationReport)
    assert report.passes is False
    assert report.errors
    assert candidate["claim_decision"]["status"] is ClaimStatus.PRIMARY_STRICT
    machine_details = report.machine_details()
    assert machine_details["passes"] is False
    assert omitted_field.split(".")[-1] in str(machine_details)
