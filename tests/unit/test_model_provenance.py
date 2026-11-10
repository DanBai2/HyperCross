"""Targeted tests for task 6.3 provenance and reproduction gating."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.models import (
    ModelRegistration,
    ModelRegistry,
    ModelRegistryError,
    ReproductionGateError,
)
from benchmark.registry import ErrorJournal
from benchmark.schema import (
    ImplementationKind,
    InputContract,
    ModelIdentity,
    ModelProvenance,
    UpstreamReproductionRecord,
    UpstreamReproductionStatus,
)


HASHES = {
    "source": "1" * 64,
    "patch": "2" * 64,
    "environment": "3" * 64,
    "result": "4" * 64,
}
REPOSITORY = "https://example.org/upstream/model"
LICENSE = "Apache-2.0"
COMMIT = "fd08eb5edc9565f86b925a6b9abf8ebb63a5f065"
MODEL_KEY = "external.model.v1"


class ExternalToyModel:
    __benchmark_model_key__ = MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.ADAPTED_OFFICIAL
    __benchmark_source_revision__ = COMMIT

    def parameters(self):
        return iter(())


def _contract() -> InputContract:
    return InputContract(
        representation_key="synthetic.spectrum.v1",
        modalities=("spectrum",),
        tensor_shapes={"spectrum": (128,)},
        dtypes={"spectrum": "float32"},
        channel_or_band_order={"spectrum": ("wavelength_ascending",)},
        preprocessing_key="synthetic.fold.v1",
    )


def _record(
    *,
    status: UpstreamReproductionStatus = UpstreamReproductionStatus.PASSED,
    source_hash: str = HASHES["source"],
) -> UpstreamReproductionRecord:
    return UpstreamReproductionRecord(
        model_key=MODEL_KEY,
        upstream_repository=REPOSITORY,
        upstream_commit=COMMIT,
        license_reference=LICENSE,
        source_hash=source_hash,
        dependency_environment_hash=HASHES["environment"],
        result_artifact_hash=HASHES["result"],
        status=status,
        completed_utc=datetime(2027, 1, 2, tzinfo=timezone.utc),
    )


def _provenance(
    *,
    status: UpstreamReproductionStatus = UpstreamReproductionStatus.PASSED,
    record: UpstreamReproductionRecord | None = None,
    adaptations: tuple[str, ...] = ("resize spatial input to 16x16",),
) -> ModelProvenance:
    return ModelProvenance(
        upstream_repository=REPOSITORY,
        upstream_commit=COMMIT,
        license_reference=LICENSE,
        downloaded_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        source_hash=HASHES["source"],
        local_patch_hash=HASHES["patch"],
        dependency_environment_hash=HASHES["environment"],
        upstream_reproduction_status=status,
        upstream_reproduction_record=record,
        adaptation_notes=adaptations,
        ignored_modalities=("depth", "text"),
    )


def _registration(
    kind: ImplementationKind,
    *,
    display_name: str,
    provenance: ModelProvenance | None = None,
) -> ModelRegistration:
    identity = ModelIdentity(
        model_key=MODEL_KEY,
        display_name=display_name,
        class_fully_qualified_name=(
            f"{ExternalToyModel.__module__}.{ExternalToyModel.__qualname__}"
        ),
        implementation_kind=kind,
        parameter_count=0,
        source_revision=COMMIT,
    )
    return ModelRegistration(
        model_key=MODEL_KEY,
        identity=identity,
        input_contract=_contract(),
        builder=ExternalToyModel,
        implementation_kind=kind,
        source_revision=COMMIT,
        provenance=provenance,
    )


def test_official_registration_preserves_complete_external_provenance() -> None:
    provenance = _provenance(
        status=UpstreamReproductionStatus.NOT_RUN,
        record=None,
        adaptations=(),
    )
    registry = ModelRegistry()

    registry.register(
        _registration(
            ImplementationKind.OFFICIAL,
            display_name="Official upstream model",
            provenance=provenance,
        )
    )

    stored = registry.resolve(MODEL_KEY).provenance
    assert stored is provenance
    assert stored is not None
    assert stored.upstream_repository == REPOSITORY
    assert stored.upstream_commit == COMMIT
    assert stored.license_reference == LICENSE
    assert stored.source_hash == HASHES["source"]
    assert stored.local_patch_hash == HASHES["patch"]
    assert stored.dependency_environment_hash == HASHES["environment"]
    assert stored.upstream_reproduction_status is UpstreamReproductionStatus.NOT_RUN
    assert stored.ignored_modalities == ("depth", "text")


def test_official_registration_requires_complete_provenance() -> None:
    registry = ModelRegistry()
    registration = _registration(
        ImplementationKind.OFFICIAL,
        display_name="Official upstream model",
        provenance=None,
    )

    with pytest.raises(ModelRegistryError) as caught:
        registry.register(registration)

    error = caught.value.error_record
    assert error.error_code == "OFFICIAL_PROVENANCE_MISSING"
    assert error.failing_field == "provenance"
    assert error.expected_value == "complete external ModelProvenance"
    assert error.observed_value is None
    assert registry.snapshot() == ()


def test_proxy_registration_requires_proxy_as_a_display_name_word() -> None:
    registry = ModelRegistry()
    invalid = _registration(
        ImplementationKind.PROXY,
        display_name="Compact inspired baseline",
    )

    with pytest.raises(ModelRegistryError) as caught:
        registry.register(invalid)

    assert caught.value.error_record.error_code == "PROXY_DISPLAY_NAME_MISSING_LABEL"
    assert caught.value.error_record.failing_field == "display_name"
    assert registry.snapshot() == ()

    registry.register(
        _registration(
            ImplementationKind.PROXY,
            display_name="Compact proxy baseline",
        )
    )
    assert registry.resolve(MODEL_KEY).identity.display_name == "Compact proxy baseline"


def test_adaptation_without_reproduction_record_is_blocked_and_journaled(
    tmp_path: Path,
) -> None:
    journal = ErrorJournal(tmp_path / "errors.jsonl")
    registry = ModelRegistry(journal)
    registration = _registration(
        ImplementationKind.ADAPTED_OFFICIAL,
        display_name="Adapted official model",
        provenance=_provenance(
            status=UpstreamReproductionStatus.NOT_RUN,
            record=None,
        ),
    )
    before = registry.snapshot()

    with pytest.raises(ReproductionGateError) as caught:
        registry.register(registration)

    error = caught.value.error_record
    assert error.error_code == "UPSTREAM_REPRODUCTION_RECORD_MISSING"
    assert error.failing_field == "upstream_reproduction_record"
    assert error.affected_identifiers == {"model_key": MODEL_KEY}
    assert error.invariant_checks["registered_upstream_artifacts_unchanged"] is True
    assert error.invariant_checks["completed_runs_unchanged"] is True
    assert registry.snapshot() == before
    persisted = json.loads(journal.path.read_text(encoding="utf-8"))
    assert persisted["error_code"] == "UPSTREAM_REPRODUCTION_RECORD_MISSING"


@pytest.mark.parametrize(
    ("status", "record"),
    (
        (
            UpstreamReproductionStatus.FAILED,
            _record(status=UpstreamReproductionStatus.FAILED),
        ),
        (
            UpstreamReproductionStatus.PASSED,
            _record(source_hash="9" * 64),
        ),
    ),
)
def test_adaptation_rejects_failed_or_mismatched_reproduction_evidence(
    status: UpstreamReproductionStatus,
    record: UpstreamReproductionRecord,
) -> None:
    registry = ModelRegistry()
    registration = _registration(
        ImplementationKind.ADAPTED_OFFICIAL,
        display_name="Adapted official model",
        provenance=_provenance(status=status, record=record),
    )

    with pytest.raises(ReproductionGateError) as caught:
        registry.register(registration)

    assert caught.value.error_record.error_code == "UPSTREAM_REPRODUCTION_RECORD_INVALID"
    assert registry.snapshot() == ()


def test_passing_reproduction_record_allows_formal_adaptation() -> None:
    record = _record()
    provenance = _provenance(record=record)
    registry = ModelRegistry()
    registration = _registration(
        ImplementationKind.ADAPTED_OFFICIAL,
        display_name="Adapted official model",
        provenance=provenance,
    )

    registry.register(registration)
    decision = registry.approve_adaptation(MODEL_KEY)

    assert decision.approved is True
    assert decision.formal_project_comparison_allowed is True
    assert all(decision.prerequisite_checks.values())
    assert registry.resolve(MODEL_KEY).provenance is provenance
    assert provenance.adaptation_notes == ("resize spatial input to 16x16",)
    assert provenance.ignored_modalities == ("depth", "text")
