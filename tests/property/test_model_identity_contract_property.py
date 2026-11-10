"""Property 13: Stable model identity and input contract.

**Validates: Requirements 6.1-6.13, 6.24-6.31**
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.models import (
    ModelRegistration,
    ModelRegistry,
    ModelRegistryError,
    ReproductionGateError,
)
from benchmark.registry import ErrorJournal
from benchmark.schema import (
    ErrorRecord,
    ImplementationKind,
    InputContract,
    ModelIdentity,
    ModelProvenance,
    UpstreamReproductionRecord,
    UpstreamReproductionStatus,
)


_MODEL_KEY = "property13.external-model.v1"
_COMMIT = "property13-revision"
_REPOSITORY = "https://example.org/property13/model"
_LICENSE = "Apache-2.0"
_MUTATIONS = st.sampled_from(
    (
        "model_key",
        "class",
        "kind",
        "provenance",
        "contract.representation_key",
        "contract.modalities",
        "contract.tensor_shapes",
        "contract.dtypes",
        "contract.channel_or_band_order",
        "contract.preprocessing_key",
    )
)


class _Parameter:
    def __init__(self, size: int) -> None:
        self._size = size

    def numel(self) -> int:
        return self._size


class Property13Model:
    __benchmark_model_key__ = _MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.ADAPTED_OFFICIAL
    __benchmark_source_revision__ = _COMMIT

    def parameters(self):
        return iter((_Parameter(4), _Parameter(6)))


class DifferentProperty13Model(Property13Model):
    pass


class WrongKeyProperty13Model(Property13Model):
    __benchmark_model_key__ = "property13.wrong-model.v1"


@st.composite
def _model_cases(draw: st.DrawFn) -> tuple[int, int, str]:
    height = draw(st.integers(min_value=1, max_value=64))
    bands = draw(st.integers(min_value=2, max_value=256))
    mutation = draw(_MUTATIONS)
    return height, bands, mutation


def _hash(namespace: str, height: int, bands: int) -> str:
    payload = f"{namespace}:{height}:{bands}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract(height: int, bands: int, **updates: object) -> InputContract:
    values: dict[str, object] = {
        "representation_key": "property13.rgb-spectrum.v1",
        "modalities": ("rgb", "spectrum"),
        "tensor_shapes": {"rgb": (3, height, height), "spectrum": (bands,)},
        "dtypes": {"rgb": "float32", "spectrum": "float32"},
        "channel_or_band_order": {
            "rgb": ("red", "green", "blue"),
            "spectrum": ("wavelength_ascending",),
        },
        "preprocessing_key": "property13.fold-scoped.v1",
    }
    values.update(updates)
    return InputContract.model_validate(values)


def _reproduction_record(height: int, bands: int) -> UpstreamReproductionRecord:
    return UpstreamReproductionRecord(
        model_key=_MODEL_KEY,
        upstream_repository=_REPOSITORY,
        upstream_commit=_COMMIT,
        license_reference=_LICENSE,
        source_hash=_hash("source", height, bands),
        dependency_environment_hash=_hash("environment", height, bands),
        result_artifact_hash=_hash("result", height, bands),
        status=UpstreamReproductionStatus.PASSED,
        completed_utc=datetime(2027, 1, 2, tzinfo=timezone.utc),
    )


def _provenance(
    height: int,
    bands: int,
    *,
    record_source_hash: str | None = None,
) -> ModelProvenance:
    record = _reproduction_record(height, bands)
    if record_source_hash is not None:
        record = record.model_copy(update={"source_hash": record_source_hash})
    return ModelProvenance(
        upstream_repository=_REPOSITORY,
        upstream_commit=_COMMIT,
        license_reference=_LICENSE,
        downloaded_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        source_hash=_hash("source", height, bands),
        local_patch_hash=_hash("patch", height, bands),
        dependency_environment_hash=_hash("environment", height, bands),
        upstream_reproduction_status=UpstreamReproductionStatus.PASSED,
        upstream_reproduction_record=record,
        adaptation_notes=("adapt generated inputs to the declared contract",),
        ignored_modalities=(),
    )


def _identity(
    *,
    model_class: type[Property13Model] = Property13Model,
    kind: ImplementationKind = ImplementationKind.ADAPTED_OFFICIAL,
) -> ModelIdentity:
    return ModelIdentity(
        model_key=_MODEL_KEY,
        display_name="Property 13 adapted official model",
        class_fully_qualified_name=(
            f"{model_class.__module__}.{model_class.__qualname__}"
        ),
        implementation_kind=kind,
        parameter_count=10,
        source_revision=_COMMIT,
    )


def _registration(
    height: int,
    bands: int,
    *,
    identity: ModelIdentity | None = None,
    contract: InputContract | None = None,
    builder: type[Property13Model] = Property13Model,
    provenance: ModelProvenance | None = None,
) -> ModelRegistration:
    return ModelRegistration(
        model_key=_MODEL_KEY,
        identity=identity or _identity(),
        input_contract=contract or _contract(height, bands),
        builder=builder,
        implementation_kind=ImplementationKind.ADAPTED_OFFICIAL,
        source_revision=_COMMIT,
        provenance=provenance or _provenance(height, bands),
    )


def _changed_contract(height: int, bands: int, mutation: str) -> InputContract:
    if mutation == "contract.representation_key":
        return _contract(
            height,
            bands,
            representation_key="property13.changed-representation.v1",
        )
    if mutation == "contract.modalities":
        return _contract(
            height,
            bands,
            modalities=("cube",),
            tensor_shapes={"cube": (height, height, bands)},
            dtypes={"cube": "float32"},
            channel_or_band_order={"cube": ("wavelength_ascending",)},
        )
    if mutation == "contract.tensor_shapes":
        return _contract(
            height,
            bands,
            tensor_shapes={"rgb": (3, height + 1, height), "spectrum": (bands,)},
        )
    if mutation == "contract.dtypes":
        return _contract(
            height,
            bands,
            dtypes={"rgb": "float16", "spectrum": "float32"},
        )
    if mutation == "contract.channel_or_band_order":
        return _contract(
            height,
            bands,
            channel_or_band_order={
                "rgb": ("blue", "green", "red"),
                "spectrum": ("wavelength_ascending",),
            },
        )
    if mutation == "contract.preprocessing_key":
        return _contract(
            height,
            bands,
            preprocessing_key="property13.changed-preprocessing.v1",
        )
    raise AssertionError(f"unsupported contract mutation: {mutation}")


def _write_completed_state(root: Path) -> None:
    completed = {
        "registry/run-registry.jsonl": b'{"status":"complete"}\n',
        "checkpoints/completed.bin": b"completed-checkpoint",
        "predictions/completed.json": b'{"complete":true}',
    }
    for relative_path, content in completed.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _completed_state(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_valid_registration_records_identity(
    height: int,
    bands: int,
) -> None:
    registry = ModelRegistry()
    registration = _registration(height, bands)
    registry.register(registration)

    resolved = registry.resolve(_MODEL_KEY)
    assert resolved is registration
    assert resolved.identity is registration.identity
    assert resolved.input_contract is registration.input_contract

    instance = registry.build(_MODEL_KEY)
    metadata = registry.run_metadata(
        _MODEL_KEY,
        instance,
        config_hash=_hash("config", height, bands),
        supplied_contract=_contract(
            height,
            bands,
            modalities=("spectrum", "rgb"),
        ),
    )
    assert metadata.source_revision == _COMMIT
    assert metadata.config_hash == _hash("config", height, bands)
    assert metadata.parameter_count == 10
    assert metadata.representation_key == "property13.rgb-spectrum.v1"
    assert metadata.preprocessing_key == "property13.fold-scoped.v1"


@settings(max_examples=100, deadline=None)
@given(case=_model_cases())
def test_generated_identity_provenance_and_contract_drift_fails_atomically(
    case: tuple[int, int, str],
) -> None:
    """Every generated drift is rejected before training or completed-state mutation."""

    height, bands, mutation = case
    _assert_valid_registration_records_identity(height, bands)

    with TemporaryDirectory(prefix="property13-") as directory:
        root = Path(directory) / "completed state with spaces"
        _write_completed_state(root)
        completed_before = _completed_state(root)
        journal = ErrorJournal(Path(directory) / "errors" / "model-errors.jsonl")
        registry = ModelRegistry(journal)
        training_started = False

        identity = _identity()
        builder: type[Property13Model] = Property13Model
        provenance = _provenance(height, bands)
        supplied_contract = _contract(height, bands)
        expected_field: str

        if mutation == "model_key":
            identity = _identity(model_class=WrongKeyProperty13Model)
            builder = WrongKeyProperty13Model
            expected_field = "model_key"
        elif mutation == "class":
            identity = _identity(model_class=DifferentProperty13Model)
            expected_field = "class_fully_qualified_name"
        elif mutation == "kind":
            identity = _identity(kind=ImplementationKind.OFFICIAL)
            expected_field = "implementation_kind"
        elif mutation == "provenance":
            provenance = _provenance(
                height,
                bands,
                record_source_hash=_hash("changed-source", height, bands),
            )
            expected_field = "source_hash"
        else:
            supplied_contract = _changed_contract(height, bands, mutation)
            expected_field = mutation.removeprefix("contract.")

        registration = _registration(
            height,
            bands,
            identity=identity,
            builder=builder,
            provenance=provenance,
        )
        registry_before = registry.snapshot()

        with pytest.raises((ModelRegistryError, ReproductionGateError)) as caught:
            registry.register(registration)
            resolved = registry.resolve(_MODEL_KEY)
            assert resolved.identity is registration.identity
            assert resolved.input_contract is registration.input_contract
            instance = registry.build(_MODEL_KEY)
            registry.run_metadata(
                _MODEL_KEY,
                instance,
                config_hash=_hash("config", height, bands),
                supplied_contract=supplied_contract,
            )
            training_started = True

        error = caught.value.error_record
        assert isinstance(error, ErrorRecord)
        assert error.affected_identifiers == {"model_key": _MODEL_KEY}
        assert error.failing_field == expected_field
        assert error.expected_value is not None
        assert error.observed_value is not None
        assert not training_started
        assert _completed_state(root) == completed_before

        if mutation == "provenance":
            assert registry.snapshot() == registry_before
            assert error.error_code == "UPSTREAM_REPRODUCTION_RECORD_INVALID"
            assert error.invariant_checks["training_not_started"] is True
            assert error.invariant_checks["completed_runs_unchanged"] is True
        elif mutation.startswith("contract."):
            assert registry.snapshot() != registry_before
            assert error.error_code.startswith("INPUT_CONTRACT_")
            assert error.invariant_checks == {
                "training_not_started": True,
                "checkpoint_unchanged": True,
                "prediction_artifact_unchanged": True,
                "completed_run_records_unchanged": True,
            }
        else:
            assert registry.snapshot() != registry_before
            assert error.error_code.startswith("MODEL_")
            assert error.invariant_checks == {
                "model_not_returned": True,
                "training_not_started": True,
                "completed_artifacts_unchanged": True,
            }

        persisted_errors = journal.path.read_text(encoding="utf-8").splitlines()
        assert len(persisted_errors) == 1
