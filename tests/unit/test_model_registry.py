"""Targeted tests for tasks 6.1 and 6.2 model registry validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.cli import main
from benchmark.models import ModelRegistration, ModelRegistry, ModelRegistryError
from benchmark.registry import ErrorJournal
from benchmark.schema import ImplementationKind, InputContract, ModelIdentity


class _Parameter:
    def __init__(self, size: int) -> None:
        self._size = size

    def numel(self) -> int:
        return self._size


class ToyModel:
    __benchmark_model_key__ = "toy.stable.v1"
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = "revision-abc"

    def parameters(self):
        return iter((_Parameter(6), _Parameter(4)))


class DifferentModel(ToyModel):
    pass


class HistoricalRouteOnlyModel:
    __benchmark_model_key__ = "leaf.route-only.v1"
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = "historical-selector-regression-v1"

    def parameters(self):
        return iter(())


class HistoricalTokenOnlyModel:
    __benchmark_model_key__ = "leaf.token-only.v1"
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = "historical-selector-regression-v1"

    def parameters(self):
        return iter(())


def _identity(**updates: object) -> ModelIdentity:
    values: dict[str, object] = {
        "model_key": "toy.stable.v1",
        "display_name": "Toy stable model",
        "class_fully_qualified_name": f"{ToyModel.__module__}.{ToyModel.__qualname__}",
        "implementation_kind": ImplementationKind.IN_REPO,
        "parameter_count": 10,
        "source_revision": "revision-abc",
    }
    values.update(updates)
    return ModelIdentity.model_validate(values)


def _contract(**updates: object) -> InputContract:
    values: dict[str, object] = {
        "representation_key": "synthetic.rgb-spectrum.v1",
        "modalities": ("rgb", "spectrum"),
        "tensor_shapes": {"rgb": (3, 16, 16), "spectrum": (128,)},
        "dtypes": {"rgb": "float32", "spectrum": "float32"},
        "channel_or_band_order": {
            "rgb": ("red", "green", "blue"),
            "spectrum": ("wavelength_ascending",),
        },
        "preprocessing_key": "synthetic.fold-scoped.v1",
    }
    values.update(updates)
    return InputContract.model_validate(values)


def _registration(
    *,
    identity: ModelIdentity | None = None,
    builder=ToyModel,
    implementation_kind: ImplementationKind = ImplementationKind.IN_REPO,
    source_revision: str = "revision-abc",
) -> ModelRegistration:
    return ModelRegistration(
        model_key="toy.stable.v1",
        identity=identity or _identity(),
        input_contract=_contract(),
        builder=builder,
        implementation_kind=implementation_kind,
        source_revision=source_revision,
    )


def _historical_registration(
    model_key: str,
    model_class: type[HistoricalRouteOnlyModel] | type[HistoricalTokenOnlyModel],
) -> ModelRegistration:
    revision = "historical-selector-regression-v1"
    return ModelRegistration(
        model_key=model_key,
        identity=ModelIdentity(
            model_key=model_key,
            display_name=model_key,
            class_fully_qualified_name=(
                f"{model_class.__module__}.{model_class.__qualname__}"
            ),
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=0,
            source_revision=revision,
        ),
        input_contract=_contract(),
        builder=model_class,
        implementation_kind=ImplementationKind.IN_REPO,
        source_revision=revision,
    )


def test_stable_key_resolves_exactly_one_identity_and_contract() -> None:
    registry = ModelRegistry()
    registration = _registration()
    registry.register(registration)

    resolved = registry.resolve("toy.stable.v1")

    assert resolved is registration
    assert resolved.identity is registration.identity
    assert resolved.input_contract is registration.input_contract
    assert registry.model_keys == ("toy.stable.v1",)
    with pytest.raises(ModelRegistryError) as duplicate:
        registry.register(_registration())
    assert duplicate.value.error_record.error_code == "MODEL_KEY_ALREADY_REGISTERED"
    assert registry.resolve("toy.stable.v1") is registration


def test_build_validates_identity_and_creates_run_metadata() -> None:
    registry = ModelRegistry()
    registry.register(_registration())

    model = registry.build("toy.stable.v1")
    metadata = registry.run_metadata(
        "toy.stable.v1",
        model,
        config_hash="c" * 64,
        supplied_contract=_contract(modalities=("spectrum", "rgb")),
    )

    assert isinstance(model, ToyModel)
    assert metadata.model_key == "toy.stable.v1"
    assert metadata.source_revision == "revision-abc"
    assert metadata.parameter_count == 10
    assert metadata.representation_key == "synthetic.rgb-spectrum.v1"
    assert metadata.preprocessing_key == "synthetic.fold-scoped.v1"
    assert metadata.config_hash == "c" * 64


def test_registration_rejects_identity_with_a_different_model_key() -> None:
    registry = ModelRegistry()
    registration = _registration(identity=_identity(model_key="toy.other.v1"))

    with pytest.raises(ModelRegistryError) as caught:
        registry.register(registration)

    error = caught.value.error_record
    assert error.error_code == "MODEL_KEY_IDENTITY_MISMATCH"
    assert error.failing_field == "model_key"
    assert error.expected_value == "toy.stable.v1"
    assert error.observed_value == "toy.other.v1"
    assert registry.snapshot() == ()


def test_build_rejects_instance_with_a_different_model_key() -> None:
    class WrongKeyModel(ToyModel):
        __benchmark_model_key__ = "toy.other.v1"

    registry = ModelRegistry()
    registration = _registration(
        identity=_identity(
            class_fully_qualified_name=(
                f"{WrongKeyModel.__module__}.{WrongKeyModel.__qualname__}"
            )
        ),
        builder=WrongKeyModel,
    )
    registry.register(registration)
    before = registry.snapshot()

    with pytest.raises(ModelRegistryError) as caught:
        registry.build("toy.stable.v1")

    error = caught.value.error_record
    assert error.error_code == "MODEL_KEY_INSTANCE_MISMATCH"
    assert error.failing_field == "model_key"
    assert error.expected_value == "toy.stable.v1"
    assert error.observed_value == "toy.other.v1"
    assert registry.snapshot() == before


@pytest.mark.parametrize(
    ("supplied_contract", "error_code", "field"),
    (
        (
            _contract(representation_key="synthetic.other-representation.v1"),
            "INPUT_CONTRACT_REPRESENTATION_KEY_MISMATCH",
            "representation_key",
        ),
        (
            _contract(
                modalities=("rgb",),
                tensor_shapes={"rgb": (3, 16, 16)},
                dtypes={"rgb": "float32"},
                channel_or_band_order={"rgb": ("red", "green", "blue")},
            ),
            "INPUT_CONTRACT_MODALITIES_MISMATCH",
            "modalities",
        ),
        (
            _contract(tensor_shapes={"rgb": (3, 32, 32), "spectrum": (128,)}),
            "INPUT_CONTRACT_TENSOR_SHAPES_MISMATCH",
            "tensor_shapes",
        ),
        (
            _contract(dtypes={"rgb": "float16", "spectrum": "float32"}),
            "INPUT_CONTRACT_DTYPES_MISMATCH",
            "dtypes",
        ),
        (
            _contract(
                channel_or_band_order={
                    "rgb": ("blue", "green", "red"),
                    "spectrum": ("wavelength_ascending",),
                }
            ),
            "INPUT_CONTRACT_CHANNEL_OR_BAND_ORDER_MISMATCH",
            "channel_or_band_order",
        ),
        (
            _contract(preprocessing_key="synthetic.global-fit.v1"),
            "INPUT_CONTRACT_PREPROCESSING_KEY_MISMATCH",
            "preprocessing_key",
        ),
    ),
)
def test_run_start_blocks_each_input_contract_mismatch_before_training(
    supplied_contract: InputContract,
    error_code: str,
    field: str,
) -> None:
    registry = ModelRegistry()
    registry.register(_registration())
    model = registry.build("toy.stable.v1")
    before = registry.snapshot()
    training_started = False

    with pytest.raises(ModelRegistryError) as caught:
        registry.run_metadata(
            "toy.stable.v1",
            model,
            config_hash="c" * 64,
            supplied_contract=supplied_contract,
        )
        training_started = True

    error = caught.value.error_record
    assert error.error_code == error_code
    assert error.affected_identifiers == {"model_key": "toy.stable.v1"}
    assert error.failing_field == field
    assert error.expected_value is not None
    assert error.observed_value is not None
    assert error.invariant_checks == {
        "training_not_started": True,
        "checkpoint_unchanged": True,
        "prediction_artifact_unchanged": True,
        "completed_run_records_unchanged": True,
    }
    assert not training_started
    assert registry.snapshot() == before


def test_input_contract_validation_reports_every_mismatched_field() -> None:
    supplied = InputContract(
        representation_key="synthetic.other.v1",
        modalities=("cube",),
        tensor_shapes={"cube": (8, 8, 64)},
        dtypes={"cube": "float64"},
        channel_or_band_order={"cube": ("band_descending",)},
        preprocessing_key="synthetic.other-preprocessing.v1",
    )

    report = ModelRegistry.validate_input_contract(_contract(), supplied)

    assert not report.passes
    assert tuple(issue.field_path[-1] for issue in report.errors) == (
        "representation_key",
        "modalities",
        "tensor_shapes",
        "dtypes",
        "channel_or_band_order",
        "preprocessing_key",
    )


def test_input_contract_failure_is_appended_to_error_journal(tmp_path: Path) -> None:
    journal = ErrorJournal(tmp_path / "input contract errors.jsonl")
    registry = ModelRegistry(journal)
    registry.register(_registration())
    model = registry.build("toy.stable.v1")

    with pytest.raises(ModelRegistryError):
        registry.run_metadata(
            "toy.stable.v1",
            model,
            config_hash="c" * 64,
            supplied_contract=_contract(
                dtypes={"rgb": "float16", "spectrum": "float32"}
            ),
        )

    record = json.loads(journal.path.read_text(encoding="utf-8").strip())
    assert record["error_code"] == "INPUT_CONTRACT_DTYPES_MISMATCH"
    assert record["failed_stage"] == "model_validation"
    assert record["failing_field"] == "dtypes"
    assert record["affected_identifiers"] == {"model_key": "toy.stable.v1"}


@pytest.mark.parametrize(
    ("registration", "error_code", "field"),
    (
        (
            _registration(
                identity=_identity(
                    class_fully_qualified_name=(
                        f"{DifferentModel.__module__}.{DifferentModel.__qualname__}"
                    )
                )
            ),
            "MODEL_CLASS_MISMATCH",
            "class_fully_qualified_name",
        ),
        (
            _registration(
                identity=_identity(implementation_kind=ImplementationKind.CLASSIC)
            ),
            "MODEL_IMPLEMENTATION_KIND_MISMATCH",
            "implementation_kind",
        ),
        (
            _registration(implementation_kind=ImplementationKind.PROXY),
            "MODEL_IMPLEMENTATION_KIND_MISMATCH",
            "implementation_kind",
        ),
        (
            _registration(identity=_identity(parameter_count=11)),
            "MODEL_PARAMETER_COUNT_MISMATCH",
            "parameter_count",
        ),
        (
            _registration(identity=_identity(source_revision="revision-other")),
            "MODEL_SOURCE_REVISION_MISMATCH",
            "source_revision",
        ),
        (
            _registration(source_revision="revision-other"),
            "MODEL_SOURCE_REVISION_MISMATCH",
            "source_revision",
        ),
    ),
)
def test_build_blocks_each_identity_mismatch_before_return(
    registration: ModelRegistration,
    error_code: str,
    field: str,
) -> None:
    registry = ModelRegistry()
    registry.register(registration)
    before = registry.snapshot()

    with pytest.raises(ModelRegistryError) as caught:
        registry.build("toy.stable.v1")

    assert caught.value.error_record.error_code == error_code
    assert caught.value.error_record.failing_field == field
    assert caught.value.error_record.invariant_checks == {
        "model_not_returned": True,
        "training_not_started": True,
        "completed_artifacts_unchanged": True,
    }
    assert registry.snapshot() == before


def test_unknown_or_non_string_selector_is_rejected_and_journaled(
    tmp_path: Path,
) -> None:
    journal = ErrorJournal(tmp_path / "errors.jsonl")
    registry = ModelRegistry(journal)
    registry.register(_registration())

    with pytest.raises(ModelRegistryError) as unknown:
        registry.resolve("missing.model.v1")
    with pytest.raises(ModelRegistryError) as positional:
        registry.resolve(0)  # type: ignore[arg-type]

    assert unknown.value.error_record.failing_field == "model_key"
    assert positional.value.error_record.error_code == "MODEL_SELECTOR_NOT_STABLE_KEY"
    records = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [record["error_code"] for record in records] == [
        "MODEL_KEY_NOT_REGISTERED",
        "MODEL_SELECTOR_NOT_STABLE_KEY",
    ]


def test_historical_route_and_token_selectors_cannot_drift_with_list_order() -> None:
    registrations = (
        _historical_registration("leaf.route-only.v1", HistoricalRouteOnlyModel),
        _historical_registration("leaf.token-only.v1", HistoricalTokenOnlyModel),
    )

    for registration_order in (registrations, tuple(reversed(registrations))):
        registry = ModelRegistry()
        for registration in registration_order:
            registry.register(registration)

        route_model = registry.build("leaf.route-only.v1")
        token_model = registry.build("leaf.token-only.v1")

        assert type(route_model) is HistoricalRouteOnlyModel
        assert type(token_model) is HistoricalTokenOnlyModel
        assert registry.resolve("leaf.route-only.v1").identity.model_key == (
            "leaf.route-only.v1"
        )
        assert registry.resolve("leaf.token-only.v1").identity.model_key == (
            "leaf.token-only.v1"
        )


def test_cli_resolves_only_an_exact_stable_model_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = ModelRegistry()
    registry.register(_registration())

    assert (
        main(
            ["model-resolve", "--model-key", "toy.stable.v1"],
            model_registry=registry,
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["model_key"] == "toy.stable.v1"
    assert output["model_identity"]["class_fully_qualified_name"].endswith("ToyModel")
    assert output["input_contract"]["representation_key"] == (
        "synthetic.rgb-spectrum.v1"
    )

    with pytest.raises(SystemExit) as positional:
        main(["model-resolve", "--model-key", "0"], model_registry=registry)
    assert positional.value.code == 2

    with pytest.raises(SystemExit) as list_selector:
        main(
            ["model-resolve", "--model-key", "toy.stable.v1,other.model"],
            model_registry=registry,
        )
    assert list_selector.value.code == 2
