"""Task 14.4 coverage for HyperLeaf classic matched-input controls.

Validates Requirements 7.6-7.14, 8.9-8.13, 10.12-10.17.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from benchmark.adapters import (
    HyperLeafClassicError,
    HyperLeafClassicKind,
    HyperLeafClassicRunner,
    MatchedRunContext,
    classic_model_key,
    register_hyperleaf_classic_controls,
)
from benchmark.catalog import GLOBAL_SPECTRUM_KEY, HyperLeaf2024Adapter, HyperLeafRepresentationBuilder
from benchmark.engine import OuterEvaluationGate
from benchmark.metrics import PredictionArtifactWriter
from benchmark.models import ModelRegistry
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    ArtifactReference,
    EvaluationBudget,
    FitBoundary,
    FitBoundaryScope,
    FrozenOuterCheckpoint,
    InnerFold,
    LimitEnforcement,
    MetricDirection,
    OuterFold,
    ProtocolManifest,
    ResourceBudget,
    ResourceLimit,
    SelectionCandidate,
    SelectionDecision,
    SelectionPolicy,
)
from tests.unit.test_hyperleaf_adapter import TEST_IDS, TRAIN_IDS, leaf_root


_FROZEN_AT = datetime(2027, 1, 2, 3, 4, tzinfo=timezone.utc)


def _fold() -> OuterFold:
    return OuterFold(
        fold_id="outer-1",
        outer_train_ids=TRAIN_IDS,
        outer_test_ids=TEST_IDS,
        inner_folds=(
            InnerFold(
                fold_id="inner-1",
                train_ids=TRAIN_IDS[:3],
                validation_ids=TRAIN_IDS[3:],
            ),
        ),
    )


def _protocol() -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1.0",
        protocol_key="leaf.official.synthetic.classic.v1",
        dataset_hash="a" * 64,
        folds=(_fold(),),
        selection_policy=SelectionPolicy(
            policy_key="inner-mean-r2-v1",
            objective_metric="mean-r2",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule="candidate_id_utf8_ascending",
        ),
        seed_set=(7, 42, 123),
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=1),
    )


def _budget(step_limit: int = 1000) -> ResourceBudget:
    return ResourceBudget(
        limits=(
            ResourceLimit(
                field="completed_training_steps",
                value=step_limit,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )


def _context(protocol: ProtocolManifest, *, budget: ResourceBudget | None = None) -> MatchedRunContext:
    return MatchedRunContext.from_protocol(
        protocol,
        metric_configuration={
            "regression": ("r2_per_target", "mean_r2", "mae"),
            "numeric_tolerance": 1e-12,
        },
        resource_budget=budget or _budget(),
        representation_key=GLOBAL_SPECTRUM_KEY,
    )


def _install_non_degenerate_spectra(
    builder: HyperLeafRepresentationBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def spectrum(sample_id: str) -> np.ndarray:
        index = int(sample_id)
        values = np.linspace(0.0, 1.0, builder.spectrum_length, dtype=np.float32)
        values = values + (index + 1) * np.linspace(
            0.05, 0.4, builder.spectrum_length, dtype=np.float32
        ) ** ((index % 3) + 1)
        return values

    monkeypatch.setattr(builder, "global_spectrum", spectrum)


def _decision() -> SelectionDecision:
    return SelectionDecision.create(
        decision_id="classic-decision-outer-1",
        fold_id="outer-1",
        candidates=(
            SelectionCandidate(
                candidate_id="classic-global-spectrum",
                configuration={"representation_key": GLOBAL_SPECTRUM_KEY},
            ),
        ),
        candidate_scores={"classic-global-spectrum": 0.0},
        source_sample_ids=TRAIN_IDS[3:],
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
        source_prediction_hash="b" * 64,
        selected_candidate="classic-global-spectrum",
        frozen_utc=_FROZEN_AT,
    )


class _SyntheticVectorLabelVault:
    def __init__(self) -> None:
        self.unlock_calls = 0

    def unlock(self, lease, sample_ids):
        self.unlock_calls += 1
        return {
            TEST_IDS[0]: (14.0, 4.1, 4.5),
            TEST_IDS[1]: (15.0, 5.1, 5.5),
        }


def test_registers_plsr_and_rf_with_the_same_representation_contract(
    leaf_root: Path,
) -> None:
    builder = HyperLeafRepresentationBuilder(
        HyperLeaf2024Adapter(leaf_root), spectrum_length=8
    )
    registry = ModelRegistry()

    plsr, rf = register_hyperleaf_classic_controls(
        registry,
        builder,
        representation_key=GLOBAL_SPECTRUM_KEY,
    )

    assert registry.model_keys == tuple(sorted((plsr.model_key, rf.model_key)))
    assert plsr.model_key == classic_model_key(
        HyperLeafClassicKind.PLSR, GLOBAL_SPECTRUM_KEY
    )
    assert rf.model_key == classic_model_key(
        HyperLeafClassicKind.RANDOM_FOREST, GLOBAL_SPECTRUM_KEY
    )
    assert plsr.input_contract == rf.input_contract
    assert plsr.input_contract.representation_key == GLOBAL_SPECTRUM_KEY
    assert plsr.input_contract.tensor_shapes == {"global_spectrum": (8,)}


def test_runner_fits_each_classic_control_with_exact_fold_scoped_lineage(
    leaf_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    context = _context(protocol)
    builder = HyperLeafRepresentationBuilder(
        HyperLeaf2024Adapter(leaf_root), spectrum_length=8
    )
    _install_non_degenerate_spectra(builder, monkeypatch)
    registry = ModelRegistry()
    registrations = register_hyperleaf_classic_controls(
        registry, builder, representation_key=GLOBAL_SPECTRUM_KEY
    )
    runner = HyperLeafClassicRunner(
        registry=registry,
        representations=builder,
        protocol=protocol,
        context=context,
    )
    boundary = FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=TRAIN_IDS,
        outer_test_sample_ids=TEST_IDS,
    )

    fitted_models = tuple(
        runner.fit(
            registration.model_key,
            seed=42,
            boundary=boundary,
            estimator_parameters=(
                {"n_components": 2}
                if ".plsr." in registration.model_key
                else {"n_estimators": 8, "max_depth": 3}
            ),
        )
        for registration in registrations
    )

    for fitted in fitted_models:
        assert fitted.context is context
        assert fitted.seed == 42
        assert all(artifact.fit_sample_ids == TRAIN_IDS for artifact in fitted.fitted_artifacts)
        assert all(artifact.dataset_hash == protocol.dataset_hash for artifact in fitted.fitted_artifacts)
        assert all(artifact.protocol_hash == protocol.protocol_hash for artifact in fitted.fitted_artifacts)
        prediction = runner.predict(fitted, TEST_IDS[0])
        assert isinstance(prediction.y_pred, tuple)
        assert len(prediction.y_pred) == 3


def test_runner_delegates_label_access_and_unified_prediction_publication_to_gate(
    leaf_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    context = _context(protocol)
    builder = HyperLeafRepresentationBuilder(
        HyperLeaf2024Adapter(leaf_root), spectrum_length=8
    )
    _install_non_degenerate_spectra(builder, monkeypatch)
    registry = ModelRegistry()
    plsr, _ = register_hyperleaf_classic_controls(
        registry, builder, representation_key=GLOBAL_SPECTRUM_KEY
    )
    runner = HyperLeafClassicRunner(
        registry=registry,
        representations=builder,
        protocol=protocol,
        context=context,
    )
    boundary = FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=TRAIN_IDS,
        outer_test_sample_ids=TEST_IDS,
    )
    fitted = runner.fit(
        plsr.model_key,
        seed=42,
        boundary=boundary,
        estimator_parameters={"n_components": 2},
    )
    decision = _decision()
    checkpoint = FrozenOuterCheckpoint(
        artifact=ArtifactReference(
            artifact_id="classic-checkpoint-outer-1",
            relative_path="runs_v2/checkpoints/classic-outer-1.canonical.json",
            content_hash="c" * 64,
        ),
        fold_id="outer-1",
        seed=42,
        inference_branch="primary",
        selection_decision_hash=decision.decision_hash,
        frozen_utc=_FROZEN_AT,
    )
    vault = _SyntheticVectorLabelVault()
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    gate = OuterEvaluationGate(
        protocol,
        vault,
        PredictionArtifactWriter(store),
    )

    published = runner.evaluate(
        fitted,
        gate=gate,
        outer_fold=_fold(),
        checkpoint=checkpoint,
        selection_decision=decision,
        artifact_id="classic-prediction-outer-1",
        run_id="classic-run-outer-1",
        relative_path="predictions/classic-outer-1.canonical.json",
    )

    assert vault.unlock_calls == 1
    assert published.artifact.complete is True
    assert published.artifact.expected_sample_ids == TEST_IDS
    assert all(record.key.seed == 42 for record in published.artifact.records)
    assert all(record.key.fold_id == "outer-1" for record in published.artifact.records)
    assert all(len(record.y_pred) == 3 for record in published.artifact.records)
    assert published.reference.relative_path == "predictions/classic-outer-1.canonical.json"


def test_matched_context_rejects_seed_metric_budget_or_representation_drift() -> None:
    protocol = _protocol()
    candidate = _context(protocol)

    for field, value in (
        ("dataset_hash", "e" * 64),
        ("protocol_hash", "f" * 64),
        ("fold_manifest_hashes", (("outer-1", "0" * 64),)),
        ("seed_set", (42,)),
        ("metric_configuration_hash", "d" * 64),
        ("resource_budget", _budget(999)),
        ("representation_key", "hyperleaf.visible-rgb.v1"),
    ):
        control = MatchedRunContext(
            **{
                **candidate.__dict__,
                field: value,
            }
        )
        with pytest.raises(HyperLeafClassicError, match=field):
            candidate.require_matched(control)


def test_runner_rejects_unfrozen_seed_and_fold_membership_before_data_access(
    leaf_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    context = _context(protocol)
    builder = HyperLeafRepresentationBuilder(
        HyperLeaf2024Adapter(leaf_root), spectrum_length=8
    )
    registry = ModelRegistry()
    plsr, _ = register_hyperleaf_classic_controls(
        registry, builder, representation_key=GLOBAL_SPECTRUM_KEY
    )
    runner = HyperLeafClassicRunner(
        registry=registry,
        representations=builder,
        protocol=protocol,
        context=context,
    )

    monkeypatch.setattr(
        builder,
        "global_spectrum",
        lambda sample_id: pytest.fail(f"representation read for {sample_id}"),
    )
    valid_boundary = FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=TRAIN_IDS,
        outer_test_sample_ids=TEST_IDS,
    )
    with pytest.raises(HyperLeafClassicError, match="Seed_Set"):
        runner.fit(plsr.model_key, seed=999, boundary=valid_boundary)

    wrong_boundary = FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=TRAIN_IDS[:-1],
        outer_test_sample_ids=TEST_IDS,
    )
    with pytest.raises(HyperLeafClassicError, match="fold manifest"):
        runner.fit(plsr.model_key, seed=42, boundary=wrong_boundary)
