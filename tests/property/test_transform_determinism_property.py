"""Property 10: transform determinism.

**Validates: Requirements 4.29**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import given, settings, strategies as st

from benchmark.schema import FitBoundary, FitBoundaryScope, hash_object
from benchmark.transforms import PreprocessingManager


_DATASET_HASH = "a" * 64
_PROTOCOL_HASH = "b" * 64


@dataclass(frozen=True)
class _TransformCase:
    seed: int
    sample_count: int
    feature_count: int
    component_count: int
    whiten: bool
    epsilon: float
    imputation_strategy: str
    fill_value: float
    threshold_objective: str
    calibration_input_kind: str
    calibration_iterations: int


@st.composite
def _transform_cases(draw: st.DrawFn) -> _TransformCase:
    sample_count = draw(st.integers(min_value=4, max_value=10))
    feature_count = draw(st.integers(min_value=1, max_value=6))
    return _TransformCase(
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        sample_count=sample_count,
        feature_count=feature_count,
        component_count=draw(
            st.integers(min_value=1, max_value=min(sample_count, feature_count))
        ),
        whiten=draw(st.booleans()),
        epsilon=draw(st.sampled_from((1e-12, 1e-9, 1e-6))),
        imputation_strategy=draw(st.sampled_from(("mean", "median", "constant"))),
        fill_value=draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            )
        ),
        threshold_objective=draw(
            st.sampled_from(("accuracy", "balanced_accuracy"))
        ),
        calibration_input_kind=draw(st.sampled_from(("probability", "logit"))),
        calibration_iterations=draw(st.integers(min_value=8, max_value=24)),
    )


def _inputs(case: _TransformCase) -> dict[str, np.ndarray]:
    """Derive all transform inputs solely from the generated seed and shape."""

    generator = np.random.default_rng(case.seed)
    features = generator.normal(
        loc=0.0,
        scale=3.0,
        size=(case.sample_count, case.feature_count),
    )
    missing_features = features.copy()
    missing_features[0, 0] = np.nan

    labels = np.arange(case.sample_count, dtype=np.int64) % 2
    generator.shuffle(labels)
    logits = generator.normal(size=case.sample_count)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    class_labels = generator.integers(0, 3, size=case.sample_count, dtype=np.int64)

    return {
        "features": features,
        "missing_features": missing_features,
        "binary_labels": labels,
        "logits": logits,
        "probabilities": probabilities,
        "class_labels": class_labels,
    }


def _fit_all(case: _TransformCase) -> dict[str, object]:
    """Fit every learned preprocessing kind with one frozen case."""

    np.random.seed(case.seed)
    inputs = _inputs(case)
    sample_ids = tuple(f"sample-{index:02d}" for index in range(case.sample_count))
    boundary = FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=sample_ids,
        outer_test_sample_ids=("held-out",),
    )
    identity = {
        "dataset_hash": _DATASET_HASH,
        "protocol_hash": _PROTOCOL_HASH,
    }
    manager = PreprocessingManager()

    calibration_scores = (
        inputs["probabilities"]
        if case.calibration_input_kind == "probability"
        else inputs["logits"]
    )
    return {
        "target_normalization": manager.fit_target_normalization(
            inputs["features"],
            sample_ids,
            boundary,
            epsilon=case.epsilon,
            **identity,
        ),
        "pca": manager.fit_pca(
            inputs["features"],
            sample_ids,
            boundary,
            n_components=case.component_count,
            whiten=case.whiten,
            epsilon=case.epsilon,
            **identity,
        ),
        "band_normalization": manager.fit_band_normalization(
            inputs["features"],
            sample_ids,
            boundary,
            epsilon=case.epsilon,
            **identity,
        ),
        "imputation": manager.fit_imputation(
            inputs["missing_features"],
            sample_ids,
            boundary,
            strategy=case.imputation_strategy,
            fill_value=case.fill_value,
            **identity,
        ),
        "class_weights": manager.fit_class_weights(
            inputs["class_labels"],
            sample_ids,
            boundary,
            **identity,
        ),
        "threshold": manager.fit_threshold(
            inputs["probabilities"],
            inputs["binary_labels"],
            sample_ids,
            boundary,
            objective=case.threshold_objective,
            **identity,
        ),
        "calibration": manager.fit_calibration(
            calibration_scores,
            inputs["binary_labels"],
            sample_ids,
            boundary,
            input_kind=case.calibration_input_kind,
            iterations=case.calibration_iterations,
            epsilon=1e-7,
            **identity,
        ),
    }


@settings(max_examples=100, deadline=None)
@given(case=_transform_cases())
def test_identical_inputs_config_and_seed_produce_identical_state_hashes(
    case: _TransformCase,
) -> None:
    """Every fitted transform has stable state identity for a repeated case."""

    first = _fit_all(case)
    second = _fit_all(case)

    assert first.keys() == second.keys()
    for transform_name in first:
        first_artifact = first[transform_name]
        second_artifact = second[transform_name]

        assert first_artifact.input_hash == second_artifact.input_hash
        assert first_artifact.config_hash == second_artifact.config_hash
        assert first_artifact.state == second_artifact.state
        assert first_artifact.state_hash == second_artifact.state_hash
        assert first_artifact.state_hash == hash_object(first_artifact.state)
        assert first_artifact.artifact_id == second_artifact.artifact_id
