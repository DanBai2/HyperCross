"""Unit coverage for task 5.4 fold-scoped preprocessing behavior."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.schema import (
    FitBoundary,
    FitBoundaryScope,
    FitDataRole,
    FittedArtifactKind,
)
from benchmark.transforms import PreprocessingManager, PreprocessingValidationError


DATASET_HASH = "a" * 64
PROTOCOL_HASH = "b" * 64


def _outer_boundary(*sample_ids: str) -> FitBoundary:
    return FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=sample_ids,
        outer_test_sample_ids=("held-out-1",),
    )


def _inner_boundary(*sample_ids: str) -> FitBoundary:
    return FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.INNER_TRAIN,
        inner_fold_id="inner-1",
        allowed_fit_sample_ids=sample_ids,
        outer_test_sample_ids=("held-out-1",),
    )


def _identity() -> dict[str, str]:
    return {"dataset_hash": DATASET_HASH, "protocol_hash": PROTOCOL_HASH}


def test_target_normalization_uses_exact_permitted_training_ids() -> None:
    """Requirements 4.1, 4.14-4.16: target statistics use all and only training IDs."""

    manager = PreprocessingManager()
    boundary = _outer_boundary("train-b", "train-a")

    artifact = manager.fit_target_normalization(
        np.array([[1.0, 10.0], [3.0, 14.0]]),
        ("train-a", "train-b"),
        boundary,
        **_identity(),
    )
    transformed = manager.apply(
        artifact,
        np.array([[1.0, 10.0], [3.0, 14.0]]),
        boundary,
    )

    assert artifact.fit_sample_ids == ("train-a", "train-b")
    assert artifact.kind is FittedArtifactKind.TARGET_NORMALIZATION
    assert artifact.state["mean"] == (2.0, 12.0)
    assert artifact.state["scale"] == (1.0, 2.0)
    np.testing.assert_allclose(transformed, [[-1.0, -1.0], [1.0, 1.0]])


def test_identical_inputs_and_config_produce_deterministic_state_hash() -> None:
    """Requirement 4.29: repeated deterministic fits have identical state identity."""

    manager = PreprocessingManager()
    boundary = _inner_boundary("s1", "s2", "s3")
    values = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 8.0]])

    first = manager.fit_pca(
        values,
        ("s1", "s2", "s3"),
        boundary,
        n_components=2,
        whiten=True,
        **_identity(),
    )
    second = manager.fit_pca(
        values.copy(),
        ("s1", "s2", "s3"),
        boundary,
        n_components=2,
        whiten=True,
        **_identity(),
    )

    assert first.fit_sample_ids == second.fit_sample_ids == ("s1", "s2", "s3")
    assert first.config_hash == second.config_hash
    assert first.input_hash == second.input_hash
    assert first.state_hash == second.state_hash
    assert first.artifact_id == second.artifact_id
    assert first.state == second.state


def test_pca_band_normalization_and_imputation_fit_apply() -> None:
    """Requirements 4.2-4.4: fitted numeric transforms preserve exact lineage."""

    manager = PreprocessingManager()
    boundary = _inner_boundary("s1", "s2", "s3")
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])

    pca = manager.fit_pca(
        values,
        ("s1", "s2", "s3"),
        boundary,
        n_components=1,
        **_identity(),
    )
    projected = manager.apply(pca, values, boundary)
    assert projected.shape == (3, 1)
    np.testing.assert_allclose(projected.mean(axis=0), [0.0], atol=1e-12)

    bands = manager.fit_band_normalization(
        values,
        ("s1", "s2", "s3"),
        boundary,
        **_identity(),
    )
    normalized = manager.apply(bands, values, boundary)
    np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(normalized.std(axis=0), [1.0, 1.0], atol=1e-12)

    missing = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]])
    imputer = manager.fit_imputation(
        missing,
        ("s1", "s2", "s3"),
        boundary,
        **_identity(),
    )
    imputed = manager.apply(imputer, [[np.nan, np.nan]], boundary)
    np.testing.assert_allclose(imputed, [[3.0, 6.0]])

    assert pca.fit_sample_ids == bands.fit_sample_ids == imputer.fit_sample_ids
    assert pca.inner_fold_id == "inner-1"


def test_class_weights_threshold_and_calibration_fit_apply() -> None:
    """Requirements 4.5-4.7, 9.33-9.34: label-derived state has guarded sources."""

    manager = PreprocessingManager()
    training_boundary = _inner_boundary("s1", "s2", "s3", "s4")
    oof_boundary = _outer_boundary("s1", "s2", "s3", "s4")
    sample_ids = ("s1", "s2", "s3", "s4")

    weights = manager.fit_class_weights(
        [0, 0, 0, 1], sample_ids, training_boundary, **_identity()
    )
    np.testing.assert_allclose(
        manager.apply(weights, [0, 1], training_boundary), [2.0 / 3.0, 2.0]
    )

    threshold = manager.fit_threshold(
        [0.1, 0.4, 0.6, 0.9],
        [0, 0, 1, 1],
        sample_ids,
        oof_boundary,
        **_identity(),
    )
    assert threshold.fit_data_role is FitDataRole.INNER_OOF_PREDICTIONS
    assert threshold.state["threshold"] == pytest.approx(0.5)
    np.testing.assert_array_equal(
        manager.apply(threshold, [0.49, 0.5], oof_boundary), [0, 1]
    )

    calibration = manager.fit_calibration(
        [0.2, 0.35, 0.65, 0.8],
        [0, 0, 1, 1],
        sample_ids,
        oof_boundary,
        **_identity(),
    )
    calibrated = manager.apply(calibration, [0.25, 0.75], oof_boundary)
    assert calibration.fit_data_role is FitDataRole.INNER_OOF_PREDICTIONS
    assert calibration.state["temperature"] > 0
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_apply_rejects_boundary_mismatch_before_reading_values() -> None:
    """Requirements 4.8-4.13: leakage is rejected before artifact application."""

    manager = PreprocessingManager()
    fitted_boundary = _inner_boundary("s1", "s2")
    artifact = manager.fit_band_normalization(
        [[1.0], [2.0]], ("s1", "s2"), fitted_boundary, **_identity()
    )
    narrower_boundary = _inner_boundary("s1")

    class ExplodingArray:
        touched = False

        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
            self.touched = True
            raise AssertionError("values were read before boundary validation")

    values = ExplodingArray()
    with pytest.raises(PreprocessingValidationError) as captured:
        manager.apply(artifact, values, narrower_boundary)

    assert values.touched is False
    assert captured.value.error_record.error_code == "FIT_SAMPLE_IDS_OUTSIDE_BOUNDARY"
    assert captured.value.error_record.affected_identifiers["fold_id"] == "outer-1"
    assert captured.value.error_record.unexpected_ids == ("s2",)
    assert captured.value.claim_status.value == "invalid_or_leaky"


def test_fit_rejects_forbidden_ids_and_incomplete_target_set_before_values() -> None:
    """Requirements 4.9-4.16: fit membership is checked before target access."""

    manager = PreprocessingManager()
    boundary = _outer_boundary("s1", "s2")

    class ExplodingArray:
        touched = False

        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
            self.touched = True
            raise AssertionError("targets were read before fit-ID validation")

    forbidden_values = ExplodingArray()
    with pytest.raises(PreprocessingValidationError) as forbidden:
        manager.fit_target_normalization(
            forbidden_values,
            ("s1", "held-out-1"),
            boundary,
            **_identity(),
        )
    assert forbidden_values.touched is False
    assert forbidden.value.error_record.error_code == "FIT_SAMPLE_IDS_OUTSIDE_BOUNDARY"

    incomplete_values = ExplodingArray()
    with pytest.raises(PreprocessingValidationError) as incomplete:
        manager.fit_target_normalization(
            incomplete_values,
            ("s1",),
            boundary,
            **_identity(),
        )
    assert incomplete_values.touched is False
    assert incomplete.value.error_record.error_code == (
        "TARGET_NORMALIZATION_INCOMPLETE_TRAINING_SET"
    )
    assert incomplete.value.error_record.missing_ids == ("s2",)


def test_threshold_and_calibration_reject_non_oof_fit_role() -> None:
    """Requirements 9.33-9.34: threshold and calibration require inner OOF data."""

    manager = PreprocessingManager()
    boundary = _outer_boundary("s1", "s2")

    for kind in (FittedArtifactKind.THRESHOLD, FittedArtifactKind.CALIBRATION):
        with pytest.raises(PreprocessingValidationError) as captured:
            manager.fit(
                kind,
                ([0.2, 0.8], [0, 1]),
                ("s1", "s2"),
                boundary,
                fit_data_role=FitDataRole.TRAINING_VALUES,
                **_identity(),
            )
        assert captured.value.error_record.error_code == "INNER_OOF_EVIDENCE_REQUIRED"
