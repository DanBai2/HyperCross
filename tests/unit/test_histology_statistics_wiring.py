"""Unit coverage for Histology threshold/calibration and patient-statistics wiring."""

from __future__ import annotations

import pytest

from benchmark.runners import (
    HISTOLOGY_FITTED_THRESHOLD_RESULT_LABEL,
    HISTOLOGY_PRIMARY_RESULT_LABEL,
    HISTOLOGY_TEMPERATURE_RESULT_LABEL,
    HistologyStatisticsWiring,
    HistologyStatisticsWiringError,
)
from benchmark.schema import (
    ConfidenceInterval,
    FitDataRole,
    OuterFold,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    StatisticalComparison,
)


DATASET_HASH = "a" * 64
PROTOCOL_HASH = "b" * 64


def _fold() -> OuterFold:
    return OuterFold(
        fold_id="outer-1",
        outer_train_ids=("train-1", "train-2", "train-3", "train-4"),
        outer_test_ids=("test-1", "test-2", "test-3", "test-4"),
    )


def _artifact(
    artifact_id: str,
    rows: tuple[tuple[str, int, float], ...],
    *,
    fold_id: str,
    branch: str,
) -> PredictionArtifact:
    records = tuple(
        PredictionRecord(
            key=PredictionKey(
                sample_id=sample_id,
                fold_id=fold_id,
                seed=42,
                inference_branch=branch,
            ),
            y_true=label,
            y_pred=int(probability >= 0.5),
            probability=probability,
        )
        for sample_id, label, probability in rows
    )
    return PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=f"run-{artifact_id}",
        records=records,
        expected_sample_ids=tuple(row[0] for row in rows),
        complete=True,
    )


def _inner_oof() -> PredictionArtifact:
    return _artifact(
        "inner-oof",
        (
            ("train-1", 0, 0.1),
            ("train-2", 0, 0.4),
            ("train-3", 1, 0.6),
            ("train-4", 1, 0.9),
        ),
        fold_id="outer-1-inner-oof",
        branch="inner-oof",
    )


def _outer(artifact_id: str, probabilities: tuple[float, ...]) -> PredictionArtifact:
    return _artifact(
        artifact_id,
        tuple(
            (sample_id, label, probability)
            for sample_id, label, probability in zip(
                ("test-1", "test-2", "test-3", "test-4"),
                (0, 1, 0, 1),
                probabilities,
            )
        ),
        fold_id="outer-1",
        branch="pca-only",
    )


def test_primary_is_fixed_and_inner_oof_secondary_results_are_separately_labeled() -> None:
    """Validates: Requirements 9.33, 9.34, 9.35, 9.36."""

    wiring = HistologyStatisticsWiring()
    fitted = wiring.fit_inner_oof(
        _fold(),
        _inner_oof(),
        dataset_hash=DATASET_HASH,
        protocol_hash=PROTOCOL_HASH,
    )
    variants = wiring.evaluate_variants(
        _fold(),
        _outer("candidate", (0.49, 0.51, 0.2, 0.8)),
        fitted,
    )

    assert fitted.threshold.fit_data_role is FitDataRole.INNER_OOF_PREDICTIONS
    assert fitted.temperature.fit_data_role is FitDataRole.INNER_OOF_PREDICTIONS
    assert set(fitted.threshold.fit_sample_ids) == set(_fold().outer_train_ids)
    assert set(fitted.temperature.fit_sample_ids) == set(_fold().outer_train_ids)
    assert variants.primary.result_label == HISTOLOGY_PRIMARY_RESULT_LABEL
    assert variants.primary.metric_schema.decision_threshold == 0.5
    assert variants.primary.primary is True
    assert variants.fitted_threshold.result_label == HISTOLOGY_FITTED_THRESHOLD_RESULT_LABEL
    assert variants.fitted_threshold.primary is False
    assert variants.fitted_temperature.result_label == HISTOLOGY_TEMPERATURE_RESULT_LABEL
    assert variants.fitted_temperature.primary is False
    assert variants.fitted_temperature.prediction_artifact.content_hash != (
        variants.primary.prediction_artifact.content_hash
    )


def test_fitting_rejects_incomplete_or_outer_patient_evidence_before_tuning() -> None:
    """Validates: Requirements 9.33, 9.34."""

    wiring = HistologyStatisticsWiring()
    contaminated = _artifact(
        "contaminated",
        (
            ("train-1", 0, 0.1),
            ("train-2", 0, 0.4),
            ("train-3", 1, 0.6),
            ("test-1", 1, 0.9),
        ),
        fold_id="inner-oof",
        branch="inner-oof",
    )

    with pytest.raises(HistologyStatisticsWiringError, match="exact complete inner OOF"):
        wiring.fit_inner_oof(
            _fold(),
            contaminated,
            dataset_hash=DATASET_HASH,
            protocol_hash=PROTOCOL_HASH,
        )


def test_patient_bootstrap_and_paired_comparison_keep_variant_label() -> None:
    """Validates: Requirements 9.27, 9.28, 9.29, 9.30, 9.31, 9.32."""

    wiring = HistologyStatisticsWiring()
    fitted = wiring.fit_inner_oof(
        _fold(),
        _inner_oof(),
        dataset_hash=DATASET_HASH,
        protocol_hash=PROTOCOL_HASH,
    )
    candidate = wiring.evaluate_variants(
        _fold(), _outer("candidate", (0.1, 0.9, 0.2, 0.8)), fitted
    ).primary
    control = wiring.evaluate_variants(
        _fold(), _outer("control", (0.6, 0.4, 0.2, 0.8)), fitted
    ).primary
    groups = {
        "test-1": "patient-a",
        "test-2": "patient-a",
        "test-3": "patient-b",
        "test-4": "patient-b",
    }

    interval = wiring.patient_bootstrap(
        candidate,
        "accuracy",
        group_by_sample_id=groups,
        seed=123,
    )
    comparison = wiring.paired_patient_compare(
        candidate,
        control,
        "accuracy",
        candidate_group_by_sample_id=groups,
        control_group_by_sample_id=groups,
        seed=123,
    )

    assert interval.result_label == HISTOLOGY_PRIMARY_RESULT_LABEL
    assert isinstance(interval.statistic, ConfidenceInterval)
    assert interval.statistic.resampling_unit == "patient_group_id"
    assert interval.statistic.resamples == 10_000
    assert interval.statistic.confidence_level == 0.95
    assert interval.statistic.sidedness == "two_sided"
    assert comparison.result_label == HISTOLOGY_PRIMARY_RESULT_LABEL
    assert isinstance(comparison.statistic, StatisticalComparison)
    assert comparison.statistic.common_group_ids == ("patient-a", "patient-b")
    assert comparison.statistic.resamples == 10_000
