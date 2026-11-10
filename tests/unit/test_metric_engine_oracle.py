"""Independent hand-calculated metric oracles for requirements 16.21-16.27."""

from __future__ import annotations

import pytest

from benchmark.metrics import binary_metrics, regression_metrics, remote_metrics
from benchmark.schema import (
    MetricSchema,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    TaskKind,
    UNDEFINED_SINGLE_CLASS_AUROC,
    UNDEFINED_ZERO_VARIANCE_R2,
)

ABS_TOLERANCE = 1e-12


def _artifact(
    artifact_id: str,
    truths: list[object],
    predictions: list[object],
    probabilities: list[object | None] | None = None,
) -> PredictionArtifact:
    """Build a complete prediction artifact without using metric implementation helpers."""

    supplied_probabilities = probabilities or [None] * len(truths)
    records = tuple(
        PredictionRecord(
            key=PredictionKey(
                sample_id=f"oracle-sample-{index}",
                fold_id="oracle-fold",
                seed=101,
                inference_branch="primary",
            ),
            y_true=truth,
            y_pred=prediction,
            probability=probability,
        )
        for index, (truth, prediction, probability) in enumerate(
            zip(truths, predictions, supplied_probabilities)
        )
    )
    return PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=f"run-{artifact_id}",
        records=records,
        expected_sample_ids=tuple(record.key.sample_id for record in records),
        complete=True,
    )


def test_regression_metrics_match_hand_calculated_oracle() -> None:
    """R² uses SSE/TSS per target and MAE uses absolute residuals. Req. 16.21."""

    artifact = _artifact(
        "regression-oracle",
        truths=[(0.0, -2.0), (1.0, 0.0), (2.0, 2.0), (3.0, 4.0)],
        predictions=[(0.0, -1.0), (2.0, 0.0), (1.0, 1.0), (3.0, 5.0)],
    )
    schema = MetricSchema(
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        numeric_tolerance=ABS_TOLERANCE,
        target_names=("first", "second"),
    )

    result = regression_metrics(artifact, schema)

    # first: TSS=5, SSE=2, absolute-error sum=2.
    assert result.metric("r2.first").value == pytest.approx(3.0 / 5.0, abs=ABS_TOLERANCE)
    assert result.metric("mae.first").value == pytest.approx(1.0 / 2.0, abs=ABS_TOLERANCE)
    # second: TSS=20, SSE=3, absolute-error sum=3.
    assert result.metric("r2.second").value == pytest.approx(17.0 / 20.0, abs=ABS_TOLERANCE)
    assert result.metric("mae.second").value == pytest.approx(3.0 / 4.0, abs=ABS_TOLERANCE)
    assert result.metric("mean_r2").value == pytest.approx(29.0 / 40.0, abs=ABS_TOLERANCE)
    assert result.metric("mean_mae").value == pytest.approx(5.0 / 8.0, abs=ABS_TOLERANCE)


def test_classification_metrics_match_hand_calculated_oracle() -> None:
    """Confusion-derived metrics and AUROC match a manual fixture. Req. 16.22."""

    artifact = _artifact(
        "classification-oracle",
        truths=[0, 0, 0, 1, 1, 1],
        predictions=[0, 0, 1, 0, 1, 1],
        probabilities=[0.1, 0.4, 0.6, 0.3, 0.7, 0.9],
    )
    schema = MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=ABS_TOLERANCE,
        decision_threshold=0.5,
        ece_bins=2,
    )

    result = binary_metrics(artifact, schema)

    # Matrix is ((TN=2, FP=1), (FN=1, TP=2)); seven of nine
    # positive/negative probability pairs are correctly ordered.
    assert result.confusion_matrices["confusion_matrix"] == ((2, 1), (1, 2))
    assert result.metric("accuracy").value == pytest.approx(2.0 / 3.0, abs=ABS_TOLERANCE)
    assert result.metric("balanced_accuracy").value == pytest.approx(2.0 / 3.0, abs=ABS_TOLERANCE)
    assert result.metric("sensitivity").value == pytest.approx(2.0 / 3.0, abs=ABS_TOLERANCE)
    assert result.metric("specificity").value == pytest.approx(2.0 / 3.0, abs=ABS_TOLERANCE)
    assert result.metric("auroc").value == pytest.approx(7.0 / 9.0, abs=ABS_TOLERANCE)
    assert result.metric("ece").value == pytest.approx(1.0 / 15.0, abs=ABS_TOLERANCE)


def test_remote_metrics_match_hand_calculated_confusion_oracle() -> None:
    """OA, macro class recall, and kappa use a manual matrix. Req. 16.23."""

    artifact = _artifact(
        "remote-oracle",
        truths=["a", "a", "a", "b", "b", "c", "c"],
        predictions=["a", "a", "b", "b", "c", "a", "c"],
    )
    schema = MetricSchema(
        task_kind=TaskKind.REMOTE_CLASSIFICATION,
        numeric_tolerance=ABS_TOLERANCE,
        class_labels=("a", "b", "c"),
    )

    result = remote_metrics(artifact, schema)

    # Confusion matrix [[2,1,0],[0,1,1],[1,0,1]] gives expected agreement 17/49.
    assert result.metric("oa").value == pytest.approx(4.0 / 7.0, abs=ABS_TOLERANCE)
    assert result.metric("aa").value == pytest.approx(5.0 / 9.0, abs=ABS_TOLERANCE)
    assert result.metric("kappa").value == pytest.approx(11.0 / 32.0, abs=ABS_TOLERANCE)


def test_auroc_midranks_match_hand_calculated_tie_oracle() -> None:
    """A tied score block receives its average rank. Req. 16.24."""

    artifact = _artifact(
        "auroc-tie-oracle",
        truths=[0, 1, 0, 1],
        predictions=[0, 0, 0, 1],
        probabilities=[0.2, 0.2, 0.2, 0.8],
    )
    schema = MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=ABS_TOLERANCE,
    )

    result = binary_metrics(artifact, schema)

    # The first three observations share rank 2. Positive rank sum is 2+4=6;
    # U=(6 - 2*3/2)=3 and AUROC=U/(2*2)=3/4.
    assert result.metric("auroc").value == pytest.approx(3.0 / 4.0, abs=ABS_TOLERANCE)


def test_ece_assigns_exact_boundaries_to_upper_bins_and_one_to_last_bin() -> None:
    """ECE bins are [lower, upper), while probability 1 is in the last. Req. 16.25."""

    artifact = _artifact(
        "ece-boundary-oracle",
        truths=[0, 1, 0, 1, 1],
        predictions=[0, 0, 1, 1, 1],
        probabilities=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    schema = MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=ABS_TOLERANCE,
        decision_threshold=0.5,
        ece_bins=4,
    )

    result = binary_metrics(artifact, schema)

    # Weighted gaps: 0/5 + 0.75/5 + 0.5/5 + 2*(1-0.875)/5 = 0.3.
    assert result.metric("ece").value == pytest.approx(3.0 / 10.0, abs=ABS_TOLERANCE)


def test_zero_variance_r2_has_required_null_and_reason() -> None:
    """A constant target has no defined R². Req. 16.26."""

    artifact = _artifact(
        "zero-variance-oracle",
        truths=[4.0, 4.0, 4.0],
        predictions=[4.0, 5.0, 3.0],
    )
    schema = MetricSchema(
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        numeric_tolerance=ABS_TOLERANCE,
        target_names=("constant",),
    )

    result = regression_metrics(artifact, schema)

    assert result.metric("r2.constant").value is None
    assert result.metric("r2.constant").reason == UNDEFINED_ZERO_VARIANCE_R2
    assert result.metric("r2.constant").reason == "zero_variance_target"


def test_single_class_auroc_has_required_null_and_reason() -> None:
    """A fold with no positive class has no defined AUROC. Req. 16.27."""

    artifact = _artifact(
        "single-class-oracle",
        truths=[0, 0, 0],
        predictions=[0, 0, 1],
        probabilities=[0.1, 0.2, 0.8],
    )
    schema = MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=ABS_TOLERANCE,
    )

    result = binary_metrics(artifact, schema)

    assert result.metric("auroc").value is None
    assert result.metric("auroc").reason == UNDEFINED_SINGLE_CLASS_AUROC
    assert result.metric("auroc").reason == "single_class_fold"
