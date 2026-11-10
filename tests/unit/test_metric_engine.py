"""Targeted unit coverage for task 10.2 metric computation."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from benchmark.metrics import MetricInputError, binary_metrics, regression_metrics, remote_metrics
from benchmark.schema import (
    MetricSchema,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    TaskKind,
    UNDEFINED_SINGLE_CLASS_AUROC,
    UNDEFINED_ZERO_VARIANCE_R2,
)


def _artifact(
    artifact_id: str,
    truths: list[object],
    predictions: list[object],
    probabilities: list[object | None] | None = None,
) -> PredictionArtifact:
    probabilities = probabilities or [None] * len(truths)
    records = tuple(
        PredictionRecord(
            key=PredictionKey(
                sample_id=f"sample-{index}",
                fold_id="outer-1",
                seed=42,
                inference_branch="primary",
            ),
            y_true=truth,
            y_pred=prediction,
            probability=probability,
        )
        for index, (truth, prediction, probability) in enumerate(
            zip(truths, predictions, probabilities)
        )
    )
    return PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=f"run-{artifact_id}",
        records=records,
        expected_sample_ids=tuple(record.key.sample_id for record in records),
        complete=True,
    )


def test_regression_metrics_report_per_target_r2_mae_and_means() -> None:
    artifact = _artifact(
        "regression",
        truths=[(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)],
        predictions=[(1.0, 1.0), (2.0, 5.0), (4.0, 7.0)],
    )
    schema = MetricSchema(
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        numeric_tolerance=1e-12,
        target_names=("GrainWeight", "Gsw"),
    )

    result = regression_metrics(artifact, schema)

    assert result.source_prediction_hash == artifact.content_hash
    assert result.metric_schema.numeric_tolerance == 1e-12
    assert result.metric("r2.GrainWeight").value == pytest.approx(0.5, abs=1e-12)
    assert result.metric("r2.Gsw").value == pytest.approx(0.625, abs=1e-12)
    assert result.metric("mean_r2").value == pytest.approx(0.5625, abs=1e-12)
    assert result.metric("mae.GrainWeight").value == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert result.metric("mae.Gsw").value == pytest.approx(1.0, abs=1e-12)
    assert result.metric("mean_mae").value == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_zero_variance_r2_is_null_with_fixed_reason() -> None:
    artifact = _artifact("constant-target", [2.0, 2.0], [2.0, 3.0])
    schema = MetricSchema(
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        numeric_tolerance=1e-12,
        target_names=("PhiPS2",),
    )

    result = regression_metrics(artifact, schema)

    assert result.metric("r2.PhiPS2").value is None
    assert result.metric("r2.PhiPS2").reason == UNDEFINED_ZERO_VARIANCE_R2
    assert result.metric("mean_r2").value is None
    assert result.metric("mean_r2").reason == UNDEFINED_ZERO_VARIANCE_R2


def test_binary_metrics_use_fixed_threshold_tie_aware_auroc_and_ece_bins() -> None:
    artifact = _artifact(
        "binary",
        truths=[0, 0, 1, 1],
        predictions=[0, 0, 0, 1],
        probabilities=[0.1, 0.4, 0.35, 0.8],
    )
    schema = MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=1e-12,
        decision_threshold=0.5,
        ece_bins=2,
    )

    result = binary_metrics(artifact, schema)

    assert result.metric("accuracy").value == pytest.approx(0.75, abs=1e-12)
    assert result.metric("balanced_accuracy").value == pytest.approx(0.75, abs=1e-12)
    assert result.metric("auroc").value == pytest.approx(0.75, abs=1e-12)
    assert result.metric("sensitivity").value == pytest.approx(0.5, abs=1e-12)
    assert result.metric("specificity").value == pytest.approx(1.0, abs=1e-12)
    assert result.metric("ece").value == pytest.approx(0.0875, abs=1e-12)
    assert result.confusion_matrices["confusion_matrix"] == ((2, 0), (1, 1))


def test_binary_auroc_ties_and_single_class_reason_are_deterministic() -> None:
    tie_artifact = _artifact(
        "binary-tie",
        truths=[0, 1, 0, 1],
        predictions=[0, 1, 1, 1],
        probabilities=[0.2, 0.2, 0.8, 0.8],
    )
    single_class = _artifact(
        "single-class",
        truths=[1, 1],
        predictions=[1, 0],
        probabilities=[0.9, 0.4],
    )
    schema = MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=1e-12,
    )

    assert binary_metrics(tie_artifact, schema).metric("auroc").value == pytest.approx(
        0.5, abs=1e-12
    )
    undefined = binary_metrics(single_class, schema).metric("auroc")
    assert undefined.value is None
    assert undefined.reason == UNDEFINED_SINGLE_CLASS_AUROC


def test_remote_metrics_report_oa_aa_and_kappa() -> None:
    artifact = _artifact(
        "remote",
        truths=[0, 0, 1, 1, 2, 2],
        predictions=[0, 1, 1, 1, 2, 0],
    )
    schema = MetricSchema(
        task_kind=TaskKind.REMOTE_CLASSIFICATION,
        numeric_tolerance=1e-12,
        class_labels=(0, 1, 2),
    )

    result = remote_metrics(artifact, schema)

    assert result.metric("oa").value == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert result.metric("aa").value == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert result.metric("kappa").value == pytest.approx(0.5, abs=1e-12)


def test_metric_schema_requires_finite_positive_numeric_tolerance() -> None:
    for tolerance in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValidationError):
            MetricSchema(
                task_kind=TaskKind.MULTI_TARGET_REGRESSION,
                numeric_tolerance=tolerance,
                target_names=("target",),
            )


def test_metrics_reject_incomplete_artifacts_and_wrong_task_schema() -> None:
    complete = _artifact("complete-regression", [1.0, 2.0], [1.0, 2.0])
    incomplete = PredictionArtifact.create(
        artifact_id="incomplete-regression",
        run_id="run-incomplete-regression",
        records=complete.records,
        expected_sample_ids=complete.expected_sample_ids,
        complete=False,
    )
    regression_schema = MetricSchema(
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        numeric_tolerance=1e-12,
        target_names=("target",),
    )
    binary_schema = MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=1e-12,
    )

    with pytest.raises(MetricInputError, match="complete PredictionArtifact"):
        regression_metrics(incomplete, regression_schema)
    with pytest.raises(MetricInputError, match="task_kind"):
        regression_metrics(complete, binary_schema)
