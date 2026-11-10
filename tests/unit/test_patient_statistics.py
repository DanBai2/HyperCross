"""Targeted unit coverage for task 10.4 patient-level statistics."""

from __future__ import annotations

import math
import random

import pytest
from pydantic import ValidationError

from benchmark.metrics import (
    MetricEngine,
    MetricInputError,
    paired_patient_compare,
    patient_bootstrap,
)
from benchmark.schema import (
    ConfidenceInterval,
    MetricSchema,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    StatisticalComparison,
    TaskKind,
)


def _artifact(
    artifact_id: str,
    rows: list[tuple[str, int, float]],
) -> PredictionArtifact:
    records = tuple(
        PredictionRecord(
            key=PredictionKey(
                sample_id=sample_id,
                fold_id="outer-oof",
                seed=17,
                inference_branch="primary",
            ),
            y_true=truth,
            y_pred=int(probability >= 0.5),
            probability=probability,
        )
        for sample_id, truth, probability in rows
    )
    return PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=f"run-{artifact_id}",
        records=records,
        expected_sample_ids=tuple(record.key.sample_id for record in records),
        complete=True,
    )


_SCHEMA = MetricSchema(
    task_kind=TaskKind.BINARY_CLASSIFICATION,
    numeric_tolerance=1e-12,
    decision_threshold=0.5,
)


def _linear_percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def test_patient_bootstrap_keeps_all_rois_and_records_fixed_policy() -> None:
    artifact = _artifact(
        "patient-bootstrap",
        [
            ("a-1", 1, 0.9),
            ("b-1", 1, 0.8),
            ("b-2", 1, 0.2),
            ("c-1", 0, 0.9),
            ("c-2", 0, 0.8),
            ("c-3", 0, 0.7),
            ("c-4", 0, 0.6),
            ("d-1", 0, 0.1),
            ("d-2", 1, 0.9),
            ("d-3", 1, 0.8),
        ],
    )
    groups = {
        "a-1": "patient-a",
        "b-1": "patient-b",
        "b-2": "patient-b",
        "c-1": "patient-c",
        "c-2": "patient-c",
        "c-3": "patient-c",
        "c-4": "patient-c",
        "d-1": "patient-d",
        "d-2": "patient-d",
        "d-3": "patient-d",
    }

    result = patient_bootstrap(
        artifact,
        "accuracy",
        group_by_sample_id=groups,
        metric_schema=_SCHEMA,
        seed=1234,
    )

    # Independent reference: each draw chooses four patients and includes every
    # ROI from each chosen patient, including repeated patient clusters.
    patient_outcomes = {
        "patient-a": (1,),
        "patient-b": (1, 0),
        "patient-c": (0, 0, 0, 0),
        "patient-d": (1, 1, 1),
    }
    patient_ids = tuple(sorted(patient_outcomes))
    rng = random.Random(1234)
    reference = []
    for _ in range(10_000):
        selected = tuple(
            patient_ids[rng.randrange(len(patient_ids))] for _ in patient_ids
        )
        outcomes = tuple(
            outcome
            for patient_id in selected
            for outcome in patient_outcomes[patient_id]
        )
        reference.append(sum(outcomes) / len(outcomes))

    assert isinstance(result, ConfidenceInterval)
    assert result.source_prediction_hash == artifact.content_hash
    assert result.point_estimate == pytest.approx(0.5, abs=1e-12)
    assert result.lower_bound == pytest.approx(
        _linear_percentile(reference, 0.025), abs=1e-12
    )
    assert result.upper_bound == pytest.approx(
        _linear_percentile(reference, 0.975), abs=1e-12
    )
    assert result.resampling_unit == "patient_group_id"
    assert result.resamples == 10_000
    assert result.confidence_level == 0.95
    assert result.sidedness == "two_sided"
    assert result.seed == 1234
    assert MetricEngine.patient_bootstrap(
        artifact,
        "accuracy",
        group_by_sample_id=groups,
        metric_schema=_SCHEMA,
        seed=1234,
    ) == result


def test_paired_comparison_uses_only_common_patients_and_shared_draws() -> None:
    candidate = _artifact(
        "candidate",
        [
            ("candidate-a", 1, 0.1),
            ("candidate-b", 1, 0.9),
            ("candidate-c", 0, 0.1),
        ],
    )
    control = _artifact(
        "control",
        [
            ("control-b", 1, 0.1),
            ("control-c", 0, 0.1),
            ("control-d", 1, 0.9),
        ],
    )

    result = paired_patient_compare(
        candidate,
        control,
        "accuracy",
        candidate_group_by_sample_id={
            "candidate-a": "patient-a",
            "candidate-b": "patient-b",
            "candidate-c": "patient-c",
        },
        control_group_by_sample_id={
            "control-b": "patient-b",
            "control-c": "patient-c",
            "control-d": "patient-d",
        },
        metric_schema=_SCHEMA,
        seed=77,
    )

    assert isinstance(result, StatisticalComparison)
    assert result.common_group_ids == ("patient-b", "patient-c")
    assert result.candidate_estimate == pytest.approx(1.0, abs=1e-12)
    assert result.control_estimate == pytest.approx(0.5, abs=1e-12)
    assert result.estimated_difference == pytest.approx(0.5, abs=1e-12)
    assert result.lower_bound == pytest.approx(0.0, abs=1e-12)
    assert result.upper_bound == pytest.approx(1.0, abs=1e-12)
    assert result.candidate_prediction_hash == candidate.content_hash
    assert result.control_prediction_hash == control.content_hash
    assert result.resampling_unit == "patient_group_id"
    assert result.resamples == 10_000
    assert result.confidence_level == 0.95
    assert result.sidedness == "two_sided"
    assert result.seed == 77


def test_patient_statistics_reject_missing_or_nonoverlapping_group_ids() -> None:
    artifact = _artifact("missing-group", [("sample-1", 0, 0.1)])
    with pytest.raises(MetricInputError, match="missing prediction samples"):
        patient_bootstrap(
            artifact,
            "accuracy",
            group_by_sample_id={},
            metric_schema=_SCHEMA,
            seed=1,
        )

    control = _artifact("no-common-control", [("sample-2", 0, 0.1)])
    with pytest.raises(MetricInputError, match="common patient Group_ID"):
        paired_patient_compare(
            artifact,
            control,
            "accuracy",
            candidate_group_by_sample_id={"sample-1": "patient-a"},
            control_group_by_sample_id={"sample-2": "patient-b"},
            metric_schema=_SCHEMA,
            seed=1,
        )


def test_statistics_schema_cannot_override_frozen_resampling_policy() -> None:
    with pytest.raises(ValidationError):
        ConfidenceInterval(
            metric_name="accuracy",
            source_prediction_hash="a" * 64,
            point_estimate=0.5,
            lower_bound=0.4,
            upper_bound=0.6,
            resamples=9999,
            seed=5,
        )
