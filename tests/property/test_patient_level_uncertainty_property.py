"""Property 18: Patient-level uncertainty.

**Validates: Requirements 9.27-9.32, 11.24-11.31**
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from benchmark.metrics import paired_patient_compare, patient_bootstrap
from benchmark.schema import (
    MetricSchema,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    TaskKind,
)


_SCHEMA = MetricSchema(
    task_kind=TaskKind.BINARY_CLASSIFICATION,
    numeric_tolerance=1e-12,
    decision_threshold=0.5,
)


@dataclass(frozen=True)
class _PatientUncertaintyCase:
    seed: int
    roi_counts: tuple[int, ...]
    phase: int


@st.composite
def _patient_uncertainty_cases(draw: st.DrawFn) -> _PatientUncertaintyCase:
    """Generate bounded clusters with unequal ROI counts and opposing methods."""

    patient_count = draw(st.integers(min_value=2, max_value=4))
    extra_counts = draw(
        st.lists(
            st.integers(min_value=1, max_value=3),
            min_size=patient_count - 2,
            max_size=patient_count - 2,
        )
    )
    return _PatientUncertaintyCase(
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        # Unequal first clusters make patient resampling observably different
        # from treating every ROI as an independent resampling unit.
        roi_counts=(1, 2, *extra_counts),
        phase=draw(st.integers(min_value=0, max_value=1)),
    )


def _artifact_and_groups(
    artifact_id: str,
    roi_counts: tuple[int, ...],
    patient_correctness: tuple[bool, ...],
) -> tuple[PredictionArtifact, dict[str, str]]:
    records: list[PredictionRecord] = []
    groups: dict[str, str] = {}
    for patient_index, (roi_count, correct) in enumerate(
        zip(roi_counts, patient_correctness)
    ):
        group_id = f"patient-{patient_index:02d}"
        probability = 0.9 if correct else 0.1
        for roi_index in range(roi_count):
            sample_id = f"{artifact_id}-p{patient_index:02d}-roi{roi_index:02d}"
            groups[sample_id] = group_id
            records.append(
                PredictionRecord(
                    key=PredictionKey(
                        sample_id=sample_id,
                        fold_id="outer-oof",
                        seed=17,
                        inference_branch="primary",
                    ),
                    y_true=1,
                    y_pred=int(correct),
                    probability=probability,
                )
            )

    artifact = PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=f"run-{artifact_id}",
        records=tuple(records),
        expected_sample_ids=tuple(groups),
        complete=True,
    )
    return artifact, groups


def _accuracy(
    selected_patient_indices: tuple[int, ...],
    roi_counts: tuple[int, ...],
    patient_correctness: tuple[bool, ...],
) -> float:
    correct = sum(
        roi_counts[index] * int(patient_correctness[index])
        for index in selected_patient_indices
    )
    total = sum(roi_counts[index] for index in selected_patient_indices)
    return correct / total


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
@given(case=_patient_uncertainty_cases())
def test_patient_bootstrap_and_paired_comparison_preserve_fixed_cluster_policy(
    case: _PatientUncertaintyCase,
) -> None:
    """The fixed policy records its seed and resamples matched patient clusters."""

    patient_count = len(case.roi_counts)
    candidate_correctness = tuple(
        (patient_index + case.phase) % 2 == 0
        for patient_index in range(patient_count)
    )
    control_correctness = tuple(not value for value in candidate_correctness)
    candidate, candidate_groups = _artifact_and_groups(
        f"candidate-{case.seed}-{case.phase}",
        case.roi_counts,
        candidate_correctness,
    )
    control, control_groups = _artifact_and_groups(
        f"control-{case.seed}-{case.phase}",
        case.roi_counts,
        control_correctness,
    )

    interval = patient_bootstrap(
        candidate,
        "accuracy",
        group_by_sample_id=candidate_groups,
        metric_schema=_SCHEMA,
        seed=case.seed,
    )
    comparison = paired_patient_compare(
        candidate,
        control,
        "accuracy",
        candidate_group_by_sample_id=candidate_groups,
        control_group_by_sample_id=control_groups,
        metric_schema=_SCHEMA,
        seed=case.seed,
    )

    # This independent reference selects one shared patient sequence per draw.
    # Every selected patient contributes all of its ROIs to both methods.
    rng = random.Random(case.seed)
    candidate_estimates: list[float] = []
    paired_differences: list[float] = []
    for _ in range(10_000):
        selected = tuple(
            rng.randrange(patient_count) for _ in range(patient_count)
        )
        candidate_value = _accuracy(
            selected, case.roi_counts, candidate_correctness
        )
        control_value = _accuracy(selected, case.roi_counts, control_correctness)
        candidate_estimates.append(candidate_value)
        paired_differences.append(candidate_value - control_value)

    all_patients = tuple(range(patient_count))
    expected_candidate = _accuracy(
        all_patients, case.roi_counts, candidate_correctness
    )
    expected_control = _accuracy(
        all_patients, case.roi_counts, control_correctness
    )
    expected_groups = tuple(f"patient-{index:02d}" for index in all_patients)

    assert interval.point_estimate == pytest.approx(expected_candidate, abs=1e-12)
    assert interval.lower_bound == pytest.approx(
        _percentile(candidate_estimates, 0.025), abs=1e-12
    )
    assert interval.upper_bound == pytest.approx(
        _percentile(candidate_estimates, 0.975), abs=1e-12
    )
    assert interval.source_prediction_hash == candidate.content_hash
    assert interval.resampling_unit == "patient_group_id"
    assert interval.resamples == 10_000
    assert interval.confidence_level == 0.95
    assert interval.sidedness == "two_sided"
    assert interval.seed == case.seed

    assert comparison.candidate_estimate == pytest.approx(
        expected_candidate, abs=1e-12
    )
    assert comparison.control_estimate == pytest.approx(expected_control, abs=1e-12)
    assert comparison.estimated_difference == pytest.approx(
        expected_candidate - expected_control, abs=1e-12
    )
    assert comparison.lower_bound == pytest.approx(
        _percentile(paired_differences, 0.025), abs=1e-12
    )
    assert comparison.upper_bound == pytest.approx(
        _percentile(paired_differences, 0.975), abs=1e-12
    )
    assert comparison.common_group_ids == expected_groups
    assert comparison.candidate_prediction_hash == candidate.content_hash
    assert comparison.control_prediction_hash == control.content_hash
    assert comparison.resampling_unit == "patient_group_id"
    assert comparison.resamples == 10_000
    assert comparison.confidence_level == 0.95
    assert comparison.sidedness == "two_sided"
    assert comparison.seed == case.seed
