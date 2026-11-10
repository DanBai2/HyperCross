"""Property 23: Prediction-first metric reproducibility.

**Validates: Requirements 11.9, 11.10, 17.34**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.metrics import (
    MetricInputError,
    PredictionArtifactWriter,
    binary_metrics,
    regression_metrics,
    remote_metrics,
)
from benchmark.schema import (
    MetricSchema,
    MetricSet,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    TaskKind,
    canonical_deserialize,
    canonical_serialize,
)


_STRICT_RECOMPUTATION_TOLERANCE = 1e-6
MetricFunction = Callable[[PredictionArtifact, MetricSchema], MetricSet]


@dataclass(frozen=True)
class _MetricCase:
    artifact: PredictionArtifact
    schema: MetricSchema
    compute: MetricFunction


def _artifact(
    *,
    case_id: str,
    truths: tuple[object, ...],
    predictions: tuple[object, ...],
    probabilities: tuple[object | None, ...] | None = None,
) -> PredictionArtifact:
    supplied_probabilities = probabilities or (None,) * len(truths)
    records = tuple(
        PredictionRecord(
            key=PredictionKey(
                sample_id=f"sample-{index:02d}",
                fold_id="outer-property-23",
                seed=23,
                inference_branch="strict-primary",
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
        artifact_id=f"prediction-{case_id}",
        run_id=f"run-{case_id}",
        records=records,
        expected_sample_ids=tuple(record.key.sample_id for record in records),
        complete=True,
    )


@st.composite
def _regression_cases(draw: st.DrawFn) -> _MetricCase:
    sample_count = draw(st.integers(min_value=2, max_value=12))
    target_count = draw(st.integers(min_value=1, max_value=3))
    values = st.integers(min_value=-10_000, max_value=10_000)
    truths_by_sample = draw(
        st.lists(
            st.lists(values, min_size=target_count, max_size=target_count),
            min_size=sample_count,
            max_size=sample_count,
        )
    )
    predictions_by_sample = draw(
        st.lists(
            st.lists(values, min_size=target_count, max_size=target_count),
            min_size=sample_count,
            max_size=sample_count,
        )
    )
    truths = tuple(tuple(float(value) for value in row) for row in truths_by_sample)
    predictions = tuple(
        tuple(float(value) for value in row) for row in predictions_by_sample
    )
    tolerance = draw(st.sampled_from((1e-12, 1e-10, 1e-8)))
    case_id = draw(st.uuids()).hex
    return _MetricCase(
        artifact=_artifact(
            case_id=case_id,
            truths=truths,
            predictions=predictions,
        ),
        schema=MetricSchema(
            task_kind=TaskKind.MULTI_TARGET_REGRESSION,
            numeric_tolerance=tolerance,
            target_names=tuple(f"target-{index}" for index in range(target_count)),
        ),
        compute=regression_metrics,
    )


@st.composite
def _binary_cases(draw: st.DrawFn) -> _MetricCase:
    sample_count = draw(st.integers(min_value=2, max_value=20))
    truths = tuple(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=sample_count,
                max_size=sample_count,
            )
        )
    )
    probabilities = tuple(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=1000).map(
                    lambda value: value / 1000.0
                ),
                min_size=sample_count,
                max_size=sample_count,
            )
        )
    )
    threshold = draw(
        st.integers(min_value=0, max_value=1000).map(lambda value: value / 1000.0)
    )
    predictions = tuple(int(probability >= threshold) for probability in probabilities)
    tolerance = draw(st.sampled_from((1e-12, 1e-10, 1e-8)))
    case_id = draw(st.uuids()).hex
    return _MetricCase(
        artifact=_artifact(
            case_id=case_id,
            truths=truths,
            predictions=predictions,
            probabilities=probabilities,
        ),
        schema=MetricSchema(
            task_kind=TaskKind.BINARY_CLASSIFICATION,
            numeric_tolerance=tolerance,
            decision_threshold=threshold,
            ece_bins=draw(st.integers(min_value=1, max_value=20)),
        ),
        compute=binary_metrics,
    )


@st.composite
def _remote_cases(draw: st.DrawFn) -> _MetricCase:
    class_count = draw(st.integers(min_value=2, max_value=6))
    labels = tuple(f"class-{index}" for index in range(class_count))
    sample_count = draw(st.integers(min_value=2, max_value=20))
    truths = tuple(
        draw(
            st.lists(
                st.sampled_from(labels),
                min_size=sample_count,
                max_size=sample_count,
            )
        )
    )
    predictions = tuple(
        draw(
            st.lists(
                st.sampled_from(labels),
                min_size=sample_count,
                max_size=sample_count,
            )
        )
    )
    tolerance = draw(st.sampled_from((1e-12, 1e-10, 1e-8)))
    case_id = draw(st.uuids()).hex
    return _MetricCase(
        artifact=_artifact(
            case_id=case_id,
            truths=truths,
            predictions=predictions,
        ),
        schema=MetricSchema(
            task_kind=TaskKind.REMOTE_CLASSIFICATION,
            numeric_tolerance=tolerance,
            class_labels=labels,
        ),
        compute=remote_metrics,
    )


_METRIC_CASES = st.one_of(_regression_cases(), _binary_cases(), _remote_cases())


def _assert_metric_sets_close(
    expected: MetricSet,
    observed: MetricSet,
    *,
    tolerance: float,
) -> None:
    assert observed.source_prediction_hash == expected.source_prediction_hash
    assert observed.metric_schema == expected.metric_schema
    assert observed.metrics.keys() == expected.metrics.keys()
    assert observed.confusion_matrices == expected.confusion_matrices
    for name, expected_metric in expected.metrics.items():
        observed_metric = observed.metrics[name]
        if expected_metric.value is None:
            assert observed_metric.value is None
            assert observed_metric.reason == expected_metric.reason
        else:
            assert observed_metric.reason is None
            assert observed_metric.value is not None
            assert abs(observed_metric.value - expected_metric.value) <= tolerance


@settings(max_examples=100, deadline=None)
@given(case=_METRIC_CASES)
def test_validated_prediction_metrics_are_reproducible_from_frozen_artifacts(
    case: _MetricCase,
) -> None:
    """Only validated artifacts compute, and frozen recomputation stays within bounds."""

    reported = case.compute(case.artifact, case.schema)
    repeated = case.compute(case.artifact, case.schema)
    _assert_metric_sets_close(
        reported,
        repeated,
        tolerance=case.schema.numeric_tolerance,
    )

    frozen_artifact = PredictionArtifactWriter.reload(
        canonical_serialize(case.artifact)
    )
    frozen_schema = MetricSchema.model_validate(
        canonical_deserialize(canonical_serialize(case.schema))
    )
    strict_recomputed = case.compute(frozen_artifact, frozen_schema)
    _assert_metric_sets_close(
        reported,
        strict_recomputed,
        tolerance=_STRICT_RECOMPUTATION_TOLERANCE,
    )

    incomplete = PredictionArtifact.create(
        artifact_id=f"{case.artifact.artifact_id}-incomplete",
        run_id=f"{case.artifact.run_id}-incomplete",
        records=case.artifact.records,
        expected_sample_ids=case.artifact.expected_sample_ids,
        complete=False,
    )
    with pytest.raises(MetricInputError, match="complete PredictionArtifact"):
        case.compute(incomplete, case.schema)

    altered_hash = (
        ("0" if case.artifact.content_hash[0] != "0" else "1")
        + case.artifact.content_hash[1:]
    )
    hash_invalid = case.artifact.model_copy(
        update={"content_hash": altered_hash}
    )
    with pytest.raises(MetricInputError, match="content_hash is invalid"):
        case.compute(hash_invalid, case.schema)

    with pytest.raises(TypeError, match="validated PredictionArtifact"):
        case.compute(case.artifact.records, case.schema)  # type: ignore[arg-type]
