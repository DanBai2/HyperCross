"""Property 24: Merge-order invariance and undefined metrics.

**Validates: Requirements 11.11-11.16**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from hypothesis import given, settings, strategies as st

from benchmark.metrics import (
    binary_metrics,
    merge_prediction_artifacts,
    regression_metrics,
)
from benchmark.schema import (
    MetricSchema,
    MetricSet,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    TaskKind,
    UNDEFINED_SINGLE_CLASS_AUROC,
    UNDEFINED_ZERO_VARIANCE_R2,
)


_REGRESSION_SCHEMA = MetricSchema(
    task_kind=TaskKind.MULTI_TARGET_REGRESSION,
    numeric_tolerance=1e-12,
    target_names=("target",),
)
_BINARY_SCHEMA = MetricSchema(
    task_kind=TaskKind.BINARY_CLASSIFICATION,
    numeric_tolerance=1e-12,
    decision_threshold=0.5,
    ece_bins=5,
)


@dataclass(frozen=True)
class _MergeCase:
    constant_truth: float
    regression_predictions: tuple[float, ...]
    binary_label: int
    probabilities: tuple[float, ...]
    permutation: tuple[int, ...]


@st.composite
def _merge_cases(draw: st.DrawFn) -> _MergeCase:
    sample_count = draw(st.integers(min_value=2, max_value=8))
    identity = tuple(range(sample_count))
    permutation = draw(st.permutations(identity))
    if permutation == identity:
        permutation = permutation[1:] + permutation[:1]

    finite_value = st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    probability = st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    return _MergeCase(
        constant_truth=draw(finite_value),
        regression_predictions=tuple(
            draw(
                st.lists(
                    finite_value,
                    min_size=sample_count,
                    max_size=sample_count,
                )
            )
        ),
        binary_label=draw(st.integers(min_value=0, max_value=1)),
        probabilities=tuple(
            draw(
                st.lists(
                    probability,
                    min_size=sample_count,
                    max_size=sample_count,
                )
            )
        ),
        permutation=permutation,
    )


def _singleton_shards(
    *,
    prefix: str,
    truths: Iterable[object],
    predictions: Iterable[object],
    probabilities: Iterable[float | None],
) -> tuple[PredictionArtifact, ...]:
    shards = []
    for index, (truth, prediction, probability) in enumerate(
        zip(truths, predictions, probabilities)
    ):
        sample_id = f"{prefix}-sample-{index:02d}"
        record = PredictionRecord(
            key=PredictionKey(
                sample_id=sample_id,
                fold_id="outer-1",
                seed=17,
                inference_branch="primary",
            ),
            y_true=truth,
            y_pred=prediction,
            probability=probability,
        )
        shards.append(
            PredictionArtifact.create(
                artifact_id=f"{prefix}-shard-{index:02d}",
                run_id=f"{prefix}-run-{index:02d}",
                records=(record,),
                expected_sample_ids=(sample_id,),
                complete=True,
            )
        )
    return tuple(shards)


def _complete_artifact(
    artifact_id: str, records: tuple[PredictionRecord, ...]
) -> PredictionArtifact:
    return PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=f"run-{artifact_id}",
        records=records,
        expected_sample_ids=tuple(record.key.sample_id for record in records),
        complete=True,
    )


def _assert_metrics_within_tolerance(
    first: MetricSet, second: MetricSet, tolerance: float
) -> None:
    assert first.metrics.keys() == second.metrics.keys()
    for name in first.metrics:
        first_value = first.metric(name)
        second_value = second.metric(name)
        assert first_value.reason == second_value.reason
        if first_value.value is None:
            assert second_value.value is None
        else:
            assert second_value.value is not None
            assert abs(first_value.value - second_value.value) <= tolerance
    assert first.confusion_matrices == second.confusion_matrices


@settings(max_examples=100, deadline=None)
@given(case=_merge_cases())
def test_shard_permutations_preserve_membership_metrics_and_undefined_reasons(
    case: _MergeCase,
) -> None:
    """Generated shard orders preserve results and both fixed undefined reasons."""

    sample_count = len(case.permutation)
    identity = tuple(range(sample_count))
    assert case.permutation != identity

    regression_shards = _singleton_shards(
        prefix="regression",
        truths=(case.constant_truth,) * sample_count,
        predictions=case.regression_predictions,
        probabilities=(None,) * sample_count,
    )
    regression_forward = merge_prediction_artifacts(regression_shards)
    regression_permuted = merge_prediction_artifacts(
        regression_shards[index] for index in case.permutation
    )
    assert regression_permuted == regression_forward
    assert {record.key for record in regression_permuted} == {
        record.key for record in regression_forward
    }

    regression_first = regression_metrics(
        _complete_artifact("regression-aggregate", regression_forward),
        _REGRESSION_SCHEMA,
    )
    regression_second = regression_metrics(
        _complete_artifact("regression-aggregate", regression_permuted),
        _REGRESSION_SCHEMA,
    )
    _assert_metrics_within_tolerance(
        regression_first,
        regression_second,
        _REGRESSION_SCHEMA.numeric_tolerance,
    )
    for metric_name in ("r2.target", "mean_r2"):
        undefined_r2 = regression_first.metric(metric_name)
        assert undefined_r2.value is None
        assert undefined_r2.reason == UNDEFINED_ZERO_VARIANCE_R2

    binary_shards = _singleton_shards(
        prefix="binary",
        truths=(case.binary_label,) * sample_count,
        predictions=tuple(
            int(probability >= _BINARY_SCHEMA.decision_threshold)
            for probability in case.probabilities
        ),
        probabilities=case.probabilities,
    )
    binary_forward = merge_prediction_artifacts(binary_shards)
    binary_permuted = merge_prediction_artifacts(
        binary_shards[index] for index in case.permutation
    )
    assert binary_permuted == binary_forward
    assert {record.key for record in binary_permuted} == {
        record.key for record in binary_forward
    }

    binary_first = binary_metrics(
        _complete_artifact("binary-aggregate", binary_forward),
        _BINARY_SCHEMA,
    )
    binary_second = binary_metrics(
        _complete_artifact("binary-aggregate", binary_permuted),
        _BINARY_SCHEMA,
    )
    _assert_metrics_within_tolerance(
        binary_first,
        binary_second,
        _BINARY_SCHEMA.numeric_tolerance,
    )
    undefined_auroc = binary_first.metric("auroc")
    assert undefined_auroc.value is None
    assert undefined_auroc.reason == UNDEFINED_SINGLE_CLASS_AUROC
