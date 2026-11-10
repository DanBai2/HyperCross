"""Property 22: Prediction key uniqueness and exact coverage.

**Validates: Requirements 11.1-11.8**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.metrics import (
    PredictionArtifactValidationError,
    PredictionArtifactWriter,
    regression_metrics,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    MetricSchema,
    PredictionKey,
    PredictionRecord,
    PredictionShard,
    TaskKind,
)


_DEFECTS = ("missing", "unexpected", "duplicate")


@dataclass(frozen=True)
class _InvalidShardCase:
    expected_sample_ids: tuple[str, ...]
    shards: tuple[PredictionShard, ...]
    defects: frozenset[str]
    missing_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]


@st.composite
def _invalid_shard_cases(draw: st.DrawFn) -> _InvalidShardCase:
    sample_count = draw(st.integers(min_value=2, max_value=10))
    token = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            min_size=1,
            max_size=8,
        )
    )
    fold_id = f"outer-{draw(st.integers(min_value=0, max_value=12))}"
    seed = draw(st.integers(min_value=-(2**16), max_value=2**16))
    branch = draw(st.sampled_from(("primary", "pca-only", "ensemble")))
    expected_ids = tuple(f"sample-{token}-{index}" for index in range(sample_count))
    defects = frozenset(
        draw(
            st.sets(
                st.sampled_from(_DEFECTS),
                min_size=1,
                max_size=len(_DEFECTS),
            )
        )
    )

    records = [
        PredictionRecord(
            key=PredictionKey(
                sample_id=sample_id,
                fold_id=fold_id,
                seed=seed,
                inference_branch=branch,
            ),
            y_true=float(index),
            y_pred=float(index) + 0.25,
        )
        for index, sample_id in enumerate(expected_ids)
    ]

    missing_ids: tuple[str, ...] = ()
    if "missing" in defects:
        missing_index = draw(st.integers(min_value=0, max_value=sample_count - 1))
        missing_ids = (records.pop(missing_index).key.sample_id,)

    unexpected_ids: tuple[str, ...] = ()
    if "unexpected" in defects:
        unexpected_id = f"unexpected-{token}-{sample_count}"
        unexpected_ids = (unexpected_id,)
        records.append(
            PredictionRecord(
                key=PredictionKey(
                    sample_id=unexpected_id,
                    fold_id=fold_id,
                    seed=seed,
                    inference_branch=branch,
                ),
                y_true=-1.0,
                y_pred=-0.75,
            )
        )

    if "duplicate" in defects:
        duplicate_candidates = [
            record for record in records if record.key.sample_id in expected_ids
        ]
        duplicate = draw(st.sampled_from(duplicate_candidates))
        records.append(duplicate.model_copy(deep=True))

    permutation = draw(st.permutations(tuple(range(len(records)))))
    ordered_records = [records[index] for index in permutation]
    cut_flags = draw(
        st.lists(
            st.booleans(),
            min_size=max(0, len(records) - 1),
            max_size=max(0, len(records) - 1),
        )
    )
    chunks: list[tuple[PredictionRecord, ...]] = []
    start = 0
    for position, should_cut in enumerate(cut_flags, start=1):
        if should_cut:
            chunks.append(tuple(ordered_records[start:position]))
            start = position
    chunks.append(tuple(ordered_records[start:]))
    shards = tuple(
        PredictionShard(shard_id=f"shard-{index}", records=chunk)
        for index, chunk in enumerate(chunks)
    )

    return _InvalidShardCase(
        expected_sample_ids=expected_ids,
        shards=shards,
        defects=defects,
        missing_ids=missing_ids,
        unexpected_ids=unexpected_ids,
    )


@settings(max_examples=100, deadline=None)
@given(case=_invalid_shard_cases())
def test_invalid_prediction_keys_are_rejected_before_metrics_and_preserve_artifacts(
    case: _InvalidShardCase,
) -> None:
    """Missing, unexpected, or duplicate keys cannot reach metric computation."""

    with TemporaryDirectory(prefix="property22-") as directory:
        root = Path(directory) / "artifact root with spaces"
        store = FailureAtomicArtifactStore(root)
        writer = PredictionArtifactWriter(store)
        completed_prediction = store.publish_bytes(
            artifact_id="completed-prediction",
            relative_path="predictions/completed.json",
            content=b'{"status":"complete"}',
            media_type="application/json",
        )
        completed_aggregate = store.publish_bytes(
            artifact_id="completed-aggregate",
            relative_path="aggregates/completed.json",
            content=b'{"status":"complete"}',
            media_type="application/json",
        )
        prediction_path = root / Path(completed_prediction.relative_path)
        aggregate_path = root / Path(completed_aggregate.relative_path)
        prediction_before = prediction_path.read_bytes()
        aggregate_before = aggregate_path.read_bytes()
        metric_calls = 0

        def publish_then_compute_metrics() -> None:
            nonlocal metric_calls
            published = writer.publish_shards(
                artifact_id="generated-invalid-prediction",
                run_id="generated-invalid-run",
                expected_sample_ids=case.expected_sample_ids,
                shards=case.shards,
                relative_path="predictions/rejected.json",
            )
            metric_calls += 1
            regression_metrics(
                published.artifact,
                MetricSchema(
                    task_kind=TaskKind.MULTI_TARGET_REGRESSION,
                    numeric_tolerance=1e-12,
                    target_names=("target",),
                ),
            )

        with pytest.raises(PredictionArtifactValidationError) as caught:
            publish_then_compute_metrics()

        issue_codes = {issue.code for issue in caught.value.report.errors}
        if case.defects & {"missing", "unexpected"}:
            assert "PREDICTION_COVERAGE_MISMATCH" in issue_codes
        if "duplicate" in case.defects:
            assert {
                "DUPLICATE_PREDICTION_KEYS",
                "DUPLICATE_PREDICTION_SAMPLE_IDS",
            } <= issue_codes

        error = caught.value.error_record
        assert error.missing_ids == case.missing_ids
        assert error.unexpected_ids == case.unexpected_ids
        assert bool(error.observed_value["duplicate_prediction_keys"]) == (
            "duplicate" in case.defects
        )
        assert error.invariant_checks["metrics_not_computed"] is True
        assert error.invariant_checks["prediction_artifact_unchanged"] is True
        assert error.invariant_checks["aggregate_records_unchanged"] is True
        assert metric_calls == 0
        assert not (root / "predictions" / "rejected.json").exists()
        assert prediction_path.read_bytes() == prediction_before
        assert aggregate_path.read_bytes() == aggregate_before
