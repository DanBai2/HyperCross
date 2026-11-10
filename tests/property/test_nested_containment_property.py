"""Property 6: Nested containment.

**Validates: Requirements 3.6, 3.7, 3.16, 8.17, 8.18, 9.2, 9.4-9.6**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.protocols import ProtocolPlanner, ProtocolPlanningError
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    EvaluationBudget,
    InnerFold,
    MetricDirection,
    OuterFold,
    SampleManifest,
    SampleRecord,
    SelectionPolicy,
    canonical_sample_order,
    hash_object,
)


_INPUT_HASH = "b" * 64
_CREATED_UTC = datetime(2027, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _NestedGroupedCase:
    manifest: SampleManifest
    outer_fold: OuterFold
    inner_validation_group: str
    track: str


@st.composite
def _nested_grouped_folds(draw: st.DrawFn) -> _NestedGroupedCase:
    """Generate nested folds whose sample boundaries follow whole groups."""

    track = draw(st.sampled_from(("leaf", "histology")))
    group_count = draw(st.integers(min_value=3, max_value=8))
    group_sizes = draw(
        st.lists(
            st.integers(min_value=2, max_value=5),
            min_size=group_count,
            max_size=group_count,
        )
    )
    group_prefix = "variety" if track == "leaf" else "patient"
    groups = tuple(f"{group_prefix}-{index:02d}" for index in range(group_count))
    outer_test_index = draw(st.integers(min_value=0, max_value=group_count - 1))
    inner_validation_index = draw(
        st.integers(min_value=0, max_value=group_count - 1).filter(
            lambda index: index != outer_test_index
        )
    )

    records: list[SampleRecord] = []
    samples_by_group: dict[str, tuple[str, ...]] = {}
    for group_index, (group_id, group_size) in enumerate(zip(groups, group_sizes)):
        sample_ids = tuple(
            f"{track}/group-{group_index:02d}/sample-{sample_index:02d}"
            for sample_index in range(group_size)
        )
        samples_by_group[group_id] = sample_ids
        records.extend(
            SampleRecord(
                sample_id=sample_id,
                group_id=group_id,
                label_reference=f"labels/{sample_id}.json",
                input_references={"cube": f"inputs/{sample_id}.bin"},
                input_content_hashes={"cube": _INPUT_HASH},
                label_schema_metadata={"dimension": 3 if track == "leaf" else 1},
            )
            for sample_id in sample_ids
        )

    permutation = draw(st.permutations(tuple(records)))
    record_tuple = tuple(permutation)
    dataset_identity = {
        "track": track,
        "records": tuple(
            (record.sample_id, record.group_id) for record in record_tuple
        ),
    }
    manifest = SampleManifest(
        schema_version="1",
        dataset_key=f"property6.{track}.nested",
        records=record_tuple,
        canonical_order=canonical_sample_order(
            tuple(record.sample_id for record in record_tuple)
        ),
        dataset_hash=hash_object(dataset_identity),
        created_utc=_CREATED_UTC,
    )

    outer_test_group = groups[outer_test_index]
    inner_validation_group = groups[inner_validation_index]
    outer_test_ids = samples_by_group[outer_test_group]
    outer_train_ids = tuple(
        sample_id
        for group_id in groups
        if group_id != outer_test_group
        for sample_id in samples_by_group[group_id]
    )
    inner_validation_ids = samples_by_group[inner_validation_group]
    inner_train_ids = tuple(
        sample_id
        for group_id in groups
        if group_id not in {outer_test_group, inner_validation_group}
        for sample_id in samples_by_group[group_id]
    )
    outer_fold = OuterFold(
        fold_id=f"outer-{outer_test_index:02d}",
        outer_train_ids=outer_train_ids,
        outer_test_ids=outer_test_ids,
        inner_folds=(
            InnerFold(
                fold_id=f"inner-{inner_validation_index:02d}",
                train_ids=inner_train_ids,
                validation_ids=inner_validation_ids,
            ),
        ),
    )
    return _NestedGroupedCase(
        manifest=manifest,
        outer_fold=outer_fold,
        inner_validation_group=inner_validation_group,
        track=track,
    )


def _groups(manifest: SampleManifest, sample_ids: tuple[str, ...]) -> set[str]:
    group_by_sample = {
        record.sample_id: record.group_id for record in manifest.records
    }
    return {
        group_by_sample[sample_id]
        for sample_id in sample_ids
        if group_by_sample[sample_id] is not None
    }


def _plan(planner: ProtocolPlanner, case: _NestedGroupedCase, fold: OuterFold) -> None:
    planner.plan(
        case.manifest,
        schema_version="1",
        protocol_key=f"property6.{case.track}.nested.v1",
        folds=(fold,),
        selection_policy=SelectionPolicy(
            policy_key="inner-group-selection-v1",
            objective_metric="mean_r2" if case.track == "leaf" else "balanced_accuracy",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule="candidate_id_utf8_ascending",
        ),
        seed_set=(7, 42),
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=1),
    )


def _require_rejection(
    planner: ProtocolPlanner,
    case: _NestedGroupedCase,
    fold: OuterFold,
    expected_code: str,
) -> None:
    with pytest.raises(ProtocolPlanningError) as caught:
        _plan(planner, case, fold)

    matching_errors = tuple(
        error for error in caught.value.error_records if error.error_code == expected_code
    )
    assert matching_errors
    assert all(error.affected_identifiers["fold_id"] == fold.fold_id for error in matching_errors)
    assert all(
        error.affected_identifiers["inner_fold_id"] == fold.inner_folds[0].fold_id
        for error in matching_errors
    )


@settings(max_examples=100, deadline=None)
@given(case=_nested_grouped_folds())
def test_nested_grouped_folds_enforce_sample_containment_and_group_boundaries(
    case: _NestedGroupedCase,
) -> None:
    """Nested selection stays disjoint and within outer-train group boundaries."""

    with TemporaryDirectory(prefix="property6-") as directory:
        planner = ProtocolPlanner(
            FailureAtomicArtifactStore(Path(directory) / "artifacts")
        )
        _plan(planner, case, case.outer_fold)

        inner = case.outer_fold.inner_folds[0]
        outer_train = set(case.outer_fold.outer_train_ids)
        inner_train = set(inner.train_ids)
        inner_validation = set(inner.validation_ids)
        train_groups = _groups(case.manifest, inner.train_ids)
        validation_groups = _groups(case.manifest, inner.validation_ids)

        assert inner_train.isdisjoint(inner_validation)
        assert inner_train | inner_validation <= outer_train
        assert train_groups.isdisjoint(validation_groups)
        assert train_groups | validation_groups <= _groups(
            case.manifest, case.outer_fold.outer_train_ids
        )
        assert case.inner_validation_group in validation_groups

        sample_overlap = inner.validation_ids[0]
        overlapping_inner = InnerFold(
            fold_id=inner.fold_id,
            train_ids=inner.train_ids + (sample_overlap,),
            validation_ids=inner.validation_ids,
        )
        overlapping_fold = OuterFold(
            fold_id=case.outer_fold.fold_id,
            outer_train_ids=case.outer_fold.outer_train_ids,
            outer_test_ids=case.outer_fold.outer_test_ids,
            inner_folds=(overlapping_inner,),
        )
        _require_rejection(
            planner,
            case,
            overlapping_fold,
            "INNER_SAMPLE_ISOLATION_FAILED",
        )

        outer_test_sample = case.outer_fold.outer_test_ids[0]
        out_of_bound_inner = InnerFold(
            fold_id=inner.fold_id,
            train_ids=inner.train_ids,
            validation_ids=inner.validation_ids + (outer_test_sample,),
        )
        out_of_bound_fold = OuterFold(
            fold_id=case.outer_fold.fold_id,
            outer_train_ids=case.outer_fold.outer_train_ids,
            outer_test_ids=case.outer_fold.outer_test_ids,
            inner_folds=(out_of_bound_inner,),
        )
        _require_rejection(
            planner,
            case,
            out_of_bound_fold,
            "INNER_CONTAINMENT_FAILED",
        )

        crossing_sample = inner.validation_ids[0]
        crossing_inner = InnerFold(
            fold_id=inner.fold_id,
            train_ids=inner.train_ids + (crossing_sample,),
            validation_ids=inner.validation_ids[1:],
        )
        group_crossing_fold = OuterFold(
            fold_id=case.outer_fold.fold_id,
            outer_train_ids=case.outer_fold.outer_train_ids,
            outer_test_ids=case.outer_fold.outer_test_ids,
            inner_folds=(crossing_inner,),
        )
        assert set(crossing_inner.train_ids).isdisjoint(crossing_inner.validation_ids)
        assert set(crossing_inner.train_ids) | set(crossing_inner.validation_ids) <= outer_train
        _require_rejection(
            planner,
            case,
            group_crossing_fold,
            "INNER_GROUP_ISOLATION_FAILED",
        )
