"""Property 5: outer sample and group isolation.

**Validates: Requirements 3.3-3.5, 3.10, 3.12, 3.14**
"""

from __future__ import annotations

import hashlib
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
    MetricDirection,
    OuterFold,
    SampleManifest,
    SampleRecord,
    SelectionPolicy,
    canonical_sample_order,
    hash_object,
)


_CREATED_UTC = datetime(2027, 1, 1, tzinfo=timezone.utc)
_INPUT_HASH = hashlib.sha256(b"property-5-input").hexdigest()


@dataclass(frozen=True)
class _GroupedManifestCase:
    manifest: SampleManifest
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    group_by_sample: dict[str, str]


@st.composite
def _grouped_manifest_cases(draw: st.DrawFn) -> _GroupedManifestCase:
    """Generate group-preserving outer partitions with splittable groups."""

    group_count = draw(st.integers(min_value=2, max_value=7))
    group_sizes = draw(
        st.lists(
            st.integers(min_value=2, max_value=5),
            min_size=group_count,
            max_size=group_count,
        )
    )
    held_out_group_count = draw(
        st.integers(min_value=1, max_value=group_count - 1)
    )
    held_out_group_indices = set(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=group_count - 1),
                min_size=held_out_group_count,
                max_size=held_out_group_count,
                unique=True,
            )
        )
    )

    records: list[SampleRecord] = []
    train_ids: list[str] = []
    test_ids: list[str] = []
    group_by_sample: dict[str, str] = {}
    for group_index, group_size in enumerate(group_sizes):
        group_id = f"patient-{group_index:02d}"
        destination = test_ids if group_index in held_out_group_indices else train_ids
        for sample_index in range(group_size):
            sample_id = f"sample-{group_index:02d}-{sample_index:02d}"
            group_by_sample[sample_id] = group_id
            destination.append(sample_id)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    group_id=group_id,
                    label_reference=f"labels/{sample_id}.json",
                    input_references={"cube": f"cubes/{sample_id}.tiff"},
                    input_content_hashes={"cube": _INPUT_HASH},
                    label_schema_metadata={"dimension": 1},
                )
            )

    record_tuple = tuple(records)
    identity = {
        "records": record_tuple,
        "canonical_order": canonical_sample_order(
            tuple(record.sample_id for record in record_tuple)
        ),
    }
    manifest = SampleManifest(
        schema_version="1",
        dataset_key="property5.grouped-histology",
        records=record_tuple,
        canonical_order=identity["canonical_order"],
        dataset_hash=hash_object(identity),
        created_utc=_CREATED_UTC,
    )
    return _GroupedManifestCase(
        manifest=manifest,
        train_ids=tuple(train_ids),
        test_ids=tuple(test_ids),
        group_by_sample=group_by_sample,
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="property5-selection",
        objective_metric="balanced_accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def _plan_kwargs(fold: OuterFold) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol_key": "property5.outer-isolation.v1",
        "folds": (fold,),
        "selection_policy": _policy(),
        "seed_set": (42,),
        "evaluation_budget": EvaluationBudget(
            maximum_successful_evaluations=1
        ),
    }


def _groups(sample_ids: tuple[str, ...], group_by_sample: dict[str, str]) -> set[str]:
    return {group_by_sample[sample_id] for sample_id in sample_ids}


def _assert_execution_blocked(
    planner: ProtocolPlanner,
    manifest: SampleManifest,
    fold: OuterFold,
    expected_code: str,
    expected_ids: set[str],
    build_calls: list[str],
) -> None:
    with pytest.raises(ProtocolPlanningError) as caught:
        planner.plan_publish_and_build(
            manifest,
            model_builder=lambda protocol: build_calls.append(protocol.protocol_hash),
            **_plan_kwargs(fold),
        )

    matching = [
        record
        for record in caught.value.error_records
        if record.error_code == expected_code
    ]
    assert len(matching) == 1
    assert matching[0].affected_identifiers["fold_id"] == fold.fold_id
    observed_ids = set(matching[0].missing_ids) | set(matching[0].unexpected_ids)
    assert observed_ids == expected_ids
    assert build_calls == []


@settings(max_examples=100, deadline=None)
@given(case=_grouped_manifest_cases())
def test_outer_fold_sample_coverage_and_group_isolation(
    case: _GroupedManifestCase,
) -> None:
    """Valid group partitions pass; overlap, coverage, and group leaks block execution."""

    with TemporaryDirectory(prefix="property5-") as directory:
        artifact_root = Path(directory) / "artifacts"
        planner = ProtocolPlanner(FailureAtomicArtifactStore(artifact_root))
        valid_fold = OuterFold(
            fold_id="outer-valid",
            outer_train_ids=case.train_ids,
            outer_test_ids=case.test_ids,
        )

        protocol = planner.plan(case.manifest, **_plan_kwargs(valid_fold))
        planned_fold = protocol.folds[0]
        expected_ids = set(case.manifest.canonical_order)
        assert set(planned_fold.outer_train_ids).isdisjoint(
            planned_fold.outer_test_ids
        )
        assert (
            set(planned_fold.outer_train_ids) | set(planned_fold.outer_test_ids)
        ) == expected_ids
        assert _groups(
            planned_fold.outer_train_ids, case.group_by_sample
        ).isdisjoint(_groups(planned_fold.outer_test_ids, case.group_by_sample))

        build_calls: list[str] = []
        overlapping_id = case.train_ids[0]
        sample_overlap_fold = OuterFold(
            fold_id="outer-sample-overlap",
            outer_train_ids=case.train_ids,
            outer_test_ids=case.test_ids + (overlapping_id,),
        )
        _assert_execution_blocked(
            planner,
            case.manifest,
            sample_overlap_fold,
            "OUTER_SAMPLE_ISOLATION_FAILED",
            {overlapping_id},
            build_calls,
        )

        missing_id = case.train_ids[0]
        coverage_failure_fold = OuterFold(
            fold_id="outer-coverage-failure",
            outer_train_ids=case.train_ids[1:],
            outer_test_ids=case.test_ids,
        )
        _assert_execution_blocked(
            planner,
            case.manifest,
            coverage_failure_fold,
            "OUTER_COVERAGE_FAILED",
            {missing_id},
            build_calls,
        )

        train_crossing_id = case.train_ids[0]
        test_crossing_id = case.test_ids[0]
        group_overlap_fold = OuterFold(
            fold_id="outer-group-overlap",
            outer_train_ids=case.train_ids[1:] + (test_crossing_id,),
            outer_test_ids=case.test_ids[1:] + (train_crossing_id,),
        )
        _assert_execution_blocked(
            planner,
            case.manifest,
            group_overlap_fold,
            "OUTER_GROUP_ISOLATION_FAILED",
            {
                case.group_by_sample[train_crossing_id],
                case.group_by_sample[test_crossing_id],
            },
            build_calls,
        )

        assert not list(artifact_root.glob("protocols/**/*.json"))
