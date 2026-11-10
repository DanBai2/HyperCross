"""Targeted unit coverage for task 3.2 split planning and validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.protocols import ProtocolPlanner, ProtocolPlanningError
from benchmark.recovery import ArtifactStoreError, FailureAtomicArtifactStore
from benchmark.schema import (
    EvaluationBudget,
    InnerFold,
    MetricDirection,
    OuterFold,
    ProtocolManifest,
    SampleManifest,
    SampleRecord,
    SelectionPolicy,
    canonical_sample_order,
    canonical_serialize,
)


DATASET_HASH = "a" * 64
INPUT_HASH = "b" * 64


def _sample_manifest() -> SampleManifest:
    records = tuple(
        SampleRecord(
            sample_id=sample_id,
            group_id=group_id,
            label_reference=f"labels/{sample_id}.json",
            input_references={"cube": f"cubes/{sample_id}.tiff"},
            input_content_hashes={"cube": INPUT_HASH},
            label_schema_metadata={"dimension": 1},
        )
        for sample_id, group_id in (
            ("sample-1", "patient-a"),
            ("sample-2", "patient-a"),
            ("sample-3", "patient-b"),
            ("sample-4", "patient-b"),
            ("sample-5", "patient-c"),
            ("sample-6", "patient-c"),
        )
    )
    return SampleManifest(
        schema_version="1",
        dataset_key="histology.synthetic",
        records=records,
        canonical_order=canonical_sample_order(
            tuple(record.sample_id for record in records)
        ),
        dataset_hash=DATASET_HASH,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="patient-inner-v1",
        objective_metric="balanced_accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def _valid_fold() -> OuterFold:
    return OuterFold(
        fold_id="outer-c",
        outer_train_ids=("sample-4", "sample-1", "sample-3", "sample-2"),
        outer_test_ids=("sample-6", "sample-5"),
        inner_folds=(
            InnerFold(
                fold_id="inner-a-b",
                train_ids=("sample-2", "sample-1"),
                validation_ids=("sample-4", "sample-3"),
            ),
        ),
    )


def _planner(tmp_path: Path) -> ProtocolPlanner:
    return ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts"))


def _plan_kwargs(fold: OuterFold) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol_key": "hist.patient-nested.synthetic.v1",
        "folds": (fold,),
        "selection_policy": _policy(),
        "seed_set": (42, 7),
        "evaluation_budget": EvaluationBudget(
            maximum_successful_evaluations=1
        ),
    }


def test_valid_plan_is_canonical_and_atomically_published_before_model_build(
    tmp_path: Path,
) -> None:
    planner = _planner(tmp_path)
    artifact_root = tmp_path / "artifacts"
    build_observations: list[bool] = []

    def build_model(protocol: ProtocolManifest) -> str:
        expected_path = artifact_root / Path(
            f"protocols/{protocol.protocol_hash}.canonical.json"
        )
        build_observations.append(expected_path.is_file())
        assert expected_path.read_bytes() == canonical_serialize(protocol)
        return "built-model"

    published, model = planner.plan_publish_and_build(
        _sample_manifest(),
        model_builder=build_model,
        **_plan_kwargs(_valid_fold()),
    )

    assert model == "built-model"
    assert build_observations == [True]
    assert published.artifact.content_hash
    assert published.manifest.folds[0].outer_train_ids == (
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
    )
    assert published.manifest.seed_set == (7, 42)


@pytest.mark.parametrize(
    ("fold", "expected_code"),
    (
        (
            OuterFold(
                fold_id="outer-overlap",
                outer_train_ids=(
                    "sample-1",
                    "sample-2",
                    "sample-3",
                    "sample-4",
                    "sample-5",
                ),
                outer_test_ids=("sample-5", "sample-6"),
            ),
            "OUTER_SAMPLE_ISOLATION_FAILED",
        ),
        (
            OuterFold(
                fold_id="outer-group-overlap",
                outer_train_ids=("sample-1", "sample-3", "sample-4", "sample-5"),
                outer_test_ids=("sample-2", "sample-6"),
            ),
            "OUTER_GROUP_ISOLATION_FAILED",
        ),
        (
            OuterFold(
                fold_id="outer-missing",
                outer_train_ids=("sample-1", "sample-2", "sample-3"),
                outer_test_ids=("sample-5", "sample-6"),
            ),
            "OUTER_COVERAGE_FAILED",
        ),
        (
            OuterFold(
                fold_id="outer-inner-out-of-bound",
                outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-4"),
                outer_test_ids=("sample-5", "sample-6"),
                inner_folds=(
                    InnerFold(
                        fold_id="inner-invalid",
                        train_ids=("sample-1", "sample-2"),
                        validation_ids=("sample-5",),
                    ),
                ),
            ),
            "INNER_CONTAINMENT_FAILED",
        ),
        (
            OuterFold(
                fold_id="outer-inner-sample-overlap",
                outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-4"),
                outer_test_ids=("sample-5", "sample-6"),
                inner_folds=(
                    InnerFold(
                        fold_id="inner-invalid",
                        train_ids=("sample-1", "sample-2"),
                        validation_ids=("sample-2", "sample-3", "sample-4"),
                    ),
                ),
            ),
            "INNER_SAMPLE_ISOLATION_FAILED",
        ),
        (
            OuterFold(
                fold_id="outer-inner-group-overlap",
                outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-4"),
                outer_test_ids=("sample-5", "sample-6"),
                inner_folds=(
                    InnerFold(
                        fold_id="inner-invalid",
                        train_ids=("sample-1", "sample-3"),
                        validation_ids=("sample-2", "sample-4"),
                    ),
                ),
            ),
            "INNER_GROUP_ISOLATION_FAILED",
        ),
    ),
)
def test_invalid_split_invariant_blocks_protocol_publication(
    tmp_path: Path,
    fold: OuterFold,
    expected_code: str,
) -> None:
    planner = _planner(tmp_path)

    with pytest.raises(ProtocolPlanningError) as captured:
        planner.plan_and_publish(
            _sample_manifest(),
            **_plan_kwargs(fold),
        )

    codes = {issue.code for issue in captured.value.report.errors}
    assert expected_code in codes
    assert expected_code in {
        record.error_code for record in captured.value.error_records
    }
    assert not list((tmp_path / "artifacts").glob("protocols/**/*.json"))


def test_coverage_error_records_fold_missing_and_unexpected_sample_ids(
    tmp_path: Path,
) -> None:
    planner = _planner(tmp_path)
    fold = OuterFold(
        fold_id="outer-coverage",
        outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-extra"),
        outer_test_ids=("sample-5", "sample-6"),
    )

    with pytest.raises(ProtocolPlanningError) as captured:
        planner.plan(_sample_manifest(), **_plan_kwargs(fold))

    error = next(
        record
        for record in captured.value.error_records
        if record.error_code == "OUTER_COVERAGE_FAILED"
    )
    assert error.affected_identifiers == {
        "protocol_key": "hist.patient-nested.synthetic.v1",
        "fold_id": "outer-coverage",
    }
    assert error.missing_ids == ("sample-4",)
    assert error.unexpected_ids == ("sample-extra",)
    assert error.failed_stage == "protocol_validation"
    assert error.invariant_checks["protocol_artifact_unchanged"] is True


def test_validation_failure_preserves_existing_completed_artifact(
    tmp_path: Path,
) -> None:
    planner = _planner(tmp_path)
    completed = tmp_path / "artifacts" / "protocols" / "completed.json"
    completed.parent.mkdir(parents=True)
    completed.write_bytes(b"completed-protocol")
    before = completed.read_bytes()
    invalid_fold = OuterFold(
        fold_id="outer-overlap",
        outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-4"),
        outer_test_ids=("sample-4", "sample-5", "sample-6"),
    )

    with pytest.raises(ProtocolPlanningError):
        planner.plan_and_publish(
            _sample_manifest(),
            **_plan_kwargs(invalid_fold),
        )

    assert completed.read_bytes() == before


def test_disjointness_and_containment_errors_report_exact_offending_ids(
    tmp_path: Path,
) -> None:
    """Requirements 3.10-3.17: failures identify the fold and offending IDs."""

    cases = (
        (
            OuterFold(
                fold_id="outer-sample-overlap",
                outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-4"),
                outer_test_ids=("sample-4", "sample-5", "sample-6"),
            ),
            "OUTER_SAMPLE_ISOLATION_FAILED",
            {"protocol_key": "hist.patient-nested.synthetic.v1", "fold_id": "outer-sample-overlap"},
            ("sample-4",),
        ),
        (
            OuterFold(
                fold_id="outer-group-overlap",
                outer_train_ids=("sample-1", "sample-3", "sample-4", "sample-5"),
                outer_test_ids=("sample-2", "sample-6"),
            ),
            "OUTER_GROUP_ISOLATION_FAILED",
            {"protocol_key": "hist.patient-nested.synthetic.v1", "fold_id": "outer-group-overlap"},
            ("patient-a", "patient-c"),
        ),
        (
            OuterFold(
                fold_id="outer-inner-boundary",
                outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-4"),
                outer_test_ids=("sample-5", "sample-6"),
                inner_folds=(
                    InnerFold(
                        fold_id="inner-out-of-bound",
                        train_ids=("sample-1", "sample-2"),
                        validation_ids=("sample-5",),
                    ),
                ),
            ),
            "INNER_CONTAINMENT_FAILED",
            {
                "protocol_key": "hist.patient-nested.synthetic.v1",
                "fold_id": "outer-inner-boundary",
                "inner_fold_id": "inner-out-of-bound",
            },
            ("sample-5",),
        ),
    )

    for fold, expected_code, affected_identifiers, offending_ids in cases:
        planner = _planner(tmp_path / fold.fold_id)
        with pytest.raises(ProtocolPlanningError) as captured:
            planner.plan(_sample_manifest(), **_plan_kwargs(fold))

        error = next(
            record
            for record in captured.value.error_records
            if record.error_code == expected_code
        )
        assert error.affected_identifiers == affected_identifiers
        assert error.unexpected_ids == offending_ids
        assert error.failed_stage == "protocol_validation"


def test_round_trip_preserves_all_fold_memberships_and_protocol_hash(
    tmp_path: Path,
) -> None:
    """Requirements 3.19-3.21 and 16.11: canonical reload has no drift."""

    planner = _planner(tmp_path)
    published = planner.plan_and_publish(
        _sample_manifest(),
        **_plan_kwargs(_valid_fold()),
    )
    artifact_path = (
        tmp_path
        / "artifacts"
        / Path(*published.artifact.relative_path.split("/"))
    )

    reloaded = planner.reload(artifact_path.read_bytes())

    assert reloaded.protocol_hash == published.manifest.protocol_hash
    assert reloaded.folds == published.manifest.folds
    assert reloaded.folds[0].outer_train_ids == published.manifest.folds[0].outer_train_ids
    assert reloaded.folds[0].outer_test_ids == published.manifest.folds[0].outer_test_ids
    assert reloaded.folds[0].inner_folds == published.manifest.folds[0].inner_folds


def test_model_builder_is_not_called_when_protocol_validation_fails(
    tmp_path: Path,
) -> None:
    """Requirements 3.1 and 3.10: invalid plans block model construction."""

    planner = _planner(tmp_path)
    build_calls: list[str] = []
    invalid_fold = OuterFold(
        fold_id="outer-invalid",
        outer_train_ids=("sample-1", "sample-2", "sample-3", "sample-4"),
        outer_test_ids=("sample-4", "sample-5", "sample-6"),
    )

    with pytest.raises(ProtocolPlanningError):
        planner.plan_publish_and_build(
            _sample_manifest(),
            model_builder=lambda protocol: build_calls.append(protocol.protocol_hash),
            **_plan_kwargs(invalid_fold),
        )

    assert build_calls == []
    assert not list((tmp_path / "artifacts").glob("protocols/**/*.json"))


def test_model_builder_is_not_called_until_protocol_publication_succeeds(
    tmp_path: Path,
) -> None:
    """Requirement 3.1: a failed publication cannot invoke model construction."""

    planner = _planner(tmp_path)
    protocol = planner.plan(_sample_manifest(), **_plan_kwargs(_valid_fold()))
    target = (
        tmp_path
        / "artifacts"
        / "protocols"
        / f"{protocol.protocol_hash}.canonical.json"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already-complete")
    build_calls: list[str] = []

    with pytest.raises(ArtifactStoreError) as captured:
        planner.plan_publish_and_build(
            _sample_manifest(),
            model_builder=lambda manifest: build_calls.append(manifest.protocol_hash),
            **_plan_kwargs(_valid_fold()),
        )

    assert captured.value.error_record.error_code == "ARTIFACT_TARGET_ALREADY_COMPLETE"
    assert build_calls == []
    assert target.read_bytes() == b"already-complete"
