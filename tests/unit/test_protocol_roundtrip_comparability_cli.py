"""Targeted unit coverage for task 3.3 protocol reuse and dry-run APIs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.cli import main
from benchmark.protocols import (
    ProtocolComparabilityError,
    ProtocolPlanner,
    ProtocolReloadError,
)
from benchmark.recovery import FailureAtomicArtifactStore
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


def _fold() -> OuterFold:
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


def _plan_kwargs() -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol_key": "hist.patient-nested.synthetic.v1",
        "folds": (_fold(),),
        "selection_policy": _policy(),
        "seed_set": (42, 7),
        "evaluation_budget": EvaluationBudget(
            maximum_successful_evaluations=1
        ),
    }


def _protocol() -> ProtocolManifest:
    return ProtocolManifest.create(
        dataset_hash=DATASET_HASH,
        **_plan_kwargs(),
    )


def test_published_protocol_round_trip_preserves_memberships_and_hash(
    tmp_path: Path,
) -> None:
    planner = ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts"))
    published = planner.dry_run(_sample_manifest(), **_plan_kwargs())
    artifact_path = (
        tmp_path / "artifacts" / Path(*published.artifact.relative_path.split("/"))
    )

    reloaded = planner.reload(artifact_path.read_bytes())

    assert reloaded == published.manifest
    assert reloaded.protocol_hash == published.manifest.protocol_hash
    assert reloaded.folds == published.manifest.folds
    assert canonical_serialize(reloaded) == artifact_path.read_bytes()


def test_reload_rejects_noncanonical_or_hash_tampered_protocol_bytes() -> None:
    protocol = _protocol()
    with pytest.raises(ProtocolReloadError, match="exact canonical"):
        ProtocolPlanner.reload(canonical_serialize(protocol) + b" ")

    tampered = protocol.model_copy(update={"protocol_hash": "0" * 64})
    with pytest.raises(ProtocolReloadError, match="schema or hash"):
        ProtocolPlanner.reload(canonical_serialize(tampered))


def test_comparability_requires_hash_and_fold_memberships_independently() -> None:
    reference = _protocol()
    same_memberships_new_identity = ProtocolManifest.create(
        dataset_hash=DATASET_HASH,
        **{**_plan_kwargs(), "protocol_key": "hist.changed.v1"},
    )
    hash_report = ProtocolPlanner.validate_comparability(
        reference,
        same_memberships_new_identity,
    )
    assert {issue.code for issue in hash_report.errors} == {
        "PROTOCOL_COMPARISON_HASH_MISMATCH"
    }

    changed_fold = OuterFold(
        fold_id="outer-b",
        outer_train_ids=("sample-1", "sample-2", "sample-5", "sample-6"),
        outer_test_ids=("sample-3", "sample-4"),
    )
    same_hash_changed_memberships = reference.model_copy(update={"folds": (changed_fold,)})
    membership_report = ProtocolPlanner.validate_comparability(
        reference,
        same_hash_changed_memberships,
    )
    assert {issue.code for issue in membership_report.errors} == {
        "PROTOCOL_COMPARISON_FOLD_MEMBERSHIP_MISMATCH"
    }

    with pytest.raises(ProtocolComparabilityError):
        ProtocolPlanner.require_comparable(reference, same_hash_changed_memberships)
    ProtocolPlanner.require_comparable(reference, ProtocolPlanner.reload(canonical_serialize(reference)))


def test_protocol_dry_run_cli_handles_spaced_paths_without_model_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    working = tmp_path / "directory with spaces"
    working.mkdir()
    manifest_path = working / "sample manifest.json"
    spec_path = working / "protocol spec.json"
    output_root = working / "published artifacts"
    manifest_path.write_text(
        json.dumps(_sample_manifest().model_dump(mode="json")),
        encoding="utf-8",
    )
    spec_path.write_text(
        json.dumps(
            _protocol().model_dump(
                mode="json",
                exclude={"dataset_hash", "protocol_hash"},
            )
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "protocol-dry-run",
            "--sample-manifest",
            str(manifest_path),
            "--protocol-spec",
            str(spec_path),
            "--output-root",
            str(output_root),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert result == 0
    assert summary["model_built"] is False
    assert summary["fold_ids"] == ["outer-c"]
    artifact_path = output_root / Path(*summary["artifact"]["relative_path"].split("/"))
    assert artifact_path.is_file()
    assert ProtocolPlanner.reload(artifact_path.read_bytes()).protocol_hash == summary["protocol_hash"]
