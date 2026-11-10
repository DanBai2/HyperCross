"""Targeted unit coverage for task 10.1 prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.metrics import (
    PredictionArtifactReloadError,
    PredictionArtifactValidationError,
    PredictionArtifactWriter,
    validate_prediction_records,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    PredictionShard,
    canonical_serialize,
    hash_file,
    hash_object,
)


def _record(
    sample_id: str,
    *,
    fold_id: str = "outer-1",
    seed: int = 42,
    branch: str = "primary",
    y_true: object = 1,
    y_pred: object = 1,
    probability: object = 0.8,
) -> PredictionRecord:
    return PredictionRecord(
        key=PredictionKey(
            sample_id=sample_id,
            fold_id=fold_id,
            seed=seed,
            inference_branch=branch,
        ),
        y_true=y_true,
        y_pred=y_pred,
        probability=probability,
    )


def _writer(tmp_path: Path) -> PredictionArtifactWriter:
    return PredictionArtifactWriter(
        FailureAtomicArtifactStore(tmp_path / "artifact root with spaces")
    )


def test_prediction_schema_records_full_key_truth_prediction_and_probability() -> None:
    record = PredictionRecord(
        key={
            "sample_id": "sample\\e\u0301",
            "fold_id": "outer-1",
            "seed": 7,
            "inference_branch": "pca-only",
        },
        target_name="three-target-output",
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.1, 1.9, 3.2],
        probability=[0.1, 0.7, 0.2],
    )

    assert record.key.sample_id == "sample/é"
    assert record.key.fold_id == "outer-1"
    assert record.key.seed == 7
    assert record.key.inference_branch == "pca-only"
    assert record.y_true == (1.0, 2.0, 3.0)
    assert record.y_pred == (1.1, 1.9, 3.2)
    assert record.probability == (0.1, 0.7, 0.2)


@pytest.mark.parametrize(
    "values",
    (
        {"probability": float("nan")},
        {"probability": 1.01},
        {"y_true": [1.0, 2.0], "y_pred": [1.0]},
        {"y_true": [1.0], "y_pred": 1.0},
    ),
)
def test_prediction_schema_rejects_non_finite_invalid_or_mismatched_values(
    values: dict[str, object],
) -> None:
    base = {
        "key": {
            "sample_id": "sample-1",
            "fold_id": "outer-1",
            "seed": 1,
            "inference_branch": "primary",
        },
        "y_true": 1,
        "y_pred": 1,
        "probability": 0.5,
    }

    with pytest.raises(ValidationError):
        PredictionRecord.model_validate({**base, **values})

    with pytest.raises(ValidationError, match="Extra inputs"):
        PredictionRecord.model_validate({**base, "undeclared": "field"})


def test_shards_round_trip_and_merge_to_one_atomic_complete_artifact(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    shard_b = PredictionShard(
        shard_id="batch-b",
        records=(_record("sample-3", y_true=0, y_pred=1),),
    )
    shard_a = PredictionShard(
        shard_id="batch-a",
        records=(_record("sample-2"), _record("sample-1")),
    )

    encoded_shard = writer.encode_shard(shard_a)
    assert writer.reload_shard(encoded_shard) == shard_a

    published = writer.publish_shards(
        artifact_id="prediction-run-1",
        run_id="run-1",
        expected_sample_ids=("sample-3", "sample-1", "sample-2"),
        shards=(shard_b, shard_a),
        relative_path=r"predictions\fold 1.canonical.json",
    )

    artifact = published.artifact
    target = tmp_path / "artifact root with spaces" / "predictions" / "fold 1.canonical.json"
    assert artifact.complete is True
    assert artifact.expected_sample_ids == ("sample-1", "sample-2", "sample-3")
    assert tuple(record.key.sample_id for record in artifact.records) == (
        "sample-1",
        "sample-2",
        "sample-3",
    )
    assert artifact.content_hash == hash_object(artifact.hash_payload())
    assert target.read_bytes() == canonical_serialize(artifact)
    assert published.reference.relative_path == "predictions/fold 1.canonical.json"
    assert published.reference.content_hash == hash_file(target)
    assert writer.reload(target.read_bytes()) == artifact


def test_publish_accepts_generator_shards_without_loading_outer_labels(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    def shards():
        yield (_record("sample-2", y_true=0, y_pred=0),)
        yield (_record("sample-1", y_true=1, y_pred=1),)

    published = writer.publish_shards(
        artifact_id="prediction-generator",
        run_id="run-generator",
        expected_sample_ids=("sample-1", "sample-2"),
        shards=shards(),
    )

    assert published.artifact.complete
    assert tuple(record.key.sample_id for record in published.artifact.records) == (
        "sample-1",
        "sample-2",
    )


def test_missing_unexpected_and_duplicate_predictions_block_before_publication(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    root = tmp_path / "artifact root with spaces"
    completed = root / "predictions" / "completed.canonical.json"
    completed.parent.mkdir(parents=True)
    completed.write_bytes(b"previous-completed-prediction")
    before = completed.read_bytes()

    duplicate = _record("sample-1")
    with pytest.raises(PredictionArtifactValidationError) as captured:
        writer.publish_shards(
            artifact_id="prediction-invalid",
            run_id="run-invalid",
            expected_sample_ids=("sample-1", "sample-2", "sample-3"),
            shards=((duplicate, duplicate), (_record("sample-extra"),)),
            relative_path="predictions/rejected.canonical.json",
        )

    error = captured.value.error_record
    assert {issue.code for issue in captured.value.report.errors} >= {
        "DUPLICATE_PREDICTION_KEYS",
        "DUPLICATE_PREDICTION_SAMPLE_IDS",
        "PREDICTION_COVERAGE_MISMATCH",
    }
    assert error.missing_ids == ("sample-2", "sample-3")
    assert error.unexpected_ids == ("sample-extra",)
    assert error.observed_value["duplicate_prediction_keys"]
    assert error.invariant_checks == {
        "prediction_artifact_unchanged": True,
        "aggregate_records_unchanged": True,
        "metrics_not_computed": True,
        "publication_target_not_opened": True,
    }
    assert not (root / "predictions" / "rejected.canonical.json").exists()
    assert completed.read_bytes() == before

    journal_lines = writer.error_journal.path.read_text(encoding="utf-8").splitlines()
    persisted = json.loads(journal_lines[-1])
    assert persisted["error_code"] == "PREDICTION_ARTIFACT_VALIDATION_FAILED"
    assert persisted["missing_ids"] == ["sample-2", "sample-3"]
    assert persisted["unexpected_ids"] == ["sample-extra"]


def test_context_mismatch_is_rejected_even_when_sample_coverage_matches() -> None:
    report = validate_prediction_records(
        (
            _record("sample-1", fold_id="outer-1", seed=1),
            _record("sample-2", fold_id="outer-2", seed=1),
        ),
        ("sample-1", "sample-2"),
    )

    assert not report.passes
    assert {issue.code for issue in report.errors} == {
        "PREDICTION_CONTEXT_MISMATCH"
    }


def test_complete_artifact_cannot_bypass_coverage_or_hash_validation() -> None:
    valid = PredictionArtifact.create(
        artifact_id="prediction-valid",
        run_id="run-valid",
        records=(_record("sample-1"),),
        expected_sample_ids=("sample-1",),
        complete=True,
    )

    with pytest.raises(ValidationError, match="content_hash"):
        PredictionArtifact.model_validate(
            {**valid.model_dump(mode="python"), "content_hash": "f" * 64}
        )

    with pytest.raises(ValidationError, match="sample coverage"):
        PredictionArtifact.create(
            artifact_id="prediction-incomplete",
            run_id="run-incomplete",
            records=(_record("sample-1"),),
            expected_sample_ids=("sample-1", "sample-2"),
            complete=True,
        )


def test_reload_rejects_incomplete_or_noncanonical_artifact_bytes(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    incomplete = PredictionArtifact.create(
        artifact_id="prediction-draft",
        run_id="run-draft",
        records=(_record("sample-1"),),
        expected_sample_ids=("sample-1",),
        complete=False,
    )

    with pytest.raises(PredictionArtifactReloadError, match="not marked complete"):
        writer.reload(canonical_serialize(incomplete))

    with pytest.raises(PredictionArtifactReloadError):
        writer.reload(b'{"ordinary":"json"}')
