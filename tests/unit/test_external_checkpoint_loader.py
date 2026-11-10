"""Targeted tests for task 6.4 controlled external checkpoint loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.models import ExternalCheckpointLoader, ExternalCheckpointLoadError
from benchmark.registry import ErrorJournal


torch = pytest.importorskip("torch")
SOURCE_REFERENCE = "https://example.org/releases/model-v1/checkpoint.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_marker(path: str) -> None:
    Path(path).write_text("unsafe deserialization executed", encoding="utf-8")


class _UnsafeCheckpointPayload:
    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self):
        return _write_marker, (str(self._marker),)


def test_matching_source_and_hash_load_weights_only_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "external checkpoint.pt"
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0, 2.0, 3.0])},
            "epoch": 7,
        },
        checkpoint,
    )
    original = checkpoint.read_bytes()

    loaded = ExternalCheckpointLoader().load(
        checkpoint,
        source_reference=SOURCE_REFERENCE,
        expected_content_hash=_sha256(checkpoint).upper(),
    )

    assert loaded["epoch"] == 7
    assert torch.equal(loaded["model"]["weight"], torch.tensor([1.0, 2.0, 3.0]))
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("source_reference", (None, ""))
def test_missing_source_reference_blocks_before_deserialization(
    tmp_path: Path,
    source_reference: str | None,
) -> None:
    checkpoint = tmp_path / "not-a-serialized-checkpoint.pt"
    original = b"these bytes would fail deserialization"
    checkpoint.write_bytes(original)

    with pytest.raises(ExternalCheckpointLoadError) as caught:
        ExternalCheckpointLoader().load(
            checkpoint,
            source_reference=source_reference,
            expected_content_hash=hashlib.sha256(original).hexdigest(),
        )

    error = caught.value.error_record
    assert error.error_code == "EXTERNAL_CHECKPOINT_SOURCE_REFERENCE_MISSING"
    assert error.failed_stage == "checkpoint_validation"
    assert error.failing_field == "source_reference"
    assert error.affected_identifiers == {
        "checkpoint_reference": str(checkpoint),
    }
    assert error.invariant_checks["deserialization_not_started"] is True
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("expected_hash", (None, ""))
def test_missing_content_hash_blocks_before_checkpoint_is_opened(
    tmp_path: Path,
    expected_hash: str | None,
) -> None:
    missing_checkpoint = tmp_path / "checkpoint-does-not-exist.pt"

    with pytest.raises(ExternalCheckpointLoadError) as caught:
        ExternalCheckpointLoader().load(
            missing_checkpoint,
            source_reference=SOURCE_REFERENCE,
            expected_content_hash=expected_hash,
        )

    error = caught.value.error_record
    assert error.error_code == "EXTERNAL_CHECKPOINT_CONTENT_HASH_MISSING"
    assert error.failing_field == "content_hash"
    assert error.invariant_checks["deserialization_not_started"] is True


def test_hash_mismatch_is_journaled_before_deserialization_and_preserves_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "external.pt"
    checkpoint_bytes = b"not a checkpoint because hash validation must win"
    checkpoint.write_bytes(checkpoint_bytes)
    completed_artifact = tmp_path / "completed-prediction.json"
    completed_bytes = b'{"status":"complete"}'
    completed_artifact.write_bytes(completed_bytes)
    journal = ErrorJournal(tmp_path / "errors" / "checkpoint-errors.jsonl")
    deserialization_attempted = False

    def record_unexpected_deserialization(*args, **kwargs):
        nonlocal deserialization_attempted
        deserialization_attempted = True
        raise AssertionError("deserialization must not run before hash validation")

    monkeypatch.setattr(
        ExternalCheckpointLoader,
        "_restricted_torch_load",
        staticmethod(record_unexpected_deserialization),
    )

    with pytest.raises(ExternalCheckpointLoadError) as caught:
        ExternalCheckpointLoader(journal).load(
            checkpoint,
            source_reference=SOURCE_REFERENCE,
            expected_content_hash="0" * 64,
        )

    error = caught.value.error_record
    observed_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    assert error.error_code == "EXTERNAL_CHECKPOINT_CONTENT_HASH_MISMATCH"
    assert error.failed_stage == "checkpoint_validation"
    assert error.failing_field == "content_hash"
    assert error.expected_value == "0" * 64
    assert error.observed_value == observed_hash
    assert error.source_content_hash == observed_hash
    assert error.affected_identifiers == {
        "checkpoint_reference": str(checkpoint),
        "source_reference": SOURCE_REFERENCE,
    }
    assert error.invariant_checks == {
        "deserialization_not_started": True,
        "restricted_loading_required": True,
        "checkpoint_bytes_unchanged": True,
        "completed_artifacts_unchanged": True,
    }
    assert deserialization_attempted is False
    persisted = json.loads(journal.path.read_text(encoding="utf-8"))
    assert persisted["error_code"] == error.error_code
    assert persisted["expected_value"] == "0" * 64
    assert persisted["observed_value"] == observed_hash
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert completed_artifact.read_bytes() == completed_bytes


def test_restricted_loader_rejects_unsafe_pickle_global_without_executing_it(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "unsafe-marker.txt"
    checkpoint = tmp_path / "unsafe-external.pt"
    torch.save(_UnsafeCheckpointPayload(marker), checkpoint)
    original = checkpoint.read_bytes()

    with pytest.raises(ExternalCheckpointLoadError) as caught:
        ExternalCheckpointLoader().load(
            checkpoint,
            source_reference=SOURCE_REFERENCE,
            expected_content_hash=_sha256(checkpoint),
        )

    error = caught.value.error_record
    assert error.error_code == "EXTERNAL_CHECKPOINT_DESERIALIZATION_FAILED"
    assert error.failed_stage == "checkpoint_deserialization"
    assert error.failing_field == "restricted_load"
    assert error.invariant_checks["restricted_loading_required"] is True
    assert error.invariant_checks["deserialization_not_started"] is False
    assert not marker.exists()
    assert checkpoint.read_bytes() == original
