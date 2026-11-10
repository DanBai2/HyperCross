"""Focused unit coverage for task 1.4 failure-atomic storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.recovery import ArtifactStoreError, FailureAtomicArtifactStore
from benchmark.registry import ErrorJournal
from benchmark.schema import ErrorRecord


def _record(code: str) -> ErrorRecord:
    return ErrorRecord(
        error_code=code,
        failed_stage="unit_test",
        failing_field="fixture",
    )


def _journal_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_error_journal_appends_without_rewriting_existing_records(tmp_path: Path) -> None:
    path = tmp_path / "journal with spaces" / "errors.jsonl"
    journal = ErrorJournal(path)

    journal.append(_record("FIRST_FAILURE"))
    first_bytes = path.read_bytes()
    journal.append(_record("SECOND_FAILURE"))

    assert journal.path == path
    assert path.read_bytes().startswith(first_bytes)
    assert [record["error_code"] for record in _journal_records(path)] == [
        "FIRST_FAILURE",
        "SECOND_FAILURE",
    ]


def test_publish_bytes_flushes_hash_checks_and_atomically_publishes(tmp_path: Path) -> None:
    root = tmp_path / "artifact root with spaces"
    store = FailureAtomicArtifactStore(root)
    content = b"immutable prediction bytes"

    reference = store.publish_bytes(
        artifact_id="prediction-fold-1",
        relative_path=r"predictions\fold 1.bin",
        content=content,
        media_type="application/octet-stream",
    )

    target = root / "predictions" / "fold 1.bin"
    assert target.read_bytes() == content
    assert reference.relative_path == "predictions/fold 1.bin"
    assert reference.content_hash == hashlib.sha256(content).hexdigest()
    assert not list((root / "predictions").glob("*.tmp"))


def test_windows_and_parent_traversal_paths_are_rejected_before_staging(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    store = FailureAtomicArtifactStore(root)

    for unsafe_path in (
        r"C:\private\result.bin",
        r"..\outside.bin",
        "../outside.bin",
        "/absolute/result.bin",
    ):
        with pytest.raises(ValueError):
            store.publish_bytes(
                artifact_id="unsafe-path",
                relative_path=unsafe_path,
                content=b"must not be written",
            )

    assert list(root.rglob("*.bin")) == []
    assert list(root.rglob("*.tmp")) == []


def test_completed_target_is_rejected_before_staging_and_remains_unchanged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    target = root / "manifests" / "dataset.json"
    target.parent.mkdir(parents=True)
    original = b"completed artifact"
    target.write_bytes(original)
    journal_path = tmp_path / "errors.jsonl"
    store = FailureAtomicArtifactStore(root, error_journal=ErrorJournal(journal_path))

    with pytest.raises(ArtifactStoreError) as caught:
        store.publish_bytes(
            artifact_id="dataset-v1",
            relative_path="manifests/dataset.json",
            content=b"replacement must not win",
        )

    assert caught.value.error_record.error_code == "ARTIFACT_TARGET_ALREADY_COMPLETE"
    assert caught.value.error_record.failed_stage == "completed_target_precheck"
    assert caught.value.error_record.invariant_checks["temporary_file_created"] is False
    assert target.read_bytes() == original
    assert not list(target.parent.glob("*.tmp"))
    assert _journal_records(journal_path)[0]["error_code"] == caught.value.error_record.error_code


def test_hash_mismatch_removes_only_current_temp_and_preserves_completed_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    completed = root / "predictions" / "completed.bin"
    completed.parent.mkdir(parents=True)
    completed.write_bytes(b"old completed bytes")
    journal_path = tmp_path / "errors.jsonl"
    store = FailureAtomicArtifactStore(root, error_journal=ErrorJournal(journal_path))

    with pytest.raises(ArtifactStoreError) as caught:
        store.publish_bytes(
            artifact_id="new-artifact",
            relative_path="predictions/new.bin",
            content=b"new staged bytes",
            expected_hash="0" * 64,
        )

    assert caught.value.error_record.error_code == "ARTIFACT_HASH_MISMATCH"
    assert caught.value.error_record.failing_field == "content_hash"
    assert caught.value.error_record.invariant_checks["temporary_file_disposed"] is True
    assert completed.read_bytes() == b"old completed bytes"
    assert not (root / "predictions" / "new.bin").exists()
    assert not list((root / "predictions").glob("*.tmp"))


def test_interrupted_writer_is_recorded_and_can_quarantine_only_its_temp(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    journal_path = tmp_path / "errors.jsonl"
    store = FailureAtomicArtifactStore(
        root,
        error_journal=ErrorJournal(journal_path),
        quarantine_failed_temps=True,
    )

    completed = root / "manifests" / "completed.json"
    completed.parent.mkdir(parents=True)
    completed_bytes = b"previously completed"
    completed.write_bytes(completed_bytes)

    def interrupted_writer(stream) -> None:
        stream.write(b"partial")
        raise RuntimeError("simulated interruption")

    with pytest.raises(ArtifactStoreError) as caught:
        store.publish_writer(
            artifact_id="interrupted-artifact",
            relative_path="aggregates/result.json",
            expected_hash=hashlib.sha256(b"complete").hexdigest(),
            writer=interrupted_writer,
        )

    record = caught.value.error_record
    assert record.error_code == "ARTIFACT_TEMP_WRITE_FAILED"
    assert record.invariant_checks["temporary_file_disposed"] is True
    assert record.invariant_checks["temporary_file_quarantined"] is True
    assert not (root / "aggregates" / "result.json").exists()
    assert completed.read_bytes() == completed_bytes
    quarantined = list((root / ".quarantine").glob("*.tmp"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"partial"


def test_windows_replace_sharing_violation_has_stable_error_code_and_no_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    journal_path = tmp_path / "errors.jsonl"
    store = FailureAtomicArtifactStore(root, error_journal=ErrorJournal(journal_path))

    def sharing_violation(source: Path, target: Path) -> Path:
        error = OSError("file is used by another process")
        error.winerror = 32  # type: ignore[attr-defined]
        raise error

    monkeypatch.setattr(Path, "replace", sharing_violation)

    with pytest.raises(ArtifactStoreError) as caught:
        store.publish_bytes(
            artifact_id="windows-locked-artifact",
            relative_path="publication/table.json",
            content=b"publication bytes",
        )

    record = caught.value.error_record
    assert record.error_code == "WINDOWS_ATOMIC_REPLACE_SHARING_VIOLATION"
    assert record.failed_stage == "atomic_publish"
    assert record.observed_value["winerror"] == 32
    assert record.invariant_checks["temporary_file_disposed"] is True
    assert not (root / "publication" / "table.json").exists()
    assert not list((root / "publication").glob("*.tmp"))
