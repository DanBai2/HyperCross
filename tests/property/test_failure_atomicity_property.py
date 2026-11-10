"""Property 38: failure atomicity and Error Record completeness.

**Validates: Requirements 1.28, 1.29, 2.11, 2.13, 2.15, 2.17, 2.19,
3.11, 3.13, 3.15, 3.17, 3.18, 4.12, 4.13, 4.27, 4.28, 5.17,
5.18, 6.29, 6.30, 16.48-16.50**
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given, strategies as st

from benchmark.recovery import ArtifactStoreError, FailureAtomicArtifactStore
from benchmark.registry import ErrorJournal
from benchmark.schema import ErrorRecord, hash_file


_FAILURE_KINDS = st.sampled_from(
    (
        "invalid_artifact_id",
        "invalid_expected_hash",
        "completed_target",
        "interrupted_write",
        "content_hash_mismatch",
    )
)
_SAFE_COMPONENTS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=24,
)
_NONEMPTY_BYTES = st.binary(min_size=1, max_size=256)
_MACHINE_VALUES = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=32)),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=12), children, max_size=4),
    ),
    max_leaves=10,
)


def _write_completed_state(root: Path, registry_bytes: bytes, artifact_bytes: bytes) -> None:
    completed = {
        "registry/run-registry.jsonl": registry_bytes,
        "manifests/completed.json": artifact_bytes,
        "checkpoints/completed.bin": b"checkpoint:" + artifact_bytes,
        "predictions/completed.json": b"prediction:" + artifact_bytes,
        "cache/completed.bin": b"cache:" + artifact_bytes,
    }
    for relative_path, content in completed.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _registry_snapshot(root: Path) -> bytes:
    return (root / "registry" / "run-registry.jsonl").read_bytes()


def _completed_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hash_file(path)
        for directory in ("manifests", "checkpoints", "predictions", "cache")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file()
    }


def _journal_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line.decode("utf-8"))
        for line in path.read_bytes().split(b"\n")
        if line
    ]


@given(
    failure_kind=_FAILURE_KINDS,
    artifact_suffix=_SAFE_COMPONENTS,
    registry_bytes=_NONEMPTY_BYTES,
    completed_bytes=_NONEMPTY_BYTES,
    attempted_bytes=_NONEMPTY_BYTES,
)
def test_validation_failures_preserve_registry_and_completed_artifacts(
    failure_kind: str,
    artifact_suffix: str,
    registry_bytes: bytes,
    completed_bytes: bytes,
    attempted_bytes: bytes,
) -> None:
    """Every generated example uses fresh filesystem state."""

    with TemporaryDirectory() as temporary_directory:
        _assert_validation_failure_is_atomic(
            Path(temporary_directory),
            failure_kind,
            artifact_suffix,
            registry_bytes,
            completed_bytes,
            attempted_bytes,
        )


def _assert_validation_failure_is_atomic(
    tmp_path: Path,
    failure_kind: str,
    artifact_suffix: str,
    registry_bytes: bytes,
    completed_bytes: bytes,
    attempted_bytes: bytes,
) -> None:
    """Every rejected publication leaves protected state unchanged and is auditable."""

    root = tmp_path / "artifact root"
    _write_completed_state(root, registry_bytes, completed_bytes)
    registry_before = _registry_snapshot(root)
    hashes_before = _completed_hashes(root)

    journal_path = tmp_path / "error journal" / "errors.jsonl"
    store = FailureAtomicArtifactStore(root, error_journal=ErrorJournal(journal_path))
    artifact_id = f"property38-{artifact_suffix}"
    relative_path = f"attempts/{artifact_suffix}.bin"
    expected_code: str
    expected_stage: str
    expected_field: str
    expected_invariants: dict[str, bool]

    if failure_kind == "invalid_artifact_id":
        artifact_id = f"{artifact_id}\x00"
        expected_code = "INVALID_ARTIFACT_ID"
        expected_stage = "request_validation"
        expected_field = "artifact_id"
        expected_invariants = {"temporary_file_created": False}
        operation = lambda: store.publish_bytes(  # noqa: E731
            artifact_id=artifact_id,
            relative_path=relative_path,
            content=attempted_bytes,
        )
    elif failure_kind == "invalid_expected_hash":
        expected_code = "INVALID_EXPECTED_CONTENT_HASH"
        expected_stage = "request_validation"
        expected_field = "expected_hash"
        expected_invariants = {"temporary_file_created": False}
        operation = lambda: store.publish_bytes(  # noqa: E731
            artifact_id=artifact_id,
            relative_path=relative_path,
            content=attempted_bytes,
            expected_hash="not-a-sha256",
        )
    elif failure_kind == "completed_target":
        relative_path = "manifests/completed.json"
        expected_code = "ARTIFACT_TARGET_ALREADY_COMPLETE"
        expected_stage = "completed_target_precheck"
        expected_field = "target_path"
        expected_invariants = {
            "completed_target_unchanged": True,
            "temporary_file_created": False,
        }
        operation = lambda: store.publish_bytes(  # noqa: E731
            artifact_id=artifact_id,
            relative_path=relative_path,
            content=attempted_bytes,
        )
    elif failure_kind == "interrupted_write":
        expected_code = "ARTIFACT_TEMP_WRITE_FAILED"
        expected_stage = "temporary_write"
        expected_field = "temporary_file"
        expected_invariants = {
            "completed_target_unchanged": True,
            "temporary_file_disposed": True,
            "temporary_file_quarantined": False,
        }

        def operation() -> object:
            def interrupted_writer(stream: Any) -> None:
                stream.write(attempted_bytes)
                raise RuntimeError("property38 injected write interruption")

            return store.publish_writer(
                artifact_id=artifact_id,
                relative_path=relative_path,
                expected_hash=hashlib.sha256(attempted_bytes).hexdigest(),
                writer=interrupted_writer,
            )

    else:
        expected_code = "ARTIFACT_HASH_MISMATCH"
        expected_stage = "hash_validation"
        expected_field = "content_hash"
        expected_invariants = {
            "completed_target_unchanged": True,
            "temporary_file_disposed": True,
            "temporary_file_quarantined": False,
        }
        wrong_hash = hashlib.sha256(attempted_bytes + b"different").hexdigest()
        operation = lambda: store.publish_bytes(  # noqa: E731
            artifact_id=artifact_id,
            relative_path=relative_path,
            content=attempted_bytes,
            expected_hash=wrong_hash,
        )

    with pytest.raises(ArtifactStoreError) as caught:
        operation()

    record = caught.value.error_record
    assert _registry_snapshot(root) == registry_before
    assert _completed_hashes(root) == hashes_before
    assert not list((root / "attempts").glob("*.tmp"))
    if failure_kind != "completed_target":
        assert not (root / relative_path).exists()

    assert record.error_code == expected_code
    assert record.failed_stage == expected_stage
    assert record.affected_identifiers == {
        "artifact_id": artifact_id,
        "relative_path": relative_path,
    }
    assert record.failing_field == expected_field
    assert record.expected_value is not None
    assert record.observed_value is not None
    assert record.invariant_checks == expected_invariants

    persisted = _journal_records(journal_path)
    assert len(persisted) == 1
    assert persisted[0]["error_id"] == str(record.error_id)
    assert persisted[0]["error_code"] == record.error_code
    assert persisted[0]["failed_stage"] == record.failed_stage
    assert persisted[0]["affected_identifiers"] == record.affected_identifiers
    assert persisted[0]["failing_field"] == record.failing_field
    assert persisted[0]["expected_value"] == record.expected_value
    assert persisted[0]["observed_value"] == record.observed_value
    assert persisted[0]["invariant_checks"] == record.invariant_checks


@given(
    source_path=st.text(min_size=1, max_size=64).filter(lambda value: bool(value.strip())),
    source_bytes=_NONEMPTY_BYTES,
    fold_id=_SAFE_COMPONENTS,
    affected_ids=st.sets(_SAFE_COMPONENTS, min_size=1, max_size=6),
    expected_value=_MACHINE_VALUES,
    observed_value=_MACHINE_VALUES,
)
def test_complete_validation_diagnostics_round_trip_without_business_state_mutation(
    source_path: str,
    source_bytes: bytes,
    fold_id: str,
    affected_ids: set[str],
    expected_value: object,
    observed_value: object,
) -> None:
    """Every generated example uses fresh filesystem state."""

    with TemporaryDirectory() as temporary_directory:
        _assert_validation_diagnostics_round_trip(
            Path(temporary_directory),
            source_path,
            source_bytes,
            fold_id,
            affected_ids,
            expected_value,
            observed_value,
        )


def _assert_validation_diagnostics_round_trip(
    tmp_path: Path,
    source_path: str,
    source_bytes: bytes,
    fold_id: str,
    affected_ids: set[str],
    expected_value: object,
    observed_value: object,
) -> None:
    """Full boundary diagnostics survive append-only persistence exactly."""

    root = tmp_path / "protected state"
    _write_completed_state(root, b'{"run_id":"complete"}\n', b"completed")
    registry_before = _registry_snapshot(root)
    hashes_before = _completed_hashes(root)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    sorted_ids = tuple(sorted(affected_ids))
    journal_path = tmp_path / "errors.jsonl"

    record = ErrorRecord(
        error_code="VALIDATION_BOUNDARY_REJECTED",
        failed_stage="pre_mutation_validation",
        affected_identifiers={
            "fold_id": fold_id,
            "artifact_id": "property38-existing-artifact",
        },
        failing_field="sample_ids",
        expected_value=expected_value,
        observed_value=observed_value,
        missing_ids=sorted_ids,
        unexpected_ids=tuple(reversed(sorted_ids)),
        invariant_checks={
            "registry_snapshot_unchanged": True,
            "completed_hashes_unchanged": True,
            "mutation_started": False,
        },
        source_path=source_path,
        source_content_hash=source_hash,
    )
    ErrorJournal(journal_path).append(record)

    assert _registry_snapshot(root) == registry_before
    assert _completed_hashes(root) == hashes_before

    persisted = _journal_records(journal_path)
    assert len(persisted) == 1
    assert persisted[0] == record.model_dump(mode="json")
    assert persisted[0]["failed_stage"] == "pre_mutation_validation"
    assert persisted[0]["affected_identifiers"]["fold_id"] == fold_id
    assert persisted[0]["failing_field"] == "sample_ids"
    assert persisted[0]["expected_value"] == expected_value
    assert persisted[0]["observed_value"] == observed_value
    assert persisted[0]["missing_ids"] == list(sorted_ids)
    assert persisted[0]["unexpected_ids"] == list(reversed(sorted_ids))
    assert persisted[0]["source_path"] == source_path
    assert persisted[0]["source_content_hash"] == source_hash
    assert persisted[0]["invariant_checks"] == {
        "registry_snapshot_unchanged": True,
        "completed_hashes_unchanged": True,
        "mutation_started": False,
    }
