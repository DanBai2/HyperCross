"""Failure-atomic publication of immutable benchmark artifacts."""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Callable, Optional, Union

from pydantic import TypeAdapter, ValidationError

from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import ArtifactId, ContentHash
from benchmark.schema.canonical import hash_file
from benchmark.schema.records import ArtifactReference, ErrorRecord


ArtifactWriter = Callable[[BinaryIO], None]
_WINDOWS_SHARING_VIOLATIONS = {32, 33}
_WINDOWS_ACCESS_DENIED = 5


class ArtifactStoreError(RuntimeError):
    """Publication failure carrying the exact machine-readable error record."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        super().__init__(
            f"{error_record.error_code}: artifact publication failed at "
            f"{error_record.failed_stage} ({error_record.failing_field})"
        )


class _PublishFailure(Exception):
    def __init__(
        self,
        *,
        code: str,
        field: str,
        expected: object = None,
        observed: object = None,
    ) -> None:
        self.code = code
        self.field = field
        self.expected = expected
        self.observed = observed
        super().__init__(code)


class FailureAtomicArtifactStore:
    """Publish immutable files through same-directory staging and replacement.

    A destination that already exists is treated as a completed artifact and is
    rejected before any write handle is opened. Temporary files are allocated
    in the destination directory, flushed, synchronized, and hash-checked before
    ``Path.replace`` performs the same-filesystem atomic publication step.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        error_journal: Optional[ErrorJournal] = None,
        quarantine_failed_temps: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.error_journal = error_journal or ErrorJournal(
            self.root / ".errors" / "error-journal.jsonl"
        )
        self.quarantine_failed_temps = quarantine_failed_temps

    def publish_bytes(
        self,
        *,
        artifact_id: str,
        relative_path: str,
        content: bytes,
        expected_hash: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> ArtifactReference:
        """Publish bytes after checking their staged SHA-256 digest."""

        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        declared_hash = expected_hash or hashlib.sha256(content).hexdigest()
        return self.publish_writer(
            artifact_id=artifact_id,
            relative_path=relative_path,
            expected_hash=declared_hash,
            writer=lambda stream: stream.write(content),
            media_type=media_type,
        )

    def publish_writer(
        self,
        *,
        artifact_id: str,
        relative_path: str,
        expected_hash: str,
        writer: ArtifactWriter,
        media_type: Optional[str] = None,
    ) -> ArtifactReference:
        """Stage, flush, verify, and atomically publish writer output."""

        if not callable(writer):
            raise TypeError("writer must be callable")

        normalized_path, target = self._resolve_target(relative_path)
        validated_artifact_id = self._validate_artifact_id(artifact_id, normalized_path)
        validated_hash = self._validate_expected_hash(
            expected_hash, validated_artifact_id, normalized_path
        )

        if target.exists():
            self._raise_recorded_failure(
                artifact_id=validated_artifact_id,
                relative_path=normalized_path,
                code="ARTIFACT_TARGET_ALREADY_COMPLETE",
                stage="completed_target_precheck",
                field="target_path",
                expected="absent destination",
                observed="completed artifact exists",
                invariant_checks={
                    "completed_target_unchanged": True,
                    "temporary_file_created": False,
                },
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        phase = "temporary_write"
        try:
            descriptor, temp_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            temp_path = Path(temp_name)
            with os.fdopen(descriptor, "wb") as stream:
                writer(stream)
                phase = "flush_and_sync"
                stream.flush()
                os.fsync(stream.fileno())

            phase = "hash_validation"
            observed_hash = hash_file(temp_path)
            if observed_hash != validated_hash:
                raise _PublishFailure(
                    code="ARTIFACT_HASH_MISMATCH",
                    field="content_hash",
                    expected=validated_hash,
                    observed=observed_hash,
                )

            # Build the reference before publication so no schema validation can
            # turn a successful atomic replacement into a reported failure.
            reference = ArtifactReference(
                artifact_id=validated_artifact_id,
                relative_path=normalized_path,
                content_hash=observed_hash,
                media_type=media_type,
            )

            phase = "completed_target_precheck"
            if target.exists():
                raise _PublishFailure(
                    code="ARTIFACT_TARGET_ALREADY_COMPLETE",
                    field="target_path",
                    expected="absent destination",
                    observed="destination appeared before publication",
                )

            phase = "atomic_publish"
            temp_path.replace(target)
            temp_path = None
            return reference
        except Exception as exc:
            cleanup_ok, quarantined = self._dispose_temp(temp_path)
            code, field, expected, observed = self._classify_failure(exc, phase)
            self._raise_recorded_failure(
                artifact_id=validated_artifact_id,
                relative_path=normalized_path,
                code=code,
                stage=phase,
                field=field,
                expected=expected,
                observed=observed,
                invariant_checks={
                    "completed_target_unchanged": not target.exists(),
                    "temporary_file_disposed": cleanup_ok,
                    "temporary_file_quarantined": quarantined,
                },
                cause=exc,
            )
        raise AssertionError("unreachable")

    def _resolve_target(self, relative_path: str) -> tuple[str, Path]:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("relative_path must be a non-empty string")
        if relative_path != relative_path.strip():
            raise ValueError("relative_path must not have surrounding whitespace")

        windows_path = PureWindowsPath(relative_path)
        normalized = relative_path.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
            raise ValueError("relative_path must be relative")
        if ".." in windows_path.parts or ".." in posix_path.parts:
            raise ValueError("relative_path must not traverse parent directories")
        if normalized in {"", "."}:
            raise ValueError("relative_path must identify a file")

        target = (self.root / Path(*posix_path.parts)).resolve(strict=False)
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("relative_path resolves outside the artifact root") from exc
        if target == self.root:
            raise ValueError("relative_path must identify a file")
        return posix_path.as_posix(), target

    def _validate_artifact_id(self, artifact_id: str, relative_path: str) -> str:
        try:
            return TypeAdapter(ArtifactId).validate_python(artifact_id)
        except ValidationError as exc:
            self._raise_recorded_failure(
                artifact_id=str(artifact_id),
                relative_path=relative_path,
                code="INVALID_ARTIFACT_ID",
                stage="request_validation",
                field="artifact_id",
                expected="non-empty identifier without control characters",
                observed=repr(artifact_id),
                invariant_checks={"temporary_file_created": False},
                cause=exc,
            )
        raise AssertionError("unreachable")

    def _validate_expected_hash(
        self,
        expected_hash: str,
        artifact_id: str,
        relative_path: str,
    ) -> str:
        try:
            return TypeAdapter(ContentHash).validate_python(expected_hash)
        except ValidationError as exc:
            self._raise_recorded_failure(
                artifact_id=artifact_id,
                relative_path=relative_path,
                code="INVALID_EXPECTED_CONTENT_HASH",
                stage="request_validation",
                field="expected_hash",
                expected="64-character hexadecimal SHA-256 digest",
                observed=repr(expected_hash),
                invariant_checks={"temporary_file_created": False},
                cause=exc,
            )
        raise AssertionError("unreachable")

    def _dispose_temp(self, temp_path: Optional[Path]) -> tuple[bool, bool]:
        if temp_path is None or not temp_path.exists():
            return True, False
        try:
            if self.quarantine_failed_temps:
                quarantine = self.root / ".quarantine"
                quarantine.mkdir(parents=True, exist_ok=True)
                destination = quarantine / temp_path.name
                temp_path.replace(destination)
                return True, True
            temp_path.unlink()
            return True, False
        except OSError:
            return False, False

    @staticmethod
    def _classify_failure(
        exc: Exception,
        phase: str,
    ) -> tuple[str, str, object, object]:
        if isinstance(exc, _PublishFailure):
            return exc.code, exc.field, exc.expected, exc.observed
        if phase == "temporary_write":
            return "ARTIFACT_TEMP_WRITE_FAILED", "temporary_file", "complete write", repr(exc)
        if phase == "flush_and_sync":
            return "ARTIFACT_FLUSH_FAILED", "temporary_file", "flushed and synchronized bytes", repr(exc)
        if phase == "hash_validation":
            return "ARTIFACT_HASH_READ_FAILED", "content_hash", "readable staged bytes", repr(exc)
        if phase == "completed_target_precheck":
            return "ARTIFACT_TARGET_PRECHECK_FAILED", "target_path", "absent destination", repr(exc)
        if phase == "atomic_publish" and isinstance(exc, OSError):
            winerror = getattr(exc, "winerror", None)
            if winerror in _WINDOWS_SHARING_VIOLATIONS:
                return (
                    "WINDOWS_ATOMIC_REPLACE_SHARING_VIOLATION",
                    "target_path",
                    "destination not locked by another process",
                    {"winerror": winerror, "message": str(exc)},
                )
            if winerror == _WINDOWS_ACCESS_DENIED:
                return (
                    "WINDOWS_ATOMIC_REPLACE_ACCESS_DENIED",
                    "target_path",
                    "replace permission",
                    {"winerror": winerror, "message": str(exc)},
                )
            if exc.errno == errno.EXDEV:
                return (
                    "ATOMIC_REPLACE_CROSS_DEVICE",
                    "target_path",
                    "temporary and target paths on one filesystem",
                    {"errno": exc.errno, "message": str(exc)},
                )
            return (
                "ATOMIC_REPLACE_FAILED",
                "target_path",
                "successful same-filesystem atomic replace",
                {"errno": exc.errno, "message": str(exc)},
            )
        return "ARTIFACT_PUBLICATION_FAILED", "artifact", "completed publication", repr(exc)

    def _raise_recorded_failure(
        self,
        *,
        artifact_id: str,
        relative_path: str,
        code: str,
        stage: str,
        field: str,
        expected: object,
        observed: object,
        invariant_checks: dict[str, bool],
        cause: Optional[BaseException] = None,
    ) -> None:
        record = ErrorRecord(
            error_code=code,
            failed_stage=stage,
            affected_identifiers={
                "artifact_id": artifact_id,
                "relative_path": relative_path,
            },
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            invariant_checks=invariant_checks,
        )
        failure = ArtifactStoreError(record)
        try:
            self.error_journal.append(record)
        except OSError as journal_error:
            failure.add_note(f"error journal append also failed: {journal_error!r}")
        if cause is None:
            raise failure
        raise failure from cause
