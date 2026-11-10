"""Controlled, hash-gated loading for external PyTorch checkpoints."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from pydantic import TypeAdapter, ValidationError

from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import ContentHash
from benchmark.schema.records import ErrorRecord


_CHECKPOINT_READ_CHUNK_BYTES = 1024 * 1024
_CHECKPOINT_SPOOL_MEMORY_BYTES = 64 * 1024 * 1024


class ExternalCheckpointLoadError(RuntimeError):
    """Controlled checkpoint loading failure with a machine-readable record."""

    def __init__(self, message: str, error_record: ErrorRecord) -> None:
        super().__init__(message)
        self.error_record = error_record


class ExternalCheckpointLoader:
    """Validate provenance and bytes before restricted deserialization.

    The loader copies the checkpoint from a read-only source handle into a
    bounded-memory spool while computing SHA-256. Restricted deserialization
    then reads that exact verified snapshot, avoiding a second path lookup and
    preventing path replacement between validation and loading. Large files
    spill to an automatically removed temporary file rather than remaining in
    memory.
    """

    def __init__(self, error_journal: Optional[ErrorJournal] = None) -> None:
        self._error_journal = error_journal

    def load(
        self,
        checkpoint_reference: Union[str, Path],
        *,
        source_reference: Optional[str],
        expected_content_hash: Optional[str],
        map_location: Any = "cpu",
    ) -> Any:
        """Load one external checkpoint after source and hash validation.

        ``source_reference`` identifies where the external checkpoint was
        obtained (for example, a pinned release URL or artifact reference).
        ``expected_content_hash`` must be its registered SHA-256 digest.
        Neither value is inferred from the checkpoint itself.
        """

        checkpoint_path, checkpoint_display = self._checkpoint_path(
            checkpoint_reference
        )
        validated_source = self._validate_source_reference(
            source_reference,
            checkpoint_display,
        )
        validated_hash = self._validate_expected_hash(
            expected_content_hash,
            checkpoint_display,
            validated_source,
        )

        try:
            with checkpoint_path.open("rb") as source_stream:
                with tempfile.SpooledTemporaryFile(
                    max_size=_CHECKPOINT_SPOOL_MEMORY_BYTES,
                    mode="w+b",
                ) as verified_stream:
                    observed_hash = self._copy_and_hash(
                        source_stream,
                        verified_stream,
                    )
                    if observed_hash != validated_hash:
                        self._fail(
                            code="EXTERNAL_CHECKPOINT_CONTENT_HASH_MISMATCH",
                            checkpoint_reference=checkpoint_display,
                            source_reference=validated_source,
                            field="content_hash",
                            expected=validated_hash,
                            observed=observed_hash,
                            message="external checkpoint content hash does not match",
                            source_content_hash=observed_hash,
                        )

                    verified_stream.seek(0)
                    try:
                        return self._restricted_torch_load(
                            verified_stream,
                            map_location=map_location,
                        )
                    except ExternalCheckpointLoadError:
                        raise
                    except Exception as exc:
                        self._fail(
                            code="EXTERNAL_CHECKPOINT_DESERIALIZATION_FAILED",
                            checkpoint_reference=checkpoint_display,
                            source_reference=validated_source,
                            field="restricted_load",
                            expected="a weights-only-compatible PyTorch checkpoint",
                            observed={"exception_type": type(exc).__name__},
                            message="restricted external checkpoint loading failed",
                            source_content_hash=observed_hash,
                            deserialization_started=True,
                            cause=exc,
                        )
        except ExternalCheckpointLoadError:
            raise
        except OSError as exc:
            self._fail(
                code="EXTERNAL_CHECKPOINT_READ_FAILED",
                checkpoint_reference=checkpoint_display,
                source_reference=validated_source,
                field="checkpoint_reference",
                expected="a readable external checkpoint file",
                observed={
                    "exception_type": type(exc).__name__,
                    "errno": exc.errno,
                },
                message="external checkpoint could not be read",
                cause=exc,
            )
        raise AssertionError("unreachable")

    @staticmethod
    def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> str:
        digest = hashlib.sha256()
        while True:
            chunk = source.read(_CHECKPOINT_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            destination.write(chunk)
        destination.flush()
        return digest.hexdigest()

    @staticmethod
    def _restricted_torch_load(stream: BinaryIO, *, map_location: Any) -> Any:
        """Use PyTorch's restricted weights-only unpickler without fallback."""

        import torch

        return torch.load(
            stream,
            map_location=map_location,
            weights_only=True,
        )

    def _checkpoint_path(
        self,
        checkpoint_reference: Union[str, Path],
    ) -> tuple[Path, str]:
        if not isinstance(checkpoint_reference, (str, Path)):
            display = repr(checkpoint_reference)
            self._fail(
                code="EXTERNAL_CHECKPOINT_REFERENCE_INVALID",
                checkpoint_reference=display,
                source_reference=None,
                field="checkpoint_reference",
                expected="a non-empty local path",
                observed=type(checkpoint_reference).__name__,
                message="external checkpoint reference must be a local path",
            )
        display = str(checkpoint_reference)
        if not display.strip() or display != display.strip():
            self._fail(
                code="EXTERNAL_CHECKPOINT_REFERENCE_INVALID",
                checkpoint_reference=display,
                source_reference=None,
                field="checkpoint_reference",
                expected="a non-empty local path without surrounding whitespace",
                observed=display,
                message="external checkpoint reference is invalid",
            )
        return Path(checkpoint_reference), display

    def _validate_source_reference(
        self,
        source_reference: Optional[str],
        checkpoint_reference: str,
    ) -> str:
        if not isinstance(source_reference, str) or not source_reference.strip():
            self._fail(
                code="EXTERNAL_CHECKPOINT_SOURCE_REFERENCE_MISSING",
                checkpoint_reference=checkpoint_reference,
                source_reference=None,
                field="source_reference",
                expected="a non-empty registered source reference",
                observed=source_reference,
                message="external checkpoint source reference is required",
            )
        if source_reference != source_reference.strip() or any(
            ord(character) < 32 or ord(character) == 127
            for character in source_reference
        ):
            self._fail(
                code="EXTERNAL_CHECKPOINT_SOURCE_REFERENCE_INVALID",
                checkpoint_reference=checkpoint_reference,
                source_reference=source_reference,
                field="source_reference",
                expected="a source reference without surrounding whitespace or control characters",
                observed=source_reference,
                message="external checkpoint source reference is invalid",
            )
        return source_reference

    def _validate_expected_hash(
        self,
        expected_content_hash: Optional[str],
        checkpoint_reference: str,
        source_reference: str,
    ) -> str:
        if expected_content_hash is None or expected_content_hash == "":
            self._fail(
                code="EXTERNAL_CHECKPOINT_CONTENT_HASH_MISSING",
                checkpoint_reference=checkpoint_reference,
                source_reference=source_reference,
                field="content_hash",
                expected="a registered SHA-256 content hash",
                observed=expected_content_hash,
                message="external checkpoint content hash is required",
            )
        try:
            return TypeAdapter(ContentHash).validate_python(expected_content_hash)
        except ValidationError as exc:
            self._fail(
                code="EXTERNAL_CHECKPOINT_CONTENT_HASH_INVALID",
                checkpoint_reference=checkpoint_reference,
                source_reference=source_reference,
                field="content_hash",
                expected="a 64-character hexadecimal SHA-256 digest",
                observed=expected_content_hash,
                message="external checkpoint content hash is invalid",
                cause=exc,
            )
        raise AssertionError("unreachable")

    def _fail(
        self,
        *,
        code: str,
        checkpoint_reference: str,
        source_reference: Optional[str],
        field: str,
        expected: Any,
        observed: Any,
        message: str,
        source_content_hash: Optional[str] = None,
        deserialization_started: bool = False,
        cause: Optional[BaseException] = None,
    ) -> None:
        identifiers = {"checkpoint_reference": checkpoint_reference}
        if source_reference is not None:
            identifiers["source_reference"] = source_reference
        error = ErrorRecord(
            error_code=code,
            failed_stage=(
                "checkpoint_deserialization"
                if deserialization_started
                else "checkpoint_validation"
            ),
            affected_identifiers=identifiers,
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            invariant_checks={
                "deserialization_not_started": not deserialization_started,
                "restricted_loading_required": True,
                "checkpoint_bytes_unchanged": True,
                "completed_artifacts_unchanged": True,
            },
            source_path=checkpoint_reference,
            source_content_hash=source_content_hash,
        )
        failure = ExternalCheckpointLoadError(message, error)
        if self._error_journal is not None:
            try:
                self._error_journal.append(error)
            except OSError as journal_error:
                failure.add_note(
                    f"error journal append also failed: {journal_error!r}"
                )
        if cause is None:
            raise failure
        raise failure from cause
