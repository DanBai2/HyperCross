"""Property 30: atomic write and interruption safety.

**Validates: Requirements 13.3-13.8, 13.35, 13.36**
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st

from benchmark.recovery import ArtifactStoreError, FailureAtomicArtifactStore
from benchmark.recovery import artifact_store as artifact_store_module
from benchmark.schema.canonical import hash_file as real_hash_file


_BINARY_CONTENT = st.binary(min_size=1, max_size=4096)


@contextmanager
def _isolated_root() -> Iterator[Path]:
    """Give every generated example a fresh filesystem transaction boundary."""

    with tempfile.TemporaryDirectory(prefix="property30-") as directory:
        yield Path(directory)


@st.composite
def interrupted_writes(draw: st.DrawFn) -> tuple[bytes, int]:
    content = draw(_BINARY_CONTENT)
    return content, draw(st.integers(min_value=0, max_value=len(content)))


@given(completed_content=_BINARY_CONTENT, interrupted=interrupted_writes())
def test_generated_write_interruptions_dispose_only_the_current_temporary_file(
    completed_content: bytes,
    interrupted: tuple[bytes, int],
) -> None:
    """An interrupted staged write cannot publish or alter completed bytes."""

    attempted_content, interrupt_after = interrupted
    with _isolated_root() as root:
        store = FailureAtomicArtifactStore(root)
        completed_path = root / "completed" / "stable.bin"
        completed_path.parent.mkdir(parents=True)
        completed_path.write_bytes(completed_content)
        completed_hash = real_hash_file(completed_path)
        target = root / "pending" / "candidate.bin"
        observed_temp_paths: list[Path] = []
        original_mkstemp = artifact_store_module.tempfile.mkstemp

        def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
            descriptor, temp_name = original_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
            observed_temp_paths.append(Path(temp_name))
            return descriptor, temp_name

        def interrupted_writer(stream: object) -> None:
            assert len(observed_temp_paths) == 1
            temp_path = observed_temp_paths[0]
            assert temp_path.parent == target.parent
            assert temp_path.name.startswith(f".{target.name}.")
            assert temp_path.suffix == ".tmp"
            assert temp_path.exists()
            assert not target.exists()
            stream.write(attempted_content[:interrupt_after])  # type: ignore[attr-defined]
            raise InterruptedError("generated write-stage interruption")

        with (
            patch.object(artifact_store_module.tempfile, "mkstemp", recording_mkstemp),
            pytest.raises(ArtifactStoreError) as caught,
        ):
            store.publish_writer(
                artifact_id="property30-interrupted",
                relative_path="pending/candidate.bin",
                expected_hash=hashlib.sha256(attempted_content).hexdigest(),
                writer=interrupted_writer,
            )

        assert len(observed_temp_paths) == 1
        assert not observed_temp_paths[0].exists()
        assert not target.exists()
        assert not list(target.parent.glob("*.tmp"))
        assert completed_path.read_bytes() == completed_content
        assert real_hash_file(completed_path) == completed_hash
        assert caught.value.error_record.failed_stage == "temporary_write"
        assert caught.value.error_record.error_code == "ARTIFACT_TEMP_WRITE_FAILED"
        assert caught.value.error_record.invariant_checks["temporary_file_disposed"] is True


@given(content=_BINARY_CONTENT, completed_content=_BINARY_CONTENT)
def test_hash_validation_precedes_single_atomic_publication_and_blocks_recompletion(
    content: bytes,
    completed_content: bytes,
) -> None:
    """Hashing happens before one publish, and the completed target is immutable."""

    with _isolated_root() as root:
        store = FailureAtomicArtifactStore(root)
        old_path = root / "completed" / "old.bin"
        old_path.parent.mkdir(parents=True)
        old_path.write_bytes(completed_content)
        old_hash = real_hash_file(old_path)
        target = root / "new" / "artifact.bin"
        events: list[str] = []

        def recording_hash(path: Path) -> str:
            assert not target.exists()
            events.append("hash")
            return real_hash_file(path)

        original_replace = Path.replace

        def recording_replace(source: Path, destination: Path) -> Path:
            assert events == ["hash"]
            assert source.exists()
            assert not destination.exists()
            events.append("publish")
            return original_replace(source, destination)

        with (
            patch.object(artifact_store_module, "hash_file", recording_hash),
            patch.object(Path, "replace", recording_replace),
        ):
            reference = store.publish_bytes(
                artifact_id="property30-complete",
                relative_path="new/artifact.bin",
                content=content,
            )

            with pytest.raises(ArtifactStoreError) as caught:
                store.publish_bytes(
                    artifact_id="property30-complete",
                    relative_path="new/artifact.bin",
                    content=b"second completion must be rejected",
                )

        assert events == ["hash", "publish"]
        assert target.read_bytes() == content
        assert reference.content_hash == hashlib.sha256(content).hexdigest()
        assert caught.value.error_record.error_code == "ARTIFACT_TARGET_ALREADY_COMPLETE"
        assert caught.value.error_record.failed_stage == "completed_target_precheck"
        assert caught.value.error_record.invariant_checks["temporary_file_created"] is False
        assert old_path.read_bytes() == completed_content
        assert real_hash_file(old_path) == old_hash
        assert not list(target.parent.glob("*.tmp"))


@given(content=_BINARY_CONTENT, completed_content=_BINARY_CONTENT)
def test_generated_hash_mismatch_never_reaches_atomic_publication(
    content: bytes,
    completed_content: bytes,
) -> None:
    """A staged artifact with a mismatched declaration is rejected before publish."""

    with _isolated_root() as root:
        store = FailureAtomicArtifactStore(root)
        old_path = root / "completed.bin"
        old_path.write_bytes(completed_content)
        old_hash = real_hash_file(old_path)
        target = root / "new" / "mismatch.bin"
        actual_hash = hashlib.sha256(content).hexdigest()
        wrong_hash = ("0" if actual_hash[0] != "0" else "1") + actual_hash[1:]
        publish_calls = 0

        def forbidden_replace(source: Path, destination: Path) -> Path:
            nonlocal publish_calls
            publish_calls += 1
            raise AssertionError("atomic publication ran before hash acceptance")

        with patch.object(Path, "replace", forbidden_replace):
            with pytest.raises(ArtifactStoreError) as caught:
                store.publish_bytes(
                    artifact_id="property30-hash-mismatch",
                    relative_path="new/mismatch.bin",
                    content=content,
                    expected_hash=wrong_hash,
                )

        assert publish_calls == 0
        assert caught.value.error_record.error_code == "ARTIFACT_HASH_MISMATCH"
        assert caught.value.error_record.failed_stage == "hash_validation"
        assert not target.exists()
        assert not list(target.parent.glob("*.tmp"))
        assert old_path.read_bytes() == completed_content
        assert real_hash_file(old_path) == old_hash
