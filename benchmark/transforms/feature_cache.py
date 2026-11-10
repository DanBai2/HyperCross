"""Immutable, lineage-validated cache for fold-scoped feature tensors."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from pydantic import ValidationError

from benchmark.recovery.artifact_store import FailureAtomicArtifactStore
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.canonical import canonical_deserialize, canonical_serialize
from benchmark.schema.preprocessing import FeatureCacheKey, FeatureLineage
from benchmark.schema.records import ArtifactReference, ErrorRecord


_CACHE_MAGIC = b"HCFCACHE\x01"
_HEADER_LENGTH_BYTES = 8
_MAX_HEADER_BYTES = 16 * 1024 * 1024


class FeatureCacheValidationError(ValueError):
    """Cache rejection carrying a machine-readable error record."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        super().__init__(
            f"{error_record.error_code}: feature cache rejected at "
            f"{error_record.failed_stage} ({error_record.failing_field})"
        )


class FeatureCache:
    """Store tensors atomically and validate all lineage before returning data.

    Each entry is one immutable file containing canonical key/lineage metadata
    followed by one NumPy ``.npy`` payload. Keeping both parts in one atomic
    publication prevents a tensor from being paired with partially published or
    stale lineage.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.error_journal = error_journal or ErrorJournal(
            self.root / ".errors" / "error-journal.jsonl"
        )
        self._store = FailureAtomicArtifactStore(
            self.root,
            error_journal=self.error_journal,
        )

    def put_atomic(
        self,
        key: FeatureCacheKey,
        value: Any,
        lineage: FeatureLineage,
    ) -> ArtifactReference:
        """Validate and atomically publish a new immutable cache entry."""

        self._require_models(key, lineage)
        if key.transform_state_hash != lineage.transform_hash:
            self._reject(
                key,
                code="FEATURE_CACHE_TRANSFORM_HASH_MISMATCH",
                stage="feature_cache_put_validation",
                field="transform_hash",
                expected=key.transform_state_hash,
                observed=lineage.transform_hash,
            )

        tensor = np.asarray(value)
        if tensor.dtype.hasobject:
            self._reject(
                key,
                code="FEATURE_CACHE_UNSAFE_DTYPE",
                stage="feature_cache_put_validation",
                field="tensor.dtype",
                expected="a non-object NumPy dtype",
                observed=str(tensor.dtype),
            )
        if tuple(tensor.shape) != lineage.output_shape:
            self._reject(
                key,
                code="FEATURE_CACHE_OUTPUT_SHAPE_MISMATCH",
                stage="feature_cache_put_validation",
                field="output_shape",
                expected=list(lineage.output_shape),
                observed=list(tensor.shape),
            )

        content = self._encode_entry(key, lineage, tensor)
        return self._store.publish_bytes(
            artifact_id=f"feature-cache-{key.identity_hash}",
            relative_path=self._relative_path(key),
            content=content,
            media_type="application/vnd.hypercross.feature-cache",
        )

    def get(
        self,
        key: FeatureCacheKey,
        expected_lineage: FeatureLineage,
    ) -> np.ndarray:
        """Return cached tensor only after key, lineage, and shape validation."""

        self._require_models(key, expected_lineage)
        entry_path = self.entry_path(key)
        if not entry_path.is_file():
            self._reject(
                key,
                code="FEATURE_CACHE_ENTRY_NOT_FOUND",
                stage="feature_cache_lookup",
                field="cache_key",
                expected="an immutable cache entry",
                observed="missing",
            )

        try:
            content = entry_path.read_bytes()
            stored_key, stored_lineage, tensor_payload = self._decode_metadata(content)
        except (OSError, ValueError, ValidationError) as exc:
            self._reject(
                key,
                code="FEATURE_CACHE_ENTRY_CORRUPT",
                stage="feature_cache_metadata_validation",
                field="cache_entry",
                expected="valid canonical metadata and tensor payload",
                observed=repr(exc),
            )

        self._validate_key(key, stored_key)
        self._validate_lineage(key, expected_lineage, stored_lineage)

        # Tensor bytes are deliberately decoded only after all metadata lineage
        # checks pass. No mismatched source or fitted boundary can expose data.
        try:
            tensor = np.load(io.BytesIO(tensor_payload), allow_pickle=False)
            if not isinstance(tensor, np.ndarray):
                raise ValueError("cache payload is not one ndarray")
        except (OSError, ValueError) as exc:
            self._reject(
                key,
                code="FEATURE_CACHE_TENSOR_CORRUPT",
                stage="feature_cache_tensor_validation",
                field="tensor",
                expected="a safe NumPy ndarray payload",
                observed=repr(exc),
            )

        if tuple(tensor.shape) != stored_lineage.output_shape:
            self._reject(
                key,
                code="FEATURE_CACHE_OUTPUT_SHAPE_MISMATCH",
                stage="feature_cache_tensor_validation",
                field="output_shape",
                expected=list(stored_lineage.output_shape),
                observed=list(tensor.shape),
            )
        tensor.setflags(write=False)
        return tensor

    def entry_path(self, key: FeatureCacheKey) -> Path:
        """Return the deterministic path for a key without reading the entry."""

        if not isinstance(key, FeatureCacheKey):
            raise TypeError("key must be a FeatureCacheKey")
        return self.root / Path(*self._relative_path(key).split("/"))

    @staticmethod
    def _require_models(key: FeatureCacheKey, lineage: FeatureLineage) -> None:
        if not isinstance(key, FeatureCacheKey):
            raise TypeError("key must be a FeatureCacheKey")
        if not isinstance(lineage, FeatureLineage):
            raise TypeError("lineage must be a FeatureLineage")

    @staticmethod
    def _relative_path(key: FeatureCacheKey) -> str:
        return f"entries/{key.identity_hash}.feature-cache"

    @staticmethod
    def _encode_entry(
        key: FeatureCacheKey,
        lineage: FeatureLineage,
        tensor: np.ndarray,
    ) -> bytes:
        metadata = canonical_serialize(
            {
                "key": key.model_dump(mode="python"),
                "lineage": lineage.model_dump(mode="python"),
            }
        )
        tensor_stream = io.BytesIO()
        np.save(tensor_stream, tensor, allow_pickle=False)
        return (
            _CACHE_MAGIC
            + struct.pack(">Q", len(metadata))
            + metadata
            + tensor_stream.getvalue()
        )

    @staticmethod
    def _decode_metadata(
        content: bytes,
    ) -> tuple[FeatureCacheKey, FeatureLineage, bytes]:
        prefix_size = len(_CACHE_MAGIC) + _HEADER_LENGTH_BYTES
        if len(content) < prefix_size or content[: len(_CACHE_MAGIC)] != _CACHE_MAGIC:
            raise ValueError("invalid feature-cache magic")
        header_length = struct.unpack(
            ">Q", content[len(_CACHE_MAGIC) : prefix_size]
        )[0]
        if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
            raise ValueError("invalid feature-cache metadata length")
        header_end = prefix_size + header_length
        if header_end >= len(content):
            raise ValueError("truncated feature-cache entry")
        decoded = canonical_deserialize(content[prefix_size:header_end])
        if not isinstance(decoded, dict) or set(decoded) != {"key", "lineage"}:
            raise ValueError("feature-cache metadata has invalid fields")
        stored_key = FeatureCacheKey.model_validate(decoded["key"])
        stored_lineage = FeatureLineage.model_validate(decoded["lineage"])
        return stored_key, stored_lineage, content[header_end:]

    def _validate_key(
        self,
        expected: FeatureCacheKey,
        observed: FeatureCacheKey,
    ) -> None:
        fields = (
            "dataset_hash",
            "protocol_hash",
            "fold_id",
            "representation_key",
            "transform_state_hash",
        )
        for field in fields:
            expected_value = getattr(expected, field)
            observed_value = getattr(observed, field)
            if observed_value != expected_value:
                self._reject(
                    expected,
                    code="FEATURE_CACHE_KEY_MISMATCH",
                    stage="feature_cache_key_validation",
                    field=f"cache_key.{field}",
                    expected=expected_value,
                    observed=observed_value,
                )

    def _validate_lineage(
        self,
        key: FeatureCacheKey,
        expected: FeatureLineage,
        observed: FeatureLineage,
    ) -> None:
        checks = (
            (
                "source_hashes",
                "FEATURE_CACHE_SOURCE_HASH_MISMATCH",
            ),
            (
                "transform_config",
                "FEATURE_CACHE_CONFIG_MISMATCH",
            ),
            (
                "config_hash",
                "FEATURE_CACHE_CONFIG_HASH_MISMATCH",
            ),
            (
                "fit_sample_ids",
                "FEATURE_CACHE_FIT_BOUNDARY_MISMATCH",
            ),
            (
                "fit_sample_ids_hash",
                "FEATURE_CACHE_FIT_BOUNDARY_MISMATCH",
            ),
            (
                "transform_hash",
                "FEATURE_CACHE_TRANSFORM_HASH_MISMATCH",
            ),
            (
                "output_shape",
                "FEATURE_CACHE_OUTPUT_SHAPE_MISMATCH",
            ),
        )
        for field, code in checks:
            expected_value = getattr(expected, field)
            observed_value = getattr(observed, field)
            if observed_value != expected_value:
                self._reject(
                    key,
                    code=code,
                    stage="feature_cache_lineage_validation",
                    field=field,
                    expected=self._machine_value(expected_value),
                    observed=self._machine_value(observed_value),
                )

        if observed.transform_hash != key.transform_state_hash:
            self._reject(
                key,
                code="FEATURE_CACHE_TRANSFORM_HASH_MISMATCH",
                stage="feature_cache_lineage_validation",
                field="transform_hash",
                expected=key.transform_state_hash,
                observed=observed.transform_hash,
            )

    @staticmethod
    def _machine_value(value: Any) -> Any:
        if isinstance(value, tuple):
            return list(value)
        return value

    def _reject(
        self,
        key: FeatureCacheKey,
        *,
        code: str,
        stage: str,
        field: str,
        expected: Any,
        observed: Any,
    ) -> None:
        entry_path = self.entry_path(key)
        record = ErrorRecord(
            error_code=code,
            failed_stage=stage,
            affected_identifiers={
                "cache_key": key.identity_hash,
                "dataset_hash": key.dataset_hash,
                "protocol_hash": key.protocol_hash,
                "fold_id": key.fold_id,
                "representation_key": key.representation_key,
                "transform_state_hash": key.transform_state_hash,
            },
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            invariant_checks={
                "cached_tensor_not_returned": True,
                "rejected_entry_unchanged": True,
                "completed_artifacts_unchanged": True,
                "entry_existed_at_rejection": entry_path.is_file(),
            },
        )
        failure = FeatureCacheValidationError(record)
        try:
            self.error_journal.append(record)
        except OSError as journal_error:
            failure.add_note(f"error journal append also failed: {journal_error!r}")
        raise failure


__all__ = ["FeatureCache", "FeatureCacheValidationError"]
