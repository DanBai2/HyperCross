"""Property 9: PCA and cache lineage.

**Validates: Requirements 4.17-4.27, 14.21-14.27**
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from benchmark.schema import ErrorRecord, FeatureCacheKey, FeatureLineage
from benchmark.transforms import FeatureCache, FeatureCacheValidationError


_KEY_FIELDS = (
    "dataset_hash",
    "protocol_hash",
    "fold_id",
    "representation_key",
    "transform_state_hash",
)
_MISMATCH_KINDS = st.sampled_from(
    (
        "source_hashes",
        "transform_config",
        "transform_hash",
        "fit_sample_ids",
        "output_shape",
        *tuple(f"key.{field}" for field in _KEY_FIELDS),
    )
)


def _content_hash(namespace: str, value: object) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()


@st.composite
def _cache_cases(draw: st.DrawFn) -> tuple[
    FeatureCacheKey,
    FeatureLineage,
    np.ndarray,
]:
    token = draw(st.integers(min_value=0, max_value=2**32 - 1))
    fit_numbers = draw(
        st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    rows = draw(st.integers(min_value=1, max_value=5))
    columns = draw(st.integers(min_value=1, max_value=4))
    dtype = draw(st.sampled_from((np.dtype("float32"), np.dtype("float64"))))
    transform_hash = _content_hash("transform", token)
    key = FeatureCacheKey(
        dataset_hash=_content_hash("dataset", token),
        protocol_hash=_content_hash("protocol", token),
        fold_id=f"outer-{token}",
        representation_key=f"pca-representation-{token}",
        transform_state_hash=transform_hash,
    )
    lineage = FeatureLineage.create(
        source_hashes={
            "cube": _content_hash("cube", token),
            "labels": _content_hash("labels", token),
        },
        transform_config={
            "n_components": columns,
            "seed": token,
            "whiten": bool(token % 2),
        },
        fit_sample_ids=tuple(f"sample-{number}" for number in fit_numbers),
        transform_hash=transform_hash,
        output_shape=(rows, columns),
    )
    tensor = np.arange(rows * columns, dtype=dtype).reshape(rows, columns)
    return key, lineage, tensor


def _changed_hash(current: str, namespace: str) -> str:
    candidate = _content_hash(namespace, current)
    assert candidate != current
    return candidate


def _expected_lineage_with_mismatch(
    lineage: FeatureLineage,
    mismatch_kind: str,
) -> FeatureLineage:
    source_hashes = dict(lineage.source_hashes)
    transform_config = dict(lineage.transform_config)
    fit_sample_ids = lineage.fit_sample_ids
    transform_hash = lineage.transform_hash
    output_shape = lineage.output_shape

    if mismatch_kind == "source_hashes":
        source_hashes["cube"] = _changed_hash(source_hashes["cube"], "changed-source")
    elif mismatch_kind == "transform_config":
        transform_config["generated_variant"] = "different"
    elif mismatch_kind == "transform_hash":
        transform_hash = _changed_hash(transform_hash, "changed-transform")
    elif mismatch_kind == "fit_sample_ids":
        outsider = "generated-outside-fit-boundary"
        assert outsider not in fit_sample_ids
        fit_sample_ids = (*fit_sample_ids, outsider)
    elif mismatch_kind == "output_shape":
        output_shape = (output_shape[0] + 1, *output_shape[1:])
    else:
        raise AssertionError(f"not a lineage mismatch: {mismatch_kind}")

    return FeatureLineage.create(
        source_hashes=source_hashes,
        transform_config=transform_config,
        fit_sample_ids=fit_sample_ids,
        transform_hash=transform_hash,
        output_shape=output_shape,
    )


def _expected_key_with_mismatch(
    key: FeatureCacheKey,
    field: str,
) -> FeatureCacheKey:
    values = key.model_dump(mode="python")
    if field in {"dataset_hash", "protocol_hash", "transform_state_hash"}:
        values[field] = _changed_hash(values[field], f"changed-key-{field}")
    elif field == "fold_id":
        values[field] = f"{values[field]}-different"
    elif field == "representation_key":
        values[field] = f"{values[field]}-different"
    else:
        raise AssertionError(f"unsupported key field: {field}")
    return FeatureCacheKey.model_validate(values)


def _protected_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".errors" not in path.parts
    }


@settings(max_examples=100, deadline=None)
@given(case=_cache_cases(), mismatch_kind=_MISMATCH_KINDS)
def test_generated_pca_cache_lineage_mismatches_reject_before_return(
    case: tuple[FeatureCacheKey, FeatureLineage, np.ndarray],
    mismatch_kind: str,
) -> None:
    """Every source, transform, fit, shape, or key mismatch is atomic and auditable."""

    key, lineage, tensor = case
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "cache root with spaces"
        cache = FeatureCache(root)
        cache.put_atomic(key, tensor, lineage)
        sentinel = root / "completed" / "sentinel.bin"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"completed-artifact-must-not-change")

        expected_key = key
        expected_lineage = lineage
        if mismatch_kind.startswith("key."):
            key_field = mismatch_kind.removeprefix("key.")
            expected_key = _expected_key_with_mismatch(key, key_field)
            misplaced_entry = cache.entry_path(expected_key)
            misplaced_entry.parent.mkdir(parents=True, exist_ok=True)
            misplaced_entry.write_bytes(cache.entry_path(key).read_bytes())
            expected_code = "FEATURE_CACHE_KEY_MISMATCH"
            expected_field = f"cache_key.{key_field}"
            expected_value = getattr(expected_key, key_field)
            observed_value = getattr(key, key_field)
        else:
            expected_lineage = _expected_lineage_with_mismatch(lineage, mismatch_kind)
            expected_code_by_field = {
                "source_hashes": "FEATURE_CACHE_SOURCE_HASH_MISMATCH",
                "transform_config": "FEATURE_CACHE_CONFIG_MISMATCH",
                "transform_hash": "FEATURE_CACHE_TRANSFORM_HASH_MISMATCH",
                "fit_sample_ids": "FEATURE_CACHE_FIT_BOUNDARY_MISMATCH",
                "output_shape": "FEATURE_CACHE_OUTPUT_SHAPE_MISMATCH",
            }
            expected_code = expected_code_by_field[mismatch_kind]
            expected_field = mismatch_kind
            expected_value = getattr(expected_lineage, mismatch_kind)
            observed_value = getattr(lineage, mismatch_kind)
            if isinstance(expected_value, tuple):
                expected_value = list(expected_value)
            if isinstance(observed_value, tuple):
                observed_value = list(observed_value)

        protected_before = _protected_bytes(root)
        with patch.object(
            np,
            "load",
            side_effect=AssertionError("tensor decoded before lineage rejection"),
        ):
            with pytest.raises(FeatureCacheValidationError) as caught:
                cache.get(expected_key, expected_lineage)

        record = caught.value.error_record
        assert isinstance(record, ErrorRecord)
        assert record.error_code == expected_code
        assert record.failed_stage in {
            "feature_cache_key_validation",
            "feature_cache_lineage_validation",
        }
        assert record.failing_field == expected_field
        assert record.expected_value == expected_value
        assert record.observed_value == observed_value
        assert record.affected_identifiers == {
            "cache_key": expected_key.identity_hash,
            "dataset_hash": expected_key.dataset_hash,
            "protocol_hash": expected_key.protocol_hash,
            "fold_id": expected_key.fold_id,
            "representation_key": expected_key.representation_key,
            "transform_state_hash": expected_key.transform_state_hash,
        }
        assert record.invariant_checks == {
            "cached_tensor_not_returned": True,
            "rejected_entry_unchanged": True,
            "completed_artifacts_unchanged": True,
            "entry_existed_at_rejection": True,
        }
        assert _protected_bytes(root) == protected_before

        journal_path = root / ".errors" / "error-journal.jsonl"
        journal_records = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
        ]
        assert journal_records == [record.model_dump(mode="json")]
