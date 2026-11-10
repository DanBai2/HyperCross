"""Unit coverage for task 5.4 PCA identity and Feature Cache behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.schema import (
    FeatureCacheKey,
    FeatureLineage,
    FitBoundary,
    FitBoundaryScope,
    PCAIdentity,
)
from benchmark.transforms import (
    FeatureCache,
    FeatureCacheValidationError,
    PreprocessingManager,
)


DATASET_HASH = "a" * 64
PROTOCOL_HASH = "b" * 64
SOURCE_HASH = "c" * 64
OTHER_HASH = "d" * 64


def _boundary() -> FitBoundary:
    return FitBoundary(
        fold_id="outer-1",
        scope=FitBoundaryScope.INNER_TRAIN,
        inner_fold_id="inner-1",
        allowed_fit_sample_ids=("sample-2", "sample-1", "sample-3"),
        outer_test_sample_ids=("held-out",),
    )


def _pca_artifact():
    return PreprocessingManager().fit_pca(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 5.0]],
        ("sample-1", "sample-2", "sample-3"),
        _boundary(),
        dataset_hash=DATASET_HASH,
        protocol_hash=PROTOCOL_HASH,
        n_components=1,
        whiten=True,
    )


def _key_and_lineage(shape: tuple[int, ...] = (3, 1)):
    artifact = _pca_artifact()
    identity = PCAIdentity.from_fitted_artifact(
        artifact,
        representation_key="pca-pseudo-rgb.v1",
    )
    key = FeatureCacheKey.from_pca_identity(identity)
    lineage = FeatureLineage.from_fitted_artifact(
        artifact,
        source_hashes={"cube": SOURCE_HASH},
        output_shape=shape,
    )
    return artifact, identity, key, lineage


def test_pca_identity_binds_dataset_protocol_fold_fit_config_and_state() -> None:
    """Requirements 4.17-4.22, 14.21-14.25: PCA/cache identity is complete."""

    artifact, identity, key, lineage = _key_and_lineage()

    assert identity.dataset_hash == key.dataset_hash == DATASET_HASH
    assert identity.protocol_hash == key.protocol_hash == PROTOCOL_HASH
    assert identity.fold_id == key.fold_id == "outer-1"
    assert identity.representation_key == key.representation_key == "pca-pseudo-rgb.v1"
    assert identity.fit_sample_ids == ("sample-1", "sample-2", "sample-3")
    assert identity.config_hash == artifact.config_hash == lineage.config_hash
    assert identity.transform_state_hash == artifact.state_hash
    assert key.transform_state_hash == lineage.transform_hash
    assert lineage.transform_config == artifact.config
    assert lineage.fit_sample_ids_hash == identity.fit_sample_ids_hash


def test_feature_cache_atomically_round_trips_read_only_tensor(tmp_path: Path) -> None:
    """Requirements 4.22, 14.21-14.26: validated lineage permits exact reuse."""

    _, _, key, lineage = _key_and_lineage()
    cache = FeatureCache(tmp_path / "cache root with spaces")
    expected = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)

    reference = cache.put_atomic(key, expected, lineage)
    observed = cache.get(key, lineage)

    np.testing.assert_array_equal(observed, expected)
    assert observed.dtype == expected.dtype
    assert observed.flags.writeable is False
    assert reference.relative_path == f"entries/{key.identity_hash}.feature-cache"
    assert cache.entry_path(key).is_file()
    assert not list(cache.entry_path(key).parent.glob("*.tmp"))


def test_source_mismatch_rejects_before_tensor_decode_and_preserves_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirements 4.23, 4.27-4.28, 14.26-14.28: source mismatch is atomic."""

    artifact, _, key, lineage = _key_and_lineage()
    cache = FeatureCache(tmp_path / "cache")
    cache.put_atomic(key, np.arange(3, dtype=np.float64).reshape(3, 1), lineage)
    entry = cache.entry_path(key)
    before = entry.read_bytes()
    mismatched = FeatureLineage.from_fitted_artifact(
        artifact,
        source_hashes={"cube": OTHER_HASH},
        output_shape=(3, 1),
    )

    def forbidden_tensor_decode(*args: object, **kwargs: object) -> object:
        raise AssertionError("tensor bytes decoded before lineage validation")

    monkeypatch.setattr(np, "load", forbidden_tensor_decode)
    with pytest.raises(FeatureCacheValidationError) as caught:
        cache.get(key, mismatched)

    record = caught.value.error_record
    assert record.error_code == "FEATURE_CACHE_SOURCE_HASH_MISMATCH"
    assert record.failing_field == "source_hashes"
    assert record.affected_identifiers["cache_key"] == key.identity_hash
    assert record.invariant_checks["cached_tensor_not_returned"] is True
    assert record.invariant_checks["rejected_entry_unchanged"] is True
    assert entry.read_bytes() == before


def test_transform_fit_boundary_and_shape_mismatches_preserve_cached_bytes(
    tmp_path: Path,
) -> None:
    """Requirements 4.24-4.28: every declared lineage mismatch rejects reuse."""

    artifact, _, key, lineage = _key_and_lineage()
    cache = FeatureCache(tmp_path / "cache")
    cache.put_atomic(key, np.arange(3).reshape(3, 1), lineage)
    entry = cache.entry_path(key)
    before = entry.read_bytes()

    cases = (
        (
            FeatureLineage.create(
                source_hashes={"cube": SOURCE_HASH},
                transform_config=artifact.config,
                fit_sample_ids=artifact.fit_sample_ids,
                transform_hash=OTHER_HASH,
                output_shape=(3, 1),
            ),
            "FEATURE_CACHE_TRANSFORM_HASH_MISMATCH",
            "transform_hash",
        ),
        (
            FeatureLineage.create(
                source_hashes={"cube": SOURCE_HASH},
                transform_config=artifact.config,
                fit_sample_ids=("sample-1", "sample-2"),
                transform_hash=artifact.state_hash,
                output_shape=(3, 1),
            ),
            "FEATURE_CACHE_FIT_BOUNDARY_MISMATCH",
            "fit_sample_ids",
        ),
        (
            FeatureLineage.create(
                source_hashes={"cube": SOURCE_HASH},
                transform_config=artifact.config,
                fit_sample_ids=artifact.fit_sample_ids,
                transform_hash=artifact.state_hash,
                output_shape=(1, 3),
            ),
            "FEATURE_CACHE_OUTPUT_SHAPE_MISMATCH",
            "output_shape",
        ),
    )

    for expected_lineage, error_code, field in cases:
        with pytest.raises(FeatureCacheValidationError) as caught:
            cache.get(key, expected_lineage)
        assert caught.value.error_record.error_code == error_code
        assert caught.value.error_record.failing_field == field
        assert entry.read_bytes() == before


def test_put_rejects_tensor_shape_mismatch_without_publishing_entry(
    tmp_path: Path,
) -> None:
    """Requirements 4.26-4.28: invalid output shape cannot create cache data."""

    _, _, key, lineage = _key_and_lineage(shape=(3, 1))
    cache = FeatureCache(tmp_path / "cache")

    with pytest.raises(FeatureCacheValidationError) as caught:
        cache.put_atomic(key, np.zeros((1, 3)), lineage)

    assert caught.value.error_record.error_code == "FEATURE_CACHE_OUTPUT_SHAPE_MISMATCH"
    assert not cache.entry_path(key).exists()


def test_cache_rejection_appends_machine_readable_error_record(tmp_path: Path) -> None:
    """Requirement 4.27: cache errors name key, field, expected, and observed values."""

    artifact, _, key, lineage = _key_and_lineage()
    cache = FeatureCache(tmp_path / "cache")
    cache.put_atomic(key, np.ones((3, 1)), lineage)
    mismatched = FeatureLineage.from_fitted_artifact(
        artifact,
        source_hashes={"cube": OTHER_HASH},
        output_shape=(3, 1),
    )

    with pytest.raises(FeatureCacheValidationError):
        cache.get(key, mismatched)

    journal_path = tmp_path / "cache" / ".errors" / "error-journal.jsonl"
    records = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["error_code"] == "FEATURE_CACHE_SOURCE_HASH_MISMATCH"
    assert records[-1]["affected_identifiers"]["cache_key"] == key.identity_hash
    assert records[-1]["failing_field"] == "source_hashes"
    assert records[-1]["expected_value"] == {"cube": OTHER_HASH}
    assert records[-1]["observed_value"] == {"cube": SOURCE_HASH}
