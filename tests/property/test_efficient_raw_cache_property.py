"""Property 35: Efficient raw access and immutable cache reuse.

**Validates: Requirements 14.19, 14.20, 14.26-14.28**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import tifffile
from hypothesis import given, settings, strategies as st

from benchmark.catalog import open_envi_cube, open_tiff_cube
from benchmark.schema import FeatureCacheKey, FeatureLineage, hash_file, hash_object
from benchmark.transforms import FeatureCache, FeatureCacheValidationError


_TIFF_AXIS_ORDERS = ("BHW", "BWH", "HBW", "HWB", "WBH", "WHB")
_ENVI_INTERLEAVES = ("bsq", "bil", "bip")


@dataclass(frozen=True)
class _RawCacheCase:
    storage_kind: str
    bands: int
    rows: int
    columns: int
    band_chunk_size: int
    row_chunk_size: int
    column_chunk_size: int
    value_offset: int


@st.composite
def _raw_cache_cases(draw: st.DrawFn) -> _RawCacheCase:
    bands = draw(st.integers(min_value=2, max_value=5))
    rows = draw(st.integers(min_value=2, max_value=5))
    columns = draw(st.integers(min_value=2, max_value=6))
    return _RawCacheCase(
        storage_kind=draw(
            st.sampled_from(
                tuple(f"tiff-{axes}" for axes in _TIFF_AXIS_ORDERS)
                + tuple(f"envi-{interleave}" for interleave in _ENVI_INTERLEAVES)
            )
        ),
        bands=bands,
        rows=rows,
        columns=columns,
        band_chunk_size=draw(st.integers(min_value=1, max_value=bands + 2)),
        row_chunk_size=draw(st.integers(min_value=1, max_value=rows + 2)),
        column_chunk_size=draw(st.integers(min_value=1, max_value=columns + 2)),
        value_offset=draw(st.integers(min_value=0, max_value=65_000)),
    )


def _cube(case: _RawCacheCase) -> np.ndarray:
    size = case.bands * case.rows * case.columns
    values = (np.arange(size, dtype=np.uint32) + case.value_offset) % 65_536
    return values.astype(np.uint16).reshape(case.bands, case.rows, case.columns)


def _write_tiff(root: Path, cube: np.ndarray, axis_order: str) -> Path:
    permutation = tuple("BHW".index(axis) for axis in axis_order)
    path = root / "generated cube.tiff"
    tifffile.imwrite(
        path,
        np.transpose(cube, permutation),
        contiguous=True,
        metadata=None,
    )
    return path


def _write_envi(root: Path, cube: np.ndarray, interleave: str) -> tuple[Path, Path]:
    header = root / "generated cube.hdr"
    raw = root / "generated cube.raw"
    storage = {
        "bsq": cube,
        "bil": cube.transpose(1, 0, 2),
        "bip": cube.transpose(1, 2, 0),
    }[interleave]
    storage.astype("<u2").tofile(raw)
    bands, lines, samples = cube.shape
    header.write_text(
        "\n".join(
            (
                "ENVI",
                f"samples = {samples}",
                f"lines = {lines}",
                f"bands = {bands}",
                "header offset = 0",
                "data type = 12",
                f"interleave = {interleave}",
                "byte order = 0",
                f"data file = {raw.name}",
            )
        ),
        encoding="utf-8",
    )
    return header, raw


def _is_memmap_backed(array: np.ndarray) -> bool:
    current: object = array
    visited: set[int] = set()
    while isinstance(current, np.ndarray) and id(current) not in visited:
        if isinstance(current, np.memmap):
            return True
        visited.add(id(current))
        current = current.base
    return False


def _change_one_source_byte(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        assert original
        stream.seek(-1, 2)
        stream.write(bytes((original[0] ^ 0xFF,)))
        stream.flush()


@settings(max_examples=100, deadline=None)
@given(case=_raw_cache_cases())
def test_chunked_memmap_access_and_source_bound_cache_reuse_are_safe(
    case: _RawCacheCase,
) -> None:
    """Chunks reconstruct the mapped cube and stale raw-derived cache data is rejected."""

    cube = _cube(case)
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "property 35 with spaces"
        root.mkdir(parents=True)

        if case.storage_kind.startswith("tiff-"):
            axis_order = case.storage_kind.removeprefix("tiff-")
            source_path = _write_tiff(root, cube, axis_order)
            reader_context = open_tiff_cube(
                source_path,
                axis_order=axis_order,
                expected_shape=cube.shape,
                expected_dtype=np.uint16,
            )
        else:
            interleave = case.storage_kind.removeprefix("envi-")
            header_path, source_path = _write_envi(root, cube, interleave)
            reader_context = open_envi_cube(
                header_path,
                expected_shape=cube.shape,
                expected_dtype=np.uint16,
            )

        with reader_context as reader:
            mapped = reader.memmap_view()
            assert _is_memmap_backed(mapped)
            assert mapped.flags.writeable is False
            np.testing.assert_array_equal(mapped, cube)

            rebuilt = np.empty_like(cube)
            coverage = np.zeros(cube.shape, dtype=np.uint8)
            for chunk in reader.iter_chunks(
                band_chunk_size=case.band_chunk_size,
                spatial_chunk_shape=(case.row_chunk_size, case.column_chunk_size),
            ):
                selectors = (
                    chunk.band_slice,
                    chunk.row_slice,
                    chunk.column_slice,
                )
                assert chunk.data.flags.writeable is False
                assert np.shares_memory(chunk.data, mapped)
                np.testing.assert_array_equal(chunk.data, mapped[selectors])
                rebuilt[selectors] = chunk.data
                coverage[selectors] += 1

            np.testing.assert_array_equal(coverage, np.ones(cube.shape, dtype=np.uint8))
            np.testing.assert_array_equal(rebuilt, mapped)
            feature = rebuilt.mean(axis=(1, 2), dtype=np.float64)

        source_hash = hash_file(source_path)
        transform_config = {"kind": "band-spatial-mean", "version": 1}
        transform_hash = hash_object(transform_config)
        key = FeatureCacheKey(
            dataset_hash=hash_object({"dataset": case.storage_kind}),
            protocol_hash=hash_object({"protocol": "property-35"}),
            fold_id="outer-1",
            representation_key="raw-band-spatial-mean.v1",
            transform_state_hash=transform_hash,
        )
        lineage = FeatureLineage.create(
            source_hashes={"cube": source_hash},
            transform_config=transform_config,
            fit_sample_ids=("raw-sample-1",),
            transform_hash=transform_hash,
            output_shape=feature.shape,
        )
        cache = FeatureCache(root / "feature cache")
        cache.put_atomic(key, feature, lineage)

        reused = cache.get(key, lineage)
        np.testing.assert_array_equal(reused, feature)
        assert reused.flags.writeable is False

        entry_path = cache.entry_path(key)
        sentinel_path = root / "completed" / "sentinel.bin"
        sentinel_path.parent.mkdir(parents=True)
        sentinel_path.write_bytes(b"completed-artifact-must-not-change")
        entry_before = entry_path.read_bytes()
        sentinel_before = sentinel_path.read_bytes()

        _change_one_source_byte(source_path)
        current_source_hash = hash_file(source_path)
        assert current_source_hash != source_hash
        current_lineage = FeatureLineage.create(
            source_hashes={"cube": current_source_hash},
            transform_config=transform_config,
            fit_sample_ids=("raw-sample-1",),
            transform_hash=transform_hash,
            output_shape=feature.shape,
        )

        with pytest.raises(FeatureCacheValidationError) as caught:
            cache.get(key, current_lineage)

        record = caught.value.error_record
        assert record.error_code == "FEATURE_CACHE_SOURCE_HASH_MISMATCH"
        assert record.failed_stage == "feature_cache_lineage_validation"
        assert record.failing_field == "source_hashes"
        assert record.expected_value == {"cube": current_source_hash}
        assert record.observed_value == {"cube": source_hash}
        assert record.invariant_checks["cached_tensor_not_returned"] is True
        assert record.invariant_checks["rejected_entry_unchanged"] is True
        assert record.invariant_checks["completed_artifacts_unchanged"] is True
        assert entry_path.read_bytes() == entry_before
        assert sentinel_path.read_bytes() == sentinel_before
