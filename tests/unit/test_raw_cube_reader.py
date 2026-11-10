"""Unit coverage for task 5.4 bounded TIFF and ENVI cube access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmark.catalog import (
    RawCubeError,
    RawCubeFormatError,
    RawCubeValidationError,
    open_envi_cube,
    open_tiff_cube,
)


def _write_envi_bil(root: Path, cube: np.ndarray) -> tuple[Path, Path]:
    """Write canonical BHW uint16 data as ENVI BIL."""

    header = root / "raw.hdr"
    raw = root / "raw"
    bands, lines, samples = cube.shape
    header.write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {samples}",
                f"lines = {lines}",
                f"bands = {bands}",
                "header offset = 0",
                "data type = 12",
                "interleave = bil",
                "byte order = 0",
            ]
        ),
        encoding="utf-8",
    )
    cube.transpose(1, 0, 2).astype("<u2").tofile(raw)
    return header, raw


def test_envi_bil_exposes_read_only_memmap_windows_and_complete_chunks(
    tmp_path: Path,
) -> None:
    cube = np.arange(5 * 4 * 7, dtype=np.uint16).reshape(5, 4, 7)
    header, raw = _write_envi_bil(tmp_path, cube)

    with open_envi_cube(
        header,
        expected_shape=(5, 4, 7),
        expected_dtype=np.uint16,
    ) as reader:
        assert reader.source_path == raw
        assert reader.source_format == "envi-bil"
        assert reader.shape == cube.shape
        assert reader.dtype.kind == "u"
        assert reader.dtype.itemsize == 2
        assert not reader.memmap_view().flags.writeable

        window = reader.read_window(
            bands=slice(1, 4), rows=slice(0, 4, 2), columns=slice(2, 7)
        )
        np.testing.assert_array_equal(window, cube[1:4, 0:4:2, 2:7])
        assert not window.flags.writeable
        with pytest.raises(ValueError):
            window[0, 0, 0] = 99

        rebuilt = np.empty_like(cube)
        chunks = list(
            reader.iter_chunks(
                band_chunk_size=2,
                spatial_chunk_shape=(3, 4),
            )
        )
        assert len(chunks) == 3 * 2 * 2
        for chunk in chunks:
            rebuilt[
                chunk.band_slice,
                chunk.row_slice,
                chunk.column_slice,
            ] = chunk.data
            assert not chunk.data.flags.writeable
        np.testing.assert_array_equal(rebuilt, cube)

    with pytest.raises(RawCubeError, match="closed"):
        reader.read_window(bands=slice(0, 1))


def test_envi_rejects_shape_dtype_and_file_size_mismatches(tmp_path: Path) -> None:
    cube = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    header, raw = _write_envi_bil(tmp_path, cube)

    with pytest.raises(RawCubeValidationError, match="shape mismatch"):
        open_envi_cube(header, expected_shape=(4, 2, 4))
    with pytest.raises(RawCubeValidationError, match="dtype mismatch"):
        open_envi_cube(header, expected_dtype=np.float32)

    raw.write_bytes(raw.read_bytes()[:-2])
    with pytest.raises(RawCubeValidationError, match="byte size"):
        open_envi_cube(header)


def test_tiff_axis_order_windows_and_band_chunks_are_memory_mapped(
    tmp_path: Path,
) -> None:
    canonical = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    stored_hwb = canonical.transpose(1, 2, 0)
    path = tmp_path / "cube.tiff"
    tifffile.imwrite(path, stored_hwb, contiguous=True, metadata=None)

    with open_tiff_cube(
        path,
        axis_order="HWB",
        expected_shape=(4, 3, 5),
        expected_dtype=np.float32,
    ) as reader:
        assert reader.source_format == "tiff"
        assert isinstance(reader.memmap_view().base, np.memmap)
        np.testing.assert_array_equal(
            reader.read_window(
                bands=slice(0, 4, 2),
                rows=slice(1, 3),
                columns=slice(0, 5, 2),
            ),
            canonical[0:4:2, 1:3, 0:5:2],
        )
        chunks = list(reader.iter_chunks(band_chunk_size=3))
        assert [chunk.data.shape for chunk in chunks] == [(3, 3, 5), (1, 3, 5)]
        np.testing.assert_array_equal(
            np.concatenate([chunk.data for chunk in chunks], axis=0), canonical
        )


def test_tiff_and_chunk_validation_fail_before_data_access(tmp_path: Path) -> None:
    path = tmp_path / "cube.tiff"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 4), dtype=np.uint16),
        contiguous=True,
        metadata=None,
    )

    with pytest.raises(RawCubeValidationError, match="axis_order"):
        open_tiff_cube(path, axis_order="BBH")
    with pytest.raises(RawCubeValidationError, match="dtype mismatch"):
        open_tiff_cube(path, expected_dtype=np.float32)

    with open_tiff_cube(path) as reader:
        with pytest.raises(RawCubeValidationError, match="positive integer"):
            list(reader.iter_chunks(band_chunk_size=0))
        with pytest.raises(RawCubeValidationError, match="selector must be a slice"):
            reader.read_window(bands=0)  # type: ignore[arg-type]

    missing = tmp_path / "missing.tiff"
    with pytest.raises(RawCubeFormatError, match="does not exist"):
        open_tiff_cube(missing)
