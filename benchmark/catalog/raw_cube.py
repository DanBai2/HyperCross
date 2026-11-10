"""Read-only, bounded access to TIFF and ENVI hyperspectral cubes.

Readers expose a canonical ``(bands, rows, columns)`` view without loading a
whole cube into memory. TIFF input must be directly memory-mappable; compressed
or otherwise unsupported TIFF layouts are rejected instead of falling back to
an eager read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile


PathLike = Union[str, Path]
CubeShape = Tuple[int, int, int]
SpatialChunkShape = Tuple[int, int]


class RawCubeError(RuntimeError):
    """Base error for raw hyperspectral cube access."""


class RawCubeFormatError(RawCubeError):
    """Raised when a TIFF or ENVI file cannot be memory-mapped safely."""


class RawCubeValidationError(RawCubeError):
    """Raised when cube shape, dtype, header, or chunk inputs are invalid."""


@dataclass(frozen=True)
class CubeChunk:
    """One bounded, read-only cube view and its canonical source slices."""

    band_slice: slice
    row_slice: slice
    column_slice: slice
    data: np.ndarray


class HyperspectralCubeReader:
    """A read-only memory-mapped cube with canonical band-first indexing."""

    def __init__(
        self,
        *,
        source_path: Path,
        storage: np.memmap,
        canonical_view: np.ndarray,
        source_format: str,
    ) -> None:
        if canonical_view.ndim != 3:
            raise RawCubeValidationError(
                f"hyperspectral cube must be three-dimensional, got {canonical_view.ndim} dimensions"
            )
        if any(dimension <= 0 for dimension in canonical_view.shape):
            raise RawCubeValidationError(
                f"hyperspectral cube dimensions must be positive, got {canonical_view.shape}"
            )
        storage.flags.writeable = False
        canonical_view.flags.writeable = False
        self._source_path = source_path
        self._storage = storage
        self._view = canonical_view
        self._source_format = source_format
        self._closed = False

    @property
    def source_path(self) -> Path:
        return self._source_path

    @property
    def source_format(self) -> str:
        return self._source_format

    @property
    def shape(self) -> CubeShape:
        """Return canonical ``(bands, rows, columns)`` shape."""

        return tuple(int(value) for value in self._view.shape)  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return self._view.dtype

    def memmap_view(self) -> np.ndarray:
        """Return the read-only canonical view backed by the mapped file."""

        self._ensure_open()
        return self._view

    def read_window(
        self,
        *,
        bands: Optional[slice] = None,
        rows: Optional[slice] = None,
        columns: Optional[slice] = None,
    ) -> np.ndarray:
        """Return a read-only band/spatial window without copying the cube."""

        self._ensure_open()
        selectors = (
            _normalized_slice("bands", bands, self.shape[0]),
            _normalized_slice("rows", rows, self.shape[1]),
            _normalized_slice("columns", columns, self.shape[2]),
        )
        window = np.asarray(self._view[selectors])
        window.flags.writeable = False
        return window

    def iter_chunks(
        self,
        *,
        band_chunk_size: int,
        spatial_chunk_shape: Optional[SpatialChunkShape] = None,
    ) -> Iterator[CubeChunk]:
        """Yield deterministic band-first chunks as read-only mapped views.

        ``spatial_chunk_shape`` is ``(row_count, column_count)``. Omitting it
        keeps each chunk at full spatial extent while still chunking bands.
        """

        self._ensure_open()
        band_size = _positive_integer("band_chunk_size", band_chunk_size)
        if spatial_chunk_shape is None:
            row_size, column_size = self.shape[1], self.shape[2]
        else:
            if (
                not isinstance(spatial_chunk_shape, Sequence)
                or isinstance(spatial_chunk_shape, (str, bytes))
                or len(spatial_chunk_shape) != 2
            ):
                raise RawCubeValidationError(
                    "spatial_chunk_shape must contain exactly two positive integers"
                )
            row_size = _positive_integer("spatial_chunk_shape[0]", spatial_chunk_shape[0])
            column_size = _positive_integer(
                "spatial_chunk_shape[1]", spatial_chunk_shape[1]
            )

        bands, rows, columns = self.shape
        for band_start in range(0, bands, band_size):
            band_slice = slice(band_start, min(band_start + band_size, bands))
            for row_start in range(0, rows, row_size):
                row_slice = slice(row_start, min(row_start + row_size, rows))
                for column_start in range(0, columns, column_size):
                    column_slice = slice(
                        column_start, min(column_start + column_size, columns)
                    )
                    yield CubeChunk(
                        band_slice=band_slice,
                        row_slice=row_slice,
                        column_slice=column_slice,
                        data=self.read_window(
                            bands=band_slice,
                            rows=row_slice,
                            columns=column_slice,
                        ),
                    )

    def close(self) -> None:
        """Close the underlying memory map. Calling this twice is safe."""

        if self._closed:
            return
        mapped_file = getattr(self._storage, "_mmap", None)
        if mapped_file is not None:
            mapped_file.close()
        self._closed = True

    def __enter__(self) -> "HyperspectralCubeReader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RawCubeError("hyperspectral cube reader is closed")


def open_tiff_cube(
    path: PathLike,
    *,
    axis_order: str = "BHW",
    expected_shape: Optional[CubeShape] = None,
    expected_dtype: Optional[Union[str, np.dtype, type]] = None,
) -> HyperspectralCubeReader:
    """Open a directly memory-mappable TIFF as a canonical band-first cube.

    ``axis_order`` describes the on-disk array with ``B`` for bands, ``H`` for
    rows, and ``W`` for columns. No eager TIFF fallback is used.
    """

    source = _existing_file(path, "TIFF")
    axes = _validate_axis_order(axis_order)
    try:
        storage = tifffile.memmap(source, mode="r")
    except Exception as error:
        raise RawCubeFormatError(
            f"TIFF is not directly memory-mappable: {source}"
        ) from error

    try:
        if storage.ndim != 3:
            raise RawCubeValidationError(
                f"TIFF hyperspectral cube must be three-dimensional, got shape {storage.shape}"
            )
        permutation = tuple(axes.index(axis) for axis in "BHW")
        canonical_view = np.transpose(storage, permutation)
        _validate_cube(canonical_view, expected_shape, expected_dtype)
        return HyperspectralCubeReader(
            source_path=source,
            storage=storage,
            canonical_view=canonical_view,
            source_format="tiff",
        )
    except Exception:
        mapped_file = getattr(storage, "_mmap", None)
        if mapped_file is not None:
            mapped_file.close()
        raise


def open_envi_cube(
    header_path: PathLike,
    *,
    raw_path: Optional[PathLike] = None,
    expected_shape: Optional[CubeShape] = None,
    expected_dtype: Optional[Union[str, np.dtype, type]] = None,
) -> HyperspectralCubeReader:
    """Open an ENVI BSQ/BIL/BIP cube using a read-only ``numpy.memmap``."""

    header = _existing_file(header_path, "ENVI header")
    fields = _parse_envi_header(header)
    samples = _required_positive_header_integer(fields, "samples")
    lines = _required_positive_header_integer(fields, "lines")
    bands = _required_positive_header_integer(fields, "bands")
    data_type = _required_header_integer(fields, "data type")
    interleave = fields.get("interleave", "").strip().lower()
    if interleave not in {"bsq", "bil", "bip"}:
        raise RawCubeValidationError(
            f"unsupported ENVI interleave {interleave!r}; expected bsq, bil, or bip"
        )
    byte_order = _optional_header_integer(fields, "byte order", default=0)
    if byte_order not in {0, 1}:
        raise RawCubeValidationError("ENVI byte order must be 0 (little) or 1 (big)")
    header_offset = _optional_header_integer(fields, "header offset", default=0)
    if header_offset < 0:
        raise RawCubeValidationError("ENVI header offset must be non-negative")

    dtype = _envi_dtype(data_type, byte_order)
    source = _resolve_envi_raw_path(header, fields, raw_path)
    source = _existing_file(source, "ENVI raw cube")
    expected_bytes = header_offset + samples * lines * bands * dtype.itemsize
    observed_bytes = source.stat().st_size
    if observed_bytes != expected_bytes:
        raise RawCubeValidationError(
            "ENVI raw cube byte size does not match header: "
            f"expected {expected_bytes}, observed {observed_bytes}"
        )

    storage_shape: Mapping[str, CubeShape] = {
        "bsq": (bands, lines, samples),
        "bil": (lines, bands, samples),
        "bip": (lines, samples, bands),
    }
    try:
        storage = np.memmap(
            source,
            dtype=dtype,
            mode="r",
            offset=header_offset,
            shape=storage_shape[interleave],
            order="C",
        )
    except Exception as error:
        raise RawCubeFormatError(f"cannot memory-map ENVI raw cube: {source}") from error

    if interleave == "bsq":
        canonical_view = storage
    elif interleave == "bil":
        canonical_view = np.transpose(storage, (1, 0, 2))
    else:
        canonical_view = np.transpose(storage, (2, 0, 1))

    try:
        _validate_cube(canonical_view, expected_shape, expected_dtype)
        return HyperspectralCubeReader(
            source_path=source,
            storage=storage,
            canonical_view=canonical_view,
            source_format=f"envi-{interleave}",
        )
    except Exception:
        mapped_file = getattr(storage, "_mmap", None)
        if mapped_file is not None:
            mapped_file.close()
        raise


def _parse_envi_header(path: Path) -> Mapping[str, str]:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
    except (OSError, UnicodeError) as error:
        raise RawCubeFormatError(f"cannot read ENVI header: {path}") from error
    lines = text.splitlines()
    if not lines or lines[0].strip().upper() != "ENVI":
        raise RawCubeValidationError("ENVI header must begin with the ENVI marker")

    fields: dict[str, str] = {}
    current_key: Optional[str] = None
    current_value: list[str] = []
    brace_depth = 0
    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped or (stripped.startswith(";") and current_key is None):
            continue
        if current_key is not None:
            current_value.append(stripped)
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                fields[current_key] = " ".join(current_value).strip().strip("{}").strip()
                current_key = None
                current_value = []
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        normalized_key = key.strip().lower()
        value = value.strip()
        depth = value.count("{") - value.count("}")
        if depth > 0:
            current_key = normalized_key
            current_value = [value]
            brace_depth = depth
        else:
            fields[normalized_key] = value.strip().strip("{}").strip()
    if current_key is not None:
        raise RawCubeValidationError(f"unterminated ENVI header value for {current_key!r}")
    return fields


def _resolve_envi_raw_path(
    header: Path, fields: Mapping[str, str], raw_path: Optional[PathLike]
) -> Path:
    if raw_path is not None:
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else header.parent / candidate
    declared = fields.get("data file")
    if declared:
        candidate = Path(declared)
        return candidate if candidate.is_absolute() else header.parent / candidate
    return header.with_suffix("")


def _envi_dtype(data_type: int, byte_order: int) -> np.dtype:
    dtype_by_code = {
        1: np.dtype("u1"),
        2: np.dtype("i2"),
        3: np.dtype("i4"),
        4: np.dtype("f4"),
        5: np.dtype("f8"),
        6: np.dtype("c8"),
        9: np.dtype("c16"),
        12: np.dtype("u2"),
        13: np.dtype("u4"),
        14: np.dtype("i8"),
        15: np.dtype("u8"),
    }
    if data_type not in dtype_by_code:
        raise RawCubeValidationError(f"unsupported ENVI data type code: {data_type}")
    dtype = dtype_by_code[data_type]
    if dtype.itemsize == 1:
        return dtype
    return dtype.newbyteorder("<" if byte_order == 0 else ">")


def _validate_cube(
    view: np.ndarray,
    expected_shape: Optional[CubeShape],
    expected_dtype: Optional[Union[str, np.dtype, type]],
) -> None:
    actual_shape = tuple(int(value) for value in view.shape)
    if any(value <= 0 for value in actual_shape):
        raise RawCubeValidationError(f"cube dimensions must be positive, got {actual_shape}")
    if expected_shape is not None:
        normalized_shape = _normalize_expected_shape(expected_shape)
        if actual_shape != normalized_shape:
            raise RawCubeValidationError(
                f"cube shape mismatch: expected {normalized_shape}, observed {actual_shape}"
            )
    if expected_dtype is not None:
        normalized_dtype = np.dtype(expected_dtype)
        if not _dtype_matches(view.dtype, normalized_dtype):
            raise RawCubeValidationError(
                f"cube dtype mismatch: expected {normalized_dtype}, observed {view.dtype}"
            )


def _dtype_matches(observed: np.dtype, expected: np.dtype) -> bool:
    if observed.kind != expected.kind or observed.itemsize != expected.itemsize:
        return False
    if expected.itemsize == 1 or expected.byteorder in {"=", "|"}:
        return True
    return observed.byteorder == expected.byteorder


def _normalize_expected_shape(shape: CubeShape) -> CubeShape:
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 3
    ):
        raise RawCubeValidationError("expected_shape must contain exactly three integers")
    return tuple(
        _positive_integer(f"expected_shape[{index}]", value)
        for index, value in enumerate(shape)
    )  # type: ignore[return-value]


def _validate_axis_order(axis_order: str) -> str:
    if not isinstance(axis_order, str):
        raise RawCubeValidationError("axis_order must be a string")
    normalized = axis_order.upper()
    if len(normalized) != 3 or set(normalized) != {"B", "H", "W"}:
        raise RawCubeValidationError(
            "axis_order must contain B, H, and W exactly once"
        )
    return normalized


def _required_positive_header_integer(fields: Mapping[str, str], key: str) -> int:
    return _positive_integer(f"ENVI {key}", _required_header_integer(fields, key))


def _required_header_integer(fields: Mapping[str, str], key: str) -> int:
    if key not in fields:
        raise RawCubeValidationError(f"ENVI header is missing required field {key!r}")
    return _parse_header_integer(key, fields[key])


def _optional_header_integer(
    fields: Mapping[str, str], key: str, *, default: int
) -> int:
    if key not in fields:
        return default
    return _parse_header_integer(key, fields[key])


def _parse_header_integer(key: str, value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise RawCubeValidationError(
            f"ENVI header field {key!r} must be an integer, got {value!r}"
        ) from error


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise RawCubeValidationError(f"{name} must be a positive integer")
    return int(value)


def _normalized_slice(name: str, value: Optional[slice], size: int) -> slice:
    if value is None:
        return slice(0, size, 1)
    if not isinstance(value, slice):
        raise RawCubeValidationError(f"{name} selector must be a slice")
    start, stop, step = value.indices(size)
    if step <= 0:
        raise RawCubeValidationError(f"{name} selector step must be positive")
    return slice(start, stop, step)


def _existing_file(path: PathLike, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise RawCubeFormatError(f"{description} file does not exist: {candidate}")
    return candidate


__all__ = [
    "CubeChunk",
    "CubeShape",
    "HyperspectralCubeReader",
    "RawCubeError",
    "RawCubeFormatError",
    "RawCubeValidationError",
    "SpatialChunkShape",
    "open_envi_cube",
    "open_tiff_cube",
]
