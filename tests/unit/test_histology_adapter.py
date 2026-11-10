"""Targeted coverage for task 15.1 Histology dataset adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark.adapters.histology import (
    CACHED_SPATIAL_MEAN_PCA_KEY,
    DEFAULT_HISTOLOGY_ROOT,
    EXPECTED_PATIENT_COUNT,
    EXPECTED_ROI_COUNT,
    FULL_CUBE_SPATIAL_PCA_KEY,
    HistologyAdapterError,
    HistologyHSIGBAdapter,
    HistologyInputHasher,
    RepresentationSourceKind,
)
from benchmark.schema.canonical import hash_file


VALID_HASH = "a" * 64


def metadata_only_adapter() -> HistologyHSIGBAdapter:
    # Discovery and schema tests avoid re-reading 622 GB of raw bytes. The real
    # materialization path uses HistologyInputHasher and is exercised separately.
    return HistologyHSIGBAdapter(input_hasher=lambda reference: VALID_HASH)  # type: ignore[arg-type]


def test_discovers_exact_roi_patient_and_label_sets_from_approved_root() -> None:
    records = tuple(metadata_only_adapter().iter_patient_grouped_samples())

    assert len(records) == EXPECTED_ROI_COUNT == 469
    assert len({record.sample_id for record in records}) == 469
    assert {record.group_id for record in records} == {f"P{index}" for index in range(1, 14)}
    assert len({record.group_id for record in records}) == EXPECTED_PATIENT_COUNT == 13

    labels = [record.label_schema_metadata["label_value"] for record in records]
    assert labels.count(0) == 303
    assert labels.count(1) == 166
    assert all(len(record.label_schema_metadata["label_content_hash"]) == 64 for record in records)
    assert all(record.group_id == record.sample_id.split("/", 1)[0] for record in records)


def test_manifest_inputs_are_relative_hashed_and_cover_both_representation_sources() -> None:
    adapter = metadata_only_adapter()
    records = tuple(adapter.iter_samples())
    expected_names = {
        "raw_cube",
        "raw_header",
        "rgb_preview",
        "cached_spatial_mean_pca",
    }

    assert all(set(record.input_references) == expected_names for record in records)
    assert all(set(record.input_content_hashes) == expected_names for record in records)
    assert all(
        not Path(reference).is_absolute() and not reference.casefold().startswith("f:")
        for record in records
        for reference in record.input_references.values()
    )
    for representation in adapter.representations.values():
        assert set(representation.required_input_names).issubset(expected_names)


def test_cached_and_full_cube_pca_have_distinct_governed_keys_and_access_policies() -> None:
    adapter = metadata_only_adapter()
    cached = adapter.representation(CACHED_SPATIAL_MEAN_PCA_KEY)
    full_cube = adapter.representation(FULL_CUBE_SPATIAL_PCA_KEY)

    assert cached.representation_key != full_cube.representation_key
    assert cached.source_kind is RepresentationSourceKind.IMMUTABLE_DERIVED_CACHE
    assert full_cube.source_kind is RepresentationSourceKind.READ_ONLY_RAW_CUBE
    assert "raw_cube" not in cached.required_input_names
    assert "raw_cube" in full_cube.required_input_names
    assert "fold-scoped" in full_cube.pca_fit_scope

    with pytest.raises(HistologyAdapterError, match="unknown Histology Representation_Key"):
        adapter.representation("histology.ambiguous-pca")


def test_raw_cube_opens_as_read_only_memmap_and_supports_bounded_chunks() -> None:
    adapter = metadata_only_adapter()
    sample_id = next(iter(adapter.iter_samples())).sample_id

    with adapter.open_raw_cube(sample_id) as reader:
        assert reader.source_format == "envi-bil"
        assert reader.shape == (826, 800, 1004)
        assert reader.dtype == np.dtype("uint16")
        assert not reader.memmap_view().flags.writeable
        chunk = next(reader.iter_chunks(band_chunk_size=2, spatial_chunk_shape=(3, 4)))
        assert chunk.data.shape == (2, 3, 4)
        assert not chunk.data.flags.writeable


def test_input_hasher_returns_real_content_hash_and_rejects_root_escape() -> None:
    hasher = HistologyInputHasher(DEFAULT_HISTOLOGY_ROOT)
    reference = "P1/ROI_01_C01_T/rgb.png"

    assert hasher(reference) == hash_file(DEFAULT_HISTOLOGY_ROOT / reference)
    # The second call uses the stat-keyed memoized digest.
    assert hasher(reference) == hasher(reference)

    for forbidden in ("../outside", "F:/forbidden/raw", "/absolute/path"):
        with pytest.raises(HistologyAdapterError, match="relative|approved"):
            hasher(forbidden)


def test_adapter_rejects_unapproved_or_non_g_drive_root(tmp_path: Path) -> None:
    with pytest.raises(HistologyAdapterError, match="differs from approved root"):
        HistologyHSIGBAdapter(DEFAULT_HISTOLOGY_ROOT, approved_root=tmp_path)

    with pytest.raises(HistologyAdapterError, match="approved G: drive"):
        HistologyHSIGBAdapter(
            Path(r"F:\Datasets\PKG - HistologyHSI-GB"),
            approved_root=Path(r"F:\Datasets\PKG - HistologyHSI-GB"),
        )
