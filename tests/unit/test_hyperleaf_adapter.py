"""Targeted coverage for task 14.1 HyperLeaf data and representations."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

import benchmark.catalog.hyperleaf as hyperleaf
from benchmark.catalog import (
    FULL_CUBE_KEY,
    GLOBAL_SPECTRUM_KEY,
    HYPERLEAF_REPRESENTATION_KEYS,
    PCA_PSEUDO_RGB_KEY,
    VISIBLE_RGB_KEY,
    DatasetCatalog,
    HyperLeaf2024Adapter,
    HyperLeafFileHasher,
    HyperLeafPartition,
    HyperLeafRepresentationBuilder,
    OfficialLabelAccessError,
    materialize_hyperleaf_manifests,
)
from benchmark.recovery.artifact_store import FailureAtomicArtifactStore
from benchmark.schema import FitBoundary, FitBoundaryScope

TRAIN_IDS = ("00000", "00001", "00003", "00007")
TEST_IDS = ("00002", "00004")

@pytest.fixture
def leaf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hyperleaf, "HYPERLEAF_TRAIN_COUNT", len(TRAIN_IDS))
    monkeypatch.setattr(hyperleaf, "HYPERLEAF_OFFICIAL_TEST_COUNT", len(TEST_IDS))
    root = tmp_path / "HyperLeaf2024"
    (root / "images").mkdir(parents=True)
    varieties = ("Heerup", "Kvium", "Rembrandt", "Sheriff")
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ImageId", "GrainWeight", "Gsw", "PhiPS2", *varieties))
        for index, sample_id in enumerate(TRAIN_IDS):
            one_hot = [int(position == index) for position in range(4)]
            writer.writerow((sample_id, 10 + index, 0.1 + index, 0.5 + index, *one_hot))
    with (root / "test.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ImageId",))
        writer.writerows((sample_id,) for sample_id in TEST_IDS)
    # Its presence must not grant adapter code official-label access.
    (root / "solution.csv").write_text("DO NOT OPEN\n", encoding="utf-8")
    for index, sample_id in enumerate((*TRAIN_IDS, *TEST_IDS)):
        cube = (np.arange(4 * 4 * 5, dtype=np.uint16).reshape(4, 4, 5) + index * 11)
        tifffile.imwrite(root / "images" / f"{sample_id}.tiff", cube)
    return root


def _boundary(*ids: str) -> FitBoundary:
    return FitBoundary(fold_id="outer-1", scope=FitBoundaryScope.OUTER_TRAIN,
                       allowed_fit_sample_ids=ids, outer_test_sample_ids=TEST_IDS)


def test_materializes_partition_manifests_with_variety_groups_and_no_label_leak(
    leaf_root: Path, tmp_path: Path
) -> None:
    hasher = HyperLeafFileHasher(leaf_root)
    adapter = HyperLeaf2024Adapter(leaf_root, input_hasher=hasher)
    catalog = DatasetCatalog(FailureAtomicArtifactStore(tmp_path / "artifacts"), input_hasher=hasher)

    manifests = materialize_hyperleaf_manifests(catalog, adapter)

    assert len(manifests.training.records) == 4
    assert len(manifests.official_test.records) == 2
    assert {record.group_id for record in manifests.training.records} == {
        "Heerup", "Kvium", "Rembrandt", "Sheriff"
    }
    assert all(record.group_id is None for record in manifests.official_test.records)
    assert all(record.label_reference.startswith("withheld://") for record in manifests.official_test.records)
    assert all(record.label_schema_metadata["labels_available"] is False
               for record in manifests.official_test.records)
    assert not set(manifests.training.canonical_order) & set(manifests.official_test.canonical_order)
    assert adapter.frozen_config() == {"dataset_key": "hyperleaf2024"}
    assert str(leaf_root) not in str(adapter.frozen_config())

def test_official_adapter_reads_test_metadata_without_opening_solution(
    leaf_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name == "solution.csv":
            raise AssertionError("official solution.csv was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    adapter = HyperLeaf2024Adapter(leaf_root, partition=HyperLeafPartition.OFFICIAL_TEST)
    assert tuple(record.sample_id for record in adapter.iter_official_test_samples()) == TEST_IDS


def test_representation_builders_publish_stable_keys_and_expected_shapes(leaf_root: Path) -> None:
    adapter = HyperLeaf2024Adapter(leaf_root)
    builder = HyperLeafRepresentationBuilder(adapter, visible_size=(3, 6), spectrum_length=8)

    assert builder.representation_keys == HYPERLEAF_REPRESENTATION_KEYS
    assert set(builder.representation_keys) == {
        VISIBLE_RGB_KEY, GLOBAL_SPECTRUM_KEY, PCA_PSEUDO_RGB_KEY, FULL_CUBE_KEY
    }
    rgb = builder.visible_rgb("00000")
    spectrum = builder.global_spectrum("00000")
    assert rgb.shape == (3, 3, 6) and rgb.dtype == np.float32
    assert spectrum.shape == (8,) and spectrum.dtype == np.float32
    assert np.isfinite(rgb).all() and np.isfinite(spectrum).all()
    with builder.open_full_cube("00000") as cube:
        assert cube.shape == (4, 4, 5)
        assert cube.memmap_view().flags.writeable is False


def test_fitted_representations_use_only_protocol_permitted_training_ids(leaf_root: Path) -> None:
    adapter = HyperLeaf2024Adapter(leaf_root)
    builder = HyperLeafRepresentationBuilder(adapter)
    boundary = _boundary(*TRAIN_IDS)
    dataset_hash, protocol_hash = "a" * 64, "b" * 64

    target_state = builder.fit_target_normalization(
        TRAIN_IDS, boundary, dataset_hash=dataset_hash, protocol_hash=protocol_hash)
    pca_state = builder.fit_pca_pseudo_rgb(
        TRAIN_IDS, boundary, dataset_hash=dataset_hash, protocol_hash=protocol_hash)
    pseudo_rgb = builder.pca_pseudo_rgb("00000", pca_state, boundary)

    assert target_state.fit_sample_ids == TRAIN_IDS
    assert pca_state.fit_sample_ids == TRAIN_IDS
    assert pca_state.dataset_hash == dataset_hash and pca_state.protocol_hash == protocol_hash
    assert pseudo_rgb.shape == (3, 4, 5) and pseudo_rgb.dtype == np.float32
    assert np.isfinite(pseudo_rgb).all()

def test_official_ids_are_rejected_before_pca_or_target_values_are_read(
    leaf_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HyperLeaf2024Adapter(leaf_root)
    builder = HyperLeafRepresentationBuilder(adapter)
    boundary = FitBoundary(fold_id="bad", scope=FitBoundaryScope.OUTER_TRAIN,
                           allowed_fit_sample_ids=(TEST_IDS[0],), outer_test_sample_ids=())

    def explode(sample_id: str) -> np.ndarray:
        raise AssertionError(f"cube was read for forbidden sample {sample_id}")

    monkeypatch.setattr(builder, "_raw_global_spectrum", explode)
    with pytest.raises(OfficialLabelAccessError):
        builder.fit_pca_pseudo_rgb((TEST_IDS[0],), boundary,
                                   dataset_hash="a" * 64, protocol_hash="b" * 64)
    with pytest.raises(OfficialLabelAccessError):
        builder.fit_target_normalization((TEST_IDS[0],), boundary,
                                         dataset_hash="a" * 64, protocol_hash="b" * 64)
