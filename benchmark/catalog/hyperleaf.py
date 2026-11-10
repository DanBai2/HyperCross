"""Governed HyperLeaf2024 manifests and representation builders.

Runtime paths stay outside frozen configuration. Official metadata comes only
from ``test.csv``; this module never opens the official ``solution.csv``.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from benchmark.catalog.materialization import DatasetCatalog, LocalInputHasher
from benchmark.catalog.raw_cube import HyperspectralCubeReader, open_tiff_cube
from benchmark.schema.base import TaskKind
from benchmark.schema.canonical import hash_object
from benchmark.schema.datasets import DatasetSpec, LabelSchema, LabelValueKind, SampleManifest, SampleRecord
from benchmark.schema.preprocessing import FitBoundary, FittedArtifact
from benchmark.transforms.preprocessing import PreprocessingManager

HYPERLEAF_DATASET_KEY = "hyperleaf2024"
HYPERLEAF_TRAIN_COUNT = 1590
HYPERLEAF_OFFICIAL_TEST_COUNT = 820
HYPERLEAF_TARGETS = ("GrainWeight", "Gsw", "PhiPS2")
HYPERLEAF_VARIETIES = ("Heerup", "Kvium", "Rembrandt", "Sheriff")
VISIBLE_RGB_KEY = "hyperleaf.visible-rgb.v1"
GLOBAL_SPECTRUM_KEY = "hyperleaf.global-spectrum.v1"
PCA_PSEUDO_RGB_KEY = "hyperleaf.pca-pseudo-rgb.fold-fitted.v1"
FULL_CUBE_KEY = "hyperleaf.full-cube.bhw.v1"
HYPERLEAF_REPRESENTATION_KEYS = (VISIBLE_RGB_KEY, GLOBAL_SPECTRUM_KEY, PCA_PSEUDO_RGB_KEY, FULL_CUBE_KEY)

class HyperLeafPartition(str, Enum):
    TRAIN = "train"
    OFFICIAL_TEST = "official_test"

class HyperLeafDataError(RuntimeError):
    """The approved HyperLeaf layout or metadata is invalid."""

class OfficialLabelAccessError(PermissionError):
    """Withheld official-test targets were requested."""

@dataclass(frozen=True)
class HyperLeafManifestPair:
    training: SampleManifest
    official_test: SampleManifest

@dataclass(frozen=True)
class RepresentationDescriptor:
    representation_key: str
    modalities: tuple[str, ...]
    dtype: str
    channel_or_band_order: tuple[str, ...]
    learned_state_required: bool

REPRESENTATION_DESCRIPTORS: Mapping[str, RepresentationDescriptor] = {
    VISIBLE_RGB_KEY: RepresentationDescriptor(VISIBLE_RGB_KEY, ("visible_rgb",), "float32", ("red", "green", "blue"), False),
    GLOBAL_SPECTRUM_KEY: RepresentationDescriptor(GLOBAL_SPECTRUM_KEY, ("global_spectrum",), "float32", ("wavelength_ascending",), False),
    PCA_PSEUDO_RGB_KEY: RepresentationDescriptor(PCA_PSEUDO_RGB_KEY, ("pca_pseudo_rgb",), "float32", ("PC1", "PC2", "PC3"), True),
    FULL_CUBE_KEY: RepresentationDescriptor(FULL_CUBE_KEY, ("full_cube",), "source", ("band", "row", "column"), False),
}

class HyperLeafFileHasher(LocalInputHasher):
    """Local hasher with a process-local cache for repeated validation."""
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self._cache: dict[Path, str] = {}

    def __call__(self, reference: str) -> str:
        path = self.resolve(reference)
        digest = self._cache.get(path)
        if digest is None:
            digest = super().__call__(reference)
            self._cache[path] = digest
        return digest

class HyperLeaf2024Adapter:
    """Read-only adapter for one official HyperLeaf2024 partition."""
    def __init__(self, root: str | Path, *, partition: HyperLeafPartition | str = HyperLeafPartition.TRAIN,
                 input_hasher: HyperLeafFileHasher | None = None) -> None:
        self._root = Path(root).resolve()
        self._reject_disallowed_drive(self._root)
        if not self._root.is_dir():
            raise HyperLeafDataError(f"HyperLeaf root is not a directory: {self._root}")
        self.partition = HyperLeafPartition(partition)
        self.input_hasher = input_hasher or HyperLeafFileHasher(self._root)
        if self.input_hasher.root != self._root:
            raise HyperLeafDataError("input hasher root must equal the HyperLeaf runtime root")
        self._row_cache: dict[HyperLeafPartition, tuple[dict[str, str], ...]] = {}
        self._sample_cache: dict[HyperLeafPartition, dict[str, dict[str, str]]] = {}

    @staticmethod
    def _reject_disallowed_drive(path: Path) -> None:
        drive = PureWindowsPath(str(path)).drive.upper()
        if drive and drive != "G:":
            raise HyperLeafDataError("HyperLeaf runtime data must be read from the G: drive")

    @property
    def label_schema(self) -> LabelSchema:
        return LabelSchema(schema_key="hyperleaf.targets.v1", value_kind=LabelValueKind.CONTINUOUS,
                           dimension=3, target_names=HYPERLEAF_TARGETS)

    @property
    def training_spec(self) -> DatasetSpec:
        return DatasetSpec(dataset_key=HYPERLEAF_DATASET_KEY, task_kind=TaskKind.MULTI_TARGET_REGRESSION,
                           expected_sample_count=HYPERLEAF_TRAIN_COUNT, label_schema=self.label_schema, grouped=True)

    @property
    def official_test_spec(self) -> DatasetSpec:
        # test.csv has no wheat-variety field. It remains ungrouped rather than
        # fabricating a variety or reading the label-bearing solution.csv.
        return DatasetSpec(dataset_key=HYPERLEAF_DATASET_KEY, task_kind=TaskKind.MULTI_TARGET_REGRESSION,
                           expected_sample_count=HYPERLEAF_OFFICIAL_TEST_COUNT,
                           label_schema=self.label_schema, grouped=False)

    @property
    def spec(self) -> DatasetSpec:
        return self.training_spec if self.partition is HyperLeafPartition.TRAIN else self.official_test_spec

    def frozen_config(self) -> dict[str, str]:
        return {"dataset_key": HYPERLEAF_DATASET_KEY}

    def with_partition(self, partition: HyperLeafPartition | str) -> "HyperLeaf2024Adapter":
        adapter = HyperLeaf2024Adapter(self._root, partition=partition, input_hasher=self.input_hasher)
        adapter._row_cache, adapter._sample_cache = self._row_cache, self._sample_cache
        return adapter

    def iter_samples(self) -> Iterable[SampleRecord]:
        return self.iter_training_samples() if self.partition is HyperLeafPartition.TRAIN else self.iter_official_test_samples()

    def iter_training_samples(self) -> Iterable[SampleRecord]:
        for row in self._rows(HyperLeafPartition.TRAIN):
            sample_id = self._sample_id(row)
            variety = self._training_variety(row, sample_id)
            targets = self._target_values(row, sample_id)
            yield self._record(sample_id, group_id=variety,
                label_reference=f"labels://{HYPERLEAF_DATASET_KEY}/train/{sample_id}",
                label_metadata={"schema_key": "hyperleaf.targets.v1", "dimension": 3,
                    "partition": "train", "labels_available": True,
                    "label_content_hash": hash_object(dict(zip(HYPERLEAF_TARGETS, targets))),
                    "group_semantics": "wheat_variety"})

    def iter_official_test_samples(self) -> Iterable[SampleRecord]:
        for row in self._rows(HyperLeafPartition.OFFICIAL_TEST):
            sample_id = self._sample_id(row)
            yield self._record(sample_id, group_id=None,
                label_reference=f"withheld://{HYPERLEAF_DATASET_KEY}/official-test/{sample_id}",
                label_metadata={"schema_key": "hyperleaf.targets.v1", "dimension": 3,
                    "partition": "official_test", "labels_available": False,
                    "group_semantics": "withheld_in_test_csv"})

    def validate_partition_isolation(self) -> None:
        overlap = set(self.sample_ids(HyperLeafPartition.TRAIN)) & set(self.sample_ids(HyperLeafPartition.OFFICIAL_TEST))
        if overlap:
            raise HyperLeafDataError(f"training and official-test Sample_IDs overlap: {sorted(overlap)}")

    def sample_ids(self, partition: HyperLeafPartition | str) -> tuple[str, ...]:
        return tuple(self._sample_id(row) for row in self._rows(HyperLeafPartition(partition)))

    def cube_reference(self, sample_id: str) -> str:
        normalized = self._normalize_sample_id(sample_id)
        if not any(normalized in self._sample_rows(partition) for partition in HyperLeafPartition):
            raise KeyError(f"unknown HyperLeaf Sample_ID: {normalized}")
        return f"images/{normalized}.tiff"

    def cube_path(self, sample_id: str) -> Path:
        return self.input_hasher.resolve(self.cube_reference(sample_id))

    def training_targets(self, sample_ids: Sequence[str], boundary: FitBoundary,
                         *, require_complete_boundary: bool) -> np.ndarray:
        ids = self.validate_training_fit_ids(sample_ids, boundary,
                                             require_complete_boundary=require_complete_boundary)
        rows = self._sample_rows(HyperLeafPartition.TRAIN)
        return np.asarray([self._target_values(rows[sample_id], sample_id) for sample_id in ids], dtype=np.float64)

    def validate_training_fit_ids(self, sample_ids: Sequence[str], boundary: FitBoundary,
                                  *, require_complete_boundary: bool) -> tuple[str, ...]:
        ids = tuple(self._normalize_sample_id(sample_id) for sample_id in sample_ids)
        if not ids or len(ids) != len(set(ids)):
            raise HyperLeafDataError("fit Sample_IDs must be non-empty and unique")
        supplied, allowed = set(ids), set(boundary.allowed_fit_sample_ids)
        forbidden = supplied - allowed
        if forbidden or supplied & set(boundary.outer_test_sample_ids):
            raise OfficialLabelAccessError(
                f"fit Sample_IDs are outside the permitted training boundary: {sorted(forbidden)}")
        if supplied & set(self.sample_ids(HyperLeafPartition.OFFICIAL_TEST)):
            raise OfficialLabelAccessError("official-test samples cannot provide fitted state")
        unknown = supplied - set(self.sample_ids(HyperLeafPartition.TRAIN))
        if unknown:
            raise HyperLeafDataError(f"unknown training Sample_IDs: {sorted(unknown)}")
        if require_complete_boundary and supplied != allowed:
            raise HyperLeafDataError("target normalization requires all and only allowed training Sample_IDs")
        return ids

    def _record(self, sample_id: str, *, group_id: str | None, label_reference: str,
                label_metadata: dict[str, object]) -> SampleRecord:
        reference = f"images/{sample_id}.tiff"
        return SampleRecord(sample_id=sample_id, group_id=group_id,
            label_reference=label_reference, input_references={"full_cube": reference},
            input_content_hashes={"full_cube": self.input_hasher(reference)},
            label_schema_metadata=label_metadata)

    def _rows(self, partition: HyperLeafPartition) -> tuple[dict[str, str], ...]:
        cached = self._row_cache.get(partition)
        if cached is not None:
            return cached
        filename = "train.csv" if partition is HyperLeafPartition.TRAIN else "test.csv"
        path = self._root / filename
        if not path.is_file():
            raise HyperLeafDataError(f"required HyperLeaf metadata is missing: {filename}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            required = ("ImageId",) + (HYPERLEAF_TARGETS + HYPERLEAF_VARIETIES
                                       if partition is HyperLeafPartition.TRAIN else ())
            missing = tuple(field for field in required if field not in fields)
            if missing:
                raise HyperLeafDataError(f"{filename} is missing required columns: {missing}")
            if partition is HyperLeafPartition.OFFICIAL_TEST:
                forbidden = tuple(field for field in HYPERLEAF_TARGETS if field in fields)
                if forbidden:
                    raise HyperLeafDataError(f"official test metadata exposes forbidden target columns: {forbidden}")
            rows = tuple(dict(row) for row in reader)
        expected = HYPERLEAF_TRAIN_COUNT if partition is HyperLeafPartition.TRAIN else HYPERLEAF_OFFICIAL_TEST_COUNT
        if len(rows) != expected:
            raise HyperLeafDataError(f"{filename} sample count mismatch: expected {expected}, observed {len(rows)}")
        ids = tuple(self._sample_id(row) for row in rows)
        if len(ids) != len(set(ids)):
            raise HyperLeafDataError(f"{filename} contains duplicate ImageId values")
        self._row_cache[partition] = rows
        self._sample_cache[partition] = dict(zip(ids, rows))
        return rows

    def _sample_rows(self, partition: HyperLeafPartition) -> dict[str, dict[str, str]]:
        self._rows(partition)
        return self._sample_cache[partition]

    @staticmethod
    def _normalize_sample_id(value: object) -> str:
        text = str(value).strip()
        if not text or not text.isdigit():
            raise HyperLeafDataError(f"invalid HyperLeaf ImageId: {value!r}")
        return text.zfill(5)

    def _sample_id(self, row: Mapping[str, str]) -> str:
        return self._normalize_sample_id(row.get("ImageId", ""))

    @staticmethod
    def _target_values(row: Mapping[str, str], sample_id: str) -> tuple[float, float, float]:
        try:
            values = tuple(float(row[name]) for name in HYPERLEAF_TARGETS)
        except (KeyError, TypeError, ValueError) as exc:
            raise HyperLeafDataError(f"invalid training targets for {sample_id}") from exc
        if not all(math.isfinite(value) for value in values):
            raise HyperLeafDataError(f"non-finite training targets for {sample_id}")
        return values  # type: ignore[return-value]

    @staticmethod
    def _training_variety(row: Mapping[str, str], sample_id: str) -> str:
        try:
            indicators = tuple(float(row[name]) for name in HYPERLEAF_VARIETIES)
        except (KeyError, TypeError, ValueError) as exc:
            raise HyperLeafDataError(f"invalid variety indicators for {sample_id}") from exc
        active = tuple(name for name, value in zip(HYPERLEAF_VARIETIES, indicators) if value == 1.0)
        if len(active) != 1 or any(value not in (0.0, 1.0) for value in indicators):
            raise HyperLeafDataError(f"sample {sample_id} must have exactly one binary wheat-variety indicator")
        return active[0]

class HyperLeafRepresentationBuilder:
    """Build deterministic inputs and leakage-safe fitted representations."""
    def __init__(self, adapter: HyperLeaf2024Adapter, *, preprocessing: PreprocessingManager | None = None,
                 visible_size: tuple[int, int] = (32, 64), spectrum_length: int = 128) -> None:
        if any(value <= 0 for value in (*visible_size, spectrum_length)):
            raise ValueError("representation dimensions must be positive")
        self.adapter = adapter
        self.preprocessing = preprocessing or PreprocessingManager()
        self.visible_size, self.spectrum_length = visible_size, spectrum_length

    @property
    def representation_keys(self) -> tuple[str, ...]:
        return HYPERLEAF_REPRESENTATION_KEYS

    def descriptor(self, representation_key: str) -> RepresentationDescriptor:
        try:
            return REPRESENTATION_DESCRIPTORS[representation_key]
        except KeyError as exc:
            raise KeyError(f"unknown HyperLeaf Representation_Key: {representation_key}") from exc

    def open_full_cube(self, sample_id: str) -> HyperspectralCubeReader:
        return open_tiff_cube(self.adapter.cube_path(sample_id), axis_order="BHW")

    def visible_rgb(self, sample_id: str) -> np.ndarray:
        with self.open_full_cube(sample_id) as reader:
            bands = reader.shape[0]
            indices = tuple(min(bands - 1, int(bands * ratio)) for ratio in (0.72, 0.48, 0.20))
            cube = reader.memmap_view()
            rgb = np.stack([np.asarray(cube[index], dtype=np.float32) for index in indices])
        rgb = self._source_scale(rgb)
        height, width = self.visible_size
        image = Image.fromarray(np.moveaxis((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 0, -1))
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        result = np.moveaxis(np.asarray(resized, dtype=np.float32) / 255.0, -1, 0)
        result.flags.writeable = False
        return result

    def global_spectrum(self, sample_id: str) -> np.ndarray:
        raw = self._raw_global_spectrum(sample_id)
        spectrum = np.interp(np.linspace(0, 1, self.spectrum_length), np.linspace(0, 1, raw.size), raw)
        scale = float(spectrum.std())
        result = ((spectrum - spectrum.mean()) / (scale if scale > 1e-12 else 1.0)).astype(np.float32)
        result.flags.writeable = False
        return result

    def fit_target_normalization(self, sample_ids: Sequence[str], boundary: FitBoundary,
                                 *, dataset_hash: str, protocol_hash: str) -> FittedArtifact:
        targets = self.adapter.training_targets(sample_ids, boundary, require_complete_boundary=True)
        return self.preprocessing.fit_target_normalization(targets, sample_ids, boundary,
            dataset_hash=dataset_hash, protocol_hash=protocol_hash)

    def fit_pca_pseudo_rgb(self, sample_ids: Sequence[str], boundary: FitBoundary,
                           *, dataset_hash: str, protocol_hash: str) -> FittedArtifact:
        ids = self.adapter.validate_training_fit_ids(sample_ids, boundary, require_complete_boundary=False)
        spectra = np.stack([self._raw_global_spectrum(sample_id) for sample_id in ids])
        return self.preprocessing.fit_pca(spectra, ids, boundary, dataset_hash=dataset_hash,
                                          protocol_hash=protocol_hash, n_components=3, whiten=False)

    def pca_pseudo_rgb(self, sample_id: str, artifact: FittedArtifact,
                       boundary: FitBoundary) -> np.ndarray:
        with self.open_full_cube(sample_id) as reader:
            cube = np.asarray(reader.memmap_view(), dtype=np.float32)
            bands, height, width = cube.shape
            pixels = np.moveaxis(cube, 0, -1).reshape(-1, bands)
            projected = self.preprocessing.apply(artifact, pixels, boundary)
        image = projected.reshape(height, width, 3)
        minimum, maximum = image.min((0, 1), keepdims=True), image.max((0, 1), keepdims=True)
        denominator = np.where(maximum - minimum > 1e-12, maximum - minimum, 1.0)
        result = np.moveaxis(((image - minimum) / denominator).astype(np.float32), -1, 0)
        result.flags.writeable = False
        return result

    def _raw_global_spectrum(self, sample_id: str) -> np.ndarray:
        with self.open_full_cube(sample_id) as reader:
            result = np.asarray(reader.memmap_view(), dtype=np.float64).mean(axis=(1, 2))
        if not np.all(np.isfinite(result)):
            raise HyperLeafDataError(f"cube contains non-finite values: {sample_id}")
        return result

    @staticmethod
    def _source_scale(values: np.ndarray) -> np.ndarray:
        if np.issubdtype(values.dtype, np.integer):
            return values.astype(np.float32) / float(np.iinfo(values.dtype).max)
        maximum = float(np.nanmax(values))
        return values.astype(np.float32) / (65535.0 if maximum > 1.0 else 1.0)

def materialize_hyperleaf_manifests(catalog: DatasetCatalog,
                                    adapter: HyperLeaf2024Adapter) -> HyperLeafManifestPair:
    """Validate partition isolation and atomically publish both manifests."""
    adapter.validate_partition_isolation()
    training = catalog.materialize_manifest(adapter.with_partition(HyperLeafPartition.TRAIN))
    official = catalog.materialize_manifest(adapter.with_partition(HyperLeafPartition.OFFICIAL_TEST))
    if set(training.canonical_order) & set(official.canonical_order):
        raise HyperLeafDataError("materialized train and official-test manifests overlap")
    return HyperLeafManifestPair(training=training, official_test=official)
