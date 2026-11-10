"""Governed HistologyHSI-GB dataset adapter.

The adapter discovers ROI samples under one explicitly approved G-drive root,
keeps all manifest references relative to that root, and delegates raw access to
the benchmark's read-only ENVI memory-map reader. It does not import or modify
legacy training code or caches.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath
from threading import Lock
import time
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union
from uuid import uuid4

from benchmark.catalog.materialization import DatasetCatalog
from benchmark.catalog.raw_cube import HyperspectralCubeReader, open_envi_cube
from benchmark.recovery.artifact_store import FailureAtomicArtifactStore
from benchmark.schema import (
    DatasetSpec,
    LabelSchema,
    LabelValueKind,
    SampleManifest,
    SampleRecord,
    TaskKind,
)
from benchmark.schema.canonical import hash_file, hash_object


DEFAULT_HISTOLOGY_ROOT = Path(r"G:\Datasets\PKG - HistologyHSI-GB")
DATASET_KEY = "histology-hsi-gb"
EXPECTED_ROI_COUNT = 469
EXPECTED_PATIENT_COUNT = 13

CACHED_SPATIAL_MEAN_PCA_KEY = "histology.cached-spatial-mean-pca.v1"
FULL_CUBE_SPATIAL_PCA_KEY = "histology.full-cube-spatial-pca.v1"
RGB_PREVIEW_KEY = "histology.rgb-preview.v1"


class HistologyAdapterError(RuntimeError):
    """Raised when source discovery or representation access violates policy."""


class RepresentationSourceKind(str, Enum):
    """How a governed representation obtains its spectral input."""

    IMMUTABLE_SOURCE_IMAGE = "immutable_source_image"
    IMMUTABLE_DERIVED_CACHE = "immutable_derived_cache"
    READ_ONLY_RAW_CUBE = "read_only_raw_cube"


@dataclass(frozen=True)
class HistologyRepresentation:
    """Immutable identity and access policy for one Histology representation."""

    representation_key: str
    source_kind: RepresentationSourceKind
    required_input_names: Tuple[str, ...]
    spatial_semantics: str
    pca_fit_scope: str
    raw_access_policy: str


HISTOLOGY_REPRESENTATIONS: Mapping[str, HistologyRepresentation] = {
    RGB_PREVIEW_KEY: HistologyRepresentation(
        representation_key=RGB_PREVIEW_KEY,
        source_kind=RepresentationSourceKind.IMMUTABLE_SOURCE_IMAGE,
        required_input_names=("rgb_preview",),
        spatial_semantics="visible RGB preview resized to the declared input resolution",
        pca_fit_scope="not applicable; this representation contains no PCA state",
        raw_access_policy="raw cube access is forbidden for RGB-only inference",
    ),
    CACHED_SPATIAL_MEAN_PCA_KEY: HistologyRepresentation(
        representation_key=CACHED_SPATIAL_MEAN_PCA_KEY,
        source_kind=RepresentationSourceKind.IMMUTABLE_DERIVED_CACHE,
        required_input_names=("rgb_preview", "cached_spatial_mean_pca"),
        spatial_semantics="precomputed spatial-mean spectral cache",
        pca_fit_scope="cache provenance; never relabel as full-cube spatial PCA",
        raw_access_policy="raw cube access not required for cached inference",
    ),
    FULL_CUBE_SPATIAL_PCA_KEY: HistologyRepresentation(
        representation_key=FULL_CUBE_SPATIAL_PCA_KEY,
        source_kind=RepresentationSourceKind.READ_ONLY_RAW_CUBE,
        required_input_names=("rgb_preview", "raw_header", "raw_cube"),
        spatial_semantics="spatial PCA derived from the complete ROI cube",
        pca_fit_scope="fit IDs must be supplied by fold-scoped preprocessing",
        raw_access_policy="read-only ENVI memmap with bounded windows or chunks",
    ),
}

if len(HISTOLOGY_REPRESENTATIONS) != len(
    {item.representation_key for item in HISTOLOGY_REPRESENTATIONS.values()}
):
    raise RuntimeError("Histology Representation_Key values must be unique")
if CACHED_SPATIAL_MEAN_PCA_KEY == FULL_CUBE_SPATIAL_PCA_KEY:
    raise RuntimeError("cached and full-cube PCA must use distinct Representation_Key values")


class HistologyInputHasher:
    """Root-confined, memoizing SHA-256 hasher for large Histology inputs.

    An optional private cache stores only relative references, file stat
    identities, and completed digests. This makes the 622 GB source scan
    resumable without weakening content hashing.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        cache_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.root = _validate_approved_root(root, approved_root=root)
        self.cache_path = Path(cache_path).resolve(strict=False) if cache_path else None
        if self.cache_path is not None and self.cache_path.drive.upper() != "G:":
            raise HistologyAdapterError("Histology hash cache must remain on the G: drive")
        self._lock = Lock()
        self._cache: Dict[Tuple[str, int, int], str] = {}
        self._load_cache()

    def __call__(self, reference: str) -> str:
        path = _resolve_relative_input(self.root, reference)
        stat = path.stat()
        key = (reference, int(stat.st_size), int(stat.st_mtime_ns))
        with self._lock:
            digest = self._cache.get(key)
        if digest is not None:
            return digest

        digest = hash_file(path, chunk_size=8 * 1024 * 1024)
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                return existing
            self._cache[key] = digest
            self._persist_cache()
        return digest

    def prime(self, references: Iterable[str], *, workers: int = 4) -> None:
        """Hash references concurrently and persist each completed digest."""

        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("workers must be a positive integer")
        unique = tuple(sorted(set(references), key=lambda value: value.encode("utf-8")))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, _ in enumerate(executor.map(self, unique), start=1):
                if index == 1 or index % 25 == 0 or index == len(unique):
                    print(f"histology input hashes: {index}/{len(unique)}", flush=True)

    def _load_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if payload.get("root") != str(self.root):
            raise HistologyAdapterError("hash cache root differs from the approved source root")
        for item in payload.get("entries", ()):
            key = (item["reference"], int(item["size"]), int(item["mtime_ns"]))
            digest = item["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise HistologyAdapterError("hash cache contains an invalid SHA-256 digest")
            self._cache[key] = digest

    def _persist_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "root": str(self.root),
            "entries": [
                {
                    "reference": reference,
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "sha256": digest,
                }
                for (reference, size, mtime_ns), digest in sorted(self._cache.items())
            ],
        }
        temporary = self.cache_path.with_name(
            f".{self.cache_path.name}.{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(10):
                try:
                    os.replace(temporary, self.cache_path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.1 * (attempt + 1))
        finally:
            if temporary.exists():
                temporary.unlink()


class HistologyHSIGBAdapter:
    """Concrete patient-grouped adapter for the approved HistologyHSI-GB root."""

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_HISTOLOGY_ROOT,
        *,
        approved_root: Union[str, Path] = DEFAULT_HISTOLOGY_ROOT,
        input_hasher: Optional[HistologyInputHasher] = None,
    ) -> None:
        self.root = _validate_approved_root(root, approved_root=approved_root)
        self.input_hasher = input_hasher or HistologyInputHasher(self.root)

    @property
    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            dataset_key=DATASET_KEY,
            task_kind=TaskKind.BINARY_CLASSIFICATION,
            expected_sample_count=EXPECTED_ROI_COUNT,
            label_schema=LabelSchema(
                schema_key="histology.binary.roi-suffix.v1",
                value_kind=LabelValueKind.BINARY,
                dimension=1,
                target_names=("tumor",),
                class_names=("non_tumor", "tumor"),
            ),
            grouped=True,
        )

    @property
    def representations(self) -> Mapping[str, HistologyRepresentation]:
        """Return the immutable representation registry keyed by Representation_Key."""

        return HISTOLOGY_REPRESENTATIONS

    def representation(self, representation_key: str) -> HistologyRepresentation:
        try:
            return self.representations[representation_key]
        except KeyError as error:
            raise HistologyAdapterError(
                f"unknown Histology Representation_Key: {representation_key}"
            ) from error

    def iter_samples(self) -> Iterable[SampleRecord]:
        return self.iter_patient_grouped_samples()

    def iter_patient_grouped_samples(self) -> Iterable[SampleRecord]:
        patient_directories = tuple(
            sorted(
                (path for path in self.root.iterdir() if path.is_dir() and _is_patient_name(path.name)),
                key=lambda path: _patient_number(path.name),
            )
        )
        if len(patient_directories) != EXPECTED_PATIENT_COUNT:
            raise HistologyAdapterError(
                f"expected {EXPECTED_PATIENT_COUNT} patient directories, observed {len(patient_directories)}"
            )

        seen: set[str] = set()
        for patient_directory in patient_directories:
            _assert_confined(self.root, patient_directory)
            roi_directories = sorted(
                (path for path in patient_directory.iterdir() if path.is_dir() and path.name.startswith("ROI_")),
                key=lambda path: path.name.encode("utf-8"),
            )
            for roi_directory in roi_directories:
                _assert_confined(self.root, roi_directory)
                sample_id = f"{patient_directory.name}/{roi_directory.name}"
                if sample_id in seen:
                    raise HistologyAdapterError(f"duplicate ROI Sample_ID: {sample_id}")
                seen.add(sample_id)
                yield self._build_record(patient_directory.name, roi_directory)

        if len(seen) != EXPECTED_ROI_COUNT:
            raise HistologyAdapterError(
                f"expected {EXPECTED_ROI_COUNT} ROI samples, observed {len(seen)}"
            )

    def open_raw_cube(self, sample_id: str) -> HyperspectralCubeReader:
        """Open one ROI cube as a read-only canonical ``(bands, rows, columns)`` memmap."""

        roi = self._roi_path(sample_id)
        return open_envi_cube(roi / "raw.hdr", raw_path=roi / "raw")

    def materialize_manifest(
        self,
        artifact_root: Union[str, Path],
    ) -> SampleManifest:
        """Hash all required inputs once and atomically publish the private manifest."""

        store = FailureAtomicArtifactStore(Path(artifact_root))
        catalog = DatasetCatalog(store, input_hasher=self.input_hasher)
        return catalog.materialize_manifest(self)

    def _build_record(self, patient_id: str, roi: Path) -> SampleRecord:
        class_name, label_value = _label_from_roi_name(roi.name)
        cache = self.root / "cache_routeA_v2" / f"{patient_id}_{roi.name}.npz"
        required = {
            "raw_cube": roi / "raw",
            "raw_header": roi / "raw.hdr",
            "rgb_preview": roi / "rgb.png",
            "cached_spatial_mean_pca": cache,
        }
        references: Dict[str, str] = {}
        hashes: Dict[str, str] = {}
        for input_name, path in required.items():
            confined = _assert_confined(self.root, path)
            if not confined.is_file():
                raise HistologyAdapterError(
                    f"required Histology input is missing for {patient_id}/{roi.name}: {input_name}"
                )
            reference = confined.relative_to(self.root).as_posix()
            references[input_name] = reference
            hashes[input_name] = self.input_hasher(reference)

        label_identity = {
            "schema_key": self.spec.label_schema.schema_key,
            "sample_id": f"{patient_id}/{roi.name}",
            "value": label_value,
        }
        return SampleRecord(
            sample_id=f"{patient_id}/{roi.name}",
            group_id=patient_id,
            label_reference="roi-directory-class-suffix",
            input_references=references,
            input_content_hashes=hashes,
            label_schema_metadata={
                "schema_key": self.spec.label_schema.schema_key,
                "dimension": 1,
                "label_value": label_value,
                "class_name": class_name,
                "label_content_hash": hash_object(label_identity),
            },
        )

    def _roi_path(self, sample_id: str) -> Path:
        normalized = PurePosixPath(sample_id)
        if normalized.is_absolute() or len(normalized.parts) != 2:
            raise HistologyAdapterError("sample_id must be a patient/ROI relative identifier")
        patient, roi = normalized.parts
        if not _is_patient_name(patient) or not roi.startswith("ROI_"):
            raise HistologyAdapterError(f"invalid Histology Sample_ID: {sample_id}")
        path = _assert_confined(self.root, self.root / patient / roi)
        if not path.is_dir():
            raise HistologyAdapterError(f"Histology Sample_ID does not exist: {sample_id}")
        return path


def _validate_approved_root(
    root: Union[str, Path], *, approved_root: Union[str, Path]
) -> Path:
    candidate = Path(root).resolve(strict=False)
    approved = Path(approved_root).resolve(strict=False)
    if candidate.drive.upper() != "G:":
        raise HistologyAdapterError("Histology source must remain on the approved G: drive")
    if candidate != approved:
        raise HistologyAdapterError(
            f"Histology source differs from approved root: {candidate}"
        )
    if not candidate.is_dir():
        raise HistologyAdapterError(f"Histology source root does not exist: {candidate}")
    return candidate


def _resolve_relative_input(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise HistologyAdapterError("input reference must be a non-empty relative path")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[0].endswith(":"):
        raise HistologyAdapterError("input references must remain relative to the approved root")
    path = _assert_confined(root, root.joinpath(*pure.parts))
    if not path.is_file():
        raise HistologyAdapterError(f"Histology input does not exist: {reference}")
    return path


def _assert_confined(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise HistologyAdapterError(
            f"Histology path escapes the approved G-drive root: {path}"
        ) from error
    if resolved.drive.upper() != "G:":
        raise HistologyAdapterError("Histology path resolves outside the approved G: drive")
    return resolved


def _is_patient_name(name: str) -> bool:
    return name.startswith("P") and name[1:].isdigit() and int(name[1:]) > 0


def _patient_number(name: str) -> int:
    if not _is_patient_name(name):
        raise HistologyAdapterError(f"invalid patient directory: {name}")
    return int(name[1:])


def _label_from_roi_name(name: str) -> Tuple[str, int]:
    if name.endswith("_NT"):
        return "non_tumor", 0
    if name.endswith("_T"):
        return "tumor", 1
    raise HistologyAdapterError(
        f"ROI name must end in _T or _NT to define its governed label: {name}"
    )


__all__ = [
    "CACHED_SPATIAL_MEAN_PCA_KEY",
    "DATASET_KEY",
    "DEFAULT_HISTOLOGY_ROOT",
    "EXPECTED_PATIENT_COUNT",
    "EXPECTED_ROI_COUNT",
    "FULL_CUBE_SPATIAL_PCA_KEY",
    "HISTOLOGY_REPRESENTATIONS",
    "HistologyAdapterError",
    "HistologyHSIGBAdapter",
    "HistologyInputHasher",
    "HistologyRepresentation",
    "RepresentationSourceKind",
]
