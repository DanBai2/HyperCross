"""Dataset Catalog unit coverage for tasks 2.2 and 2.4.

The tests exercise the validation records and failure boundary required by
Requirements 2.10-2.23 and 16.1-16.6.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pytest

from benchmark.catalog import DatasetCatalog, LocalInputHasher, ManifestMaterializationError
from benchmark.recovery.artifact_store import FailureAtomicArtifactStore
from benchmark.schema import DatasetSpec, LabelSchema, LabelValueKind, SampleManifest, SampleRecord, TaskKind


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def regression_spec(count: int) -> DatasetSpec:
    return DatasetSpec(
        dataset_key="synthetic-leaf",
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        expected_sample_count=count,
        label_schema=LabelSchema(
            schema_key="leaf.targets.v1",
            value_kind=LabelValueKind.CONTINUOUS,
            dimension=3,
            target_names=("GrainWeight", "Gsw", "PhiPS2"),
        ),
        grouped=True,
    )


def record(
    sample_id: str,
    input_reference: str,
    content_hash: str,
    *,
    group_id: Optional[str] = "variety-a",
    dimension: object = 3,
    label_reference: Optional[str] = None,
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        group_id=group_id,
        label_reference=label_reference or f"labels/{sample_id}.json",
        input_references={"cube": input_reference},
        input_content_hashes={"cube": content_hash},
        label_schema_metadata={"schema_key": "leaf.targets.v1", "dimension": dimension},
    )


@dataclass
class StreamingAdapter:
    spec: DatasetSpec
    records: tuple[SampleRecord, ...]
    iterations: int = 0

    def iter_samples(self) -> Iterable[SampleRecord]:
        self.iterations += 1
        yield from self.records


def make_catalog(tmp_path: Path) -> DatasetCatalog:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    return DatasetCatalog(store, input_hasher=LocalInputHasher(input_root))


def test_materializes_stream_once_in_canonical_order_and_publishes(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    input_root = tmp_path / "inputs"
    (input_root / "z.bin").write_bytes(b"z")
    (input_root / "a.bin").write_bytes(b"a")
    adapter = StreamingAdapter(
        regression_spec(2),
        (
            record("z-sample", "z.bin", sha256(b"z")),
            record("a-sample", "a.bin", sha256(b"a")),
        ),
    )

    manifest = catalog.materialize_manifest(
        adapter,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert adapter.iterations == 1
    assert manifest.canonical_order == ("a-sample", "z-sample")
    assert tuple(item.sample_id for item in manifest.records) == manifest.canonical_order
    assert {item.sample_id for item in manifest.records} == {"a-sample", "z-sample"}
    assert len({item.sample_id for item in manifest.records}) == len(manifest.records)
    assert all(item.group_id for item in manifest.records)
    path = catalog.artifact_store.root / "manifests" / "synthetic-leaf" / f"{manifest.dataset_hash}.json"
    assert SampleManifest.model_validate_json(path.read_bytes()) == manifest
    assert catalog.validate_manifest(manifest, adapter.spec).passes


def test_validation_aggregates_duplicate_group_dimension_and_input_failures(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    input_root = tmp_path / "inputs"
    (input_root / "readable.bin").write_bytes(b"content")
    records = (
        record("folder\\sample", "readable.bin", sha256(b"content")),
        record(
            "folder/sample",
            "missing.bin",
            sha256(b"missing"),
            group_id=None,
            dimension=2,
        ),
    )

    with pytest.raises(ManifestMaterializationError) as caught:
        catalog.materialize_manifest(StreamingAdapter(regression_spec(2), records))

    error = caught.value
    assert error.protocol_creation_allowed is False
    by_code = {item.error_code: item for item in error.error_records}
    assert {
        "MANIFEST_DUPLICATE_SAMPLE_ID",
        "MANIFEST_GROUP_ID_MISSING",
        "MANIFEST_LABEL_DIMENSION_INVALID",
        "MANIFEST_REQUIRED_INPUT_UNREADABLE",
    } <= set(by_code)
    assert by_code["MANIFEST_DUPLICATE_SAMPLE_ID"].failing_field == "sample_id"
    assert by_code["MANIFEST_DUPLICATE_SAMPLE_ID"].unexpected_ids == ("folder/sample",)
    assert by_code["MANIFEST_GROUP_ID_MISSING"].failing_field == "group_id"
    assert by_code["MANIFEST_GROUP_ID_MISSING"].missing_ids == ("folder/sample",)
    label_error = by_code["MANIFEST_LABEL_DIMENSION_INVALID"]
    assert label_error.failing_field == "label_dimension"
    assert label_error.expected_value == 3
    assert label_error.observed_value == {"sample_dimensions": {"folder/sample": 2}}
    unreadable_error = by_code["MANIFEST_REQUIRED_INPUT_UNREADABLE"]
    assert unreadable_error.failing_field == "input_reference"
    unreadable = unreadable_error.observed_value["inputs"]
    assert unreadable[0]["sample_id"] == "folder/sample"
    assert unreadable[0]["input_reference"] == "missing.bin"
    assert unreadable[0]["read_error_code"] == "FileNotFoundError"
    assert not (catalog.artifact_store.root / "manifests").exists()
    assert catalog.error_journal.path.read_text(encoding="utf-8").count("\n") == 4


def test_exact_count_and_hash_mismatch_block_publication(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    input_root = tmp_path / "inputs"
    (input_root / "sample.bin").write_bytes(b"actual")
    baseline = catalog.materialize_manifest(
        StreamingAdapter(
            regression_spec(1),
            (record("sample-1", "sample.bin", sha256(b"actual")),),
        )
    )
    baseline_path = (
        catalog.artifact_store.root
        / "manifests"
        / "synthetic-leaf"
        / f"{baseline.dataset_hash}.json"
    )
    baseline_bytes = baseline_path.read_bytes()
    adapter = StreamingAdapter(
        regression_spec(2),
        (record("sample-1", "sample.bin", sha256(b"declared")),),
    )

    with pytest.raises(ManifestMaterializationError) as caught:
        catalog.materialize_manifest(adapter)

    by_code = {item.error_code: item for item in caught.value.error_records}
    assert set(by_code) == {"MANIFEST_SAMPLE_COUNT_MISMATCH", "MANIFEST_INPUT_HASH_MISMATCH"}
    count_error = by_code["MANIFEST_SAMPLE_COUNT_MISMATCH"]
    assert count_error.failing_field == "sample_count"
    assert count_error.expected_value == 2
    assert count_error.observed_value == 1
    assert all(
        item.invariant_checks["protocol_creation_blocked"]
        and item.invariant_checks["previous_manifests_unchanged"]
        and item.invariant_checks["new_manifest_unpublished"]
        for item in caught.value.error_records
    )
    assert baseline_path.read_bytes() == baseline_bytes
    assert list(baseline_path.parent.glob("*.json")) == [baseline_path]


def test_content_and_label_changes_create_new_versions_without_rewriting_old(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    input_file = tmp_path / "inputs" / "sample.bin"
    input_file.write_bytes(b"version-one")
    first_record = record("sample-1", "sample.bin", sha256(b"version-one"))
    first = catalog.materialize_manifest(StreamingAdapter(regression_spec(1), (first_record,)))
    first_path = catalog.artifact_store.root / "manifests" / "synthetic-leaf" / f"{first.dataset_hash}.json"
    first_bytes = first_path.read_bytes()

    # Identical content reuses the immutable version, including its first timestamp.
    repeated = catalog.materialize_manifest(
        StreamingAdapter(regression_spec(1), (first_record,)),
        created_utc=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    assert repeated == first
    assert first_path.read_bytes() == first_bytes

    input_file.write_bytes(b"version-two")
    changed_content = catalog.materialize_manifest(
        StreamingAdapter(
            regression_spec(1),
            (record("sample-1", "sample.bin", sha256(b"version-two")),),
        )
    )
    changed_content_path = (
        catalog.artifact_store.root
        / "manifests"
        / "synthetic-leaf"
        / f"{changed_content.dataset_hash}.json"
    )
    changed_content_bytes = changed_content_path.read_bytes()
    changed_label = catalog.materialize_manifest(
        StreamingAdapter(
            regression_spec(1),
            (
                record(
                    "sample-1",
                    "sample.bin",
                    sha256(b"version-two"),
                    label_reference="labels/sample-1-v2.json",
                ),
            ),
        )
    )

    assert len({first.dataset_hash, changed_content.dataset_hash, changed_label.dataset_hash}) == 3
    assert first_path.read_bytes() == first_bytes
    assert changed_content_path.read_bytes() == changed_content_bytes
    changed_label_path = (
        catalog.artifact_store.root
        / "manifests"
        / "synthetic-leaf"
        / f"{changed_label.dataset_hash}.json"
    )
    assert SampleManifest.model_validate_json(changed_label_path.read_bytes()) == changed_label
    version_paths = list((catalog.artifact_store.root / "manifests" / "synthetic-leaf").glob("*.json"))
    assert set(version_paths) == {first_path, changed_content_path, changed_label_path}


def test_loaded_manifest_hash_validation_is_machine_readable(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    input_file = tmp_path / "inputs" / "sample.bin"
    input_file.write_bytes(b"sample")
    spec = regression_spec(1)
    manifest = catalog.materialize_manifest(
        StreamingAdapter(spec, (record("sample-1", "sample.bin", sha256(b"sample")),))
    )
    tampered = manifest.model_copy(update={"dataset_hash": "f" * 64})

    report = catalog.validate_manifest(tampered, spec)

    assert not report.passes
    assert report.errors[0].code == "MANIFEST_DATASET_HASH_MISMATCH"
    assert report.errors[0].field_path == ("dataset_hash",)
