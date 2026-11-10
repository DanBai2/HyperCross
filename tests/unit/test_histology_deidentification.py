"""Targeted unit coverage for task 2.3 Histology public export."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.catalog import DatasetCatalog, HistologyPublicExportPolicy
from benchmark.recovery.artifact_store import FailureAtomicArtifactStore
from benchmark.schema import (
    PublicSampleManifest,
    SampleManifest,
    SampleRecord,
    canonical_sample_order,
)


KEY = b"stable-test-only-deidentification-key-32-bytes"
HASH_RAW = "a" * 64
HASH_RGB = "b" * 64


def source_record(patient: str, roi: str) -> SampleRecord:
    return SampleRecord(
        sample_id=f"{patient}/{roi}",
        group_id=patient,
        label_reference=f"private/labels/{patient}/{roi}.json",
        input_references={
            "raw_cube": f"private/{patient}/{roi}/raw.hsi",
            "rgb_preview": f"derived/{patient}/{roi}/rgb.png",
        },
        input_content_hashes={
            "raw_cube": HASH_RAW,
            "rgb_preview": HASH_RGB,
        },
        label_schema_metadata={
            "schema_key": "histology.binary.v1",
            "dimension": 1,
            "diagnosis": "malignant",
            "patient_identifier": patient,
        },
    )


def source_manifest() -> SampleManifest:
    records = (
        source_record("patient-B", "roi-1"),
        source_record("patient-A", "roi-2"),
        source_record("patient-A", "roi-1"),
    )
    return SampleManifest(
        schema_version="1.0",
        dataset_key="histology-hsi-gb",
        records=records,
        canonical_order=canonical_sample_order(tuple(record.sample_id for record in records)),
        dataset_hash="c" * 64,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


def catalog(tmp_path: Path) -> DatasetCatalog:
    return DatasetCatalog(
        FailureAtomicArtifactStore(tmp_path / "artifacts"),
        input_hasher=lambda reference: HASH_RAW,
    )


def public_policy() -> HistologyPublicExportPolicy:
    return HistologyPublicExportPolicy(
        deidentification_key_id="histology-public-key-v1",
        approved_input_names=("rgb_preview",),
        approved_label_metadata_fields=("schema_key", "dimension"),
    )


def test_export_is_stable_and_maps_each_patient_to_one_pseudonymous_group(tmp_path: Path) -> None:
    dataset_catalog = catalog(tmp_path)
    source = source_manifest()

    first = dataset_catalog.export_histology_public_manifest(
        source,
        deidentification_key=KEY,
        policy=public_policy(),
        created_utc=datetime(2027, 2, 1, tzinfo=timezone.utc),
    )
    second = dataset_catalog.export_histology_public_manifest(
        source,
        deidentification_key=KEY,
        policy=public_policy(),
        created_utc=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )

    assert second == first
    assert len({record.sample_id for record in first.records}) == 3
    # Two ROIs from patient-A share one pseudonymous group; patient-B receives
    # a different group, without exposing which opaque group belongs to either.
    group_sizes = sorted(
        sum(candidate.group_id == group_id for candidate in first.records)
        for group_id in {record.group_id for record in first.records}
    )
    assert group_sizes == [1, 2]
    assert all(record.group_id.startswith("group-") for record in first.records)
    assert all(record.sample_id.startswith("sample-") for record in first.records)

    path = (
        dataset_catalog.artifact_store.root
        / "publication"
        / "manifests"
        / "histology-hsi-gb"
        / f"{first.public_manifest_hash}.json"
    )
    original_bytes = path.read_bytes()
    assert PublicSampleManifest.model_validate_json(original_bytes) == first
    assert path.read_bytes() == original_bytes


def test_export_omits_patient_identifiers_raw_cube_paths_and_unapproved_labels(tmp_path: Path) -> None:
    source = source_manifest()
    exported = catalog(tmp_path).export_histology_public_manifest(
        source,
        deidentification_key=KEY,
        policy=public_policy(),
    )
    payload = exported.model_dump_json()

    for patient_identifier in ("patient-A", "patient-B"):
        assert patient_identifier not in payload
    assert "private/" not in payload
    assert "label_reference" not in payload
    assert "raw_cube" not in payload
    assert HASH_RAW not in payload
    assert "diagnosis" not in payload
    assert "malignant" not in payload
    assert "patient_identifier" not in payload

    assert all(record.input_content_hashes == {"rgb_preview": HASH_RGB} for record in exported.records)
    assert all(
        record.approved_label_metadata
        == {"schema_key": "histology.binary.v1", "dimension": 1}
        for record in exported.records
    )


def test_export_policy_rejects_raw_cube_approval_and_identifier_leaks(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="raw hyperspectral cube"):
        HistologyPublicExportPolicy(
            deidentification_key_id="histology-public-key-v1",
            approved_input_names=("raw_cube",),
        )

    leaking_policy = HistologyPublicExportPolicy(
        deidentification_key_id="histology-public-key-v1",
        approved_label_metadata_fields=("patient_identifier",),
    )
    with pytest.raises(ValueError, match="source patient identifier"):
        catalog(tmp_path).export_histology_public_manifest(
            source_manifest(),
            deidentification_key=KEY,
            policy=leaking_policy,
        )


def test_export_requires_a_secret_key_and_grouped_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        catalog(tmp_path).export_histology_public_manifest(
            source_manifest(),
            deidentification_key=b"short",
            policy=public_policy(),
        )

    source = source_manifest()
    ungrouped = source.model_copy(
        update={
            "records": (
                source.records[0].model_copy(update={"group_id": None}),
                *source.records[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="patient Group_ID"):
        catalog(tmp_path).export_histology_public_manifest(
            ungrouped,
            deidentification_key=KEY,
            policy=public_policy(),
        )
