"""Property 3: Stable Histology de-identification.

**Validates: Requirements 2.24, 2.25, 17.41-17.43**
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from hypothesis import given, settings, strategies as st

from benchmark.catalog import DatasetCatalog, HistologyPublicExportPolicy
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import SampleManifest, SampleRecord, canonical_sample_order


_SCHEME = "hmac-sha256-v1"
_DATASET_KEY = "histology-hsi-gb"
_POLICY = HistologyPublicExportPolicy(
    deidentification_key_id="histology-public-key-v1",
    approved_input_names=("rgb_preview",),
    approved_label_metadata_fields=("schema_key", "dimension"),
)
_PATIENT_SUFFIXES = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=8,
    max_size=16,
)


class HistologyCase(NamedTuple):
    manifest: SampleManifest
    key: bytes
    patient_ids: tuple[str, ...]
    restricted_tokens: tuple[str, ...]


@st.composite
def histology_manifests(draw: st.DrawFn) -> HistologyCase:
    """Generate grouped manifests carrying unmistakable private data."""

    suffixes = draw(st.lists(_PATIENT_SUFFIXES, min_size=1, max_size=5, unique=True))
    roi_counts = draw(
        st.lists(
            st.integers(min_value=1, max_value=3),
            min_size=len(suffixes),
            max_size=len(suffixes),
        )
    )
    key_tail = draw(st.binary(min_size=0, max_size=32))
    key = b"property-3-private-key-material-" + key_tail
    records: list[SampleRecord] = []
    patient_ids: list[str] = []
    restricted_tokens = {"raw_cube", "private_diagnosis", "patient_identifier"}

    for suffix, roi_count in zip(suffixes, roi_counts):
        patient_id = f"PATIENT_SECRET_{suffix}"
        patient_ids.append(patient_id)
        for roi_index in range(roi_count):
            roi_id = f"ROI_{roi_index}"
            raw_reference = f"private/{patient_id}/{roi_id}/RAW_CUBE_RESTRICTED.bin"
            label_reference = f"private/{patient_id}/{roi_id}/LABEL_RESTRICTED.json"
            diagnosis = f"RESTRICTED_LABEL_{suffix}_{roi_index}"
            raw_hash = hashlib.sha256(raw_reference.encode("utf-8")).hexdigest()
            rgb_hash = hashlib.sha256(
                f"approved-preview:{patient_id}:{roi_id}".encode("utf-8")
            ).hexdigest()
            restricted_tokens.update(
                (patient_id, raw_reference, label_reference, diagnosis, raw_hash)
            )
            records.append(
                SampleRecord(
                    sample_id=f"{patient_id}/{roi_id}",
                    group_id=patient_id,
                    label_reference=label_reference,
                    input_references={
                        "raw_cube": raw_reference,
                        "rgb_preview": f"derived/{patient_id}/{roi_id}/preview.png",
                    },
                    input_content_hashes={
                        "raw_cube": raw_hash,
                        "rgb_preview": rgb_hash,
                    },
                    label_schema_metadata={
                        "schema_key": "histology.binary.v1",
                        "dimension": 1,
                        "private_diagnosis": diagnosis,
                        "patient_identifier": patient_id,
                    },
                )
            )

    # Enumeration order is deliberately generated independently of canonical order.
    permutation = draw(st.permutations(tuple(range(len(records)))))
    records = [records[index] for index in permutation]
    sample_ids = tuple(record.sample_id for record in records)
    dataset_hash = hashlib.sha256(
        json.dumps(
            sorted(
                (record.sample_id, record.group_id, record.label_reference)
                for record in records
            ),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = SampleManifest(
        schema_version="1.0",
        dataset_key=_DATASET_KEY,
        records=tuple(records),
        canonical_order=canonical_sample_order(sample_ids),
        dataset_hash=dataset_hash,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    return HistologyCase(
        manifest=manifest,
        key=key,
        patient_ids=tuple(patient_ids),
        restricted_tokens=tuple(sorted(restricted_tokens)),
    )


def _catalog(root: Path) -> DatasetCatalog:
    return DatasetCatalog(
        FailureAtomicArtifactStore(root),
        input_hasher=lambda reference: hashlib.sha256(reference.encode("utf-8")).hexdigest(),
    )


def _expected_pseudonym(key: bytes, kind: str, identifier: str) -> str:
    message = "\x00".join((_SCHEME, kind, _DATASET_KEY, identifier)).encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).hexdigest()
    prefix = "group" if kind == "patient" else "sample"
    return f"{prefix}-{digest}"


@settings(max_examples=100, deadline=None)
@given(case=histology_manifests())
def test_repeated_exports_preserve_mapping_and_exclude_restricted_histology_data(
    case: HistologyCase,
) -> None:
    """Stable pseudonyms expose only explicitly approved public fields."""

    with TemporaryDirectory(prefix="property3-first-") as first_directory, TemporaryDirectory(
        prefix="property3-second-"
    ) as second_directory:
        first_root = Path(first_directory)
        second_root = Path(second_directory)
        first = _catalog(first_root).export_histology_public_manifest(
            case.manifest,
            deidentification_key=case.key,
            policy=_POLICY,
            created_utc=datetime(2027, 2, 1, tzinfo=timezone.utc),
        )
        second = _catalog(second_root).export_histology_public_manifest(
            case.manifest,
            deidentification_key=case.key,
            policy=_POLICY,
            created_utc=datetime(2030, 3, 1, tzinfo=timezone.utc),
        )

        first_mapping = {record.sample_id: record.group_id for record in first.records}
        second_mapping = {record.sample_id: record.group_id for record in second.records}
        assert first_mapping == second_mapping

        expected_mapping = {
            _expected_pseudonym(case.key, "sample", source.sample_id): _expected_pseudonym(
                case.key, "patient", source.group_id
            )
            for source in case.manifest.records
            if source.group_id is not None
        }
        assert first_mapping == expected_mapping
        assert sorted(Counter(first_mapping.values()).values()) == sorted(
            Counter(source.group_id for source in case.manifest.records).values()
        )

        for exported in (first, second):
            assert all(set(record.input_content_hashes) == {"rgb_preview"} for record in exported.records)
            assert all(
                set(record.approved_label_metadata) == {"schema_key", "dimension"}
                for record in exported.records
            )

        payloads = tuple(path.read_bytes() for root in (first_root, second_root) for path in root.rglob("*.json"))
        assert len(payloads) == 2
        for payload in payloads:
            for token in case.restricted_tokens:
                assert token.encode("utf-8") not in payload
