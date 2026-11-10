"""Stable, publication-safe Histology manifest projection.

The exporter uses keyed HMAC pseudonyms. The key is supplied at export time and
is never serialized, logged, or stored in the public artifact. Public records
contain neither source paths nor source label references; copied fields are
strictly allowlisted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from urllib.parse import quote

from pydantic import field_validator, model_validator

from benchmark.recovery.artifact_store import ArtifactStoreError, FailureAtomicArtifactStore
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.canonical import hash_object, normalize_id
from benchmark.schema.datasets import (
    FrozenDatasetModel,
    PublicSampleManifest,
    PublicSampleRecord,
    SampleManifest,
    canonical_sample_order,
)
from benchmark.schema.base import Identifier


_DEIDENTIFICATION_SCHEME = "hmac-sha256-v1"
_MINIMUM_KEY_BYTES = 32
_RAW_CUBE_NAMES = {
    "cube",
    "full_cube",
    "hsi",
    "hsi_cube",
    "hyperspectral",
    "hyperspectral_cube",
    "raw_cube",
    "raw_hsi",
}


class HistologyPublicExportPolicy(FrozenDatasetModel):
    """Explicit allowlist for fields copied into a public Histology manifest."""

    deidentification_key_id: Identifier
    approved_input_names: Tuple[Identifier, ...] = ()
    approved_label_metadata_fields: Tuple[Identifier, ...] = ()

    @field_validator("approved_input_names", "approved_label_metadata_fields")
    @classmethod
    def validate_unique_fields(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("approved field names must be unique")
        return tuple(sorted(value, key=lambda item: item.encode("utf-8")))

    @model_validator(mode="after")
    def prohibit_raw_cube_approval(self) -> "HistologyPublicExportPolicy":
        forbidden = tuple(name for name in self.approved_input_names if _is_raw_cube_name(name))
        if forbidden:
            raise ValueError(
                "raw hyperspectral cube inputs cannot be approved for public export: "
                + ", ".join(forbidden)
            )
        return self


class HistologyPublicExporter:
    """Build and atomically publish deterministic de-identified manifests."""

    def __init__(
        self,
        artifact_store: FailureAtomicArtifactStore,
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.error_journal = error_journal or artifact_store.error_journal

    def export(
        self,
        manifest: SampleManifest,
        *,
        deidentification_key: Union[bytes, bytearray],
        policy: HistologyPublicExportPolicy,
        schema_version: str = "1.0",
        created_utc: Optional[datetime] = None,
    ) -> PublicSampleManifest:
        """Return one immutable public projection, reusing identical exports.

        Source Sample_ID values are also pseudonymized because Histology ROI IDs
        may embed a patient identifier. Only hashes for approved, non-raw inputs
        and explicitly approved scalar label metadata are retained.
        """

        key = _validate_key(deidentification_key)
        records = self._project_records(manifest, key, policy)
        order = canonical_sample_order(tuple(record.sample_id for record in records))
        ordered_records = tuple(
            sorted(records, key=lambda record: normalize_id(record.sample_id).encode("utf-8"))
        )
        policy_hash = hash_object(policy)
        public_hash = _public_manifest_hash(
            schema_version=schema_version,
            dataset_key=manifest.dataset_key,
            source_dataset_hash=manifest.dataset_hash,
            records=ordered_records,
            canonical_order=order,
            key_id=policy.deidentification_key_id,
            policy_hash=policy_hash,
        )
        relative_path = _public_manifest_relative_path(manifest.dataset_key, public_hash)

        existing = self._load_existing(relative_path, public_hash)
        if existing is not None:
            return existing

        public_manifest = PublicSampleManifest(
            schema_version=schema_version,
            dataset_key=manifest.dataset_key,
            source_dataset_hash=manifest.dataset_hash,
            records=ordered_records,
            canonical_order=order,
            deidentification_scheme=_DEIDENTIFICATION_SCHEME,
            deidentification_key_id=policy.deidentification_key_id,
            export_policy_hash=policy_hash,
            public_manifest_hash=public_hash,
            created_utc=created_utc or datetime.now(timezone.utc),
        )
        payload = _encode_public_manifest(public_manifest)
        try:
            self.artifact_store.publish_bytes(
                artifact_id=f"histology-public-manifest:{manifest.dataset_key}:{public_hash}",
                relative_path=relative_path,
                content=payload,
                media_type="application/json",
            )
        except ArtifactStoreError as exc:
            if exc.error_record.error_code == "ARTIFACT_TARGET_ALREADY_COMPLETE":
                existing = self._load_existing(relative_path, public_hash)
                if existing is not None:
                    return existing
            raise
        return public_manifest

    @staticmethod
    def _project_records(
        manifest: SampleManifest,
        key: bytes,
        policy: HistologyPublicExportPolicy,
    ) -> Tuple[PublicSampleRecord, ...]:
        projected = []
        for source in manifest.records:
            if source.group_id is None:
                raise ValueError("Histology public export requires one patient Group_ID per sample")

            approved_hashes = {
                name: source.input_content_hashes[name]
                for name in policy.approved_input_names
                if name in source.input_content_hashes
            }
            approved_metadata = {
                name: source.label_schema_metadata[name]
                for name in policy.approved_label_metadata_fields
                if name in source.label_schema_metadata
            }
            _reject_identifier_leak(
                source.group_id,
                tuple(approved_hashes),
                approved_metadata,
            )
            projected.append(
                PublicSampleRecord(
                    sample_id=_pseudonym(key, "sample", manifest.dataset_key, source.sample_id),
                    group_id=_pseudonym(key, "patient", manifest.dataset_key, source.group_id),
                    input_content_hashes=approved_hashes,
                    approved_label_metadata=approved_metadata,
                )
            )
        return tuple(projected)

    def _load_existing(
        self,
        relative_path: str,
        expected_hash: str,
    ) -> Optional[PublicSampleManifest]:
        path = self.artifact_store.root.joinpath(*relative_path.split("/"))
        if not path.exists():
            return None
        loaded = PublicSampleManifest.model_validate_json(path.read_bytes())
        observed_hash = _public_manifest_hash(
            schema_version=loaded.schema_version,
            dataset_key=loaded.dataset_key,
            source_dataset_hash=loaded.source_dataset_hash,
            records=loaded.records,
            canonical_order=loaded.canonical_order,
            key_id=loaded.deidentification_key_id,
            policy_hash=loaded.export_policy_hash,
        )
        if loaded.public_manifest_hash != expected_hash or observed_hash != expected_hash:
            raise ValueError("existing public manifest does not match its content identity")
        return loaded


def _validate_key(value: Union[bytes, bytearray]) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError("deidentification_key must be bytes")
    key = bytes(value)
    if len(key) < _MINIMUM_KEY_BYTES:
        raise ValueError("deidentification_key must contain at least 32 bytes")
    return key


def _pseudonym(key: bytes, kind: str, dataset_key: str, identifier: str) -> str:
    message = "\x00".join(
        (_DEIDENTIFICATION_SCHEME, kind, normalize_id(dataset_key), normalize_id(identifier))
    ).encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).hexdigest()
    prefix = "group" if kind == "patient" else "sample"
    return f"{prefix}-{digest}"


def _is_raw_cube_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return normalized in _RAW_CUBE_NAMES or (
        "raw" in normalized.split("_")
        and any(token in normalized.split("_") for token in ("cube", "hsi", "hyperspectral"))
    )


def _reject_identifier_leak(
    patient_identifier: str,
    approved_input_names: Iterable[str],
    approved_metadata: dict[str, object],
) -> None:
    source = normalize_id(patient_identifier)
    copied_strings = list(approved_input_names) + list(approved_metadata)
    copied_strings.extend(value for value in approved_metadata.values() if isinstance(value, str))
    for copied in copied_strings:
        normalized = normalize_id(copied)
        if normalized == source or (len(source) >= 4 and source in normalized):
            raise ValueError(
                "approved public fields must not contain the source patient identifier"
            )


def _public_manifest_hash(
    *,
    schema_version: str,
    dataset_key: str,
    source_dataset_hash: str,
    records: Tuple[PublicSampleRecord, ...],
    canonical_order: Tuple[str, ...],
    key_id: str,
    policy_hash: str,
) -> str:
    return hash_object(
        {
            "schema_version": schema_version,
            "dataset_key": dataset_key,
            "source_dataset_hash": source_dataset_hash,
            "records": records,
            "canonical_order": canonical_order,
            "deidentification_scheme": _DEIDENTIFICATION_SCHEME,
            "deidentification_key_id": key_id,
            "export_policy_hash": policy_hash,
        }
    )


def _encode_public_manifest(manifest: PublicSampleManifest) -> bytes:
    return json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _public_manifest_relative_path(dataset_key: str, public_hash: str) -> str:
    encoded_key = quote(dataset_key, safe="-._")
    return f"publication/manifests/{encoded_key}/{public_hash}.json"
