"""Streaming validation and immutable publication of sample manifests."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterable, Optional, Protocol, Tuple, Union
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from benchmark.catalog.adapters import DatasetAdapter
from benchmark.recovery.artifact_store import ArtifactStoreError, FailureAtomicArtifactStore
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import ValidationIssue, ValidationReport, ValidationSeverity
from benchmark.schema.canonical import hash_file, hash_object, normalize_id
from benchmark.schema.datasets import DatasetSpec, SampleManifest, SampleRecord, canonical_sample_order
from benchmark.schema.records import ErrorRecord


InputHasher = Callable[[str], str]


class ManifestMaterializationError(RuntimeError):
    """Reject manifest materialization without exposing an invalid manifest."""

    def __init__(self, error_records: Iterable[ErrorRecord]) -> None:
        self.error_records = tuple(error_records)
        if not self.error_records:
            raise ValueError("at least one error record is required")
        self.protocol_creation_allowed = False
        codes = ", ".join(record.error_code for record in self.error_records)
        super().__init__(f"sample manifest materialization failed: {codes}")


class ManifestPublicationError(RuntimeError):
    """Reject publication when an existing manifest cannot be safely reused."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        self.protocol_creation_allowed = False
        super().__init__(f"sample manifest publication failed: {error_record.error_code}")


class SupportsInputHashing(Protocol):
    """Interface for adapter-specific readable-input verification."""

    def __call__(self, reference: str) -> str:
        """Read a required input and return its SHA-256 digest."""


class LocalInputHasher:
    """Stream hashes from local paths or ``file:`` URIs.

    Relative references are resolved against the configured root. Dataset
    adapters that use another storage backend can inject their own hasher into
    :class:`DatasetCatalog`.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root).resolve()

    def __call__(self, reference: str) -> str:
        return hash_file(self.resolve(reference))

    def resolve(self, reference: str) -> Path:
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("input reference must be a non-empty string")

        # A Windows drive path must not be mistaken for a URI scheme.
        windows_path = PureWindowsPath(reference)
        parsed = urlparse(reference)
        if parsed.scheme and not windows_path.drive:
            if parsed.scheme.lower() != "file":
                raise ValueError(f"unsupported input URI scheme: {parsed.scheme}")
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError("remote file URI authorities are not supported")
            path = Path(unquote(parsed.path.lstrip("/") if windows_path.drive else parsed.path))
        else:
            path = Path(reference)

        if not path.is_absolute():
            path = self.root / path
        return path.resolve(strict=False)


class DatasetCatalog:
    """Materialize validated manifests and publish content-addressed versions."""

    def __init__(
        self,
        artifact_store: FailureAtomicArtifactStore,
        *,
        input_hasher: SupportsInputHashing,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        if not callable(input_hasher):
            raise TypeError("input_hasher must be callable")
        self.artifact_store = artifact_store
        self.input_hasher = input_hasher
        self.error_journal = error_journal or artifact_store.error_journal

    def export_histology_public_manifest(
        self,
        manifest: SampleManifest,
        *,
        deidentification_key: Union[bytes, bytearray],
        policy: "HistologyPublicExportPolicy",
        schema_version: str = "1.0",
        created_utc: Optional[datetime] = None,
    ) -> "PublicSampleManifest":
        """Create a stable, allowlisted Histology publication projection."""

        from benchmark.catalog.deidentification import HistologyPublicExporter

        return HistologyPublicExporter(
            self.artifact_store,
            error_journal=self.error_journal,
        ).export(
            manifest,
            deidentification_key=deidentification_key,
            policy=policy,
            schema_version=schema_version,
            created_utc=created_utc,
        )

    def materialize_manifest(
        self,
        adapter: DatasetAdapter,
        *,
        schema_version: str = "1.0",
        created_utc: Optional[datetime] = None,
    ) -> SampleManifest:
        """Consume one sample stream, validate it, and publish one version.

        Validation completes before any manifest path is created. A repeated
        materialization of identical content reuses the already published
        version; changed sample, input-hash, label-reference, or label-metadata
        content receives a new Dataset_Hash and therefore a new path.
        """

        spec = adapter.spec
        records, stream_errors = self._consume_stream(adapter.iter_samples(), spec)
        errors = stream_errors + self._validate_records(spec, records)
        if errors:
            self._reject(errors)

        ordered_records = tuple(
            sorted(records, key=lambda record: normalize_id(record.sample_id).encode("utf-8"))
        )
        order = canonical_sample_order(tuple(record.sample_id for record in ordered_records))
        dataset_hash = self._dataset_hash(spec, schema_version, ordered_records, order)
        relative_path = self._manifest_relative_path(spec.dataset_key, dataset_hash)

        existing = self._load_existing(relative_path, spec, schema_version, dataset_hash)
        if existing is not None:
            return existing

        timestamp = created_utc or datetime.now(timezone.utc)
        manifest = SampleManifest(
            schema_version=schema_version,
            dataset_key=spec.dataset_key,
            records=ordered_records,
            canonical_order=order,
            dataset_hash=dataset_hash,
            created_utc=timestamp,
        )
        payload = self._encode_manifest(manifest)
        try:
            self.artifact_store.publish_bytes(
                artifact_id=f"sample-manifest:{spec.dataset_key}:{dataset_hash}",
                relative_path=relative_path,
                content=payload,
                media_type="application/json",
            )
        except ArtifactStoreError as exc:
            # A concurrent publisher may have completed the identical version
            # after our precheck. Reuse only after full identity validation.
            if exc.error_record.error_code == "ARTIFACT_TARGET_ALREADY_COMPLETE":
                existing = self._load_existing(
                    relative_path, spec, schema_version, dataset_hash
                )
                if existing is not None:
                    return existing
            raise
        return manifest

    def validate_manifest(
        self,
        manifest: SampleManifest,
        spec: DatasetSpec,
        *,
        verify_inputs: bool = True,
    ) -> ValidationReport:
        """Return machine-readable validation findings for a loaded manifest."""

        errors = self._validate_records(
            spec,
            tuple(manifest.records),
            verify_inputs=verify_inputs,
        )
        if manifest.dataset_key != spec.dataset_key:
            errors.append(
                self._error(
                    spec,
                    code="MANIFEST_DATASET_KEY_MISMATCH",
                    field="dataset_key",
                    expected=spec.dataset_key,
                    observed=manifest.dataset_key,
                )
            )
        expected_hash = self._dataset_hash(
            spec,
            manifest.schema_version,
            tuple(sorted(manifest.records, key=lambda r: normalize_id(r.sample_id).encode("utf-8"))),
            manifest.canonical_order,
        )
        if manifest.dataset_hash != expected_hash:
            errors.append(
                self._error(
                    spec,
                    code="MANIFEST_DATASET_HASH_MISMATCH",
                    field="dataset_hash",
                    expected=expected_hash,
                    observed=manifest.dataset_hash,
                )
            )
        return ValidationReport(
            issues=tuple(
                ValidationIssue(
                    code=error.error_code,
                    severity=ValidationSeverity.ERROR,
                    field_path=(error.failing_field,),
                    message=error.error_code,
                    expected_value=error.expected_value,
                    observed_value=error.observed_value,
                )
                for error in errors
            )
        )

    @staticmethod
    def _dataset_hash(
        spec: DatasetSpec,
        schema_version: str,
        records: Tuple[SampleRecord, ...],
        canonical_order: Tuple[str, ...],
    ) -> str:
        identity = {
            "schema_version": schema_version,
            "dataset_spec": spec,
            "records": records,
            "canonical_order": canonical_order,
        }
        return hash_object(identity)

    def _consume_stream(
        self,
        stream: Iterable[SampleRecord],
        spec: DatasetSpec,
    ) -> tuple[Tuple[SampleRecord, ...], list[ErrorRecord]]:
        records: list[SampleRecord] = []
        errors: list[ErrorRecord] = []
        try:
            for index, candidate in enumerate(stream):
                try:
                    record = SampleRecord.model_validate(candidate)
                except ValidationError as exc:
                    errors.append(
                        self._error(
                            spec,
                            code="SAMPLE_RECORD_SCHEMA_INVALID",
                            field="sample_record",
                            expected="valid SampleRecord",
                            observed={"index": index, "errors": exc.error_count()},
                        )
                    )
                    continue
                records.append(record)
        except Exception as exc:
            errors.append(
                self._error(
                    spec,
                    code="SAMPLE_STREAM_READ_FAILED",
                    field="sample_stream",
                    expected="complete readable sample metadata stream",
                    observed={"error_code": type(exc).__name__, "message": str(exc)},
                )
            )
        return tuple(records), errors

    def _validate_records(
        self,
        spec: DatasetSpec,
        records: Tuple[SampleRecord, ...],
        *,
        verify_inputs: bool = True,
    ) -> list[ErrorRecord]:
        errors: list[ErrorRecord] = []
        if len(records) != spec.expected_sample_count:
            errors.append(
                self._error(
                    spec,
                    code="MANIFEST_SAMPLE_COUNT_MISMATCH",
                    field="sample_count",
                    expected=spec.expected_sample_count,
                    observed=len(records),
                )
            )

        normalized_ids = [normalize_id(record.sample_id) for record in records]
        duplicates = tuple(
            sorted(
                (sample_id for sample_id, count in Counter(normalized_ids).items() if count > 1),
                key=lambda value: value.encode("utf-8"),
            )
        )
        if duplicates:
            errors.append(
                self._error(
                    spec,
                    code="MANIFEST_DUPLICATE_SAMPLE_ID",
                    field="sample_id",
                    expected="unique Sample_ID values after canonical normalization",
                    observed={"duplicate_sample_ids": list(duplicates)},
                    unexpected_ids=duplicates,
                )
            )

        if spec.grouped:
            missing_groups = tuple(
                sorted(
                    (normalize_id(record.sample_id) for record in records if record.group_id is None),
                    key=lambda value: value.encode("utf-8"),
                )
            )
            if missing_groups:
                errors.append(
                    self._error(
                        spec,
                        code="MANIFEST_GROUP_ID_MISSING",
                        field="group_id",
                        expected="one non-empty Group_ID per sample",
                        observed={"sample_ids": list(missing_groups)},
                        missing_ids=missing_groups,
                    )
                )

        bad_dimensions: dict[str, object] = {}
        for record in records:
            observed_dimension = record.label_schema_metadata.get("dimension")
            if (
                isinstance(observed_dimension, bool)
                or not isinstance(observed_dimension, int)
                or observed_dimension != spec.label_schema.dimension
            ):
                bad_dimensions[normalize_id(record.sample_id)] = observed_dimension
        if bad_dimensions:
            errors.append(
                self._error(
                    spec,
                    code="MANIFEST_LABEL_DIMENSION_INVALID",
                    field="label_dimension",
                    expected=spec.label_schema.dimension,
                    observed={"sample_dimensions": bad_dimensions},
                )
            )

        if verify_inputs:
            unreadable: list[dict[str, str]] = []
            hash_mismatches: list[dict[str, str]] = []
            for record in records:
                for input_name, reference in record.input_references.items():
                    try:
                        observed_hash = self.input_hasher(reference)
                    except Exception as exc:
                        unreadable.append(
                            {
                                "sample_id": normalize_id(record.sample_id),
                                "input_name": input_name,
                                "input_reference": reference,
                                "read_error_code": type(exc).__name__,
                            }
                        )
                        continue
                    expected_hash = record.input_content_hashes[input_name]
                    if observed_hash.lower() != expected_hash.lower():
                        hash_mismatches.append(
                            {
                                "sample_id": normalize_id(record.sample_id),
                                "input_name": input_name,
                                "input_reference": reference,
                                "expected_hash": expected_hash,
                                "observed_hash": observed_hash,
                            }
                        )
            if unreadable:
                errors.append(
                    self._error(
                        spec,
                        code="MANIFEST_REQUIRED_INPUT_UNREADABLE",
                        field="input_reference",
                        expected="every required input is readable",
                        observed={"inputs": unreadable},
                    )
                )
            if hash_mismatches:
                errors.append(
                    self._error(
                        spec,
                        code="MANIFEST_INPUT_HASH_MISMATCH",
                        field="input_content_hashes",
                        expected="declared hashes match readable input bytes",
                        observed={"inputs": hash_mismatches},
                    )
                )
        return errors

    def _load_existing(
        self,
        relative_path: str,
        spec: DatasetSpec,
        schema_version: str,
        dataset_hash: str,
    ) -> Optional[SampleManifest]:
        path = self.artifact_store.root.joinpath(*relative_path.split("/"))
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            manifest = SampleManifest.model_validate_json(raw)
            if manifest.schema_version != schema_version:
                raise ValueError("schema version differs")
            if manifest.dataset_key != spec.dataset_key:
                raise ValueError("dataset key differs")
            expected = self._dataset_hash(
                spec,
                manifest.schema_version,
                tuple(sorted(manifest.records, key=lambda r: normalize_id(r.sample_id).encode("utf-8"))),
                manifest.canonical_order,
            )
            if manifest.dataset_hash != dataset_hash or expected != dataset_hash:
                raise ValueError("dataset hash differs from manifest identity")
            return manifest
        except Exception as exc:
            error = self._error(
                spec,
                code="MANIFEST_EXISTING_VERSION_INVALID",
                field="manifest_artifact",
                expected={"dataset_hash": dataset_hash, "schema_version": schema_version},
                observed={"path": relative_path, "error_code": type(exc).__name__},
            )
            self.error_journal.append(error)
            raise ManifestPublicationError(error) from exc

    @staticmethod
    def _encode_manifest(manifest: SampleManifest) -> bytes:
        return json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _manifest_relative_path(dataset_key: str, dataset_hash: str) -> str:
        # Dataset keys are identifiers rather than paths. Encode separators and
        # other path-significant characters while retaining readable keys.
        from urllib.parse import quote

        encoded_key = quote(dataset_key, safe="-._")
        return f"manifests/{encoded_key}/{dataset_hash}.json"

    def _reject(self, errors: Iterable[ErrorRecord]) -> None:
        records = tuple(errors)
        for error in records:
            self.error_journal.append(error)
        raise ManifestMaterializationError(records)

    @staticmethod
    def _error(
        spec: DatasetSpec,
        *,
        code: str,
        field: str,
        expected: object,
        observed: object,
        missing_ids: Tuple[str, ...] = (),
        unexpected_ids: Tuple[str, ...] = (),
    ) -> ErrorRecord:
        return ErrorRecord(
            error_code=code,
            failed_stage="manifest_validation",
            affected_identifiers={"dataset_key": spec.dataset_key},
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            missing_ids=missing_ids,
            unexpected_ids=unexpected_ids,
            invariant_checks={
                "protocol_creation_blocked": True,
                "previous_manifests_unchanged": True,
                "new_manifest_unpublished": True,
            },
        )
