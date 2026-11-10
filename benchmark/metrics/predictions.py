"""Shard-aware validation and atomic publication of prediction artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union

from pydantic import ValidationError

from benchmark.recovery.artifact_store import FailureAtomicArtifactStore
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from benchmark.schema.canonical import canonical_deserialize, canonical_serialize, normalize_id
from benchmark.schema.predictions import (
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    PredictionShard,
    canonical_prediction_key,
    prediction_key_text,
)
from benchmark.schema.records import ArtifactReference, ErrorRecord


PredictionRecordInput = Union[PredictionRecord, dict[str, Any]]
PredictionShardInput = Union[PredictionShard, Iterable[PredictionRecordInput]]


@dataclass(frozen=True)
class PublishedPredictionArtifact:
    """A validated prediction artifact and its immutable file reference."""

    artifact: PredictionArtifact
    reference: ArtifactReference


class PredictionArtifactValidationError(ValueError):
    """Prediction coverage or key validation failed before publication."""

    def __init__(self, report: ValidationReport, error_record: ErrorRecord) -> None:
        self.report = report
        self.error_record = error_record
        codes = ", ".join(issue.code for issue in report.errors)
        super().__init__(f"prediction artifact validation failed: {codes}")


class PredictionArtifactReloadError(ValueError):
    """Published prediction bytes failed canonical or schema validation."""


def _ordered_ids(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


def _issue(
    code: str,
    field_path: Tuple[str, ...],
    message: str,
    expected: Any,
    observed: Any,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=ValidationSeverity.ERROR,
        field_path=field_path,
        message=message,
        expected_value=expected,
        observed_value=observed,
    )


def validate_prediction_records(
    records: Iterable[PredictionRecord],
    expected_sample_ids: Iterable[str],
) -> ValidationReport:
    """Validate exact sample coverage, key uniqueness, and run context.

    Validation does not read labels or compute metrics. It operates entirely on
    already-produced records and is safe to call before any artifact target is
    opened.
    """

    records = tuple(records)
    expected = tuple(normalize_id(sample_id) for sample_id in expected_sample_ids)
    expected_set = set(expected)
    issues = []

    if not records:
        issues.append(
            _issue(
                "PREDICTION_RECORDS_EMPTY",
                ("records",),
                "prediction records must not be empty",
                "one record for every expected Sample_ID",
                [],
            )
        )

    key_counts = Counter(record.key for record in records)
    duplicate_keys = tuple(
        prediction_key_text(key)
        for key, count in sorted(
            key_counts.items(), key=lambda item: canonical_prediction_key(item[0])
        )
        if count > 1
    )
    if duplicate_keys:
        issues.append(
            _issue(
                "DUPLICATE_PREDICTION_KEYS",
                ("records", "key"),
                "every Prediction_Key must be unique",
                [],
                list(duplicate_keys),
            )
        )

    sample_counts = Counter(record.key.sample_id for record in records)
    duplicate_sample_ids = _ordered_ids(
        sample_id for sample_id, count in sample_counts.items() if count > 1
    )
    if duplicate_sample_ids:
        issues.append(
            _issue(
                "DUPLICATE_PREDICTION_SAMPLE_IDS",
                ("records", "key", "sample_id"),
                "each expected Sample_ID must have exactly one prediction record",
                [],
                list(duplicate_sample_ids),
            )
        )

    observed_set = set(sample_counts)
    missing = _ordered_ids(expected_set - observed_set)
    unexpected = _ordered_ids(observed_set - expected_set)
    if missing or unexpected:
        issues.append(
            _issue(
                "PREDICTION_COVERAGE_MISMATCH",
                ("records", "key", "sample_id"),
                "prediction Sample_IDs must exactly equal expected_sample_ids",
                {"missing_ids": [], "unexpected_ids": []},
                {
                    "missing_ids": list(missing),
                    "unexpected_ids": list(unexpected),
                },
            )
        )

    contexts = {
        (record.key.fold_id, record.key.seed, record.key.inference_branch)
        for record in records
    }
    if len(contexts) > 1:
        observed_contexts = [
            {
                "fold_id": fold_id,
                "seed": seed,
                "inference_branch": inference_branch,
            }
            for fold_id, seed, inference_branch in sorted(
                contexts,
                key=lambda item: (
                    item[0].encode("utf-8"),
                    item[1],
                    item[2].encode("utf-8"),
                ),
            )
        ]
        issues.append(
            _issue(
                "PREDICTION_CONTEXT_MISMATCH",
                ("records", "key"),
                "one artifact must contain one Fold_ID, seed, and Inference_Branch",
                "one evaluation context",
                observed_contexts,
            )
        )

    return ValidationReport(issues=tuple(issues))


class PredictionArtifactWriter:
    """Merge bounded prediction shards, validate, then publish atomically."""

    def __init__(
        self,
        artifact_store: FailureAtomicArtifactStore,
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.error_journal = error_journal or artifact_store.error_journal

    @staticmethod
    def encode_shard(shard: PredictionShard) -> bytes:
        """Encode a validated shard for transport without publishing it."""

        if not isinstance(shard, PredictionShard):
            raise TypeError("shard must be a PredictionShard")
        return canonical_serialize(shard)

    @staticmethod
    def reload_shard(content: bytes) -> PredictionShard:
        """Reload exact canonical shard bytes."""

        if not isinstance(content, bytes):
            raise TypeError("prediction shard content must be bytes")
        try:
            shard = PredictionShard.model_validate(canonical_deserialize(content))
        except Exception as exc:
            raise PredictionArtifactReloadError(
                "prediction shard failed schema validation"
            ) from exc
        if canonical_serialize(shard) != content:
            raise PredictionArtifactReloadError(
                "prediction shard is valid but not exactly canonical"
            )
        return shard

    def build(
        self,
        *,
        artifact_id: str,
        run_id: str,
        expected_sample_ids: Iterable[str],
        shards: Iterable[PredictionShardInput],
        schema_version: str = "1.0",
    ) -> PredictionArtifact:
        """Merge shards and return a complete artifact after all checks pass."""

        expected_ids = tuple(expected_sample_ids)
        records = self._merge_shards(shards)
        report = validate_prediction_records(records, expected_ids)
        if not report.passes:
            error_record = self._error_record(
                artifact_id=artifact_id,
                run_id=run_id,
                records=records,
                expected_sample_ids=expected_ids,
                report=report,
            )
            self.error_journal.append(error_record)
            raise PredictionArtifactValidationError(report, error_record)

        return PredictionArtifact.create(
            schema_version=schema_version,
            artifact_id=artifact_id,
            run_id=run_id,
            records=records,
            expected_sample_ids=expected_ids,
            complete=True,
        )

    def publish_shards(
        self,
        *,
        artifact_id: str,
        run_id: str,
        expected_sample_ids: Iterable[str],
        shards: Iterable[PredictionShardInput],
        schema_version: str = "1.0",
        relative_path: Optional[str] = None,
    ) -> PublishedPredictionArtifact:
        """Validate all shards before opening an immutable publication target."""

        artifact = self.build(
            artifact_id=artifact_id,
            run_id=run_id,
            expected_sample_ids=expected_sample_ids,
            shards=shards,
            schema_version=schema_version,
        )
        path = relative_path or (
            f"predictions/{artifact.content_hash}.canonical.json"
        )
        reference = self.artifact_store.publish_bytes(
            artifact_id=artifact.artifact_id,
            relative_path=path,
            content=canonical_serialize(artifact),
            media_type="application/json",
        )
        return PublishedPredictionArtifact(artifact=artifact, reference=reference)

    @staticmethod
    def reload(content: bytes) -> PredictionArtifact:
        """Reload exact canonical bytes for a complete validated artifact."""

        if not isinstance(content, bytes):
            raise TypeError("prediction artifact content must be bytes")
        try:
            artifact = PredictionArtifact.model_validate(canonical_deserialize(content))
        except Exception as exc:
            raise PredictionArtifactReloadError(
                "prediction artifact failed schema, hash, or completeness validation"
            ) from exc
        if not artifact.complete:
            raise PredictionArtifactReloadError(
                "prediction artifact is not marked complete"
            )
        if canonical_serialize(artifact) != content:
            raise PredictionArtifactReloadError(
                "prediction artifact is valid but not exactly canonical"
            )
        return artifact

    @staticmethod
    def _merge_shards(
        shards: Iterable[PredictionShardInput],
    ) -> Tuple[PredictionRecord, ...]:
        records = []
        shard_ids = set()
        for index, shard in enumerate(shards):
            if isinstance(shard, PredictionShard):
                if shard.shard_id in shard_ids:
                    raise ValueError(f"duplicate prediction shard_id: {shard.shard_id}")
                shard_ids.add(shard.shard_id)
                shard_records = shard.records
            else:
                try:
                    shard_records = tuple(
                        record
                        if isinstance(record, PredictionRecord)
                        else PredictionRecord.model_validate(record)
                        for record in shard
                    )
                except (TypeError, ValidationError) as exc:
                    raise ValueError(
                        f"prediction shard {index} contains an invalid record"
                    ) from exc
            records.extend(shard_records)
        return tuple(
            sorted(records, key=lambda record: canonical_prediction_key(record.key))
        )

    @staticmethod
    def _error_record(
        *,
        artifact_id: str,
        run_id: str,
        records: Tuple[PredictionRecord, ...],
        expected_sample_ids: Tuple[str, ...],
        report: ValidationReport,
    ) -> ErrorRecord:
        observed_ids = {record.key.sample_id for record in records}
        expected_ids = {normalize_id(sample_id) for sample_id in expected_sample_ids}
        duplicate_keys = tuple(
            prediction_key_text(key)
            for key, count in sorted(
                Counter(record.key for record in records).items(),
                key=lambda item: canonical_prediction_key(item[0]),
            )
            if count > 1
        )
        return ErrorRecord(
            error_code="PREDICTION_ARTIFACT_VALIDATION_FAILED",
            failed_stage="prediction_artifact_validation",
            affected_identifiers={
                "artifact_id": str(artifact_id),
                "run_id": str(run_id),
            },
            failing_field="records",
            expected_value={
                "sample_ids": list(_ordered_ids(expected_ids)),
                "unique_prediction_keys": True,
            },
            observed_value={
                "sample_ids": list(_ordered_ids(observed_ids)),
                "duplicate_prediction_keys": list(duplicate_keys),
                "issue_codes": [issue.code for issue in report.errors],
            },
            missing_ids=_ordered_ids(expected_ids - observed_ids),
            unexpected_ids=_ordered_ids(observed_ids - expected_ids),
            invariant_checks={
                "prediction_artifact_unchanged": True,
                "aggregate_records_unchanged": True,
                "metrics_not_computed": True,
                "publication_target_not_opened": True,
            },
        )


__all__ = [
    "PredictionArtifactReloadError",
    "PredictionArtifactValidationError",
    "PredictionArtifactWriter",
    "PublishedPredictionArtifact",
    "validate_prediction_records",
]
