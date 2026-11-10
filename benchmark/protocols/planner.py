"""Protocol-first split planning, invariant validation, and publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar

from benchmark.recovery.artifact_store import FailureAtomicArtifactStore
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import ValidationIssue, ValidationReport, ValidationSeverity
from benchmark.schema.canonical import canonical_deserialize, canonical_serialize
from benchmark.schema.datasets import SampleManifest
from benchmark.schema.protocol import (
    EvaluationBudget,
    OuterFold,
    ProtocolManifest,
    SelectionPolicy,
)
from benchmark.schema.records import ArtifactReference, ErrorRecord


ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class PublishedProtocol:
    """A validated protocol and its immutable published artifact reference."""

    manifest: ProtocolManifest
    artifact: ArtifactReference


class ProtocolPlanningError(ValueError):
    """Protocol invariant failure with all emitted machine-readable records."""

    def __init__(
        self,
        report: ValidationReport,
        error_records: Tuple[ErrorRecord, ...],
    ) -> None:
        self.report = report
        self.error_records = error_records
        codes = ", ".join(issue.code for issue in report.errors)
        super().__init__(f"protocol invariant validation failed: {codes}")


class ProtocolReloadError(ValueError):
    """Published protocol bytes could not be reloaded without identity drift."""


class ProtocolComparabilityError(ValueError):
    """Two model runs do not share the same protocol identity and memberships."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.errors)
        super().__init__(f"protocol comparison rejected: {codes}")


class ProtocolPlanner:
    """Create and publish protocol manifests before model construction.

    Split memberships are supplied independently of any model. The planner
    canonicalizes them through :class:`ProtocolManifest`, validates all outer
    and inner sample/group boundaries against the immutable sample manifest,
    and only then writes the protocol through the failure-atomic artifact store.
    """

    def __init__(
        self,
        artifact_store: FailureAtomicArtifactStore,
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.error_journal = error_journal or artifact_store.error_journal

    def validate(
        self,
        sample_manifest: SampleManifest,
        protocol_manifest: ProtocolManifest,
    ) -> ValidationReport:
        """Validate dataset binding, coverage, and all split boundaries."""

        expected_ids = set(sample_manifest.canonical_order)
        group_by_sample = {
            record.sample_id: record.group_id for record in sample_manifest.records
        }
        grouped = any(group_id is not None for group_id in group_by_sample.values())
        issues = []

        if protocol_manifest.dataset_hash != sample_manifest.dataset_hash:
            issues.append(
                self._issue(
                    "PROTOCOL_DATASET_HASH_MISMATCH",
                    ("dataset_hash",),
                    "protocol must bind to the supplied Sample_Manifest",
                    sample_manifest.dataset_hash,
                    protocol_manifest.dataset_hash,
                )
            )

        for outer_index, outer_fold in enumerate(protocol_manifest.folds):
            outer_path = ("folds", str(outer_index))
            train_ids = set(outer_fold.outer_train_ids)
            test_ids = set(outer_fold.outer_test_ids)
            overlap = self._ordered(train_ids & test_ids)
            if overlap:
                issues.append(
                    self._issue(
                        "OUTER_SAMPLE_ISOLATION_FAILED",
                        outer_path + ("sample_ids",),
                        "outer train and test Sample_IDs must be disjoint",
                        [],
                        overlap,
                    )
                )

            observed_ids = train_ids | test_ids
            missing = self._ordered(expected_ids - observed_ids)
            unexpected = self._ordered(observed_ids - expected_ids)
            if missing or unexpected:
                issues.append(
                    self._issue(
                        "OUTER_COVERAGE_FAILED",
                        outer_path + ("coverage",),
                        "outer train/test union must equal the Sample_Manifest Sample_ID set",
                        {"missing_ids": [], "unexpected_ids": []},
                        {"missing_ids": missing, "unexpected_ids": unexpected},
                    )
                )

            if grouped:
                outer_group_overlap = self._ordered(
                    self._groups(train_ids, group_by_sample)
                    & self._groups(test_ids, group_by_sample)
                )
                if outer_group_overlap:
                    issues.append(
                        self._issue(
                            "OUTER_GROUP_ISOLATION_FAILED",
                            outer_path + ("group_ids",),
                            "outer train and test Group_IDs must be disjoint",
                            [],
                            outer_group_overlap,
                        )
                    )

            for inner_index, inner_fold in enumerate(outer_fold.inner_folds):
                inner_path = outer_path + ("inner_folds", str(inner_index))
                inner_train = set(inner_fold.train_ids)
                inner_validation = set(inner_fold.validation_ids)
                inner_overlap = self._ordered(inner_train & inner_validation)
                if inner_overlap:
                    issues.append(
                        self._issue(
                            "INNER_SAMPLE_ISOLATION_FAILED",
                            inner_path + ("sample_ids",),
                            "inner train and validation Sample_IDs must be disjoint",
                            [],
                            inner_overlap,
                        )
                    )

                out_of_bound = self._ordered(
                    (inner_train | inner_validation) - train_ids
                )
                if out_of_bound:
                    issues.append(
                        self._issue(
                            "INNER_CONTAINMENT_FAILED",
                            inner_path + ("sample_ids",),
                            "inner train/validation Sample_IDs must be contained in outer train",
                            [],
                            out_of_bound,
                        )
                    )

                if grouped:
                    inner_group_overlap = self._ordered(
                        self._groups(inner_train, group_by_sample)
                        & self._groups(inner_validation, group_by_sample)
                    )
                    if inner_group_overlap:
                        issues.append(
                            self._issue(
                                "INNER_GROUP_ISOLATION_FAILED",
                                inner_path + ("group_ids",),
                                "inner train and validation Group_IDs must be disjoint",
                                [],
                                inner_group_overlap,
                            )
                        )

        return ValidationReport(issues=tuple(issues))

    def plan(
        self,
        sample_manifest: SampleManifest,
        *,
        schema_version: str,
        protocol_key: str,
        folds: Iterable[OuterFold],
        selection_policy: SelectionPolicy,
        seed_set: Iterable[int],
        evaluation_budget: EvaluationBudget,
    ) -> ProtocolManifest:
        """Build a canonical protocol and reject it if any invariant fails."""

        protocol = ProtocolManifest.create(
            schema_version=schema_version,
            protocol_key=protocol_key,
            dataset_hash=sample_manifest.dataset_hash,
            folds=tuple(folds),
            selection_policy=selection_policy,
            seed_set=tuple(seed_set),
            evaluation_budget=evaluation_budget,
        )
        report = self.validate(sample_manifest, protocol)
        if not report.passes:
            records = tuple(
                self._error_record(protocol, issue) for issue in report.errors
            )
            for record in records:
                self.error_journal.append(record)
            raise ProtocolPlanningError(report, records)
        return protocol

    def plan_and_publish(
        self,
        sample_manifest: SampleManifest,
        *,
        schema_version: str,
        protocol_key: str,
        folds: Iterable[OuterFold],
        selection_policy: SelectionPolicy,
        seed_set: Iterable[int],
        evaluation_budget: EvaluationBudget,
        relative_path: Optional[str] = None,
    ) -> PublishedProtocol:
        """Validate first, then atomically publish one immutable protocol."""

        protocol = self.plan(
            sample_manifest,
            schema_version=schema_version,
            protocol_key=protocol_key,
            folds=folds,
            selection_policy=selection_policy,
            seed_set=seed_set,
            evaluation_budget=evaluation_budget,
        )
        path = relative_path or self._default_relative_path(protocol)
        artifact = self.artifact_store.publish_bytes(
            artifact_id=f"protocol-{protocol.protocol_hash}",
            relative_path=path,
            content=canonical_serialize(protocol),
            media_type="application/json",
        )
        return PublishedProtocol(manifest=protocol, artifact=artifact)

    def dry_run(
        self,
        sample_manifest: SampleManifest,
        *,
        schema_version: str,
        protocol_key: str,
        folds: Iterable[OuterFold],
        selection_policy: SelectionPolicy,
        seed_set: Iterable[int],
        evaluation_budget: EvaluationBudget,
        relative_path: Optional[str] = None,
    ) -> PublishedProtocol:
        """Plan, validate, and publish a protocol without constructing a model."""

        return self.plan_and_publish(
            sample_manifest,
            schema_version=schema_version,
            protocol_key=protocol_key,
            folds=folds,
            selection_policy=selection_policy,
            seed_set=seed_set,
            evaluation_budget=evaluation_budget,
            relative_path=relative_path,
        )

    @staticmethod
    def reload(content: bytes) -> ProtocolManifest:
        """Reload exact canonical protocol bytes and verify membership and hash."""

        if not isinstance(content, bytes):
            raise TypeError("protocol content must be bytes")
        try:
            decoded = canonical_deserialize(content)
            manifest = ProtocolManifest.model_validate(decoded)
        except Exception as exc:
            raise ProtocolReloadError("protocol content failed schema or hash validation") from exc
        if canonical_serialize(manifest) != content:
            raise ProtocolReloadError(
                "protocol content is valid but not the exact canonical serialization"
            )
        return manifest

    @classmethod
    def validate_comparability(
        cls,
        reference: ProtocolManifest,
        candidate: ProtocolManifest,
    ) -> ValidationReport:
        """Check protocol hash and every outer/inner fold sample set."""

        issues = []
        if candidate.protocol_hash != reference.protocol_hash:
            issues.append(
                cls._issue(
                    "PROTOCOL_COMPARISON_HASH_MISMATCH",
                    ("protocol_hash",),
                    "compared models must use the same Protocol_Hash",
                    reference.protocol_hash,
                    candidate.protocol_hash,
                )
            )

        reference_memberships = cls._fold_memberships(reference)
        candidate_memberships = cls._fold_memberships(candidate)
        if candidate_memberships != reference_memberships:
            issues.append(
                cls._issue(
                    "PROTOCOL_COMPARISON_FOLD_MEMBERSHIP_MISMATCH",
                    ("folds",),
                    "compared models must use identical fold Sample_ID sets",
                    reference_memberships,
                    candidate_memberships,
                )
            )
        return ValidationReport(issues=tuple(issues))

    @classmethod
    def require_comparable(
        cls,
        reference: ProtocolManifest,
        candidate: ProtocolManifest,
    ) -> None:
        """Reject a model comparison unless protocol hash and memberships match."""

        report = cls.validate_comparability(reference, candidate)
        if not report.passes:
            raise ProtocolComparabilityError(report)

    def plan_publish_and_build(
        self,
        sample_manifest: SampleManifest,
        *,
        model_builder: Callable[[ProtocolManifest], ModelT],
        schema_version: str,
        protocol_key: str,
        folds: Iterable[OuterFold],
        selection_policy: SelectionPolicy,
        seed_set: Iterable[int],
        evaluation_budget: EvaluationBudget,
        relative_path: Optional[str] = None,
    ) -> tuple[PublishedProtocol, ModelT]:
        """Publish the protocol completely before invoking a model builder."""

        published = self.plan_and_publish(
            sample_manifest,
            schema_version=schema_version,
            protocol_key=protocol_key,
            folds=folds,
            selection_policy=selection_policy,
            seed_set=seed_set,
            evaluation_budget=evaluation_budget,
            relative_path=relative_path,
        )
        model = model_builder(published.manifest)
        return published, model

    @staticmethod
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

    @staticmethod
    def _ordered(values: Iterable[str]) -> list[str]:
        return sorted(values, key=lambda value: value.encode("utf-8"))

    @staticmethod
    def _groups(
        sample_ids: Iterable[str],
        group_by_sample: dict[str, Optional[str]],
    ) -> set[str]:
        return {
            group_id
            for sample_id in sample_ids
            if (group_id := group_by_sample.get(sample_id)) is not None
        }

    @staticmethod
    def _fold_memberships(protocol: ProtocolManifest) -> dict[str, Any]:
        return {
            fold.fold_id: {
                "outer_train_ids": fold.outer_train_ids,
                "outer_test_ids": fold.outer_test_ids,
                "inner_folds": {
                    inner.fold_id: {
                        "train_ids": inner.train_ids,
                        "validation_ids": inner.validation_ids,
                    }
                    for inner in fold.inner_folds
                },
            }
            for fold in protocol.folds
        }

    @staticmethod
    def _default_relative_path(protocol: ProtocolManifest) -> str:
        return f"protocols/{protocol.protocol_hash}.canonical.json"

    @staticmethod
    def _error_record(
        protocol: ProtocolManifest,
        issue: ValidationIssue,
    ) -> ErrorRecord:
        outer_index = int(issue.field_path[1]) if len(issue.field_path) > 1 else None
        outer_fold = protocol.folds[outer_index] if outer_index is not None else None
        affected = {"protocol_key": protocol.protocol_key}
        if outer_fold is not None:
            affected["fold_id"] = outer_fold.fold_id
        if "inner_folds" in issue.field_path and outer_fold is not None:
            marker = issue.field_path.index("inner_folds")
            inner_index = int(issue.field_path[marker + 1])
            affected["inner_fold_id"] = outer_fold.inner_folds[inner_index].fold_id

        observed = issue.observed_value
        missing_ids: Tuple[str, ...] = ()
        unexpected_ids: Tuple[str, ...] = ()
        if issue.code == "OUTER_COVERAGE_FAILED" and isinstance(observed, dict):
            missing_ids = tuple(observed.get("missing_ids", ()))
            unexpected_ids = tuple(observed.get("unexpected_ids", ()))
        elif isinstance(observed, list):
            unexpected_ids = tuple(observed)

        return ErrorRecord(
            error_code=issue.code,
            failed_stage="protocol_validation",
            affected_identifiers=affected,
            failing_field=".".join(issue.field_path),
            expected_value=issue.expected_value,
            observed_value=issue.observed_value,
            missing_ids=missing_ids,
            unexpected_ids=unexpected_ids,
            invariant_checks={"protocol_artifact_unchanged": True},
        )


__all__ = [
    "ProtocolComparabilityError",
    "ProtocolPlanner",
    "ProtocolPlanningError",
    "ProtocolReloadError",
    "PublishedProtocol",
]
