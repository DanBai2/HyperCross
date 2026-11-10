"""Inner-only candidate selection and frozen outer-refit planning."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar
from uuid import uuid4

from pydantic import ValidationError

from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import (
    ClaimStatus,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from benchmark.schema.canonical import normalize_id
from benchmark.schema.preprocessing import canonical_sample_ids
from benchmark.schema.protocol import OuterFold, SelectionPolicy
from benchmark.schema.records import ErrorRecord
from benchmark.schema.selection import (
    OuterRefitRequest,
    SelectionCandidate,
    SelectionDecision,
)


RefitT = TypeVar("RefitT")


class SelectionValidationError(ValueError):
    """Selection or refit rejection with machine-readable failure records."""

    def __init__(
        self,
        report: ValidationReport,
        error_records: Tuple[ErrorRecord, ...],
    ) -> None:
        self.report = report
        self.error_records = error_records
        self.error_record = error_records[0]
        self.claim_status = (
            ClaimStatus.INVALID_OR_LEAKY
            if any(
                issue.code
                in {
                    "SELECTION_OUTER_EVIDENCE_FORBIDDEN",
                    "SELECTION_SOURCE_OUTSIDE_OUTER_TRAIN",
                    "SELECTION_SOURCE_NOT_INNER_EVIDENCE",
                }
                for issue in report.errors
            )
            else None
        )
        codes = ", ".join(issue.code for issue in report.errors)
        super().__init__(f"selection validation failed: {codes}")


class SelectionController:
    """Freeze decisions from inner predictions and safely prepare outer refits.

    This component accepts scores, sample identities, and prediction hashes only;
    it has no label-reader dependency or outer-label capability. Every evidence
    Sample_ID must come from an inner validation split and remain inside the
    corresponding Outer_Train_Set.
    """

    _UTF8_TIE_BREAK = "candidate_id_utf8_ascending"

    def __init__(self, *, error_journal: Optional[ErrorJournal] = None) -> None:
        self.error_journal = error_journal

    def freeze_selection(
        self,
        outer_fold: OuterFold,
        *,
        candidates: Iterable[SelectionCandidate],
        candidate_scores: Mapping[str, float],
        source_sample_ids: Iterable[str],
        source_prediction_hash: str,
        selection_policy: SelectionPolicy,
        decision_id: Optional[str] = None,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Validate inner-only evidence, choose deterministically, and freeze it."""

        pending_id = normalize_id(decision_id or f"selection-{uuid4()}")
        try:
            candidate_tuple = tuple(
                candidate
                if isinstance(candidate, SelectionCandidate)
                else SelectionCandidate.model_validate(candidate)
                for candidate in candidates
            )
            source_ids = canonical_sample_ids(
                source_sample_ids, field_name="source_sample_ids"
            )
        except (TypeError, ValueError, ValidationError) as exc:
            self._reject_single(
                code="SELECTION_INPUT_INVALID",
                field="source_sample_ids",
                expected="a non-empty unique inner-validation Sample_ID set",
                observed=str(exc),
                outer_fold=outer_fold,
                decision_id=pending_id,
            )

        boundary_report = self.validate_source_boundary(outer_fold, source_ids)
        if not boundary_report.passes:
            self._reject_report(boundary_report, outer_fold, pending_id)

        try:
            normalized_scores = self._normalize_scores(candidate_scores)
            selected_candidate = self._select_candidate(
                candidate_tuple,
                normalized_scores,
                selection_policy,
            )
            create_values: dict[str, Any] = {
                "decision_id": pending_id,
                "fold_id": outer_fold.fold_id,
                "candidates": candidate_tuple,
                "candidate_scores": normalized_scores,
                "source_sample_ids": source_ids,
                "metric_direction": selection_policy.metric_direction,
                "tie_break_rule": selection_policy.tie_break_rule,
                "source_prediction_hash": source_prediction_hash,
                "selected_candidate": selected_candidate,
            }
            if frozen_utc is not None:
                create_values["frozen_utc"] = frozen_utc
            return SelectionDecision.create(**create_values)
        except (TypeError, ValueError, ValidationError) as exc:
            self._reject_single(
                code="SELECTION_DECISION_INVALID",
                field="candidate_scores",
                expected=(
                    "one finite score for every unique candidate and a deterministic "
                    "predeclared tie-break"
                ),
                observed=str(exc),
                outer_fold=outer_fold,
                decision_id=pending_id,
            )

    # Design-level concise operation name.
    freeze = freeze_selection

    @classmethod
    def validate_source_boundary(
        cls,
        outer_fold: OuterFold,
        source_sample_ids: Iterable[str],
    ) -> ValidationReport:
        """Check that all selection evidence is inner validation evidence."""

        source_ids = set(source_sample_ids)
        outer_train = set(outer_fold.outer_train_ids)
        outer_test = set(outer_fold.outer_test_ids)
        inner_validation = {
            sample_id
            for inner_fold in outer_fold.inner_folds
            for sample_id in inner_fold.validation_ids
        }
        issues = []

        forbidden = cls._ordered(source_ids & outer_test)
        if forbidden:
            issues.append(
                cls._issue(
                    "SELECTION_OUTER_EVIDENCE_FORBIDDEN",
                    ("source_sample_ids",),
                    "selection evidence must not reference Outer_Test_Set samples",
                    [],
                    forbidden,
                )
            )

        outside_outer_train = cls._ordered(source_ids - outer_train)
        if outside_outer_train:
            issues.append(
                cls._issue(
                    "SELECTION_SOURCE_OUTSIDE_OUTER_TRAIN",
                    ("source_sample_ids",),
                    "selection evidence must be contained in Outer_Train_Set",
                    [],
                    outside_outer_train,
                )
            )

        not_inner_evidence = cls._ordered(source_ids - inner_validation)
        if not_inner_evidence:
            issues.append(
                cls._issue(
                    "SELECTION_SOURCE_NOT_INNER_EVIDENCE",
                    ("source_sample_ids",),
                    "selection evidence must come from declared inner validation splits",
                    [],
                    not_inner_evidence,
                )
            )

        if not outer_fold.inner_folds:
            issues.append(
                cls._issue(
                    "SELECTION_INNER_FOLDS_REQUIRED",
                    ("inner_folds",),
                    "candidate selection requires declared Inner_Folds",
                    "at least one Inner_Fold",
                    0,
                )
            )
        return ValidationReport(issues=tuple(issues))

    def prepare_refit(
        self,
        outer_fold: OuterFold,
        decision: SelectionDecision,
    ) -> OuterRefitRequest:
        """Bind an unchanged decision to the complete Outer_Train_Set."""

        try:
            decision.require_integrity()
        except ValueError as exc:
            self._reject_single(
                code="SELECTION_DECISION_MUTATED",
                field="decision_hash",
                expected=decision.decision_hash,
                observed=str(exc),
                outer_fold=outer_fold,
                decision_id=decision.decision_id,
                stage="outer_refit_validation",
            )

        if decision.fold_id != outer_fold.fold_id:
            self._reject_single(
                code="SELECTION_FOLD_MISMATCH",
                field="fold_id",
                expected=outer_fold.fold_id,
                observed=decision.fold_id,
                outer_fold=outer_fold,
                decision_id=decision.decision_id,
                stage="outer_refit_validation",
            )

        boundary_report = self.validate_source_boundary(
            outer_fold, decision.source_sample_ids
        )
        if not boundary_report.passes:
            self._reject_report(
                boundary_report,
                outer_fold,
                decision.decision_id,
                stage="outer_refit_validation",
            )

        return OuterRefitRequest(
            fold_id=outer_fold.fold_id,
            outer_train_ids=outer_fold.outer_train_ids,
            selection_decision=decision,
        )

    def apply_refit(
        self,
        outer_fold: OuterFold,
        decision: SelectionDecision,
        refit: Callable[[OuterRefitRequest], RefitT],
    ) -> RefitT:
        """Invoke refit once with all outer-train IDs and the frozen decision."""

        if not callable(refit):
            raise TypeError("refit must be callable")
        request = self.prepare_refit(outer_fold, decision)
        result = refit(request)
        try:
            request.selection_decision.require_integrity()
        except ValueError as exc:
            self._reject_single(
                code="SELECTION_DECISION_MUTATED",
                field="decision_hash",
                expected=decision.decision_hash,
                observed=str(exc),
                outer_fold=outer_fold,
                decision_id=decision.decision_id,
                stage="outer_refit_validation",
            )
        return result

    @staticmethod
    def _normalize_scores(candidate_scores: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(candidate_scores, Mapping):
            raise TypeError("candidate_scores must be a mapping")
        normalized: dict[str, float] = {}
        for candidate_id, score in candidate_scores.items():
            if not isinstance(candidate_id, str):
                raise TypeError("candidate score keys must be candidate IDs")
            key = normalize_id(candidate_id)
            if key in normalized:
                raise ValueError("candidate score IDs collide after normalization")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("candidate scores must be finite numbers")
            normalized[key] = float(score)
        return normalized

    @classmethod
    def _select_candidate(
        cls,
        candidates: Sequence[SelectionCandidate],
        scores: Mapping[str, float],
        policy: SelectionPolicy,
    ) -> str:
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if not candidate_ids:
            raise ValueError("candidate set must not be empty")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate set contains duplicate candidate IDs")
        if set(scores) != set(candidate_ids):
            raise ValueError("candidate scores must cover the complete candidate set")

        best_score = (
            max(scores.values())
            if policy.metric_direction.value == "maximize"
            else min(scores.values())
        )
        tied = [candidate_id for candidate_id in candidate_ids if scores[candidate_id] == best_score]
        if len(tied) > 1 and policy.tie_break_rule != cls._UTF8_TIE_BREAK:
            raise ValueError(
                "tied scores require tie_break_rule candidate_id_utf8_ascending"
            )
        return min(tied, key=lambda value: value.encode("utf-8"))

    @staticmethod
    def _ordered(values: Iterable[str]) -> list[str]:
        return sorted(values, key=lambda value: value.encode("utf-8"))

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

    def _reject_report(
        self,
        report: ValidationReport,
        outer_fold: OuterFold,
        decision_id: str,
        *,
        stage: str = "selection_validation",
    ) -> None:
        records = tuple(
            self._error_record(issue, outer_fold, decision_id, stage=stage)
            for issue in report.errors
        )
        if self.error_journal is not None:
            for record in records:
                self.error_journal.append(record)
        raise SelectionValidationError(report, records)

    def _reject_single(
        self,
        *,
        code: str,
        field: str,
        expected: Any,
        observed: Any,
        outer_fold: OuterFold,
        decision_id: str,
        stage: str = "selection_validation",
    ) -> None:
        issue = self._issue(code, (field,), code, expected, observed)
        self._reject_report(
            ValidationReport(issues=(issue,)),
            outer_fold,
            decision_id,
            stage=stage,
        )

    @staticmethod
    def _error_record(
        issue: ValidationIssue,
        outer_fold: OuterFold,
        decision_id: str,
        *,
        stage: str,
    ) -> ErrorRecord:
        observed = issue.observed_value
        unexpected_ids = tuple(observed) if isinstance(observed, list) else ()
        return ErrorRecord(
            error_code=issue.code,
            failed_stage=stage,
            affected_identifiers={
                "selection_decision_id": decision_id,
                "fold_id": outer_fold.fold_id,
            },
            failing_field=".".join(issue.field_path),
            expected_value=issue.expected_value,
            observed_value=observed,
            unexpected_ids=unexpected_ids,
            invariant_checks={
                "checkpoint_unchanged": True,
                "prediction_artifact_unchanged": True,
                "completed_artifacts_unchanged": True,
            },
        )


__all__ = ["SelectionController", "SelectionValidationError"]
