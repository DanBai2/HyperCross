"""Targeted unit coverage for task 9.2 frozen inner-only selection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.engine import SelectionController, SelectionValidationError
from benchmark.registry import ErrorJournal
from benchmark.schema import (
    InnerFold,
    MetricDirection,
    OuterFold,
    SelectionCandidate,
    SelectionPolicy,
    hash_object,
)


PREDICTION_HASH = "a" * 64
FROZEN_AT = datetime(2027, 2, 3, 4, 5, tzinfo=timezone.utc)


def _fold() -> OuterFold:
    return OuterFold(
        fold_id="outer-patient-c",
        outer_train_ids=("p2-b", "p1-a", "p2-a", "p1-b"),
        outer_test_ids=("p3-b", "p3-a"),
        inner_folds=(
            InnerFold(
                fold_id="inner-patient-a",
                train_ids=("p2-a", "p2-b"),
                validation_ids=("p1-a", "p1-b"),
            ),
            InnerFold(
                fold_id="inner-patient-b",
                train_ids=("p1-a", "p1-b"),
                validation_ids=("p2-a", "p2-b"),
            ),
        ),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="inner-balanced-accuracy-v1",
        objective_metric="balanced_accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def _candidates() -> tuple[SelectionCandidate, ...]:
    return (
        SelectionCandidate(
            candidate_id="route-pca",
            configuration={
                "epoch": 12,
                "target_route": "pca",
                "threshold": 0.5,
                "calibration": "none",
            },
        ),
        SelectionCandidate(
            candidate_id="route-rgb",
            configuration={
                "epoch": 8,
                "target_route": "rgb",
                "threshold": 0.5,
                "calibration": "temperature",
            },
        ),
    )


def _decision(controller: SelectionController | None = None):
    return (controller or SelectionController()).freeze_selection(
        _fold(),
        candidates=_candidates(),
        candidate_scores={"route-rgb": 0.8, "route-pca": 0.8},
        source_sample_ids=("p2-b", "p1-a", "p2-a", "p1-b"),
        source_prediction_hash=PREDICTION_HASH,
        selection_policy=_policy(),
        decision_id="decision-patient-c",
        frozen_utc=FROZEN_AT,
    )


def test_freeze_saves_complete_inner_evidence_and_uses_deterministic_tie_break() -> None:
    """Requirements 5.7-5.13 and 10.1-10.11: all frozen inputs are retained."""

    decision = _decision()

    assert tuple(candidate.candidate_id for candidate in decision.candidates) == (
        "route-pca",
        "route-rgb",
    )
    assert decision.candidate_scores == {"route-pca": 0.8, "route-rgb": 0.8}
    assert decision.source_sample_ids == ("p1-a", "p1-b", "p2-a", "p2-b")
    assert decision.metric_direction is MetricDirection.MAXIMIZE
    assert decision.tie_break_rule == "candidate_id_utf8_ascending"
    assert decision.source_prediction_hash == PREDICTION_HASH
    assert decision.selected_candidate == "route-pca"
    assert decision.frozen_utc == FROZEN_AT
    assert decision.decision_hash == hash_object(decision.hash_payload())
    assert decision.candidates[0].configuration["epoch"] == 12

    with pytest.raises(ValidationError):
        decision.selected_candidate = "route-rgb"  # type: ignore[misc]


def test_outer_or_non_inner_source_ids_are_rejected_before_decision_creation(
    tmp_path: Path,
) -> None:
    """Requirements 5.15-5.18, 8.4-8.8, 9.5-9.8: evidence stays inner-only."""

    sentinel = tmp_path / "completed.predictions"
    sentinel.write_bytes(b"unchanged")
    journal = ErrorJournal(tmp_path / "errors.jsonl")
    controller = SelectionController(error_journal=journal)

    with pytest.raises(SelectionValidationError) as captured:
        controller.freeze_selection(
            _fold(),
            candidates=_candidates(),
            candidate_scores={"route-pca": 0.7, "route-rgb": 0.8},
            source_sample_ids=("p1-a", "p3-a"),
            source_prediction_hash=PREDICTION_HASH,
            selection_policy=_policy(),
            decision_id="rejected-decision",
        )

    codes = {issue.code for issue in captured.value.report.errors}
    assert "SELECTION_OUTER_EVIDENCE_FORBIDDEN" in codes
    assert "SELECTION_SOURCE_OUTSIDE_OUTER_TRAIN" in codes
    forbidden = next(
        record
        for record in captured.value.error_records
        if record.error_code == "SELECTION_OUTER_EVIDENCE_FORBIDDEN"
    )
    assert forbidden.affected_identifiers == {
        "selection_decision_id": "rejected-decision",
        "fold_id": "outer-patient-c",
    }
    assert forbidden.unexpected_ids == ("p3-a",)
    assert forbidden.invariant_checks["checkpoint_unchanged"] is True
    assert forbidden.invariant_checks["prediction_artifact_unchanged"] is True
    assert sentinel.read_bytes() == b"unchanged"
    assert journal.path.read_text(encoding="utf-8").count("\n") >= 1


def test_outer_train_sample_that_is_not_inner_validation_evidence_is_rejected() -> None:
    controller = SelectionController()

    with pytest.raises(SelectionValidationError) as captured:
        controller.freeze_selection(
            OuterFold(
                fold_id="outer-one-inner",
                outer_train_ids=("train-a", "train-b", "validation-a"),
                outer_test_ids=("test-a",),
                inner_folds=(
                    InnerFold(
                        fold_id="inner-1",
                        train_ids=("train-a", "train-b"),
                        validation_ids=("validation-a",),
                    ),
                ),
            ),
            candidates=_candidates(),
            candidate_scores={"route-pca": 0.7, "route-rgb": 0.8},
            source_sample_ids=("train-a",),
            source_prediction_hash=PREDICTION_HASH,
            selection_policy=_policy(),
        )

    assert {
        issue.code for issue in captured.value.report.errors
    } == {"SELECTION_SOURCE_NOT_INNER_EVIDENCE"}


def test_refit_receives_unchanged_decision_and_complete_outer_train_set() -> None:
    """Requirement 5.14: refit applies the frozen winner to all outer-train IDs."""

    controller = SelectionController()
    decision = _decision(controller)
    calls = []

    def refit(request):
        calls.append(request)
        assert request.outer_train_ids == _fold().outer_train_ids
        assert request.selection_decision is decision
        assert request.selection_decision.decision_hash == decision.decision_hash
        assert request.selected_candidate.candidate_id == "route-pca"
        assert request.selected_candidate.configuration["target_route"] == "pca"
        return "checkpoint-ref"

    result = controller.apply_refit(_fold(), decision, refit)

    assert result == "checkpoint-ref"
    assert len(calls) == 1


def test_refit_rejects_nested_decision_mutation_before_callback() -> None:
    controller = SelectionController()
    decision = _decision(controller)
    decision.candidate_scores["route-pca"] = 0.1
    calls = []

    with pytest.raises(SelectionValidationError) as captured:
        controller.apply_refit(_fold(), decision, calls.append)

    assert captured.value.error_record.error_code == "SELECTION_DECISION_MUTATED"
    assert captured.value.error_record.failed_stage == "outer_refit_validation"
    assert calls == []


def test_candidate_scores_must_cover_the_complete_frozen_candidate_set() -> None:
    with pytest.raises(SelectionValidationError) as captured:
        SelectionController().freeze_selection(
            _fold(),
            candidates=_candidates(),
            candidate_scores={"route-pca": 0.7},
            source_sample_ids=("p1-a", "p1-b", "p2-a", "p2-b"),
            source_prediction_hash=PREDICTION_HASH,
            selection_policy=_policy(),
        )

    assert captured.value.error_record.error_code == "SELECTION_DECISION_INVALID"
