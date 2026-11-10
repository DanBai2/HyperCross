"""Focused non-smoke tests for task-24.2 nested selection boundaries."""
from __future__ import annotations

from dataclasses import replace

import pytest

from benchmark.runners.hyperleaf_nested_remediation import (
    MAX_SELECTION_EPOCH,
    OuterContext,
    REQUIRED_CANDIDATE_IDS,
    build_remediation_candidates,
    select_outer_decision,
)
from benchmark.runners.hyperleaf_screening import ScreeningFold


def _context() -> OuterContext:
    return OuterContext(
        fold=ScreeningFold(
            fold_id="outer-held",
            validation_variety="Held",
            train_ids=("a", "b", "c"),
            validation_ids=("held",),
        ),
        inner_folds=(
            ScreeningFold("outer-held--inner-a", "A", ("b", "c"), ("a",)),
            ScreeningFold("outer-held--inner-b", "B", ("a", "c"), ("b",)),
            ScreeningFold("outer-held--inner-c", "C", ("a", "b"), ("c",)),
        ),
    )


def _evidence(candidates, *, tie: bool = False):
    result = {}
    for index, candidate in enumerate(candidates):
        result[candidate.candidate_id] = {
            "source_sample_ids": ("a", "b", "c"),
            "score": 0.25 if tie else 0.10 + index / 100.0,
            "selected_epoch": min(MAX_SELECTION_EPOCH, index + 1),
            "source_prediction_hashes": {"7": "1" * 64, "42": "2" * 64, "123": "3" * 64},
        }
    return result


def test_candidate_set_is_exact_frozen_winner_and_matched_controls() -> None:
    candidates = build_remediation_candidates()

    assert tuple(candidate.candidate_id for candidate in candidates) == REQUIRED_CANDIDATE_IDS
    assert candidates[1].inference_branch == "target-routed"
    assert candidates[1].target_routes == ("rgb", "rgb-pca-average", "pca")
    assert candidates[1].consistency_weight == 0.10
    assert candidates[-1].consistency_weight == 0.05


def test_outer_decision_uses_exact_outer_train_inner_oof_and_freezes_epoch_route() -> None:
    candidates = build_remediation_candidates()
    decision = select_outer_decision(
        context=_context(), candidates=candidates,
        candidate_evidence=_evidence(candidates), execution_id="task-24-2-test",
    )

    assert set(decision["source_sample_ids"]) == {"a", "b", "c"}
    assert "held" not in decision["source_sample_ids"]
    assert decision["outer_test_overlap_count"] == 0
    assert decision["selected_candidate"] == candidates[-1].candidate_id
    assert decision["selected_epoch"] == len(candidates)
    assert decision["selected_route"] == candidates[-1].inference_branch
    assert decision["legacy_task_14_12_scores_used"] is False
    assert decision["legacy_task_14_13_metrics_used"] is False
    assert decision["outer_metrics_available_at_freeze"] is False
    assert len(decision["decision_hash"]) == 64


def test_outer_decision_tie_breaks_by_candidate_id_bytes() -> None:
    candidates = build_remediation_candidates()
    decision = select_outer_decision(
        context=_context(), candidates=candidates,
        candidate_evidence=_evidence(candidates, tie=True), execution_id="task-24-2-test",
    )

    assert decision["selected_candidate"] == min(REQUIRED_CANDIDATE_IDS, key=lambda value: value.encode("utf-8"))


def test_outer_decision_rejects_held_out_contamination() -> None:
    candidates = build_remediation_candidates()
    evidence = _evidence(candidates)
    contaminated = dict(evidence[candidates[0].candidate_id])
    contaminated["source_sample_ids"] = ("a", "b", "c", "held")
    evidence[candidates[0].candidate_id] = contaminated

    with pytest.raises(RuntimeError, match="source boundary mismatch"):
        select_outer_decision(
            context=_context(), candidates=candidates,
            candidate_evidence=evidence, execution_id="task-24-2-test",
        )


def test_candidate_declaration_drift_is_rejected() -> None:
    candidates = list(build_remediation_candidates())
    candidates[1] = replace(candidates[1], consistency_weight=0.20)

    with pytest.raises(RuntimeError, match="declaration drift"):
        build_remediation_candidates(candidates)
