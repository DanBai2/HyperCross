from __future__ import annotations

from benchmark.runners.histology_candidates import (
    ATTENTION_DISABLED_CONTROL_KEY,
    LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
    LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
    LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
    LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
    MIL_CONCAT_CONTROL_KEY,
    PCA_TASK_ONLY_CONTROL_KEY,
)
from benchmark.runners.histology_screening import (
    PRIMARY_THRESHOLD,
    build_screening_candidates,
    compute_screening_metrics,
)


def _row(sample_id: str, group_id: str, outer: str, fold: str, truth: int, probability: float):
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "outer_fold_id": outer,
        "fold_id": fold,
        "seed": 42,
        "inference_branch": "test",
        "y_true": truth,
        "probability": probability,
        "y_pred_fixed_0_5": int(probability >= PRIMARY_THRESHOLD),
    }


def test_screening_candidate_set_contains_required_modality_attention_and_loss_controls() -> None:
    candidates = build_screening_candidates()
    keys = {candidate.candidate_id for candidate in candidates}
    assert len(candidates) == 8
    assert {
        PCA_TASK_ONLY_CONTROL_KEY,
        "histology.rgb-only.task-only-control.v1",
        MIL_CONCAT_CONTROL_KEY,
        LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
        ATTENTION_DISABLED_CONTROL_KEY,
        LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
    } == keys
    local = next(item for item in candidates if item.candidate_id == LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY)
    assert set(local.required_control_keys) == {
        ATTENTION_DISABLED_CONTROL_KEY,
        MIL_CONCAT_CONTROL_KEY,
        LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
    }


def test_screening_metrics_keep_outer_inner_and_patient_evidence_named() -> None:
    rows = [
        _row("P1/a", "P1", "outer-01", "outer-01.inner-01", 0, 0.1),
        _row("P2/a", "P2", "outer-01", "outer-01.inner-02", 1, 0.9),
        _row("P1/a", "P1", "outer-02", "outer-02.inner-01", 0, 0.2),
        _row("P2/a", "P2", "outer-02", "outer-02.inner-02", 1, 0.8),
    ]
    metrics = compute_screening_metrics(rows)
    assert metrics["primary_objective"] == "outer_fold_macro.inner_oof.balanced_accuracy"
    assert metrics["outer_fold_macro"]["accuracy"]["value"] == 1.0
    assert metrics["outer_fold_macro"]["balanced_accuracy"]["value"] == 1.0
    assert metrics["cross_context_pooled_oof"]["accuracy"]["value"] == 1.0
    assert set(metrics["per_outer_fold_inner_oof"]) == {"outer-01", "outer-02"}
    assert set(metrics["per_patient_cross_context"]) == {"P1", "P2"}
