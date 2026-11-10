"""Focused task-24.4 runner tests; these do not execute training or smoke suites."""
from __future__ import annotations

from pathlib import Path

from benchmark.runners.hyperleaf_inner_optimization import (
    _deltas,
    _load_permitted_targets,
    _promotion_evidence,
    _screen_decision,
)
from benchmark.runners.hyperleaf_nested_remediation import build_nested_contexts
from benchmark.runners.hyperleaf_optimization_registry import (
    CandidateRole,
    build_hyperleaf_optimization_plan,
)
from benchmark.runners.hyperleaf_screening import _load_train_manifest
from benchmark.protocols.hyperleaf_lovo import plan_hyperleaf_lovo
from benchmark.protocols.planner import ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore


def _result(candidate_id: str, gsw: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_prediction_hashes": {str(seed): f"{index}" * 64 for index, seed in enumerate((7, 42, 123), start=1)},
        "per_seed_inner_metrics": {
            str(seed): {
                "pooled_oof": {
                    "targets": {
                        "GrainWeight": {"r2": 0.50, "mae": 1.0},
                        "Gsw": {"r2": gsw + index * 0.001, "mae": 0.01},
                        "PhiPS2": {"r2": 0.40, "mae": 0.01},
                    }
                }
            }
            for index, seed in enumerate((7, 42, 123))
        },
    }


def _first_context():
    plan = build_hyperleaf_optimization_plan()
    manifest = _load_train_manifest()
    protocol = plan_hyperleaf_lovo(
        ProtocolPlanner(FailureAtomicArtifactStore(Path(".tmp") / "pytest-task-24-4")),
        manifest,
        seed_set=(7, 42, 123),
        publish=False,
    )
    return plan, build_nested_contexts(protocol, manifest)[0]


def test_permitted_target_reader_materializes_only_requested_outer_train_ids() -> None:
    ids, targets, audit = _load_permitted_targets(("00000", "00001"))
    assert ids == ("00000", "00001")
    assert targets.shape == (2, 3)
    assert audit["materialized_count"] == 2
    assert audit["held_out_target_fields_materialized"] == 0
    assert audit["official_target_fields_materialized"] == 0


def test_paired_screen_advances_at_most_three_optimization_candidates_and_guards() -> None:
    plan, context = _first_context()
    results = {}
    optimization_index = 0
    for candidate in plan.candidates:
        if candidate.role is CandidateRole.OPTIMIZATION:
            optimization_index += 1
            # Every candidate beats its guard; the deterministic ranking must
            # still cap newly advanced families at three.
            results[candidate.candidate_id] = _result(candidate.candidate_id, 0.20 + optimization_index * 0.01)
        else:
            results[candidate.candidate_id] = _result(candidate.candidate_id, 0.10)
    # Candidates guarded by PCA-routed need to beat that candidate, not controls.
    pca_routed = next(
        item for item in plan.candidates
        if item.role is CandidateRole.OPTIMIZATION and item.matched_guard_id is not None
        and item.matched_guard_id.startswith("control::")
    )
    results[pca_routed.candidate_id] = _result(pca_routed.candidate_id, 0.20)
    for index, candidate in enumerate(
        item for item in plan.candidates
        if item.role is CandidateRole.OPTIMIZATION and item.candidate_id != pca_routed.candidate_id
    ):
        results[candidate.candidate_id] = _result(candidate.candidate_id, 0.22 + index * 0.01)

    decision = _screen_decision(
        plan=plan,
        context=context,
        screen_results=results,
        execution_id="task-24-4-test",
    )
    assert len(decision["complete_candidate_set"]) == 10
    assert len(decision["advanced_optimization_candidates"]) == 3
    assert set(decision["advanced_optimization_candidates"]).issubset(
        set(decision["advanced_candidates_and_guards"])
    )
    for candidate_id in decision["advanced_optimization_candidates"]:
        guard = next(item.matched_guard_id for item in plan.candidates if item.candidate_id == candidate_id)
        assert guard in decision["advanced_candidates_and_guards"]
    assert decision["held_out_overlap_count"] == 0
    assert decision["task_14_13_metrics_used"] is False
    assert decision["task_24_2_outer_metrics_used"] is False


def test_nested_inner_delta_uses_aligned_seed_metrics_only() -> None:
    plan = build_hyperleaf_optimization_plan()
    candidate = next(item for item in plan.candidates if item.role is CandidateRole.OPTIMIZATION)
    guard_id = candidate.matched_guard_id
    candidate_evidence = _promotion_evidence(plan, candidate.candidate_id, _result(candidate.candidate_id, 0.14))
    guard_evidence = _promotion_evidence(plan, guard_id, _result(guard_id, 0.10))
    delta = _deltas(candidate_evidence, guard_evidence)
    assert all(abs(delta["per_seed_delta_r2"][str(seed)]["Gsw"] - 0.04) < 1.0e-12 for seed in (7, 42, 123))
    assert abs(delta["mean_delta_r2"]["Gsw"] - 0.04) < 1.0e-12
    assert delta["mean_delta_r2"]["GrainWeight"] == 0.0
    assert delta["mean_delta_r2"]["PhiPS2"] == 0.0
