"""Focused unit coverage for task-14.13 strict execution glue.

Validates Requirements 8.14-8.23, 10.12-10.31, 11.1-11.23, and 13.6-13.32.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS
from benchmark.runners.hyperleaf_screening import build_screening_candidates
from benchmark.runners.hyperleaf_strict import (
    REQUIRED_GSW_CONTROL,
    SELECTED_EPOCH,
    STRICT_SEEDS,
    WINNER_ID,
    aggregate_candidate_metrics,
    resolve_strict_candidates,
    _resume_plan_matches,
)
from benchmark.schema import hash_object


def _decision() -> dict:
    candidates = build_screening_candidates()
    payload = {
        "schema_version": "1.0",
        "selected_candidate": WINNER_ID,
        "candidate_set": [
            {**candidate.__dict__, "config_hash": candidate.config_hash}
            for candidate in candidates
        ],
        "source_sample_ids": ["sample-a", "sample-b"],
        "source_prediction_hash": "a" * 64,
        "official_test_labels_accessible": False,
        "official_evaluation_budget_consumed": 0,
        "frozen_configuration": {
            "candidate_id": WINNER_ID,
            "selected_epoch": SELECTED_EPOCH,
            "matched_controls": [
                "branch::independent-target",
                WINNER_ID,
                "branch::style-augmentation",
                "branch::train-fold-pca",
            ],
        },
    }
    payload["decision_hash"] = hash_object(payload)
    return payload


def _metrics(gsw_r2: float) -> dict:
    return {
        "pooled_oof": {
            "targets": {
                target: {
                    "r2": gsw_r2 + index / 100.0,
                    "mae": 1.0 + index,
                }
                for index, target in enumerate(HYPERLEAF_TARGETS)
            }
        }
    }


def test_resolve_strict_candidates_uses_frozen_winner_and_all_required_controls() -> None:
    resolved = resolve_strict_candidates(_decision())
    ids = {candidate.candidate_id for candidate in resolved}

    assert ids == {
        WINNER_ID,
        REQUIRED_GSW_CONTROL,
        "branch::independent-target",
        "branch::style-augmentation",
        "branch::train-fold-pca",
    }
    assert next(candidate for candidate in resolved if candidate.candidate_id == WINNER_ID).inference_branch == "target-routed"
    assert next(candidate for candidate in resolved if candidate.candidate_id == REQUIRED_GSW_CONTROL).consistency_weight == 0.05


def test_resolve_strict_candidates_rejects_any_frozen_config_hash_drift() -> None:
    decision = _decision()
    decision["candidate_set"][0]["config_hash"] = "0" * 64
    decision.pop("decision_hash")
    decision["decision_hash"] = hash_object(decision)

    with pytest.raises(RuntimeError, match="config hash mismatch"):
        resolve_strict_candidates(decision)


def test_resolve_strict_candidates_rejects_selection_artifact_mutation() -> None:
    decision = _decision()
    mutated = deepcopy(decision)
    mutated["frozen_configuration"]["selected_epoch"] = SELECTED_EPOCH + 1

    with pytest.raises(RuntimeError, match="decision hash mismatch"):
        resolve_strict_candidates(mutated)


def test_seed_aggregation_reports_exact_seed_variance_and_stability() -> None:
    aggregate = aggregate_candidate_metrics(
        {7: _metrics(0.10), 42: _metrics(0.20), 123: _metrics(0.30)}
    )

    assert aggregate["seed_set"] == list(STRICT_SEEDS)
    gsw = aggregate["targets"]["Gsw"]
    assert gsw["pooled_oof_r2_seed_mean"] == pytest.approx(0.21)
    assert gsw["pooled_oof_r2_seed_variance"] == pytest.approx(0.01)
    assert aggregate["stability"]["seed_variance"] == pytest.approx(0.01)
    assert aggregate["stability"]["unstable"] is True


def test_seed_aggregation_rejects_incomplete_seed_set() -> None:
    with pytest.raises(RuntimeError, match="exact seeds"):
        aggregate_candidate_metrics({7: _metrics(0.1), 42: _metrics(0.2)})


def test_resume_plan_comparison_normalizes_json_sequence_types() -> None:
    frozen = {"candidate_table": [{"target_routes": ["rgb", "pca"]}], "seed_set": [7, 42, 123]}
    regenerated = {"candidate_table": [{"target_routes": ("rgb", "pca")}], "seed_set": (7, 42, 123)}

    assert _resume_plan_matches(frozen, regenerated)


def test_resume_plan_comparison_rejects_scientific_identity_drift() -> None:
    frozen = {"dataset_hash": "a" * 64, "selected_epoch": 47}
    changed = {"dataset_hash": "b" * 64, "selected_epoch": 47}

    assert not _resume_plan_matches(frozen, changed)
