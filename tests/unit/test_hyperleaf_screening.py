"""Focused tests for the task-14.12 screening execution glue."""
from __future__ import annotations

import numpy as np

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS, HYPERLEAF_VARIETIES
from benchmark.runners.hyperleaf_candidates import GSW_CANDIDATE_CONFIGS
from benchmark.runners.hyperleaf_screening import (
    RUNG_EPOCHS,
    RUNG_SURVIVORS,
    build_inner_lovo_folds,
    build_screening_candidates,
    compute_grouped_metrics,
)


def test_screening_space_contains_all_frozen_branches_and_gsw_candidates() -> None:
    candidates = build_screening_candidates()

    assert len(candidates) == 4 + len(GSW_CANDIDATE_CONFIGS)
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert {candidate.family for candidate in candidates} == {
        "grouped-branch-control",
        "gsw-specialist",
    }
    assert {candidate.style_key for candidate in candidates} >= {"none", "mild"}
    assert {candidate.loss_kind for candidate in candidates} == {"mse", "huber"}
    assert {candidate.target_transform for candidate in candidates} == {
        "standardized",
        "asinh-scale-0.05",
    }
    assert RUNG_EPOCHS == (20, 40, 80)
    assert RUNG_SURVIVORS == (8, 4, 1)


def test_inner_lovo_folds_cover_each_training_sample_exactly_once() -> None:
    sample_ids = tuple(f"sample-{index}" for index in range(12))
    varieties = tuple(HYPERLEAF_VARIETIES[index % 4] for index in range(12))

    folds = build_inner_lovo_folds(sample_ids, varieties)

    validation_ids = [sample_id for fold in folds for sample_id in fold.validation_ids]
    assert len(folds) == 4
    assert set(validation_ids) == set(sample_ids)
    assert len(validation_ids) == len(set(validation_ids))
    for fold in folds:
        assert set(fold.train_ids).isdisjoint(fold.validation_ids)
        assert set(fold.train_ids) | set(fold.validation_ids) == set(sample_ids)


def test_grouped_metrics_retain_gsw_and_regression_guard_targets() -> None:
    sample_ids = tuple(f"sample-{index}" for index in range(8))
    fold_ids = tuple(f"fold-{index // 2}" for index in range(8))
    truth = np.arange(24, dtype=np.float64).reshape(8, 3)

    metrics = compute_grouped_metrics(sample_ids, fold_ids, truth, truth.copy())

    assert metrics["primary_objective"] == "pooled_oof.r2.Gsw"
    for target in HYPERLEAF_TARGETS:
        assert metrics["pooled_oof"]["targets"][target]["r2"] == 1.0
        assert metrics["pooled_oof"]["targets"][target]["mae"] == 0.0
        assert target in metrics["fold_macro"]["targets"]
