"""Focused non-smoke validation for immutable task-24.3 declarations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmark.models.hyperleaf_optimization import (
    HETEROSCEDASTIC_MODEL_KEY,
    LEARNED_GATED_BRANCH,
    LEARNED_GATED_MODEL_KEY,
    HyperLeafHeteroscedasticRgbPca,
    HyperLeafLearnedGatedFusion,
    create_hyperleaf_optimization_model_registry,
)
from benchmark.runners.hyperleaf_optimization_registry import (
    CANDIDATE_GATED,
    CANDIDATE_HETEROSCEDASTIC,
    CANDIDATE_PCA_ROUTED,
    CANDIDATE_PHI_CONSISTENCY,
    CANDIDATE_ROBUST_NORMALIZATION,
    CANDIDATE_TARGET_BALANCED,
    CONTROL_CONSISTENCY_005,
    CONTROL_PCA,
    CONTROL_RGB,
    FALLBACK_EXISTING,
    HyperLeafOptimizationError,
    InnerOofPromotionEvidence,
    SeedTargetR2,
    build_hyperleaf_optimization_plan,
    evaluate_promotion,
)
from benchmark.transforms.hyperleaf_optimization import (
    HETEROSCEDASTIC_LOSS,
    TARGET_BALANCED_LOSS,
    VARIETY_ROBUST_SPECTRAL_NORMALIZATION,
    fit_target_balance_weights,
    fit_variety_robust_spectral_normalization,
    heteroscedastic_regression_loss,
    target_balanced_regression_loss,
)


def _evidence(plan, candidate_id: str, rows: tuple[tuple[float, float, float], ...]):
    return InnerOofPromotionEvidence.create(
        candidate_id=candidate_id,
        comparison_context_hash=plan.context.comparison_context_hash,
        evidence_scope="nested_inner_oof_only",
        seed_metrics=tuple(
            SeedTargetR2(seed=seed, grainweight=row[0], gsw=row[1], phips2=row[2])
            for seed, row in zip((7, 42, 123), rows)
        ),
        source_prediction_hashes=("1" * 64, "2" * 64, "3" * 64),
        outer_metrics_used=False,
        official_labels_used=False,
    )


def test_registry_freezes_disciplined_candidate_controls_and_common_identity() -> None:
    plan = build_hyperleaf_optimization_plan()
    by_id = {item.candidate_id: item for item in plan.candidates}

    assert set(by_id) == {
        CONTROL_RGB,
        CONTROL_PCA,
        CONTROL_CONSISTENCY_005,
        FALLBACK_EXISTING,
        CANDIDATE_PCA_ROUTED,
        CANDIDATE_GATED,
        CANDIDATE_PHI_CONSISTENCY,
        CANDIDATE_ROBUST_NORMALIZATION,
        CANDIDATE_HETEROSCEDASTIC,
        CANDIDATE_TARGET_BALANCED,
    }
    assert plan.fallback_candidate_id == FALLBACK_EXISTING
    assert plan.context.seed_set == (7, 42, 123)
    assert plan.context.fold_ids == (
        "outer-heerup",
        "outer-kvium",
        "outer-rembrandt",
        "outer-sheriff",
    )
    assert plan.context.dataset_hash == "66b78e98dca72c272f5dd764e2c72b2543e4be4c7f47a49bf97a01fa8714efd1"
    assert plan.context.protocol_hash == "b22c46f6b44d2824f672b73aab331799642d776e5b7ce296614db2270586bcdf"
    assert plan.context.precision_profile_config_hash == "6a546010f4ba2793e5dc6150376b4be482fed918f5e5fd6bda19d3f207043fe9"
    assert len({item.comparison_context_hash for item in plan.candidates}) == 1
    assert all(not value.startswith(("/", "G:", "F:")) for value in (
        plan.context.output_root,
        plan.context.recovery_root,
        plan.context.artifact_root,
    ))
    assert plan.design_basis["selection_score_allowed"] is False
    assert plan.promotion_guard.gsw_mean_delta_r2_minimum == 0.0100
    assert tuple(stage.candidate_epoch_cap for stage in plan.screening_schedule) == (20, 80)
    assert all(stage.seed_set == (7, 42, 123) for stage in plan.screening_schedule)
    plan.candidate_space.require_integrity()

    frozen = json.loads(
        Path("configs/optimization/v1/hyperleaf-task-24.3-preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["registry_hashes"]["plan_hash"] == plan.plan_hash
    assert frozen["registry_hashes"]["candidate_space_hash"] == plan.candidate_space.candidate_space_hash
    assert frozen["candidate_configuration_hashes"] == {
        item.candidate_id: item.configuration_hash for item in plan.candidates
    }

    optimization = [item for item in plan.candidates if item.role.value == "optimization_candidate"]
    assert all(item.target_routes[0] == "pca" for item in optimization)
    assert by_id[CANDIDATE_GATED].target_routes[1] == LEARNED_GATED_BRANCH
    assert by_id[CANDIDATE_ROBUST_NORMALIZATION].spectral_normalization_key == VARIETY_ROBUST_SPECTRAL_NORMALIZATION
    assert by_id[CANDIDATE_HETEROSCEDASTIC].loss_key == HETEROSCEDASTIC_LOSS
    assert by_id[CANDIDATE_TARGET_BALANCED].loss_key == TARGET_BALANCED_LOSS
    assert all(item.matched_guard_id in by_id for item in optimization)


def test_model_registry_builds_gated_and_heteroscedastic_candidates() -> None:
    registry = create_hyperleaf_optimization_model_registry()
    gated = registry.build(LEARNED_GATED_MODEL_KEY)
    heteroscedastic = registry.build(HETEROSCEDASTIC_MODEL_KEY)
    assert isinstance(gated, HyperLeafLearnedGatedFusion)
    assert isinstance(heteroscedastic, HyperLeafHeteroscedasticRgbPca)

    rgb = torch.rand(2, 3, 32, 64)
    pca = torch.rand(2, 3, 32, 64)
    spectrum = torch.rand(2, 128)
    with torch.no_grad():
        prediction, branches = gated(rgb, pca, spectrum, return_branches=True)
        mean, log_variance = heteroscedastic(rgb, pca, spectrum)
    assert prediction.shape == mean.shape == log_variance.shape == (2, 3)
    assert LEARNED_GATED_BRANCH in branches
    lower = torch.minimum(branches["rgb"], branches["pca"])
    upper = torch.maximum(branches["rgb"], branches["pca"])
    assert torch.all(branches[LEARNED_GATED_BRANCH] >= lower)
    assert torch.all(branches[LEARNED_GATED_BRANCH] <= upper)
    assert torch.all((-8.0 <= log_variance) & (log_variance <= 8.0))


def test_fold_scoped_robust_normalization_and_new_losses_are_implementable() -> None:
    spectra = torch.tensor(
        [[1.0, 10.0], [3.0, 12.0], [100.0, 20.0], [102.0, 22.0]],
        dtype=torch.float32,
    )
    state = fit_variety_robust_spectral_normalization(
        spectra,
        sample_ids=("a", "b", "c", "d"),
        variety_ids=("v1", "v1", "v2", "v2"),
        allowed_sample_ids=("a", "b", "c", "d", "unused"),
        dataset_hash="a" * 64,
        protocol_hash="b" * 64,
        fold_id="inner-v3",
    )
    transformed = state.transform(spectra)
    assert transformed.shape == spectra.shape
    assert torch.isfinite(transformed).all()
    assert state.fit_sample_ids == ("a", "b", "c", "d")
    assert len(state.config_hash) == len(state.state_hash) == 64

    with pytest.raises(ValueError, match="permitted training boundary"):
        fit_variety_robust_spectral_normalization(
            spectra,
            sample_ids=("a", "b", "c", "held-out"),
            variety_ids=("v1", "v1", "v2", "held"),
            allowed_sample_ids=("a", "b", "c"),
            dataset_hash="a" * 64,
            protocol_hash="b" * 64,
            fold_id="inner-v3",
        )

    target = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.5, 8.0]])
    prediction = target + 0.25
    weights = fit_target_balance_weights(target)
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert target_balanced_regression_loss(
        prediction, target, target_weights=weights
    ).ndim == 0
    assert heteroscedastic_regression_loss(
        prediction, torch.zeros_like(prediction), target
    ).ndim == 0


def test_promotion_requires_exact_mean_seed_direction_and_target_guards() -> None:
    plan = build_hyperleaf_optimization_plan()
    guard = _evidence(
        plan,
        CONTROL_CONSISTENCY_005,
        ((0.50, 0.10, 0.40), (0.51, 0.11, 0.41), (0.52, 0.12, 0.42)),
    )
    passing = _evidence(
        plan,
        CANDIDATE_PCA_ROUTED,
        ((0.491, 0.111, 0.391), (0.501, 0.122, 0.401), (0.511, 0.133, 0.411)),
    )
    decision = evaluate_promotion(plan, passing, guard)
    assert decision.passed
    assert decision.mean_delta_r2["gsw"] >= 0.0100
    assert all(delta > 0 for _, delta in decision.gsw_seed_deltas)
    assert decision.mean_delta_r2["grainweight"] >= -0.0100
    assert decision.mean_delta_r2["phips2"] >= -0.0100

    wrong_direction = _evidence(
        plan,
        CANDIDATE_PCA_ROUTED,
        ((0.50, 0.099, 0.40), (0.51, 0.131, 0.41), (0.52, 0.151, 0.42)),
    )
    rejected = evaluate_promotion(plan, wrong_direction, guard)
    assert not rejected.passed
    assert "gsw_seed_direction_not_positive_for_all_7_42_123" in rejected.failed_guards
    assert rejected.disposition == "retain_as_negative_evidence_no_outer_revival"

    target_regression = _evidence(
        plan,
        CANDIDATE_PCA_ROUTED,
        ((0.489, 0.12, 0.389), (0.499, 0.13, 0.399), (0.509, 0.14, 0.409)),
    )
    rejected = evaluate_promotion(plan, target_regression, guard)
    assert not rejected.passed
    assert "grainweight_mean_pooled_oof_regression_exceeds_0.0100" in rejected.failed_guards
    assert "phips2_mean_pooled_oof_regression_exceeds_0.0100" in rejected.failed_guards


def test_promotion_rejects_outer_or_identity_mismatched_evidence() -> None:
    plan = build_hyperleaf_optimization_plan()
    with pytest.raises(ValueError, match="nested inner OOF only"):
        InnerOofPromotionEvidence.create(
            candidate_id=CANDIDATE_PCA_ROUTED,
            comparison_context_hash=plan.context.comparison_context_hash,
            evidence_scope="outer_oof",
            seed_metrics=tuple(
                SeedTargetR2(seed=seed, grainweight=0.5, gsw=0.2, phips2=0.4)
                for seed in (7, 42, 123)
            ),
            source_prediction_hashes=("1" * 64, "2" * 64, "3" * 64),
            outer_metrics_used=True,
            official_labels_used=False,
        )

    candidate = _evidence(
        plan,
        CANDIDATE_GATED,
        ((0.5, 0.2, 0.4), (0.5, 0.2, 0.4), (0.5, 0.2, 0.4)),
    )
    wrong_guard = _evidence(
        plan,
        CONTROL_CONSISTENCY_005,
        ((0.5, 0.1, 0.4), (0.5, 0.1, 0.4), (0.5, 0.1, 0.4)),
    )
    with pytest.raises(HyperLeafOptimizationError, match="declared matched guard"):
        evaluate_promotion(plan, candidate, wrong_guard)
