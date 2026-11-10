"""Unit coverage for task 17.6 HyperLeaf review configuration and controls."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.protocols.hyperleaf_lovo import HYPERLEAF_LOVO_PROTOCOL_KEY
from benchmark.runners import (
    HYPERLEAF_REVIEW_SPECS,
    HYPERLEAF_RGB_PCA_MODALITIES,
    HYPERLEAF_RGB_PCA_REPRESENTATION_KEY,
    HyperLeafAugmentation,
    HyperLeafCandidateReviewRunner,
    HyperLeafModelBranch,
    HyperLeafReviewError,
    HyperLeafReviewRole,
    build_hyperleaf_review_plan,
    validate_hyperleaf_matched_control,
)
from benchmark.schema import (
    EnvironmentLock,
    EvaluationBudget,
    InnerFold,
    LimitEnforcement,
    MetricDirection,
    OuterFold,
    ProtocolManifest,
    ResourceBudget,
    ResourceField,
    ResourceLimit,
    ResourceUnit,
    SelectionPolicy,
)


FROZEN_AT = datetime(2027, 5, 1, tzinfo=timezone.utc)


def _budget() -> ResourceBudget:
    return ResourceBudget(
        limits=(
            ResourceLimit(
                field=ResourceField.COMPLETED_EPOCHS,
                value=30,
                unit=ResourceUnit.COUNT,
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )


def _protocol() -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1.0",
        protocol_key=HYPERLEAF_LOVO_PROTOCOL_KEY,
        dataset_hash="a" * 64,
        folds=(
            OuterFold(
                fold_id="outer-sheriff",
                outer_train_ids=("heerup-1", "kvium-1", "rembrandt-1"),
                outer_test_ids=("sheriff-1",),
                inner_folds=(
                    InnerFold(
                        fold_id="outer-sheriff.inner-1",
                        train_ids=("kvium-1", "rembrandt-1"),
                        validation_ids=("heerup-1",),
                    ),
                    InnerFold(
                        fold_id="outer-sheriff.inner-2",
                        train_ids=("heerup-1", "rembrandt-1"),
                        validation_ids=("kvium-1",),
                    ),
                    InnerFold(
                        fold_id="outer-sheriff.inner-3",
                        train_ids=("heerup-1", "kvium-1"),
                        validation_ids=("rembrandt-1",),
                    ),
                ),
            ),
        ),
        selection_policy=_policy(),
        seed_set=(123, 7, 42),
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=1),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="hyperleaf.review.inner-variety.v1",
        objective_metric="mean-r2",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def _plan():
    return build_hyperleaf_review_plan(
        _protocol(),
        metric_configuration={"targets": ("GrainWeight", "Gsw", "PhiPS2"), "r2": True},
        resource_budget=_budget(),
        frozen_utc=FROZEN_AT,
    )


def test_plan_encodes_every_named_model_augmentation_and_negative_ablation() -> None:
    plan = _plan()
    specs = {binding.spec.review_key: binding.spec for binding in plan.bindings}

    assert {spec.model_branch for spec in specs.values()} == set(HyperLeafModelBranch)
    assert {spec.augmentation for spec in specs.values()} == set(HyperLeafAugmentation)
    assert {
        HyperLeafModelBranch.TARGET_ROUTED_RGB_PCA_CONSISTENCY,
        HyperLeafModelBranch.INDEPENDENT_EXPERTS,
        HyperLeafModelBranch.SHARED_HEADS,
    } <= {spec.model_branch for spec in specs.values()}
    negatives = [
        spec for spec in specs.values()
        if spec.role is HyperLeafReviewRole.NEGATIVE_ABLATION
    ]
    assert negatives
    assert all(spec.evidence_disposition == "append_only_negative" for spec in negatives)
    assert all(
        spec.matched_control_key in specs
        for spec in specs.values()
        if spec.role is not HyperLeafReviewRole.MATCHED_INPUT_CONTROL
    )
    target_routed = specs["candidate.target-routed-rgb-pca-consistency.style"]
    assert tuple(
        (route.target_name, route.inference_branch)
        for route in target_routed.target_routes
    ) == (
        ("GrainWeight", "visible_rgb"),
        ("Gsw", "rgb_pca_consistency"),
        ("PhiPS2", "train_fold_pca"),
    )
    assert target_routed.consistency_weight == 0.1


def test_every_candidate_and_control_has_an_identical_frozen_comparison_context() -> None:
    plan = _plan()
    by_key = {binding.spec.review_key: binding for binding in plan.bindings}

    for candidate in plan.bindings:
        if candidate.spec.role is HyperLeafReviewRole.MATCHED_INPUT_CONTROL:
            continue
        control = by_key[candidate.spec.matched_control_key]
        validate_hyperleaf_matched_control(candidate, control)
        assert candidate.context is plan.context
        assert control.context is plan.context
        assert candidate.representation_key == control.representation_key
        assert candidate.input_modalities == control.input_modalities
        assert candidate.run_config.resource_budget == control.run_config.resource_budget

    assert plan.context.dataset_hash == _protocol().dataset_hash
    assert plan.context.protocol_hash == _protocol().protocol_hash
    assert plan.context.seed_set == (7, 42, 123)
    assert len(plan.context.fold_manifest_hashes) == 1


def test_candidate_space_and_run_configs_are_hash_bound_before_evaluation() -> None:
    plan = _plan()
    resolved = plan.frozen_candidate_registry.require_for_evaluation(
        plan.candidate_space.candidate_space_key,
        expected_hash=plan.candidate_space.candidate_space_hash,
    )

    assert resolved is plan.candidate_space
    assert {option.option_key for option in resolved.augmentation_candidates} == {
        augmentation.value for augmentation in HyperLeafAugmentation
    }
    assert {option.option_key for option in resolved.model_candidates} == {
        branch.value for branch in HyperLeafModelBranch
    }
    assert all(binding.run_config.execution_mode.value == "deterministic" for binding in plan.bindings)
    assert all(binding.run_config.framework_settings.deterministic_algorithms for binding in plan.bindings)
    assert all(binding.run_config.checkpoint_retention.preserve_selected for binding in plan.bindings)
    assert all(binding.run_config.checkpoint_retention.preserve_last for binding in plan.bindings)
    assert all(binding.run_config.checkpoint_retention.preserve_recovery_state for binding in plan.bindings)


def test_route_and_consistency_selection_requires_complete_inner_variety_oof() -> None:
    runner = HyperLeafCandidateReviewRunner(_plan())
    keys = (
        "candidate.target-routed-rgb-pca-consistency.style",
        "candidate.independent-experts.style",
    )
    decision = runner.freeze_route_and_consistency(
        _protocol().folds[0],
        review_keys=keys,
        candidate_scores={keys[0]: 0.41, keys[1]: 0.39},
        inner_source_sample_ids=("rembrandt-1", "heerup-1", "kvium-1"),
        inner_prediction_hash="b" * 64,
        selection_policy=_policy(),
        decision_id="hyperleaf-review-inner-only",
        frozen_utc=FROZEN_AT,
    )

    assert decision.selected_candidate == keys[0]
    assert set(decision.source_sample_ids) == {"heerup-1", "kvium-1", "rembrandt-1"}
    assert decision.candidates[0].configuration["context_hash"] == runner.plan.context.context_hash

    with pytest.raises(HyperLeafReviewError, match="inner-variety OOF"):
        runner.freeze_route_and_consistency(
            _protocol().folds[0],
            review_keys=(keys[0],),
            candidate_scores={keys[0]: 1.0},
            inner_source_sample_ids=("sheriff-1",),
            inner_prediction_hash="c" * 64,
            selection_policy=_policy(),
        )


def test_execution_contract_is_tcfm_rtx5080_single_process_relative_and_recoverable() -> None:
    runner = HyperLeafCandidateReviewRunner(_plan())
    metadata = runner.metadata()
    environment = EnvironmentLock.create(
        python_version="3.10",
        pytorch_version="2.7",
        numpy_version="2.0",
        pandas_version="2.0",
        pillow_version="11",
        tifffile_version="2025",
        sklearn_version="1.6",
        scipy_version="1.15",
        cuda_version="12.8",
        cuda_available=True,
        cuda_device_count=1,
        device_identity="cuda:0:NVIDIA GeForce RTX 5080",
    )

    runner.validate_launch(environment, conda_environment="TCFM", process_count=1)
    assert metadata["conda_environment"] == "TCFM"
    assert metadata["expected_device_name"] == "NVIDIA GeForce RTX 5080"
    assert metadata["process_count"] == 1
    assert metadata["deterministic_execution"] is True
    assert metadata["official_label_access"] is False
    assert not Path(metadata["output_root"]).is_absolute()
    assert metadata["output_root"].startswith("runs_v2/candidates/hyperleaf/")
    assert set(metadata["recovery_state_components"]) == {
        "model", "optimizer", "scheduler", "stage"
    }

    with pytest.raises(HyperLeafReviewError, match="exactly G"):
        HyperLeafCandidateReviewRunner(_plan(), r"F:\Datasets\HyperCross")


def test_review_spec_catalog_is_immutable_and_complete() -> None:
    assert HYPERLEAF_REVIEW_SPECS
    assert all(spec.target_routes for spec in HYPERLEAF_REVIEW_SPECS)
    assert HYPERLEAF_RGB_PCA_MODALITIES == (
        "visible_rgb", "train_fold_pca_pseudo_rgb"
    )
    assert "pca" in HYPERLEAF_RGB_PCA_REPRESENTATION_KEY
