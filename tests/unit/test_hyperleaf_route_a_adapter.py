"""Task 14.5 coverage for HyperLeaf in-repository Route A adapters.

Validates Requirements 1.21, 1.23, 1.24, 6.31, 10.8, 10.9, 10.32, 10.33.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import torch

from benchmark.models.hyperleaf_route_a import (
    GLOBAL_FUSION_MODEL_KEY,
    LOCAL_MIL_MODEL_KEY,
    SHARED_HEADS_MODEL_KEY,
    TARGET_EXPERTS_MODEL_KEY,
    TOKEN_MODEL_KEY,
    HyperLeafGlobalFusionAdapter,
    HyperLeafLocalMILRouteAAdapter,
    HyperLeafSharedHeadsAdapter,
    HyperLeafTargetExpertsAdapter,
    HyperLeafTokenRouteAAdapter,
    create_hyperleaf_route_a_registry,
)
from benchmark.runners.hyperleaf_route_a import HyperLeafRouteAError, HyperLeafRouteARunner
from benchmark.schema import InnerFold, MetricDirection, OuterFold, SelectionPolicy


EXPECTED_CLASSES = {
    GLOBAL_FUSION_MODEL_KEY: HyperLeafGlobalFusionAdapter,
    TARGET_EXPERTS_MODEL_KEY: HyperLeafTargetExpertsAdapter,
    SHARED_HEADS_MODEL_KEY: HyperLeafSharedHeadsAdapter,
    TOKEN_MODEL_KEY: HyperLeafTokenRouteAAdapter,
    LOCAL_MIL_MODEL_KEY: HyperLeafLocalMILRouteAAdapter,
}


def _outer_fold() -> OuterFold:
    return OuterFold(
        fold_id="held-out-variety",
        outer_train_ids=("heerup-1", "kvium-1", "rembrandt-1", "rembrandt-2"),
        outer_test_ids=("sheriff-1",),
        inner_folds=(
            InnerFold(
                fold_id="inner-variety-1",
                train_ids=("heerup-1", "kvium-1"),
                validation_ids=("rembrandt-1", "rembrandt-2"),
            ),
        ),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="inner-variety-mean-r2-v1",
        objective_metric="mean-r2",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def test_stable_model_keys_build_exact_adapter_classes() -> None:
    registry = create_hyperleaf_route_a_registry()

    assert set(registry.model_keys) == set(EXPECTED_CLASSES)
    for model_key, expected_class in EXPECTED_CLASSES.items():
        registration = registry.resolve(model_key)
        model = registry.build(model_key)
        assert type(model) is expected_class
        assert registration.identity.model_key == model_key
        assert registration.identity.class_fully_qualified_name == (
            f"{expected_class.__module__}.{expected_class.__qualname__}"
        )
        assert registration.identity.parameter_count == sum(
            parameter.numel() for parameter in model.parameters()
        )


def test_route_a_adapters_preserve_three_target_output_contracts() -> None:
    rgb = torch.rand(2, 3, 32, 64)
    spectrum = torch.rand(2, 128)
    local_tokens = torch.rand(2, 16, 128)

    with torch.no_grad():
        for adapter in (
            HyperLeafGlobalFusionAdapter(embedding_dim=16),
            HyperLeafTargetExpertsAdapter(embedding_dim=16),
            HyperLeafSharedHeadsAdapter(embedding_dim=16),
            HyperLeafTokenRouteAAdapter(embedding_dim=16),
        ):
            assert adapter(rgb, spectrum).shape == (2, 3)
        assert HyperLeafLocalMILRouteAAdapter(embedding_dim=16)(
            rgb, local_tokens
        ).shape == (2, 3)

    experts = HyperLeafTargetExpertsAdapter(embedding_dim=16)
    assert len(experts.experts) == 3
    assert len({id(expert.ve) for expert in experts.experts}) == 3
    shared = HyperLeafSharedHeadsAdapter(embedding_dim=16)
    assert len(shared.model.heads) == 3


def test_global_metadata_declares_fusion_baseline_without_correspondence_claim() -> None:
    runner = HyperLeafRouteARunner()

    metadata = runner.metadata(GLOBAL_FUSION_MODEL_KEY)

    assert metadata["method_description"] == (
        "Global Route A global fusion baseline; each modality is globally "
        "encoded before fusion, with no single-token correspondence claim"
    )
    assert metadata["artifact_root"] == "runs_v2/artifacts/hyperleaf-route-a"
    assert metadata["implementation_kind"] == "in_repo"


def test_legacy_flag_and_positional_selectors_are_rejected_instead_of_drifting() -> None:
    runner = HyperLeafRouteARunner()

    for selector in ("route_only", "--route_only", "token_only", "0"):
        with pytest.raises(HyperLeafRouteAError, match="stable Model_Key|unknown"):
            runner.resolve(selector)

    assert type(runner.build(GLOBAL_FUSION_MODEL_KEY)) is HyperLeafGlobalFusionAdapter
    assert type(runner.build(TOKEN_MODEL_KEY)) is HyperLeafTokenRouteAAdapter


def test_route_and_consistency_selection_uses_only_inner_variety_evidence() -> None:
    runner = HyperLeafRouteARunner()
    option_a = runner.selection_candidate(GLOBAL_FUSION_MODEL_KEY, "alignment-nce-v1")
    option_b = runner.selection_candidate(TOKEN_MODEL_KEY, "task-only-v1")
    decision = runner.freeze_route_and_consistency(
        _outer_fold(),
        options=(
            (GLOBAL_FUSION_MODEL_KEY, "alignment-nce-v1"),
            (TOKEN_MODEL_KEY, "task-only-v1"),
        ),
        candidate_scores={option_a.candidate_id: 0.3, option_b.candidate_id: 0.4},
        inner_source_sample_ids=("rembrandt-1", "rembrandt-2"),
        inner_prediction_hash="a" * 64,
        selection_policy=_policy(),
        decision_id="route-a-inner-variety-selection",
        frozen_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert decision.selected_candidate == option_b.candidate_id
    assert decision.source_sample_ids == ("rembrandt-1", "rembrandt-2")
    assert decision.candidates[0].configuration["consistency_setting"] == "alignment-nce-v1"
    assert decision.candidates[1].configuration["consistency_setting"] == "task-only-v1"

    with pytest.raises(Exception) as outer_evidence:
        runner.freeze_route_and_consistency(
            _outer_fold(),
            options=((GLOBAL_FUSION_MODEL_KEY, "alignment-nce-v1"),),
            candidate_scores={option_a.candidate_id: 0.3},
            inner_source_sample_ids=("sheriff-1",),
            inner_prediction_hash="b" * 64,
            selection_policy=_policy(),
        )
    assert getattr(outer_evidence.value, "error_record").error_code == (
        "SELECTION_OUTER_EVIDENCE_FORBIDDEN"
    )


def test_undeclared_consistency_candidate_is_rejected_before_selection() -> None:
    runner = HyperLeafRouteARunner()

    with pytest.raises(HyperLeafRouteAError, match="not declared"):
        runner.selection_candidate(GLOBAL_FUSION_MODEL_KEY, "outer-tuned-setting")
