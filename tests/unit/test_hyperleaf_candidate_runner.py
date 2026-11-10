"""Unit coverage for task 14.11 HyperLeaf candidate reconstruction.

Validates Requirements 4.14-4.21, 5.1-5.18, 10.1-10.11, 13.1-13.29.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark.catalog.hyperleaf import HyperLeafPartition
from benchmark.models.hyperleaf_candidates import (
    AVERAGE_BRANCH,
    INDEPENDENT_TARGET_MODEL_KEY,
    PCA_INDEPENDENT_TARGET_MODEL_KEY,
    RGB_PCA_CONSISTENCY_MODEL_KEY,
    HyperLeafIndependentTargets,
    HyperLeafPcaIndependentTargets,
    HyperLeafRgbPcaConsistency,
    rgb_pca_consistency_loss,
)
from benchmark.runners.hyperleaf_candidates import (
    GSW_CANDIDATE_CONFIGS,
    HYPERLEAF_CANDIDATE_SPECS,
    GswRoute,
    HyperLeafCandidateBranch,
    HyperLeafCandidateError,
    HyperLeafCandidateRunner,
)
from benchmark.schema import EnvironmentLock, FitBoundaryScope, InnerFold, OuterFold
from benchmark.transforms.hyperleaf_candidates import (
    GswLossKind,
    GswTargetTransform,
    HYPERLEAF_STYLE_CONFIGS,
    STRONG_STYLE,
    apply_hyperleaf_style,
    gsw_regression_loss,
    inverse_gsw_target,
    transform_gsw_target,
)
from benchmark.transforms.preprocessing import PreprocessingManager


NOW = datetime(2027, 1, 1, tzinfo=timezone.utc)
DATASET_HASH = "a" * 64
PROTOCOL_HASH = "b" * 64


def _fold() -> OuterFold:
    return OuterFold(
        fold_id="held-out-sheriff",
        outer_train_ids=tuple(f"train-{index}" for index in range(6)),
        outer_test_ids=("test-0", "test-1"),
        inner_folds=(
            InnerFold(
                fold_id="inner-a",
                train_ids=("train-0", "train-1", "train-2"),
                validation_ids=("train-3", "train-4", "train-5"),
            ),
            InnerFold(
                fold_id="inner-b",
                train_ids=("train-3", "train-4", "train-5"),
                validation_ids=("train-0", "train-1", "train-2"),
            ),
        ),
    )


class _FoldRepresentations:
    def __init__(self) -> None:
        self.adapter = SimpleNamespace(partition=HyperLeafPartition.TRAIN)
        self.preprocessing = PreprocessingManager()

    def fit_target_normalization(self, sample_ids, boundary, *, dataset_hash, protocol_hash):
        values = np.arange(len(sample_ids) * 3, dtype=np.float64).reshape(len(sample_ids), 3)
        return self.preprocessing.fit_target_normalization(
            values,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
        )

    def fit_pca_pseudo_rgb(self, sample_ids, boundary, *, dataset_hash, protocol_hash):
        values = np.arange(len(sample_ids) * 5, dtype=np.float64).reshape(len(sample_ids), 5)
        return self.preprocessing.fit_pca(
            values,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            n_components=3,
        )


def _environment() -> EnvironmentLock:
    return EnvironmentLock.create(
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


def test_all_named_branches_use_new_registered_candidate_models() -> None:
    runner = HyperLeafCandidateRunner()

    assert set(HYPERLEAF_CANDIDATE_SPECS) == set(HyperLeafCandidateBranch)
    assert set(runner.model_keys) == {
        INDEPENDENT_TARGET_MODEL_KEY,
        PCA_INDEPENDENT_TARGET_MODEL_KEY,
        RGB_PCA_CONSISTENCY_MODEL_KEY,
    }
    expected_types = {
        HyperLeafCandidateBranch.INDEPENDENT_TARGET: HyperLeafIndependentTargets,
        HyperLeafCandidateBranch.STYLE_AUGMENTATION: HyperLeafIndependentTargets,
        HyperLeafCandidateBranch.TRAIN_FOLD_PCA: HyperLeafPcaIndependentTargets,
        HyperLeafCandidateBranch.RGB_PCA_CONSISTENCY: HyperLeafRgbPcaConsistency,
    }
    for branch, expected_type in expected_types.items():
        model = runner.build_branch(branch)
        assert type(model) is expected_type
        assert model.__class__.__module__ == "benchmark.models.hyperleaf_candidates"


def test_candidate_models_expose_independent_and_rgb_pca_predictions() -> None:
    rgb = torch.rand(2, 3, 32, 64)
    pca = torch.rand(2, 3, 32, 64)
    spectrum = torch.rand(2, 128)

    independent = HyperLeafIndependentTargets(embedding_dim=16)
    consistency = HyperLeafRgbPcaConsistency(embedding_dim=16)
    with torch.no_grad():
        assert independent(rgb, spectrum).shape == (2, 3)
        routed, branches = consistency(
            rgb,
            pca,
            spectrum,
            inference_branch="target-routed",
            target_routes=("rgb", AVERAGE_BRANCH, "pca"),
            return_branches=True,
        )
    assert routed.shape == (2, 3)
    assert set(branches) == {"rgb", "pca", AVERAGE_BRANCH}
    assert rgb_pca_consistency_loss(
        branches["rgb"], branches["pca"], weight=0.1
    ).ndim == 0
    assert len({id(expert) for expert in independent.experts}) == 3


def test_style_and_gsw_loss_transform_wiring_is_deterministic() -> None:
    rgb = torch.linspace(0, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    spectrum = torch.linspace(-1, 1, 2 * 16).reshape(2, 16)

    first = apply_hyperleaf_style(rgb, spectrum, config=STRONG_STYLE, seed=42)
    second = apply_hyperleaf_style(rgb, spectrum, config=STRONG_STYLE, seed=42)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert all(torch.isfinite(value).all() for value in first)
    assert torch.all((first[0] >= 0) & (first[0] <= 1))

    target = torch.tensor([-1.0, 0.0, 1.0])
    transformed = transform_gsw_target(target, GswTargetTransform.ASINH_005)
    assert torch.allclose(inverse_gsw_target(transformed, GswTargetTransform.ASINH_005), target)
    assert gsw_regression_loss(
        transformed, transformed, loss_kind=GswLossKind.HUBER
    ).item() == 0.0


def test_gsw_space_is_complete_and_frozen_from_exact_inner_oof() -> None:
    runner = HyperLeafCandidateRunner()
    expected = {
        (route, loss, transform)
        for route in GswRoute
        for loss in GswLossKind
        for transform in GswTargetTransform
    }
    observed = {
        (candidate.route, candidate.loss_kind, candidate.target_transform)
        for candidate in GSW_CANDIDATE_CONFIGS
    }
    assert observed == expected
    assert len({candidate.candidate_id for candidate in GSW_CANDIDATE_CONFIGS}) == len(expected)
    runner.candidate_space.require_integrity()
    assert {option.option_key for option in runner.candidate_space.target_route_candidates} == {
        route.value for route in GswRoute
    }
    assert {option.option_key for option in runner.candidate_space.loss_candidates} == {
        loss.value for loss in GswLossKind
    }

    scores = {
        candidate.candidate_id: float(index)
        for index, candidate in enumerate(GSW_CANDIDATE_CONFIGS)
    }
    decision = runner.freeze_gsw_selection(
        _fold(),
        candidate_scores=scores,
        inner_source_sample_ids=_fold().outer_train_ids,
        inner_prediction_hash="c" * 64,
        decision_id="gsw-inner-selection",
        frozen_utc=NOW,
    )
    assert len(decision.candidates) == len(expected)
    assert decision.source_sample_ids == _fold().outer_train_ids
    assert decision.selected_candidate == max(scores, key=scores.get)  # type: ignore[arg-type]
    assert runner.prepare_outer_refit(_fold(), decision).outer_train_ids == _fold().outer_train_ids

    with pytest.raises(HyperLeafCandidateError, match="exact complete"):
        runner.freeze_gsw_selection(
            _fold(),
            candidate_scores=scores,
            inner_source_sample_ids=("train-0",),
            inner_prediction_hash="d" * 64,
        )


def test_preprocessing_uses_exact_inner_or_outer_train_boundaries() -> None:
    runner = HyperLeafCandidateRunner()
    representations = _FoldRepresentations()

    inner = runner.fit_preprocessing(
        representations,  # type: ignore[arg-type]
        _fold(),
        dataset_hash=DATASET_HASH,
        protocol_hash=PROTOCOL_HASH,
        branch=HyperLeafCandidateBranch.RGB_PCA_CONSISTENCY,
        inner_fold_id="inner-a",
    )
    assert inner.boundary.scope is FitBoundaryScope.INNER_TRAIN
    assert inner.target_normalization.fit_sample_ids == ("train-0", "train-1", "train-2")
    assert inner.pca is not None
    assert inner.pca.fit_sample_ids == inner.target_normalization.fit_sample_ids
    assert set(inner.pca.fit_sample_ids).isdisjoint(_fold().outer_test_ids)

    outer = runner.fit_preprocessing(
        representations,  # type: ignore[arg-type]
        _fold(),
        dataset_hash=DATASET_HASH,
        protocol_hash=PROTOCOL_HASH,
        branch=HyperLeafCandidateBranch.TRAIN_FOLD_PCA,
    )
    assert outer.boundary.scope is FitBoundaryScope.OUTER_TRAIN
    assert outer.target_normalization.fit_sample_ids == _fold().outer_train_ids
    assert outer.pca is not None and outer.pca.fit_sample_ids == _fold().outer_train_ids

    representations.adapter.partition = HyperLeafPartition.OFFICIAL_TEST
    with pytest.raises(HyperLeafCandidateError, match="training partition"):
        runner.fit_preprocessing(
            representations,  # type: ignore[arg-type]
            _fold(),
            dataset_hash=DATASET_HASH,
            protocol_hash=PROTOCOL_HASH,
            branch=HyperLeafCandidateBranch.INDEPENDENT_TARGET,
        )


def test_launch_output_and_recovery_wiring_are_frozen_and_relative() -> None:
    runner = HyperLeafCandidateRunner()
    runner.validate_launch(
        _environment(),
        conda_environment="TCFM",
        process_count=1,
        deterministic_algorithms=True,
        cudnn_benchmark=False,
    )
    metadata = runner.metadata()
    assert metadata["output_root"] == "runs_v2/candidates/hyperleaf"
    assert not Path(metadata["output_root"]).is_absolute()
    assert metadata["official_test_labels_allowed"] is False
    assert metadata["legacy_src_imported"] is False
    assert metadata["execution"]["checkpoint_boundary"] == "epoch"  # type: ignore[index]

    with pytest.raises(HyperLeafCandidateError, match="exactly G"):
        HyperLeafCandidateRunner(r"F:\Datasets\HyperCross")
    with pytest.raises(HyperLeafCandidateError, match="execution contract"):
        runner.validate_launch(
            _environment(),
            conda_environment="base",
            process_count=2,
            deterministic_algorithms=False,
            cudnn_benchmark=True,
        )

    captured = {}
    active = SimpleNamespace(
        frozen_identity=SimpleNamespace(model_key=INDEPENDENT_TARGET_MODEL_KEY)
    )
    fake_manager = SimpleNamespace(
        registry=SimpleNamespace(latest_run=lambda run_id: active),
        save_checkpoint_boundary=lambda **values: captured.update(values) or values,
    )
    result = runner.save_recovery_checkpoint(
        fake_manager,  # type: ignore[arg-type]
        run_id="run-1",
        epoch=3,
        model_state=b"model",
        optimizer_state=b"optimizer",
        scheduler_state=b"scheduler",
        stage_state=b"stage",
        created_utc=NOW,
    )
    assert result["relative_path"] == (
        "runs_v2/candidates/hyperleaf/recovery/run-1/epoch-000003.canonical.json"
    )
    assert result["completed_checkpoint_boundary"] == 3
