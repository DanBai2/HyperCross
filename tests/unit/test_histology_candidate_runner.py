"""Unit coverage for task 15.5 Histology candidate and control wiring."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import torch
from torch import nn

from benchmark.adapters import (
    FULL_CUBE_SPATIAL_PCA_KEY,
    RGB_PREVIEW_KEY,
    HistologyHSIGBAdapter,
    RepresentationSourceKind,
)
from benchmark.engine import SelectionValidationError
from benchmark.models.histology import (
    ATTENTION_DISABLED_MODEL_KEY,
    LOCAL_ROUTE_A_MODEL_KEY,
    MIL_CONCAT_MODEL_KEY,
    PCA_ONLY_INFERENCE_BRANCH,
    PCA_ONLY_MODEL_KEY,
    RF_MODEL_KEY,
    RGB_ONLY_MODEL_KEY,
    SVM_MODEL_KEY,
)
from benchmark.runners import (
    ATTENTION_DISABLED_CONTROL_KEY,
    CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY,
    HISTOLOGY_CANDIDATE_SPECS,
    HISTOLOGY_NORMALIZATION_KEYS,
    HISTOLOGY_REVIEW_CANDIDATES,
    HISTOLOGY_REVIEW_READ_ONLY_SOURCES,
    LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
    LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
    LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
    LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
    MIL_CONCAT_CONTROL_KEY,
    PCA_TASK_ONLY_CONTROL_KEY,
    HistologyCandidateError,
    HistologyCandidateRunner,
    MatchedRunContext,
)
from benchmark.schema import (
    InnerFold,
    MetricDirection,
    OuterFold,
    RuntimeMeasurements,
    SelectionPolicy,
    canonical_deserialize,
)


HASHES = {name: f"{index:064x}" for index, name in enumerate(
    ("dataset", "protocol", "folds", "metrics", "budget"), start=1
)}


def _context(**updates: object) -> MatchedRunContext:
    values: dict[str, object] = {
        "dataset_hash": HASHES["dataset"],
        "protocol_hash": HASHES["protocol"],
        "seed_set": (7, 42, 123),
        "fold_manifest_hash": HASHES["folds"],
        "metric_configuration_hash": HASHES["metrics"],
        "resource_budget_hash": HASHES["budget"],
    }
    values.update(updates)
    return MatchedRunContext(**values)  # type: ignore[arg-type]


def _fold() -> OuterFold:
    return OuterFold(
        fold_id="outer-patient-1",
        outer_train_ids=("P2/ROI_A_T", "P3/ROI_B_NT"),
        outer_test_ids=("P1/ROI_C_T",),
        inner_folds=(
            InnerFold(
                fold_id="outer-patient-1.inner-1",
                train_ids=("P2/ROI_A_T",),
                validation_ids=("P3/ROI_B_NT",),
            ),
        ),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="hist.inner-patient.branch.v1",
        objective_metric="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def test_rgb_representation_is_governed_without_raw_cube_access() -> None:
    adapter = HistologyHSIGBAdapter(input_hasher=lambda reference: "a" * 64)  # type: ignore[arg-type]
    rgb = adapter.representation(RGB_PREVIEW_KEY)

    assert rgb.source_kind is RepresentationSourceKind.IMMUTABLE_SOURCE_IMAGE
    assert rgb.required_input_names == ("rgb_preview",)
    assert "forbidden" in rgb.raw_access_policy


def test_registry_exposes_all_stable_candidate_and_control_keys() -> None:
    runner = HistologyCandidateRunner()

    assert set(runner.model_keys) == set(HISTOLOGY_CANDIDATE_SPECS)
    assert set(runner.model_keys) == {
        PCA_ONLY_MODEL_KEY,
        RGB_ONLY_MODEL_KEY,
        LOCAL_ROUTE_A_MODEL_KEY,
        MIL_CONCAT_MODEL_KEY,
        ATTENTION_DISABLED_MODEL_KEY,
        SVM_MODEL_KEY,
        RF_MODEL_KEY,
    }
    for model_key in runner.model_keys:
        spec, registration = runner.resolve(model_key)
        assert spec.inference_branch
        assert registration.identity.model_key == model_key
        assert registration.input_contract.representation_key


def test_neural_adapters_build_and_follow_declared_shapes() -> None:
    runner = HistologyCandidateRunner()
    batch = 2
    rgb = torch.zeros(batch, 3, 64, 64)
    pca = torch.zeros(batch, 128)
    local = torch.zeros(batch, 16, 128)

    with torch.no_grad():
        assert runner.registry.build(PCA_ONLY_MODEL_KEY)(pca).shape == (batch, 2)
        assert runner.registry.build(RGB_ONLY_MODEL_KEY)(rgb).shape == (batch, 2)
        for model_key in (
            LOCAL_ROUTE_A_MODEL_KEY,
            MIL_CONCAT_MODEL_KEY,
            ATTENTION_DISABLED_MODEL_KEY,
        ):
            assert runner.registry.build(model_key)(rgb, local).shape == (batch, 2)


def test_attention_disabled_control_preserves_contract_but_removes_attention() -> None:
    runner = HistologyCandidateRunner()
    _, candidate = runner.resolve(LOCAL_ROUTE_A_MODEL_KEY)
    _, control = runner.resolve(ATTENTION_DISABLED_MODEL_KEY)
    candidate_model = runner.registry.build(LOCAL_ROUTE_A_MODEL_KEY)
    control_model = runner.registry.build(ATTENTION_DISABLED_MODEL_KEY)

    assert candidate.input_contract == control.input_contract
    assert candidate.input_contract.representation_key == FULL_CUBE_SPATIAL_PCA_KEY
    assert any(isinstance(module, nn.MultiheadAttention) for module in candidate_model.modules())
    assert not any(isinstance(module, nn.MultiheadAttention) for module in control_model.modules())


def test_classic_controls_build_with_fixed_pca_contract() -> None:
    runner = HistologyCandidateRunner()

    for model_key in (SVM_MODEL_KEY, RF_MODEL_KEY):
        model = runner.registry.build(model_key)
        _, registration = runner.resolve(model_key)
        assert tuple(model.parameters()) == ()
        assert registration.identity.parameter_count == 0
        assert registration.input_contract.modalities == ("pca_spectrum",)


def test_pca_only_inference_branch_is_frozen_from_inner_patient_evidence() -> None:
    runner = HistologyCandidateRunner()
    frozen_at = datetime(2027, 1, 2, tzinfo=timezone.utc)

    decision = runner.freeze_inference_branch(
        _fold(),
        model_key=PCA_ONLY_MODEL_KEY,
        inner_source_sample_ids=("P3/ROI_B_NT",),
        inner_prediction_hash="a" * 64,
        inner_score=0.75,
        selection_policy=_policy(),
        decision_id="hist-pca-branch",
        frozen_utc=frozen_at,
    )

    assert decision.selected_candidate == PCA_ONLY_INFERENCE_BRANCH
    assert decision.source_sample_ids == ("P3/ROI_B_NT",)
    assert decision.frozen_utc == frozen_at
    assert decision.candidates[0].configuration["inference_branch"] == "pca-only"

    with pytest.raises(SelectionValidationError):
        runner.freeze_inference_branch(
            _fold(),
            model_key=PCA_ONLY_MODEL_KEY,
            inner_source_sample_ids=("P1/ROI_C_T",),
            inner_prediction_hash="b" * 64,
            inner_score=1.0,
            selection_policy=_policy(),
        )


def test_local_route_controls_require_every_matched_comparison_field() -> None:
    runner = HistologyCandidateRunner()
    runner.validate_matched_control(
        LOCAL_ROUTE_A_MODEL_KEY,
        ATTENTION_DISABLED_MODEL_KEY,
        candidate_context=_context(),
        control_context=_context(seed_set=(123, 7, 42)),
    )

    for field in (
        "dataset_hash",
        "protocol_hash",
        "fold_manifest_hash",
        "metric_configuration_hash",
        "resource_budget_hash",
    ):
        with pytest.raises(HistologyCandidateError) as caught:
            runner.validate_matched_control(
                LOCAL_ROUTE_A_MODEL_KEY,
                MIL_CONCAT_MODEL_KEY,
                candidate_context=_context(),
                control_context=_context(**{field: "f" * 64}),
            )
        assert caught.value.mismatched_fields == (field,)


def test_undeclared_or_different_representation_is_not_a_matched_control() -> None:
    runner = HistologyCandidateRunner()

    with pytest.raises(HistologyCandidateError, match="not a declared matched control"):
        runner.validate_matched_control(
            LOCAL_ROUTE_A_MODEL_KEY,
            PCA_ONLY_MODEL_KEY,
            candidate_context=_context(),
            control_context=_context(),
        )


def test_method_wording_falls_back_to_combined_method_without_attention_control() -> None:
    runner = HistologyCandidateRunner()

    fallback = runner.method_description(
        LOCAL_ROUTE_A_MODEL_KEY,
        available_control_keys=(MIL_CONCAT_MODEL_KEY,),
    )
    controlled = runner.method_description(
        LOCAL_ROUTE_A_MODEL_KEY,
        available_control_keys=(ATTENTION_DISABLED_MODEL_KEY,),
    )

    assert fallback == (
        "combined method: local tokenization, MIL pooling, and cross-modal interaction"
    )
    assert "attention-disabled control" in controlled


def test_run_metadata_wires_required_comparison_columns() -> None:
    runner = HistologyCandidateRunner()
    _, registration = runner.resolve(LOCAL_ROUTE_A_MODEL_KEY)
    measurements = RuntimeMeasurements(
        completed_epochs=2,
        completed_training_steps=12,
        wall_clock_duration_ms=345,
        peak_device_memory_bytes=678,
        device_identity="cpu",
        numeric_precision="float32",
        parameter_count=registration.identity.parameter_count,
        input_volume_bytes=1024,
    )

    metadata = runner.run_metadata(
        LOCAL_ROUTE_A_MODEL_KEY,
        protocol_key="hist.patient-nested5.v1",
        measurements=measurements,
        available_control_keys=(ATTENTION_DISABLED_MODEL_KEY,),
    )

    assert metadata["protocol_key"] == "hist.patient-nested5.v1"
    assert metadata["representation_key"] == FULL_CUBE_SPATIAL_PCA_KEY
    assert metadata["parameter_count"] == registration.identity.parameter_count
    assert metadata["completed_training_steps"] == 12
    assert metadata["input_resolution"] == "3x64x64 RGB + 16x128 local PCA tokens"
    assert metadata["pretraining_data_status"] == "none"
    assert metadata["wall_clock_duration_ms"] == 345
    assert metadata["peak_device_memory_bytes"] == 678


def test_runner_rejects_non_g_workspace() -> None:
    with pytest.raises(HistologyCandidateError, match="remain on G"):
        HistologyCandidateRunner(r"F:\Datasets\HyperCross")


def test_task_17_7_review_configuration_freezes_all_named_options() -> None:
    """Validates: Requirements 10.10-10.25, 10.34."""

    runner = HistologyCandidateRunner()
    frozen_at = datetime(2027, 2, 3, 4, 5, tzinfo=timezone.utc)
    review = runner.freeze_review_configuration(frozen_utc=frozen_at)
    options = {
        option.option_key: option.parameters
        for option in review.candidate_space.model_candidates
    }

    assert set(options) == {spec.candidate_key for spec in HISTOLOGY_REVIEW_CANDIDATES}
    assert {
        CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY,
        LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
        MIL_CONCAT_CONTROL_KEY,
        ATTENTION_DISABLED_CONTROL_KEY,
        LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
        PCA_TASK_ONLY_CONTROL_KEY,
    }.issubset(options)
    assert options[CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY][
        "inference_branch"
    ] == PCA_ONLY_INFERENCE_BRANCH
    assert options[CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY][
        "consistency_weight"
    ] == 0.1
    assert options[LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY][
        "alignment_weight"
    ] == 0.08
    assert options[LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY]["nce_weight"] == 0.04
    assert tuple(option.option_key for option in review.normalization_candidates) == (
        HISTOLOGY_NORMALIZATION_KEYS
    )
    assert len(review.negative_candidate_keys) == 10
    assert all(key.startswith("histology.negative.") for key in review.negative_candidate_keys)
    assert review.execution_template.conda_environment == "TCFM"
    assert review.execution_template.expected_device_name == "NVIDIA GeForce RTX 5080"
    assert review.execution_template.process_count == 1
    assert review.execution_template.execution_mode.value == "deterministic"
    assert review.relative_output_root == "runs_v2/strict/histology-candidate-review"
    assert review.read_only_sources == HISTOLOGY_REVIEW_READ_ONLY_SOURCES
    assert set(review.execution_template.read_only_roots) >= {"runs", "src"}
    assert review.outer_label_access_allowed is False
    assert review.outer_evaluation_allowed is False
    assert len(review.review_hash) == 64


def test_task_17_7_pca_branch_requires_frozen_review_and_inner_patient_evidence() -> None:
    """Validates: Requirements 10.10, 10.11."""

    runner = HistologyCandidateRunner()
    frozen_at = datetime(2027, 2, 3, 4, 5, tzinfo=timezone.utc)
    review = runner.freeze_review_configuration(frozen_utc=frozen_at)

    decision = runner.freeze_pca_only_inference_branch(
        _fold(),
        review_configuration=review,
        inner_source_sample_ids=("P3/ROI_B_NT",),
        inner_prediction_hash="d" * 64,
        inner_score=0.8,
        selection_policy=_policy(),
        decision_id="consistency-pca-only",
        frozen_utc=frozen_at,
    )

    assert decision.selected_candidate == CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY
    assert decision.source_sample_ids == ("P3/ROI_B_NT",)
    assert decision.candidates[0].configuration["inference_branch"] == "pca-only"
    assert decision.candidates[0].configuration["candidate_space_hash"] == (
        review.candidate_space_hash
    )

    with pytest.raises(HistologyCandidateError, match="before outer labels"):
        runner.freeze_pca_only_inference_branch(
            _fold(),
            review_configuration=review,
            inner_source_sample_ids=("P3/ROI_B_NT",),
            inner_prediction_hash="e" * 64,
            inner_score=0.9,
            selection_policy=_policy(),
            outer_labels_accessible=True,
        )
    with pytest.raises(SelectionValidationError):
        runner.freeze_pca_only_inference_branch(
            _fold(),
            review_configuration=review,
            inner_source_sample_ids=("P1/ROI_C_T",),
            inner_prediction_hash="f" * 64,
            inner_score=1.0,
            selection_policy=_policy(),
        )


def test_task_17_7_review_controls_require_frozen_matched_identities() -> None:
    """Validates: Requirements 10.12-10.17, 10.34."""

    runner = HistologyCandidateRunner()
    review = runner.freeze_review_configuration(
        frozen_utc=datetime(2027, 2, 3, 4, 5, tzinfo=timezone.utc)
    )
    for control_key in (
        MIL_CONCAT_CONTROL_KEY,
        ATTENTION_DISABLED_CONTROL_KEY,
        LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
        LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
    ):
        runner.validate_review_matched_control(
            LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
            control_key,
            review_configuration=review,
            candidate_context=_context(),
            control_context=_context(seed_set=(123, 42, 7)),
        )
    runner.validate_review_matched_control(
        CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY,
        PCA_TASK_ONLY_CONTROL_KEY,
        review_configuration=review,
        candidate_context=_context(),
        control_context=_context(),
    )

    with pytest.raises(HistologyCandidateError) as caught:
        runner.validate_review_matched_control(
            LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
            MIL_CONCAT_CONTROL_KEY,
            review_configuration=review,
            candidate_context=_context(),
            control_context=_context(protocol_hash="f" * 64),
        )
    assert caught.value.mismatched_fields == ("protocol_hash",)


def test_task_17_7_later_training_job_is_strict_relative_and_recoverable() -> None:
    """Validates immutable later-run constraints without starting training."""

    runner = HistologyCandidateRunner()
    review = runner.freeze_review_configuration(
        frozen_utc=datetime(2027, 2, 3, 4, 5, tzinfo=timezone.utc)
    )
    job = runner.freeze_training_job(
        review_configuration=review,
        candidate_key=LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
        job_name="histology-review-fold-1-seed-7",
        dataset_hash="1" * 64,
        protocol_hash="2" * 64,
        protocol_key="hist.patient-nested5.v1",
        fold_id="outer-patient-1",
        seed=7,
        input_volume_bytes=4096,
    )
    parameters = canonical_deserialize(
        job.frozen_config.parameters_canonical_serialization
    )

    assert job.stage.value == "strict"
    assert job.output_root == "runs_v2/strict"
    assert job.frozen_identity_relative_path.startswith("runs_v2/strict/jobs/")
    assert job.recovery_directory.startswith("runs_v2/strict/jobs/")
    assert job.recovery_state_names == ("model", "optimizer", "scheduler", "stage")
    assert job.persist_identity_before_training is True
    assert job.frozen_config.execution_mode.value == "deterministic"
    assert job.frozen_config.checkpoint_retention.preserve_selected is True
    assert job.frozen_config.checkpoint_retention.preserve_last is True
    assert job.frozen_config.checkpoint_retention.preserve_recovery_state is True
    assert parameters["conda_environment"] == "TCFM"
    assert parameters["device"] == "cuda:0"
    assert parameters["process_count"] == 1
    assert parameters["outer_evaluation_allowed"] is False
    assert parameters["outer_label_access_allowed"] is False
    assert tuple(parameters["read_only_sources"]) == HISTOLOGY_REVIEW_READ_ONLY_SOURCES
    assert parameters["candidate_parameters"]["alignment_weight"] == 0.08
    assert parameters["candidate_parameters"]["nce_weight"] == 0.04
    assert parameters["review_artifact_root"] == (
        "runs_v2/strict/histology-candidate-review"
    )
