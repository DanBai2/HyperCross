"""Targeted validation for task 13.9 fast-lane execution contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.fast_lane import (
    FastLaneConfigurationError,
    FastLaneStage,
    FastLaneStageTemplate,
    load_fast_lane_template,
)
from benchmark.schema import (
    EnvironmentLock,
    LimitEnforcement,
    MetricDirection,
    NumericPrecision,
    canonical_deserialize,
)


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = ROOT / "configs" / "fast_lane"
AUDIT_PATH = ROOT / "artifacts" / "audit" / "training_fast_lane_audit.json"
EXPECTED = {
    FastLaneStage.SMOKE: ("runs_v2/smoke", (42,), 2, 128, 900_000),
    FastLaneStage.BASELINE: (
        "runs_v2/baselines",
        (42,),
        300,
        500_000,
        172_800_000,
    ),
    FastLaneStage.SCREENING: (
        "runs_v2/screening",
        (42,),
        80,
        100_000,
        43_200_000,
    ),
    FastLaneStage.STRICT: (
        "runs_v2/strict",
        (7, 42, 123),
        400,
        1_000_000,
        259_200_000,
    ),
    FastLaneStage.OFFICIAL: (
        "runs_v2/official",
        (42,),
        400,
        1_000_000,
        259_200_000,
    ),
}


def _load(stage: FastLaneStage) -> FastLaneStageTemplate:
    return load_fast_lane_template(TEMPLATE_ROOT / f"{stage.value}.json")


def _budget_values(template: FastLaneStageTemplate) -> dict[str, int]:
    return {limit.field: limit.value for limit in template.resource_budget.limits}


@pytest.mark.parametrize("stage", tuple(FastLaneStage))
def test_stage_templates_are_hash_bound_and_freeze_all_execution_fields(
    stage: FastLaneStage,
) -> None:
    template = _load(stage)
    output_root, seeds, epochs, steps, duration_ms = EXPECTED[stage]
    budget = _budget_values(template)

    assert template.template_hash == template.model_copy().template_hash
    assert template.conda_environment == "TCFM"
    assert template.device == "cuda:0"
    assert template.expected_device_name == "NVIDIA GeForce RTX 5080"
    assert template.visible_cuda_device_count == 1
    assert template.process_count == 1
    assert template.numeric_precision is NumericPrecision.FLOAT32
    assert template.execution_mode.value == "deterministic"
    assert template.framework_settings.deterministic_algorithms is True
    assert template.framework_settings.cudnn_deterministic is True
    assert template.framework_settings.cudnn_benchmark is False
    assert template.seed_set == seeds
    assert template.checkpoint_interval_epochs == 1
    assert template.output_root == output_root
    assert template.read_only_roots == ("runs", "src")
    assert template.recovery_state_names == (
        "model",
        "optimizer",
        "scheduler",
        "stage",
    )
    assert template.persist_frozen_identity_before_training is True
    assert template.checkpoint_retention.preserve_selected is True
    assert template.checkpoint_retention.preserve_last is True
    assert template.checkpoint_retention.preserve_recovery_state is True
    assert budget["completed_epochs"] == epochs
    assert budget["completed_training_steps"] == steps
    assert budget["wall_clock_duration_ms"] == duration_ms
    assert budget["peak_device_memory_bytes"] == 16_000_000_000
    assert {limit.enforcement for limit in template.resource_budget.limits} == {
        LimitEnforcement.HARD,
        LimitEnforcement.ADVISORY,
    }


def test_template_validates_audited_environment_and_only_g_workspace() -> None:
    template = _load(FastLaneStage.SMOKE)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    lock = EnvironmentLock.model_validate(audit["environment_lock"])

    template.validate_environment(lock, conda_environment="TCFM")
    assert str(template.validate_workspace_root(r"G:\Datasets\HyperCross")) == (
        r"G:\Datasets\HyperCross"
    )

    with pytest.raises(FastLaneConfigurationError, match="conda_environment"):
        template.validate_environment(lock, conda_environment="base")
    with pytest.raises(FastLaneConfigurationError, match="workspace root"):
        template.validate_workspace_root(r"F:\Datasets\HyperCross")


def test_job_freeze_binds_config_identity_budget_and_recovery_locations() -> None:
    template = _load(FastLaneStage.STRICT)
    job = template.freeze_job(
        job_name="hyperleaf-fold-03-seed-123",
        dataset_hash="1" * 64,
        protocol_hash="2" * 64,
        protocol_key="leaf.lovo.nested.v1",
        model_key="hyperleaf.route-a.v1",
        fold_id="variety-03",
        seed=123,
        input_volume_bytes=8192,
        selection_metric_direction=MetricDirection.MAXIMIZE,
        parameters={"learning_rate": 0.0001, "batch_size": 8},
    )
    parameters = canonical_deserialize(
        job.frozen_config.parameters_canonical_serialization.encode("utf-8")
    )

    assert job.template_hash == template.template_hash
    assert job.frozen_identity.config_hash == job.frozen_config.config_hash
    assert job.frozen_identity.environment_lock_hash == template.environment_lock_hash
    assert job.frozen_identity.environment_compatibility_policy == "exact"
    assert job.frozen_identity.seed == 123
    assert job.frozen_config.resource_budget == template.resource_budget
    assert parameters["device"] == "cuda:0"
    assert parameters["process_count"] == 1
    assert parameters["checkpoint_interval_epochs"] == 1
    assert parameters["epochs"] == 400
    assert job.frozen_identity_relative_path == (
        "runs_v2/strict/jobs/hyperleaf-fold-03-seed-123/"
        "frozen_run_identity.json"
    )
    assert job.recovery_directory == (
        "runs_v2/strict/jobs/hyperleaf-fold-03-seed-123/recovery"
    )
    assert job.recovery_state_names == ("model", "optimizer", "scheduler", "stage")
    assert job.persist_identity_before_training is True


def test_path_and_override_gates_reject_f_drive_legacy_and_cross_stage_resume() -> None:
    template = _load(FastLaneStage.BASELINE)

    assert template.validate_write_path("runs_v2/baselines/hyperleaf/job-1.bin") == (
        "runs_v2/baselines/hyperleaf/job-1.bin"
    )
    assert template.validate_write_path("artifacts/environment/job-1.json") == (
        "artifacts/environment/job-1.json"
    )
    assert template.validate_recovery_source(
        "runs_v2/baselines/hyperleaf/job-1/recovery/epoch-1.json"
    ).startswith("runs_v2/baselines/")

    rejected = (
        (template.validate_recovery_source, r"F:\checkpoints\epoch-1.pt"),
        (template.validate_recovery_source, "runs/legacy/checkpoint.pt"),
        (template.validate_recovery_source, "runs_v2/strict/job/recovery.json"),
        (template.validate_write_path, "src/new_runner.py"),
        (template.validate_write_path, "runs/overwritten.pt"),
        (template.validate_write_path, "../outside.bin"),
    )
    for validator, path in rejected:
        with pytest.raises((FastLaneConfigurationError, ValueError)):
            validator(path)

    common = {
        "job_name": "baseline-job",
        "dataset_hash": "1" * 64,
        "protocol_hash": "2" * 64,
        "protocol_key": "leaf.baseline.v1",
        "model_key": "classic.plsr.v1",
        "fold_id": "fold-1",
        "seed": 42,
        "input_volume_bytes": 1,
        "selection_metric_direction": MetricDirection.MAXIMIZE,
    }
    with pytest.raises(FastLaneConfigurationError, match="cannot override"):
        template.freeze_job(**common, parameters={"device": "cpu"})
    with pytest.raises(FastLaneConfigurationError, match="outside"):
        template.freeze_job(**{**common, "seed": 7})


def test_tampered_template_is_rejected_before_scheduling() -> None:
    payload = json.loads(
        (TEMPLATE_ROOT / "official.json").read_text(encoding="utf-8")
    )
    payload["process_count"] = 2

    with pytest.raises(ValidationError, match="process_count"):
        FastLaneStageTemplate.model_validate(payload)
