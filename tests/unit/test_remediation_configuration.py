"""Focused non-smoke tests for task 24.1 remediation templates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.remediation import (
    RemediationConfigurationError,
    RemediationPlanningContext,
    RemediationProfile,
    load_remediation_template,
)
from benchmark.schema import (
    EnvironmentLock,
    ExecutionMode,
    MetricDirection,
    NumericPrecision,
    canonical_deserialize,
)


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = ROOT / "configs" / "remediation" / "v1"
AUDIT_PATH = ROOT / "artifacts" / "audit" / "training_fast_lane_audit.json"
FAST_LANE_BYTE_HASHES = {
    "baseline.json": "075f6b41c7b42f97e11a3651b552d8df21656ab902e0a47bfa32bf1be88cc81e",
    "official.json": "215ca0d98f437f5f25f7cf2f930aaf51a564b46ae70bbcd85ee30f34e79f4b4e",
    "screening.json": "6e04d1e435169d56058386e4d9bee7a30c026af4a5789027fc706143777970dc",
    "smoke-job.example.json": "fbbc9ae79f679a68eb65469b17d07d732e98fc0fd01da44bb7601a90a4b0a222",
    "smoke.json": "8710c0e333b711330c03ffd5fa34de05207915483965035a2fec4299d574e3c6",
    "strict.json": "098630830b96bffa1d3faa4e5a967fcc23ca12dcb7cc3717c4e9dbd2b80e135d",
}


def _templates():
    deterministic = load_remediation_template(
        TEMPLATE_ROOT / "deterministic-float32.json"
    )
    performance = load_remediation_template(TEMPLATE_ROOT / "performance-bf16.json")
    return deterministic, performance


def _environment_lock() -> EnvironmentLock:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    return EnvironmentLock.model_validate(audit["environment_lock"])


def _context(template, **updates) -> RemediationPlanningContext:
    values = {
        "conda_environment": template.conda_environment,
        "environment_lock": _environment_lock(),
        "device": template.device,
        "process_count": template.process_count,
        "numeric_precision": template.numeric_precision,
        "execution_mode": template.execution_mode,
        "autocast_policy": template.autocast_policy,
        "gradient_accumulation": template.gradient_accumulation,
    }
    values.update(updates)
    return RemediationPlanningContext(**values)


def _freeze(template, context=None, resume_identity=None):
    return template.freeze_job(
        planning_context=context or _context(template),
        job_name="hyperleaf-variety-1-seed-42",
        dataset_hash="1" * 64,
        protocol_hash="2" * 64,
        protocol_key="leaf.lovo.nested.v1",
        model_key="hyperleaf.route-a.v1",
        fold_id="variety-1",
        seed=42,
        input_volume_bytes=8192,
        selection_metric_direction=MetricDirection.MAXIMIZE,
        parameters={"batch_size": 8, "learning_rate": 0.0001},
        resume_identity=resume_identity,
    )


def test_versioned_templates_have_independent_canonical_identity_and_namespaces() -> None:
    deterministic, performance = _templates()

    assert deterministic.configuration_namespace == "remediation/v1"
    assert performance.configuration_namespace == "remediation/v1"
    assert deterministic.template_version == performance.template_version == "v1"
    assert deterministic.profile is RemediationProfile.DETERMINISTIC_FLOAT32
    assert performance.profile is RemediationProfile.PERFORMANCE_BF16
    assert deterministic.config_hash == (
        "6a546010f4ba2793e5dc6150376b4be482fed918f5e5fd6bda19d3f207043fe9"
    )
    assert performance.config_hash == (
        "832be123aee4305010f1a65795daf6931e83ef9cebced7a27aadf5b71c3ee78d"
    )
    assert deterministic.config_hash != performance.config_hash
    assert deterministic.output_root != performance.output_root
    assert deterministic.recovery_root != performance.recovery_root
    assert deterministic.artifact_root != performance.artifact_root

    for template in (deterministic, performance):
        assert template.conda_environment == "TCFM"
        assert template.expected_device_name == "NVIDIA GeForce RTX 5080"
        assert template.visible_cuda_device_count == template.process_count == 1
        assert template.seed_set == (7, 42, 123)
        assert template.checkpoint_interval_epochs == 1
        assert template.recovery_state_names == (
            "model",
            "optimizer",
            "scheduler",
            "stage",
        )
        assert template.input_volume_accounting.method == "actual-bytes-read"
        assert template.input_volume_accounting.count_repeated_reads is True
        assert template.input_volume_accounting.include_preprocessing_reads is True
        assert template.input_volume_accounting.require_runtime_measurement is True


def test_profiles_freeze_float32_and_bf16_autocast_performance_policies() -> None:
    deterministic, performance = _templates()

    assert deterministic.numeric_precision is NumericPrecision.FLOAT32
    assert deterministic.execution_mode is ExecutionMode.DETERMINISTIC
    assert deterministic.framework_settings.deterministic_algorithms is True
    assert deterministic.framework_settings.cudnn_benchmark is False
    assert deterministic.autocast_policy.enabled is False
    assert deterministic.autocast_policy.dtype is NumericPrecision.FLOAT32

    assert performance.numeric_precision is NumericPrecision.MIXED_BFLOAT16
    assert performance.execution_mode is ExecutionMode.PERFORMANCE
    assert performance.framework_settings.deterministic_algorithms is False
    assert performance.framework_settings.cudnn_benchmark is True
    assert performance.framework_settings.allow_tf32 is True
    assert performance.autocast_policy.enabled is True
    assert performance.autocast_policy.device_type == "cuda"
    assert performance.autocast_policy.dtype is NumericPrecision.BFLOAT16

    for template in (deterministic, performance):
        assert template.gradient_accumulation.accumulation_steps == 1
        assert template.gradient_accumulation.divide_loss_by_accumulation_steps is True
        assert template.gradient_accumulation.flush_final_partial_accumulation is True


def test_planning_binds_precision_policy_budget_and_distinct_job_config_hashes() -> None:
    deterministic, performance = _templates()
    deterministic_job = _freeze(deterministic)
    performance_job = _freeze(performance)
    deterministic_parameters = canonical_deserialize(
        deterministic_job.frozen_config.parameters_canonical_serialization.encode("utf-8")
    )
    performance_parameters = canonical_deserialize(
        performance_job.frozen_config.parameters_canonical_serialization.encode("utf-8")
    )

    assert deterministic_job.frozen_identity.config_hash != (
        performance_job.frozen_identity.config_hash
    )
    assert deterministic_parameters["remediation_template_config_hash"] == (
        deterministic.config_hash
    )
    assert performance_parameters["remediation_template_config_hash"] == (
        performance.config_hash
    )
    assert deterministic_parameters["numeric_precision"] == "float32"
    assert performance_parameters["numeric_precision"] == "mixed_bfloat16"
    assert performance_parameters["autocast_policy"]["dtype"] == "bfloat16"
    assert performance_parameters["gradient_accumulation"]["accumulation_steps"] == 1
    assert deterministic_job.frozen_config.resource_budget == deterministic.resource_budget
    assert performance_job.frozen_config.resource_budget == performance.resource_budget
    assert deterministic_job.frozen_config.input_volume_bytes == 8192
    assert performance_job.frozen_config.input_volume_bytes == 8192
    assert deterministic_job.recovery_directory.startswith(deterministic.recovery_root)
    assert performance_job.recovery_directory.startswith(performance.recovery_root)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("conda_environment", "base"),
        ("device", "cuda:1"),
        ("process_count", 2),
        ("numeric_precision", NumericPrecision.FLOAT32),
        ("execution_mode", ExecutionMode.DETERMINISTIC),
    ),
)
def test_bf16_planning_rejects_runtime_identity_mismatch(
    field: str,
    bad_value: object,
) -> None:
    _, performance = _templates()
    context = _context(performance).model_copy(update={field: bad_value})

    with pytest.raises(RemediationConfigurationError, match=field):
        _freeze(performance, context=context)


def _changed_lock(lock: EnvironmentLock, **updates) -> EnvironmentLock:
    values = lock.model_dump(mode="python", exclude={"lock_hash"})
    values.update(updates)
    return EnvironmentLock.create(**values)


def test_planning_rejects_wrong_environment_lock_and_device_identity() -> None:
    deterministic, _ = _templates()
    lock = _environment_lock()

    for changed_lock in (
        _changed_lock(lock, python_version="3.10.19"),
        _changed_lock(lock, device_identity="cuda:0:NVIDIA GeForce RTX 4090"),
        _changed_lock(lock, cuda_device_count=2),
    ):
        with pytest.raises(RemediationConfigurationError):
            _freeze(
                deterministic,
                context=_context(deterministic, environment_lock=changed_lock),
            )


def test_cross_precision_resume_and_recovery_paths_are_rejected() -> None:
    deterministic, performance = _templates()
    deterministic_job = _freeze(deterministic)
    performance_job = _freeze(performance)

    deterministic_job.require_exact_resume_identity(deterministic_job.frozen_identity)
    with pytest.raises(RemediationConfigurationError, match="config_hash"):
        deterministic_job.require_exact_resume_identity(performance_job.frozen_identity)
    with pytest.raises(RemediationConfigurationError, match="config_hash"):
        _freeze(
            deterministic,
            resume_identity=performance_job.frozen_identity,
        )
    with pytest.raises(RemediationConfigurationError, match="precision-specific"):
        deterministic_job.validate_recovery_source(
            f"{performance.recovery_root}/hyperleaf-variety-1-seed-42/epoch-1.json"
        )

    accepted = (
        f"{deterministic_job.recovery_directory}/epoch-1.json"
    )
    assert deterministic_job.validate_recovery_source(accepted) == accepted


def test_read_only_and_workspace_guards_preserve_legacy_state() -> None:
    deterministic, _ = _templates()

    assert str(deterministic.validate_workspace_root(r"G:\Datasets\HyperCross")) == (
        r"G:\Datasets\HyperCross"
    )
    with pytest.raises(RemediationConfigurationError, match="workspace root"):
        deterministic.validate_workspace_root(r"F:\Datasets\HyperCross")

    for path in (
        "configs/fast_lane/strict.json",
        "src/new_runner.py",
        "runs/legacy.pt",
        "checkpoints/legacy.pt",
        "datasets/materialized.bin",
        "caches/features.bin",
        "artifacts/audit/training_fast_lane_audit.json",
    ):
        with pytest.raises(RemediationConfigurationError):
            deterministic.validate_write_path(path)


def test_existing_float32_templates_remain_byte_identical() -> None:
    fast_lane_root = ROOT / "configs" / "fast_lane"
    observed = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in fast_lane_root.glob("*.json")
    }

    assert observed == FAST_LANE_BYTE_HASHES
