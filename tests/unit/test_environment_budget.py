"""Unit coverage for environment locks and frozen resource configuration."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from benchmark.schema import (
    CheckpointRetentionPolicy,
    EnvironmentLock,
    ExecutionMode,
    FrozenRunConfig,
    LimitEnforcement,
    NumericPrecision,
    ResourceBudget,
    ResourceLimit,
    RuntimeMeasurements,
    assess_resource_budget,
    capture_environment_lock,
)


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "NVIDIA GeForce RTX 5080"


def _version_provider(distribution: str) -> str:
    return {
        "torch": "2.8.0",
        "numpy": "2.2.0",
        "pandas": "2.2.3",
        "Pillow": "11.0.0",
        "tifffile": "2025.1.1",
        "scikit-learn": "1.6.0",
        "scipy": "1.15.0",
    }[distribution]


def _budget() -> ResourceBudget:
    return ResourceBudget(
        limits=(
            ResourceLimit(
                field="completed_epochs",
                value=10,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            ),
            ResourceLimit(
                field="peak_device_memory_bytes",
                value=1_000,
                unit="bytes",
                enforcement=LimitEnforcement.ADVISORY,
            ),
        )
    )


def test_capture_environment_lock_records_cuda_device_and_valid_hash() -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=_FakeCuda(),
    )

    lock = capture_environment_lock(
        version_provider=_version_provider,
        torch_module=fake_torch,
    )

    assert lock.pytorch_version == "2.8.0"
    assert lock.cuda_version == "12.8"
    assert lock.cuda_device_count == 1
    assert lock.device_identity == "cuda:0:NVIDIA GeForce RTX 5080"
    assert len(lock.lock_hash) == 64

    tampered = lock.model_dump()
    tampered["numpy_version"] = "0.0.0"
    with pytest.raises(ValidationError, match="lock_hash does not match"):
        EnvironmentLock.model_validate(tampered)


def test_frozen_config_hash_is_order_independent_and_records_execution_policy() -> None:
    first = FrozenRunConfig.create(
        model_key="model.example",
        protocol_key="protocol.example.v1",
        execution_mode=ExecutionMode.DETERMINISTIC,
        numeric_precision=NumericPrecision.BFLOAT16,
        input_volume_bytes=4096,
        resource_budget=_budget(),
        parameters={"learning_rate": 0.001, "batch_size": 8},
    )
    second = FrozenRunConfig.create(
        model_key="model.example",
        protocol_key="protocol.example.v1",
        execution_mode=ExecutionMode.DETERMINISTIC,
        numeric_precision=NumericPrecision.BFLOAT16,
        input_volume_bytes=4096,
        resource_budget=_budget(),
        parameters={"batch_size": 8, "learning_rate": 0.001},
    )

    assert first.config_hash == second.config_hash
    assert first.framework_settings.deterministic_algorithms is True
    assert first.framework_settings.cudnn_benchmark is False
    assert first.checkpoint_retention == CheckpointRetentionPolicy(
        preserve_selected=True,
        preserve_last=True,
        preserve_recovery_state=True,
    )
    with pytest.raises(ValidationError):
        first.numeric_precision = NumericPrecision.FLOAT32


def test_budget_assessment_stops_for_hard_limit_and_only_warns_for_advisory() -> None:
    measurements = RuntimeMeasurements(
        completed_epochs=11,
        completed_training_steps=100,
        wall_clock_duration_ms=2_000,
        peak_device_memory_bytes=1_500,
        device_identity="cuda:0:NVIDIA GeForce RTX 5080",
        numeric_precision="bfloat16",
        parameter_count=50,
        input_volume_bytes=4_096,
    )

    result = assess_resource_budget(
        _budget(), measurements, checkpoint_boundary=11
    )

    assert result.stop_at_next_checkpoint_boundary is True
    assert result.stopping_checkpoint == 11
    assert [item.field.value for item in result.hard_exceeded] == [
        "completed_epochs"
    ]
    assert [item.field.value for item in result.advisory_exceeded] == [
        "peak_device_memory_bytes"
    ]


def test_resource_budget_rejects_non_positive_or_non_integer_values() -> None:
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValidationError):
            ResourceLimit(
                field="completed_epochs",
                value=invalid,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            )
