"""Property 34: resource budget validity and enforcement.

**Validates: Requirements 14.1-14.18, 14.29-14.33**
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hypothesis import given, strategies as st

from benchmark.schema import (
    ExecutionMode,
    FrozenRunConfig,
    LimitEnforcement,
    NumericPrecision,
    ResourceBudget,
    ResourceField,
    ResourceLimit,
    ResourceUnit,
    RuntimeMeasurements,
    assess_resource_budget,
    validate_schema,
)


_EXPECTED_UNITS = {
    ResourceField.COMPLETED_EPOCHS: ResourceUnit.COUNT,
    ResourceField.COMPLETED_TRAINING_STEPS: ResourceUnit.COUNT,
    ResourceField.WALL_CLOCK_DURATION_MS: ResourceUnit.MILLISECONDS,
    ResourceField.PEAK_DEVICE_MEMORY_BYTES: ResourceUnit.BYTES,
    ResourceField.PARAMETER_COUNT: ResourceUnit.COUNT,
    ResourceField.INPUT_VOLUME_BYTES: ResourceUnit.BYTES,
}
_MEASUREMENT_FIELDS = {
    "completed_epochs",
    "completed_training_steps",
    "wall_clock_duration_ms",
    "peak_device_memory_bytes",
    "device_identity",
    "numeric_precision",
    "parameter_count",
    "input_volume_bytes",
}


@st.composite
def resource_limits(draw: st.DrawFn) -> tuple[ResourceLimit, ...]:
    fields = draw(
        st.lists(
            st.sampled_from(list(ResourceField)),
            min_size=1,
            max_size=len(ResourceField),
            unique=True,
        )
    )
    limits: list[ResourceLimit] = []
    for field in fields:
        limits.append(
            ResourceLimit(
                field=field.value,
                value=draw(st.integers(min_value=1, max_value=10**12)),
                unit=_EXPECTED_UNITS[field].value,
                enforcement=draw(st.sampled_from(list(LimitEnforcement))),
            )
        )
    return tuple(limits)


@st.composite
def runtime_measurements(draw: st.DrawFn) -> RuntimeMeasurements:
    measured_values = {
        field.value: draw(st.integers(min_value=0, max_value=10**12))
        for field in ResourceField
    }
    return RuntimeMeasurements(
        **measured_values,
        device_identity=draw(
            st.sampled_from(
                ["cpu", "cuda:0:NVIDIA GeForce RTX 5080", "cuda:1:test-device"]
            )
        ),
        numeric_precision=draw(st.sampled_from([item.value for item in NumericPrecision])),
    )


_INVALID_LIMIT_VALUES = st.one_of(
    st.integers(max_value=0),
    st.floats(allow_nan=False, allow_infinity=False),
    st.sampled_from([float("nan"), float("inf"), float("-inf"), True, False]),
)


@given(invalid_value=_INVALID_LIMIT_VALUES)
def test_invalid_budget_values_are_rejected_before_scheduling(
    invalid_value: object,
) -> None:
    """Invalid values never cross the validation boundary into scheduling."""

    scheduling_calls = 0
    candidate: Mapping[str, Any] = {
        "limits": [
            {
                "field": ResourceField.COMPLETED_EPOCHS.value,
                "value": invalid_value,
                "unit": ResourceUnit.COUNT.value,
                "enforcement": LimitEnforcement.HARD.value,
            }
        ]
    }

    budget, report = validate_schema(ResourceBudget, candidate)
    if budget is not None:
        scheduling_calls += 1

    assert budget is None
    assert report.passes is False
    assert scheduling_calls == 0
    assert any(issue.field_path == ("limits", "0", "value") for issue in report.errors)
    assert report.machine_details()["passes"] is False


@given(
    limits=resource_limits(),
    measurements=runtime_measurements(),
    checkpoint_boundary=st.integers(min_value=0, max_value=10**9),
)
def test_valid_budgets_partition_hard_stops_and_advisory_warnings(
    limits: tuple[ResourceLimit, ...],
    measurements: RuntimeMeasurements,
    checkpoint_boundary: int,
) -> None:
    """Every exceeded limit has the specified checkpoint-boundary behavior."""

    budget = ResourceBudget(limits=limits)
    assessment = assess_resource_budget(
        budget,
        measurements,
        checkpoint_boundary=checkpoint_boundary,
    )

    expected_hard = {
        limit.field
        for limit in limits
        if limit.enforcement is LimitEnforcement.HARD
        and measurements.value_for(ResourceField(limit.field)) > limit.value
    }
    expected_advisory = {
        limit.field
        for limit in limits
        if limit.enforcement is LimitEnforcement.ADVISORY
        and measurements.value_for(ResourceField(limit.field)) > limit.value
    }

    assert {item.field.value for item in assessment.hard_exceeded} == expected_hard
    assert {
        item.field.value for item in assessment.advisory_exceeded
    } == expected_advisory
    assert assessment.stop_at_next_checkpoint_boundary is bool(expected_hard)
    assert assessment.stopping_checkpoint == (
        checkpoint_boundary if expected_hard else None
    )
    assert all(
        item.enforcement is LimitEnforcement.ADVISORY
        for item in assessment.advisory_exceeded
    )

    persisted_measurements = measurements.model_dump(mode="json")
    assert set(persisted_measurements) == _MEASUREMENT_FIELDS
    for field in ResourceField:
        assert persisted_measurements[field.value] == measurements.value_for(field)
    assert persisted_measurements["device_identity"]
    assert persisted_measurements["numeric_precision"]


@given(
    limits=resource_limits(),
    execution_mode=st.sampled_from(list(ExecutionMode)),
    numeric_precision=st.sampled_from(list(NumericPrecision)),
    input_volume_bytes=st.integers(min_value=0, max_value=10**12),
)
def test_frozen_run_config_records_budget_retention_and_execution_policy(
    limits: tuple[ResourceLimit, ...],
    execution_mode: ExecutionMode,
    numeric_precision: NumericPrecision,
    input_volume_bytes: int,
) -> None:
    """A schedulable run preserves its budget and required execution policy."""

    budget = ResourceBudget(limits=limits)
    config = FrozenRunConfig.create(
        model_key="property34.model",
        protocol_key="property34.protocol.v1",
        execution_mode=execution_mode,
        numeric_precision=numeric_precision,
        input_volume_bytes=input_volume_bytes,
        resource_budget=budget,
        parameters={"batch_size": 1, "property": 34},
    )

    assert config.resource_budget == budget
    assert config.input_volume_bytes == input_volume_bytes
    assert config.execution_mode is execution_mode
    assert config.numeric_precision is numeric_precision
    assert config.checkpoint_retention.preserve_selected is True
    assert config.checkpoint_retention.preserve_last is True
    assert config.checkpoint_retention.preserve_recovery_state is True

    settings = config.framework_settings
    if execution_mode is ExecutionMode.DETERMINISTIC:
        assert settings.deterministic_algorithms is True
        assert settings.cudnn_deterministic is True
        assert settings.cudnn_benchmark is False
        assert settings.allow_tf32 is False
    else:
        assert settings.deterministic_algorithms is False
        assert settings.cudnn_deterministic is False
        assert settings.cudnn_benchmark is True
        assert settings.allow_tf32 is True
