"""Runtime measurement and resource-budget enforcement records."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from .base import LimitEnforcement
from .environment import FrozenSchemaModel
from .records import ResourceBudget, ResourceLimit


class ResourceField(str, Enum):
    COMPLETED_EPOCHS = "completed_epochs"
    COMPLETED_TRAINING_STEPS = "completed_training_steps"
    WALL_CLOCK_DURATION_MS = "wall_clock_duration_ms"
    PEAK_DEVICE_MEMORY_BYTES = "peak_device_memory_bytes"
    PARAMETER_COUNT = "parameter_count"
    INPUT_VOLUME_BYTES = "input_volume_bytes"


class ResourceUnit(str, Enum):
    COUNT = "count"
    MILLISECONDS = "milliseconds"
    BYTES = "bytes"


_EXPECTED_UNITS = {
    ResourceField.COMPLETED_EPOCHS: ResourceUnit.COUNT,
    ResourceField.COMPLETED_TRAINING_STEPS: ResourceUnit.COUNT,
    ResourceField.WALL_CLOCK_DURATION_MS: ResourceUnit.MILLISECONDS,
    ResourceField.PEAK_DEVICE_MEMORY_BYTES: ResourceUnit.BYTES,
    ResourceField.PARAMETER_COUNT: ResourceUnit.COUNT,
    ResourceField.INPUT_VOLUME_BYTES: ResourceUnit.BYTES,
}


class RuntimeMeasurements(FrozenSchemaModel):
    """Complete resource measurements persisted for an executing run."""

    completed_epochs: int = Field(strict=True, ge=0)
    completed_training_steps: int = Field(strict=True, ge=0)
    wall_clock_duration_ms: int = Field(strict=True, ge=0)
    peak_device_memory_bytes: int = Field(strict=True, ge=0)
    device_identity: str = Field(min_length=1)
    numeric_precision: str = Field(min_length=1)
    parameter_count: int = Field(strict=True, ge=0)
    input_volume_bytes: int = Field(strict=True, ge=0)

    def value_for(self, field: ResourceField) -> int:
        return int(getattr(self, field.value))


class BudgetObservation(FrozenSchemaModel):
    """Machine-readable evidence for one exceeded limit."""

    field: ResourceField
    unit: ResourceUnit
    limit: int = Field(strict=True, gt=0)
    observed: int = Field(strict=True, ge=0)
    enforcement: LimitEnforcement

    @model_validator(mode="after")
    def validate_exceeded(self) -> "BudgetObservation":
        if self.observed <= self.limit:
            raise ValueError("observed value must exceed the limit")
        return self


class BudgetAssessment(FrozenSchemaModel):
    """Checkpoint-boundary decision produced by budget enforcement."""

    hard_exceeded: tuple[BudgetObservation, ...] = ()
    advisory_exceeded: tuple[BudgetObservation, ...] = ()
    stop_at_next_checkpoint_boundary: bool
    stopping_checkpoint: Optional[int] = Field(default=None, strict=True, ge=0)

    @model_validator(mode="after")
    def validate_decision(self) -> "BudgetAssessment":
        if self.stop_at_next_checkpoint_boundary != bool(self.hard_exceeded):
            raise ValueError("hard-limit observations must determine stop behavior")
        if self.hard_exceeded and self.stopping_checkpoint is None:
            raise ValueError("hard-limit stop must record the checkpoint boundary")
        if not self.hard_exceeded and self.stopping_checkpoint is not None:
            raise ValueError("stopping_checkpoint is only valid for a hard-limit stop")
        if any(item.enforcement is not LimitEnforcement.HARD for item in self.hard_exceeded):
            raise ValueError("hard_exceeded may contain only hard limits")
        if any(
            item.enforcement is not LimitEnforcement.ADVISORY
            for item in self.advisory_exceeded
        ):
            raise ValueError("advisory_exceeded may contain only advisory limits")
        return self


def assess_resource_budget(
    budget: ResourceBudget,
    measurements: RuntimeMeasurements,
    *,
    checkpoint_boundary: int,
) -> BudgetAssessment:
    """Evaluate hard stops and advisory warnings at a checkpoint boundary."""

    if isinstance(checkpoint_boundary, bool) or not isinstance(checkpoint_boundary, int):
        raise TypeError("checkpoint_boundary must be an integer")
    if checkpoint_boundary < 0:
        raise ValueError("checkpoint_boundary must be non-negative")

    hard: list[BudgetObservation] = []
    advisory: list[BudgetObservation] = []
    for limit in budget.limits:
        try:
            field = ResourceField(limit.field)
        except ValueError as exc:
            raise ValueError(f"unsupported resource budget field: {limit.field}") from exc
        expected_unit = _EXPECTED_UNITS[field]
        try:
            unit = ResourceUnit(limit.unit)
        except ValueError as exc:
            raise ValueError(f"unsupported resource unit: {limit.unit}") from exc
        if unit is not expected_unit:
            raise ValueError(
                f"unit for {field.value} must be {expected_unit.value}, got {unit.value}"
            )

        observed = measurements.value_for(field)
        if observed <= limit.value:
            continue
        observation = BudgetObservation(
            field=field,
            unit=unit,
            limit=limit.value,
            observed=observed,
            enforcement=limit.enforcement,
        )
        if limit.enforcement is LimitEnforcement.HARD:
            hard.append(observation)
        else:
            advisory.append(observation)

    return BudgetAssessment(
        hard_exceeded=tuple(hard),
        advisory_exceeded=tuple(advisory),
        stop_at_next_checkpoint_boundary=bool(hard),
        stopping_checkpoint=checkpoint_boundary if hard else None,
    )


__all__ = [
    "BudgetAssessment",
    "BudgetObservation",
    "ResourceBudget",
    "ResourceField",
    "ResourceLimit",
    "ResourceUnit",
    "RuntimeMeasurements",
    "assess_resource_budget",
]
