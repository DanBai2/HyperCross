"""Immutable execution configuration records."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Mapping

from pydantic import Field, field_validator, model_validator

from .base import ContentHash, ModelKey, ProtocolKey
from .canonical import canonical_serialize, hash_object
from .environment import FrozenSchemaModel
from .records import ResourceBudget


class ExecutionMode(str, Enum):
    DETERMINISTIC = "deterministic"
    PERFORMANCE = "performance"


class NumericPrecision(str, Enum):
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    TF32 = "tf32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    MIXED_FLOAT16 = "mixed_float16"
    MIXED_BFLOAT16 = "mixed_bfloat16"


class FrameworkDeterminismSettings(FrozenSchemaModel):
    """Framework switches needed to reproduce execution behavior."""

    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    allow_tf32: bool

    @classmethod
    def for_mode(cls, mode: ExecutionMode) -> "FrameworkDeterminismSettings":
        if mode is ExecutionMode.DETERMINISTIC:
            return cls(
                deterministic_algorithms=True,
                cudnn_deterministic=True,
                cudnn_benchmark=False,
                allow_tf32=False,
            )
        return cls(
            deterministic_algorithms=False,
            cudnn_deterministic=False,
            cudnn_benchmark=True,
            allow_tf32=True,
        )


class CheckpointRetentionPolicy(FrozenSchemaModel):
    """Checkpoint classes retained by a run."""

    preserve_selected: bool = True
    preserve_last: bool = True
    preserve_recovery_state: bool = True


class FrozenRunConfig(FrozenSchemaModel):
    """Hash-bound configuration frozen before benchmark scheduling."""

    schema_version: str = Field(default="1.0", min_length=1)
    model_key: ModelKey
    protocol_key: ProtocolKey
    execution_mode: ExecutionMode
    numeric_precision: NumericPrecision
    framework_settings: FrameworkDeterminismSettings
    input_volume_bytes: int = Field(strict=True, ge=0)
    checkpoint_retention: CheckpointRetentionPolicy
    resource_budget: ResourceBudget
    parameters_canonical_serialization: str = Field(min_length=2)
    config_hash: ContentHash

    @classmethod
    def create(
        cls,
        *,
        model_key: str,
        protocol_key: str,
        execution_mode: ExecutionMode,
        numeric_precision: NumericPrecision,
        input_volume_bytes: int,
        resource_budget: ResourceBudget,
        parameters: Mapping[str, Any],
        framework_settings: FrameworkDeterminismSettings | None = None,
        checkpoint_retention: CheckpointRetentionPolicy | None = None,
        schema_version: str = "1.0",
    ) -> "FrozenRunConfig":
        """Canonicalize free-form parameters and freeze the complete config."""

        parameters_serialization = canonical_serialize(dict(parameters)).decode("utf-8")
        payload: dict[str, Any] = {
            "schema_version": schema_version,
            "model_key": model_key,
            "protocol_key": protocol_key,
            "execution_mode": execution_mode,
            "numeric_precision": numeric_precision,
            "framework_settings": framework_settings
            or FrameworkDeterminismSettings.for_mode(execution_mode),
            "input_volume_bytes": input_volume_bytes,
            "checkpoint_retention": checkpoint_retention
            or CheckpointRetentionPolicy(),
            "resource_budget": resource_budget,
            "parameters_canonical_serialization": parameters_serialization,
        }
        payload["config_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    @field_validator("parameters_canonical_serialization")
    @classmethod
    def validate_parameter_serialization(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "parameters_canonical_serialization must contain canonical JSON bytes"
            ) from exc
        compact = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if value != compact:
            raise ValueError("parameter serialization is not canonical")
        return value

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"config_hash"})

    @model_validator(mode="after")
    def validate_frozen_config(self) -> "FrozenRunConfig":
        if self.execution_mode is ExecutionMode.DETERMINISTIC:
            settings = self.framework_settings
            if not settings.deterministic_algorithms:
                raise ValueError(
                    "deterministic mode requires deterministic_algorithms"
                )
            if not settings.cudnn_deterministic or settings.cudnn_benchmark:
                raise ValueError(
                    "deterministic mode requires deterministic cuDNN without benchmarking"
                )
        if self.config_hash != hash_object(self.hash_payload()):
            raise ValueError("config_hash does not match the frozen configuration")
        return self
