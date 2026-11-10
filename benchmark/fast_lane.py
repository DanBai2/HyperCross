"""Frozen execution contracts for the training-first benchmark fast lane.

The templates in ``configs/fast_lane`` contain no dataset paths.  This module
binds them to the audited TCFM/CUDA environment and produces the existing
``FrozenRunConfig``/``FrozenRunIdentity`` pair before a runner allocates or
starts a training job.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from pydantic import Field, StrictInt, field_validator, model_validator

from benchmark.registry.records import FrozenRunIdentity
from benchmark.schema.base import ContentHash, LimitEnforcement, MetricDirection
from benchmark.schema.canonical import hash_object
from benchmark.schema.configuration import (
    CheckpointRetentionPolicy,
    ExecutionMode,
    FrameworkDeterminismSettings,
    FrozenRunConfig,
    NumericPrecision,
)
from benchmark.schema.environment import EnvironmentLock, FrozenSchemaModel
from benchmark.schema.records import ResourceBudget
from benchmark.schema.resources import ResourceField, ResourceUnit


_AUDITED_ENVIRONMENT_LOCK = (
    "a381dbcd2b3aaf8723b92ce4ed7f44c95d039ff46c04e8d343b82c88eb36a275"
)
_EXPECTED_WORKSPACE = PureWindowsPath(r"G:\Datasets\HyperCross")
_EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5080"
_REQUIRED_RECOVERY_STATES = ("model", "optimizer", "scheduler", "stage")
_READ_ONLY_ROOTS = ("runs", "src")
_ALLOWED_SHARED_WRITE_ROOT = "artifacts"
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXPECTED_UNITS = {
    ResourceField.COMPLETED_EPOCHS: ResourceUnit.COUNT,
    ResourceField.COMPLETED_TRAINING_STEPS: ResourceUnit.COUNT,
    ResourceField.WALL_CLOCK_DURATION_MS: ResourceUnit.MILLISECONDS,
    ResourceField.PEAK_DEVICE_MEMORY_BYTES: ResourceUnit.BYTES,
    ResourceField.PARAMETER_COUNT: ResourceUnit.COUNT,
    ResourceField.INPUT_VOLUME_BYTES: ResourceUnit.BYTES,
}


class FastLaneConfigurationError(ValueError):
    """A fast-lane contract failed before scheduling or state restoration."""


class FastLaneStage(str, Enum):
    SMOKE = "smoke"
    BASELINE = "baseline"
    SCREENING = "screening"
    STRICT = "strict"
    OFFICIAL = "official"


_STAGE_OUTPUT_ROOTS = {
    FastLaneStage.SMOKE: "runs_v2/smoke",
    FastLaneStage.BASELINE: "runs_v2/baselines",
    FastLaneStage.SCREENING: "runs_v2/screening",
    FastLaneStage.STRICT: "runs_v2/strict",
    FastLaneStage.OFFICIAL: "runs_v2/official",
}


class FastLaneStageTemplate(FrozenSchemaModel):
    """Hash-bound policy shared by all jobs in one fast-lane stage."""

    schema_version: str = Field(default="1.0", min_length=1)
    stage: FastLaneStage
    conda_environment: str = Field(min_length=1)
    environment_lock_artifact: str = Field(min_length=1)
    environment_lock_hash: ContentHash
    device: str = Field(min_length=1)
    expected_device_name: str = Field(min_length=1)
    visible_cuda_device_count: StrictInt = Field(gt=0)
    process_count: StrictInt = Field(gt=0)
    numeric_precision: NumericPrecision
    execution_mode: ExecutionMode
    framework_settings: FrameworkDeterminismSettings
    seed_set: tuple[StrictInt, ...] = Field(min_length=1)
    resource_budget: ResourceBudget
    checkpoint_interval_epochs: StrictInt = Field(gt=0)
    checkpoint_retention: CheckpointRetentionPolicy
    output_root: str = Field(min_length=1)
    shared_artifact_root: str = Field(min_length=1)
    read_only_roots: tuple[str, ...] = Field(min_length=1)
    recovery_state_names: tuple[str, ...] = Field(min_length=1)
    persist_frozen_identity_before_training: bool
    template_hash: ContentHash

    @classmethod
    def create(
        cls,
        *,
        stage: FastLaneStage,
        seed_set: tuple[int, ...],
        resource_budget: ResourceBudget,
        checkpoint_interval_epochs: int = 1,
        numeric_precision: NumericPrecision = NumericPrecision.FLOAT32,
        execution_mode: ExecutionMode = ExecutionMode.DETERMINISTIC,
    ) -> "FastLaneStageTemplate":
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "stage": stage,
            "conda_environment": "TCFM",
            "environment_lock_artifact": (
                "artifacts/audit/training_fast_lane_audit.json"
            ),
            "environment_lock_hash": _AUDITED_ENVIRONMENT_LOCK,
            "device": "cuda:0",
            "expected_device_name": _EXPECTED_DEVICE_NAME,
            "visible_cuda_device_count": 1,
            "process_count": 1,
            "numeric_precision": numeric_precision,
            "execution_mode": execution_mode,
            "framework_settings": FrameworkDeterminismSettings.for_mode(
                execution_mode
            ),
            "seed_set": seed_set,
            "resource_budget": resource_budget,
            "checkpoint_interval_epochs": checkpoint_interval_epochs,
            "checkpoint_retention": CheckpointRetentionPolicy(),
            "output_root": _STAGE_OUTPUT_ROOTS[stage],
            "shared_artifact_root": _ALLOWED_SHARED_WRITE_ROOT,
            "read_only_roots": _READ_ONLY_ROOTS,
            "recovery_state_names": _REQUIRED_RECOVERY_STATES,
            "persist_frozen_identity_before_training": True,
        }
        payload["template_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    @field_validator(
        "environment_lock_artifact",
        "output_root",
        "shared_artifact_root",
    )
    @classmethod
    def validate_relative_paths(cls, value: str) -> str:
        return _normalized_relative_path(value)

    @field_validator("read_only_roots")
    @classmethod
    def validate_read_only_roots(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_normalized_relative_path(value) for value in values)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("read_only_roots must be unique and canonically sorted")
        return normalized

    @field_validator("seed_set")
    @classmethod
    def validate_seed_set(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("seed_set must be unique and canonically sorted")
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("seed_set values must be non-negative integers")
        return values

    @model_validator(mode="after")
    def validate_contract(self) -> "FastLaneStageTemplate":
        exact_values = {
            "conda_environment": (self.conda_environment, "TCFM"),
            "device": (self.device, "cuda:0"),
            "expected_device_name": (
                self.expected_device_name,
                _EXPECTED_DEVICE_NAME,
            ),
            "visible_cuda_device_count": (self.visible_cuda_device_count, 1),
            "process_count": (self.process_count, 1),
            "environment_lock_hash": (
                self.environment_lock_hash,
                _AUDITED_ENVIRONMENT_LOCK,
            ),
            "output_root": (self.output_root, _STAGE_OUTPUT_ROOTS[self.stage]),
            "shared_artifact_root": (
                self.shared_artifact_root,
                _ALLOWED_SHARED_WRITE_ROOT,
            ),
            "read_only_roots": (self.read_only_roots, _READ_ONLY_ROOTS),
            "recovery_state_names": (
                self.recovery_state_names,
                _REQUIRED_RECOVERY_STATES,
            ),
            "persist_frozen_identity_before_training": (
                self.persist_frozen_identity_before_training,
                True,
            ),
            # The current TrainingEngine publishes recovery state at every epoch.
            "checkpoint_interval_epochs": (self.checkpoint_interval_epochs, 1),
            "seed_set": (self.seed_set, _stage_seed_set(self.stage)),
        }
        mismatched = [
            name for name, (observed, expected) in exact_values.items()
            if observed != expected
        ]
        if mismatched:
            raise ValueError(
                "fast-lane fixed fields differ from the audited contract: "
                + ", ".join(mismatched)
            )
        if self.numeric_precision is not NumericPrecision.FLOAT32:
            raise ValueError("fast-lane precision is frozen to float32")
        if self.execution_mode is not ExecutionMode.DETERMINISTIC:
            raise ValueError("fast-lane execution mode is frozen to deterministic")
        if self.framework_settings != FrameworkDeterminismSettings.for_mode(
            self.execution_mode
        ):
            raise ValueError("framework settings differ from the frozen execution mode")
        if self.checkpoint_retention != CheckpointRetentionPolicy():
            raise ValueError("all selected, last, and recovery checkpoints must be retained")
        self._validate_budget()
        if self.template_hash != hash_object(self.hash_payload()):
            raise ValueError("template_hash does not match the fast-lane template")
        return self

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"template_hash"})

    def _validate_budget(self) -> None:
        enforcement = {limit.enforcement for limit in self.resource_budget.limits}
        if enforcement != {LimitEnforcement.HARD, LimitEnforcement.ADVISORY}:
            raise ValueError("resource budget must contain hard and advisory limits")
        hard_fields: set[ResourceField] = set()
        for limit in self.resource_budget.limits:
            try:
                field = ResourceField(limit.field)
                unit = ResourceUnit(limit.unit)
            except ValueError as exc:
                raise ValueError("resource budget uses an unsupported field or unit") from exc
            if unit is not _EXPECTED_UNITS[field]:
                raise ValueError(
                    f"unit for {field.value} must be {_EXPECTED_UNITS[field].value}"
                )
            if limit.enforcement is LimitEnforcement.HARD:
                hard_fields.add(field)
        required_hard = {
            ResourceField.COMPLETED_EPOCHS,
            ResourceField.COMPLETED_TRAINING_STEPS,
            ResourceField.WALL_CLOCK_DURATION_MS,
            ResourceField.PEAK_DEVICE_MEMORY_BYTES,
        }
        if not required_hard.issubset(hard_fields):
            missing = sorted(field.value for field in required_hard - hard_fields)
            raise ValueError("resource budget lacks hard limits: " + ", ".join(missing))

    def validate_environment(
        self,
        environment_lock: EnvironmentLock,
        *,
        conda_environment: str,
    ) -> None:
        """Reject scheduling if the active environment differs from the audit."""

        mismatches: list[str] = []
        if conda_environment != self.conda_environment:
            mismatches.append("conda_environment")
        if environment_lock.lock_hash != self.environment_lock_hash:
            mismatches.append("environment_lock_hash")
        if not environment_lock.cuda_available:
            mismatches.append("cuda_available")
        if environment_lock.cuda_device_count != self.visible_cuda_device_count:
            mismatches.append("cuda_device_count")
        expected_identity = f"cuda:0:{self.expected_device_name}"
        if environment_lock.device_identity != expected_identity:
            mismatches.append("device_identity")
        if mismatches:
            raise FastLaneConfigurationError(
                "active environment differs from frozen fast-lane contract: "
                + ", ".join(mismatches)
            )

    def validate_workspace_root(self, workspace_root: str | Path) -> Path:
        """Require the sole writable workspace to be G:\\Datasets\\HyperCross."""

        value = str(workspace_root)
        candidate = PureWindowsPath(value)
        if (
            not candidate.is_absolute()
            or candidate.drive.casefold() != "g:"
            or tuple(part.casefold() for part in candidate.parts[1:])
            != tuple(part.casefold() for part in _EXPECTED_WORKSPACE.parts[1:])
        ):
            raise FastLaneConfigurationError(
                r"workspace root must be G:\Datasets\HyperCross"
            )
        return Path(value)

    def validate_write_path(self, relative_path: str) -> str:
        """Allow writes only below this stage output or the shared artifact root."""

        normalized = _normalized_relative_path(relative_path)
        if not (
            _is_at_or_below(normalized, self.output_root)
            or _is_at_or_below(normalized, self.shared_artifact_root)
        ):
            raise FastLaneConfigurationError(
                "write path must stay under the stage output root or artifacts"
            )
        if any(_is_at_or_below(normalized, root) for root in self.read_only_roots):
            raise FastLaneConfigurationError("src and legacy runs are read-only")
        return normalized

    def validate_recovery_source(self, path: str) -> str:
        """Permit resume only from this stage's workspace-relative runs_v2 tree."""

        normalized = _normalized_relative_path(path)
        if not _is_at_or_below(normalized, self.output_root):
            raise FastLaneConfigurationError(
                "recovery source must be workspace-relative and inside the stage output root"
            )
        return normalized

    def freeze_job(
        self,
        *,
        job_name: str,
        dataset_hash: str,
        protocol_hash: str,
        protocol_key: str,
        model_key: str,
        fold_id: str,
        seed: int,
        input_volume_bytes: int,
        selection_metric_direction: MetricDirection,
        parameters: Mapping[str, Any] | None = None,
    ) -> "FrozenFastLaneJob":
        """Bind a stage template to one immutable, resumable job identity."""

        if not _JOB_NAME.fullmatch(job_name):
            raise FastLaneConfigurationError(
                "job_name must use only letters, digits, dot, underscore, or hyphen"
            )
        if seed not in self.seed_set:
            raise FastLaneConfigurationError("job seed is outside the frozen seed_set")
        supplied = dict(parameters or {})
        reserved = {
            "checkpoint_interval_epochs",
            "conda_environment",
            "device",
            "epochs",
            "fast_lane_stage",
            "output_root",
            "process_count",
            "selection_metric_direction",
        }
        conflict = sorted(reserved.intersection(supplied))
        if conflict:
            raise FastLaneConfigurationError(
                "job parameters cannot override frozen fields: " + ", ".join(conflict)
            )
        epoch_limit = next(
            limit.value
            for limit in self.resource_budget.limits
            if limit.field == ResourceField.COMPLETED_EPOCHS.value
            and limit.enforcement is LimitEnforcement.HARD
        )
        supplied.update(
            {
                "checkpoint_interval_epochs": self.checkpoint_interval_epochs,
                "conda_environment": self.conda_environment,
                "device": self.device,
                "epochs": epoch_limit,
                "fast_lane_stage": self.stage.value,
                "output_root": self.output_root,
                "process_count": self.process_count,
                "selection_metric_direction": selection_metric_direction.value,
            }
        )
        config = FrozenRunConfig.create(
            model_key=model_key,
            protocol_key=protocol_key,
            execution_mode=self.execution_mode,
            numeric_precision=self.numeric_precision,
            framework_settings=self.framework_settings,
            input_volume_bytes=input_volume_bytes,
            checkpoint_retention=self.checkpoint_retention,
            resource_budget=self.resource_budget,
            parameters=supplied,
        )
        identity = FrozenRunIdentity(
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            model_key=model_key,
            config_hash=config.config_hash,
            fold_id=fold_id,
            seed=seed,
            environment_lock_hash=self.environment_lock_hash,
            environment_compatibility_policy="exact",
        )
        job_root = f"{self.output_root}/jobs/{job_name}"
        return FrozenFastLaneJob(
            stage=self.stage,
            template_hash=self.template_hash,
            output_root=self.output_root,
            frozen_config=config,
            frozen_identity=identity,
            frozen_identity_relative_path=f"{job_root}/frozen_run_identity.json",
            recovery_directory=f"{job_root}/recovery",
            recovery_state_names=self.recovery_state_names,
            persist_identity_before_training=self.persist_frozen_identity_before_training,
        )


class FrozenFastLaneJob(FrozenSchemaModel):
    """Complete pre-training handoff for one fast-lane runner job."""

    schema_version: str = Field(default="1.0", min_length=1)
    stage: FastLaneStage
    template_hash: ContentHash
    output_root: str
    frozen_config: FrozenRunConfig
    frozen_identity: FrozenRunIdentity
    frozen_identity_relative_path: str
    recovery_directory: str
    recovery_state_names: tuple[str, ...]
    persist_identity_before_training: bool

    @field_validator(
        "output_root", "frozen_identity_relative_path", "recovery_directory"
    )
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _normalized_relative_path(value)

    @model_validator(mode="after")
    def validate_job(self) -> "FrozenFastLaneJob":
        expected_root = _STAGE_OUTPUT_ROOTS[self.stage]
        if self.output_root != expected_root:
            raise ValueError("job output_root differs from its fast-lane stage")
        for path in (self.frozen_identity_relative_path, self.recovery_directory):
            if not _is_at_or_below(path, self.output_root):
                raise ValueError("job artifacts must remain under the stage output root")
        if self.frozen_identity.config_hash != self.frozen_config.config_hash:
            raise ValueError("Frozen_Run_Identity config hash does not match config")
        if self.frozen_identity.model_key != self.frozen_config.model_key:
            raise ValueError("Frozen_Run_Identity model key does not match config")
        if self.frozen_identity.environment_lock_hash != _AUDITED_ENVIRONMENT_LOCK:
            raise ValueError("job is not bound to the audited Environment_Lock")
        if self.frozen_identity.seed not in _stage_seed_set(self.stage):
            raise ValueError("job seed is outside the frozen stage seed_set")
        if self.recovery_state_names != _REQUIRED_RECOVERY_STATES:
            raise ValueError("recovery must include model, optimizer, scheduler, and stage")
        if not self.persist_identity_before_training:
            raise ValueError("Frozen_Run_Identity must be persisted before training")
        return self


def load_fast_lane_template(path: str | Path) -> FastLaneStageTemplate:
    """Read and validate one immutable JSON template without starting training."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FastLaneConfigurationError(
            f"cannot read fast-lane template {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FastLaneConfigurationError("fast-lane template must contain a JSON object")
    return FastLaneStageTemplate.model_validate(payload)


def _stage_seed_set(stage: FastLaneStage) -> tuple[int, ...]:
    if stage is FastLaneStage.STRICT:
        return (7, 42, 123)
    return (42,)


def _normalized_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("path must be a non-empty trimmed string")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        if windows.drive.casefold() == "f:":
            raise FastLaneConfigurationError("F: paths are forbidden by fast-lane policy")
        raise FastLaneConfigurationError("fast-lane paths must be workspace-relative")
    if ".." in windows.parts or ".." in posix.parts:
        raise FastLaneConfigurationError("fast-lane paths must not traverse parents")
    normalized = PurePosixPath(value.replace("\\", "/")).as_posix()
    if normalized in {"", "."}:
        raise ValueError("path must identify a workspace-relative target")
    return normalized


def _is_at_or_below(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


__all__ = [
    "FastLaneConfigurationError",
    "FastLaneStage",
    "FastLaneStageTemplate",
    "FrozenFastLaneJob",
    "load_fast_lane_template",
]
