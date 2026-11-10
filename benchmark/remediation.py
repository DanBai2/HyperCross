"""Versioned, hash-bound execution contracts for task 24 remediation jobs.

This namespace is intentionally independent from ``configs/fast_lane``.  A
precision profile is part of both the canonical template hash and every frozen
job configuration, so deterministic float32 and performance BF16 jobs cannot
share configuration identity or recovery state.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from pydantic import Field, StrictInt, ValidationError, field_validator, model_validator

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
_EXPECTED_DEVICE = "cuda:0"
_EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5080"
_REQUIRED_RECOVERY_STATES = ("model", "optimizer", "scheduler", "stage")
_READ_ONLY_ROOTS = (
    "caches",
    "checkpoints",
    "configs/fast_lane",
    "datasets",
    "runs",
    "src",
)
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXPECTED_UNITS = {
    ResourceField.COMPLETED_EPOCHS: ResourceUnit.COUNT,
    ResourceField.COMPLETED_TRAINING_STEPS: ResourceUnit.COUNT,
    ResourceField.WALL_CLOCK_DURATION_MS: ResourceUnit.MILLISECONDS,
    ResourceField.PEAK_DEVICE_MEMORY_BYTES: ResourceUnit.BYTES,
    ResourceField.PARAMETER_COUNT: ResourceUnit.COUNT,
    ResourceField.INPUT_VOLUME_BYTES: ResourceUnit.BYTES,
}
_REQUIRED_HARD_LIMITS = {
    ResourceField.COMPLETED_EPOCHS,
    ResourceField.COMPLETED_TRAINING_STEPS,
    ResourceField.WALL_CLOCK_DURATION_MS,
    ResourceField.PEAK_DEVICE_MEMORY_BYTES,
}


class RemediationConfigurationError(ValueError):
    """A remediation contract failed before scheduling or restoration."""


class RemediationProfile(str, Enum):
    """Precision identities approved for the task 24 remediation wave."""

    DETERMINISTIC_FLOAT32 = "deterministic-float32"
    PERFORMANCE_BF16 = "performance-bf16"


_PROFILE_OUTPUT_ROOTS = {
    RemediationProfile.DETERMINISTIC_FLOAT32: (
        "runs_v2/remediation/v1/deterministic-float32"
    ),
    RemediationProfile.PERFORMANCE_BF16: (
        "runs_v2/remediation/v1/performance-bf16"
    ),
}
_PROFILE_ARTIFACT_ROOTS = {
    profile: f"artifacts/remediation/v1/{profile.value}"
    for profile in RemediationProfile
}


class AutocastPolicy(FrozenSchemaModel):
    """Exact autocast behavior included in configuration identity."""

    enabled: bool
    device_type: str = Field(min_length=1)
    dtype: NumericPrecision
    cache_enabled: bool


class GradientAccumulationPolicy(FrozenSchemaModel):
    """Exact microbatch-to-optimizer-step policy."""

    accumulation_steps: StrictInt = Field(gt=0)
    divide_loss_by_accumulation_steps: bool
    flush_final_partial_accumulation: bool


class InputVolumeAccountingPolicy(FrozenSchemaModel):
    """How runtime input bytes are measured against the declared budget."""

    method: str = Field(min_length=1)
    count_repeated_reads: bool
    include_preprocessing_reads: bool
    budget_field: str = Field(min_length=1)
    require_runtime_measurement: bool


class RemediationPlanningContext(FrozenSchemaModel):
    """Observed scheduling context that must exactly match a template."""

    conda_environment: str = Field(min_length=1)
    environment_lock: EnvironmentLock
    device: str = Field(min_length=1)
    process_count: StrictInt = Field(gt=0)
    numeric_precision: NumericPrecision
    execution_mode: ExecutionMode
    autocast_policy: AutocastPolicy
    gradient_accumulation: GradientAccumulationPolicy


class RemediationTemplate(FrozenSchemaModel):
    """One immutable, explicitly versioned remediation precision template."""

    schema_version: str = Field(min_length=1)
    configuration_namespace: str = Field(min_length=1)
    template_version: str = Field(min_length=1)
    profile: RemediationProfile
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
    autocast_policy: AutocastPolicy
    gradient_accumulation: GradientAccumulationPolicy
    seed_set: tuple[StrictInt, ...] = Field(min_length=1)
    resource_budget: ResourceBudget
    checkpoint_interval_epochs: StrictInt = Field(gt=0)
    checkpoint_retention: CheckpointRetentionPolicy
    input_volume_accounting: InputVolumeAccountingPolicy
    output_root: str = Field(min_length=1)
    recovery_root: str = Field(min_length=1)
    artifact_root: str = Field(min_length=1)
    read_only_roots: tuple[str, ...] = Field(min_length=1)
    recovery_state_names: tuple[str, ...] = Field(min_length=1)
    persist_frozen_identity_before_training: bool
    environment_compatibility_policy: str = Field(min_length=1)
    config_hash: ContentHash

    @field_validator(
        "environment_lock_artifact",
        "output_root",
        "recovery_root",
        "artifact_root",
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
    def validate_contract(self) -> "RemediationTemplate":
        expected_output = _PROFILE_OUTPUT_ROOTS[self.profile]
        expected = {
            "schema_version": (self.schema_version, "1.0.0"),
            "configuration_namespace": (
                self.configuration_namespace,
                "remediation/v1",
            ),
            "template_version": (self.template_version, "v1"),
            "conda_environment": (self.conda_environment, "TCFM"),
            "environment_lock_hash": (
                self.environment_lock_hash,
                _AUDITED_ENVIRONMENT_LOCK,
            ),
            "device": (self.device, _EXPECTED_DEVICE),
            "expected_device_name": (
                self.expected_device_name,
                _EXPECTED_DEVICE_NAME,
            ),
            "visible_cuda_device_count": (self.visible_cuda_device_count, 1),
            "process_count": (self.process_count, 1),
            "seed_set": (self.seed_set, (7, 42, 123)),
            "checkpoint_interval_epochs": (self.checkpoint_interval_epochs, 1),
            "checkpoint_retention": (
                self.checkpoint_retention,
                CheckpointRetentionPolicy(),
            ),
            "output_root": (self.output_root, expected_output),
            "recovery_root": (self.recovery_root, f"{expected_output}/recovery"),
            "artifact_root": (
                self.artifact_root,
                _PROFILE_ARTIFACT_ROOTS[self.profile],
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
            "environment_compatibility_policy": (
                self.environment_compatibility_policy,
                "exact",
            ),
        }
        mismatched = [
            name for name, (observed, required) in expected.items()
            if observed != required
        ]
        if mismatched:
            raise ValueError(
                "remediation fixed fields differ from the v1 contract: "
                + ", ".join(mismatched)
            )
        self._validate_profile()
        self._validate_budget()
        expected_accounting = InputVolumeAccountingPolicy(
            method="actual-bytes-read",
            count_repeated_reads=True,
            include_preprocessing_reads=True,
            budget_field=ResourceField.INPUT_VOLUME_BYTES.value,
            require_runtime_measurement=True,
        )
        if self.input_volume_accounting != expected_accounting:
            raise ValueError("input-volume accounting differs from the v1 contract")
        if self.config_hash != hash_object(self.hash_payload()):
            raise ValueError("config_hash does not match the remediation template")
        return self

    def _validate_profile(self) -> None:
        if self.profile is RemediationProfile.DETERMINISTIC_FLOAT32:
            expected_precision = NumericPrecision.FLOAT32
            expected_mode = ExecutionMode.DETERMINISTIC
            expected_autocast = AutocastPolicy(
                enabled=False,
                device_type="cuda",
                dtype=NumericPrecision.FLOAT32,
                cache_enabled=False,
            )
        else:
            expected_precision = NumericPrecision.MIXED_BFLOAT16
            expected_mode = ExecutionMode.PERFORMANCE
            expected_autocast = AutocastPolicy(
                enabled=True,
                device_type="cuda",
                dtype=NumericPrecision.BFLOAT16,
                cache_enabled=True,
            )
        if self.numeric_precision is not expected_precision:
            raise ValueError("numeric_precision differs from the remediation profile")
        if self.execution_mode is not expected_mode:
            raise ValueError("execution_mode differs from the remediation profile")
        if self.framework_settings != FrameworkDeterminismSettings.for_mode(expected_mode):
            raise ValueError("framework settings differ from the execution mode")
        if self.autocast_policy != expected_autocast:
            raise ValueError("autocast policy differs from the remediation profile")
        expected_accumulation = GradientAccumulationPolicy(
            accumulation_steps=1,
            divide_loss_by_accumulation_steps=True,
            flush_final_partial_accumulation=True,
        )
        if self.gradient_accumulation != expected_accumulation:
            raise ValueError("gradient accumulation differs from the v1 contract")

    def _validate_budget(self) -> None:
        hard_fields: set[ResourceField] = set()
        observed_fields: set[ResourceField] = set()
        enforcement = {limit.enforcement for limit in self.resource_budget.limits}
        if enforcement != {LimitEnforcement.HARD, LimitEnforcement.ADVISORY}:
            raise ValueError("resource budget must contain hard and advisory limits")
        for limit in self.resource_budget.limits:
            try:
                field = ResourceField(limit.field)
                unit = ResourceUnit(limit.unit)
            except ValueError as exc:
                raise ValueError("resource budget uses an unsupported field or unit") from exc
            observed_fields.add(field)
            if unit is not _EXPECTED_UNITS[field]:
                raise ValueError(
                    f"unit for {field.value} must be {_EXPECTED_UNITS[field].value}"
                )
            if limit.enforcement is LimitEnforcement.HARD:
                hard_fields.add(field)
        if observed_fields != set(_EXPECTED_UNITS):
            missing = sorted(field.value for field in set(_EXPECTED_UNITS) - observed_fields)
            raise ValueError("resource budget lacks required fields: " + ", ".join(missing))
        if not _REQUIRED_HARD_LIMITS.issubset(hard_fields):
            missing = sorted(field.value for field in _REQUIRED_HARD_LIMITS - hard_fields)
            raise ValueError("resource budget lacks hard limits: " + ", ".join(missing))

    def hash_payload(self) -> dict[str, Any]:
        # JSON mode normalizes enums and tuple-backed schema fields to the
        # portable representation stored in the versioned template files.
        return self.model_dump(mode="json", exclude={"config_hash"})

    def validate_workspace_root(self, workspace_root: str | Path) -> Path:
        value = str(workspace_root)
        candidate = PureWindowsPath(value)
        if (
            not candidate.is_absolute()
            or candidate.drive.casefold() != "g:"
            or tuple(part.casefold() for part in candidate.parts[1:])
            != tuple(part.casefold() for part in _EXPECTED_WORKSPACE.parts[1:])
        ):
            raise RemediationConfigurationError(
                r"workspace root must be G:\Datasets\HyperCross"
            )
        return Path(value)

    def validate_planning_context(self, context: RemediationPlanningContext) -> None:
        expected = {
            "conda_environment": self.conda_environment,
            "environment_lock_hash": self.environment_lock_hash,
            "cuda_available": True,
            "cuda_device_count": self.visible_cuda_device_count,
            "device_identity": f"{self.device}:{self.expected_device_name}",
            "device": self.device,
            "process_count": self.process_count,
            "numeric_precision": self.numeric_precision,
            "execution_mode": self.execution_mode,
            "autocast_policy": self.autocast_policy,
            "gradient_accumulation": self.gradient_accumulation,
        }
        observed = {
            "conda_environment": context.conda_environment,
            "environment_lock_hash": context.environment_lock.lock_hash,
            "cuda_available": context.environment_lock.cuda_available,
            "cuda_device_count": context.environment_lock.cuda_device_count,
            "device_identity": context.environment_lock.device_identity,
            "device": context.device,
            "process_count": context.process_count,
            "numeric_precision": context.numeric_precision,
            "execution_mode": context.execution_mode,
            "autocast_policy": context.autocast_policy,
            "gradient_accumulation": context.gradient_accumulation,
        }
        mismatches = [name for name, value in observed.items() if value != expected[name]]
        if mismatches:
            raise RemediationConfigurationError(
                "planning context differs from the remediation template: "
                + ", ".join(mismatches)
            )

    def validate_write_path(self, relative_path: str) -> str:
        normalized = _normalized_relative_path(relative_path)
        if not (
            _is_at_or_below(normalized, self.output_root)
            or _is_at_or_below(normalized, self.artifact_root)
        ):
            raise RemediationConfigurationError(
                "write path must stay in this precision profile's output namespace"
            )
        if any(_is_at_or_below(normalized, root) for root in self.read_only_roots):
            raise RemediationConfigurationError("path targets a read-only legacy root")
        return normalized

    def freeze_job(
        self,
        *,
        planning_context: RemediationPlanningContext,
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
        resume_identity: FrozenRunIdentity | None = None,
    ) -> "FrozenRemediationJob":
        """Validate the runtime and freeze one job before it can be scheduled."""

        self.validate_planning_context(planning_context)
        if not _JOB_NAME.fullmatch(job_name):
            raise RemediationConfigurationError(
                "job_name must use only letters, digits, dot, underscore, or hyphen"
            )
        if seed not in self.seed_set:
            raise RemediationConfigurationError("job seed is outside the frozen seed_set")
        if isinstance(input_volume_bytes, bool) or input_volume_bytes <= 0:
            raise RemediationConfigurationError(
                "input_volume_bytes must be a positive measured byte count"
            )
        supplied = dict(parameters or {})
        reserved = {
            "autocast_policy",
            "checkpoint_interval_epochs",
            "conda_environment",
            "device",
            "environment_lock_hash",
            "execution_mode",
            "gradient_accumulation",
            "input_volume_accounting",
            "numeric_precision",
            "output_root",
            "process_count",
            "remediation_profile",
            "remediation_template_config_hash",
            "remediation_template_version",
            "selection_metric_direction",
        }
        conflict = sorted(reserved.intersection(supplied))
        if conflict:
            raise RemediationConfigurationError(
                "job parameters cannot override frozen fields: " + ", ".join(conflict)
            )
        supplied.update(
            {
                "autocast_policy": self.autocast_policy.model_dump(mode="json"),
                "checkpoint_interval_epochs": self.checkpoint_interval_epochs,
                "conda_environment": self.conda_environment,
                "device": self.device,
                "environment_lock_hash": self.environment_lock_hash,
                "execution_mode": self.execution_mode.value,
                "gradient_accumulation": self.gradient_accumulation.model_dump(mode="json"),
                "input_volume_accounting": self.input_volume_accounting.model_dump(mode="json"),
                "numeric_precision": self.numeric_precision.value,
                "output_root": self.output_root,
                "process_count": self.process_count,
                "remediation_profile": self.profile.value,
                "remediation_template_config_hash": self.config_hash,
                "remediation_template_version": self.template_version,
                "selection_metric_direction": selection_metric_direction.value,
            }
        )
        config = FrozenRunConfig.create(
            schema_version="1.0.0",
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
            environment_compatibility_policy=self.environment_compatibility_policy,
        )
        job_root = f"{self.output_root}/jobs/{job_name}"
        job = FrozenRemediationJob(
            profile=self.profile,
            template_version=self.template_version,
            template_config_hash=self.config_hash,
            output_root=self.output_root,
            artifact_root=self.artifact_root,
            frozen_config=config,
            frozen_identity=identity,
            frozen_identity_relative_path=f"{job_root}/frozen_run_identity.json",
            recovery_directory=f"{self.recovery_root}/{job_name}",
            recovery_state_names=self.recovery_state_names,
            persist_identity_before_training=self.persist_frozen_identity_before_training,
        )
        if resume_identity is not None:
            job.require_exact_resume_identity(resume_identity)
        return job


class FrozenRemediationJob(FrozenSchemaModel):
    """Complete, profile-isolated remediation scheduler handoff."""

    schema_version: str = Field(default="1.0.0", min_length=1)
    profile: RemediationProfile
    template_version: str = Field(min_length=1)
    template_config_hash: ContentHash
    output_root: str
    artifact_root: str
    frozen_config: FrozenRunConfig
    frozen_identity: FrozenRunIdentity
    frozen_identity_relative_path: str
    recovery_directory: str
    recovery_state_names: tuple[str, ...]
    persist_identity_before_training: bool

    @field_validator(
        "output_root",
        "artifact_root",
        "frozen_identity_relative_path",
        "recovery_directory",
    )
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _normalized_relative_path(value)

    @model_validator(mode="after")
    def validate_job(self) -> "FrozenRemediationJob":
        expected_output = _PROFILE_OUTPUT_ROOTS[self.profile]
        expected_recovery = f"{expected_output}/recovery"
        if self.output_root != expected_output:
            raise ValueError("job output_root differs from its remediation profile")
        if self.artifact_root != _PROFILE_ARTIFACT_ROOTS[self.profile]:
            raise ValueError("job artifact_root differs from its remediation profile")
        if not _is_at_or_below(self.frozen_identity_relative_path, self.output_root):
            raise ValueError("Frozen_Run_Identity must stay under the profile output root")
        if not _is_at_or_below(self.recovery_directory, expected_recovery):
            raise ValueError("recovery state must stay under the profile recovery root")
        if self.frozen_identity.config_hash != self.frozen_config.config_hash:
            raise ValueError("Frozen_Run_Identity config hash does not match config")
        if self.frozen_identity.model_key != self.frozen_config.model_key:
            raise ValueError("Frozen_Run_Identity model key does not match config")
        if self.frozen_identity.environment_lock_hash != _AUDITED_ENVIRONMENT_LOCK:
            raise ValueError("job is not bound to the audited Environment_Lock")
        if self.frozen_identity.environment_compatibility_policy != "exact":
            raise ValueError("remediation recovery requires exact environment identity")
        if self.recovery_state_names != _REQUIRED_RECOVERY_STATES:
            raise ValueError("recovery must include model, optimizer, scheduler, and stage")
        if not self.persist_identity_before_training:
            raise ValueError("Frozen_Run_Identity must be persisted before training")
        return self

    def require_exact_resume_identity(self, observed: FrozenRunIdentity) -> None:
        """Reject cross-profile or otherwise inexact recovery before state loading."""

        fields = tuple(type(self.frozen_identity).model_fields)
        mismatches = [
            field
            for field in fields
            if getattr(observed, field) != getattr(self.frozen_identity, field)
        ]
        if mismatches:
            raise RemediationConfigurationError(
                "resume identity differs from the planned Frozen_Run_Identity: "
                + ", ".join(mismatches)
            )

    def validate_recovery_source(self, relative_path: str) -> str:
        normalized = _normalized_relative_path(relative_path)
        if not _is_at_or_below(normalized, self.recovery_directory):
            raise RemediationConfigurationError(
                "recovery source must stay inside this job's precision-specific namespace"
            )
        return normalized


def load_remediation_template(path: str | Path) -> RemediationTemplate:
    """Read and validate one immutable remediation JSON template."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemediationConfigurationError(
            f"cannot read remediation template {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RemediationConfigurationError(
            "remediation template must contain a JSON object"
        )
    try:
        return RemediationTemplate.model_validate(payload)
    except ValidationError as exc:
        raise RemediationConfigurationError(
            f"invalid remediation template {source}: {exc}"
        ) from exc


def _normalized_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("path must be a non-empty trimmed string")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        if windows.drive.casefold() == "f:":
            raise RemediationConfigurationError("F: paths are forbidden")
        raise RemediationConfigurationError("paths must be workspace-relative")
    if ".." in windows.parts or ".." in posix.parts:
        raise RemediationConfigurationError("paths must not traverse parents")
    normalized = PurePosixPath(value.replace("\\", "/")).as_posix()
    if normalized in {"", "."}:
        raise ValueError("path must identify a workspace-relative target")
    return normalized


def _is_at_or_below(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


__all__ = [
    "AutocastPolicy",
    "FrozenRemediationJob",
    "GradientAccumulationPolicy",
    "InputVolumeAccountingPolicy",
    "RemediationConfigurationError",
    "RemediationPlanningContext",
    "RemediationProfile",
    "RemediationTemplate",
    "load_remediation_template",
]
