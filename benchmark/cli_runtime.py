"""Portable CLI request validation and shell-free fast-lane launching."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import Field, StrictInt, ValidationError, field_validator, model_validator

from benchmark.fast_lane import (
    FastLaneConfigurationError,
    FastLaneStage,
    FastLaneStageTemplate,
    FrozenFastLaneJob,
    load_fast_lane_template,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    ContentHash,
    EnvironmentLock,
    MetricDirection,
    ModelKey,
    ProtocolKey,
    capture_environment_lock,
    canonical_serialize,
)
from benchmark.schema.environment import FrozenSchemaModel


_RUNNER_MODULE = re.compile(r"^benchmark(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_ALLOWED_DATASET_KEYS = {"hyperleaf2024", "histology-hsi-gb", "remote-sensing-sanity"}


class FastLaneJobRequest(FrozenSchemaModel):
    """Portable, hash-bound inputs used to freeze one fast-lane job."""

    schema_version: str = Field(default="1.0", min_length=1)
    stage: FastLaneStage
    job_name: str = Field(min_length=1)
    dataset_key: str = Field(min_length=1)
    dataset_hash: ContentHash
    protocol_key: ProtocolKey
    protocol_hash: ContentHash
    model_key: ModelKey
    fold_id: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    input_volume_bytes: StrictInt = Field(ge=0)
    selection_metric_direction: MetricDirection
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset_key")
    @classmethod
    def validate_dataset_key(cls, value: str) -> str:
        if value not in _ALLOWED_DATASET_KEYS:
            raise ValueError(
                "dataset_key must identify an approved catalog entry: "
                + ", ".join(sorted(_ALLOWED_DATASET_KEYS))
            )
        return value

    @model_validator(mode="after")
    def validate_portability(self) -> "FastLaneJobRequest":
        # Canonical serialization rejects non-finite values and unsupported types.
        canonical_serialize(self.parameters)
        _validate_portable_value(self.parameters, field="parameters")
        return self

    def freeze(self, template: FastLaneStageTemplate) -> FrozenFastLaneJob:
        if template.stage is not self.stage:
            raise FastLaneConfigurationError(
                "job stage differs from the selected fast-lane template"
            )
        return template.freeze_job(
            job_name=self.job_name,
            dataset_hash=self.dataset_hash,
            protocol_hash=self.protocol_hash,
            protocol_key=self.protocol_key,
            model_key=self.model_key,
            fold_id=self.fold_id,
            seed=self.seed,
            input_volume_bytes=self.input_volume_bytes,
            selection_metric_direction=self.selection_metric_direction,
            parameters=self.parameters,
        )


class FastLaneRuntimeOptions(FrozenSchemaModel):
    """Explicit command-line execution settings checked against a template."""

    workspace_root: str = Field(min_length=1)
    conda_environment: str = Field(min_length=1)
    device: str = Field(min_length=1)
    process_count: StrictInt = Field(gt=0)
    precision: str = Field(min_length=1)
    determinism: str = Field(min_length=1)
    output_root: str = Field(min_length=1)

    def validate_against(self, template: FastLaneStageTemplate) -> Path:
        workspace = template.validate_workspace_root(self.workspace_root)
        expected = {
            "conda_environment": template.conda_environment,
            "device": template.device,
            "process_count": template.process_count,
            "precision": template.numeric_precision.value,
            "determinism": template.execution_mode.value,
            "output_root": template.output_root,
        }
        observed = {
            "conda_environment": self.conda_environment,
            "device": self.device,
            "process_count": self.process_count,
            "precision": self.precision,
            "determinism": self.determinism,
            "output_root": self.output_root.replace("\\", "/"),
        }
        mismatches = [key for key, value in observed.items() if value != expected[key]]
        if mismatches:
            raise FastLaneConfigurationError(
                "CLI options differ from the frozen fast-lane template: "
                + ", ".join(mismatches)
            )
        template.validate_write_path(observed["output_root"])
        return workspace


def load_fast_lane_job_request(path: str | Path) -> FastLaneJobRequest:
    """Load a strict portable job request without allocating a run or writing files."""

    payload = _load_json_object(Path(path), description="fast-lane job config")
    try:
        return FastLaneJobRequest.model_validate(payload)
    except ValidationError as exc:
        raise FastLaneConfigurationError(f"invalid fast-lane job config: {exc}") from exc


def load_audited_environment_lock(
    workspace_root: str | Path,
    template: FastLaneStageTemplate,
) -> EnvironmentLock:
    """Load the passed audit lock from the template's workspace-relative reference."""

    workspace = template.validate_workspace_root(workspace_root)
    relative = template.validate_write_path(template.environment_lock_artifact)
    audit_path = workspace / Path(*PurePosixPath(relative).parts)
    payload = _load_json_object(audit_path, description="environment audit")
    if payload.get("status") != "pass":
        raise FastLaneConfigurationError("environment audit status is not pass")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise FastLaneConfigurationError("environment audit contains a failed check")
    try:
        lock = EnvironmentLock.model_validate(payload["environment_lock"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FastLaneConfigurationError(
            "environment audit does not contain a valid Environment_Lock"
        ) from exc
    template.validate_environment(lock, conda_environment=template.conda_environment)
    return lock


def validate_live_environment(
    template: FastLaneStageTemplate,
    *,
    environment_provider: Callable[[], EnvironmentLock] = capture_environment_lock,
) -> EnvironmentLock:
    """Validate the active TCFM/CUDA process immediately before runner launch."""

    conda_environment = os.environ.get("CONDA_DEFAULT_ENV", "")
    lock = environment_provider()
    template.validate_environment(lock, conda_environment=conda_environment)
    return lock


def prepare_fast_lane_job(
    *,
    template_path: str | Path,
    job_config_path: str | Path,
    options: FastLaneRuntimeOptions,
) -> tuple[FastLaneStageTemplate, FastLaneJobRequest, FrozenFastLaneJob, Path]:
    """Validate a template and request, then freeze a side-effect-free job plan."""

    template = load_fast_lane_template(template_path)
    workspace = options.validate_against(template)
    load_audited_environment_lock(workspace, template)
    request = load_fast_lane_job_request(job_config_path)
    job = request.freeze(template)
    return template, request, job, workspace


def fast_lane_summary(
    template: FastLaneStageTemplate,
    request: FastLaneJobRequest,
    job: FrozenFastLaneJob,
    *,
    operation: str,
    recovery_source: str | None = None,
) -> dict[str, Any]:
    """Return the machine-readable plan shown by dry-run commands."""

    summary: dict[str, Any] = {
        "operation": operation,
        "dry_run": True,
        "training_started": False,
        "dataset_key": request.dataset_key,
        "stage": template.stage.value,
        "job_name": request.job_name,
        "model_key": request.model_key,
        "protocol_key": request.protocol_key,
        "seed": request.seed,
        "device": template.device,
        "process_count": template.process_count,
        "numeric_precision": template.numeric_precision.value,
        "determinism": template.execution_mode.value,
        "resource_budget": template.resource_budget.model_dump(mode="json"),
        "output_root": template.output_root,
        "frozen_config_hash": job.frozen_config.config_hash,
        "frozen_identity": job.frozen_identity.model_dump(mode="json"),
        "frozen_identity_relative_path": job.frozen_identity_relative_path,
        "recovery_directory": job.recovery_directory,
    }
    if recovery_source is not None:
        summary["recovery_source"] = template.validate_recovery_source(recovery_source)
    return summary


def persist_launch_handoff(
    *,
    workspace: Path,
    template: FastLaneStageTemplate,
    request: FastLaneJobRequest,
    job: FrozenFastLaneJob,
    operation: str,
    recovery_source: str | None = None,
) -> Path:
    """Persist immutable identity and a unique launch handoff before execution."""

    store = FailureAtomicArtifactStore(workspace)
    if operation == "run":
        identity_bytes = _json_bytes(job.frozen_identity.model_dump(mode="json"))
        store.publish_bytes(
            artifact_id=f"frozen-identity-{request.job_name}",
            relative_path=job.frozen_identity_relative_path,
            content=identity_bytes,
            media_type="application/json",
        )
    elif operation != "resume":
        raise ValueError("operation must be run or resume")

    launch_id = str(uuid.uuid4())
    launch_relative_path = (
        f"{job.output_root}/jobs/{request.job_name}/launches/"
        f"{operation}-{launch_id}.json"
    )
    template.validate_write_path(launch_relative_path)
    launch = fast_lane_summary(
        template,
        request,
        job,
        operation=operation,
        recovery_source=recovery_source,
    )
    launch.update(
        {
            "dry_run": False,
            "launch_id": launch_id,
            "training_started": False,
            "frozen_config": job.frozen_config.model_dump(mode="json"),
        }
    )
    store.publish_bytes(
        artifact_id=f"fast-lane-launch-{launch_id}",
        relative_path=launch_relative_path,
        content=_json_bytes(launch),
        media_type="application/json",
    )
    return workspace / Path(*PurePosixPath(launch_relative_path).parts)


def launch_runner_module(
    *,
    runner_module: str,
    launch_manifest: Path,
    workspace: Path,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    """Launch a package runner with an argument vector and no command shell."""

    if not _RUNNER_MODULE.fullmatch(runner_module):
        raise FastLaneConfigurationError(
            "runner module must be a dotted module inside the benchmark package"
        )
    completed = process_runner(
        [
            sys.executable,
            "-m",
            runner_module,
            "--launch-manifest",
            str(launch_manifest),
        ],
        cwd=workspace,
        check=False,
        shell=False,
    )
    return int(completed.returncode)


def validate_delegated_request(path: str | Path) -> dict[str, Any]:
    """Validate portable aggregate/report/audit request payloads at the CLI edge."""

    payload = _load_json_object(Path(path), description="operation request")
    canonical_serialize(payload)
    _validate_portable_value(payload, field="request")
    return payload


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FastLaneConfigurationError(f"cannot read {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FastLaneConfigurationError(f"{description} must contain a JSON object")
    return value


def _validate_portable_value(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_portable_value(item, field=f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_portable_value(item, field=f"{field}[{index}]")
        return
    if not isinstance(value, str):
        return
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.drive.casefold() == "f:":
        raise FastLaneConfigurationError(f"{field} contains a forbidden F: path")
    if _WINDOWS_DRIVE.match(value) or windows.is_absolute() or posix.is_absolute():
        raise FastLaneConfigurationError(
            f"{field} must use dataset keys or workspace-relative references"
        )
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise FastLaneConfigurationError(f"{field} must not traverse parent directories")
    if parts and parts[0].casefold() in {"src", "runs"}:
        raise FastLaneConfigurationError(
            f"{field} must not target read-only legacy src/runs"
        )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "FastLaneJobRequest",
    "FastLaneRuntimeOptions",
    "fast_lane_summary",
    "launch_runner_module",
    "load_audited_environment_lock",
    "load_fast_lane_job_request",
    "persist_launch_handoff",
    "prepare_fast_lane_job",
    "validate_delegated_request",
    "validate_live_environment",
]
