"""PowerShell-safe command-line entry points for benchmark infrastructure."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any, Optional

from benchmark import __version__
from benchmark.cli_runtime import (
    FastLaneRuntimeOptions,
    fast_lane_summary,
    launch_runner_module,
    persist_launch_handoff,
    prepare_fast_lane_job,
    validate_delegated_request,
    validate_live_environment,
)
from benchmark.fast_lane import FastLaneConfigurationError, FastLaneStage
from benchmark.models import ModelRegistry
from benchmark.protocols import ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import EnvironmentLock, SampleManifest, capture_environment_lock

_PROTOCOL_SPEC_FIELDS = {
    "schema_version",
    "protocol_key",
    "folds",
    "selection_policy",
    "seed_set",
    "evaluation_budget",
}


def _stable_model_key(value: str) -> str:
    """Accept one stable key while rejecting positional or list selectors."""

    if not value or value != value.strip():
        raise argparse.ArgumentTypeError("Model_Key must be a non-empty trimmed string")
    if value.isdecimal() or "," in value or value.startswith("[") or value.endswith("]"):
        raise argparse.ArgumentTypeError(
            "model selection requires exactly one stable Model_Key, not an index or list"
        )
    return value


def _add_fast_lane_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_runner: bool,
) -> None:
    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="hash-bound stage template under configs/fast_lane",
    )
    parser.add_argument(
        "--job-config",
        required=True,
        type=Path,
        help="portable job request containing dataset keys and relative references",
    )
    parser.add_argument(
        "--workspace-root",
        default=r"G:\Datasets\HyperCross",
        help=r"sole writable workspace; must be G:\Datasets\HyperCross",
    )
    parser.add_argument("--conda-environment", default="TCFM")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process-count", default=1, type=int)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--determinism", default="deterministic")
    parser.add_argument(
        "--output-root",
        required=True,
        help="workspace-relative stage output root, for example runs_v2/smoke",
    )
    if include_runner:
        parser.add_argument(
            "--runner-module",
            help="dotted module inside benchmark; required unless --dry-run",
        )


def build_parser(model_registry: Optional[ModelRegistry] = None) -> argparse.ArgumentParser:
    """Build the package-level command-line parser."""

    registry = model_registry if model_registry is not None else ModelRegistry()

    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="HyperCross protocol-first benchmark infrastructure.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    dry_run = subparsers.add_parser(
        "protocol-dry-run",
        help="validate and publish a protocol without constructing a model",
    )
    dry_run.add_argument(
        "--sample-manifest",
        required=True,
        type=Path,
        help="path to a JSON SampleManifest",
    )
    dry_run.add_argument(
        "--protocol-spec",
        required=True,
        type=Path,
        help="path to a JSON protocol planning request",
    )
    dry_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="artifact root where the validated protocol is published",
    )
    dry_run.add_argument(
        "--relative-path",
        help="optional artifact path relative to output-root",
    )
    dry_run.set_defaults(handler=_run_protocol_dry_run)

    # ``plan`` is the task-18.4 fast-lane planning command.  The older
    # ``protocol-dry-run`` name remains available for protocol-only callers.
    plan = subparsers.add_parser(
        "plan",
        help="validate and freeze one fast-lane job without writing or training",
    )
    _add_fast_lane_arguments(plan, include_runner=False)
    plan.set_defaults(handler=_run_fast_lane_plan)

    run = subparsers.add_parser(
        "run",
        help="launch one validated fast-lane job through a Python runner module",
    )
    _add_fast_lane_arguments(run, include_runner=True)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the frozen job without writing or launching",
    )
    run.set_defaults(handler=_run_fast_lane_run)

    resume = subparsers.add_parser(
        "resume",
        help="resume one fast-lane job from a validated workspace-relative checkpoint",
    )
    _add_fast_lane_arguments(resume, include_runner=True)
    resume.add_argument(
        "--recovery-source",
        required=True,
        help="workspace-relative recovery artifact inside the selected stage root",
    )
    resume.add_argument(
        "--dry-run",
        action="store_true",
        help="validate resume inputs without restoring state or launching",
    )
    resume.set_defaults(handler=_run_fast_lane_resume)

    smoke = subparsers.add_parser(
        "smoke",
        help="launch the bounded smoke stage; use --dry-run to validate only",
    )
    _add_fast_lane_arguments(smoke, include_runner=True)
    smoke.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the smoke launch without writing or training",
    )
    smoke.set_defaults(handler=_run_fast_lane_smoke)

    for operation in ("aggregate", "report", "audit"):
        delegated = subparsers.add_parser(
            operation,
            help=f"validate and dispatch a portable {operation} request",
        )
        delegated.add_argument("--request", required=True, type=Path)
        delegated.add_argument(
            "--workspace-root",
            default=r"G:\Datasets\HyperCross",
            help=r"must be G:\Datasets\HyperCross",
        )
        delegated.add_argument(
            "--runner-module",
            help="dotted module inside benchmark; required unless --dry-run",
        )
        delegated.add_argument("--dry-run", action="store_true")
        delegated.set_defaults(handler=_run_delegated_operation, operation=operation)

    model_resolve = subparsers.add_parser(
        "model-resolve",
        help="resolve one registered model by stable Model_Key",
    )
    model_resolve.add_argument(
        "--model-key",
        required=True,
        type=_stable_model_key,
        help="exact stable Model_Key; positional indices and lists are rejected",
    )
    model_resolve.set_defaults(handler=_run_model_resolve, model_registry=registry)
    return parser


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} JSON must contain an object")
    return value


def _run_protocol_dry_run(arguments: argparse.Namespace) -> int:
    """Plan and publish one protocol without importing or building models."""

    manifest_payload = _load_json_object(
        arguments.sample_manifest,
        description="sample manifest",
    )
    sample_manifest = SampleManifest.model_validate(manifest_payload)
    protocol_spec = _load_json_object(
        arguments.protocol_spec,
        description="protocol spec",
    )
    unknown = sorted(set(protocol_spec) - _PROTOCOL_SPEC_FIELDS)
    missing = sorted(_PROTOCOL_SPEC_FIELDS - set(protocol_spec))
    if unknown or missing:
        raise ValueError(
            "protocol spec fields do not match the dry-run contract; "
            f"missing={missing}, unknown={unknown}"
        )

    planner = ProtocolPlanner(FailureAtomicArtifactStore(arguments.output_root))
    published = planner.dry_run(
        sample_manifest,
        relative_path=arguments.relative_path,
        **protocol_spec,
    )
    summary = {
        "artifact": published.artifact.model_dump(mode="json"),
        "dataset_hash": published.manifest.dataset_hash,
        "fold_ids": [fold.fold_id for fold in published.manifest.folds],
        "model_built": False,
        "protocol_hash": published.manifest.protocol_hash,
        "protocol_key": published.manifest.protocol_key,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _run_model_resolve(arguments: argparse.Namespace) -> int:
    """Resolve one identity/contract pair without constructing a model."""

    registration = arguments.model_registry.resolve(arguments.model_key)
    summary = {
        "input_contract": registration.input_contract.model_dump(mode="json"),
        "model_identity": registration.identity.model_dump(mode="json"),
        "model_key": registration.model_key,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _runtime_options(arguments: argparse.Namespace) -> FastLaneRuntimeOptions:
    return FastLaneRuntimeOptions(
        workspace_root=arguments.workspace_root,
        conda_environment=arguments.conda_environment,
        device=arguments.device,
        process_count=arguments.process_count,
        precision=arguments.precision,
        determinism=arguments.determinism,
        output_root=arguments.output_root,
    )


def _prepare_from_arguments(arguments: argparse.Namespace):
    return prepare_fast_lane_job(
        template_path=arguments.template,
        job_config_path=arguments.job_config,
        options=_runtime_options(arguments),
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _run_fast_lane_plan(arguments: argparse.Namespace) -> int:
    template, request, job, _ = _prepare_from_arguments(arguments)
    _print_json(fast_lane_summary(template, request, job, operation="plan"))
    return 0


def _run_fast_lane_run(arguments: argparse.Namespace) -> int:
    template, request, job, workspace = _prepare_from_arguments(arguments)
    if arguments.dry_run:
        _print_json(fast_lane_summary(template, request, job, operation="run"))
        return 0
    return _launch_fast_lane(
        arguments,
        template=template,
        request=request,
        job=job,
        workspace=workspace,
        operation="run",
    )


def _run_fast_lane_resume(arguments: argparse.Namespace) -> int:
    template, request, job, workspace = _prepare_from_arguments(arguments)
    recovery_source = template.validate_recovery_source(arguments.recovery_source)
    if arguments.dry_run:
        _print_json(
            fast_lane_summary(
                template,
                request,
                job,
                operation="resume",
                recovery_source=recovery_source,
            )
        )
        return 0
    return _launch_fast_lane(
        arguments,
        template=template,
        request=request,
        job=job,
        workspace=workspace,
        operation="resume",
        recovery_source=recovery_source,
    )


def _run_fast_lane_smoke(arguments: argparse.Namespace) -> int:
    template, request, job, workspace = _prepare_from_arguments(arguments)
    if template.stage is not FastLaneStage.SMOKE or request.stage is not FastLaneStage.SMOKE:
        raise FastLaneConfigurationError("smoke command requires the frozen smoke stage")
    if arguments.dry_run:
        _print_json(fast_lane_summary(template, request, job, operation="smoke"))
        return 0
    return _launch_fast_lane(
        arguments,
        template=template,
        request=request,
        job=job,
        workspace=workspace,
        operation="run",
    )


def _launch_fast_lane(
    arguments: argparse.Namespace,
    *,
    template: Any,
    request: Any,
    job: Any,
    workspace: Path,
    operation: str,
    recovery_source: str | None = None,
) -> int:
    if not arguments.runner_module:
        raise FastLaneConfigurationError("--runner-module is required without --dry-run")
    validate_live_environment(
        template,
        environment_provider=arguments.environment_provider,
    )
    manifest = persist_launch_handoff(
        workspace=workspace,
        template=template,
        request=request,
        job=job,
        operation=operation,
        recovery_source=recovery_source,
    )
    return launch_runner_module(
        runner_module=arguments.runner_module,
        launch_manifest=manifest,
        workspace=workspace,
        process_runner=arguments.process_runner,
    )


def _run_delegated_operation(arguments: argparse.Namespace) -> int:
    payload = validate_delegated_request(arguments.request)
    workspace_path = PureWindowsPath(arguments.workspace_root)
    if str(workspace_path).casefold() != r"g:\datasets\hypercross":
        raise FastLaneConfigurationError(
            r"workspace root must be G:\Datasets\HyperCross"
        )
    summary = {
        "operation": arguments.operation,
        "dry_run": bool(arguments.dry_run),
        "training_started": False,
        "request": payload,
        "workspace_root": str(workspace_path),
    }
    if arguments.dry_run:
        _print_json(summary)
        return 0
    if not arguments.runner_module:
        raise FastLaneConfigurationError("--runner-module is required without --dry-run")
    return launch_runner_module(
        runner_module=arguments.runner_module,
        launch_manifest=arguments.request.resolve(),
        workspace=Path(arguments.workspace_root),
        process_runner=arguments.process_runner,
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    model_registry: Optional[ModelRegistry] = None,
    environment_provider: Callable[[], EnvironmentLock] = capture_environment_lock,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    """Run one command without shell-specific assumptions."""

    parser = build_parser(model_registry)
    arguments = parser.parse_args(argv)
    arguments.environment_provider = environment_provider
    arguments.process_runner = process_runner
    handler = getattr(arguments, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(arguments))
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
