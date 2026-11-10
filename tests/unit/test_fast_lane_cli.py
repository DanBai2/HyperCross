"""Targeted tests for task 18.4 PowerShell-safe fast-lane CLI wiring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.cli import build_parser, main
from benchmark.cli_runtime import (
    FastLaneConfigurationError,
    launch_runner_module,
    load_fast_lane_job_request,
)


ROOT = Path(__file__).resolve().parents[2]
SMOKE_TEMPLATE = ROOT / "configs" / "fast_lane" / "smoke.json"
SMOKE_JOB = ROOT / "configs" / "fast_lane" / "smoke-job.example.json"
WORKSPACE = r"G:\Datasets\HyperCross"


def _fast_lane_arguments(command: str, job_config: Path = SMOKE_JOB) -> list[str]:
    return [
        command,
        "--template",
        str(SMOKE_TEMPLATE),
        "--job-config",
        str(job_config),
        "--workspace-root",
        WORKSPACE,
        "--conda-environment",
        "TCFM",
        "--device",
        "cuda:0",
        "--process-count",
        "1",
        "--precision",
        "float32",
        "--determinism",
        "deterministic",
        "--output-root",
        "runs_v2/smoke",
    ]


def test_plan_handles_spaced_config_path_without_writing_or_training(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spaced = tmp_path / "request directory with spaces"
    spaced.mkdir()
    job_config = spaced / "smoke job.json"
    job_config.write_bytes(SMOKE_JOB.read_bytes())

    assert main(_fast_lane_arguments("plan", job_config)) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["operation"] == "plan"
    assert summary["dry_run"] is True
    assert summary["training_started"] is False
    assert summary["device"] == "cuda:0"
    assert summary["process_count"] == 1
    assert summary["numeric_precision"] == "float32"
    assert summary["determinism"] == "deterministic"
    assert summary["output_root"] == "runs_v2/smoke"
    assert summary["frozen_identity_relative_path"].startswith("runs_v2/smoke/")


def test_smoke_dry_run_does_not_require_or_invoke_runner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*_fast_lane_arguments("smoke"), "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["operation"] == "smoke"
    assert summary["training_started"] is False
    assert summary["resource_budget"]["limits"][0]["value"] == 2


def test_resume_dry_run_rejects_legacy_and_cross_stage_sources() -> None:
    for recovery_source in (
        "runs/legacy/checkpoint.pt",
        "runs_v2/strict/jobs/job/recovery/epoch-1.pt",
        r"F:\checkpoints\epoch-1.pt",
    ):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    *_fast_lane_arguments("resume"),
                    "--recovery-source",
                    recovery_source,
                    "--dry-run",
                ]
            )
        assert exc.value.code == 2


def test_job_config_rejects_absolute_f_drive_and_legacy_write_references(
    tmp_path: Path,
) -> None:
    base = json.loads(SMOKE_JOB.read_text(encoding="utf-8"))
    for value in (r"F:\data\sample.tif", "src/generated_runner.py", "runs/new.pt"):
        payload = {**base, "parameters": {"output_reference": value}}
        path = tmp_path / f"bad-{len(value)}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(FastLaneConfigurationError):
            load_fast_lane_job_request(path)


def test_runner_launch_uses_argument_vector_and_disables_shell(tmp_path: Path) -> None:
    manifest = tmp_path / "launch manifest with spaces.json"
    manifest.write_text("{}", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_runner(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    result = launch_runner_module(
        runner_module="benchmark.runners.hyperleaf_smoke",
        launch_manifest=manifest,
        workspace=tmp_path,
        process_runner=fake_runner,
    )

    assert result == 0
    assert observed["command"] == [
        sys.executable,
        "-m",
        "benchmark.runners.hyperleaf_smoke",
        "--launch-manifest",
        str(manifest),
    ]
    assert observed["cwd"] == tmp_path
    assert observed["check"] is False
    assert observed["shell"] is False


def test_cli_exposes_required_task_18_4_commands() -> None:
    parser = build_parser()
    action = next(
        action for action in parser._actions if action.dest == "command"  # noqa: SLF001
    )
    assert {
        "plan",
        "run",
        "resume",
        "aggregate",
        "report",
        "audit",
        "smoke",
    }.issubset(action.choices)
