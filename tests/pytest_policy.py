"""Pytest marker taxonomy and audited GPU-environment policy."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pytest

PRIMARY_MARKERS = frozenset(
    {"unit", "property", "integration", "regression", "smoke", "audit", "gpu"}
)
DIRECTORY_MARKERS = {
    "unit": "unit",
    "property": "property",
    "integration": "integration",
    "regression": "regression",
    "smoke": "smoke",
    "audit": "audit",
    "gpu": "gpu",
}


def marker_for_path(path: Path, tests_root: Path) -> str | None:
    """Return the suite marker implied by a test's first directory level."""

    try:
        relative = path.resolve().relative_to(tests_root.resolve())
    except ValueError:
        return None
    if not relative.parts:
        return None
    return DIRECTORY_MARKERS.get(relative.parts[0])


def implied_markers(marker_names: Iterable[str]) -> frozenset[str]:
    """Return execution markers implied by explicitly assigned markers."""

    names = set(marker_names)
    if "strict" in names:
        names.add("slow")
    return frozenset(names)


@lru_cache(maxsize=1)
def audited_environment_issues(workspace_root: Path) -> tuple[str, ...]:
    """Compare the active process with the audited TCFM environment lock."""

    audit_path = workspace_root / "artifacts" / "audit" / "training_fast_lane_audit.json"
    if not audit_path.is_file():
        return (f"audited environment lock is missing: {audit_path}",)

    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_environment = audit["environment"]
        expected_lock = audit["environment_lock"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return (f"audited environment lock is unreadable: {exc}",)

    issues: list[str] = []
    expected_conda = str(expected_environment.get("conda_environment", "TCFM"))
    observed_conda = os.environ.get("CONDA_DEFAULT_ENV", "")
    if observed_conda.casefold() != expected_conda.casefold():
        issues.append(
            f"CONDA_DEFAULT_ENV must be {expected_conda!r}, observed {observed_conda or '<unset>'!r}"
        )

    try:
        from benchmark.schema.environment import capture_environment_lock

        observed_lock = capture_environment_lock()
    except Exception as exc:  # pragma: no cover - depends on the host installation
        issues.append(f"environment capture failed: {type(exc).__name__}: {exc}")
        return tuple(issues)

    expected_hash = str(expected_lock.get("lock_hash", ""))
    if observed_lock.lock_hash != expected_hash:
        issues.append(
            "active dependency/CUDA/device versions do not match the audited "
            f"Environment Lock {expected_hash or '<missing>'}"
        )
    if not observed_lock.cuda_available:
        issues.append("CUDA is unavailable")
    if observed_lock.cuda_device_count != 1:
        issues.append(
            f"exactly one visible CUDA device is required, observed {observed_lock.cuda_device_count}"
        )
    expected_device = str(expected_lock.get("device_identity", ""))
    if observed_lock.device_identity != expected_device:
        issues.append(
            f"device identity must be {expected_device!r}, observed {observed_lock.device_identity!r}"
        )
    return tuple(issues)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("hypercross environment")
    group.addoption(
        "--require-audited-environment",
        action="store_true",
        default=False,
        help=(
            "Fail GPU tests when TCFM, dependency versions, CUDA, or the device "
            "does not match the audited Environment Lock. Without this option, "
            "affected GPU tests are skipped."
        ),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply directory markers before pytest evaluates marker expressions."""

    tests_root = Path(__file__).resolve().parent
    unclassified: list[str] = []
    for item in items:
        inferred = marker_for_path(Path(str(item.path)), tests_root)
        if inferred is not None:
            item.add_marker(getattr(pytest.mark, inferred))

        names = {marker.name for marker in item.iter_markers()}
        for implied in implied_markers(names) - names:
            item.add_marker(getattr(pytest.mark, implied))

        final_names = {marker.name for marker in item.iter_markers()}
        if not (final_names & PRIMARY_MARKERS):
            unclassified.append(item.nodeid)

    if unclassified:
        rendered = "\n  ".join(unclassified)
        raise pytest.UsageError(
            "every test must belong to a primary suite marker; unclassified tests:\n  "
            + rendered
        )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip or fail GPU tests before test code touches CUDA."""

    if item.get_closest_marker("gpu") is None:
        return

    workspace_root = Path(__file__).resolve().parents[1]
    issues = audited_environment_issues(workspace_root)
    if not issues:
        return

    message = "GPU environment preflight failed: " + "; ".join(issues)
    if item.config.getoption("--require-audited-environment"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
