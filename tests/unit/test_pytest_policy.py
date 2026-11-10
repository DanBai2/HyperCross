"""Unit tests for the layered pytest marker policy."""

from pathlib import Path

from tests.pytest_policy import implied_markers, marker_for_path


def test_first_test_directory_assigns_primary_suite_marker(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"

    assert marker_for_path(tests_root / "unit" / "test_example.py", tests_root) == "unit"
    assert marker_for_path(tests_root / "property" / "test_property.py", tests_root) == "property"
    assert marker_for_path(tests_root / "gpu" / "test_cuda.py", tests_root) == "gpu"
    assert marker_for_path(tests_root / "unknown" / "test_other.py", tests_root) is None


def test_strict_marker_always_implies_slow() -> None:
    assert implied_markers({"strict", "gpu"}) == frozenset({"strict", "slow", "gpu"})
    assert implied_markers({"smoke"}) == frozenset({"smoke"})
