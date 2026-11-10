"""Tests for the package-level, shell-independent entry point."""

from benchmark import __version__
from benchmark.cli import main


def test_module_entry_prints_help_without_side_effects(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "HyperCross protocol-first benchmark infrastructure" in output


def test_module_entry_reports_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("argparse version action must exit")

    assert __version__ in capsys.readouterr().out
