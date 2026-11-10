"""PowerShell-safe module entry point for ``python -m benchmark``."""

from benchmark.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
