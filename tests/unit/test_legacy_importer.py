"""Unit coverage for task 13.1 read-only legacy scanning and import."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from benchmark.legacy import (
    LegacyImportError,
    LegacyImporter,
    LegacySourceScanner,
)
from benchmark.registry import ErrorJournal, RunRegistry
from benchmark.schema import RunStatus, hash_file


NOW = datetime(2027, 2, 3, 4, 5, 6, tzinfo=timezone.utc)


@pytest.fixture
def g_tmp_path() -> Path:
    root = Path(__file__).resolve().parents[2] / ".tmp" / f"legacy-import-{uuid4()}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _require_g_drive(path: Path) -> Path:
    resolved = path.resolve()
    assert resolved.drive.upper() == "G:"
    return resolved


def test_json_import_preserves_source_protocol_numbers_caveats_and_references(
    g_tmp_path: Path,
) -> None:
    root = _require_g_drive(g_tmp_path)
    (root / "tools").mkdir()
    (root / "checkpoints").mkdir()
    runner = root / "tools" / "legacy_runner.py"
    checkpoint = root / "checkpoints" / "model.pt"
    runner.write_bytes(b"print('legacy runner')\r\n")
    checkpoint.write_bytes(b"legacy-checkpoint-bytes\x00\x01")
    source = root / "legacy evidence.json"
    source_bytes = (
        b'{"protocol":"1590 train -> 820 test; selected on inner split",'
        b'"metrics":{"mean_r2":0.915300,"tiny":1e-6},'
        b'"decision":"development only; outer fold used for early stopping",'
        b'"runner":"tools/legacy_runner.py",'
        b'"checkpoint":"checkpoints/model.pt",'
        b'"backup_checkpoint":"F:\\\\archive\\\\ignored.pth"}'
    )
    source.write_bytes(source_bytes)
    checkpoint_before = checkpoint.read_bytes()
    runner_before = runner.read_bytes()

    registry = RunRegistry(root / "runs_v2" / "registry.jsonl")
    importer = LegacyImporter(
        registry,
        ErrorJournal(root / "runs_v2" / "errors.jsonl"),
        root,
    )
    imported = importer.import_source(source, imported_utc=NOW)

    assert imported.status is RunStatus.IMPORTED
    assert imported.source_path == "legacy evidence.json"
    assert imported.source_content_hash == hash_file(source)
    assert imported.protocol_descriptions == (
        "1590 train -> 820 test; selected on inner split",
    )
    assert imported.metrics["legacy_source"]["metrics"] == {
        "mean_r2": "0.915300",
        "tiny": "1e-6",
    }
    assert imported.known_caveats == (
        "development only; outer fold used for early stopping",
    )
    assert imported.runner_metadata == {"runner": "tools/legacy_runner.py"}
    assert len(imported.checkpoint_references) == 1
    assert imported.checkpoint_references[0].source_value == "checkpoints/model.pt"
    assert imported.checkpoint_references[0].content_hash == hash_file(checkpoint)
    assert "runner" not in imported.missing_provenance_fields
    assert imported.claim_decision.prerequisite_checks["runner"] is True
    assert imported.claim_decision.prerequisite_checks["protocol_manifest"] is False

    reloaded = RunRegistry(registry.path).latest_run(imported.run_id)
    assert reloaded == imported
    assert source.read_bytes() == source_bytes
    assert checkpoint.read_bytes() == checkpoint_before
    assert runner.read_bytes() == runner_before


@pytest.mark.parametrize(
    ("name", "content", "expected_format", "expected_value", "expected_protocol", "expected_caveat"),
    [
        (
            "legacy.csv",
            b'dataset,split,metric,value,secondary\r\nHyperLeaf,random_80_20,R2_mean,0.560990,"development, not grouped"\r\n',
            "csv",
            "0.560990",
            "random_80_20",
            "development, not grouped",
        ),
        (
            "legacy.md",
            b"| Model | Protocol | Result | Decision |\r\n|---|---|---|---|\r\n| Route A | fixed 9/4 | 0.9392 | exploratory only |\r\n",
            "markdown",
            "0.9392",
            "fixed 9/4",
            "exploratory only",
        ),
    ],
)
def test_scanner_preserves_csv_and_markdown_cell_text(
    g_tmp_path: Path,
    name: str,
    content: bytes,
    expected_format: str,
    expected_value: str,
    expected_protocol: str,
    expected_caveat: str,
) -> None:
    root = _require_g_drive(g_tmp_path)
    source = root / name
    source.write_bytes(content)

    scan = LegacySourceScanner(root).scan(source)

    assert scan.source_format == expected_format
    assert scan.protocol_descriptions == (expected_protocol,)
    assert scan.known_caveats == (expected_caveat,)
    assert expected_value in json.dumps(scan.raw_payload, ensure_ascii=False)
    assert source.read_bytes() == content


def test_invalid_source_is_rejected_before_run_id_and_registry_append(
    g_tmp_path: Path,
) -> None:
    root = _require_g_drive(g_tmp_path)
    source = root / "broken.json"
    source_bytes = b'{"protocol":"strict", "metric": NaN}'
    source.write_bytes(source_bytes)
    registry_path = root / "runs_v2" / "registry.jsonl"
    journal_path = root / "runs_v2" / "errors.jsonl"
    importer = LegacyImporter(
        RunRegistry(registry_path),
        ErrorJournal(journal_path),
        root,
    )

    with pytest.raises(LegacyImportError) as caught:
        importer.import_source(source, imported_utc=NOW)

    assert not registry_path.exists()
    assert source.read_bytes() == source_bytes
    record = caught.value.error_record
    assert record.failed_stage == "legacy_preallocation_validation"
    assert record.source_path == "broken.json"
    assert record.source_content_hash == hash_file(source)
    assert record.failing_field == "source_content"
    assert record.invariant_checks == {
        "run_id_not_assigned": True,
        "registry_unchanged": True,
        "source_unchanged": True,
        "checkpoint_unchanged": True,
    }
    persisted = json.loads(journal_path.read_text(encoding="utf-8"))
    assert persisted["error_id"] == str(record.error_id)
    assert "run_id" not in persisted["affected_identifiers"]
