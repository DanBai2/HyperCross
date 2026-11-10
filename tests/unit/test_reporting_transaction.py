"""Unit tests for cross-format validation and transactional publication."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmark.registry import ErrorJournal, RunRegistry
from benchmark.reporting import (
    PublicationMatrix,
    PublicationRow,
    PublicationValidationError,
    ReportGenerator,
)


def _matrix() -> PublicationMatrix:
    columns = (
        "aggregate_id",
        "model_key",
        "protocol_hash",
        "claim_status",
        "metric.accuracy",
    )
    return PublicationMatrix(
        columns=columns,
        rows=(
            PublicationRow(
                aggregate_id="aggregate-a",
                cells={
                    "aggregate_id": "aggregate-a",
                    "model_key": "model.a",
                    "protocol_hash": "a" * 64,
                    "claim_status": "primary_strict",
                    "metric.accuracy": "0.8124",
                },
            ),
        ),
    )


def _generator(tmp_path: Path) -> tuple[ReportGenerator, Path]:
    journal_path = tmp_path / "publication-errors.jsonl"
    return (
        ReportGenerator(
            RunRegistry(tmp_path / "registry.jsonl"),
            error_journal=ErrorJournal(journal_path),
        ),
        journal_path,
    )


def test_cross_format_validator_compares_model_protocol_claim_and_metric_cells(
    tmp_path: Path,
) -> None:
    generator, _ = _generator(tmp_path)
    matrix = _matrix()
    json_bytes = generator.render_json(matrix)
    csv_bytes = generator.render_csv(matrix)
    markdown_bytes = generator.render_markdown(matrix)

    assert generator.validate_cross_format(json_bytes, csv_bytes, markdown_bytes).passes

    changed = markdown_bytes.replace(b"primary_strict", b"development")
    report = generator.validate_cross_format(json_bytes, csv_bytes, changed)

    assert not report.passes
    assert any(
        issue.code == "PUBLICATION_CELL_MISMATCH"
        and issue.field_path == ("markdown", "aggregate-a", "claim_status")
        and issue.expected_value == "primary_strict"
        and issue.observed_value == "development"
        for issue in report.errors
    )


def test_hand_edited_conflicts_are_journaled_before_any_publication_write(
    tmp_path: Path,
) -> None:
    generator, journal_path = _generator(tmp_path)
    output = tmp_path / "publication"
    output.mkdir()
    old = {
        "publication.json": b"old-json",
        "publication.csv": b"old-csv",
        "publication.md": b"old-markdown",
    }
    for name, content in old.items():
        (output / name).write_bytes(content)

    with pytest.raises(PublicationValidationError) as caught:
        generator.publish_transactional(
            _matrix(),
            output,
            hand_edited_summary={
                "aggregate_id": "aggregate-a",
                "model_key": "manually-renamed-model",
                "protocol_hash": "b" * 64,
                "claim_status": "development",
            },
        )

    assert caught.value.error_record.error_code == "HAND_EDITED_SUMMARY_CONFLICT"
    assert {name: (output / name).read_bytes() for name in old} == old
    record = json.loads(journal_path.read_text("utf-8").splitlines()[0])
    expected = record["expected_value"]["conflicts"]
    observed = record["observed_value"]["conflicts"]
    assert [item["field"] for item in expected] == [
        "claim_status",
        "model_key",
        "protocol_hash",
    ]
    assert [item["value"] for item in expected] == [
        "primary_strict",
        "model.a",
        "a" * 64,
    ]
    assert [item["value"] for item in observed] == [
        "development",
        "manually-renamed-model",
        "b" * 64,
    ]
    assert record["invariant_checks"]["publication_files_unchanged"] is True


def test_transaction_commit_failure_restores_all_existing_publication_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, journal_path = _generator(tmp_path)
    output = tmp_path / "publication"
    output.mkdir()
    old = {
        "publication.json": b"old-json",
        "publication.csv": b"old-csv",
        "publication.md": b"old-markdown",
    }
    for name, content in old.items():
        (output / name).write_bytes(content)

    real_replace = os.replace

    def fail_markdown_commit(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.suffix == ".stage" and destination_path.name == "publication.md":
            raise OSError("injected markdown commit failure")
        real_replace(source, destination)

    monkeypatch.setattr("benchmark.reporting.matrix.os.replace", fail_markdown_commit)

    with pytest.raises(PublicationValidationError) as caught:
        generator.publish_transactional(_matrix(), output)

    assert caught.value.error_record.error_code == "PUBLICATION_TRANSACTION_COMMIT_FAILED"
    assert {name: (output / name).read_bytes() for name in old} == old
    assert not list(output.glob(".*.stage"))
    assert not list(output.glob(".*.backup"))
    record = json.loads(journal_path.read_text("utf-8").splitlines()[0])
    assert record["invariant_checks"]["publication_files_unchanged"] is True


def test_transactional_publish_writes_a_valid_three_format_bundle(tmp_path: Path) -> None:
    generator, _ = _generator(tmp_path)
    output = tmp_path / "publication"

    bundle = generator.publish_transactional(
        _matrix(),
        output,
        hand_edited_summary={"aggregate-a": {"metric.accuracy": 0.8124}},
    )

    json_bytes = (output / "publication.json").read_bytes()
    csv_bytes = (output / "publication.csv").read_bytes()
    markdown_bytes = (output / "publication.md").read_bytes()
    assert generator.validate_cross_format(json_bytes, csv_bytes, markdown_bytes).passes
    assert bundle.json.content_hash != bundle.csv.content_hash
    assert bundle.markdown.relative_path == "publication.md"
