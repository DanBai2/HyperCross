"""Registry-backed publication matrix and deterministic text renderers.

Scientific metrics remain unmodified in :class:`AggregateRecord`.  This module
creates a display-only projection in which numeric metric cells use decimal
round-half-even with exactly four fractional digits.  JSON, CSV, and Markdown
renderers consume the same immutable matrix and never read scientific records
independently.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import Field, model_validator

from benchmark.registry.error_journal import ErrorJournal
from benchmark.registry.run_registry import RunRegistry
from benchmark.schema.base import ValidationIssue, ValidationReport, ValidationSeverity
from benchmark.schema.canonical import normalize_id
from benchmark.schema.environment import FrozenSchemaModel
from benchmark.schema.records import ArtifactReference, ErrorRecord


_DISPLAY_QUANTUM = Decimal("0.0001")
_UNCERTAINTY_TOKENS = (
    "uncertainty",
    "variance",
    "std",
    "confidence",
    "ci",
    "lower",
    "upper",
)

_BASE_COLUMNS: Tuple[str, ...] = (
    "aggregate_id",
    "protocol_name",
    "protocol_hash",
    "model_key",
    "model_display_name",
    "model_class",
    "implementation_kind",
    "parameter_count",
    "source_revision",
    "representation_key",
    "claim_status",
    "uncertainty_estimate",
    "evidence_limitation",
    "claim_explanation",
)


class PublicationSource(FrozenSchemaModel):
    """Non-scientific display context for one registry aggregate.

    Metric values, model identity, protocol hash, representation, and claim
    status are always loaded from the registry.  ``protocol_name`` is supplied
    because the current aggregate schema stores only the protocol hash.
    """

    aggregate_id: str = Field(min_length=1)
    protocol_name: str = Field(min_length=1)
    uncertainty_metric_paths: Tuple[str, ...] = ()
    evidence_limitations: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_display_context(self) -> "PublicationSource":
        for field_name, values in (
            ("uncertainty_metric_paths", self.uncertainty_metric_paths),
            ("evidence_limitations", self.evidence_limitations),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
            if any(not value.strip() or value != value.strip() for value in values):
                raise ValueError(f"{field_name} must contain trimmed non-blank strings")
        return self


class PublicationRow(FrozenSchemaModel):
    """One row whose scalar cells are shared verbatim by every renderer."""

    aggregate_id: str = Field(min_length=1)
    cells: Dict[str, str]


class PublicationMatrix(FrozenSchemaModel):
    """Single authoritative, renderer-independent publication table."""

    schema_version: str = "1.0"
    columns: Tuple[str, ...] = Field(min_length=1)
    rows: Tuple[PublicationRow, ...]

    @model_validator(mode="after")
    def validate_rectangular_matrix(self) -> "PublicationMatrix":
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("publication matrix columns must be unique")
        expected = set(self.columns)
        for row in self.rows:
            if set(row.cells) != expected:
                raise ValueError(
                    "every publication row must contain exactly the matrix columns"
                )
            if row.cells.get("aggregate_id") != row.aggregate_id:
                raise ValueError("row aggregate_id must match its aggregate_id cell")
        return self


@dataclass(frozen=True)
class PublicationBundle:
    """References to one successfully committed three-format publication."""

    json: ArtifactReference
    csv: ArtifactReference
    markdown: ArtifactReference


class PublicationValidationError(RuntimeError):
    """Publication rejection carrying its machine-readable error record."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        super().__init__(
            f"{error_record.error_code}: publication failed at "
            f"{error_record.failed_stage} ({error_record.failing_field})"
        )


@dataclass(frozen=True)
class _ParsedTable:
    columns: Tuple[str, ...]
    rows: Tuple[Dict[str, str], ...]


class ReportGenerator:
    """Build one matrix from registry records and render it in three formats."""

    def __init__(
        self,
        registry: RunRegistry,
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        self.registry = registry
        self.error_journal = error_journal or ErrorJournal(
            registry.path.with_name(f"{registry.path.stem}.publication-errors.jsonl")
        )

    def build_matrix(
        self,
        sources: Union[PublicationSource, Sequence[PublicationSource]],
    ) -> PublicationMatrix:
        """Load aggregates and member runs, then create a display projection.

        Rows are sorted by normalized aggregate identifier.  Metric columns are
        the union of flattened aggregate metric paths and are bytewise sorted,
        which makes output independent of caller and metric-map insertion order.
        """

        normalized_sources = self._normalize_sources(sources)
        projected: list[tuple[PublicationSource, object, object, Dict[str, str]]] = []
        metric_columns: set[str] = set()

        for source in normalized_sources:
            aggregate = self.registry.get_aggregate(source.aggregate_id)
            runs = tuple(
                self.registry.latest_run(run_id)
                for run_id in aggregate.member_run_ids
            )
            if not runs:
                raise ValueError("publication aggregate must contain member runs")
            model_identity = runs[0].model_identity
            if any(run.model_identity != model_identity for run in runs[1:]):
                raise ValueError(
                    "aggregate member runs do not share one Model_Identity"
                )
            if model_identity.model_key != aggregate.model_key:
                raise ValueError("aggregate model_key differs from member Model_Identity")
            if any(
                run.frozen_identity.protocol_hash != aggregate.protocol_hash
                for run in runs
            ):
                raise ValueError("aggregate member protocol identity mismatch")

            flattened = _flatten_metrics(aggregate.metrics)
            display_metrics = {
                path: _display_scalar(value) for path, value in flattened.items()
            }
            metric_columns.update(f"metric.{path}" for path in display_metrics)
            projected.append((source, aggregate, model_identity, display_metrics))

        columns = _BASE_COLUMNS + tuple(
            sorted(metric_columns, key=lambda value: value.encode("utf-8"))
        )
        rows = []
        for source, aggregate, model_identity, display_metrics in projected:
            metric_cells = {
                f"metric.{path}": value for path, value in display_metrics.items()
            }
            selected_uncertainty = _select_uncertainty(
                display_metrics,
                source.uncertainty_metric_paths,
            )
            limitations = tuple(
                dict.fromkeys(
                    (*source.evidence_limitations, *aggregate.claim_decision.blocking_reasons)
                )
            )
            explanation = {
                "supporting_reasons": list(
                    aggregate.claim_decision.supporting_reasons
                ),
                "blocking_reasons": list(aggregate.claim_decision.blocking_reasons),
                "prerequisite_checks": dict(
                    sorted(aggregate.claim_decision.prerequisite_checks.items())
                ),
            }
            cells = {
                "aggregate_id": aggregate.aggregate_id,
                "protocol_name": source.protocol_name,
                "protocol_hash": aggregate.protocol_hash,
                "model_key": model_identity.model_key,
                "model_display_name": model_identity.display_name,
                "model_class": model_identity.class_fully_qualified_name,
                "implementation_kind": model_identity.implementation_kind.value,
                "parameter_count": format_fixed_decimal4(
                    model_identity.parameter_count
                ),
                "source_revision": model_identity.source_revision,
                "representation_key": aggregate.representation_key,
                "claim_status": aggregate.claim_decision.status.value,
                "uncertainty_estimate": (
                    json.dumps(
                        selected_uncertainty,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if selected_uncertainty
                    else "not_reported"
                ),
                "evidence_limitation": json.dumps(
                    list(limitations),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "claim_explanation": json.dumps(
                    explanation,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **metric_cells,
            }
            for column in columns:
                cells.setdefault(column, "")
            rows.append(
                PublicationRow(
                    aggregate_id=aggregate.aggregate_id,
                    cells={column: cells[column] for column in columns},
                )
            )

        return PublicationMatrix(columns=columns, rows=tuple(rows))

    @staticmethod
    def render_json(matrix: PublicationMatrix) -> bytes:
        """Render UTF-8 JSON with every table cell encoded as a string."""

        validated = PublicationMatrix.model_validate(matrix.model_dump(mode="python"))
        payload = {
            "schema_version": validated.schema_version,
            "columns": list(validated.columns),
            "rows": [
                {column: row.cells[column] for column in validated.columns}
                for row in validated.rows
            ],
        }
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def render_csv(matrix: PublicationMatrix) -> bytes:
        """Render RFC-style CSV using the matrix's exact cell strings."""

        validated = PublicationMatrix.model_validate(matrix.model_dump(mode="python"))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=list(validated.columns),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in validated.rows:
            writer.writerow({column: row.cells[column] for column in validated.columns})
        return stream.getvalue().encode("utf-8")

    @staticmethod
    def render_markdown(matrix: PublicationMatrix) -> bytes:
        """Render a GitHub-style Markdown table from the exact matrix cells."""

        validated = PublicationMatrix.model_validate(matrix.model_dump(mode="python"))
        header = "| " + " | ".join(_escape_markdown(c) for c in validated.columns) + " |"
        divider = "| " + " | ".join("---" for _ in validated.columns) + " |"
        lines = [header, divider]
        lines.extend(
            "| "
            + " | ".join(
                _escape_markdown(row.cells[column]) for column in validated.columns
            )
            + " |"
            for row in validated.rows
        )
        return ("\n".join(lines) + "\n").encode("utf-8")

    @staticmethod
    def validate_cross_format(
        json_bytes: bytes,
        csv_bytes: bytes,
        markdown_bytes: bytes,
    ) -> ValidationReport:
        """Parse all renderings and report every cross-format cell conflict.

        Validation compares the full table, which includes model identity,
        protocol identity, claim status, explanations, and metric cells. A
        malformed format is reported as a parse issue rather than being allowed
        to reach the publication write phase.
        """

        parsed: Dict[str, _ParsedTable] = {}
        issues: list[ValidationIssue] = []
        parsers = {
            "json": _parse_json_table,
            "csv": _parse_csv_table,
            "markdown": _parse_markdown_table,
        }
        payloads = {
            "json": json_bytes,
            "csv": csv_bytes,
            "markdown": markdown_bytes,
        }
        for format_name, parser in parsers.items():
            try:
                parsed[format_name] = parser(payloads[format_name])
            except (TypeError, ValueError, UnicodeDecodeError, csv.Error, json.JSONDecodeError) as exc:
                issues.append(
                    ValidationIssue(
                        code="PUBLICATION_FORMAT_PARSE_ERROR",
                        severity=ValidationSeverity.ERROR,
                        field_path=(format_name,),
                        message=f"{format_name} publication could not be parsed",
                        expected_value="rectangular publication table",
                        observed_value=str(exc),
                    )
                )

        if "json" in parsed:
            authority = parsed["json"]
            for format_name in ("csv", "markdown"):
                candidate = parsed.get(format_name)
                if candidate is None:
                    continue
                issues.extend(
                    _table_conflicts(
                        authority,
                        candidate,
                        authority_name="json",
                        candidate_name=format_name,
                    )
                )
        return ValidationReport(issues=tuple(issues))

    def publish_transactional(
        self,
        matrix: PublicationMatrix,
        output_directory: Union[str, Path],
        *,
        basename: str = "publication",
        hand_edited_summary: Optional[Any] = None,
    ) -> PublicationBundle:
        """Validate, stage, and commit JSON, CSV, and Markdown together.

        A supplied hand-edited summary is only a consistency assertion; it is
        never used as publication data. Summary conflicts and renderer drift are
        journaled before any publication target is changed. Commit failures roll
        every target back to its exact pre-transaction bytes.
        """

        validated = PublicationMatrix.model_validate(matrix.model_dump(mode="python"))
        if (
            not isinstance(basename, str)
            or not basename
            or basename != basename.strip()
            or basename in {".", ".."}
            or "/" in basename
            or "\\" in basename
        ):
            raise ValueError("basename must be a non-empty file-name stem")

        rendered = {
            "json": self.render_json(validated),
            "csv": self.render_csv(validated),
            "markdown": self.render_markdown(validated),
        }
        report = self.validate_cross_format(
            rendered["json"], rendered["csv"], rendered["markdown"]
        )
        if not report.passes:
            self._reject(
                code="PUBLICATION_CROSS_FORMAT_VALIDATION_FAILED",
                stage="cross_format_validation",
                field="rendered_formats",
                expected={"formats": ["json", "csv", "markdown"], "cells_equal": True},
                observed=report.machine_details(),
                identifiers={
                    "aggregate_ids": ",".join(row.aggregate_id for row in validated.rows)
                },
                invariants={"publication_files_unchanged": True},
            )

        conflicts = _summary_conflicts(validated, hand_edited_summary)
        if conflicts:
            self._reject(
                code="HAND_EDITED_SUMMARY_CONFLICT",
                stage="hand_edited_conflict_gate",
                field="hand_edited_summary",
                expected={
                    "source": "run_registry",
                    "conflicts": [
                        {
                            "aggregate_id": item["aggregate_id"],
                            "field": item["field"],
                            "value": item["registry_value"],
                        }
                        for item in conflicts
                    ],
                },
                observed={
                    "source": "hand_edited_summary",
                    "conflicts": [
                        {
                            "aggregate_id": item["aggregate_id"],
                            "field": item["field"],
                            "value": item["summary_value"],
                        }
                        for item in conflicts
                    ],
                },
                identifiers={
                    "aggregate_ids": ",".join(
                        sorted(
                            {item["aggregate_id"] for item in conflicts},
                            key=lambda value: value.encode("utf-8"),
                        )
                    )
                },
                invariants={"publication_files_unchanged": True},
            )

        output_root = Path(output_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        targets = {
            "json": output_root / f"{basename}.json",
            "csv": output_root / f"{basename}.csv",
            "markdown": output_root / f"{basename}.md",
        }
        before = {
            name: (target.read_bytes() if target.exists() else None)
            for name, target in targets.items()
        }
        staged: Dict[str, Path] = {}
        backups: Dict[str, Path] = {}
        committed: list[str] = []
        try:
            for name, content in rendered.items():
                staged[name] = _stage_bytes(targets[name], content, suffix=".stage")
                if before[name] is not None:
                    backups[name] = _stage_bytes(
                        targets[name], before[name], suffix=".backup"
                    )

            for name in ("json", "csv", "markdown"):
                os.replace(staged[name], targets[name])
                committed.append(name)

            _sync_directory(output_root)
        except Exception as exc:
            rollback_errors: list[str] = []
            for name in reversed(committed):
                target = targets[name]
                try:
                    backup = backups.get(name)
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
                except OSError as rollback_error:
                    rollback_errors.append(f"{name}: {rollback_error}")
            _cleanup_paths((*staged.values(), *backups.values()))
            unchanged = all(
                (target.read_bytes() if target.exists() else None) == before[name]
                for name, target in targets.items()
            )
            self._reject(
                code="PUBLICATION_TRANSACTION_COMMIT_FAILED",
                stage="transactional_publish",
                field="publication_files",
                expected="all three formats committed together",
                observed={
                    "error": repr(exc),
                    "rollback_errors": rollback_errors,
                },
                identifiers={"basename": basename},
                invariants={"publication_files_unchanged": unchanged},
                cause=exc,
            )

        _cleanup_paths(backups.values())
        references = {
            name: ArtifactReference(
                artifact_id=f"{basename}.{name}",
                relative_path=targets[name].name,
                content_hash=hashlib.sha256(rendered[name]).hexdigest(),
                media_type={
                    "json": "application/json",
                    "csv": "text/csv",
                    "markdown": "text/markdown",
                }[name],
            )
            for name in rendered
        }
        return PublicationBundle(
            json=references["json"],
            csv=references["csv"],
            markdown=references["markdown"],
        )

    def _reject(
        self,
        *,
        code: str,
        stage: str,
        field: str,
        expected: Any,
        observed: Any,
        identifiers: Dict[str, str],
        invariants: Dict[str, bool],
        cause: Optional[BaseException] = None,
    ) -> None:
        record = ErrorRecord(
            error_code=code,
            failed_stage=stage,
            affected_identifiers=identifiers,
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            invariant_checks=invariants,
        )
        failure = PublicationValidationError(record)
        try:
            self.error_journal.append(record)
        except OSError as journal_error:
            failure.add_note(f"error journal append also failed: {journal_error!r}")
        if cause is None:
            raise failure
        raise failure from cause

    @staticmethod
    def _normalize_sources(
        sources: Union[PublicationSource, Sequence[PublicationSource]],
    ) -> Tuple[PublicationSource, ...]:
        if isinstance(sources, PublicationSource):
            values = (sources,)
        elif isinstance(sources, Sequence) and not isinstance(
            sources, (str, bytes, bytearray)
        ):
            values = tuple(sources)
        else:
            raise TypeError("sources must be PublicationSource records")
        if not values:
            raise ValueError("at least one publication source is required")
        if any(not isinstance(source, PublicationSource) for source in values):
            raise TypeError("sources must contain only PublicationSource records")
        identifiers = [source.aggregate_id for source in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("publication sources must not repeat aggregate_id")
        return tuple(
            sorted(
                values,
                key=lambda source: normalize_id(source.aggregate_id).encode("utf-8"),
            )
        )


def format_fixed_decimal4(value: Union[int, float, Decimal]) -> str:
    """Encode a finite number with decimal round-half-even and trailing zeros."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError("publication numeric value must be int, float, or Decimal")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("publication numeric value must be finite")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError("publication numeric value must be finite")
        rounded = decimal_value.quantize(_DISPLAY_QUANTUM, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise ValueError("publication numeric value cannot be represented") from exc
    if rounded == 0:
        rounded = abs(rounded)
    return format(rounded, ".4f")


def _flatten_metrics(
    value: Any,
    *,
    path: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: Dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item).encode("utf-8")):
            if not isinstance(key, str):
                raise ValueError("metric map keys must be strings")
            flattened.update(
                _flatten_metrics(value[key], path=(*path, _escape_metric_segment(key)))
            )
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_metrics(item, path=(*path, str(index))))
        return flattened
    if not path:
        raise ValueError("aggregate metrics must be a mapping or sequence")
    return {".".join(path): value}


def _escape_metric_segment(value: str) -> str:
    return value.replace("~", "~0").replace(".", "~1")


def _display_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return format_fixed_decimal4(value)
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported publication metric scalar: {type(value).__name__}")


def _select_uncertainty(
    display_metrics: Mapping[str, str],
    requested_paths: Tuple[str, ...],
) -> Dict[str, str]:
    if requested_paths:
        missing = tuple(path for path in requested_paths if path not in display_metrics)
        if missing:
            raise ValueError(
                "uncertainty metric paths are absent from aggregate metrics: "
                + ", ".join(missing)
            )
        paths = requested_paths
    else:
        paths = tuple(
            path
            for path in display_metrics
            if any(
                token in path.lower().replace("-", "_").split(".")
                or token in path.lower()
                for token in _UNCERTAINTY_TOKENS
            )
        )
    return {
        path: display_metrics[path]
        for path in sorted(paths, key=lambda value: value.encode("utf-8"))
    }


def _escape_markdown(value: str) -> str:
    escaped = html.escape(value, quote=False)
    return (
        escaped.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


def _parse_json_table(payload: bytes) -> _ParsedTable:
    if not isinstance(payload, bytes):
        raise TypeError("JSON payload must be bytes")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "columns", "rows"}:
        raise ValueError("JSON publication must contain schema_version, columns, and rows")
    columns = _validate_parsed_columns(value["columns"], "JSON")
    rows = _validate_parsed_rows(value["rows"], columns, "JSON")
    return _ParsedTable(columns=columns, rows=rows)


def _parse_csv_table(payload: bytes) -> _ParsedTable:
    if not isinstance(payload, bytes):
        raise TypeError("CSV payload must be bytes")
    records = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    if not records:
        raise ValueError("CSV publication is empty")
    columns = _validate_parsed_columns(records[0], "CSV")
    raw_rows = []
    for index, record in enumerate(records[1:], start=2):
        if len(record) != len(columns):
            raise ValueError(f"CSV row {index} has {len(record)} cells, expected {len(columns)}")
        raw_rows.append(dict(zip(columns, record)))
    rows = _validate_parsed_rows(raw_rows, columns, "CSV")
    return _ParsedTable(columns=columns, rows=rows)


def _parse_markdown_table(payload: bytes) -> _ParsedTable:
    if not isinstance(payload, bytes):
        raise TypeError("Markdown payload must be bytes")
    lines = payload.decode("utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("Markdown publication requires a header and divider")
    columns = _validate_parsed_columns(_split_markdown_row(lines[0]), "Markdown")
    divider = _split_markdown_row(lines[1], decode=False)
    if len(divider) != len(columns) or any(cell.strip() != "---" for cell in divider):
        raise ValueError("Markdown divider does not match the header")
    raw_rows = []
    for index, line in enumerate(lines[2:], start=3):
        values = _split_markdown_row(line)
        if len(values) != len(columns):
            raise ValueError(
                f"Markdown row {index} has {len(values)} cells, expected {len(columns)}"
            )
        raw_rows.append(dict(zip(columns, values)))
    rows = _validate_parsed_rows(raw_rows, columns, "Markdown")
    return _ParsedTable(columns=columns, rows=rows)


def _split_markdown_row(line: str, *, decode: bool = True) -> list[str]:
    if not line.startswith("|") or not line.endswith("|"):
        raise ValueError("Markdown table rows must begin and end with '|'")
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line[1:-1]:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        raise ValueError("Markdown table row ends with an incomplete escape")
    cells.append("".join(current).strip())
    if decode:
        return [html.unescape(cell.replace("<br>", "\n")) for cell in cells]
    return cells


def _validate_parsed_columns(value: Any, format_name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{format_name} columns must be a sequence")
    columns = tuple(value)
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{format_name} columns must be non-empty strings")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{format_name} columns must be unique")
    if "aggregate_id" not in columns:
        raise ValueError(f"{format_name} publication is missing aggregate_id")
    return columns


def _validate_parsed_rows(
    value: Any,
    columns: Tuple[str, ...],
    format_name: str,
) -> Tuple[Dict[str, str], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{format_name} rows must be a sequence")
    rows: list[Dict[str, str]] = []
    identifiers: set[str] = set()
    for index, raw_row in enumerate(value):
        if not isinstance(raw_row, Mapping) or set(raw_row) != set(columns):
            raise ValueError(f"{format_name} row {index} does not match the columns")
        row = {column: raw_row[column] for column in columns}
        if any(not isinstance(cell, str) for cell in row.values()):
            raise ValueError(f"{format_name} row {index} contains a non-string cell")
        aggregate_id = row["aggregate_id"]
        if not aggregate_id or aggregate_id in identifiers:
            raise ValueError(f"{format_name} aggregate_id values must be unique and non-empty")
        identifiers.add(aggregate_id)
        rows.append(row)
    return tuple(rows)


def _table_conflicts(
    authority: _ParsedTable,
    candidate: _ParsedTable,
    *,
    authority_name: str,
    candidate_name: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if candidate.columns != authority.columns:
        issues.append(
            ValidationIssue(
                code="PUBLICATION_COLUMN_MISMATCH",
                severity=ValidationSeverity.ERROR,
                field_path=(candidate_name, "columns"),
                message=f"{candidate_name} columns differ from {authority_name}",
                expected_value=list(authority.columns),
                observed_value=list(candidate.columns),
            )
        )

    expected_rows = {row["aggregate_id"]: row for row in authority.rows}
    observed_rows = {row["aggregate_id"]: row for row in candidate.rows}
    if tuple(expected_rows) != tuple(observed_rows):
        issues.append(
            ValidationIssue(
                code="PUBLICATION_ROW_IDENTITY_MISMATCH",
                severity=ValidationSeverity.ERROR,
                field_path=(candidate_name, "aggregate_id"),
                message=f"{candidate_name} row identities or order differ from {authority_name}",
                expected_value=list(expected_rows),
                observed_value=list(observed_rows),
            )
        )

    shared_columns = tuple(column for column in authority.columns if column in candidate.columns)
    for aggregate_id in expected_rows.keys() & observed_rows.keys():
        for column in shared_columns:
            expected = expected_rows[aggregate_id][column]
            observed = observed_rows[aggregate_id][column]
            if observed != expected:
                issues.append(
                    ValidationIssue(
                        code="PUBLICATION_CELL_MISMATCH",
                        severity=ValidationSeverity.ERROR,
                        field_path=(candidate_name, aggregate_id, column),
                        message=(
                            f"{candidate_name} cell differs from {authority_name} "
                            f"for aggregate {aggregate_id!r}"
                        ),
                        expected_value=expected,
                        observed_value=observed,
                    )
                )
    return issues


def _summary_conflicts(
    matrix: PublicationMatrix,
    summary: Optional[Any],
) -> list[Dict[str, str]]:
    if summary is None:
        return []
    summary_rows = _coerce_summary_rows(summary, matrix)
    registry_rows = {row.aggregate_id: row.cells for row in matrix.rows}
    conflicts: list[Dict[str, str]] = []
    for summary_row in summary_rows:
        aggregate_id = summary_row.get("aggregate_id")
        if not isinstance(aggregate_id, str) or aggregate_id not in registry_rows:
            conflicts.append(
                {
                    "aggregate_id": str(aggregate_id),
                    "field": "aggregate_id",
                    "registry_value": "<missing>",
                    "summary_value": str(aggregate_id),
                }
            )
            continue
        registry_row = registry_rows[aggregate_id]
        for field, raw_value in summary_row.items():
            if field == "aggregate_id":
                continue
            if field not in registry_row:
                conflicts.append(
                    {
                        "aggregate_id": aggregate_id,
                        "field": str(field),
                        "registry_value": "<unknown field>",
                        "summary_value": _summary_scalar(raw_value, None),
                    }
                )
                continue
            summary_value = _summary_scalar(raw_value, registry_row[field])
            if summary_value != registry_row[field]:
                conflicts.append(
                    {
                        "aggregate_id": aggregate_id,
                        "field": field,
                        "registry_value": registry_row[field],
                        "summary_value": summary_value,
                    }
                )
    return sorted(
        conflicts,
        key=lambda item: (
            item["aggregate_id"].encode("utf-8"),
            item["field"].encode("utf-8"),
        ),
    )


def _coerce_summary_rows(
    summary: Any,
    matrix: PublicationMatrix,
) -> Tuple[Mapping[str, Any], ...]:
    if isinstance(summary, PublicationMatrix):
        return tuple(row.cells for row in summary.rows)
    if isinstance(summary, Mapping):
        if "rows" in summary:
            raw_rows = summary["rows"]
            if isinstance(raw_rows, (str, bytes, bytearray)) or not isinstance(raw_rows, Sequence):
                raise TypeError("hand-edited summary rows must be a sequence")
            rows = tuple(raw_rows)
        elif summary and all(isinstance(value, Mapping) for value in summary.values()):
            rows = tuple(
                {"aggregate_id": aggregate_id, **dict(values)}
                for aggregate_id, values in summary.items()
            )
        else:
            rows = (summary,)
    elif isinstance(summary, Sequence) and not isinstance(summary, (str, bytes, bytearray)):
        rows = tuple(summary)
    else:
        raise TypeError("hand-edited summary must be a mapping, row sequence, or PublicationMatrix")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("every hand-edited summary row must be a mapping")
    if len(matrix.rows) == 1:
        only_id = matrix.rows[0].aggregate_id
        rows = tuple(
            row if "aggregate_id" in row else {"aggregate_id": only_id, **dict(row)}
            for row in rows
        )
    elif any("aggregate_id" not in row for row in rows):
        raise ValueError("summary rows require aggregate_id when the matrix has multiple rows")
    return rows


def _summary_scalar(value: Any, registry_value: Optional[str]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)) and registry_value is not None:
        if _looks_fixed_decimal4(registry_value):
            return format_fixed_decimal4(value)
    if value is None:
        return "null"
    return str(value)


def _looks_fixed_decimal4(value: str) -> bool:
    if value.startswith("-"):
        value = value[1:]
    integer, separator, fraction = value.partition(".")
    return bool(integer) and integer.isdigit() and separator == "." and len(fraction) == 4 and fraction.isdigit()


def _stage_bytes(target: Path, content: bytes, *, suffix: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=suffix,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if hashlib.sha256(temporary.read_bytes()).digest() != hashlib.sha256(content).digest():
            raise OSError("staged publication hash mismatch")
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _cleanup_paths(paths: Sequence[Path] | Any) -> None:
    for path in tuple(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PublicationBundle",
    "PublicationMatrix",
    "PublicationRow",
    "PublicationSource",
    "PublicationValidationError",
    "ReportGenerator",
    "format_fixed_decimal4",
]
