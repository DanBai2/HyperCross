"""Read-only scanning and append-only import of historical benchmark evidence."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Optional
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator

from benchmark.registry.error_journal import ErrorJournal
from benchmark.registry.records import (
    ClaimDecision,
    FrozenRunIdentity,
    LegacySourceReference,
    Provenance,
    RunRecord,
)
from benchmark.registry.run_registry import RunRegistry
from benchmark.schema.base import (
    ClaimStatus,
    ImplementationKind,
    LimitEnforcement,
    RunStatus,
    StrictSchemaModel,
)
from benchmark.schema.canonical import hash_file, hash_object
from benchmark.schema.models import InputContract, ModelIdentity
from benchmark.schema.records import ResourceBudget, ResourceLimit
from benchmark.schema.records import ErrorRecord


_SUPPORTED_SUFFIXES = {".json": "json", ".csv": "csv", ".md": "markdown", ".markdown": "markdown"}
_MISSING_PROVENANCE_FIELDS = (
    "code_revision",
    "environment_lock",
    "prediction_artifact",
    "protocol_manifest",
    "runner",
    "sample_manifest",
)
_PROTOCOL_KEYS = {"protocol", "protocol_description", "split"}
_CAVEAT_TOKENS = ("caveat", "decision", "limitation", "note", "reason", "secondary")
_RUNNER_TOKENS = ("runner", "script", "command")
_CHECKPOINT_TOKENS = ("checkpoint", "ckpt")
_PATH_SUFFIXES = (".py", ".pt", ".pth", ".ckpt")
_MARKDOWN_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_SAFE_KEY = re.compile(r"[^a-z0-9]+")


class LegacySourceScan(StrictSchemaModel):
    """Validated, source-bound evidence produced without mutating any file."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    source_path: str = Field(min_length=1)
    source_content_hash: str
    source_format: str
    raw_payload: Any
    protocol_descriptions: tuple[str, ...] = ()
    known_caveats: tuple[str, ...] = ()
    runner_metadata: dict[str, Any] = Field(default_factory=dict)
    runner_references: tuple[LegacySourceReference, ...] = ()
    checkpoint_references: tuple[LegacySourceReference, ...] = ()
    missing_provenance_fields: tuple[str, ...] = ()

    @field_validator("source_content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("source_content_hash must be lowercase SHA-256")
        return value


class LegacyImportError(ValueError):
    """A rejected import with its durable machine-readable diagnostic."""

    def __init__(self, error_record: ErrorRecord) -> None:
        super().__init__(
            f"{error_record.error_code}: {error_record.failing_field}"
        )
        self.error_record = error_record


class _ScanFailure(ValueError):
    def __init__(
        self,
        *,
        field: str,
        expected: Any,
        observed: Any,
        source_path: str,
        source_hash: Optional[str],
    ) -> None:
        super().__init__(f"invalid legacy source field {field}")
        self.field = field
        self.expected = expected
        self.observed = observed
        self.source_path = source_path
        self.source_hash = source_hash


class LegacySourceScanner:
    """Parse JSON, CSV, and Markdown evidence using read-only file access.

    The scanner is deliberately rooted at one G-drive repository. Absolute
    references on other drives are ignored; relative runner/checkpoint paths
    are resolved against the source file and hashed only when readable.
    """

    def __init__(self, workspace_root: Path | str) -> None:
        root = Path(workspace_root).resolve()
        if root.drive.upper() != "G:":
            raise ValueError("legacy workspace_root must be on the G drive")
        self._root = root

    @property
    def workspace_root(self) -> Path:
        return self._root

    def scan(self, source_path: Path | str) -> LegacySourceScan:
        source = self._resolve_source(source_path)
        stored_path = source.relative_to(self._root).as_posix()
        source_hash: Optional[str] = None
        try:
            if not source.is_file():
                raise _ScanFailure(
                    field="source_path",
                    expected="an existing regular file under the G-drive workspace",
                    observed=stored_path,
                    source_path=stored_path,
                    source_hash=None,
                )
            source_hash = hash_file(source)
            source_bytes = source.read_bytes()
            if not source_bytes:
                raise _ScanFailure(
                    field="source_content",
                    expected="non-empty UTF-8 content",
                    observed="empty file",
                    source_path=stored_path,
                    source_hash=source_hash,
                )
            try:
                text = source_bytes.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise _ScanFailure(
                    field="source_encoding",
                    expected="UTF-8",
                    observed=f"decode error at byte {exc.start}",
                    source_path=stored_path,
                    source_hash=source_hash,
                ) from exc

            source_format = _SUPPORTED_SUFFIXES.get(source.suffix.lower())
            if source_format is None:
                raise _ScanFailure(
                    field="source_format",
                    expected=sorted(_SUPPORTED_SUFFIXES),
                    observed=source.suffix.lower(),
                    source_path=stored_path,
                    source_hash=source_hash,
                )
            raw_payload = self._parse(text, source_format, stored_path, source_hash)
            protocols = tuple(self._values_for_keys(raw_payload, _PROTOCOL_KEYS))
            caveats = tuple(self._caveat_values(raw_payload))
            runner_metadata = self._runner_metadata(raw_payload)
            runner_refs = tuple(
                self._references(raw_payload, source.parent, "runner", _RUNNER_TOKENS)
            )
            checkpoint_refs = tuple(
                self._references(
                    raw_payload, source.parent, "checkpoint", _CHECKPOINT_TOKENS
                )
            )
            missing = self._missing_provenance(raw_payload, runner_metadata)
            return LegacySourceScan(
                source_path=stored_path,
                source_content_hash=source_hash,
                source_format=source_format,
                raw_payload=raw_payload,
                protocol_descriptions=protocols,
                known_caveats=caveats,
                runner_metadata=runner_metadata,
                runner_references=runner_refs,
                checkpoint_references=checkpoint_refs,
                missing_provenance_fields=missing,
            )
        except _ScanFailure:
            raise
        except (csv.Error, json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, _ScanFailure):
                raise
            raise _ScanFailure(
                field="source_content",
                expected="valid source content for the file format",
                observed=str(exc),
                source_path=stored_path,
                source_hash=source_hash,
            ) from exc

    def _resolve_source(self, source_path: Path | str) -> Path:
        supplied = Path(source_path)
        source = supplied if supplied.is_absolute() else self._root / supplied
        resolved = source.resolve()
        if resolved.drive.upper() != "G:":
            raise _ScanFailure(
                field="source_path",
                expected="a path on the G drive",
                observed=str(source),
                source_path=str(source),
                source_hash=None,
            )
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise _ScanFailure(
                field="source_path",
                expected="a path inside the configured workspace",
                observed=str(resolved),
                source_path=str(resolved),
                source_hash=None,
            ) from exc
        return resolved

    @staticmethod
    def _parse(
        text: str, source_format: str, source_path: str, source_hash: str
    ) -> Any:
        if source_format == "json":
            def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key {key!r}")
                    result[key] = value
                return result

            def reject_constant(value: str) -> Any:
                raise ValueError(f"non-finite JSON number {value!r}")

            return json.loads(
                text,
                parse_int=str,
                parse_float=str,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicate_keys,
            )
        if source_format == "csv":
            reader = csv.reader(io.StringIO(text, newline=""))
            rows = list(reader)
            if not rows or not rows[0]:
                raise ValueError("CSV must contain a header row")
            headers = rows[0]
            if any(not header for header in headers) or len(set(headers)) != len(headers):
                raise ValueError("CSV headers must be non-empty and unique")
            parsed_rows: list[dict[str, str]] = []
            for index, row in enumerate(rows[1:], start=2):
                if len(row) != len(headers):
                    raise ValueError(
                        f"CSV row {index} has {len(row)} cells; expected {len(headers)}"
                    )
                parsed_rows.append(dict(zip(headers, row)))
            return {"headers": headers, "rows": parsed_rows}

        tables = _parse_markdown_tables(text)
        if not tables:
            raise ValueError("Markdown source must contain at least one table")
        return {"document": text, "tables": tables}

    @staticmethod
    def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str, Any]]:
        if isinstance(value, dict):
            for key, item in value.items():
                current = (*path, key)
                yield current, key, item
                yield from LegacySourceScanner._walk(item, current)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from LegacySourceScanner._walk(item, (*path, str(index)))

    @classmethod
    def _values_for_keys(cls, payload: Any, keys: set[str]) -> Iterable[str]:
        for _, key, value in cls._walk(payload):
            if key.casefold() in keys:
                yield from _string_leaves(value)

    @classmethod
    def _caveat_values(cls, payload: Any) -> Iterable[str]:
        for _, key, value in cls._walk(payload):
            lowered = key.casefold()
            if any(token in lowered for token in _CAVEAT_TOKENS):
                yield from _string_leaves(value)

    @classmethod
    def _runner_metadata(cls, payload: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for path, key, value in cls._walk(payload):
            lowered = key.casefold()
            if any(token in lowered for token in _RUNNER_TOKENS):
                metadata["/".join(path)] = value
        return metadata

    def _references(
        self,
        payload: Any,
        source_parent: Path,
        kind: str,
        key_tokens: tuple[str, ...],
    ) -> Iterable[LegacySourceReference]:
        for _, key, value in self._walk(payload):
            if not any(token in key.casefold() for token in key_tokens):
                continue
            for source_value in _string_leaves(value):
                if not source_value.casefold().endswith(_PATH_SUFFIXES):
                    continue
                windows = PureWindowsPath(source_value)
                if windows.drive and windows.drive.upper() != "G:":
                    continue
                candidate = Path(source_value)
                if not candidate.is_absolute():
                    candidate = source_parent / candidate
                resolved = candidate.resolve()
                if resolved.drive.upper() != "G:":
                    continue
                resolved_path = self._display_reference_path(resolved)
                content_hash = hash_file(resolved) if resolved.is_file() else None
                yield LegacySourceReference(
                    reference_kind=kind,
                    source_value=source_value,
                    resolved_path=resolved_path,
                    content_hash=content_hash,
                )

    def _display_reference_path(self, path: Path) -> str:
        try:
            return path.relative_to(self._root).as_posix()
        except ValueError:
            return path.as_posix()

    @classmethod
    def _missing_provenance(
        cls, payload: Any, runner_metadata: dict[str, Any]
    ) -> tuple[str, ...]:
        observed_keys = {key.casefold() for _, key, _ in cls._walk(payload)}
        present = {
            "runner": bool(runner_metadata),
            "sample_manifest": "sample_manifest" in observed_keys,
            "protocol_manifest": "protocol_manifest" in observed_keys,
            "prediction_artifact": "prediction_artifact" in observed_keys,
            "code_revision": bool(
                observed_keys.intersection({"code_revision", "commit", "git_commit"})
            ),
            "environment_lock": "environment_lock" in observed_keys,
        }
        return tuple(field for field in _MISSING_PROVENANCE_FIELDS if not present[field])


class LegacyImporter:
    """Validate a source completely, then assign and append one imported Run_ID."""

    def __init__(
        self,
        registry: RunRegistry,
        error_journal: ErrorJournal,
        workspace_root: Path | str,
    ) -> None:
        self._registry = registry
        self._errors = error_journal
        self._scanner = LegacySourceScanner(workspace_root)

    def import_source(
        self,
        source_path: Path | str,
        *,
        imported_utc: Optional[datetime] = None,
    ) -> RunRecord:
        registry_before = _optional_bytes(self._registry.path)
        try:
            scan = self._scanner.scan(source_path)
            self._validate_snapshots(scan)
            timestamp = imported_utc or datetime.now(timezone.utc)
            if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
                raise _ScanFailure(
                    field="imported_utc",
                    expected="a UTC-aware timestamp",
                    observed=str(timestamp),
                    source_path=scan.source_path,
                    source_hash=scan.source_content_hash,
                )
            record = self._build_record(scan, timestamp, str(uuid4()))
            self._registry.append_imported(record)
            return record
        except _ScanFailure as exc:
            unchanged = registry_before == _optional_bytes(self._registry.path)
            error = ErrorRecord(
                error_code="LEGACY_IMPORT_VALIDATION_FAILED",
                failed_stage="legacy_preallocation_validation",
                affected_identifiers={"source_path": exc.source_path},
                failing_field=exc.field,
                expected_value=exc.expected,
                observed_value=exc.observed,
                invariant_checks={
                    "run_id_not_assigned": True,
                    "registry_unchanged": unchanged,
                    "source_unchanged": _hash_matches(exc.source_path, exc.source_hash, self._scanner.workspace_root),
                    "checkpoint_unchanged": True,
                },
                source_path=exc.source_path,
                source_content_hash=exc.source_hash,
            )
            self._errors.append(error)
            raise LegacyImportError(error) from exc

    def _validate_snapshots(self, scan: LegacySourceScan) -> None:
        source = self._scanner.workspace_root / Path(scan.source_path)
        current_hash = hash_file(source)
        if current_hash != scan.source_content_hash:
            raise _ScanFailure(
                field="source_content_hash",
                expected=scan.source_content_hash,
                observed=current_hash,
                source_path=scan.source_path,
                source_hash=current_hash,
            )
        for reference in (*scan.runner_references, *scan.checkpoint_references):
            if reference.content_hash is None or reference.resolved_path is None:
                continue
            path = Path(reference.resolved_path)
            if not path.is_absolute():
                path = self._scanner.workspace_root / path
            observed = hash_file(path) if path.is_file() else None
            if observed != reference.content_hash:
                raise _ScanFailure(
                    field=f"{reference.reference_kind}_content_hash",
                    expected=reference.content_hash,
                    observed=observed,
                    source_path=scan.source_path,
                    source_hash=scan.source_content_hash,
                )

    @staticmethod
    def _build_record(
        scan: LegacySourceScan, timestamp: datetime, run_id: str
    ) -> RunRecord:
        source_token = scan.source_content_hash[:16]
        stem = _SAFE_KEY.sub("-", Path(scan.source_path).stem.casefold()).strip("-") or "source"
        model_key = f"legacy.{stem}.{source_token}"
        dataset_hash = hash_object(
            {"legacy_placeholder": "dataset_unavailable", "source_hash": scan.source_content_hash}
        )
        protocol_hash = hash_object(
            {
                "legacy_protocol_descriptions": scan.protocol_descriptions,
                "source_hash": scan.source_content_hash,
            }
        )
        config_hash = hash_object(
            {"legacy_source_payload": scan.raw_payload, "source_hash": scan.source_content_hash}
        )
        environment_hash = hash_object(
            {"legacy_placeholder": "environment_lock_unavailable"}
        )
        checks = {
            field: field not in scan.missing_provenance_fields
            for field in _MISSING_PROVENANCE_FIELDS
        }
        blocking = tuple(
            f"missing_{field}" for field in scan.missing_provenance_fields
        )
        source_revision = _first_key_value(
            scan.raw_payload, {"code_revision", "commit", "git_commit"}
        ) or "legacy-unavailable"
        return RunRecord(
            run_id=run_id,
            frozen_identity=FrozenRunIdentity(
                dataset_hash=dataset_hash,
                protocol_hash=protocol_hash,
                model_key=model_key,
                config_hash=config_hash,
                fold_id="legacy-import",
                seed=0,
                environment_lock_hash=environment_hash,
                environment_compatibility_policy="legacy-missing-fields-explicit",
            ),
            model_identity=ModelIdentity(
                model_key=model_key,
                display_name=f"{Path(scan.source_path).name} legacy evidence",
                class_fully_qualified_name="benchmark.legacy.importer.ImportedLegacyEvidence",
                implementation_kind=ImplementationKind.IN_REPO,
                parameter_count=0,
                source_revision=source_revision,
            ),
            input_contract=InputContract(
                representation_key="legacy-evidence-source.v1",
                modalities=("metadata",),
                tensor_shapes={"metadata": (1,)},
                dtypes={"metadata": "utf8"},
                channel_or_band_order={"metadata": ("source_order",)},
                preprocessing_key="legacy-read-only.v1",
            ),
            provenance=Provenance(
                code_revision=source_revision,
                source_hash=scan.source_content_hash,
                environment_hash=environment_hash,
                config_hash=config_hash,
                dataset_hash=dataset_hash,
                artifact_hashes={
                    reference.resolved_path or reference.source_value: reference.content_hash
                    for reference in (*scan.runner_references, *scan.checkpoint_references)
                    if reference.content_hash is not None
                },
            ),
            metrics={"legacy_source": scan.raw_payload},
            resource_budget=ResourceBudget(
                limits=(
                    ResourceLimit(
                        field="legacy_import_records",
                        value=1,
                        unit="count",
                        enforcement=LimitEnforcement.HARD,
                    ),
                )
            ),
            status=RunStatus.IMPORTED,
            claim_decision=ClaimDecision(
                status=ClaimStatus.EXPLORATORY,
                supporting_reasons=("legacy_evidence_imported",),
                blocking_reasons=blocking,
                prerequisite_checks=checks,
            ),
            source_path=scan.source_path,
            source_content_hash=scan.source_content_hash,
            legacy_source_format=scan.source_format,
            protocol_descriptions=scan.protocol_descriptions,
            known_caveats=scan.known_caveats,
            runner_metadata=scan.runner_metadata,
            checkpoint_references=scan.checkpoint_references,
            missing_provenance_fields=scan.missing_provenance_fields,
            started_utc=timestamp,
            completed_utc=timestamp,
        )


def _parse_markdown_tables(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    tables: list[dict[str, Any]] = []
    index = 0
    while index + 1 < len(lines):
        header = _split_markdown_row(lines[index])
        separator = _split_markdown_row(lines[index + 1])
        if (
            len(header) >= 2
            and len(header) == len(separator)
            and all(_MARKDOWN_SEPARATOR.fullmatch(cell.strip()) for cell in separator)
        ):
            if any(not cell for cell in header) or len(set(header)) != len(header):
                raise ValueError("Markdown table headers must be non-empty and unique")
            rows: list[dict[str, str]] = []
            index += 2
            while index < len(lines):
                cells = _split_markdown_row(lines[index])
                if len(cells) != len(header):
                    break
                rows.append(dict(zip(header, cells)))
                index += 1
            tables.append({"headers": header, "rows": rows})
            continue
        index += 1
    return tables


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in stripped:
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
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def _string_leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_leaves(item)


def _first_key_value(payload: Any, keys: set[str]) -> Optional[str]:
    for _, key, value in LegacySourceScanner._walk(payload):
        if key.casefold() in keys:
            return next(_string_leaves(value), None)
    return None


def _optional_bytes(path: Path) -> Optional[bytes]:
    return path.read_bytes() if path.exists() else None


def _hash_matches(source_path: str, expected: Optional[str], root: Path) -> bool:
    if expected is None:
        return True
    path = Path(source_path)
    if not path.is_absolute():
        path = root / path
    try:
        return path.is_file() and hash_file(path) == expected
    except OSError:
        return False


__all__ = [
    "LegacyImportError",
    "LegacyImporter",
    "LegacySourceScan",
    "LegacySourceScanner",
]
