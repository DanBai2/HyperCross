"""Safe, allowlisted publication projection and CVPR claim audit output.

The projector deliberately accepts broad source mappings and constructs a much
smaller public schema. Machine-local dataset paths, sensitive log fields, raw
Histology data, patient mappings, and unapproved labels therefore cannot cross
the publication boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, Tuple

from pydantic import Field, field_validator, model_validator

from benchmark.reporting.promotion import BenchmarkTrack
from benchmark.schema.base import ClaimStatus, Identifier, ensure_machine_value
from benchmark.schema.environment import FrozenSchemaModel
from benchmark.schema.records import ArtifactReference


class PublicationSafetyError(ValueError):
    """Raised when required public evidence is missing or unsafe."""


class PublicationProjectionPolicy(FrozenSchemaModel):
    """Per-dataset allowlist for scalar label metadata in public output."""

    approved_label_names: Dict[Identifier, Tuple[Identifier, ...]] = Field(
        default_factory=dict
    )

    @field_validator("approved_label_names")
    @classmethod
    def canonicalize_label_allowlist(
        cls, value: Dict[str, Tuple[str, ...]]
    ) -> Dict[str, Tuple[str, ...]]:
        canonical: Dict[str, Tuple[str, ...]] = {}
        for dataset_key, names in value.items():
            _validate_dataset_key(dataset_key)
            if len(set(names)) != len(names):
                raise ValueError(
                    f"approved labels for {dataset_key!r} must be unique"
                )
            canonical[dataset_key] = tuple(
                sorted(names, key=lambda item: item.encode("utf-8"))
            )
        return dict(
            sorted(canonical.items(), key=lambda item: item[0].encode("utf-8"))
        )


class PublicationClaimView(FrozenSchemaModel):
    """One safe headline or abstract claim with all required audit context."""

    claim_id: Identifier
    track: BenchmarkTrack
    dataset_key: Identifier
    protocol_name: Identifier
    conclusion: str = Field(min_length=1)
    uncertainty_estimate: str = Field(min_length=1)
    evidence_limitation: str = Field(min_length=1)
    claim_status: ClaimStatus
    artifact_references: Tuple[ArtifactReference, ...] = ()
    approved_labels: Dict[Identifier, Any] = Field(default_factory=dict)

    @field_validator(
        "conclusion", "uncertainty_estimate", "evidence_limitation"
    )
    @classmethod
    def validate_public_text(cls, value: str) -> str:
        return _validate_public_text(value)

    @field_validator("dataset_key")
    @classmethod
    def validate_dataset_key(cls, value: str) -> str:
        return _validate_dataset_key(value)

    @field_validator("protocol_name")
    @classmethod
    def validate_protocol_name(cls, value: str) -> str:
        _reject_sensitive_text(value)
        return value

    @field_validator("approved_labels")
    @classmethod
    def validate_approved_labels(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        for item in value.values():
            if item is not None and not isinstance(item, (str, bool, int, float)):
                raise ValueError("approved labels must contain scalar values only")
            ensure_machine_value(item, field_name="approved_labels")
        return value


class TaskLevelConclusion(FrozenSchemaModel):
    """A conclusion scoped to one benchmark task and one public dataset key."""

    track: BenchmarkTrack
    dataset_key: Identifier
    conclusion: str = Field(min_length=1)

    @field_validator("dataset_key")
    @classmethod
    def validate_dataset_key(cls, value: str) -> str:
        return _validate_dataset_key(value)

    @field_validator("conclusion")
    @classmethod
    def validate_conclusion(cls, value: str) -> str:
        return _validate_public_text(value)


class SafeAuditEvent(FrozenSchemaModel):
    """A persisted audit event after recursive sensitive-value removal."""

    event_code: Identifier
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def validate_details(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return ensure_machine_value(value, field_name="details")


class RedactionSummary(FrozenSchemaModel):
    """Counts only redaction categories, never sensitive field names or values."""

    raw_content: int = Field(default=0, ge=0)
    patient_mapping: int = Field(default=0, ge=0)
    unapproved_labels: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    secrets: int = Field(default=0, ge=0)
    environment_dumps: int = Field(default=0, ge=0)
    local_absolute_paths: int = Field(default=0, ge=0)
    combined_overall_scores: int = Field(default=0, ge=0)


class PublicationAuditOutput(FrozenSchemaModel):
    """Authoritative safe claim projection with no cross-track score field."""

    schema_version: Identifier = "1.0"
    task_conclusions: Tuple[TaskLevelConclusion, ...]
    headline_claims: Tuple[PublicationClaimView, ...] = ()
    abstract_evidence: Tuple[PublicationClaimView, ...] = ()
    audit_events: Tuple[SafeAuditEvent, ...] = ()
    redaction_summary: RedactionSummary = Field(default_factory=RedactionSummary)

    @model_validator(mode="after")
    def validate_task_level_coverage(self) -> "PublicationAuditOutput":
        conclusion_keys = {
            (item.track, item.dataset_key) for item in self.task_conclusions
        }
        if len(conclusion_keys) != len(self.task_conclusions):
            raise ValueError("task conclusions must be unique per track and dataset key")

        claims = self.headline_claims + self.abstract_evidence
        missing = sorted(
            {
                (claim.track.value, claim.dataset_key)
                for claim in claims
                if (claim.track, claim.dataset_key) not in conclusion_keys
            }
        )
        if missing:
            raise ValueError(
                "every published claim requires a task-level conclusion: "
                + repr(missing)
            )
        return self


class PublicationProjector:
    """Build safe headline, abstract, task-level, and audit projections.

    Input mappings may contain private execution fields. Only declared public
    fields are copied. Sensitive discarded fields are represented solely by
    category counts in :class:`RedactionSummary`.
    """

    _CLAIM_FIELDS = {
        "claim_id",
        "track",
        "dataset_key",
        "protocol_name",
        "conclusion",
        "task_conclusion",
        "uncertainty_estimate",
        "evidence_limitation",
        "claim_status",
        "artifact_references",
        "labels",
        "destinations",
    }
    _CONCLUSION_FIELDS = {"track", "dataset_key", "conclusion"}
    _EVENT_FIELDS = {"event_code", "details"}

    def __init__(
        self, policy: PublicationProjectionPolicy | None = None
    ) -> None:
        self.policy = policy or PublicationProjectionPolicy()

    def project(
        self,
        *,
        claims: Iterable[Mapping[str, Any]],
        task_conclusions: Iterable[Mapping[str, Any]],
        audit_events: Iterable[Mapping[str, Any]] = (),
    ) -> PublicationAuditOutput:
        """Return one deterministic public audit projection.

        The output intentionally has no ``overall_score`` member. Any supplied
        combined score is discarded and recorded as a redaction category.
        """

        counts = _new_counts()
        headline: list[PublicationClaimView] = []
        abstract: list[PublicationClaimView] = []

        for raw_source in claims:
            source = _as_mapping(raw_source, kind="claim")
            self._observe_discarded(source, self._CLAIM_FIELDS, counts)
            claim, destinations = self._project_claim(source, counts)
            if "headline" in destinations:
                headline.append(claim)
            if "abstract" in destinations:
                abstract.append(claim)

        conclusions: list[TaskLevelConclusion] = []
        for raw_source in task_conclusions:
            source = _as_mapping(raw_source, kind="task conclusion")
            self._observe_discarded(source, self._CONCLUSION_FIELDS, counts)
            conclusions.append(
                TaskLevelConclusion.model_validate(
                    {
                        "track": _required(source, "track"),
                        "dataset_key": _required(source, "dataset_key"),
                        "conclusion": _required(source, "conclusion"),
                    }
                )
            )

        events: list[SafeAuditEvent] = []
        for raw_source in audit_events:
            source = _as_mapping(raw_source, kind="audit event")
            self._observe_discarded(source, self._EVENT_FIELDS, counts)
            details = source.get("details", {})
            if not isinstance(details, Mapping):
                raise PublicationSafetyError("audit event details must be a mapping")
            safe_details = _sanitize_mapping(details, counts)
            events.append(
                SafeAuditEvent(
                    event_code=_required(source, "event_code"),
                    details=safe_details,
                )
            )

        sort_claim = lambda item: (
            item.track.value.encode("utf-8"),
            item.dataset_key.encode("utf-8"),
            item.claim_id.encode("utf-8"),
        )
        conclusions.sort(
            key=lambda item: (
                item.track.value.encode("utf-8"),
                item.dataset_key.encode("utf-8"),
            )
        )
        headline.sort(key=sort_claim)
        abstract.sort(key=sort_claim)
        events.sort(key=lambda item: item.event_code.encode("utf-8"))

        return PublicationAuditOutput(
            task_conclusions=tuple(conclusions),
            headline_claims=tuple(headline),
            abstract_evidence=tuple(abstract),
            audit_events=tuple(events),
            redaction_summary=RedactionSummary(**counts),
        )

    def _project_claim(
        self,
        source: Mapping[str, Any],
        counts: Dict[str, int],
    ) -> tuple[PublicationClaimView, Tuple[str, ...]]:
        dataset_key = _required(source, "dataset_key")
        references = tuple(
            _project_artifact_reference(item)
            for item in _as_sequence(
                source.get("artifact_references", ()),
                field="artifact_references",
            )
        )
        labels = source.get("labels", {})
        if not isinstance(labels, Mapping):
            raise PublicationSafetyError("labels must be a mapping")
        approved_names = set(
            self.policy.approved_label_names.get(dataset_key, ())
            if isinstance(dataset_key, str)
            else ()
        )
        approved_labels: Dict[str, Any] = {}
        for name, value in labels.items():
            if not isinstance(name, str) or name not in approved_names:
                counts["unapproved_labels"] += 1
                continue
            if value is not None and not isinstance(value, (str, bool, int, float)):
                counts["raw_content"] += 1
                continue
            ensure_machine_value(value, field_name=f"labels.{name}")
            approved_labels[name] = value

        destinations = _destinations(source.get("destinations", ("headline", "abstract")))
        claim = PublicationClaimView.model_validate(
            {
                "claim_id": _required(source, "claim_id"),
                "track": _required(source, "track"),
                "dataset_key": dataset_key,
                "protocol_name": _required(source, "protocol_name"),
                "conclusion": _required(source, "conclusion"),
                "uncertainty_estimate": _required(source, "uncertainty_estimate"),
                "evidence_limitation": _required(source, "evidence_limitation"),
                "claim_status": _required(source, "claim_status"),
                "artifact_references": references,
                "approved_labels": approved_labels,
            }
        )
        return claim, destinations

    @staticmethod
    def _observe_discarded(
        source: Mapping[str, Any],
        allowed_fields: set[str],
        counts: Dict[str, int],
    ) -> None:
        for key, value in source.items():
            if key in allowed_fields:
                continue
            category = _sensitive_category(str(key))
            if category is not None:
                counts[category] += 1
            elif _contains_absolute_path(value):
                counts["local_absolute_paths"] += 1


_SENSITIVE_KEY_PARTS = re.compile(r"[^a-z0-9]+")
_CREDENTIAL_VALUE = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~+/=-]+|(?:access[_-]?token|password|secret|api[_-]?key)\s*[:=])"
)
_REDACTED = "[REDACTED]"
_REDACTED_PATH = "[REDACTED_LOCAL_ABSOLUTE_PATH]"
_REDACTED_CONTENT = "[REDACTED_RAW_CONTENT]"


def _new_counts() -> Dict[str, int]:
    return {
        "raw_content": 0,
        "patient_mapping": 0,
        "unapproved_labels": 0,
        "tokens": 0,
        "secrets": 0,
        "environment_dumps": 0,
        "local_absolute_paths": 0,
        "combined_overall_scores": 0,
    }


def _as_mapping(value: Any, *, kind: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    raise PublicationSafetyError(f"{kind} source must be a mapping")


def _required(source: Mapping[str, Any], field: str) -> Any:
    if field not in source:
        raise PublicationSafetyError(f"publication source is missing required field {field!r}")
    return source[field]


def _as_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PublicationSafetyError(f"{field} must be a sequence")
    return value


def _destinations(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        supplied = (value,)
    else:
        supplied = tuple(_as_sequence(value, field="destinations"))
    if not supplied:
        raise PublicationSafetyError("destinations must contain headline or abstract")
    if any(item not in {"headline", "abstract"} for item in supplied):
        raise PublicationSafetyError("destinations may contain only headline and abstract")
    return tuple(dict.fromkeys(supplied))


def _project_artifact_reference(value: Any) -> ArtifactReference:
    reference = (
        value
        if isinstance(value, ArtifactReference)
        else ArtifactReference.model_validate(value)
    )
    relative = PurePosixPath(reference.relative_path.replace("\\", "/")).as_posix()
    return ArtifactReference(
        artifact_id=reference.artifact_id,
        relative_path=relative,
        content_hash=reference.content_hash,
        media_type=reference.media_type,
    )


def _validate_dataset_key(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("dataset_key must be non-empty without surrounding whitespace")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if (
        windows.is_absolute()
        or bool(windows.drive)
        or posix.is_absolute()
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or "://" in value
    ):
        raise ValueError("dataset_key must be a stable key, not a path or URI")
    _reject_sensitive_text(value)
    return value


def _validate_public_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("public text must be non-empty without surrounding whitespace")
    if _looks_absolute_path(value):
        raise ValueError("public text must not contain a local absolute path")
    _reject_sensitive_text(value)
    return value


def _reject_sensitive_text(value: str) -> None:
    if _CREDENTIAL_VALUE.search(value):
        raise ValueError("public text must not contain credentials or secrets")


def _sensitive_category(key: str) -> str | None:
    normalized = _SENSITIVE_KEY_PARTS.sub("_", key.casefold()).strip("_")
    compact = normalized.replace("_", "")
    if compact in {"overallscore", "combinedoverallscore", "crosstrackscore"}:
        return "combined_overall_scores"
    if compact in {
        "environment",
        "environ",
        "env",
        "environmentvariables",
        "environmentvariabledump",
        "envvars",
    }:
        return "environment_dumps"
    if "patient" in compact and any(
        marker in compact for marker in ("mapping", "identity", "identifier")
    ):
        return "patient_mapping"
    if "token" in compact or compact in {"authorization", "bearer"}:
        return "tokens"
    if any(
        marker in compact
        for marker in (
            "secret",
            "password",
            "credential",
            "apikey",
            "privatekey",
            "deidentificationkey",
        )
    ):
        return "secrets"
    if compact in {"labels", "labelvalues", "groundtruth", "ytrue"}:
        return "unapproved_labels"
    if compact in {
        "raw",
        "rawcontent",
        "rawsample",
        "rawsamplecontent",
        "rawcube",
        "cubecontent",
        "samplebytes",
        "payloadbytes",
    } or ("raw" in compact and any(word in compact for word in ("cube", "sample", "content"))):
        return "raw_content"
    return None


def _sanitize_mapping(
    value: Mapping[Any, Any], counts: Dict[str, int]
) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            counts["raw_content"] += 1
            continue
        category = _sensitive_category(key)
        if category is not None:
            counts[category] += 1
            continue
        sanitized[key] = _sanitize_value(item, counts)
    return sanitized


def _sanitize_value(value: Any, counts: Dict[str, int]) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value, counts)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, counts) for item in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            counts["raw_content"] += 1
            return _REDACTED_CONTENT
        return value
    if isinstance(value, str):
        if _looks_absolute_path(value):
            counts["local_absolute_paths"] += 1
            return _REDACTED_PATH
        if _CREDENTIAL_VALUE.search(value):
            counts["secrets"] += 1
            return _REDACTED
        return value
    counts["raw_content"] += 1
    return _REDACTED_CONTENT


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return _looks_absolute_path(value)
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _looks_absolute_path(value: str) -> bool:
    if value.casefold().startswith("file:"):
        return True
    return (
        PureWindowsPath(value).is_absolute()
        or bool(PureWindowsPath(value).drive)
        or PurePosixPath(value).is_absolute()
    )


__all__ = [
    "PublicationAuditOutput",
    "PublicationClaimView",
    "PublicationProjectionPolicy",
    "PublicationProjector",
    "PublicationSafetyError",
    "RedactionSummary",
    "SafeAuditEvent",
    "TaskLevelConclusion",
]
