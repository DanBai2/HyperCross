"""Strict primitives shared by benchmark schemas.

This module validates shape and basic safety constraints only. Canonical
normalization and serialization belong to the canonical codec task.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated, TypeAlias


_IDENTIFIER_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_identifier(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("identifier must not be empty or whitespace-only")
    if value != value.strip():
        raise ValueError("identifier must not have leading or trailing whitespace")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("identifier must not contain control characters")
    return value


def _validate_content_hash(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("content hash must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _validate_utc_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a UTC timezone")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("timestamp must be expressed in UTC")
    return value


Identifier: TypeAlias = Annotated[str, AfterValidator(_validate_identifier)]
SampleId: TypeAlias = Identifier
GroupId: TypeAlias = Identifier
FoldId: TypeAlias = Identifier
ArtifactId: TypeAlias = Identifier
RunId: TypeAlias = Identifier
ProtocolKey: TypeAlias = Identifier
ModelKey: TypeAlias = Identifier
ClaimId: TypeAlias = Identifier
ContentHash: TypeAlias = Annotated[str, AfterValidator(_validate_content_hash)]
UtcTimestamp: TypeAlias = Annotated[datetime, AfterValidator(_validate_utc_timestamp)]


class StrictSchemaModel(BaseModel):
    """Base class for records that reject undeclared fields and non-finite floats."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
    )


class TaskKind(str, Enum):
    MULTI_TARGET_REGRESSION = "multi_target_regression"
    BINARY_CLASSIFICATION = "binary_classification"
    REMOTE_CLASSIFICATION = "remote_classification"


class ImplementationKind(str, Enum):
    CLASSIC = "classic"
    IN_REPO = "in_repo"
    OFFICIAL = "official"
    ADAPTED_OFFICIAL = "adapted_official"
    PROXY = "proxy"


class ClaimStatus(str, Enum):
    PRIMARY_STRICT = "primary_strict"
    STRICT_CANDIDATE = "strict_candidate"
    DEVELOPMENT = "development"
    EXPLORATORY = "exploratory"
    SANITY_ONLY = "sanity_only"
    PROXY_ONLY = "proxy_only"
    INVALID_OR_LEAKY = "invalid_or_leaky"


class RunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"
    IMPORTED = "imported"


class LimitEnforcement(str, Enum):
    HARD = "hard"
    ADVISORY = "advisory"


class FittedArtifactKind(str, Enum):
    TARGET_NORMALIZATION = "target_normalization"
    PCA = "pca"
    BAND_NORMALIZATION = "band_normalization"
    IMPUTATION = "imputation"
    CLASS_WEIGHTS = "class_weights"
    THRESHOLD = "threshold"
    CALIBRATION = "calibration"


class MetricDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(StrictSchemaModel):
    """One stable, machine-readable validation finding."""

    code: Identifier
    severity: ValidationSeverity
    field_path: Tuple[str, ...]
    message: str
    expected_value: Optional[Any] = None
    observed_value: Optional[Any] = None

    @field_validator("expected_value")
    @classmethod
    def validate_expected_value(cls, value: Any) -> Any:
        return ensure_machine_value(value, field_name="expected_value")

    @field_validator("observed_value")
    @classmethod
    def validate_observed_value(cls, value: Any) -> Any:
        return ensure_machine_value(value, field_name="observed_value")

    @classmethod
    def from_pydantic_error(cls, detail: Dict[str, Any]) -> "ValidationIssue":
        """Translate one Pydantic error without retaining non-serializable context."""
        location = tuple(str(part) for part in detail.get("loc", ()))
        observed = detail.get("input")
        if not _is_machine_value(observed):
            observed = repr(observed)
        return cls(
            code=str(detail.get("type", "validation_error")),
            severity=ValidationSeverity.ERROR,
            field_path=location,
            message=str(detail.get("msg", "validation failed")),
            observed_value=observed,
        )


class ValidationReport(StrictSchemaModel):
    """Validation outcome suitable for JSON logs and API responses."""

    issues: Tuple[ValidationIssue, ...] = ()

    @property
    def passes(self) -> bool:
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is ValidationSeverity.WARNING)

    def machine_details(self) -> Dict[str, Any]:
        """Return a serialization-ready report with an explicit pass flag."""
        return {
            "passes": self.passes,
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
        }

    @classmethod
    def from_validation_error(cls, error: ValidationError) -> "ValidationReport":
        return cls(
            issues=tuple(
                ValidationIssue.from_pydantic_error(detail)
                for detail in error.errors(include_url=False, include_context=False)
            )
        )


ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_schema(model_type: Type[ModelT], value: Any) -> Tuple[Optional[ModelT], ValidationReport]:
    """Validate input and return data or structured details without raising."""
    try:
        return model_type.model_validate(value), ValidationReport()
    except ValidationError as error:
        return None, ValidationReport.from_validation_error(error)


def ensure_machine_value(value: Any, *, field_name: str) -> Any:
    """Reject non-JSON values and non-finite numbers in diagnostic payloads."""
    if not _is_machine_value(value):
        raise ValueError(
            f"{field_name} must contain only JSON values and finite numbers"
        )
    return value


def _is_machine_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_is_machine_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_machine_value(item)
            for key, item in value.items()
        )
    return False
