"""Core records shared across benchmark components."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator
from typing_extensions import Annotated

from .base import (
    ArtifactId,
    ContentHash,
    Identifier,
    LimitEnforcement,
    StrictSchemaModel,
    UtcTimestamp,
    ensure_machine_value,
)


PositiveStrictInteger = Annotated[StrictInt, Field(gt=0)]


class ArtifactReference(StrictSchemaModel):
    """Portable reference to an immutable artifact and its verified content."""

    artifact_id: ArtifactId
    relative_path: str
    content_hash: ContentHash
    media_type: Optional[str] = None

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("artifact path must not be empty")
        if value != value.strip():
            raise ValueError("artifact path must not have surrounding whitespace")
        windows_path = PureWindowsPath(value)
        posix_path = PurePosixPath(value)
        if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
            raise ValueError("artifact path must be relative")
        if ".." in windows_path.parts or ".." in posix_path.parts:
            raise ValueError("artifact path must not traverse parent directories")
        return value


# The concise design name remains available to all components.
ArtifactRef = ArtifactReference


class ErrorRecord(StrictSchemaModel):
    """Machine-readable failure record kept outside protected business state."""

    error_id: UUID = Field(default_factory=uuid4)
    error_code: Identifier
    failed_stage: Identifier
    affected_identifiers: Dict[str, str] = Field(default_factory=dict)
    failing_field: Identifier
    expected_value: Optional[Any] = None
    observed_value: Optional[Any] = None
    missing_ids: Tuple[Identifier, ...] = ()
    unexpected_ids: Tuple[Identifier, ...] = ()
    invariant_checks: Dict[str, bool] = Field(default_factory=dict)
    source_path: Optional[str] = None
    source_content_hash: Optional[ContentHash] = None
    created_utc: UtcTimestamp = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("expected_value")
    @classmethod
    def validate_expected_value(cls, value: Any) -> Any:
        return ensure_machine_value(value, field_name="expected_value")

    @field_validator("observed_value")
    @classmethod
    def validate_observed_value(cls, value: Any) -> Any:
        return ensure_machine_value(value, field_name="observed_value")


class ResourceLimit(StrictSchemaModel):
    """One finite, positive resource bound with explicit enforcement."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    field: Identifier
    value: PositiveStrictInteger
    unit: Identifier
    enforcement: LimitEnforcement


class ResourceBudget(StrictSchemaModel):
    """Non-empty immutable collection of unambiguous resource limits."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    limits: Tuple[ResourceLimit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_fields(self) -> "ResourceBudget":
        fields = [limit.field for limit in self.limits]
        duplicates = sorted({field for field in fields if fields.count(field) > 1})
        if duplicates:
            raise ValueError(
                "resource budget contains duplicate fields: " + ", ".join(duplicates)
            )
        return self
