"""Immutable candidate selection decisions and outer-refit requests."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple
from uuid import uuid4

from pydantic import Field, ValidationInfo, field_validator, model_validator

from .base import ContentHash, FoldId, Identifier, MetricDirection, SampleId, ensure_machine_value
from .canonical import hash_object, normalize_id
from .environment import FrozenSchemaModel
from .preprocessing import canonical_sample_ids


class SelectionCandidate(FrozenSchemaModel):
    """One fully identified candidate from a predeclared selection space."""

    candidate_id: Identifier
    configuration: Dict[Identifier, Any] = Field(default_factory=dict)

    @field_validator("candidate_id")
    @classmethod
    def normalize_candidate_id(cls, value: str) -> str:
        return normalize_id(value)

    @field_validator("configuration")
    @classmethod
    def validate_configuration(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        copied = deepcopy(value)
        ensure_machine_value(copied, field_name="configuration")
        return copied


# Concise design-document name retained as an alias.
Candidate = SelectionCandidate


class SelectionDecision(FrozenSchemaModel):
    """Hash-bound, frozen result of inner-only candidate selection."""

    decision_id: Identifier
    fold_id: FoldId
    candidates: Tuple[SelectionCandidate, ...] = Field(min_length=1)
    candidate_scores: Dict[Identifier, float]
    source_sample_ids: Tuple[SampleId, ...] = Field(min_length=1)
    metric_direction: MetricDirection
    tie_break_rule: Identifier
    source_prediction_hash: ContentHash
    selected_candidate: Identifier
    frozen_utc: datetime
    decision_hash: ContentHash

    @field_validator("decision_id", "fold_id", "selected_candidate")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return normalize_id(value)

    @field_validator("candidates")
    @classmethod
    def canonicalize_candidates(
        cls, value: Tuple[SelectionCandidate, ...]
    ) -> Tuple[SelectionCandidate, ...]:
        identifiers = tuple(candidate.candidate_id for candidate in value)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidates contains duplicate candidate IDs")
        return tuple(
            sorted(value, key=lambda candidate: candidate.candidate_id.encode("utf-8"))
        )

    @field_validator("candidate_scores", mode="before")
    @classmethod
    def canonicalize_scores(cls, value: Any) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("candidate_scores must be a candidate-ID mapping")
        normalized: Dict[str, float] = {}
        for candidate_id, score in value.items():
            if not isinstance(candidate_id, str):
                raise ValueError("candidate_scores keys must be candidate IDs")
            key = normalize_id(candidate_id)
            if key in normalized:
                raise ValueError("candidate_scores contains duplicate normalized IDs")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("candidate scores must be finite numbers")
            normalized[key] = float(score)
        return {
            key: normalized[key]
            for key in sorted(normalized, key=lambda item: item.encode("utf-8"))
        }

    @field_validator("source_sample_ids", mode="before")
    @classmethod
    def canonicalize_source_ids(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("source_sample_ids must be a collection of Sample_IDs")
        try:
            return canonical_sample_ids(value, field_name="source_sample_ids")
        except TypeError as exc:
            raise ValueError(
                "source_sample_ids must be a collection of Sample_IDs"
            ) from exc

    @field_validator("frozen_utc")
    @classmethod
    def validate_frozen_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("frozen_utc must include a UTC timezone")
        if value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("frozen_utc must be expressed in UTC")
        return value

    @model_validator(mode="after")
    def validate_complete_decision(self, info: ValidationInfo) -> "SelectionDecision":
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        score_ids = set(self.candidate_scores)
        if score_ids != candidate_ids:
            raise ValueError(
                "candidate_scores keys must equal the complete candidate set"
            )
        if self.selected_candidate not in candidate_ids:
            raise ValueError("selected_candidate must belong to candidates")
        if not (info.context and info.context.get("allow_unbound_decision_hash")):
            self.require_integrity()
        return self

    @classmethod
    def create(cls, **values: Any) -> "SelectionDecision":
        """Create a decision and bind all frozen selection evidence to its hash."""

        payload = dict(values)
        payload.setdefault("decision_id", f"selection-{uuid4()}")
        payload.setdefault("frozen_utc", datetime.now(timezone.utc))
        payload.pop("decision_hash", None)
        unhashed = cls.model_validate(
            {**payload, "decision_hash": "0" * 64},
            context={"allow_unbound_decision_hash": True},
        )
        bound_payload = unhashed.hash_payload()
        bound_payload["decision_hash"] = hash_object(bound_payload)
        return cls.model_validate(bound_payload)

    def hash_payload(self) -> Dict[str, Any]:
        """Return the complete frozen decision except its self hash."""

        return self.model_dump(mode="python", exclude={"decision_hash"})

    def require_integrity(self) -> None:
        """Reject any nested mutation made after the decision was frozen."""

        if self.decision_hash != hash_object(self.hash_payload()):
            raise ValueError("decision_hash does not match the frozen selection decision")


class OuterRefitRequest(FrozenSchemaModel):
    """A refit request that carries one unchanged decision and all outer-train IDs."""

    fold_id: FoldId
    outer_train_ids: Tuple[SampleId, ...] = Field(min_length=1)
    selection_decision: SelectionDecision

    @field_validator("outer_train_ids", mode="before")
    @classmethod
    def canonicalize_outer_train_ids(cls, value: Any) -> Tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("outer_train_ids must be a collection of Sample_IDs")
        try:
            return canonical_sample_ids(value, field_name="outer_train_ids")
        except TypeError as exc:
            raise ValueError(
                "outer_train_ids must be a collection of Sample_IDs"
            ) from exc

    @model_validator(mode="after")
    def validate_decision_binding(self) -> "OuterRefitRequest":
        self.selection_decision.require_integrity()
        if self.fold_id != self.selection_decision.fold_id:
            raise ValueError("refit fold_id must equal Selection_Decision fold_id")
        return self

    @property
    def selected_candidate(self) -> SelectionCandidate:
        """Return the exact selected candidate snapshot from the frozen decision."""

        return next(
            candidate
            for candidate in self.selection_decision.candidates
            if candidate.candidate_id == self.selection_decision.selected_candidate
        )


__all__ = [
    "Candidate",
    "OuterRefitRequest",
    "SelectionCandidate",
    "SelectionDecision",
]
