"""Hash-bound candidate spaces frozen before outer evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Tuple

from pydantic import Field, field_validator, model_validator

from .base import ContentHash, Identifier, MetricDirection
from .canonical import canonical_deserialize, canonical_serialize, hash_object, normalize_id
from .environment import FrozenSchemaModel


DETERMINISTIC_TIE_BREAK = "candidate_id_utf8_ascending"


class CandidateOption(FrozenSchemaModel):
    """One immutable, independently identified option in a candidate dimension."""

    option_key: Identifier
    parameters_canonical_serialization: str = Field(min_length=2)
    option_hash: ContentHash

    @classmethod
    def create(
        cls,
        option_key: str,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> "CandidateOption":
        """Canonicalize arbitrary option parameters and bind them to a content hash."""

        serialization = canonical_serialize(dict(parameters or {})).decode("utf-8")
        payload = {
            "option_key": normalize_id(option_key),
            "parameters_canonical_serialization": serialization,
        }
        payload["option_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    @field_validator("option_key")
    @classmethod
    def normalize_option_key(cls, value: str) -> str:
        return normalize_id(value)

    @field_validator("parameters_canonical_serialization")
    @classmethod
    def validate_parameter_serialization(cls, value: str) -> str:
        try:
            parsed_json = json.loads(value)
            decoded = canonical_deserialize(value)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                "parameters_canonical_serialization must contain canonical JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError("candidate option parameters must be a mapping")
        compact = json.dumps(
            parsed_json,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if value != compact or canonical_serialize(decoded).decode("utf-8") != value:
            raise ValueError("candidate option parameter serialization is not canonical")
        return value

    @property
    def parameters(self) -> dict[str, Any]:
        """Return a fresh decoded parameter mapping; stored state remains immutable."""

        decoded = canonical_deserialize(self.parameters_canonical_serialization)
        assert isinstance(decoded, dict)
        return decoded

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"option_hash"})

    @model_validator(mode="after")
    def validate_option_hash(self) -> "CandidateOption":
        if self.option_hash != hash_object(self.hash_payload()):
            raise ValueError("option_hash does not match the frozen candidate option")
        return self


class CandidateSpace(FrozenSchemaModel):
    """Complete candidate dimensions and selection policy frozen as one artifact."""

    schema_version: str = Field(default="1.0", min_length=1)
    candidate_space_key: Identifier
    model_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    representation_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    loss_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    augmentation_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    target_route_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    inference_branch_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    threshold_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    calibration_candidates: Tuple[CandidateOption, ...] = Field(min_length=1)
    objective_metric: Identifier
    metric_direction: MetricDirection
    tie_break_rule: Identifier
    frozen_utc: datetime
    candidate_space_hash: ContentHash

    @classmethod
    def create(
        cls,
        *,
        candidate_space_key: str,
        model_candidates: Iterable[CandidateOption],
        representation_candidates: Iterable[CandidateOption],
        loss_candidates: Iterable[CandidateOption],
        augmentation_candidates: Iterable[CandidateOption],
        target_route_candidates: Iterable[CandidateOption],
        inference_branch_candidates: Iterable[CandidateOption],
        threshold_candidates: Iterable[CandidateOption],
        calibration_candidates: Iterable[CandidateOption],
        objective_metric: str,
        metric_direction: MetricDirection,
        tie_break_rule: str = DETERMINISTIC_TIE_BREAK,
        frozen_utc: datetime | None = None,
        schema_version: str = "1.0",
    ) -> "CandidateSpace":
        """Freeze every selectable dimension before registration or evaluation."""

        payload: dict[str, Any] = {
            "schema_version": schema_version,
            "candidate_space_key": normalize_id(candidate_space_key),
            "model_candidates": tuple(model_candidates),
            "representation_candidates": tuple(representation_candidates),
            "loss_candidates": tuple(loss_candidates),
            "augmentation_candidates": tuple(augmentation_candidates),
            "target_route_candidates": tuple(target_route_candidates),
            "inference_branch_candidates": tuple(inference_branch_candidates),
            "threshold_candidates": tuple(threshold_candidates),
            "calibration_candidates": tuple(calibration_candidates),
            "objective_metric": normalize_id(objective_metric),
            "metric_direction": metric_direction,
            "tie_break_rule": normalize_id(tie_break_rule),
            "frozen_utc": frozen_utc or datetime.now(timezone.utc),
        }
        payload["candidate_space_hash"] = hash_object(
            cls._canonical_hash_payload(payload)
        )
        return cls.model_validate(payload)

    @field_validator("candidate_space_key", "objective_metric", "tie_break_rule")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return normalize_id(value)

    @field_validator(
        "model_candidates",
        "representation_candidates",
        "loss_candidates",
        "augmentation_candidates",
        "target_route_candidates",
        "inference_branch_candidates",
        "threshold_candidates",
        "calibration_candidates",
    )
    @classmethod
    def canonicalize_dimension(
        cls, value: Tuple[CandidateOption, ...]
    ) -> Tuple[CandidateOption, ...]:
        keys = tuple(option.option_key for option in value)
        if len(keys) != len(set(keys)):
            raise ValueError("candidate dimension contains duplicate option keys")
        return tuple(sorted(value, key=lambda option: option.option_key.encode("utf-8")))

    @field_validator("tie_break_rule")
    @classmethod
    def require_deterministic_tie_break(cls, value: str) -> str:
        if value != DETERMINISTIC_TIE_BREAK:
            raise ValueError(
                f"tie_break_rule must be {DETERMINISTIC_TIE_BREAK!r}"
            )
        return value

    @field_validator("frozen_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("frozen_utc must include a UTC timezone")
        if value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("frozen_utc must be expressed in UTC")
        return value

    @staticmethod
    def _canonical_hash_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        for field_name in (
            "model_candidates",
            "representation_candidates",
            "loss_candidates",
            "augmentation_candidates",
            "target_route_candidates",
            "inference_branch_candidates",
            "threshold_candidates",
            "calibration_candidates",
        ):
            result[field_name] = tuple(
                sorted(
                    result[field_name],
                    key=lambda option: option.option_key.encode("utf-8"),
                )
            )
        return result

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"candidate_space_hash"})

    def require_integrity(self) -> None:
        for field_name in (
            "model_candidates",
            "representation_candidates",
            "loss_candidates",
            "augmentation_candidates",
            "target_route_candidates",
            "inference_branch_candidates",
            "threshold_candidates",
            "calibration_candidates",
        ):
            for option in getattr(self, field_name):
                if option.option_hash != hash_object(option.hash_payload()):
                    raise ValueError(f"{field_name} contains a mutated candidate option")
        if self.candidate_space_hash != hash_object(self.hash_payload()):
            raise ValueError("candidate_space_hash does not match the frozen candidate space")

    @model_validator(mode="after")
    def validate_candidate_space(self) -> "CandidateSpace":
        self.require_integrity()
        return self


class FrozenCandidateSpaceRegistry(FrozenSchemaModel):
    """Immutable registry snapshot required by an evaluation boundary."""

    candidate_spaces: Tuple[CandidateSpace, ...] = Field(min_length=1)
    frozen_utc: datetime
    registry_hash: ContentHash

    @classmethod
    def create(
        cls,
        candidate_spaces: Iterable[CandidateSpace],
        *,
        frozen_utc: datetime | None = None,
    ) -> "FrozenCandidateSpaceRegistry":
        spaces = tuple(
            sorted(
                candidate_spaces,
                key=lambda space: space.candidate_space_key.encode("utf-8"),
            )
        )
        payload = {
            "candidate_spaces": spaces,
            "frozen_utc": frozen_utc or datetime.now(timezone.utc),
        }
        payload["registry_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    @field_validator("candidate_spaces")
    @classmethod
    def validate_spaces(
        cls, value: Tuple[CandidateSpace, ...]
    ) -> Tuple[CandidateSpace, ...]:
        keys = tuple(space.candidate_space_key for space in value)
        if len(keys) != len(set(keys)):
            raise ValueError("candidate registry contains duplicate candidate_space_key values")
        for space in value:
            space.require_integrity()
        return tuple(sorted(value, key=lambda space: space.candidate_space_key.encode("utf-8")))

    @field_validator("frozen_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        return CandidateSpace.require_utc(value)

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"registry_hash"})

    @model_validator(mode="after")
    def validate_registry_hash(self) -> "FrozenCandidateSpaceRegistry":
        if self.registry_hash != hash_object(self.hash_payload()):
            raise ValueError("registry_hash does not match the frozen candidate registry")
        return self

    def require_for_evaluation(
        self,
        candidate_space_key: str,
        *,
        expected_hash: str,
    ) -> CandidateSpace:
        """Resolve only a space whose complete pre-evaluation hash is expected."""

        key = normalize_id(candidate_space_key)
        for space in self.candidate_spaces:
            if space.candidate_space_key == key:
                space.require_integrity()
                if space.candidate_space_hash != expected_hash.lower():
                    raise ValueError("candidate space hash differs from the frozen evaluation request")
                return space
        raise KeyError(f"candidate space {key!r} is not registered in the frozen snapshot")


__all__ = [
    "CandidateOption",
    "CandidateSpace",
    "DETERMINISTIC_TIE_BREAK",
    "FrozenCandidateSpaceRegistry",
]
