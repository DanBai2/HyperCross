"""Unit coverage for task 17.5 pre-evaluation candidate-space freezing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from benchmark.registry import CandidateSpaceRegistry, CandidateSpaceRegistryError
from benchmark.schema import (
    CandidateOption,
    CandidateSpace,
    DETERMINISTIC_TIE_BREAK,
    MetricDirection,
    hash_object,
)


FROZEN_AT = datetime(2027, 4, 5, 6, 7, tzinfo=timezone.utc)


def _option(key: str, **parameters: object) -> CandidateOption:
    return CandidateOption.create(key, parameters=parameters)


def _space(**overrides: object) -> CandidateSpace:
    values = {
        "candidate_space_key": "hyperleaf-review-v1",
        "model_candidates": (_option("hypercross"), _option("rf-control")),
        "representation_candidates": (
            _option("global-spectrum", bands=730),
            _option("rgb", channels=3),
        ),
        "loss_candidates": (_option("mse"), _option("huber", delta=1.0)),
        "augmentation_candidates": (_option("none"), _option("spectral-jitter")),
        "target_route_candidates": (_option("shared"), _option("target-routed")),
        "inference_branch_candidates": (_option("primary"), _option("pca-only")),
        "threshold_candidates": (_option("fixed-0.5", value=0.5),),
        "calibration_candidates": (_option("none"), _option("temperature")),
        "objective_metric": "mean-r2",
        "metric_direction": MetricDirection.MAXIMIZE,
        "tie_break_rule": DETERMINISTIC_TIE_BREAK,
        "frozen_utc": FROZEN_AT,
    }
    values.update(overrides)
    return CandidateSpace.create(**values)  # type: ignore[arg-type]


def test_registry_freezes_all_explicit_dimensions_and_policy_before_evaluation() -> None:
    """Validates: Requirements 10.1-10.7."""

    registry = CandidateSpaceRegistry()
    space = _space()
    registry.register(space)

    frozen = registry.freeze_before_evaluation(frozen_utc=FROZEN_AT)
    resolved = frozen.require_for_evaluation(
        space.candidate_space_key,
        expected_hash=space.candidate_space_hash,
    )

    assert registry.is_frozen_for_evaluation is True
    assert resolved is space
    assert tuple(option.option_key for option in resolved.model_candidates) == (
        "hypercross",
        "rf-control",
    )
    assert tuple(option.option_key for option in resolved.representation_candidates) == (
        "global-spectrum",
        "rgb",
    )
    assert {option.option_key for option in resolved.loss_candidates} == {"huber", "mse"}
    assert {option.option_key for option in resolved.augmentation_candidates} == {
        "none",
        "spectral-jitter",
    }
    assert {option.option_key for option in resolved.target_route_candidates} == {
        "shared",
        "target-routed",
    }
    assert {option.option_key for option in resolved.inference_branch_candidates} == {
        "pca-only",
        "primary",
    }
    assert resolved.threshold_candidates[0].parameters == {"value": 0.5}
    assert {option.option_key for option in resolved.calibration_candidates} == {
        "none",
        "temperature",
    }
    assert resolved.metric_direction is MetricDirection.MAXIMIZE
    assert resolved.tie_break_rule == DETERMINISTIC_TIE_BREAK
    assert resolved.candidate_space_hash == hash_object(resolved.hash_payload())
    assert frozen.registry_hash == hash_object(frozen.hash_payload())


def test_freeze_is_immutable_append_only_and_hash_checked() -> None:
    """Validates: Requirements 10.1-10.7."""

    source_parameters = {"weights": [1, 2, 3]}
    option = CandidateOption.create("weighted-loss", parameters=source_parameters)
    source_parameters["weights"].append(4)
    assert option.parameters == {"weights": (1, 2, 3)}

    registry = CandidateSpaceRegistry()
    space = _space(loss_candidates=(option,))
    registry.register(space)
    frozen = registry.freeze_before_evaluation(frozen_utc=FROZEN_AT)

    with pytest.raises(ValidationError):
        space.objective_metric = "mae"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        option.option_key = "changed"  # type: ignore[misc]
    with pytest.raises(CandidateSpaceRegistryError, match="frozen"):
        registry.register(_space(candidate_space_key="late-space"))
    with pytest.raises(ValueError, match="hash differs"):
        frozen.require_for_evaluation(
            space.candidate_space_key,
            expected_hash="f" * 64,
        )

    assert registry.freeze_before_evaluation() is frozen


def test_invalid_or_incomplete_candidate_space_is_rejected_before_registration() -> None:
    """Validates: Requirements 10.1-10.7."""

    with pytest.raises(ValidationError, match="at least 1 item"):
        _space(calibration_candidates=())
    with pytest.raises(ValidationError, match="duplicate option keys"):
        _space(model_candidates=(_option("same"), _option("same")))
    with pytest.raises(ValidationError, match="tie_break_rule"):
        _space(tie_break_rule="input_order")

    registry = CandidateSpaceRegistry()
    registry.register(_space())
    with pytest.raises(CandidateSpaceRegistryError, match="already registered"):
        registry.register(_space())
    assert len(registry) == 1


def test_registry_cannot_freeze_an_empty_candidate_space_set() -> None:
    with pytest.raises(CandidateSpaceRegistryError, match="at least one"):
        CandidateSpaceRegistry().freeze_before_evaluation()
