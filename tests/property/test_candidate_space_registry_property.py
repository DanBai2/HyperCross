"""Property coverage for task 17.5 candidate-space hash stability."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import given, strategies as st

from benchmark.schema import CandidateOption, CandidateSpace, MetricDirection


pytestmark = pytest.mark.property
FROZEN_AT = datetime(2027, 4, 5, 6, 7, tzinfo=timezone.utc)


@st.composite
def option_definitions(draw: st.DrawFn) -> list[tuple[str, dict[str, int]]]:
    keys = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12),
            min_size=1,
            max_size=8,
            unique=True,
        )
    )
    definitions: list[tuple[str, dict[str, int]]] = []
    for key in keys:
        parameters = draw(
            st.dictionaries(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
                st.integers(min_value=-1000, max_value=1000),
                max_size=5,
            )
        )
        definitions.append((key, parameters))
    return definitions


def _option(key: str) -> CandidateOption:
    return CandidateOption.create(key)


def _space(models: tuple[CandidateOption, ...]) -> CandidateSpace:
    return CandidateSpace.create(
        candidate_space_key="generated-space",
        model_candidates=models,
        representation_candidates=(_option("representation"),),
        loss_candidates=(_option("loss"),),
        augmentation_candidates=(_option("augmentation"),),
        target_route_candidates=(_option("route"),),
        inference_branch_candidates=(_option("branch"),),
        threshold_candidates=(CandidateOption.create("threshold", parameters={"value": 0.5}),),
        calibration_candidates=(_option("calibration"),),
        objective_metric="objective",
        metric_direction=MetricDirection.MINIMIZE,
        frozen_utc=FROZEN_AT,
    )


@given(option_definitions())
def test_candidate_space_hash_is_invariant_to_option_and_parameter_enumeration(
    definitions: list[tuple[str, dict[str, int]]],
) -> None:
    """Property 20: candidate space is canonical before evaluation.

    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**
    """

    forward = tuple(
        CandidateOption.create(key, parameters=parameters)
        for key, parameters in definitions
    )
    reverse = tuple(
        CandidateOption.create(
            key,
            parameters=dict(reversed(tuple(parameters.items()))),
        )
        for key, parameters in reversed(definitions)
    )

    first = _space(forward)
    second = _space(reverse)

    assert first.model_candidates == second.model_candidates
    assert first.candidate_space_hash == second.candidate_space_hash
