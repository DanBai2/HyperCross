"""Property coverage for task 15.5 matched Histology run identities.

**Validates: Requirements 10.12, 10.13, 10.14, 10.15, 10.16, 10.17**
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from benchmark.models.histology import (
    ATTENTION_DISABLED_MODEL_KEY,
    LOCAL_ROUTE_A_MODEL_KEY,
)
from benchmark.runners import (
    HistologyCandidateError,
    HistologyCandidateRunner,
    MatchedRunContext,
)


sha256s = st.binary(min_size=32, max_size=32).map(bytes.hex)
seed_sets = st.sets(
    st.integers(min_value=-(2**16), max_value=2**16), min_size=1, max_size=6
).map(lambda values: tuple(sorted(values)))


@st.composite
def matched_contexts(draw: st.DrawFn) -> MatchedRunContext:
    return MatchedRunContext(
        dataset_hash=draw(sha256s),
        protocol_hash=draw(sha256s),
        seed_set=draw(seed_sets),
        fold_manifest_hash=draw(sha256s),
        metric_configuration_hash=draw(sha256s),
        resource_budget_hash=draw(sha256s),
    )


@given(context=matched_contexts())
def test_identical_frozen_contexts_always_allow_declared_matched_control(
    context: MatchedRunContext,
) -> None:
    runner = HistologyCandidateRunner()

    runner.validate_matched_control(
        LOCAL_ROUTE_A_MODEL_KEY,
        ATTENTION_DISABLED_MODEL_KEY,
        candidate_context=context,
        control_context=context,
    )


@given(
    context=matched_contexts(),
    field=st.sampled_from(
        (
            "dataset_hash",
            "protocol_hash",
            "fold_manifest_hash",
            "metric_configuration_hash",
            "resource_budget_hash",
        )
    ),
    replacement=sha256s,
)
def test_any_changed_comparison_identity_is_rejected(
    context: MatchedRunContext,
    field: str,
    replacement: str,
) -> None:
    if replacement == getattr(context, field):
        replacement = ("0" if replacement[0] != "0" else "1") + replacement[1:]
    changed = {
        "dataset_hash": context.dataset_hash,
        "protocol_hash": context.protocol_hash,
        "seed_set": context.seed_set,
        "fold_manifest_hash": context.fold_manifest_hash,
        "metric_configuration_hash": context.metric_configuration_hash,
        "resource_budget_hash": context.resource_budget_hash,
    }
    changed[field] = replacement
    runner = HistologyCandidateRunner()

    with pytest.raises(HistologyCandidateError) as caught:
        runner.validate_matched_control(
            LOCAL_ROUTE_A_MODEL_KEY,
            ATTENTION_DISABLED_MODEL_KEY,
            candidate_context=context,
            control_context=MatchedRunContext(**changed),
        )

    assert field in caught.value.mismatched_fields
