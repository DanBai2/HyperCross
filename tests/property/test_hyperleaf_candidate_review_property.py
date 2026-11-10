"""Property coverage for task 17.6 matched HyperLeaf review controls."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import given, strategies as st

from benchmark.protocols.hyperleaf_lovo import HYPERLEAF_LOVO_PROTOCOL_KEY
from benchmark.runners import (
    HyperLeafReviewError,
    HyperLeafReviewRole,
    build_hyperleaf_review_plan,
    validate_hyperleaf_matched_control,
)
from benchmark.schema import (
    EvaluationBudget,
    InnerFold,
    LimitEnforcement,
    MetricDirection,
    OuterFold,
    ProtocolManifest,
    ResourceBudget,
    ResourceField,
    ResourceLimit,
    ResourceUnit,
    SelectionPolicy,
)


pytestmark = pytest.mark.property
FROZEN_AT = datetime(2027, 5, 1, tzinfo=timezone.utc)


def _plan(seed_set: tuple[int, ...]):
    policy = SelectionPolicy(
        policy_key="hyperleaf.review.inner-variety.v1",
        objective_metric="mean-r2",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )
    protocol = ProtocolManifest.create(
        schema_version="1.0",
        protocol_key=HYPERLEAF_LOVO_PROTOCOL_KEY,
        dataset_hash="a" * 64,
        folds=(
            OuterFold(
                fold_id="outer-v4",
                outer_train_ids=("v1-s1", "v2-s1", "v3-s1"),
                outer_test_ids=("v4-s1",),
                inner_folds=(
                    InnerFold(
                        fold_id="inner-v1",
                        train_ids=("v2-s1", "v3-s1"),
                        validation_ids=("v1-s1",),
                    ),
                    InnerFold(
                        fold_id="inner-v2",
                        train_ids=("v1-s1", "v3-s1"),
                        validation_ids=("v2-s1",),
                    ),
                    InnerFold(
                        fold_id="inner-v3",
                        train_ids=("v1-s1", "v2-s1"),
                        validation_ids=("v3-s1",),
                    ),
                ),
            ),
        ),
        selection_policy=policy,
        seed_set=seed_set,
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=1),
    )
    budget = ResourceBudget(
        limits=(
            ResourceLimit(
                field=ResourceField.COMPLETED_EPOCHS,
                value=30,
                unit=ResourceUnit.COUNT,
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )
    return build_hyperleaf_review_plan(
        protocol,
        metric_configuration={"objective": "mean-r2"},
        resource_budget=budget,
        frozen_utc=FROZEN_AT,
    )


seed_sets = st.sets(
    st.integers(min_value=0, max_value=2**16), min_size=1, max_size=6
).map(lambda values: tuple(sorted(values)))


@given(seed_set=seed_sets)
def test_all_declared_candidate_control_pairs_preserve_every_comparison_identity(
    seed_set: tuple[int, ...],
) -> None:
    """Property 20: candidate/control identities cannot drift.

    **Validates: Requirements 10.12, 10.13, 10.14, 10.15, 10.16, 10.17**
    """

    plan = _plan(seed_set)
    by_key = {binding.spec.review_key: binding for binding in plan.bindings}
    for candidate in plan.bindings:
        if candidate.spec.role is HyperLeafReviewRole.MATCHED_INPUT_CONTROL:
            continue
        control = by_key[candidate.spec.matched_control_key]
        validate_hyperleaf_matched_control(candidate, control)
        assert candidate.context.dataset_hash == control.context.dataset_hash
        assert candidate.context.protocol_hash == control.context.protocol_hash
        assert candidate.context.seed_set == control.context.seed_set
        assert candidate.context.fold_manifest_hashes == control.context.fold_manifest_hashes
        assert candidate.context.metric_configuration_hash == control.context.metric_configuration_hash
        assert candidate.context.resource_budget == control.context.resource_budget


@given(
    seed_set=seed_sets,
    field=st.sampled_from((
        "dataset_hash",
        "protocol_hash",
        "seed_set",
        "fold_manifest_hashes",
        "metric_configuration_hash",
        "resource_budget",
    )),
)
def test_any_candidate_control_identity_drift_is_rejected(
    seed_set: tuple[int, ...],
    field: str,
) -> None:
    """Property 20: one changed matched field blocks comparison.

    **Validates: Requirements 10.12, 10.13, 10.14, 10.15, 10.16, 10.17, 10.35, 10.36, 10.37**
    """

    plan = _plan(seed_set)
    candidate = next(
        binding for binding in plan.bindings
        if binding.spec.role is HyperLeafReviewRole.CANDIDATE
    )
    control = next(
        binding for binding in plan.bindings
        if binding.spec.review_key == candidate.spec.matched_control_key
    )
    replacements = {
        "dataset_hash": "b" * 64,
        "protocol_hash": "c" * 64,
        "seed_set": tuple(sorted((*seed_set, max(seed_set) + 1))),
        "fold_manifest_hashes": (("changed-fold", "d" * 64),),
        "metric_configuration_hash": "e" * 64,
        "resource_budget": ResourceBudget(
            limits=(
                ResourceLimit(
                    field=ResourceField.COMPLETED_EPOCHS,
                    value=31,
                    unit=ResourceUnit.COUNT,
                    enforcement=LimitEnforcement.HARD,
                ),
            )
        ),
    }
    changed_context = control.context.model_copy(update={field: replacements[field]})
    changed_control = control.model_copy(update={"context": changed_context})

    with pytest.raises(HyperLeafReviewError) as caught:
        validate_hyperleaf_matched_control(candidate, changed_control)

    assert field in caught.value.mismatched_fields
