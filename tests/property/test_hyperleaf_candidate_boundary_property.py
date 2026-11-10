"""Property coverage for task 14.11 HyperLeaf candidate fit boundaries."""

from __future__ import annotations

from hypothesis import given, strategies as st

from benchmark.runners.hyperleaf_candidates import HyperLeafCandidateRunner
from benchmark.schema import InnerFold, OuterFold


@given(
    inner_train_count=st.integers(min_value=1, max_value=12),
    inner_validation_count=st.integers(min_value=1, max_value=12),
    outer_test_count=st.integers(min_value=1, max_value=8),
)
def test_property_7_candidate_fit_boundaries_never_include_outer_test(
    inner_train_count: int,
    inner_validation_count: int,
    outer_test_count: int,
) -> None:
    """Property 7: fold-scoped fitted state.

    **Validates: Requirements 4.14, 4.15, 4.16, 4.17, 4.18, 4.19, 4.20, 4.21**
    """

    inner_train = tuple(f"train-{index:03d}" for index in range(inner_train_count))
    inner_validation = tuple(
        f"validation-{index:03d}" for index in range(inner_validation_count)
    )
    outer_test = tuple(f"outer-test-{index:03d}" for index in range(outer_test_count))
    outer_train = inner_train + inner_validation
    fold = OuterFold(
        fold_id="outer-property",
        outer_train_ids=outer_train,
        outer_test_ids=outer_test,
        inner_folds=(
            InnerFold(
                fold_id="inner-property",
                train_ids=inner_train,
                validation_ids=inner_validation,
            ),
        ),
    )

    inner_boundary = HyperLeafCandidateRunner.inner_fit_boundary(
        fold, "inner-property"
    )
    outer_boundary = HyperLeafCandidateRunner.outer_fit_boundary(fold)

    assert set(inner_boundary.allowed_fit_sample_ids) == set(inner_train)
    assert set(outer_boundary.allowed_fit_sample_ids) == set(outer_train)
    assert set(inner_boundary.allowed_fit_sample_ids).isdisjoint(outer_test)
    assert set(outer_boundary.allowed_fit_sample_ids).isdisjoint(outer_test)
