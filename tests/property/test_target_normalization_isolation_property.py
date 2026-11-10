"""Property 8: HyperLeaf target normalization isolation.

**Validates: Requirements 4.14-4.16, 16.12, 16.47**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from benchmark.schema import FitBoundary, FitBoundaryScope
from benchmark.transforms import PreprocessingManager, PreprocessingValidationError


_DATASET_HASH = "a" * 64
_PROTOCOL_HASH = "b" * 64


@dataclass(frozen=True)
class _HyperLeafTargetCase:
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    official_test_ids: tuple[str, ...]
    training_targets: dict[str, tuple[float, ...]]
    first_excluded_targets: dict[str, tuple[float, ...]]
    second_excluded_targets: dict[str, tuple[float, ...]]


class _TrackedTargetView:
    """Materialize selected rows while recording which target IDs were read."""

    def __init__(
        self,
        targets: dict[str, tuple[float, ...]],
        selected_ids: tuple[str, ...],
        forbidden_ids: tuple[str, ...],
    ) -> None:
        self._targets = targets
        self._selected_ids = selected_ids
        self._forbidden_ids = frozenset(forbidden_ids)
        self.accessed_ids: tuple[str, ...] = ()

    def __array__(
        self,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        forbidden_access = self._forbidden_ids.intersection(self._selected_ids)
        if forbidden_access:
            raise AssertionError(f"forbidden targets were read: {sorted(forbidden_access)}")
        self.accessed_ids = self._selected_ids
        array = np.asarray(
            [self._targets[sample_id] for sample_id in self._selected_ids],
            dtype=dtype,
        )
        return array.copy() if copy else array


class _ExplodingTargets:
    """Fail if fit validation reads targets from a contaminated request."""

    def __init__(self) -> None:
        self.touched = False

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        self.touched = True
        raise AssertionError("targets were read before Fit_Sample_ID validation")


@st.composite
def _hyperleaf_target_cases(draw: st.DrawFn) -> _HyperLeafTargetCase:
    training_count = draw(st.integers(min_value=1, max_value=12))
    validation_count = draw(st.integers(min_value=1, max_value=5))
    official_count = draw(st.integers(min_value=1, max_value=5))
    target_count = draw(st.integers(min_value=1, max_value=4))

    training_ids = tuple(f"train-{index:03d}" for index in range(training_count))
    validation_ids = tuple(
        f"validation-{index:03d}" for index in range(validation_count)
    )
    official_test_ids = tuple(
        f"official-test-{index:03d}" for index in range(official_count)
    )
    training_rows = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-10_000,
                    max_value=10_000,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
                min_size=target_count,
                max_size=target_count,
            ),
            min_size=training_count,
            max_size=training_count,
        )
    )
    training_targets = {
        sample_id: tuple(float(value) for value in row)
        for sample_id, row in zip(training_ids, training_rows)
    }

    excluded_ids = validation_ids + official_test_ids
    first_excluded_targets = {
        sample_id: tuple(
            1_000_000.0 + index * 100.0 + target for target in range(target_count)
        )
        for index, sample_id in enumerate(excluded_ids)
    }
    second_excluded_targets = {
        sample_id: tuple(
            -1_000_000.0 - index * 100.0 - target for target in range(target_count)
        )
        for index, sample_id in enumerate(excluded_ids)
    }
    return _HyperLeafTargetCase(
        training_ids=tuple(draw(st.permutations(training_ids))),
        validation_ids=validation_ids,
        official_test_ids=official_test_ids,
        training_targets=training_targets,
        first_excluded_targets=first_excluded_targets,
        second_excluded_targets=second_excluded_targets,
    )


def _all_targets(
    case: _HyperLeafTargetCase,
    excluded: dict[str, tuple[float, ...]],
) -> dict[str, tuple[float, ...]]:
    return {**case.training_targets, **excluded}


@settings(max_examples=100, deadline=None)
@given(case=_hyperleaf_target_cases())
def test_hyperleaf_target_statistics_use_only_exact_permitted_training_ids(
    case: _HyperLeafTargetCase,
) -> None:
    """Statistics use every training target and never read validation/test targets."""

    boundary = FitBoundary(
        fold_id="hyperleaf-official-inner-01",
        scope=FitBoundaryScope.INNER_TRAIN,
        inner_fold_id="inner-01",
        allowed_fit_sample_ids=case.training_ids,
        outer_test_sample_ids=case.official_test_ids,
    )
    manager = PreprocessingManager()
    forbidden_ids = case.validation_ids + case.official_test_ids
    first_view = _TrackedTargetView(
        _all_targets(case, case.first_excluded_targets),
        case.training_ids,
        forbidden_ids,
    )
    second_view = _TrackedTargetView(
        _all_targets(case, case.second_excluded_targets),
        case.training_ids,
        forbidden_ids,
    )

    first_artifact = manager.fit_target_normalization(
        first_view,
        case.training_ids,
        boundary,
        dataset_hash=_DATASET_HASH,
        protocol_hash=_PROTOCOL_HASH,
    )
    second_artifact = manager.fit_target_normalization(
        second_view,
        case.training_ids,
        boundary,
        dataset_hash=_DATASET_HASH,
        protocol_hash=_PROTOCOL_HASH,
    )

    expected_values = np.asarray(
        [case.training_targets[sample_id] for sample_id in case.training_ids],
        dtype=np.float64,
    )
    expected_mean = expected_values.mean(axis=0)
    expected_scale = expected_values.std(axis=0)
    expected_scale = np.where(expected_scale < 1e-12, 1.0, expected_scale)

    assert set(first_artifact.fit_sample_ids) == set(case.training_ids)
    assert first_view.accessed_ids == case.training_ids
    assert second_view.accessed_ids == case.training_ids
    assert set(first_view.accessed_ids).isdisjoint(forbidden_ids)
    assert set(second_view.accessed_ids).isdisjoint(forbidden_ids)
    np.testing.assert_allclose(first_artifact.state["mean"], expected_mean)
    np.testing.assert_allclose(first_artifact.state["scale"], expected_scale)
    assert first_artifact.state == second_artifact.state
    assert first_artifact.state_hash == second_artifact.state_hash

    for forbidden_id in (case.validation_ids[0], case.official_test_ids[0]):
        unreadable_targets = _ExplodingTargets()
        with pytest.raises(PreprocessingValidationError) as caught:
            manager.fit_target_normalization(
                unreadable_targets,
                case.training_ids + (forbidden_id,),
                boundary,
                dataset_hash=_DATASET_HASH,
                protocol_hash=_PROTOCOL_HASH,
            )
        assert unreadable_targets.touched is False
        assert forbidden_id in caught.value.error_record.unexpected_ids
