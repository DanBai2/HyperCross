"""Property 7: fold-scoped fitted state.

**Validates: Requirements 4.1-4.12**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from benchmark.schema import (
    ClaimStatus,
    FitBoundary,
    FitBoundaryScope,
    FittedArtifact,
    FittedArtifactKind,
)
from benchmark.transforms import PreprocessingManager, PreprocessingValidationError


_DATASET_HASH = "a" * 64
_PROTOCOL_HASH = "b" * 64


@dataclass(frozen=True)
class _ContaminatedFitCase:
    kind: FittedArtifactKind
    boundary: FitBoundary
    fit_sample_ids: tuple[str, ...]
    contamination_id: str
    contamination_source: str


@st.composite
def _contaminated_fit_cases(draw: st.DrawFn) -> _ContaminatedFitCase:
    """Generate valid inner/outer fitted states plus one forbidden fit ID."""

    kind = draw(st.sampled_from(tuple(FittedArtifactKind)))
    scope = draw(st.sampled_from(tuple(FitBoundaryScope)))
    outer_train_count = draw(st.integers(min_value=4, max_value=10))
    outer_train_ids = tuple(
        f"outer-train-{index:02d}" for index in range(outer_train_count)
    )
    outer_test_ids = tuple(
        f"outer-test-{index:02d}"
        for index in range(draw(st.integers(min_value=1, max_value=4)))
    )

    if scope is FitBoundaryScope.INNER_TRAIN:
        inner_train_count = draw(
            st.integers(min_value=2, max_value=outer_train_count - 1)
        )
        allowed_ids = outer_train_ids[:inner_train_count]
        inner_validation_ids = outer_train_ids[inner_train_count:]
        boundary = FitBoundary(
            fold_id="outer-property-7",
            scope=scope,
            inner_fold_id="inner-property-7",
            allowed_fit_sample_ids=allowed_ids,
            outer_test_sample_ids=outer_test_ids,
        )
        non_test_contamination = draw(st.sampled_from(inner_validation_ids))
    else:
        allowed_ids = outer_train_ids
        boundary = FitBoundary(
            fold_id="outer-property-7",
            scope=scope,
            allowed_fit_sample_ids=allowed_ids,
            outer_test_sample_ids=outer_test_ids,
        )
        non_test_contamination = "outside-protocol-sample"

    minimum_fit_count = (
        2
        if kind in {FittedArtifactKind.THRESHOLD, FittedArtifactKind.CALIBRATION}
        else 1
    )
    if kind is FittedArtifactKind.TARGET_NORMALIZATION:
        fit_ids = allowed_ids
    else:
        fit_count = draw(
            st.integers(min_value=minimum_fit_count, max_value=len(allowed_ids))
        )
        fit_ids = tuple(draw(st.permutations(allowed_ids))[:fit_count])

    contamination_source = draw(st.sampled_from(("outside_boundary", "outer_test")))
    contamination_id = (
        draw(st.sampled_from(outer_test_ids))
        if contamination_source == "outer_test"
        else non_test_contamination
    )
    return _ContaminatedFitCase(
        kind=kind,
        boundary=boundary,
        fit_sample_ids=fit_ids,
        contamination_id=contamination_id,
        contamination_source=contamination_source,
    )


def _fit_valid_artifact(
    manager: PreprocessingManager,
    case: _ContaminatedFitCase,
) -> FittedArtifact:
    """Fit every supported artifact kind from values matching the generated IDs."""

    sample_ids = case.fit_sample_ids
    sample_count = len(sample_ids)
    identity = {
        "dataset_hash": _DATASET_HASH,
        "protocol_hash": _PROTOCOL_HASH,
    }
    features = np.asarray(
        [[float(index), float(index * index + 1)] for index in range(sample_count)],
        dtype=np.float64,
    )

    if case.kind is FittedArtifactKind.TARGET_NORMALIZATION:
        return manager.fit_target_normalization(
            features, sample_ids, case.boundary, **identity
        )
    if case.kind is FittedArtifactKind.PCA:
        return manager.fit_pca(
            features,
            sample_ids,
            case.boundary,
            n_components=1,
            **identity,
        )
    if case.kind is FittedArtifactKind.BAND_NORMALIZATION:
        return manager.fit_band_normalization(
            features, sample_ids, case.boundary, **identity
        )
    if case.kind is FittedArtifactKind.IMPUTATION:
        return manager.fit_imputation(
            features, sample_ids, case.boundary, **identity
        )
    if case.kind is FittedArtifactKind.CLASS_WEIGHTS:
        labels = [index % 2 for index in range(sample_count)]
        return manager.fit_class_weights(
            labels, sample_ids, case.boundary, **identity
        )

    labels = [index % 2 for index in range(sample_count)]
    probabilities = np.linspace(0.1, 0.9, sample_count).tolist()
    if case.kind is FittedArtifactKind.THRESHOLD:
        return manager.fit_threshold(
            probabilities,
            labels,
            sample_ids,
            case.boundary,
            **identity,
        )
    return manager.fit_calibration(
        probabilities,
        labels,
        sample_ids,
        case.boundary,
        **identity,
    )


class _UnreadableApplyValues:
    """Detect any attempt to inspect values before boundary validation."""

    touched = False

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self.touched = True
        raise AssertionError("apply values were read before fitted-state validation")


@settings(max_examples=100, deadline=None)
@given(case=_contaminated_fit_cases())
def test_contaminated_fit_ids_are_blocked_before_apply(
    case: _ContaminatedFitCase,
) -> None:
    """Inner/outer boundary pollution, including outer-test leakage, never applies."""

    manager = PreprocessingManager()
    artifact = _fit_valid_artifact(manager, case)

    assert artifact.fit_sample_ids == tuple(
        sorted(case.fit_sample_ids, key=lambda value: value.encode("utf-8"))
    )
    assert set(artifact.fit_sample_ids) <= set(case.boundary.allowed_fit_sample_ids)
    assert set(artifact.fit_sample_ids).isdisjoint(
        case.boundary.outer_test_sample_ids
    )

    contaminated_ids = artifact.fit_sample_ids + (case.contamination_id,)
    contaminated = artifact.model_copy(update={"fit_sample_ids": contaminated_ids})
    report = manager.validate_boundary(contaminated, case.boundary)
    issue_codes = {issue.code for issue in report.errors}

    assert report.passes is False
    assert "FIT_SAMPLE_IDS_OUTSIDE_BOUNDARY" in issue_codes
    if case.contamination_source == "outer_test":
        assert "OUTER_TEST_FITTED_STATE_LEAKAGE" in issue_codes

    values = _UnreadableApplyValues()
    with pytest.raises(PreprocessingValidationError) as caught:
        manager.apply(contaminated, values, case.boundary)

    error = caught.value.error_record
    assert values.touched is False
    assert caught.value.claim_status is ClaimStatus.INVALID_OR_LEAKY
    assert error.error_code == "FIT_SAMPLE_IDS_OUTSIDE_BOUNDARY"
    assert error.affected_identifiers["fitted_artifact_id"] == artifact.artifact_id
    assert error.affected_identifiers["fold_id"] == case.boundary.fold_id
    assert error.unexpected_ids == (case.contamination_id,)
    assert error.invariant_checks == {
        "checkpoint_unchanged": True,
        "prediction_artifact_unchanged": True,
        "completed_artifacts_unchanged": True,
    }
