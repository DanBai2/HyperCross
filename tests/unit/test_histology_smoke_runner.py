"""Unit coverage for task 15.10 patient-disjoint smoke-runner glue.

Validates Requirements 5.13-5.24, 9.39-9.41, 13.1-13.29,
14.7-14.20, 16.39, 16.40.
"""
from __future__ import annotations

import pytest

from benchmark.runners.histology_smoke import (
    FOLD_ID,
    LOCAL_ROUTE_A_INFERENCE_BRANCH,
    make_prediction_records,
    select_patient_partition,
)


def _samples() -> dict[str, tuple[str, ...]]:
    return {
        f"P{patient}": (
            f"P{patient}/ROI_02_C01_NT",
            f"P{patient}/ROI_01_C02_T",
            f"P{patient}/ROI_01_C01_T",
        )
        for patient in range(1, 5)
    }


def test_patient_partition_is_deterministic_bounded_and_group_disjoint() -> None:
    observed = select_patient_partition(dict(reversed(tuple(_samples().items()))))

    assert observed.fit_groups == ("P1", "P2")
    assert observed.prediction_groups == ("P3",)
    assert set(observed.fit_groups).isdisjoint(observed.prediction_groups)
    assert observed.fit_ids == (
        "P1/ROI_01_C01_T",
        "P1/ROI_02_C01_NT",
        "P2/ROI_01_C01_T",
        "P2/ROI_02_C01_NT",
    )
    assert observed.prediction_ids == (
        "P3/ROI_01_C01_T",
        "P3/ROI_02_C01_NT",
    )


@pytest.mark.parametrize(
    ("fit_count", "prediction_count", "rois"),
    ((0, 1, 2), (2, 0, 2), (2, 1, 1)),
)
def test_patient_partition_rejects_invalid_bounds(
    fit_count: int, prediction_count: int, rois: int
) -> None:
    with pytest.raises(ValueError):
        select_patient_partition(
            _samples(),
            fit_patient_count=fit_count,
            prediction_patient_count=prediction_count,
            rois_per_patient=rois,
        )


def test_prediction_records_preserve_binary_values_and_keys() -> None:
    sample_ids = ("P3/ROI_01_C01_T", "P3/ROI_02_C01_NT")

    records = make_prediction_records(sample_ids, (1, 0), (0.75, 0.25))

    assert tuple(record.key.sample_id for record in records) == sample_ids
    assert all(record.key.fold_id == FOLD_ID for record in records)
    assert all(record.key.seed == 42 for record in records)
    assert all(record.key.inference_branch == LOCAL_ROUTE_A_INFERENCE_BRANCH for record in records)
    assert tuple(record.y_true for record in records) == (1, 0)
    assert tuple(record.y_pred for record in records) == (1, 0)
    assert tuple(record.probability for record in records) == (0.75, 0.25)


def test_prediction_records_reject_invalid_lengths_labels_and_probabilities() -> None:
    with pytest.raises(ValueError, match="equal length"):
        make_prediction_records(("sample",), (), ())
    with pytest.raises(ValueError, match="binary"):
        make_prediction_records(("sample",), (2,), (0.5,))
    with pytest.raises(ValueError, match="finite"):
        make_prediction_records(("sample",), (1,), (float("nan"),))
    with pytest.raises(ValueError, match="finite"):
        make_prediction_records(("sample",), (1,), (1.1,))
