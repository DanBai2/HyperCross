"""Unit coverage for task 14.9 train-only smoke-runner glue.

Validates Requirements 5.13-5.24, 8.28-8.30, 13.1-13.29,
14.7-14.18, 16.37, 16.38.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.runners.hyperleaf_smoke import (
    FOLD_ID,
    partition_training_ids,
    make_prediction_records,
)


def test_partition_training_ids_is_deterministic_disjoint_and_bounded() -> None:
    observed = partition_training_ids(tuple(reversed([f"{index:05d}" for index in range(12)])))

    assert observed.fit_ids == tuple(f"{index:05d}" for index in range(6))
    assert observed.prediction_ids == ("00006", "00007")
    assert len(observed.all_ids) == 8
    assert set(observed.fit_ids).isdisjoint(observed.prediction_ids)


@pytest.mark.parametrize(
    ("max_samples", "prediction_samples"),
    ((3, 1), (8, 0), (8, 8)),
)
def test_partition_training_ids_rejects_invalid_budgets(
    max_samples: int,
    prediction_samples: int,
) -> None:
    with pytest.raises(ValueError):
        partition_training_ids(
            tuple(f"{index:05d}" for index in range(12)),
            max_samples=max_samples,
            prediction_samples=prediction_samples,
        )


def test_prediction_records_preserve_each_train_only_sample_and_three_targets() -> None:
    sample_ids = ("00006", "00007")
    truth = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    prediction = truth + 0.25

    records = make_prediction_records(sample_ids, truth, prediction)

    assert tuple(record.key.sample_id for record in records) == sample_ids
    assert all(record.key.fold_id == FOLD_ID for record in records)
    assert all(record.key.seed == 42 for record in records)
    assert all(record.key.inference_branch == "global-fusion-smoke" for record in records)
    assert records[0].y_true == (1.0, 2.0, 3.0)
    assert records[1].y_pred == (4.25, 5.25, 6.25)


def test_prediction_records_reject_shape_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        make_prediction_records(("00006",), np.ones((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        make_prediction_records(
            ("00006",),
            np.ones((1, 3)),
            np.asarray([[1.0, np.nan, 3.0]]),
        )
