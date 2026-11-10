"""Unit coverage for task 14.10 governed HyperLeaf baseline glue.

Validates Requirements 1.1-1.5, 6.1-6.13, 11.1-11.10,
13.33, 13.34, and 17.1-17.24.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.models.hyperleaf_baseline import (
    CONCAT_MODEL_KEY,
    SPECTRAL_MODEL_KEY,
    VISUAL_MODEL_KEY,
    create_hyperleaf_baseline_registry,
)
from benchmark.runners.hyperleaf_baseline import (
    FOLD_ID,
    compare_metric_records,
    legacy_random_split,
    make_prediction_records,
)


def test_legacy_random_split_recreates_exact_disjoint_80_20_membership() -> None:
    sample_ids = tuple(f"{index:05d}" for index in range(10))

    observed = legacy_random_split(sample_ids)
    repeated = legacy_random_split(sample_ids)

    assert observed == repeated
    assert len(observed.train_ids) == 8
    assert len(observed.validation_ids) == 2
    assert set(observed.train_ids).isdisjoint(observed.validation_ids)
    assert set(observed.train_ids) | set(observed.validation_ids) == set(sample_ids)


def test_legacy_random_split_rejects_duplicate_or_too_small_inputs() -> None:
    with pytest.raises(ValueError, match="unique"):
        legacy_random_split(("0", "1", "2", "3", "3"))
    with pytest.raises(ValueError, match="at least five"):
        legacy_random_split(("0", "1", "2", "3"))


def test_metric_comparison_reports_each_field_and_tolerance_result() -> None:
    recorded = {"loss": 1.0, "r2_mean": 0.5, "r2": [0.2, 0.5, 0.8], "mae": 0.4}
    observed = {"loss": 1.0 + 1e-7, "r2_mean": 0.5, "r2": [0.2, 0.5, 0.8], "mae": 0.4}

    within = compare_metric_records(recorded, observed, tolerance=1e-6)
    outside = compare_metric_records(recorded, {**observed, "mae": 0.5}, tolerance=1e-6)

    assert within["all_within_tolerance"] is True
    assert tuple(item["target"] for item in within["r2"]) == ("GrainWeight", "Gsw", "PhiPS2")
    assert outside["all_within_tolerance"] is False
    assert outside["mae"]["within_tolerance"] is False


def test_prediction_records_are_complete_finite_vector_records() -> None:
    ids = ("00001", "00002")
    truth = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    prediction = truth + 0.25

    records = make_prediction_records(ids, truth, prediction, branch="route-a")

    assert tuple(record.key.sample_id for record in records) == ids
    assert all(record.key.fold_id == FOLD_ID for record in records)
    assert all(record.key.seed == 42 for record in records)
    assert all(record.key.inference_branch == "route-a" for record in records)
    assert records[1].y_pred == (2.25, 3.25, 4.25)

    with pytest.raises(ValueError, match="shape"):
        make_prediction_records(ids, truth[:, :2], prediction[:, :2], branch="bad")
    with pytest.raises(ValueError, match="finite"):
        make_prediction_records(ids, truth, np.full((2, 3), np.nan), branch="bad")


def test_baseline_control_registry_has_stable_identity_and_contracts() -> None:
    registry = create_hyperleaf_baseline_registry()

    assert registry.model_keys == tuple(sorted((CONCAT_MODEL_KEY, SPECTRAL_MODEL_KEY, VISUAL_MODEL_KEY)))
    for model_key in registry.model_keys:
        registration = registry.resolve(model_key)
        model = registry.build(model_key)
        metadata = registry.run_metadata(
            model_key,
            model,
            config_hash="a" * 64,
            supplied_contract=registration.input_contract.model_copy(deep=True),
        )
        assert metadata.model_key == registration.identity.model_key
        assert metadata.source_revision == registration.identity.source_revision
        assert metadata.parameter_count == registration.identity.parameter_count
        assert registration.input_contract.modalities == ("visible_rgb", "global_spectrum")
        assert registration.input_contract.tensor_shapes["visible_rgb"] == (3, 32, 64)
        assert registration.input_contract.tensor_shapes["global_spectrum"] == (128,)
