"""Unit coverage for task 15.11 Histology baseline glue.

Validates Requirements 1.1-1.5, 6.1-6.13, 9.9-9.26,
11.1-11.10, 13.33, and 13.34.
"""
from __future__ import annotations

import pytest
import torch

from benchmark.models.histology_baseline import (
    GLOBAL_ROUTE_A_CONTRACT,
    GLOBAL_ROUTE_A_MODEL_KEY,
    create_histology_baseline_registry,
)
from benchmark.runners.histology_baseline import (
    FOLD_ID,
    compare_binary_metric_records,
    fixed_patient_split,
    make_prediction_records,
)


def _fixed_fixture() -> tuple[tuple[str, ...], dict[str, str]]:
    sample_ids = tuple(f"P{patient}/ROI_{roi:02d}_C01_{'T' if roi == 1 else 'NT'}" for patient in range(1, 14) for roi in (1, 2))
    groups = {sample: sample.split("/", 1)[0] for sample in sample_ids}
    return sample_ids, groups


def test_fixed_patient_split_recreates_exact_legacy_groups() -> None:
    sample_ids, groups = _fixed_fixture()

    split = fixed_patient_split(sample_ids, groups)

    assert split.train_groups == ("P1", "P11", "P12", "P2", "P3", "P4", "P6", "P7", "P9")
    assert split.validation_groups == ("P10", "P13", "P5", "P8")
    assert set(split.train_groups).isdisjoint(split.validation_groups)
    assert set(split.train_ids) | set(split.validation_ids) == set(sample_ids)
    assert split == fixed_patient_split(sample_ids, groups)


def test_fixed_patient_split_rejects_bad_group_identity() -> None:
    sample_ids, groups = _fixed_fixture()
    with pytest.raises(ValueError, match="exactly one patient group"):
        fixed_patient_split(sample_ids + (sample_ids[0],), groups)
    with pytest.raises(ValueError, match="13 patients"):
        fixed_patient_split(sample_ids[:-2], {key: value for key, value in groups.items() if not key.startswith("P13/")})


def test_binary_metric_comparison_records_tolerance_and_mismatch() -> None:
    within = compare_binary_metric_records({"loss": 0.5, "acc": 0.75}, {"loss": 0.5000001, "acc": 0.75})
    outside = compare_binary_metric_records({"loss": 0.5, "acc": 0.75}, {"loss": 0.51, "acc": 0.75})

    assert within["all_within_tolerance"] is True
    assert outside["all_within_tolerance"] is False
    assert outside["loss"]["recorded"] == 0.5
    assert outside["loss"]["observed"] == 0.51


def test_prediction_records_preserve_roi_identity_and_fixed_threshold() -> None:
    records = make_prediction_records(("P1/ROI_01_C01_T", "P2/ROI_01_C01_NT"), (1, 0), (0.5, 0.49), branch="test-branch")

    assert tuple(record.key.sample_id for record in records) == ("P1/ROI_01_C01_T", "P2/ROI_01_C01_NT")
    assert all(record.key.fold_id == FOLD_ID for record in records)
    assert tuple(record.y_pred for record in records) == (1, 0)
    with pytest.raises(ValueError, match="finite probabilities"):
        make_prediction_records(("sample",), (1,), (float("nan"),), branch="bad")


def test_global_route_a_registry_has_stable_identity_and_contract() -> None:
    registry = create_histology_baseline_registry()
    registration = registry.resolve(GLOBAL_ROUTE_A_MODEL_KEY)
    model = registry.build(GLOBAL_ROUTE_A_MODEL_KEY)
    metadata = registry.run_metadata(GLOBAL_ROUTE_A_MODEL_KEY, model, config_hash="a" * 64, supplied_contract=GLOBAL_ROUTE_A_CONTRACT.model_copy(deep=True))

    assert metadata.model_key == GLOBAL_ROUTE_A_MODEL_KEY
    assert registration.input_contract.tensor_shapes == {"rgb": (3, 64, 64), "global_pca_spectrum": (128,)}
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64), torch.zeros(2, 128))
    assert output.shape == (2, 2)
