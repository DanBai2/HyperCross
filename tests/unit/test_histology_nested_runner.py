"""Unit coverage for task 15.2 strict nested Histology wiring."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.engine import SelectionValidationError
from benchmark.protocols import (
    HISTOLOGY_NESTED5_PROTOCOL_KEY,
    HistologyNested5Planner,
    ProtocolPlanner,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.runners import (
    HISTOLOGY_NESTED_RELATIVE_OUTPUT_ROOT,
    HISTOLOGY_NESTED_REQUIRED_METRICS,
    HistologyNestedRunner,
    HistologyNestedRunnerError,
)
from benchmark.schema import SampleManifest, SampleRecord, canonical_sample_order


HASH = "a" * 64


def _manifest(*, reverse: bool = False) -> SampleManifest:
    records = [
        SampleRecord(
            sample_id=f"P{patient}/ROI_{roi:02d}_{'T' if roi % 2 else 'NT'}",
            group_id=f"P{patient}",
            label_reference="roi-directory-class-suffix",
            input_references={"cube": f"P{patient}/ROI_{roi:02d}/raw"},
            input_content_hashes={"cube": f"{patient * 100 + roi:064x}"},
            label_schema_metadata={"dimension": 1, "label_value": roi % 2},
        )
        for patient in range(1, 14)
        for roi in range(1, (patient % 4) + 3)
    ]
    if reverse:
        records.reverse()
    return SampleManifest(
        schema_version="1.0",
        dataset_key="histology-hsi-gb.synthetic",
        records=tuple(records),
        canonical_order=canonical_sample_order(tuple(item.sample_id for item in records)),
        dataset_hash=HASH,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


def _patient(sample_id: str) -> str:
    return sample_id.split("/", 1)[0]


def _planner(tmp_path: Path) -> HistologyNested5Planner:
    return HistologyNested5Planner(
        ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts"))
    )


def test_nested_five_plan_is_patient_isolated_and_hash_stable(tmp_path: Path) -> None:
    first = _planner(tmp_path / "first").plan(_manifest(), seed_set=(7, 42, 123))
    second = _planner(tmp_path / "second").plan(
        _manifest(reverse=True), seed_set=(123, 7, 42)
    )

    assert first.protocol_key == HISTOLOGY_NESTED5_PROTOCOL_KEY
    assert first.protocol_hash == second.protocol_hash
    assert first.folds == second.folds
    assert len(first.folds) == 5
    assert first.evaluation_budget.maximum_successful_evaluations == 15

    outer_test_groups = []
    expected_ids = set(_manifest().canonical_order)
    for outer in first.folds:
        train_groups = {_patient(sample_id) for sample_id in outer.outer_train_ids}
        test_groups = {_patient(sample_id) for sample_id in outer.outer_test_ids}
        assert train_groups.isdisjoint(test_groups)
        assert set(outer.outer_train_ids) | set(outer.outer_test_ids) == expected_ids
        assert len(outer.inner_folds) == 5
        outer_test_groups.extend(test_groups)
        for inner in outer.inner_folds:
            inner_train_groups = {_patient(sample_id) for sample_id in inner.train_ids}
            inner_validation_groups = {
                _patient(sample_id) for sample_id in inner.validation_ids
            }
            assert inner_train_groups.isdisjoint(inner_validation_groups)
            assert set(inner.train_ids) | set(inner.validation_ids) == set(
                outer.outer_train_ids
            )
    assert set(outer_test_groups) == {f"P{patient}" for patient in range(1, 14)}
    assert len(outer_test_groups) == 13


def test_protocol_publication_is_relative_and_canonical(tmp_path: Path) -> None:
    planner = _planner(tmp_path)
    published = planner.plan_and_publish(_manifest(), seed_set=(7, 42, 123))
    target = tmp_path / "artifacts" / Path(published.artifact.relative_path)

    assert not Path(published.artifact.relative_path).is_absolute()
    assert target.is_file()
    assert planner.planner.reload(target.read_bytes()) == published.manifest


def test_runner_freezes_complete_inner_only_config_epoch_selection() -> None:
    runner = HistologyNestedRunner()
    protocol = runner.protocol_planner.plan(_manifest(), seed_set=runner.seed_set)
    outer = protocol.folds[0]
    candidates = runner.build_selection_space(
        {"base": {"learning_rate": 0.001}, "regularized": {"weight_decay": 0.01}},
        epochs=(20, 40),
    )
    source_ids = tuple(
        sample_id
        for inner in outer.inner_folds
        for sample_id in inner.validation_ids
    )
    scores = {candidate.candidate_id: float(index) for index, candidate in enumerate(candidates)}

    decision = runner.freeze_config_epoch(
        outer,
        candidates=candidates,
        candidate_scores=scores,
        inner_source_sample_ids=source_ids,
        inner_prediction_hash="b" * 64,
        decision_id="outer-01-config-epoch",
    )

    assert decision.selected_candidate == candidates[-1].candidate_id
    assert set(decision.source_sample_ids) == set(outer.outer_train_ids)
    assert set(decision.source_sample_ids).isdisjoint(outer.outer_test_ids)
    assert all("config_key" in item.configuration for item in decision.candidates)
    assert all("epoch" in item.configuration for item in decision.candidates)

    with pytest.raises(HistologyNestedRunnerError, match="complete inner-validation"):
        runner.freeze_config_epoch(
            outer,
            candidates=candidates,
            candidate_scores=scores,
            inner_source_sample_ids=source_ids[:-1],
            inner_prediction_hash="c" * 64,
        )
    with pytest.raises((HistologyNestedRunnerError, SelectionValidationError)):
        runner.freeze_config_epoch(
            outer,
            candidates=candidates,
            candidate_scores=scores,
            inner_source_sample_ids=source_ids + (outer.outer_test_ids[0],),
            inner_prediction_hash="d" * 64,
        )


def test_runner_binds_strict_local_contract_and_one_shot_jobs() -> None:
    runner = HistologyNestedRunner()
    protocol = runner.protocol_planner.plan(_manifest(), seed_set=runner.seed_set)
    jobs = runner.schedule(protocol)

    assert runner.template.conda_environment == "TCFM"
    assert runner.template.expected_device_name == "NVIDIA GeForce RTX 5080"
    assert runner.template.process_count == 1
    assert runner.template.checkpoint_interval_epochs == 1
    assert runner.template.checkpoint_retention.preserve_recovery_state is True
    assert runner.relative_output_root == HISTOLOGY_NESTED_RELATIVE_OUTPUT_ROOT
    assert not Path(runner.relative_output_root).is_absolute()
    assert len(jobs) == len(protocol.folds) * len(protocol.seed_set)
    assert len({(job.fold_id, job.seed) for job in jobs}) == len(jobs)
    assert all(job.successful_outer_evaluations == 1 for job in jobs)
    assert all(job.recovery_boundaries == ("epoch", "fold") for job in jobs)
    assert HISTOLOGY_NESTED_REQUIRED_METRICS == (
        "fold_macro_accuracy",
        "pooled_oof_accuracy",
        "pooled_oof_balanced_accuracy",
        "pooled_oof_auroc",
        "pooled_oof_sensitivity",
        "pooled_oof_specificity",
        "pooled_oof_confusion_matrix",
        "pooled_oof_ece",
    )


def test_runner_rejects_f_drive_workspace() -> None:
    with pytest.raises(HistologyNestedRunnerError, match=r"G:\\Datasets\\HyperCross"):
        HistologyNestedRunner(r"F:\Datasets\HyperCross")
