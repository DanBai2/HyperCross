"""Unit coverage for task 15.3 fixed-epoch Histology LOPO wiring."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import SimpleNamespace

import pytest

from benchmark.protocols import (
    HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY,
    HistologyFixedEpochPolicy,
    HistologyLopoFixedPlanner,
    ProtocolPlanner,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.runners import (
    HISTOLOGY_LOPO_RELATIVE_OUTPUT_ROOT,
    HISTOLOGY_LOPO_REQUIRED_METRICS,
    HistologyLopoFixedRunner,
    HistologyLopoFixedRunnerError,
)
from benchmark.schema import (
    PredictionKey,
    PredictionRecord,
    SampleManifest,
    SampleRecord,
    canonical_sample_order,
)


def _manifest(*, reverse: bool = False) -> SampleManifest:
    records = [
        SampleRecord(
            sample_id=f"P{patient}/ROI-{roi:02d}",
            group_id=f"P{patient}",
            label_reference="synthetic-label",
            input_references={"cube": f"P{patient}/ROI-{roi:02d}.bin"},
            input_content_hashes={"cube": f"{patient * 100 + roi:064x}"},
            label_schema_metadata={"dimension": 1, "label_value": roi % 2},
        )
        for patient in range(1, 5)
        for roi in range(1, patient + 2)
    ]
    if reverse:
        records.reverse()
    return SampleManifest(
        schema_version="1.0",
        dataset_key="histology.synthetic",
        records=tuple(records),
        canonical_order=canonical_sample_order(record.sample_id for record in records),
        dataset_hash="a" * 64,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


def _planner(tmp_path: Path, epochs: int = 20) -> HistologyLopoFixedPlanner:
    return HistologyLopoFixedPlanner(
        ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts")),
        epoch_policy=HistologyFixedEpochPolicy(epochs),
    )


def _patient(sample_id: str) -> str:
    return sample_id.split("/", 1)[0]


def test_lopo_plan_is_patient_isolated_exact_and_policy_hash_stable(tmp_path: Path) -> None:
    first = _planner(tmp_path / "first").plan(_manifest(), seed_set=(7, 42, 123))
    second = _planner(tmp_path / "second").plan(
        _manifest(reverse=True), seed_set=(123, 7, 42)
    )

    assert first.protocol_key == HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY
    assert first.protocol_hash == second.protocol_hash
    assert first.folds == second.folds
    assert len(first.folds) == 4
    assert first.selection_policy.policy_key == "hist.fixed-epoch-0020.v1"
    assert first.evaluation_budget.maximum_successful_evaluations == 12

    expected_ids = set(_manifest().canonical_order)
    held_out_ids: set[str] = set()
    held_out_patients: set[str] = set()
    for fold in first.folds:
        train_patients = {_patient(item) for item in fold.outer_train_ids}
        test_patients = {_patient(item) for item in fold.outer_test_ids}
        assert len(test_patients) == 1
        assert train_patients.isdisjoint(test_patients)
        assert set(fold.outer_train_ids) | set(fold.outer_test_ids) == expected_ids
        assert held_out_ids.isdisjoint(fold.outer_test_ids)
        held_out_ids.update(fold.outer_test_ids)
        held_out_patients.update(test_patients)
        assert fold.inner_folds == ()
    assert held_out_ids == expected_ids
    assert held_out_patients == {"P1", "P2", "P3", "P4"}


def test_runner_freezes_epoch_before_jobs_and_binds_strict_contract() -> None:
    runner = HistologyLopoFixedRunner(epoch_policy=30)
    protocol = runner.protocol_planner.plan(_manifest(), seed_set=runner.seed_set)
    jobs = runner.schedule(protocol)
    decision = runner.freeze_epoch_decision(
        protocol,
        fold_id=protocol.folds[0].fold_id,
        frozen_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert runner.epoch_policy.epochs == 30
    assert protocol.selection_policy.policy_key == "hist.fixed-epoch-0030.v1"
    assert decision.selected_candidate == "fixed-epoch-0030"
    assert decision.candidates[0].configuration["epoch_policy_hash"] == runner.epoch_policy.policy_hash
    assert set(decision.source_sample_ids).isdisjoint(protocol.folds[0].outer_test_ids)
    assert len(jobs) == len(protocol.folds) * len(protocol.seed_set)
    assert len({(job.fold_id, job.seed) for job in jobs}) == len(jobs)
    assert all(job.successful_outer_evaluations == 1 for job in jobs)
    assert all(job.recovery_boundary == "patient_complete" for job in jobs)
    assert all(job.epoch_policy_hash == runner.epoch_policy.policy_hash for job in jobs)

    assert runner.template.conda_environment == "TCFM"
    assert runner.template.expected_device_name == "NVIDIA GeForce RTX 5080"
    assert runner.template.visible_cuda_device_count == 1
    assert runner.template.process_count == 1
    assert runner.relative_output_root == HISTOLOGY_LOPO_RELATIVE_OUTPUT_ROOT
    assert not Path(runner.relative_output_root).is_absolute()
    assert runner.execution_contract.read_only_roots == (
        "data_caches",
        "legacy_checkpoints",
        "runs",
        "src",
    )


def test_metric_wiring_requires_exact_once_only_oof_and_reports_pooled_values() -> None:
    runner = HistologyLopoFixedRunner()
    protocol = runner.protocol_planner.plan(_manifest(), seed_set=runner.seed_set)
    fold_by_sample = {
        sample_id: fold.fold_id
        for fold in protocol.folds
        for sample_id in fold.outer_test_ids
    }
    records = tuple(
        PredictionRecord(
            key=PredictionKey(
                sample_id=sample_id,
                fold_id=fold_by_sample[sample_id],
                seed=seed,
                inference_branch="pca-only",
            ),
            y_true=index % 2,
            y_pred=index % 2,
            probability=0.9 if index % 2 else 0.1,
        )
        for seed in protocol.seed_set
        for index, sample_id in enumerate(sorted(fold_by_sample))
    )

    metrics = runner.metric_computer(protocol).compute(records, runner.metric_schema())
    assert tuple(key for key in HISTOLOGY_LOPO_REQUIRED_METRICS if key not in metrics) == ()
    assert metrics["pooled_oof_accuracy"] == 1.0
    assert metrics["pooled_oof_balanced_accuracy"] == 1.0
    assert metrics["pooled_oof_auroc"] == 1.0
    assert metrics["pooled_oof_confusion_matrix"] == ((21, 0), (0, 21))
    policy = runner.aggregation_policy(protocol)
    assert policy.fold_ids == tuple(fold.fold_id for fold in protocol.folds)
    assert policy.seed_set == protocol.seed_set
    assert policy.metric_schema.decision_threshold == 0.5

    with pytest.raises(HistologyLopoFixedRunnerError, match="once-only OOF"):
        runner.metric_computer(protocol).compute(records[:-1], runner.metric_schema())


def test_patient_boundary_recovery_is_confined_to_relative_lopo_root() -> None:
    runner = HistologyLopoFixedRunner()
    protocol = runner.protocol_planner.plan(_manifest(), seed_set=runner.seed_set)
    fold_id = protocol.folds[0].fold_id
    captured = {}

    class FakeRecovery:
        registry = SimpleNamespace(
            latest_run=lambda run_id: SimpleNamespace(
                frozen_identity=SimpleNamespace(
                    protocol_hash=protocol.protocol_hash,
                    fold_id=fold_id,
                )
            )
        )

        def save_checkpoint_boundary(self, **kwargs):
            captured.update(kwargs)
            return kwargs

    outcome = runner.save_patient_boundary(
        FakeRecovery(),  # type: ignore[arg-type]
        protocol,
        run_id="run-1",
        fold_id=fold_id,
        model_state=b"model",
        optimizer_state=b"optimizer",
        scheduler_state=b"scheduler",
        stage_state=b"stage",
    )

    assert outcome["last_completed_stage"] == "outer_patient_complete"
    assert outcome["completed_checkpoint_boundary"] == 1
    assert outcome["relative_path"].startswith("recovery/")
    assert not PurePath(outcome["relative_path"]).is_absolute()


def test_runner_rejects_f_drive_and_invalid_epoch_policy() -> None:
    with pytest.raises(HistologyLopoFixedRunnerError, match=r"G:\\Datasets\\HyperCross"):
        HistologyLopoFixedRunner(r"F:\Datasets\HyperCross")
    with pytest.raises(ValueError, match="positive integer"):
        HistologyFixedEpochPolicy(0)
