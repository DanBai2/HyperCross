"""Aggregate/statistics integration coverage for task 10.11.

Validates: Requirements 8.20-8.23, 9.9-9.32, 11.11-11.31.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

import pytest

from benchmark.metrics import (
    AggregateBuildError,
    AggregateBuilder,
    PredictionArtifactWriter,
    binary_metrics,
    merge_prediction_artifacts,
    paired_patient_compare,
    patient_bootstrap,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.registry import (
    ArtifactRecord,
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.schema import (
    AggregationPolicy,
    ClaimStatus,
    ImplementationKind,
    LimitEnforcement,
    MetricSchema,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    PredictionShard,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    RuntimeMeasurements,
    TaskKind,
)


_NOW = datetime(2027, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
_HASHES = {
    name: f"{index:064x}"
    for index, name in enumerate(
        ("dataset", "protocol", "config", "environment", "source"), start=1
    )
}
_SCHEMA = MetricSchema(
    task_kind=TaskKind.BINARY_CLASSIFICATION,
    numeric_tolerance=1e-12,
    decision_threshold=0.5,
    ece_bins=5,
)


def _record(
    sample_id: str,
    fold_id: str,
    truth: int,
    probability: float,
) -> PredictionRecord:
    return PredictionRecord(
        key=PredictionKey(
            sample_id=sample_id,
            fold_id=fold_id,
            seed=17,
            inference_branch="primary",
        ),
        y_true=truth,
        y_pred=int(probability >= 0.5),
        probability=probability,
    )


def _pooled_artifact(
    artifact_id: str,
    records: Sequence[PredictionRecord],
) -> PredictionArtifact:
    pooled = tuple(
        record.model_copy(
            update={
                "key": record.key.model_copy(update={"fold_id": "pooled-oof"})
            }
        )
        for record in records
    )
    return PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=f"run-{artifact_id}",
        records=pooled,
        expected_sample_ids=tuple(record.key.sample_id for record in pooled),
        complete=True,
    )


class _HistologyOofMetricComputer:
    """Test adapter that exposes fold-macro and pooled OOF as distinct names."""

    def compute(
        self,
        records: Tuple[PredictionRecord, ...],
        metric_schema: MetricSchema,
    ) -> Mapping[str, Any]:
        grouped: dict[str, list[PredictionRecord]] = {}
        for record in records:
            grouped.setdefault(record.key.fold_id, []).append(record)

        per_fold_accuracy: dict[str, float] = {}
        for fold_id, fold_records in sorted(grouped.items()):
            artifact = PredictionArtifact.create(
                artifact_id=f"metric-{fold_id}",
                run_id=f"metric-run-{fold_id}",
                records=tuple(fold_records),
                expected_sample_ids=tuple(
                    record.key.sample_id for record in fold_records
                ),
                complete=True,
            )
            value = binary_metrics(artifact, metric_schema).metric("accuracy").value
            assert value is not None
            per_fold_accuracy[fold_id] = value

        pooled = binary_metrics(
            _pooled_artifact("metric-pooled", records), metric_schema
        )
        pooled_values = {
            name: metric.value for name, metric in pooled.metrics.items()
        }
        assert all(value is not None for value in pooled_values.values())
        return {
            **{
                f"per_fold_accuracy.{fold_id}": value
                for fold_id, value in per_fold_accuracy.items()
            },
            "fold_macro_accuracy": math.fsum(per_fold_accuracy.values())
            / len(per_fold_accuracy),
            **{
                f"pooled_oof_{name}": value
                for name, value in pooled_values.items()
            },
            "pooled_oof_confusion_matrix": pooled.confusion_matrices[
                "confusion_matrix"
            ],
        }


def _policy(fold_ids: Iterable[str]) -> AggregationPolicy:
    return AggregationPolicy(
        policy_key="histology-fold-macro-and-pooled-oof.v1",
        metric_schema=_SCHEMA,
        variance_metric="pooled_oof_accuracy",
        seed_set=(17,),
        fold_ids=tuple(fold_ids),
    )


def _model() -> ModelIdentity:
    return ModelIdentity(
        model_key="classic.svm.v1",
        display_name="SVM",
        class_fully_qualified_name="sklearn.svm.SVC",
        implementation_kind=ImplementationKind.CLASSIC,
        parameter_count=8,
        source_revision="revision-1",
    )


def _contract() -> InputContract:
    return InputContract(
        representation_key="histology-spatial-mean-pca.v1",
        modalities=("spectrum",),
        tensor_shapes={"spectrum": (3,)},
        dtypes={"spectrum": "float32"},
        channel_or_band_order={"spectrum": ("pc1", "pc2", "pc3")},
        preprocessing_key="fold-pca.v1",
    )


def _provenance() -> Provenance:
    return Provenance(
        code_revision="revision-1",
        source_hash=_HASHES["source"],
        environment_hash=_HASHES["environment"],
        config_hash=_HASHES["config"],
        dataset_hash=_HASHES["dataset"],
    )


def _claim() -> ClaimDecision:
    return ClaimDecision(
        status=ClaimStatus.PRIMARY_STRICT,
        supporting_reasons=("patient_isolation_valid",),
        prerequisite_checks={"predictions": True, "folds": True},
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        limits=(
            ResourceLimit(
                field="completed_epochs",
                value=1,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )


def _allocate(registry: RunRegistry, fold_id: str) -> RunRecord:
    return registry.allocate_run(
        frozen_identity=FrozenRunIdentity(
            dataset_hash=_HASHES["dataset"],
            protocol_hash=_HASHES["protocol"],
            model_key="classic.svm.v1",
            config_hash=_HASHES["config"],
            fold_id=fold_id,
            seed=17,
            environment_lock_hash=_HASHES["environment"],
            environment_compatibility_policy="exact",
        ),
        model_identity=_model(),
        input_contract=_contract(),
        provenance=_provenance(),
        resource_budget=_budget(),
        claim_decision=_claim(),
        started_utc=_NOW,
    )


def _complete(
    registry: RunRegistry,
    store: FailureAtomicArtifactStore,
    fold_id: str,
    shards: Tuple[PredictionShard, ...],
) -> tuple[RunRecord, PredictionArtifact]:
    planned = _allocate(registry, fold_id)
    writer = PredictionArtifactWriter(store)
    expected_ids = tuple(
        record.key.sample_id for shard in shards for record in shard.records
    )

    forward = writer.build(
        artifact_id=f"prediction-{planned.run_id}",
        run_id=planned.run_id,
        expected_sample_ids=expected_ids,
        shards=shards,
    )
    reordered = writer.build(
        artifact_id=f"prediction-{planned.run_id}",
        run_id=planned.run_id,
        expected_sample_ids=reversed(expected_ids),
        shards=reversed(shards),
    )
    assert reordered == forward

    published = writer.publish_shards(
        artifact_id=f"prediction-{planned.run_id}",
        run_id=planned.run_id,
        expected_sample_ids=reversed(expected_ids),
        shards=reversed(shards),
    )
    checkpoint = store.publish_bytes(
        artifact_id=f"checkpoint-{planned.run_id}",
        relative_path=f"checkpoints/{planned.run_id}.bin",
        content=b"frozen checkpoint",
    )
    registry.append_artifact(
        ArtifactRecord(
            artifact=checkpoint,
            artifact_kind="checkpoint",
            run_id=planned.run_id,
            created_utc=_NOW + timedelta(seconds=1),
        )
    )
    registry.append_artifact(
        ArtifactRecord(
            artifact=published.reference,
            artifact_kind="prediction",
            run_id=planned.run_id,
            created_utc=_NOW + timedelta(seconds=2),
        )
    )
    running = RunRecord.model_validate(
        {**planned.model_dump(mode="python"), "status": RunStatus.RUNNING}
    )
    registry.append_run(running)
    complete = RunRecord.model_validate(
        {
            **running.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "checkpoint_ref": checkpoint,
            "prediction_artifact_ref": published.reference,
            "runtime_measurements": RuntimeMeasurements(
                completed_epochs=1,
                completed_training_steps=1,
                wall_clock_duration_ms=1,
                peak_device_memory_bytes=1,
                device_identity="cpu",
                numeric_precision="float32",
                parameter_count=8,
                input_volume_bytes=32,
            ),
            "completed_utc": _NOW + timedelta(seconds=3),
        }
    )
    registry.append_run(complete)
    return complete, published.artifact


def _fold_shards() -> dict[str, Tuple[PredictionShard, ...]]:
    fold_a = (
        _record("a-0", "fold-a", 0, 0.1),
        _record("a-1", "fold-a", 1, 0.9),
    )
    fold_b = (
        _record("b-0", "fold-b", 0, 0.9),
        _record("b-1", "fold-b", 1, 0.1),
        _record("b-2", "fold-b", 0, 0.9),
        _record("b-3", "fold-b", 1, 0.9),
        _record("b-4", "fold-b", 0, 0.1),
        _record("b-5", "fold-b", 1, 0.9),
    )
    return {
        "fold-a": (
            PredictionShard(shard_id="fold-a-second", records=(fold_a[1],)),
            PredictionShard(shard_id="fold-a-first", records=(fold_a[0],)),
        ),
        "fold-b": (
            PredictionShard(shard_id="fold-b-second", records=fold_b[3:]),
            PredictionShard(shard_id="fold-b-first", records=fold_b[:3]),
        ),
    }


def test_shard_reordering_preserves_aggregate_and_separates_fold_macro_from_pooled_oof(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts with spaces")
    completed_and_artifacts = [
        _complete(registry, store, fold_id, shards)
        for fold_id, shards in reversed(tuple(_fold_shards().items()))
    ]
    completed = tuple(item[0] for item in completed_and_artifacts)
    artifacts = tuple(item[1] for item in completed_and_artifacts)

    computer = _HistologyOofMetricComputer()
    forward_records = merge_prediction_artifacts(artifacts)
    reverse_records = merge_prediction_artifacts(reversed(artifacts))
    assert forward_records == reverse_records
    assert computer.compute(forward_records, _SCHEMA) == computer.compute(
        reverse_records, _SCHEMA
    )

    policy = _policy(("fold-b", "fold-a"))
    first = AggregateBuilder(registry, store.root, computer).build(
        aggregate_id="histology-oof-forward",
        member_run_ids=(run.run_id for run in completed),
        aggregation_policy=policy,
        created_utc=_NOW + timedelta(minutes=1),
    )
    second = AggregateBuilder(registry, store.root, computer).build(
        aggregate_id="histology-oof-reversed",
        member_run_ids=(run.run_id for run in reversed(completed)),
        aggregation_policy=policy,
        created_utc=_NOW + timedelta(minutes=2),
    )

    assert first.member_run_ids == second.member_run_ids
    assert first.prediction_hashes == second.prediction_hashes
    assert first.metrics == second.metrics
    assert first.metrics["per_fold_accuracy.fold-a"] == pytest.approx(1.0)
    assert first.metrics["per_fold_accuracy.fold-b"] == pytest.approx(0.5)
    assert first.metrics["fold_macro_accuracy"] == pytest.approx(0.75)
    assert first.metrics["pooled_oof_accuracy"] == pytest.approx(0.625)
    assert first.metrics["fold_macro_accuracy"] != first.metrics[
        "pooled_oof_accuracy"
    ]
    assert first.metrics["pooled_oof_confusion_matrix"] == ((2, 2), (1, 3))


def test_published_oof_predictions_drive_seeded_patient_bootstrap_and_paired_ids(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    artifacts = tuple(
        _complete(registry, store, fold_id, shards)[1]
        for fold_id, shards in _fold_shards().items()
    )
    candidate = _pooled_artifact(
        "candidate-pooled-oof", merge_prediction_artifacts(artifacts)
    )
    candidate_groups = {
        "a-0": "patient-a",
        "a-1": "patient-a",
        "b-0": "patient-b",
        "b-1": "patient-b",
        "b-2": "patient-c",
        "b-3": "patient-c",
        "b-4": "patient-d",
        "b-5": "patient-d",
    }

    control_records = (
        _record("control-b-0", "control-oof", 0, 0.1),
        _record("control-b-1", "control-oof", 1, 0.1),
        _record("control-c-0", "control-oof", 0, 0.1),
        _record("control-c-1", "control-oof", 1, 0.9),
        _record("control-e-0", "control-oof", 0, 0.9),
        _record("control-e-1", "control-oof", 1, 0.9),
    )
    control = PredictionArtifactWriter(store).build(
        artifact_id="control-pooled-oof",
        run_id="run-control-pooled-oof",
        expected_sample_ids=tuple(record.key.sample_id for record in control_records),
        shards=(
            PredictionShard(shard_id="control-tail", records=control_records[3:]),
            PredictionShard(shard_id="control-head", records=control_records[:3]),
        ),
    )
    control_groups = {
        "control-b-0": "patient-b",
        "control-b-1": "patient-b",
        "control-c-0": "patient-c",
        "control-c-1": "patient-c",
        "control-e-0": "patient-e",
        "control-e-1": "patient-e",
    }

    first_interval = patient_bootstrap(
        candidate,
        "accuracy",
        group_by_sample_id=candidate_groups,
        metric_schema=_SCHEMA,
        seed=2027,
    )
    second_interval = patient_bootstrap(
        candidate,
        "accuracy",
        group_by_sample_id=candidate_groups,
        metric_schema=_SCHEMA,
        seed=2027,
    )
    comparison = paired_patient_compare(
        candidate,
        control,
        "accuracy",
        candidate_group_by_sample_id=candidate_groups,
        control_group_by_sample_id=control_groups,
        metric_schema=_SCHEMA,
        seed=2028,
    )

    assert first_interval == second_interval
    assert first_interval.source_prediction_hash == candidate.content_hash
    assert first_interval.resampling_unit == "patient_group_id"
    assert first_interval.resamples == 10_000
    assert first_interval.confidence_level == 0.95
    assert first_interval.sidedness == "two_sided"
    assert first_interval.seed == 2027

    assert comparison.common_group_ids == ("patient-b", "patient-c")
    assert comparison.candidate_prediction_hash == candidate.content_hash
    assert comparison.control_prediction_hash == control.content_hash
    assert comparison.resampling_unit == "patient_group_id"
    assert comparison.resamples == 10_000
    assert comparison.confidence_level == 0.95
    assert comparison.sidedness == "two_sided"
    assert comparison.seed == 2028


def test_incomplete_declared_member_rejects_aggregate_without_mutating_completed_evidence(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    complete, _ = _complete(
        registry, store, "fold-a", _fold_shards()["fold-a"]
    )
    incomplete = _allocate(registry, "fold-b")
    prediction_reference = complete.prediction_artifact_ref
    assert prediction_reference is not None
    prediction_path = store.root / Path(prediction_reference.relative_path)
    prediction_before = prediction_path.read_bytes()
    registry_before = registry.path.read_bytes()

    with pytest.raises(AggregateBuildError) as captured:
        AggregateBuilder(
            registry, store.root, _HistologyOofMetricComputer()
        ).build(
            aggregate_id="histology-incomplete",
            member_run_ids=(complete.run_id, incomplete.run_id),
            aggregation_policy=_policy(("fold-a", "fold-b")),
        )

    assert captured.value.error_record.error_code == "AGGREGATE_MEMBER_INCOMPLETE"
    assert captured.value.error_record.observed_value == {
        "run_id": incomplete.run_id,
        "status": RunStatus.PLANNED.value,
    }
    assert registry.path.read_bytes() == registry_before
    assert prediction_path.read_bytes() == prediction_before
    with pytest.raises(KeyError):
        registry.get_aggregate("histology-incomplete")
