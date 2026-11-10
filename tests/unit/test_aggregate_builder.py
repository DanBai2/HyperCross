"""Targeted unit coverage for task 10.3 aggregate construction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Tuple

import pytest

from benchmark.metrics import (
    AggregateBuildError,
    AggregateBuilder,
    PredictionArtifactWriter,
    merge_prediction_artifacts,
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
    PredictionKey,
    PredictionRecord,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    RuntimeMeasurements,
    TaskKind,
    hash_object,
)


NOW = datetime(2027, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
HASHES = {
    name: f"{index:064x}"
    for index, name in enumerate(
        ("dataset", "protocol", "config", "environment", "source"), start=1
    )
}


class _MeanPredictionComputer:
    def __init__(self) -> None:
        self.calls: list[Tuple[PredictionRecord, ...]] = []

    def compute(
        self,
        records: Tuple[PredictionRecord, ...],
        metric_schema: MetricSchema,
    ) -> Mapping[str, Any]:
        assert metric_schema.task_kind is TaskKind.MULTI_TARGET_REGRESSION
        self.calls.append(records)
        return {
            "mean_prediction": sum(float(record.y_pred) for record in records)
            / len(records)
        }


def _policy(*, folds: tuple[str, ...], seeds: tuple[int, ...]) -> AggregationPolicy:
    return AggregationPolicy(
        policy_key="pooled-oof-population-variance.v1",
        metric_schema=MetricSchema(
            task_kind=TaskKind.MULTI_TARGET_REGRESSION,
            numeric_tolerance=1e-12,
            target_names=("target",),
        ),
        variance_metric="mean_prediction",
        seed_set=seeds,
        fold_ids=folds,
    )


def _model() -> ModelIdentity:
    return ModelIdentity(
        model_key="classic.rf.v1",
        display_name="Random Forest",
        class_fully_qualified_name="sklearn.ensemble.RandomForestRegressor",
        implementation_kind=ImplementationKind.CLASSIC,
        parameter_count=10,
        source_revision="revision-1",
    )


def _contract() -> InputContract:
    return InputContract(
        representation_key="global-spectrum.v1",
        modalities=("spectrum",),
        tensor_shapes={"spectrum": (4,)},
        dtypes={"spectrum": "float32"},
        channel_or_band_order={"spectrum": ("wavelength_ascending",)},
        preprocessing_key="standardize.v1",
    )


def _provenance() -> Provenance:
    return Provenance(
        code_revision="revision-1",
        source_hash=HASHES["source"],
        environment_hash=HASHES["environment"],
        config_hash=HASHES["config"],
        dataset_hash=HASHES["dataset"],
    )


def _claim() -> ClaimDecision:
    return ClaimDecision(
        status=ClaimStatus.PRIMARY_STRICT,
        supporting_reasons=("protocol_valid",),
        prerequisite_checks={"protocol": True},
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


def _allocate(
    registry: RunRegistry,
    *,
    fold_id: str,
    seed: int,
) -> RunRecord:
    return registry.allocate_run(
        frozen_identity=FrozenRunIdentity(
            dataset_hash=HASHES["dataset"],
            protocol_hash=HASHES["protocol"],
            model_key="classic.rf.v1",
            config_hash=HASHES["config"],
            fold_id=fold_id,
            seed=seed,
            environment_lock_hash=HASHES["environment"],
            environment_compatibility_policy="exact",
        ),
        model_identity=_model(),
        input_contract=_contract(),
        provenance=_provenance(),
        resource_budget=_budget(),
        claim_decision=_claim(),
        started_utc=NOW,
    )


def _complete(
    registry: RunRegistry,
    store: FailureAtomicArtifactStore,
    *,
    fold_id: str,
    seed: int,
    prediction: float,
) -> tuple[RunRecord, Any]:
    planned = _allocate(registry, fold_id=fold_id, seed=seed)
    writer = PredictionArtifactWriter(store)
    published = writer.publish_shards(
        artifact_id=f"prediction-{planned.run_id}",
        run_id=planned.run_id,
        expected_sample_ids=(f"sample-{fold_id}",),
        shards=(
            (
                PredictionRecord(
                    key=PredictionKey(
                        sample_id=f"sample-{fold_id}",
                        fold_id=fold_id,
                        seed=seed,
                        inference_branch="primary",
                    ),
                    target_name="target",
                    y_true=0.0,
                    y_pred=prediction,
                ),
            ),
        ),
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
            created_utc=NOW + timedelta(seconds=1),
        )
    )
    registry.append_artifact(
        ArtifactRecord(
            artifact=published.reference,
            artifact_kind="prediction",
            run_id=planned.run_id,
            created_utc=NOW + timedelta(seconds=2),
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
                parameter_count=10,
                input_volume_bytes=16,
            ),
            "completed_utc": NOW + timedelta(seconds=3),
        }
    )
    registry.append_run(complete)
    return complete, published.artifact


def test_builder_loads_only_explicit_complete_members_and_records_named_variances(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts with spaces")
    values = {
        ("fold-a", 1): 1.0,
        ("fold-b", 1): 3.0,
        ("fold-a", 2): 5.0,
        ("fold-b", 2): 7.0,
    }
    completed = [
        _complete(registry, store, fold_id=fold, seed=seed, prediction=value)[0]
        for (fold, seed), value in values.items()
    ]
    # This unlisted incomplete run proves the builder does not enumerate or load
    # arbitrary registry members.
    _allocate(registry, fold_id="unlisted-fold", seed=999)

    computer = _MeanPredictionComputer()
    policy = _policy(folds=("fold-b", "fold-a"), seeds=(2, 1))
    record = AggregateBuilder(registry, store.root, computer).build(
        aggregate_id="aggregate-exact-members",
        member_run_ids=reversed([run.run_id for run in completed]),
        aggregation_policy=policy,
        created_utc=NOW + timedelta(minutes=1),
    )

    expected_members = tuple(
        sorted(run.run_id for run in completed)
    )
    assert record.member_run_ids == expected_members
    assert set(record.prediction_hashes) == set(expected_members)
    assert record.prediction_hashes == {
        run.run_id: run.prediction_artifact_ref.content_hash for run in completed
    }
    assert record.metrics == {
        "mean_prediction": 4.0,
        "seed_variance": 4.0,
        "fold_variance": 1.0,
    }
    assert record.aggregation_policy == policy
    assert record.aggregation_policy_hash == hash_object(policy)
    assert {
        reference.artifact_id: reference.content_hash
        for reference in record.artifact_refs
    } == {
        run.prediction_artifact_ref.artifact_id: run.prediction_artifact_ref.content_hash
        for run in completed
    }
    assert registry.get_aggregate(record.aggregate_id) == record

    full_call = computer.calls[0]
    assert tuple(record.key for record in full_call) == tuple(
        sorted(
            (record.key for record in full_call),
            key=lambda key: (
                key.sample_id.encode("utf-8"),
                key.fold_id.encode("utf-8"),
                key.seed,
                key.inference_branch.encode("utf-8"),
            ),
        )
    )


def test_canonical_artifact_merge_is_input_order_invariant(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    _, first = _complete(
        registry, store, fold_id="fold-z", seed=2, prediction=2.0
    )
    _, second = _complete(
        registry, store, fold_id="fold-a", seed=1, prediction=1.0
    )

    forward = merge_prediction_artifacts((first, second))
    reverse = merge_prediction_artifacts((second, first))

    assert forward == reverse
    assert tuple(record.key for record in forward) == tuple(
        sorted(
            (record.key for record in forward),
            key=lambda key: (
                key.sample_id.encode("utf-8"),
                key.fold_id.encode("utf-8"),
                key.seed,
                key.inference_branch.encode("utf-8"),
            ),
        )
    )


def test_incomplete_explicit_member_is_rejected_without_aggregate_append(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    planned = _allocate(registry, fold_id="fold-a", seed=1)
    before = registry.path.read_bytes()

    with pytest.raises(AggregateBuildError) as captured:
        AggregateBuilder(registry, store.root, _MeanPredictionComputer()).build(
            aggregate_id="aggregate-rejected",
            member_run_ids=(planned.run_id,),
            aggregation_policy=_policy(folds=("fold-a",), seeds=(1,)),
        )

    assert captured.value.error_record.error_code == "AGGREGATE_MEMBER_INCOMPLETE"
    assert captured.value.error_record.invariant_checks == {
        "completed_predictions_unchanged": True,
        "aggregate_registry_unchanged": True,
    }
    assert registry.path.read_bytes() == before
    with pytest.raises(KeyError):
        registry.get_aggregate("aggregate-rejected")
