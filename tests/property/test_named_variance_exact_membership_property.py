"""Property 25: Named variance and exact aggregate membership.

**Validates: Requirements 11.17-11.23**
"""

from __future__ import annotations

import hashlib
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Tuple

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.metrics import AggregateBuilder, PredictionArtifactWriter
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


_NOW = datetime(2027, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
_HASHES = {
    name: hashlib.sha256(f"property25-{name}".encode("utf-8")).hexdigest()
    for name in ("dataset", "protocol", "config", "environment", "source")
}


@dataclass(frozen=True)
class _AggregateCase:
    fold_order: tuple[str, ...]
    seed_order: tuple[int, ...]
    values: tuple[int, ...]


@st.composite
def _aggregate_cases(draw: st.DrawFn) -> _AggregateCase:
    fold_count = draw(st.integers(min_value=1, max_value=3))
    seed_count = draw(st.integers(min_value=1, max_value=3))
    folds = tuple(f"fold-{index}" for index in range(fold_count))
    seeds = tuple(
        draw(
            st.lists(
                st.integers(min_value=-20, max_value=20),
                min_size=seed_count,
                max_size=seed_count,
                unique=True,
            )
        )
    )
    return _AggregateCase(
        fold_order=tuple(folds[index] for index in draw(st.permutations(tuple(range(fold_count))))),
        seed_order=tuple(seeds[index] for index in draw(st.permutations(tuple(range(seed_count))))),
        values=tuple(
            draw(
                st.lists(
                    st.integers(min_value=-100, max_value=100),
                    min_size=fold_count * seed_count,
                    max_size=fold_count * seed_count,
                )
            )
        ),
    )


class _RecordingMeanComputer:
    def __init__(self) -> None:
        self.calls: list[tuple[Tuple[PredictionRecord, ...], MetricSchema]] = []

    def compute(
        self,
        records: Tuple[PredictionRecord, ...],
        metric_schema: MetricSchema,
    ) -> Mapping[str, Any]:
        self.calls.append((records, metric_schema))
        return {
            "mean_prediction": sum(float(record.y_pred) for record in records)
            / len(records)
        }


def _policy(case: _AggregateCase) -> AggregationPolicy:
    return AggregationPolicy(
        policy_key="pooled-oof-population-variance.v1",
        metric_schema=MetricSchema(
            task_kind=TaskKind.MULTI_TARGET_REGRESSION,
            numeric_tolerance=1e-12,
            target_names=("target",),
        ),
        variance_metric="mean_prediction",
        seed_set=case.seed_order,
        fold_ids=case.fold_order,
    )


def _allocate(registry: RunRegistry, *, fold_id: str, seed: int) -> RunRecord:
    return registry.allocate_run(
        frozen_identity=FrozenRunIdentity(
            dataset_hash=_HASHES["dataset"],
            protocol_hash=_HASHES["protocol"],
            model_key="classic.rf.v1",
            config_hash=_HASHES["config"],
            fold_id=fold_id,
            seed=seed,
            environment_lock_hash=_HASHES["environment"],
            environment_compatibility_policy="exact",
        ),
        model_identity=ModelIdentity(
            model_key="classic.rf.v1",
            display_name="Random Forest",
            class_fully_qualified_name="property25.RandomForest",
            implementation_kind=ImplementationKind.CLASSIC,
            parameter_count=2,
            source_revision="property25-revision",
        ),
        input_contract=InputContract(
            representation_key="global-spectrum.v1",
            modalities=("spectrum",),
            tensor_shapes={"spectrum": (2,)},
            dtypes={"spectrum": "float32"},
            channel_or_band_order={"spectrum": ("ascending",)},
            preprocessing_key="none.v1",
        ),
        provenance=Provenance(
            code_revision="property25-revision",
            source_hash=_HASHES["source"],
            environment_hash=_HASHES["environment"],
            config_hash=_HASHES["config"],
            dataset_hash=_HASHES["dataset"],
        ),
        resource_budget=ResourceBudget(
            limits=(
                ResourceLimit(
                    field="completed_epochs",
                    value=1,
                    unit="count",
                    enforcement=LimitEnforcement.HARD,
                ),
            )
        ),
        claim_decision=ClaimDecision(status=ClaimStatus.PRIMARY_STRICT),
        started_utc=_NOW,
    )


def _complete(
    registry: RunRegistry,
    store: FailureAtomicArtifactStore,
    *,
    fold_id: str,
    seed: int,
    prediction: float,
) -> RunRecord:
    planned = _allocate(registry, fold_id=fold_id, seed=seed)
    published = PredictionArtifactWriter(store).publish_shards(
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
        content=b"property 25 checkpoint",
    )
    for reference, kind in (
        (checkpoint, "checkpoint"),
        (published.reference, "prediction"),
    ):
        registry.append_artifact(
            ArtifactRecord(
                artifact=reference,
                artifact_kind=kind,
                run_id=planned.run_id,
                created_utc=_NOW + timedelta(seconds=1),
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
                parameter_count=2,
                input_volume_bytes=8,
            ),
            "completed_utc": _NOW + timedelta(seconds=2),
        }
    )
    registry.append_run(complete)
    return complete


@settings(max_examples=100, deadline=None)
@given(case=_aggregate_cases())
def test_named_variances_use_frozen_policy_and_exact_prediction_membership(
    case: _AggregateCase,
) -> None:
    """Only declared member predictions determine both named variances and lineage."""

    policy = _policy(case)
    canonical_folds = policy.fold_ids
    canonical_seeds = policy.seed_set
    predictions = {
        (fold_id, seed): value
        for (fold_id, seed), value in zip(
            (
                (fold_id, seed)
                for fold_id in canonical_folds
                for seed in canonical_seeds
            ),
            case.values,
        )
    }

    with tempfile.TemporaryDirectory(prefix="property25-") as directory:
        root = Path(directory)
        registry = RunRegistry(root / "registry.jsonl")
        store = FailureAtomicArtifactStore(root / "artifacts")
        members = [
            _complete(
                registry,
                store,
                fold_id=fold_id,
                seed=seed,
                prediction=value,
            )
            for (fold_id, seed), value in predictions.items()
        ]
        unlisted = _complete(
            registry,
            store,
            fold_id="fold-unlisted",
            seed=10_000,
            prediction=1_000_000.0,
        )

        computer = _RecordingMeanComputer()
        aggregate = AggregateBuilder(registry, store.root, computer).build(
            aggregate_id="aggregate-property-25",
            member_run_ids=tuple(reversed([run.run_id for run in members])),
            aggregation_policy=policy,
            created_utc=_NOW + timedelta(minutes=1),
        )

        expected_member_ids = tuple(sorted(run.run_id for run in members))
        expected_hashes = {
            run.run_id: run.prediction_artifact_ref.content_hash  # type: ignore[union-attr]
            for run in members
        }
        seed_means = [
            statistics.fmean(
                predictions[(fold_id, seed)] for fold_id in canonical_folds
            )
            for seed in canonical_seeds
        ]
        fold_means = [
            statistics.fmean(
                predictions[(fold_id, seed)] for seed in canonical_seeds
            )
            for fold_id in canonical_folds
        ]

        assert aggregate.member_run_ids == expected_member_ids
        assert aggregate.prediction_hashes == expected_hashes
        assert unlisted.run_id not in aggregate.member_run_ids
        assert unlisted.run_id not in aggregate.prediction_hashes
        assert {
            reference.artifact_id: reference.content_hash
            for reference in aggregate.artifact_refs
        } == {
            run.prediction_artifact_ref.artifact_id: expected_hashes[run.run_id]  # type: ignore[union-attr]
            for run in members
        }
        assert aggregate.aggregation_policy == policy
        assert aggregate.aggregation_policy_hash == hash_object(policy)
        assert aggregate.metrics["seed_variance"] == pytest.approx(
            statistics.pvariance(seed_means), abs=1e-12
        )
        assert aggregate.metrics["fold_variance"] == pytest.approx(
            statistics.pvariance(fold_means), abs=1e-12
        )
        assert set(aggregate.metrics) == {
            "mean_prediction",
            "seed_variance",
            "fold_variance",
        }

        assert len(computer.calls) == 1 + len(canonical_seeds) + len(canonical_folds)
        assert all(schema == policy.metric_schema for _, schema in computer.calls)
        observed_full_keys = {record.key for record in computer.calls[0][0]}
        expected_full_keys = {
            PredictionKey(
                sample_id=f"sample-{fold_id}",
                fold_id=fold_id,
                seed=seed,
                inference_branch="primary",
            )
            for fold_id in canonical_folds
            for seed in canonical_seeds
        }
        assert observed_full_keys == expected_full_keys
        assert all(key.fold_id != "fold-unlisted" for key in observed_full_keys)
        assert RunRegistry(registry.path).get_aggregate(aggregate.aggregate_id) == aggregate
