"""Property 27: Publication aggregate completeness.

**Validates: Requirements 12.11-12.20**
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from hypothesis import given, settings, strategies as st

from benchmark.registry import (
    AggregateRecord,
    ArtifactRecord,
    ClaimDecision,
    ErrorJournal,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.reporting import (
    AggregateCompleteness,
    AggregateEvidence,
    BenchmarkTrack,
    PromotionGate,
    PublicationSource,
    ReportGenerator,
)
from benchmark.schema import (
    AggregationPolicy,
    ArtifactReference,
    ClaimStatus,
    EvaluationBudget,
    ImplementationKind,
    LimitEnforcement,
    MetricDirection,
    MetricSchema,
    OuterFold,
    ProtocolManifest,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    RuntimeMeasurements,
    SelectionPolicy,
    TaskKind,
    hash_object,
)


_NOW = datetime(2027, 4, 5, 6, 7, 8, tzinfo=timezone.utc)


def _hash(number: int) -> str:
    return f"{number:064x}"


@dataclass(frozen=True)
class _PublicationCase:
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    retained_folds: tuple[str, ...]
    retained_seeds: tuple[int, ...]
    caveat: str


@st.composite
def _publication_cases(draw: st.DrawFn) -> _PublicationCase:
    fold_count = draw(st.integers(min_value=2, max_value=4))
    seed_count = draw(st.integers(min_value=2, max_value=4))
    folds = tuple(f"fold-{index}" for index in range(fold_count))
    seeds = tuple(range(10, 10 + seed_count))
    retained_fold_indexes = draw(
        st.sets(
            st.integers(min_value=0, max_value=fold_count - 1),
            min_size=1,
            max_size=fold_count - 1,
        )
    )
    retained_seed_indexes = draw(
        st.sets(
            st.integers(min_value=0, max_value=seed_count - 1),
            min_size=1,
            max_size=seed_count - 1,
        )
    )
    caveat_suffix = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=1,
            max_size=16,
        )
    )
    return _PublicationCase(
        folds=folds,
        seeds=seeds,
        retained_folds=tuple(folds[index] for index in sorted(retained_fold_indexes)),
        retained_seeds=tuple(seeds[index] for index in sorted(retained_seed_indexes)),
        caveat=f"legacy-caveat-{caveat_suffix}",
    )


def _protocol(
    *,
    folds: tuple[str, ...],
    seeds: tuple[int, ...],
    key: str = "leaf.lovo.nested.v1",
    dataset_hash: str = _hash(1),
) -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1.0",
        protocol_key=key,
        dataset_hash=dataset_hash,
        folds=tuple(
            OuterFold(
                fold_id=fold_id,
                outer_train_ids=(f"train-{fold_id}",),
                outer_test_ids=(f"test-{fold_id}",),
            )
            for fold_id in folds
        ),
        selection_policy=SelectionPolicy(
            policy_key="inner-selection.v1",
            objective_metric="accuracy",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule="candidate-key-byte-order",
        ),
        seed_set=seeds,
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=100),
    )


def _create_evidence(
    registry: RunRegistry,
    *,
    token: str,
    protocol: ProtocolManifest,
    observed_pairs: Iterable[tuple[str, int]],
    policy_folds: tuple[str, ...],
    policy_seeds: tuple[int, ...],
    claim: ClaimDecision,
) -> AggregateEvidence:
    model_key = f"in_repo.{token}.v1"
    model_identity = ModelIdentity(
        model_key=model_key,
        display_name=token,
        class_fully_qualified_name=f"property27.{token.title()}Model",
        implementation_kind=ImplementationKind.IN_REPO,
        parameter_count=4,
        source_revision="property27-revision",
    )
    input_contract = InputContract(
        representation_key="rgb.v1",
        modalities=("image",),
        tensor_shapes={"image": (3, 4, 4)},
        dtypes={"image": "float32"},
        channel_or_band_order={"image": ("r", "g", "b")},
        preprocessing_key="normalize.v1",
    )
    budget = ResourceBudget(
        limits=(
            ResourceLimit(
                field="completed_epochs",
                value=1,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )
    members: list[str] = []
    prediction_hashes: dict[str, str] = {}
    prediction_refs: list[ArtifactReference] = []
    for index, (fold_id, seed) in enumerate(observed_pairs, start=1):
        config_hash = _hash(1000 + index)
        environment_hash = _hash(2000 + index)
        provenance = Provenance(
            code_revision="property27-revision",
            source_hash=_hash(3000 + index),
            environment_hash=environment_hash,
            config_hash=config_hash,
            dataset_hash=protocol.dataset_hash,
        )
        planned = registry.allocate_run(
            frozen_identity=FrozenRunIdentity(
                dataset_hash=protocol.dataset_hash,
                protocol_hash=protocol.protocol_hash,
                model_key=model_key,
                config_hash=config_hash,
                fold_id=fold_id,
                seed=seed,
                environment_lock_hash=environment_hash,
                environment_compatibility_policy="exact",
            ),
            model_identity=model_identity,
            input_contract=input_contract,
            provenance=provenance,
            resource_budget=budget,
            claim_decision=claim,
            started_utc=_NOW,
        )
        checkpoint = ArtifactReference(
            artifact_id=f"checkpoint-{token}-{index}",
            relative_path=f"checkpoints/{token}-{index}.bin",
            content_hash=_hash(4000 + index),
        )
        prediction = ArtifactReference(
            artifact_id=f"prediction-{token}-{index}",
            relative_path=f"predictions/{token}-{index}.json",
            content_hash=_hash(5000 + index),
        )
        for reference, artifact_kind in (
            (checkpoint, "checkpoint"),
            (prediction, "prediction"),
        ):
            registry.append_artifact(
                ArtifactRecord(
                    artifact=reference,
                    artifact_kind=artifact_kind,
                    run_id=planned.run_id,
                    created_utc=_NOW + timedelta(seconds=1),
                )
            )
        complete = RunRecord.model_validate(
            {
                **planned.model_dump(mode="python"),
                "status": RunStatus.COMPLETE,
                "checkpoint_ref": checkpoint,
                "prediction_artifact_ref": prediction,
                "runtime_measurements": RuntimeMeasurements(
                    completed_epochs=1,
                    completed_training_steps=1,
                    wall_clock_duration_ms=1,
                    peak_device_memory_bytes=1,
                    device_identity="cpu",
                    numeric_precision="float32",
                    parameter_count=4,
                    input_volume_bytes=192,
                ),
                "completed_utc": _NOW + timedelta(seconds=2),
            }
        )
        registry.append_run(complete)
        members.append(complete.run_id)
        prediction_hashes[complete.run_id] = prediction.content_hash
        prediction_refs.append(prediction)

    policy = AggregationPolicy(
        policy_key=f"policy-{token}",
        metric_schema=MetricSchema(
            task_kind=TaskKind.BINARY_CLASSIFICATION,
            numeric_tolerance=1e-12,
        ),
        variance_metric="accuracy",
        seed_set=policy_seeds,
        fold_ids=policy_folds,
    )
    aggregate = AggregateRecord(
        aggregate_id=f"aggregate-{token}",
        member_run_ids=tuple(sorted(members)),
        prediction_hashes=prediction_hashes,
        dataset_hash=protocol.dataset_hash,
        protocol_hash=protocol.protocol_hash,
        model_key=model_key,
        representation_key="rgb.v1",
        aggregation_policy=policy,
        aggregation_policy_hash=hash_object(policy),
        metrics={"accuracy": 0.75},
        artifact_refs=tuple(prediction_refs),
        claim_decision=claim,
        provenance=registry.latest_run(members[0]).provenance,
        created_utc=_NOW + timedelta(minutes=1),
    )
    registry.append_aggregate(aggregate)
    return AggregateEvidence(
        aggregate=aggregate,
        protocol=protocol,
        track=BenchmarkTrack.HYPERLEAF,
        reference_representation_key="rgb.v1",
    )


@settings(max_examples=100, deadline=None)
@given(case=_publication_cases())
def test_publication_completeness_blocks_missing_jobs_and_preserves_references(
    case: _PublicationCase,
) -> None:
    """Incomplete coverage is blocked while supersession and caveats stay traceable."""

    with tempfile.TemporaryDirectory(prefix="property27-") as directory:
        root = Path(directory)
        registry = RunRegistry(root / "registry.jsonl")
        journal_path = root / "promotion-errors.jsonl"
        gate = PromotionGate(registry, error_journal=ErrorJournal(journal_path))

        incomplete_protocol = _protocol(folds=case.folds, seeds=case.seeds)
        observed_pairs = tuple(
            (fold_id, seed)
            for fold_id in case.retained_folds
            for seed in case.retained_seeds
        )
        incomplete = _create_evidence(
            registry,
            token="incomplete",
            protocol=incomplete_protocol,
            observed_pairs=observed_pairs,
            policy_folds=case.retained_folds,
            policy_seeds=case.retained_seeds,
            claim=ClaimDecision(status=ClaimStatus.PRIMARY_STRICT),
        )
        registry_before_promotion = registry.path.read_bytes()

        decision = gate.promote(incomplete)

        expected_missing_folds = set(case.folds) - set(case.retained_folds)
        expected_missing_seeds = set(case.seeds) - set(case.retained_seeds)
        assert decision.completeness.status is AggregateCompleteness.INCOMPLETE
        assert expected_missing_folds.issubset(decision.completeness.missing_fold_ids)
        assert expected_missing_seeds.issubset(decision.completeness.missing_seeds)
        assert decision.publication_aggregate_allowed is False
        assert registry.path.read_bytes() == registry_before_promotion

        errors = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
        ]
        fold_error = next(
            item
            for item in errors
            if item["error_code"] == "PUBLICATION_AGGREGATE_INCOMPLETE_FOLDS"
        )
        seed_error = next(
            item
            for item in errors
            if item["error_code"] == "PUBLICATION_AGGREGATE_INCOMPLETE_SEEDS"
        )
        assert expected_missing_folds.issubset(fold_error["missing_ids"])
        assert {str(seed) for seed in expected_missing_seeds}.issubset(
            seed_error["missing_ids"]
        )
        assert fold_error["invariant_checks"]["publication_aggregate_not_created"]
        assert seed_error["invariant_checks"]["publication_aggregate_not_created"]

        complete_protocol = _protocol(
            folds=("fold-complete",),
            seeds=(1,),
            key="hist.patient-nested5.v1",
            dataset_hash=_hash(9),
        )
        old_claim = ClaimDecision(
            status=ClaimStatus.DEVELOPMENT,
            supporting_reasons=("historical-result",),
            blocking_reasons=("outer-early-stopping",),
            prerequisite_checks={"outer-label-isolation": False},
        )
        new_claim = ClaimDecision(
            status=ClaimStatus.PRIMARY_STRICT,
            supporting_reasons=("nested-patient-protocol",),
            prerequisite_checks={"outer-label-isolation": True},
        )
        old = _create_evidence(
            registry,
            token="prior",
            protocol=complete_protocol,
            observed_pairs=(("fold-complete", 1),),
            policy_folds=("fold-complete",),
            policy_seeds=(1,),
            claim=old_claim,
        )
        new = _create_evidence(
            registry,
            token="superseding",
            protocol=complete_protocol,
            observed_pairs=(("fold-complete", 1),),
            policy_folds=("fold-complete",),
            policy_seeds=(1,),
            claim=new_claim,
        )
        prior_record = registry.get_aggregate(old.aggregate.aggregate_id)
        supersession = registry.supersede(
            old.aggregate.aggregate_id,
            new.aggregate.aggregate_id,
            "strict rerun replaces caveated historical conclusion",
            created_utc=_NOW + timedelta(minutes=2),
        )

        matrix = ReportGenerator(registry).build_matrix(
            (
                PublicationSource(
                    aggregate_id=old.aggregate.aggregate_id,
                    protocol_name=complete_protocol.protocol_key,
                    evidence_limitations=(
                        case.caveat,
                        f"superseded_by:{new.aggregate.aggregate_id}",
                    ),
                ),
                PublicationSource(
                    aggregate_id=new.aggregate.aggregate_id,
                    protocol_name=complete_protocol.protocol_key,
                    evidence_limitations=(
                        f"supersedes:{old.aggregate.aggregate_id}",
                    ),
                ),
            )
        )
        rows = {row.aggregate_id: row.cells for row in matrix.rows}
        old_row = rows[old.aggregate.aggregate_id]
        new_row = rows[new.aggregate.aggregate_id]

        assert registry.get_aggregate(old.aggregate.aggregate_id) == prior_record
        assert registry.get_aggregate(new.aggregate.aggregate_id) == new.aggregate
        assert registry.get_supersession(old.aggregate.aggregate_id) == supersession
        assert supersession.original_record_id == old.aggregate.aggregate_id
        assert supersession.superseding_record_id == new.aggregate.aggregate_id
        assert old_row["aggregate_id"] == supersession.original_record_id
        assert new_row["aggregate_id"] == supersession.superseding_record_id
        assert json.loads(old_row["evidence_limitation"]) == [
            case.caveat,
            f"superseded_by:{new.aggregate.aggregate_id}",
            "outer-early-stopping",
        ]
        assert json.loads(new_row["evidence_limitation"]) == [
            f"supersedes:{old.aggregate.aggregate_id}"
        ]
        assert old_row["claim_status"] == ClaimStatus.DEVELOPMENT.value
        assert new_row["claim_status"] == ClaimStatus.PRIMARY_STRICT.value
        assert json.loads(old_row["claim_explanation"]) == {
            "blocking_reasons": ["outer-early-stopping"],
            "prerequisite_checks": {"outer-label-isolation": False},
            "supporting_reasons": ["historical-result"],
        }
        assert json.loads(new_row["claim_explanation"]) == {
            "blocking_reasons": [],
            "prerequisite_checks": {"outer-label-isolation": True},
            "supporting_reasons": ["nested-patient-protocol"],
        }
