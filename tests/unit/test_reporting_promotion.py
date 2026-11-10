"""Unit coverage for benchmark tier, comparability, and promotion gates."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from benchmark.registry import (
    AggregateRecord,
    ArtifactRecord,
    ClaimDecision,
    ErrorJournal,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    NegativeEvidenceReason,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.reporting import (
    AggregateCompleteness,
    AggregateEvidence,
    BenchmarkTier,
    BenchmarkTrack,
    PromotionGate,
    PublicationSource,
    ReportGenerator,
    format_fixed_decimal4,
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


NOW = datetime(2027, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


def _hash(number: int) -> str:
    return f"{number:064x}"


def _protocol(
    *,
    key: str = "leaf.lovo.nested.v1",
    dataset_hash: str = _hash(1),
    folds: tuple[str, ...] = ("fold-a",),
    seeds: tuple[int, ...] = (1,),
) -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1.0",
        protocol_key=key,
        dataset_hash=dataset_hash,
        folds=tuple(
            OuterFold(
                fold_id=fold,
                outer_train_ids=(f"train-{fold}",),
                outer_test_ids=(f"test-{fold}",),
            )
            for fold in folds
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


def _metric_schema(tolerance: float = 1e-12) -> MetricSchema:
    return MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=tolerance,
    )


def _budget(epochs: int = 2) -> ResourceBudget:
    return ResourceBudget(
        limits=(
            ResourceLimit(
                field="completed_epochs",
                value=epochs,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )


def _create_evidence(
    registry: RunRegistry,
    *,
    token: str,
    protocol: ProtocolManifest,
    kind: ImplementationKind = ImplementationKind.IN_REPO,
    representation: str = "rgb.v1",
    reference_representation: str = "rgb.v1",
    observed_pairs: Iterable[tuple[str, int]] | None = None,
    policy_folds: tuple[str, ...] | None = None,
    policy_seeds: tuple[int, ...] | None = None,
    metric_schema: MetricSchema | None = None,
    budget: ResourceBudget | None = None,
    track: BenchmarkTrack = BenchmarkTrack.HYPERLEAF,
    claim_status: ClaimStatus = ClaimStatus.PRIMARY_STRICT,
    unified_runner_reproduced: bool = True,
    strict_protocol_complete: bool = True,
    provenance_complete: bool = True,
    prediction_artifacts_reproducible: bool = True,
    matched_input_control_required: bool | None = None,
    metrics: dict[str, Any] | None = None,
) -> AggregateEvidence:
    expected_folds = tuple(fold.fold_id for fold in protocol.folds)
    pairs = tuple(
        observed_pairs
        if observed_pairs is not None
        else (
            (fold_id, seed)
            for fold_id in expected_folds
            for seed in protocol.seed_set
        )
    )
    model_key = f"{kind.value}.{token}.v1"
    identity = ModelIdentity(
        model_key=model_key,
        display_name=(
            f"{token} proxy" if kind is ImplementationKind.PROXY else token
        ),
        class_fully_qualified_name=f"tests.{token.title()}Model",
        implementation_kind=kind,
        parameter_count=4,
        source_revision="revision-1",
    )
    contract = InputContract(
        representation_key=representation,
        modalities=("image",),
        tensor_shapes={"image": (3, 8, 8)},
        dtypes={"image": "float32"},
        channel_or_band_order={"image": ("r", "g", "b")},
        preprocessing_key="normalize.v1",
    )
    selected_budget = budget or _budget()
    claim = ClaimDecision(status=claim_status)
    members = []
    prediction_hashes = {}
    references = []
    for index, (fold_id, seed) in enumerate(pairs, start=1):
        config_hash = _hash(1000 + index)
        environment_hash = _hash(2000 + index)
        provenance = Provenance(
            code_revision="revision-1",
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
            model_identity=identity,
            input_contract=contract,
            provenance=provenance,
            resource_budget=selected_budget,
            claim_decision=claim,
            started_utc=NOW,
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
                artifact=prediction,
                artifact_kind="prediction",
                run_id=planned.run_id,
                created_utc=NOW + timedelta(seconds=2),
            )
        )
        complete = RunRecord.model_validate(
            {
                **planned.model_dump(mode="python"),
                "status": RunStatus.COMPLETE,
                "checkpoint_ref": checkpoint,
                "prediction_artifact_ref": prediction,
                "runtime_measurements": RuntimeMeasurements(
                    completed_epochs=2,
                    completed_training_steps=8,
                    wall_clock_duration_ms=10,
                    peak_device_memory_bytes=20,
                    device_identity="cpu",
                    numeric_precision="float32",
                    parameter_count=4,
                    input_volume_bytes=768,
                ),
                "completed_utc": NOW + timedelta(seconds=3),
            }
        )
        registry.append_run(complete)
        members.append(complete.run_id)
        prediction_hashes[complete.run_id] = prediction.content_hash
        references.append(prediction)

    policy = AggregationPolicy(
        policy_key=f"policy-{token}",
        metric_schema=metric_schema or _metric_schema(),
        variance_metric="accuracy",
        seed_set=policy_seeds or protocol.seed_set,
        fold_ids=policy_folds or expected_folds,
    )
    aggregate = AggregateRecord(
        aggregate_id=f"aggregate-{token}",
        member_run_ids=tuple(sorted(members)),
        prediction_hashes=prediction_hashes,
        dataset_hash=protocol.dataset_hash,
        protocol_hash=protocol.protocol_hash,
        model_key=model_key,
        representation_key=representation,
        aggregation_policy=policy,
        aggregation_policy_hash=hash_object(policy),
        metrics=metrics or {"accuracy": 0.8},
        artifact_refs=tuple(references),
        claim_decision=claim,
        provenance=registry.latest_run(members[0]).provenance,
        created_utc=NOW + timedelta(minutes=1),
    )
    registry.append_aggregate(aggregate)
    return AggregateEvidence(
        aggregate=aggregate,
        protocol=protocol,
        track=track,
        reference_representation_key=reference_representation,
        unified_runner_reproduced=unified_runner_reproduced,
        strict_protocol_complete=strict_protocol_complete,
        provenance_complete=provenance_complete,
        prediction_artifacts_reproducible=prediction_artifacts_reproducible,
        matched_input_control_required=matched_input_control_required,
    )


def test_tier_groups_are_distinct_and_in_fixed_order(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol()
    evidence = [
        _create_evidence(registry, token=kind.value, protocol=protocol, kind=kind)
        for kind in ImplementationKind
    ]

    groups = PromotionGate(registry).group_by_tier(reversed(evidence))

    assert tuple(group.tier for group in groups) == tuple(BenchmarkTier)
    assert {group.tier: len(group.aggregate_ids) for group in groups} == {
        tier: 1 for tier in BenchmarkTier
    }
    assert {
        group.tier: group.aggregate_ids for group in groups
    } == {
        BenchmarkTier.CLASSIC: ("aggregate-classic",),
        BenchmarkTier.IN_REPO: ("aggregate-in_repo",),
        BenchmarkTier.OFFICIAL: ("aggregate-official",),
        BenchmarkTier.ADAPTED: ("aggregate-adapted_official",),
        BenchmarkTier.PROXY: ("aggregate-proxy",),
    }


def test_publication_matrix_exposes_tier_identity_columns(tmp_path: Path) -> None:
    """Requirements 7.1-7.9: tier identity stays explicit in report cells."""

    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol()
    evidence = _create_evidence(
        registry,
        token="official-model",
        protocol=protocol,
        kind=ImplementationKind.OFFICIAL,
        representation="pca-rgb.v1",
    )

    matrix = ReportGenerator(registry).build_matrix(
        PublicationSource(
            aggregate_id=evidence.aggregate.aggregate_id,
            protocol_name=protocol.protocol_key,
        )
    )

    assert {
        "protocol_name",
        "protocol_hash",
        "model_key",
        "model_display_name",
        "model_class",
        "implementation_kind",
        "parameter_count",
        "source_revision",
        "representation_key",
    }.issubset(matrix.columns)
    row = matrix.rows[0].cells
    assert row["implementation_kind"] == ImplementationKind.OFFICIAL.value
    assert row["protocol_name"] == protocol.protocol_key
    assert row["representation_key"] == "pca-rgb.v1"
    assert row["parameter_count"] == "4.0000"


def test_protocol_hash_mismatch_omits_rank_and_improvement_and_separates_sections(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    candidate = _create_evidence(
        registry,
        token="candidate",
        protocol=_protocol(key="leaf.official.v1"),
    )
    control = _create_evidence(
        registry,
        token="control",
        protocol=_protocol(key="leaf.lovo.nested.v1", folds=("fold-b",)),
    )

    decision = PromotionGate(registry).compare(candidate, control)

    assert decision.comparable is False
    assert decision.rank_value_allowed is False
    assert decision.improvement_value_allowed is False
    assert "PROTOCOL_HASH_MISMATCH" in decision.non_comparability_reasons
    assert decision.candidate_protocol_section != decision.control_protocol_section


def test_candidate_control_comparison_checks_seed_folds_metric_and_budget(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol(folds=("fold-a", "fold-b"), seeds=(1, 2))
    candidate = _create_evidence(
        registry,
        token="candidate",
        protocol=protocol,
    )
    control = _create_evidence(
        registry,
        token="control",
        protocol=protocol,
        policy_folds=("fold-a",),
        policy_seeds=(1,),
        metric_schema=_metric_schema(1e-6),
        budget=_budget(3),
    )

    decision = PromotionGate(registry).compare(candidate, control)

    assert decision.comparable is False
    assert {
        "SEED_SET_MISMATCH",
        "FOLD_MANIFEST_MISMATCH",
        "METRIC_CONFIGURATION_MISMATCH",
        "RESOURCE_BUDGET_MISMATCH",
        "CONTROL_AGGREGATE_INCOMPLETE",
    }.issubset(decision.non_comparability_reasons)


def test_distinct_representation_requires_complete_comparable_matched_control(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol()
    candidate = _create_evidence(
        registry,
        token="candidate",
        protocol=protocol,
        representation="pca-rgb.v1",
        reference_representation="visible-rgb.v1",
    )
    wrong_representation = _create_evidence(
        registry,
        token="wrong-control",
        protocol=protocol,
        representation="visible-rgb.v1",
    )
    matched = _create_evidence(
        registry,
        token="matched-control",
        protocol=protocol,
        kind=ImplementationKind.CLASSIC,
        representation="pca-rgb.v1",
    )
    gate = PromotionGate(registry)

    blocked = gate.promote(candidate, controls=(wrong_representation,))
    promoted = gate.promote(candidate, controls=(wrong_representation, matched))

    assert blocked.publication_aggregate_allowed is True
    assert blocked.effective_claim_status is ClaimStatus.STRICT_CANDIDATE
    assert blocked.matched_control_aggregate_id is None
    assert blocked.blocking_reasons == ("MATCHED_INPUT_CONTROL_MISSING",)
    assert promoted.effective_claim_status is ClaimStatus.PRIMARY_STRICT
    assert promoted.matched_control_aggregate_id == matched.aggregate.aggregate_id
    assert promoted.blocking_reasons == ()


def test_incomplete_folds_and_seeds_block_publication_without_registry_mutation(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol(folds=("fold-a", "fold-b"), seeds=(1, 2))
    candidate = _create_evidence(
        registry,
        token="incomplete",
        protocol=protocol,
        observed_pairs=(("fold-a", 1),),
        policy_folds=("fold-a",),
        policy_seeds=(1,),
    )
    before = registry.path.read_bytes()
    journal_path = tmp_path / "promotion-errors.jsonl"
    gate = PromotionGate(
        registry,
        error_journal=ErrorJournal(journal_path),
    )

    decision = gate.promote(candidate)

    assert decision.completeness.status is AggregateCompleteness.INCOMPLETE
    assert decision.completeness.missing_fold_ids == ("fold-a", "fold-b")
    assert decision.completeness.missing_seeds == (1, 2)
    assert decision.publication_aggregate_allowed is False
    assert decision.effective_claim_status is ClaimStatus.STRICT_CANDIDATE
    assert registry.path.read_bytes() == before
    errors = [json.loads(line) for line in journal_path.read_text("utf-8").splitlines()]
    assert {error["error_code"] for error in errors} == {
        "PUBLICATION_AGGREGATE_INCOMPLETE_FOLDS",
        "PUBLICATION_AGGREGATE_INCOMPLETE_SEEDS",
    }
    assert all(
        error["invariant_checks"]["publication_aggregate_not_created"]
        for error in errors
    )


def test_remote_sanity_track_is_excluded_from_project_rankings(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol(key="remote.indian-pines.sanity.v1")
    remote = _create_evidence(
        registry,
        token="remote",
        protocol=protocol,
        track=BenchmarkTrack.REMOTE_SENSING_SANITY,
    )
    control = _create_evidence(
        registry,
        token="remote-control",
        protocol=protocol,
        track=BenchmarkTrack.REMOTE_SENSING_SANITY,
    )
    gate = PromotionGate(registry)

    decision = gate.compare(remote, control)

    assert gate.ranking_allowed(BenchmarkTrack.HYPERLEAF) is True
    assert gate.ranking_allowed(BenchmarkTrack.HISTOLOGY) is True
    assert gate.ranking_allowed(BenchmarkTrack.REMOTE_SENSING_SANITY) is False
    assert decision.rank_value_allowed is False
    assert "REMOTE_SANITY_RANKING_EXCLUDED" in decision.non_comparability_reasons


def test_fixed_decimal4_uses_half_even_and_preserves_trailing_zeros() -> None:
    assert format_fixed_decimal4(Decimal("1.23445")) == "1.2344"
    assert format_fixed_decimal4(Decimal("1.23455")) == "1.2346"
    assert format_fixed_decimal4(0.5) == "0.5000"
    assert format_fixed_decimal4(-0.00001) == "0.0000"


def test_publication_matrix_reads_registry_and_renders_identical_cells(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol(key="hist.patient-nested5.v1")
    evidence = _create_evidence(
        registry,
        token="matrix",
        protocol=protocol,
        metrics={
            "accuracy": 0.81235,
            "confidence_interval": {"lower": 0.75, "upper": 0.875},
            "fold_counts": [2, 3],
            "undefined_reason": None,
        },
    )
    source = PublicationSource(
        aggregate_id=evidence.aggregate.aggregate_id,
        protocol_name=protocol.protocol_key,
        uncertainty_metric_paths=(
            "confidence_interval.lower",
            "confidence_interval.upper",
        ),
        evidence_limitations=("patient-level estimate",),
    )

    matrix = ReportGenerator(registry).build_matrix(source)

    row = matrix.rows[0].cells
    assert row["metric.accuracy"] == "0.8124"
    assert row["metric.confidence_interval.lower"] == "0.7500"
    assert row["metric.confidence_interval.upper"] == "0.8750"
    assert row["metric.fold_counts.0"] == "2.0000"
    assert row["metric.undefined_reason"] == "null"
    assert row["parameter_count"] == "4.0000"
    assert row["protocol_name"] == "hist.patient-nested5.v1"
    assert row["protocol_hash"] == protocol.protocol_hash
    assert row["model_key"] == evidence.aggregate.model_key
    assert row["claim_status"] == "primary_strict"
    assert json.loads(row["uncertainty_estimate"]) == {
        "confidence_interval.lower": "0.7500",
        "confidence_interval.upper": "0.8750",
    }
    assert json.loads(row["evidence_limitation"]) == ["patient-level estimate"]

    json_payload = json.loads(ReportGenerator.render_json(matrix))
    csv_rows = list(
        csv.DictReader(io.StringIO(ReportGenerator.render_csv(matrix).decode("utf-8")))
    )
    markdown_lines = ReportGenerator.render_markdown(matrix).decode("utf-8").splitlines()
    markdown_header = [cell.strip() for cell in markdown_lines[0][1:-1].split("|")]
    markdown_values = [cell.strip() for cell in markdown_lines[2][1:-1].split("|")]
    markdown_row = dict(zip(markdown_header, markdown_values))

    assert json_payload["columns"] == list(matrix.columns)
    assert json_payload["rows"] == [row]
    assert csv_rows == [row]
    for field in (
        "metric.accuracy",
        "metric.confidence_interval.lower",
        "metric.confidence_interval.upper",
        "model_key",
        "model_display_name",
        "protocol_name",
        "protocol_hash",
        "claim_status",
    ):
        assert markdown_row[field] == row[field]


def test_publication_matrix_has_canonical_row_and_metric_column_order(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol()
    second = _create_evidence(
        registry,
        token="zeta",
        protocol=protocol,
        metrics={"z_metric": 0.25},
    )
    first = _create_evidence(
        registry,
        token="alpha",
        protocol=protocol,
        metrics={"a_metric": 0.5},
    )
    generator = ReportGenerator(registry)

    matrix = generator.build_matrix(
        (
            PublicationSource(
                aggregate_id=second.aggregate.aggregate_id,
                protocol_name=protocol.protocol_key,
            ),
            PublicationSource(
                aggregate_id=first.aggregate.aggregate_id,
                protocol_name=protocol.protocol_key,
            ),
        )
    )

    assert tuple(row.aggregate_id for row in matrix.rows) == (
        "aggregate-alpha",
        "aggregate-zeta",
    )
    assert matrix.columns[-2:] == ("metric.a_metric", "metric.z_metric")
    assert matrix.rows[0].cells["metric.z_metric"] == ""
    assert matrix.rows[1].cells["metric.a_metric"] == ""


def test_superseded_and_superseding_aggregates_remain_traceable_in_report_source(
    tmp_path: Path,
) -> None:
    """Requirements 12.17-12.20: corrections retain both evidence records."""

    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol()
    original = _create_evidence(
        registry,
        token="original",
        protocol=protocol,
        claim_status=ClaimStatus.DEVELOPMENT,
    )
    correction = _create_evidence(
        registry,
        token="correction",
        protocol=protocol,
        claim_status=ClaimStatus.STRICT_CANDIDATE,
    )
    original_snapshot = registry.get_aggregate(original.aggregate.aggregate_id)

    link = registry.supersede(
        original.aggregate.aggregate_id,
        correction.aggregate.aggregate_id,
        "Corrected claim after evidence review",
        created_utc=NOW + timedelta(minutes=2),
    )
    matrix = ReportGenerator(registry).build_matrix(
        (
            PublicationSource(
                aggregate_id=original.aggregate.aggregate_id,
                protocol_name=protocol.protocol_key,
                evidence_limitations=("superseded",),
            ),
            PublicationSource(
                aggregate_id=correction.aggregate.aggregate_id,
                protocol_name=protocol.protocol_key,
                evidence_limitations=(
                    f"supersedes:{original.aggregate.aggregate_id}",
                ),
            ),
        )
    )

    rows = {row.aggregate_id: row.cells for row in matrix.rows}
    assert link.original_record_id == original.aggregate.aggregate_id
    assert link.superseding_record_id == correction.aggregate.aggregate_id
    assert registry.get_supersession(original.aggregate.aggregate_id) == link
    assert registry.get_aggregate(original.aggregate.aggregate_id) == original_snapshot
    assert set(rows) == {
        original.aggregate.aggregate_id,
        correction.aggregate.aggregate_id,
    }
    assert rows[original.aggregate.aggregate_id]["claim_status"] == "development"
    assert rows[correction.aggregate.aggregate_id]["claim_status"] == "strict_candidate"
    assert json.loads(rows[original.aggregate.aggregate_id]["evidence_limitation"]) == [
        "superseded"
    ]
    assert json.loads(rows[correction.aggregate.aggregate_id]["evidence_limitation"]) == [
        f"supersedes:{original.aggregate.aggregate_id}"
    ]


def test_comparison_failure_emits_field_errors_without_mutating_registry(
    tmp_path: Path,
) -> None:
    """Requirements 10.12-10.17, 10.35-10.37."""

    registry = RunRegistry(tmp_path / "registry.jsonl")
    candidate_protocol = _protocol(folds=("fold-a",), seeds=(1,))
    control_protocol = _protocol(
        dataset_hash=_hash(99), folds=("fold-b",), seeds=(2,)
    )
    candidate = _create_evidence(
        registry,
        token="comparison-candidate",
        protocol=candidate_protocol,
        metric_schema=_metric_schema(1e-12),
        budget=_budget(2),
    )
    control = _create_evidence(
        registry,
        token="comparison-control",
        protocol=control_protocol,
        metric_schema=_metric_schema(1e-6),
        budget=_budget(3),
    )
    before = registry.path.read_bytes()
    candidate_before = registry.get_aggregate(candidate.aggregate.aggregate_id)
    control_before = registry.get_aggregate(control.aggregate.aggregate_id)
    journal = ErrorJournal(tmp_path / "comparison-errors.jsonl")

    decision = PromotionGate(registry, error_journal=journal).compare(
        candidate, control
    )

    assert decision.comparable is False
    assert registry.path.read_bytes() == before
    assert registry.get_aggregate(candidate.aggregate.aggregate_id) == candidate_before
    assert registry.get_aggregate(control.aggregate.aggregate_id) == control_before
    errors = [json.loads(line) for line in journal.path.read_text("utf-8").splitlines()]
    assert {error["failing_field"] for error in errors} == {
        "dataset_hash",
        "protocol_hash",
        "seed_set",
        "fold_manifests",
        "metric_configuration",
        "resource_budget",
    }
    assert all(error["affected_identifiers"]["candidate_run_ids"] for error in errors)
    assert all(error["affected_identifiers"]["control_run_ids"] for error in errors)
    assert all(error["invariant_checks"]["claim_status_unchanged"] for error in errors)


def test_primary_promotion_requires_every_strict_eligibility_prerequisite(
    tmp_path: Path,
) -> None:
    """Requirements 10.18-10.21 and 10.26-10.31."""

    registry = RunRegistry(tmp_path / "registry.jsonl")
    candidate = _create_evidence(
        registry,
        token="blocked-primary",
        protocol=_protocol(),
        strict_protocol_complete=False,
        provenance_complete=False,
        unified_runner_reproduced=False,
        prediction_artifacts_reproducible=False,
        matched_input_control_required=True,
    )
    before = registry.path.read_bytes()

    decision = PromotionGate(registry).promote(candidate)

    assert decision.effective_claim_status is ClaimStatus.STRICT_CANDIDATE
    assert decision.publication_aggregate_allowed is True
    assert decision.blocking_reasons == (
        "STRICT_PROTOCOL_INCOMPLETE",
        "PROVENANCE_INCOMPLETE",
        "UNIFIED_RUNNER_REPRODUCTION_MISSING",
        "MATCHED_INPUT_CONTROL_MISSING",
        "PREDICTION_ARTIFACTS_NOT_REPRODUCIBLE",
    )
    assert decision.prerequisite_checks == {
        "strict_protocol": False,
        "provenance": False,
        "complete_declared_folds": True,
        "complete_declared_seed_set": True,
        "unified_runner_reproduced": False,
        "matched_input_control": False,
        "prediction_artifacts_reproducible": False,
    }
    assert registry.path.read_bytes() == before


def test_negative_evidence_appends_exact_configuration_and_preserves_prior_records(
    tmp_path: Path,
) -> None:
    """Requirements 10.22-10.25."""

    registry = RunRegistry(tmp_path / "registry.jsonl")
    protocol = _protocol()
    candidate = _create_evidence(
        registry,
        token="negative-candidate",
        protocol=protocol,
        metrics={"accuracy": 0.71},
    )
    control = _create_evidence(
        registry,
        token="negative-control",
        protocol=protocol,
        metrics={"accuracy": 0.76},
    )
    candidate_before = registry.get_aggregate(candidate.aggregate.aggregate_id)
    control_before = registry.get_aggregate(control.aggregate.aggregate_id)
    before = registry.path.read_bytes()
    gate = PromotionGate(registry)
    tested = {
        "candidate_config_hash": _hash(501),
        "control_config_hash": _hash(502),
        "seed_set": [1],
    }

    first = gate.record_negative_evidence(
        candidate,
        control=control,
        reason=NegativeEvidenceReason.UNDERPERFORMANCE,
        tested_configuration=tested,
        details={"metric": "accuracy", "difference": -0.05},
        evidence_id="negative-evidence-1",
        created_utc=NOW + timedelta(minutes=5),
    )
    first_bytes = registry.path.read_bytes()
    second = gate.record_negative_evidence(
        candidate,
        reason=NegativeEvidenceReason.SEED_INSTABILITY,
        tested_configuration={**tested, "seed_set": [1, 2, 3]},
        details={"seed_variance": 0.12},
        evidence_id="negative-evidence-2",
        created_utc=NOW + timedelta(minutes=6),
    )

    assert first.tested_configuration == tested
    assert first.candidate_metrics == {"accuracy": 0.71}
    assert first.control_metrics == {"accuracy": 0.76}
    assert second.reason is NegativeEvidenceReason.SEED_INSTABILITY
    assert registry.path.read_bytes().startswith(before)
    assert registry.path.read_bytes().startswith(first_bytes)
    assert registry.negative_evidence(
        candidate_aggregate_id=candidate.aggregate.aggregate_id
    ) == (first, second)
    assert registry.get_aggregate(candidate.aggregate.aggregate_id) == candidate_before
    assert registry.get_aggregate(control.aggregate.aggregate_id) == control_before
