"""Benchmark tiering, comparability, and publication-promotion gates."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from pydantic import Field

from benchmark.registry.error_journal import ErrorJournal
from benchmark.registry.records import (
    AggregateRecord,
    NegativeEvidenceReason,
    NegativeEvidenceRecord,
    RunRecord,
)
from benchmark.registry.run_registry import RunRegistry
from benchmark.schema.base import ClaimStatus, ImplementationKind
from benchmark.schema.environment import FrozenSchemaModel
from benchmark.schema.protocol import ProtocolManifest
from benchmark.schema.records import ErrorRecord


class BenchmarkTier(str, Enum):
    """Stable report groups in their required publication order."""

    CLASSIC = "Classic"
    IN_REPO = "In_Repo"
    OFFICIAL = "Official"
    ADAPTED = "Adapted"
    PROXY = "Proxy"


class BenchmarkTrack(str, Enum):
    """Independent benchmark tracks relevant to ranking eligibility."""

    HYPERLEAF = "HyperLeaf2024"
    HISTOLOGY = "HistologyHSI-GB"
    REMOTE_SENSING_SANITY = "Remote_Sensing_Sanity_Track"


class AggregateCompleteness(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


_KIND_TO_TIER: Mapping[ImplementationKind, BenchmarkTier] = {
    ImplementationKind.CLASSIC: BenchmarkTier.CLASSIC,
    ImplementationKind.IN_REPO: BenchmarkTier.IN_REPO,
    ImplementationKind.OFFICIAL: BenchmarkTier.OFFICIAL,
    ImplementationKind.ADAPTED_OFFICIAL: BenchmarkTier.ADAPTED,
    ImplementationKind.PROXY: BenchmarkTier.PROXY,
}
_CLAIM_PRECEDENCE = tuple(ClaimStatus)


class AggregateEvidence(FrozenSchemaModel):
    """Aggregate plus the protocol and reporting context used by the gate."""

    aggregate: AggregateRecord
    protocol: ProtocolManifest
    track: BenchmarkTrack
    reference_representation_key: str = Field(min_length=1)
    unified_runner_reproduced: bool = True
    strict_protocol_complete: bool = True
    provenance_complete: bool = True
    prediction_artifacts_reproducible: bool = True
    matched_input_control_required: Optional[bool] = None


class TierGroup(FrozenSchemaModel):
    """One explicit implementation-category section for downstream renderers."""

    tier: BenchmarkTier
    aggregate_ids: Tuple[str, ...] = ()


class CompletenessDecision(FrozenSchemaModel):
    """Exact declared-versus-observed fold/seed coverage."""

    status: AggregateCompleteness
    missing_fold_ids: Tuple[str, ...] = ()
    missing_seeds: Tuple[int, ...] = ()
    missing_fold_seed_pairs: Tuple[str, ...] = ()
    unexpected_fold_seed_pairs: Tuple[str, ...] = ()
    duplicate_fold_seed_pairs: Tuple[str, ...] = ()


class ComparisonDecision(FrozenSchemaModel):
    """Whether rank and improvement values may be computed for a pair."""

    comparable: bool
    rank_value_allowed: bool
    improvement_value_allowed: bool
    non_comparability_reasons: Tuple[str, ...] = ()
    candidate_protocol_section: str
    control_protocol_section: str


class PromotionDecision(FrozenSchemaModel):
    """Non-mutating result of checking publication and primary promotion."""

    aggregate_id: str
    tier: BenchmarkTier
    completeness: CompletenessDecision
    publication_aggregate_allowed: bool
    effective_claim_status: ClaimStatus
    matched_control_aggregate_id: Optional[str] = None
    blocking_reasons: Tuple[str, ...] = ()
    prerequisite_checks: Dict[str, bool]


class PromotionGate:
    """Validate aggregate comparability without mutating scientific records.

    The gate reads the supplied aggregate's exact member runs from the
    append-only registry. It emits only Error Journal records for missing
    publication coverage; it never appends or rewrites a run or aggregate.
    """

    def __init__(
        self,
        registry: RunRegistry,
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        self.registry = registry
        self.error_journal = error_journal or ErrorJournal(
            registry.path.parent / ".errors" / "promotion-errors.jsonl"
        )

    @staticmethod
    def tier_for_kind(kind: ImplementationKind) -> BenchmarkTier:
        """Map each implementation kind to exactly one report tier."""

        if not isinstance(kind, ImplementationKind):
            try:
                kind = ImplementationKind(kind)
            except (TypeError, ValueError) as exc:
                raise ValueError("implementation kind is invalid") from exc
        return _KIND_TO_TIER[kind]

    def group_by_tier(
        self, evidence: Iterable[AggregateEvidence]
    ) -> Tuple[TierGroup, ...]:
        """Return all five distinct tier sections in fixed publication order."""

        grouped: Dict[BenchmarkTier, list[str]] = {tier: [] for tier in BenchmarkTier}
        for item in evidence:
            tier = self._tier_for_evidence(item)
            grouped[tier].append(item.aggregate.aggregate_id)
        return tuple(
            TierGroup(
                tier=tier,
                aggregate_ids=tuple(
                    sorted(grouped[tier], key=lambda value: value.encode("utf-8"))
                ),
            )
            for tier in BenchmarkTier
        )

    def assess_completeness(
        self,
        evidence: AggregateEvidence,
        *,
        emit_errors: bool = True,
    ) -> CompletenessDecision:
        """Compare member jobs with the protocol fold/seed Cartesian product."""

        runs = self._registered_member_runs(evidence)
        expected_folds = tuple(fold.fold_id for fold in evidence.protocol.folds)
        expected_seeds = evidence.protocol.seed_set
        expected = {
            (fold_id, seed) for fold_id in expected_folds for seed in expected_seeds
        }
        observed_pairs = tuple(
            (run.frozen_identity.fold_id, run.frozen_identity.seed) for run in runs
        )
        observed = set(observed_pairs)
        duplicate = {
            pair for pair in observed if observed_pairs.count(pair) > 1
        }
        missing = expected - observed
        unexpected = observed - expected

        policy = evidence.aggregate.aggregation_policy
        policy_matches = (
            policy is not None
            and policy.fold_ids == expected_folds
            and policy.seed_set == expected_seeds
        )
        identity_matches = (
            evidence.aggregate.protocol_hash == evidence.protocol.protocol_hash
            and evidence.aggregate.dataset_hash == evidence.protocol.dataset_hash
        )
        complete = (
            not missing
            and not unexpected
            and not duplicate
            and policy_matches
            and identity_matches
        )

        implicated_missing_folds = {fold_id for fold_id, _ in missing}
        implicated_missing_seeds = {seed for _, seed in missing}
        if not policy_matches:
            policy_folds = set(policy.fold_ids) if policy is not None else set()
            policy_seeds = set(policy.seed_set) if policy is not None else set()
            implicated_missing_folds.update(set(expected_folds) - policy_folds)
            implicated_missing_seeds.update(set(expected_seeds) - policy_seeds)

        missing_folds = self._sorted_strings(implicated_missing_folds)
        missing_seeds = tuple(sorted(implicated_missing_seeds))
        decision = CompletenessDecision(
            status=(
                AggregateCompleteness.COMPLETE
                if complete
                else AggregateCompleteness.INCOMPLETE
            ),
            missing_fold_ids=missing_folds,
            missing_seeds=missing_seeds,
            missing_fold_seed_pairs=self._pair_text(missing),
            unexpected_fold_seed_pairs=self._pair_text(unexpected),
            duplicate_fold_seed_pairs=self._pair_text(duplicate),
        )
        if emit_errors and decision.status is AggregateCompleteness.INCOMPLETE:
            self._record_incomplete(evidence.aggregate.aggregate_id, decision)
        return decision

    def compare(
        self,
        candidate: AggregateEvidence,
        control: AggregateEvidence,
        *,
        require_matched_representation: bool = False,
        emit_errors: bool = True,
    ) -> ComparisonDecision:
        """Check every field required for a candidate/control comparison."""

        candidate_runs = self._registered_member_runs(candidate)
        control_runs = self._registered_member_runs(control)
        reasons = []

        comparisons = (
            (
                "dataset_hash",
                candidate.aggregate.dataset_hash,
                control.aggregate.dataset_hash,
                "DATASET_HASH_MISMATCH",
            ),
            (
                "protocol_hash",
                candidate.aggregate.protocol_hash,
                control.aggregate.protocol_hash,
                "PROTOCOL_HASH_MISMATCH",
            ),
            (
                "seed_set",
                self._seed_configuration(candidate),
                self._seed_configuration(control),
                "SEED_SET_MISMATCH",
            ),
            (
                "fold_manifests",
                self._fold_configuration(candidate),
                self._fold_configuration(control),
                "FOLD_MANIFEST_MISMATCH",
            ),
            (
                "metric_configuration",
                self._metric_configuration(candidate),
                self._metric_configuration(control),
                "METRIC_CONFIGURATION_MISMATCH",
            ),
            (
                "resource_budget",
                self._resource_budget_configuration(candidate_runs),
                self._resource_budget_configuration(control_runs),
                "RESOURCE_BUDGET_MISMATCH",
            ),
        )
        mismatches = tuple(
            (field, expected, observed, reason)
            for field, expected, observed, reason in comparisons
            if expected != observed
        )
        reasons.extend(reason for _, _, _, reason in mismatches)
        if emit_errors:
            for field, expected, observed, reason in mismatches:
                self._record_comparison_failure(
                    candidate,
                    control,
                    candidate_runs,
                    control_runs,
                    field=field,
                    expected=expected,
                    observed=observed,
                    reason=reason,
                )
        if (
            require_matched_representation
            and candidate.aggregate.representation_key
            != control.aggregate.representation_key
        ):
            reasons.append("REPRESENTATION_KEY_MISMATCH")
            if emit_errors:
                self._record_comparison_failure(
                    candidate,
                    control,
                    candidate_runs,
                    control_runs,
                    field="representation_key",
                    expected=candidate.aggregate.representation_key,
                    observed=control.aggregate.representation_key,
                    reason="REPRESENTATION_KEY_MISMATCH",
                )

        candidate_complete = self.assess_completeness(
            candidate, emit_errors=False
        ).status is AggregateCompleteness.COMPLETE
        control_complete = self.assess_completeness(
            control, emit_errors=False
        ).status is AggregateCompleteness.COMPLETE
        if not candidate_complete:
            reasons.append("CANDIDATE_AGGREGATE_INCOMPLETE")
        if not control_complete:
            reasons.append("CONTROL_AGGREGATE_INCOMPLETE")

        remote = (
            candidate.track is BenchmarkTrack.REMOTE_SENSING_SANITY
            or control.track is BenchmarkTrack.REMOTE_SENSING_SANITY
        )
        if remote:
            reasons.append("REMOTE_SANITY_RANKING_EXCLUDED")

        reasons = list(dict.fromkeys(reasons))
        comparable = not reasons
        return ComparisonDecision(
            comparable=comparable,
            rank_value_allowed=comparable and not remote,
            improvement_value_allowed=comparable,
            non_comparability_reasons=tuple(reasons),
            candidate_protocol_section=self.protocol_section(candidate),
            control_protocol_section=self.protocol_section(control),
        )

    def promote(
        self,
        candidate: AggregateEvidence,
        *,
        controls: Sequence[AggregateEvidence] = (),
    ) -> PromotionDecision:
        """Apply completeness, unified-runner, and matched-control ceilings."""

        completeness = self.assess_completeness(candidate)
        distinct_representation = (
            candidate.aggregate.representation_key
            != candidate.reference_representation_key
        )
        control_required = (
            distinct_representation
            if candidate.matched_input_control_required is None
            else candidate.matched_input_control_required
        )
        matched_control_id: Optional[str] = None
        if control_required:
            for control in controls:
                comparison = self.compare(
                    candidate,
                    control,
                    require_matched_representation=True,
                )
                if comparison.comparable:
                    matched_control_id = control.aggregate.aggregate_id
                    break

        checks = {
            "strict_protocol": candidate.strict_protocol_complete,
            "provenance": candidate.provenance_complete,
            "complete_declared_folds": (
                completeness.status is AggregateCompleteness.COMPLETE
                and not completeness.missing_fold_ids
                and not completeness.duplicate_fold_seed_pairs
            ),
            "complete_declared_seed_set": (
                completeness.status is AggregateCompleteness.COMPLETE
                and not completeness.missing_seeds
                and not completeness.duplicate_fold_seed_pairs
            ),
            "unified_runner_reproduced": candidate.unified_runner_reproduced,
            "matched_input_control": (
                not control_required or matched_control_id is not None
            ),
            "prediction_artifacts_reproducible": (
                candidate.prediction_artifacts_reproducible
                and self._prediction_artifacts_reproducible(candidate)
            ),
        }
        blockers = tuple(
            reason
            for field, reason in (
                ("strict_protocol", "STRICT_PROTOCOL_INCOMPLETE"),
                ("provenance", "PROVENANCE_INCOMPLETE"),
                ("complete_declared_folds", "INCOMPLETE_FOLDS"),
                ("complete_declared_seed_set", "INCOMPLETE_SEED_SET"),
                ("unified_runner_reproduced", "UNIFIED_RUNNER_REPRODUCTION_MISSING"),
                ("matched_input_control", "MATCHED_INPUT_CONTROL_MISSING"),
                (
                    "prediction_artifacts_reproducible",
                    "PREDICTION_ARTIFACTS_NOT_REPRODUCIBLE",
                ),
            )
            if not checks[field]
        )
        effective = candidate.aggregate.claim_decision.status
        if blockers:
            effective = self._lower_to(effective, ClaimStatus.STRICT_CANDIDATE)

        return PromotionDecision(
            aggregate_id=candidate.aggregate.aggregate_id,
            tier=self._tier_for_evidence(candidate),
            completeness=completeness,
            publication_aggregate_allowed=(
                completeness.status is AggregateCompleteness.COMPLETE
            ),
            effective_claim_status=effective,
            matched_control_aggregate_id=matched_control_id,
            blocking_reasons=blockers,
            prerequisite_checks=checks,
        )

    def record_negative_evidence(
        self,
        candidate: AggregateEvidence,
        *,
        reason: NegativeEvidenceReason,
        tested_configuration: Mapping[str, Any],
        control: Optional[AggregateEvidence] = None,
        details: Optional[Mapping[str, Any]] = None,
        evidence_id: Optional[str] = None,
        created_utc: Optional[datetime] = None,
    ) -> NegativeEvidenceRecord:
        """Append a negative result while preserving all prior registry entries."""

        if not isinstance(reason, NegativeEvidenceReason):
            try:
                reason = NegativeEvidenceReason(reason)
            except (TypeError, ValueError) as exc:
                raise ValueError("negative evidence reason is invalid") from exc
        candidate_runs = self._registered_member_runs(candidate)
        control_runs: Tuple[RunRecord, ...] = ()
        if control is not None:
            control_runs = self._registered_member_runs(control)
        record = NegativeEvidenceRecord(
            evidence_id=evidence_id or str(uuid4()),
            reason=reason,
            candidate_aggregate_id=candidate.aggregate.aggregate_id,
            candidate_run_ids=tuple(
                sorted(
                    (run.run_id for run in candidate_runs),
                    key=lambda value: value.encode("utf-8"),
                )
            ),
            control_aggregate_id=(
                control.aggregate.aggregate_id if control is not None else None
            ),
            control_run_ids=tuple(
                sorted(
                    (run.run_id for run in control_runs),
                    key=lambda value: value.encode("utf-8"),
                )
            ),
            tested_configuration=dict(tested_configuration),
            candidate_metrics=candidate.aggregate.metrics,
            control_metrics=(control.aggregate.metrics if control is not None else None),
            details=dict(details or {}),
            created_utc=created_utc or datetime.now(timezone.utc),
        )
        self.registry.append_negative_evidence(record)
        return record

    @staticmethod
    def protocol_section(evidence: AggregateEvidence) -> str:
        """Return a collision-resistant section key displayed by reports."""

        return f"{evidence.protocol.protocol_key}@{evidence.protocol.protocol_hash}"

    @staticmethod
    def ranking_allowed(track: BenchmarkTrack) -> bool:
        """Exclude remote sanity evidence from project performance rankings."""

        return track is not BenchmarkTrack.REMOTE_SENSING_SANITY

    def _registered_member_runs(
        self, evidence: AggregateEvidence
    ) -> Tuple[RunRecord, ...]:
        registered = self.registry.get_aggregate(evidence.aggregate.aggregate_id)
        if registered != evidence.aggregate:
            raise ValueError("aggregate evidence differs from the registered record")
        runs = tuple(
            self.registry.latest_run(run_id)
            for run_id in evidence.aggregate.member_run_ids
        )
        if any(run.frozen_identity.protocol_hash != evidence.aggregate.protocol_hash for run in runs):
            raise ValueError("aggregate member protocol identity mismatch")
        if any(run.frozen_identity.dataset_hash != evidence.aggregate.dataset_hash for run in runs):
            raise ValueError("aggregate member dataset identity mismatch")
        return runs

    def _tier_for_evidence(self, evidence: AggregateEvidence) -> BenchmarkTier:
        runs = self._registered_member_runs(evidence)
        kinds = {run.model_identity.implementation_kind for run in runs}
        if len(kinds) != 1:
            raise ValueError("aggregate members do not share one implementation kind")
        return self.tier_for_kind(next(iter(kinds)))

    @staticmethod
    def _seed_configuration(evidence: AggregateEvidence) -> Dict[str, object]:
        policy = evidence.aggregate.aggregation_policy
        return {
            "protocol_seed_set": list(evidence.protocol.seed_set),
            "aggregate_seed_set": (
                list(policy.seed_set) if policy is not None else None
            ),
        }

    @staticmethod
    def _fold_configuration(evidence: AggregateEvidence) -> Dict[str, object]:
        policy = evidence.aggregate.aggregation_policy
        return {
            "protocol_folds": [
                fold.model_dump(mode="json") for fold in evidence.protocol.folds
            ],
            "aggregate_fold_ids": (
                list(policy.fold_ids) if policy is not None else None
            ),
        }

    @staticmethod
    def _policy_seed_set(evidence: AggregateEvidence) -> Optional[Tuple[int, ...]]:
        policy = evidence.aggregate.aggregation_policy
        return policy.seed_set if policy is not None else None

    @staticmethod
    def _policy_fold_ids(evidence: AggregateEvidence) -> Optional[Tuple[str, ...]]:
        policy = evidence.aggregate.aggregation_policy
        return policy.fold_ids if policy is not None else None

    @staticmethod
    def _metric_configuration(evidence: AggregateEvidence) -> object:
        policy = evidence.aggregate.aggregation_policy
        return (
            policy.metric_schema.model_dump(mode="json")
            if policy is not None
            else None
        )

    @staticmethod
    def _resource_budget_configuration(
        runs: Sequence[RunRecord],
    ) -> Tuple[Dict[str, object], ...]:
        return tuple(
            {
                "fold_id": run.frozen_identity.fold_id,
                "seed": run.frozen_identity.seed,
                "budget": run.resource_budget.model_dump(mode="json"),
            }
            for run in sorted(
                runs,
                key=lambda item: (
                    item.frozen_identity.fold_id.encode("utf-8"),
                    item.frozen_identity.seed,
                    item.run_id.encode("utf-8"),
                ),
            )
        )

    def _prediction_artifacts_reproducible(
        self, evidence: AggregateEvidence
    ) -> bool:
        for run in self._registered_member_runs(evidence):
            reference = run.prediction_artifact_ref
            if reference is None:
                return False
            if evidence.aggregate.prediction_hashes.get(run.run_id) != reference.content_hash:
                return False
            try:
                registered = self.registry.get_artifact(reference.artifact_id)
            except KeyError:
                return False
            if registered.artifact != reference or registered.run_id != run.run_id:
                return False
        return True

    def _record_comparison_failure(
        self,
        candidate: AggregateEvidence,
        control: AggregateEvidence,
        candidate_runs: Sequence[RunRecord],
        control_runs: Sequence[RunRecord],
        *,
        field: str,
        expected: Any,
        observed: Any,
        reason: str,
    ) -> None:
        self.error_journal.append(
            ErrorRecord(
                error_code="CANDIDATE_COMPARISON_VALIDATION_FAILED",
                failed_stage="candidate_comparison",
                affected_identifiers={
                    "candidate_aggregate_id": candidate.aggregate.aggregate_id,
                    "candidate_run_ids": ",".join(
                        run.run_id for run in candidate_runs
                    ),
                    "control_aggregate_id": control.aggregate.aggregate_id,
                    "control_run_ids": ",".join(run.run_id for run in control_runs),
                    "reason_code": reason,
                },
                failing_field=field,
                expected_value=expected,
                observed_value=observed,
                invariant_checks={
                    "claim_status_unchanged": True,
                    "run_registry_unchanged": True,
                    "aggregate_registry_unchanged": True,
                },
            )
        )

    @staticmethod
    def _lower_to(current: ClaimStatus, ceiling: ClaimStatus) -> ClaimStatus:
        return _CLAIM_PRECEDENCE[
            max(_CLAIM_PRECEDENCE.index(current), _CLAIM_PRECEDENCE.index(ceiling))
        ]

    @staticmethod
    def _sorted_strings(values: Iterable[str]) -> Tuple[str, ...]:
        return tuple(sorted(values, key=lambda value: value.encode("utf-8")))

    @classmethod
    def _pair_text(cls, pairs: Iterable[Tuple[str, int]]) -> Tuple[str, ...]:
        return tuple(
            f"{fold_id}|{seed}"
            for fold_id, seed in sorted(
                pairs, key=lambda item: (item[0].encode("utf-8"), item[1])
            )
        )

    def _record_incomplete(
        self,
        aggregate_id: str,
        decision: CompletenessDecision,
    ) -> None:
        if decision.missing_fold_ids:
            self.error_journal.append(
                ErrorRecord(
                    error_code="PUBLICATION_AGGREGATE_INCOMPLETE_FOLDS",
                    failed_stage="publication_promotion",
                    affected_identifiers={"aggregate_id": aggregate_id},
                    failing_field="fold_ids",
                    expected_value="complete declared Fold_ID coverage",
                    observed_value={
                        "missing_fold_seed_pairs": list(
                            decision.missing_fold_seed_pairs
                        )
                    },
                    missing_ids=decision.missing_fold_ids,
                    invariant_checks={
                        "publication_aggregate_not_created": True,
                        "run_registry_unchanged": True,
                        "aggregate_registry_unchanged": True,
                    },
                )
            )
        if decision.missing_seeds:
            self.error_journal.append(
                ErrorRecord(
                    error_code="PUBLICATION_AGGREGATE_INCOMPLETE_SEEDS",
                    failed_stage="publication_promotion",
                    affected_identifiers={"aggregate_id": aggregate_id},
                    failing_field="seed_set",
                    expected_value="complete declared Seed_Set coverage",
                    observed_value={
                        "missing_fold_seed_pairs": list(
                            decision.missing_fold_seed_pairs
                        )
                    },
                    missing_ids=tuple(str(seed) for seed in decision.missing_seeds),
                    invariant_checks={
                        "publication_aggregate_not_created": True,
                        "run_registry_unchanged": True,
                        "aggregate_registry_unchanged": True,
                    },
                )
            )


__all__ = [
    "AggregateCompleteness",
    "AggregateEvidence",
    "BenchmarkTier",
    "BenchmarkTrack",
    "ComparisonDecision",
    "CompletenessDecision",
    "PromotionDecision",
    "PromotionGate",
    "TierGroup",
]
