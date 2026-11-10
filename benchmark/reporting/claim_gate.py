"""Deterministic evidence ceilings for CVPR claim classification."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from pydantic import StrictBool

from benchmark.registry.records import ClaimCeiling, ClaimDecision
from benchmark.schema.base import ClaimId, ClaimStatus, ImplementationKind
from benchmark.schema.environment import FrozenSchemaModel


# Highest to lowest. Keep this independent of Enum declaration order so adding
# an enum member cannot silently change publication policy.
CLAIM_STATUS_PRECEDENCE: Tuple[ClaimStatus, ...] = (
    ClaimStatus.PRIMARY_STRICT,
    ClaimStatus.STRICT_CANDIDATE,
    ClaimStatus.DEVELOPMENT,
    ClaimStatus.EXPLORATORY,
    ClaimStatus.SANITY_ONLY,
    ClaimStatus.PROXY_ONLY,
    ClaimStatus.INVALID_OR_LEAKY,
)
_STATUS_RANK = {status: rank for rank, status in enumerate(CLAIM_STATUS_PRECEDENCE)}


class ClaimEvidence(FrozenSchemaModel):
    """Validated facts used by the Claim Gate.

    Every primary-strict prerequisite is explicit. ``score`` is retained only
    as evidence metadata and intentionally never participates in
    classification.
    """

    claim_id: ClaimId
    implementation_kind: ImplementationKind

    detected_leakage: StrictBool
    model_identity_mismatch: StrictBool
    remote_sensing_sanity_track: StrictBool
    random_split: StrictBool
    outer_data_used_for_model_selection: StrictBool
    single_run: StrictBool
    outer_data_used_for_ensemble_weight_search: StrictBool
    outer_data_used_for_threshold_search: StrictBool

    strict_protocol_complete: StrictBool
    provenance_complete: StrictBool
    seed_set_complete: StrictBool
    folds_complete: StrictBool
    matched_input_control_required: StrictBool
    matched_input_control_complete: StrictBool
    prediction_artifact_complete: StrictBool
    unified_runner_reproduced: StrictBool

    score: Optional[float] = None

    def prerequisite_checks(self) -> Dict[str, bool]:
        """Return the canonical primary-strict prerequisite check set."""

        return {
            "strict_protocol": bool(self.strict_protocol_complete),
            "provenance": bool(self.provenance_complete),
            "seed_set": bool(self.seed_set_complete),
            "folds": bool(self.folds_complete),
            "required_control": bool(
                not self.matched_input_control_required
                or self.matched_input_control_complete
            ),
            "prediction_artifact": bool(self.prediction_artifact_complete),
            "reproduction": bool(self.unified_runner_reproduced),
        }


_CHECK_SUPPORTING_REASONS = {
    "strict_protocol": "strict_protocol_check_passed",
    "provenance": "provenance_check_passed",
    "seed_set": "seed_set_check_passed",
    "folds": "fold_completeness_check_passed",
    "required_control": "required_control_check_passed",
    "prediction_artifact": "prediction_artifact_check_passed",
    "reproduction": "reproduction_check_passed",
}

_CHECK_BLOCKING_REASONS = {
    "strict_protocol": "strict_protocol_incomplete",
    "provenance": "provenance_incomplete",
    "seed_set": "seed_set_incomplete",
    "folds": "folds_incomplete",
    "required_control": "required_matched_input_control_missing",
    "prediction_artifact": "prediction_artifact_incomplete",
    "reproduction": "unified_runner_reproduction_missing",
}


class ClaimGate:
    """Apply every relevant ceiling and select the fixed-precedence minimum."""

    @staticmethod
    def applicable_ceilings(evidence: ClaimEvidence) -> Tuple[ClaimCeiling, ...]:
        """Return all ceilings that apply, in deterministic policy order."""

        if not isinstance(evidence, ClaimEvidence):
            raise TypeError("evidence must be ClaimEvidence")

        ceilings = []

        def add(status: ClaimStatus, reason_code: str) -> None:
            ceilings.append(
                ClaimCeiling(
                    status=status,
                    reason_code=reason_code,
                    evidence_reference=evidence.claim_id,
                )
            )

        # Invalidating facts are recorded alongside every other applicable
        # ceiling rather than short-circuiting the audit trail.
        if evidence.detected_leakage:
            add(ClaimStatus.INVALID_OR_LEAKY, "detected_leakage")
        if evidence.model_identity_mismatch:
            add(ClaimStatus.INVALID_OR_LEAKY, "model_identity_mismatch")
        if evidence.implementation_kind is ImplementationKind.PROXY:
            add(ClaimStatus.PROXY_ONLY, "proxy_model")
        if evidence.remote_sensing_sanity_track:
            add(ClaimStatus.SANITY_ONLY, "remote_sensing_sanity_track")
        if evidence.random_split:
            add(ClaimStatus.DEVELOPMENT, "random_split")
        if evidence.outer_data_used_for_model_selection:
            add(ClaimStatus.DEVELOPMENT, "outer_data_used_for_model_selection")
        if evidence.single_run:
            add(ClaimStatus.EXPLORATORY, "single_run")
        if evidence.outer_data_used_for_ensemble_weight_search:
            add(
                ClaimStatus.EXPLORATORY,
                "outer_data_used_for_ensemble_weight_search",
            )
        if evidence.outer_data_used_for_threshold_search:
            add(ClaimStatus.EXPLORATORY, "outer_data_used_for_threshold_search")

        for check_name, passed in evidence.prerequisite_checks().items():
            if not passed:
                add(
                    ClaimStatus.STRICT_CANDIDATE,
                    _CHECK_BLOCKING_REASONS[check_name],
                )

        return tuple(ceilings)

    @classmethod
    def classify(cls, evidence: ClaimEvidence) -> ClaimDecision:
        """Return one status plus complete machine-readable decision details."""

        ceilings = cls.applicable_ceilings(evidence)
        checks = evidence.prerequisite_checks()
        status = (
            ClaimStatus.PRIMARY_STRICT
            if not ceilings
            else max(
                (ceiling.status for ceiling in ceilings),
                key=_STATUS_RANK.__getitem__,
            )
        )
        supporting_reasons = tuple(
            _CHECK_SUPPORTING_REASONS[name]
            for name, passed in checks.items()
            if passed
        )
        blocking_reasons = tuple(ceiling.reason_code for ceiling in ceilings)
        return ClaimDecision(
            status=status,
            supporting_reasons=supporting_reasons,
            blocking_reasons=blocking_reasons,
            applied_ceilings=ceilings,
            prerequisite_checks=checks,
        )


__all__ = [
    "CLAIM_STATUS_PRECEDENCE",
    "ClaimEvidence",
    "ClaimGate",
]
