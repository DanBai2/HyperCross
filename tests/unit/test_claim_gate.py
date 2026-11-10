"""Focused unit tests for the fixed-precedence Claim Gate."""

from __future__ import annotations

from typing import Any

import pytest

from benchmark.reporting import ClaimEvidence, ClaimGate
from benchmark.schema import ClaimStatus, ImplementationKind


def _evidence(**overrides: Any) -> ClaimEvidence:
    values = {
        "claim_id": "claim-1",
        "implementation_kind": ImplementationKind.IN_REPO,
        "detected_leakage": False,
        "model_identity_mismatch": False,
        "remote_sensing_sanity_track": False,
        "random_split": False,
        "outer_data_used_for_model_selection": False,
        "single_run": False,
        "outer_data_used_for_ensemble_weight_search": False,
        "outer_data_used_for_threshold_search": False,
        "strict_protocol_complete": True,
        "provenance_complete": True,
        "seed_set_complete": True,
        "folds_complete": True,
        "matched_input_control_required": True,
        "matched_input_control_complete": True,
        "prediction_artifact_complete": True,
        "unified_runner_reproduced": True,
        "score": 0.5,
    }
    values.update(overrides)
    return ClaimEvidence(**values)


def test_complete_strict_evidence_receives_one_primary_status() -> None:
    decision = ClaimGate.classify(_evidence())

    assert decision.status is ClaimStatus.PRIMARY_STRICT
    assert decision.applied_ceilings == ()
    assert decision.blocking_reasons == ()
    assert decision.prerequisite_checks == {
        "strict_protocol": True,
        "provenance": True,
        "seed_set": True,
        "folds": True,
        "required_control": True,
        "prediction_artifact": True,
        "reproduction": True,
    }
    assert decision.supporting_reasons == (
        "strict_protocol_check_passed",
        "provenance_check_passed",
        "seed_set_check_passed",
        "fold_completeness_check_passed",
        "required_control_check_passed",
        "prediction_artifact_check_passed",
        "reproduction_check_passed",
    )


@pytest.mark.parametrize(
    ("overrides", "expected_status", "expected_reason"),
    (
        (
            {"implementation_kind": ImplementationKind.PROXY},
            ClaimStatus.PROXY_ONLY,
            "proxy_model",
        ),
        (
            {"remote_sensing_sanity_track": True},
            ClaimStatus.SANITY_ONLY,
            "remote_sensing_sanity_track",
        ),
        ({"random_split": True}, ClaimStatus.DEVELOPMENT, "random_split"),
        (
            {"outer_data_used_for_model_selection": True},
            ClaimStatus.DEVELOPMENT,
            "outer_data_used_for_model_selection",
        ),
        ({"single_run": True}, ClaimStatus.EXPLORATORY, "single_run"),
        (
            {"outer_data_used_for_ensemble_weight_search": True},
            ClaimStatus.EXPLORATORY,
            "outer_data_used_for_ensemble_weight_search",
        ),
        (
            {"outer_data_used_for_threshold_search": True},
            ClaimStatus.EXPLORATORY,
            "outer_data_used_for_threshold_search",
        ),
    ),
)
def test_each_non_prerequisite_ceiling_is_applied(
    overrides: dict[str, Any],
    expected_status: ClaimStatus,
    expected_reason: str,
) -> None:
    decision = ClaimGate.classify(_evidence(**overrides))

    assert decision.status is expected_status
    assert decision.blocking_reasons == (expected_reason,)
    assert decision.applied_ceilings[0].status is expected_status
    assert decision.applied_ceilings[0].reason_code == expected_reason
    assert decision.applied_ceilings[0].evidence_reference == "claim-1"


@pytest.mark.parametrize(
    ("field", "check_name", "reason"),
    (
        ("strict_protocol_complete", "strict_protocol", "strict_protocol_incomplete"),
        ("provenance_complete", "provenance", "provenance_incomplete"),
        ("seed_set_complete", "seed_set", "seed_set_incomplete"),
        ("folds_complete", "folds", "folds_incomplete"),
        (
            "matched_input_control_complete",
            "required_control",
            "required_matched_input_control_missing",
        ),
        (
            "prediction_artifact_complete",
            "prediction_artifact",
            "prediction_artifact_incomplete",
        ),
        (
            "unified_runner_reproduced",
            "reproduction",
            "unified_runner_reproduction_missing",
        ),
    ),
)
def test_missing_strict_prerequisite_caps_claim_at_strict_candidate(
    field: str,
    check_name: str,
    reason: str,
) -> None:
    decision = ClaimGate.classify(_evidence(**{field: False}))

    assert decision.status is ClaimStatus.STRICT_CANDIDATE
    assert decision.prerequisite_checks[check_name] is False
    assert reason in decision.blocking_reasons


def test_control_check_passes_when_matched_control_is_not_required() -> None:
    decision = ClaimGate.classify(
        _evidence(
            matched_input_control_required=False,
            matched_input_control_complete=False,
        )
    )

    assert decision.status is ClaimStatus.PRIMARY_STRICT
    assert decision.prerequisite_checks["required_control"] is True


def test_all_applicable_ceilings_are_retained_and_lowest_wins() -> None:
    evidence = _evidence(
        implementation_kind=ImplementationKind.PROXY,
        remote_sensing_sanity_track=True,
        random_split=True,
        single_run=True,
        provenance_complete=False,
    )

    decision = ClaimGate.classify(evidence)

    assert decision.status is ClaimStatus.PROXY_ONLY
    assert tuple(ceiling.status for ceiling in decision.applied_ceilings) == (
        ClaimStatus.PROXY_ONLY,
        ClaimStatus.SANITY_ONLY,
        ClaimStatus.DEVELOPMENT,
        ClaimStatus.EXPLORATORY,
        ClaimStatus.STRICT_CANDIDATE,
    )
    assert decision.blocking_reasons == tuple(
        ceiling.reason_code for ceiling in decision.applied_ceilings
    )


@pytest.mark.parametrize("invalid_field", ("detected_leakage", "model_identity_mismatch"))
def test_leakage_or_identity_mismatch_forces_invalid(invalid_field: str) -> None:
    decision = ClaimGate.classify(
        _evidence(
            **{
                invalid_field: True,
                "implementation_kind": ImplementationKind.PROXY,
                "remote_sensing_sanity_track": True,
            }
        )
    )

    assert decision.status is ClaimStatus.INVALID_OR_LEAKY
    assert decision.applied_ceilings[0].status is ClaimStatus.INVALID_OR_LEAKY
    assert invalid_field in decision.blocking_reasons


def test_score_changes_do_not_raise_or_lower_the_claim_ceiling() -> None:
    low = ClaimGate.classify(_evidence(single_run=True, score=-1000.0))
    high = ClaimGate.classify(_evidence(single_run=True, score=1000.0))

    assert low == high
    assert high.status is ClaimStatus.EXPLORATORY


def test_applicable_ceilings_rejects_unvalidated_input() -> None:
    with pytest.raises(TypeError, match="evidence must be ClaimEvidence"):
        ClaimGate.applicable_ceilings({})  # type: ignore[arg-type]
