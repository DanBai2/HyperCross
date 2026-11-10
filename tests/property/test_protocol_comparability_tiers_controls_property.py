"""Property 15: Protocol comparability, model tiers, and matched controls.

**Validates: Requirements 3.8, 3.9, 7.1-7.20, 10.12-10.17**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings, strategies as st

from benchmark.registry import RunRegistry
from benchmark.reporting import (
    BenchmarkTier,
    BenchmarkTrack,
    PromotionGate,
    PublicationSource,
    ReportGenerator,
)
from benchmark.schema import ClaimStatus, ImplementationKind
from tests.unit.test_reporting_promotion import (
    _budget,
    _create_evidence,
    _hash,
    _metric_schema,
    _protocol,
)


_KIND_TO_TIER = {
    ImplementationKind.CLASSIC: BenchmarkTier.CLASSIC,
    ImplementationKind.IN_REPO: BenchmarkTier.IN_REPO,
    ImplementationKind.OFFICIAL: BenchmarkTier.OFFICIAL,
    ImplementationKind.ADAPTED_OFFICIAL: BenchmarkTier.ADAPTED,
    ImplementationKind.PROXY: BenchmarkTier.PROXY,
}
_PROTOCOL_CHANGING_DRIFTS = {"protocol", "folds", "dataset", "seeds"}
_EXPECTED_REPORT_COLUMNS = {
    "protocol_name",
    "representation_key",
    "parameter_count",
    "completed_training_steps",
    "input_resolution",
    "pretraining_data_status",
    "wall_clock_duration_ms",
    "peak_device_memory_bytes",
}


@dataclass(frozen=True)
class _ComparisonCase:
    candidate_kind: ImplementationKind
    control_kind: ImplementationKind
    representation: str
    drift: str
    matched_control_available: bool
    track: BenchmarkTrack


@st.composite
def _comparison_cases(draw: st.DrawFn) -> _ComparisonCase:
    return _ComparisonCase(
        candidate_kind=draw(st.sampled_from(tuple(ImplementationKind))),
        control_kind=draw(st.sampled_from(tuple(ImplementationKind))),
        representation=draw(
            st.sampled_from(("visible-rgb.v1", "pca-rgb.v1", "full-cube.v1"))
        ),
        drift=draw(
            st.sampled_from(
                ("none", "protocol", "folds", "dataset", "seeds", "metric", "budget")
            )
        ),
        matched_control_available=draw(st.booleans()),
        track=draw(st.sampled_from(tuple(BenchmarkTrack))),
    )


def _control_protocol(drift: str):
    if drift == "protocol":
        return _protocol(key="leaf.official.v2")
    if drift == "folds":
        return _protocol(folds=("fold-b",))
    if drift == "dataset":
        return _protocol(dataset_hash=_hash(99))
    if drift == "seeds":
        return _protocol(seeds=(2,))
    return _protocol()


def _expected_reasons(drift: str, track: BenchmarkTrack) -> set[str]:
    reasons: set[str] = set()
    if drift in _PROTOCOL_CHANGING_DRIFTS:
        reasons.add("PROTOCOL_HASH_MISMATCH")
    if drift == "folds":
        reasons.add("FOLD_MANIFEST_MISMATCH")
    elif drift == "dataset":
        reasons.add("DATASET_HASH_MISMATCH")
    elif drift == "seeds":
        reasons.add("SEED_SET_MISMATCH")
    elif drift == "metric":
        reasons.add("METRIC_CONFIGURATION_MISMATCH")
    elif drift == "budget":
        reasons.add("RESOURCE_BUDGET_MISMATCH")
    if track is BenchmarkTrack.REMOTE_SENSING_SANITY:
        reasons.add("REMOTE_SANITY_RANKING_EXCLUDED")
    return reasons


@settings(max_examples=100, deadline=None)
@given(case=_comparison_cases())
def test_protocol_comparability_tiers_matched_controls_and_report_columns(
    case: _ComparisonCase,
) -> None:
    """Generated comparisons preserve protocol, tier, control, and track boundaries."""

    with TemporaryDirectory(prefix="property15-") as directory:
        registry = RunRegistry(Path(directory) / "registry.jsonl")
        candidate_protocol = _protocol()
        control_protocol = _control_protocol(case.drift)
        candidate = _create_evidence(
            registry,
            token="property15-candidate",
            protocol=candidate_protocol,
            kind=case.candidate_kind,
            representation=case.representation,
            reference_representation="visible-rgb.v1",
            track=case.track,
        )
        control_representation = (
            case.representation
            if case.matched_control_available
            else (
                "visible-rgb.v1"
                if case.representation != "visible-rgb.v1"
                else "pca-rgb.v1"
            )
        )
        control = _create_evidence(
            registry,
            token="property15-control",
            protocol=control_protocol,
            kind=case.control_kind,
            representation=control_representation,
            track=case.track,
            metric_schema=(
                _metric_schema(1e-6) if case.drift == "metric" else _metric_schema()
            ),
            budget=_budget(3) if case.drift == "budget" else _budget(),
        )
        gate = PromotionGate(registry)

        groups = gate.group_by_tier((control, candidate))
        grouped_ids = {group.tier: set(group.aggregate_ids) for group in groups}
        assert tuple(group.tier for group in groups) == tuple(BenchmarkTier)
        assert candidate.aggregate.aggregate_id in grouped_ids[
            _KIND_TO_TIER[case.candidate_kind]
        ]
        assert control.aggregate.aggregate_id in grouped_ids[
            _KIND_TO_TIER[case.control_kind]
        ]

        comparison = gate.compare(candidate, control)
        expected_reasons = _expected_reasons(case.drift, case.track)
        expected_comparable = not expected_reasons
        assert comparison.comparable is expected_comparable
        assert comparison.rank_value_allowed is expected_comparable
        assert comparison.improvement_value_allowed is expected_comparable
        assert set(comparison.non_comparability_reasons) == expected_reasons
        assert (
            comparison.candidate_protocol_section
            != comparison.control_protocol_section
        ) is (case.drift in _PROTOCOL_CHANGING_DRIFTS)

        distinct_representation = case.representation != "visible-rgb.v1"
        control_is_matched = (
            case.matched_control_available
            and case.drift == "none"
            and case.track is not BenchmarkTrack.REMOTE_SENSING_SANITY
        )
        promotion = gate.promote(candidate, controls=(control,))
        assert promotion.matched_control_aggregate_id == (
            control.aggregate.aggregate_id
            if distinct_representation and control_is_matched
            else None
        )
        expected_primary = not distinct_representation or control_is_matched
        assert promotion.effective_claim_status is (
            ClaimStatus.PRIMARY_STRICT
            if expected_primary
            else ClaimStatus.STRICT_CANDIDATE
        )
        assert ("MATCHED_INPUT_CONTROL_MISSING" in promotion.blocking_reasons) is (
            distinct_representation and not control_is_matched
        )

        assert gate.ranking_allowed(BenchmarkTrack.HYPERLEAF) is True
        assert gate.ranking_allowed(BenchmarkTrack.HISTOLOGY) is True
        assert gate.ranking_allowed(BenchmarkTrack.REMOTE_SENSING_SANITY) is False

        matrix = ReportGenerator(registry).build_matrix(
            PublicationSource(
                aggregate_id=candidate.aggregate.aggregate_id,
                protocol_name=candidate.protocol.protocol_key,
            )
        )
        assert _EXPECTED_REPORT_COLUMNS.issubset(matrix.columns)
        row = matrix.rows[0].cells
        assert all(row[column] != "" for column in _EXPECTED_REPORT_COLUMNS)
