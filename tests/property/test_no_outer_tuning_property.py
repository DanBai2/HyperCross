"""Property 11: No outer tuning.

**Validates: Requirements 5.1-5.17**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.engine import OuterEvaluationGate, SelectionController, SelectionValidationError
from benchmark.metrics import PredictionArtifactWriter
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    ArtifactReference,
    ClaimStatus,
    EvaluationBudget,
    FrozenOuterCheckpoint,
    InnerFold,
    MetricDirection,
    OuterFold,
    ProtocolManifest,
    SelectionCandidate,
    SelectionDecision,
    SelectionPolicy,
    hash_object,
)


_FROZEN_AT = datetime(2027, 4, 5, 6, 7, tzinfo=timezone.utc)
_TIE_BREAK = "candidate_id_utf8_ascending"


@dataclass(frozen=True)
class _SelectionCase:
    fold: OuterFold
    candidates: tuple[SelectionCandidate, ...]
    candidate_scores: dict[str, float]
    source_sample_ids: tuple[str, ...]
    contaminated_source_sample_ids: tuple[str, ...]
    contamination_id: str
    metric_direction: MetricDirection
    source_prediction_hash: str
    labels: dict[str, int]


@st.composite
def _selection_cases(draw: st.DrawFn) -> _SelectionCase:
    """Generate complete selection spaces and one outer-test contamination."""

    inner_train_count = draw(st.integers(min_value=1, max_value=6))
    inner_validation_count = draw(st.integers(min_value=1, max_value=5))
    outer_test_count = draw(st.integers(min_value=1, max_value=3))

    inner_train_ids = tuple(
        f"inner-train-{index:02d}" for index in range(inner_train_count)
    )
    inner_validation_ids = tuple(
        f"inner-validation-{index:02d}"
        for index in range(inner_validation_count)
    )
    outer_test_ids = tuple(
        f"outer-test-{index:02d}" for index in range(outer_test_count)
    )
    outer_train_ids = tuple(
        draw(st.permutations(inner_train_ids + inner_validation_ids))
    )
    fold = OuterFold(
        fold_id="outer-property-11",
        outer_train_ids=outer_train_ids,
        outer_test_ids=tuple(draw(st.permutations(outer_test_ids))),
        inner_folds=(
            InnerFold(
                fold_id="inner-property-11",
                train_ids=tuple(draw(st.permutations(inner_train_ids))),
                validation_ids=tuple(draw(st.permutations(inner_validation_ids))),
            ),
        ),
    )

    candidate_count = draw(st.integers(min_value=1, max_value=6))
    candidate_ids = tuple(
        f"candidate-{index:02d}" for index in range(candidate_count)
    )
    candidates_by_id: dict[str, SelectionCandidate] = {}
    for candidate_id in candidate_ids:
        candidates_by_id[candidate_id] = SelectionCandidate(
            candidate_id=candidate_id,
            configuration={
                "epoch": draw(st.integers(min_value=1, max_value=200)),
                "model_family": draw(
                    st.sampled_from(("linear", "fusion", "local-token"))
                ),
                "target_route": draw(st.sampled_from(("rgb", "pca", "shared"))),
                "blend_weight": draw(
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                        width=32,
                    )
                ),
                "threshold": draw(
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                        width=32,
                    )
                ),
                "calibration": draw(
                    st.sampled_from(("none", "temperature", "inner-oof"))
                ),
            },
        )
    candidates = tuple(
        candidates_by_id[candidate_id]
        for candidate_id in draw(st.permutations(candidate_ids))
    )

    scores = draw(
        st.lists(
            st.floats(
                min_value=-1000.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=candidate_count,
            max_size=candidate_count,
        )
    )
    score_items = tuple(zip(candidate_ids, scores))
    candidate_scores = dict(draw(st.permutations(score_items)))
    source_sample_ids = tuple(draw(st.permutations(inner_validation_ids)))
    contamination_id = draw(st.sampled_from(outer_test_ids))
    contaminated_source_sample_ids = tuple(
        draw(st.permutations(source_sample_ids + (contamination_id,)))
    )
    metric_direction = draw(st.sampled_from(tuple(MetricDirection)))
    source_prediction_hash = hash_object(
        {
            "candidate_scores": candidate_scores,
            "source_sample_ids": source_sample_ids,
        }
    )
    labels = {
        sample_id: index % 2 for index, sample_id in enumerate(fold.outer_test_ids)
    }
    return _SelectionCase(
        fold=fold,
        candidates=candidates,
        candidate_scores=candidate_scores,
        source_sample_ids=source_sample_ids,
        contaminated_source_sample_ids=contaminated_source_sample_ids,
        contamination_id=contamination_id,
        metric_direction=metric_direction,
        source_prediction_hash=source_prediction_hash,
        labels=labels,
    )


def _policy(direction: MetricDirection) -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="property-11-inner-selection",
        objective_metric="generated-inner-objective",
        metric_direction=direction,
        tie_break_rule=_TIE_BREAK,
    )


def _expected_winner(case: _SelectionCase) -> str:
    target_score = (
        max(case.candidate_scores.values())
        if case.metric_direction is MetricDirection.MAXIMIZE
        else min(case.candidate_scores.values())
    )
    tied = (
        candidate_id
        for candidate_id, score in case.candidate_scores.items()
        if score == target_score
    )
    return min(tied, key=lambda candidate_id: candidate_id.encode("utf-8"))


class _FreezeCheckingVault:
    """Fail if outer labels are requested before an intact decision is frozen."""

    def __init__(
        self,
        labels: dict[str, int],
        decision_box: list[SelectionDecision],
        events: list[str],
    ) -> None:
        self._labels = labels
        self._decision_box = decision_box
        self._events = events
        self.unlock_calls = 0

    def unlock(self, lease: object, sample_ids: tuple[str, ...]) -> dict[str, int]:
        assert self._decision_box, "outer labels opened before selection freeze"
        self._decision_box[0].require_integrity()
        assert all(event.startswith("predict:") for event in self._events[1:])
        assert set(sample_ids) == set(self._labels)
        self._events.append("outer-label-unlock")
        self.unlock_calls += 1
        return dict(self._labels)


@settings(max_examples=100, deadline=None)
@given(case=_selection_cases())
def test_generated_selection_is_complete_outer_is_rejected_and_freeze_precedes_access(
    case: _SelectionCase,
) -> None:
    """Generated selection evidence stays complete, inner-only, and frozen first."""

    controller = SelectionController()
    policy = _policy(case.metric_direction)
    decision_box: list[SelectionDecision] = []
    events: list[str] = []
    vault = _FreezeCheckingVault(case.labels, decision_box, events)

    with pytest.raises(SelectionValidationError) as caught:
        controller.freeze_selection(
            case.fold,
            candidates=case.candidates,
            candidate_scores=case.candidate_scores,
            source_sample_ids=case.contaminated_source_sample_ids,
            source_prediction_hash=case.source_prediction_hash,
            selection_policy=policy,
            decision_id="contaminated-property-11",
            frozen_utc=_FROZEN_AT,
        )

    assert vault.unlock_calls == 0
    assert caught.value.claim_status is ClaimStatus.INVALID_OR_LEAKY
    forbidden = next(
        record
        for record in caught.value.error_records
        if record.error_code == "SELECTION_OUTER_EVIDENCE_FORBIDDEN"
    )
    assert forbidden.failed_stage == "selection_validation"
    assert forbidden.affected_identifiers == {
        "selection_decision_id": "contaminated-property-11",
        "fold_id": case.fold.fold_id,
    }
    assert forbidden.unexpected_ids == (case.contamination_id,)
    assert forbidden.invariant_checks["checkpoint_unchanged"] is True

    decision = controller.freeze_selection(
        case.fold,
        candidates=case.candidates,
        candidate_scores=case.candidate_scores,
        source_sample_ids=case.source_sample_ids,
        source_prediction_hash=case.source_prediction_hash,
        selection_policy=policy,
        decision_id="valid-property-11",
        frozen_utc=_FROZEN_AT,
    )
    decision_box.append(decision)
    events.append("selection-frozen")

    expected_candidates = {
        candidate.candidate_id: candidate.configuration for candidate in case.candidates
    }
    assert {
        candidate.candidate_id: candidate.configuration
        for candidate in decision.candidates
    } == expected_candidates
    assert decision.candidate_scores == case.candidate_scores
    assert set(decision.source_sample_ids) == set(case.source_sample_ids)
    assert set(decision.source_sample_ids) <= set(case.fold.outer_train_ids)
    assert set(decision.source_sample_ids).isdisjoint(case.fold.outer_test_ids)
    assert decision.metric_direction is case.metric_direction
    assert decision.tie_break_rule == _TIE_BREAK
    assert decision.source_prediction_hash == case.source_prediction_hash
    assert decision.selected_candidate == _expected_winner(case)
    assert decision.frozen_utc == _FROZEN_AT
    assert decision.decision_hash == hash_object(decision.hash_payload())
    decision.require_integrity()

    refit_request = controller.prepare_refit(case.fold, decision)
    assert refit_request.selection_decision is decision
    assert refit_request.outer_train_ids == case.fold.outer_train_ids

    protocol = ProtocolManifest.create(
        schema_version="1",
        protocol_key="property-11.protocol.v1",
        dataset_hash="1" * 64,
        folds=(case.fold,),
        selection_policy=policy,
        seed_set=(11,),
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=1),
    )
    checkpoint = FrozenOuterCheckpoint(
        artifact=ArtifactReference(
            artifact_id="property-11-checkpoint",
            relative_path="checkpoints/property-11.bin",
            content_hash="2" * 64,
        ),
        fold_id=case.fold.fold_id,
        seed=11,
        inference_branch="primary",
        selection_decision_hash=decision.decision_hash,
        frozen_utc=_FROZEN_AT,
    )

    with TemporaryDirectory(prefix="property-11-") as directory:
        gate = OuterEvaluationGate(
            protocol,
            vault,
            PredictionArtifactWriter(
                FailureAtomicArtifactStore(Path(directory) / "artifacts with spaces")
            ),
        )
        lease = gate.reserve(protocol.protocol_key)

        def predict(
            received_checkpoint: FrozenOuterCheckpoint,
            sample_id: str,
        ) -> dict[str, Any]:
            assert received_checkpoint is checkpoint
            decision.require_integrity()
            assert vault.unlock_calls == 0
            events.append(f"predict:{sample_id}")
            return {"y_pred": 0, "probability": 0.5}

        gate.evaluate(
            lease,
            outer_fold=case.fold,
            checkpoint=checkpoint,
            selection_decision=decision,
            fitted_artifacts=(),
            predictor=predict,
            artifact_id="property-11-predictions",
            run_id="property-11-run",
        )

    assert events[0] == "selection-frozen"
    assert events[-1] == "outer-label-unlock"
    assert events[1:-1] == [
        f"predict:{sample_id}" for sample_id in case.fold.outer_test_ids
    ]
    assert vault.unlock_calls == 1
