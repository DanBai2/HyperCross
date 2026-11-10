"""Property 12: One-shot evaluation budget.

**Validates: Requirements 5.19-5.24, 8.6, 8.7, 9.7, 9.8**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.engine import (
    EvaluationBudgetExceeded,
    EvaluationLeaseError,
    OuterEvaluationGate,
)
from benchmark.metrics import PredictionArtifactWriter
from benchmark.recovery import ArtifactStoreError, FailureAtomicArtifactStore
from benchmark.schema import (
    ArtifactReference,
    EvaluationBudget,
    FitBoundaryScope,
    FitDataRole,
    FittedArtifact,
    FittedArtifactKind,
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


_FROZEN_AT = datetime(2027, 1, 2, 3, 4, tzinfo=timezone.utc)
_FAILURE_MODES = ("prediction", "label_unlock", "publication")


@dataclass(frozen=True)
class _EvaluationSequence:
    maximum_successes: int
    middle_events: tuple[tuple[str, str | int], ...]
    duplicate_commit_retries: tuple[int, ...]
    overbudget_request: int


@st.composite
def _evaluation_sequences(draw: st.DrawFn) -> _EvaluationSequence:
    maximum = draw(st.integers(min_value=2, max_value=4))
    failure_modes = draw(
        st.lists(
            st.sampled_from(_FAILURE_MODES),
            min_size=1,
            max_size=4,
        )
    )
    middle_events: list[tuple[str, str | int]] = [
        ("success", index) for index in range(1, maximum - 1)
    ]
    middle_events.extend(("failure", mode) for mode in failure_modes)
    order = draw(st.permutations(tuple(range(len(middle_events)))))
    return _EvaluationSequence(
        maximum_successes=maximum,
        middle_events=tuple(middle_events[index] for index in order),
        duplicate_commit_retries=tuple(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=3),
                    min_size=maximum,
                    max_size=maximum,
                )
            )
        ),
        overbudget_request=draw(st.integers(min_value=1, max_value=4)),
    )


def _fold() -> OuterFold:
    return OuterFold(
        fold_id="outer-1",
        outer_train_ids=("train-a", "train-b"),
        outer_test_ids=("official-a", "official-b"),
        inner_folds=(
            InnerFold(
                fold_id="inner-1",
                train_ids=("train-a",),
                validation_ids=("train-b",),
            ),
        ),
    )


def _protocol(maximum: int) -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1",
        protocol_key="leaf.official.property12.v1",
        dataset_hash="1" * 64,
        folds=(_fold(),),
        selection_policy=SelectionPolicy(
            policy_key="inner-r2-v1",
            objective_metric="mean-r2",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule="candidate-id-ascending",
        ),
        seed_set=(42,),
        evaluation_budget=EvaluationBudget(
            maximum_successful_evaluations=maximum,
        ),
    )


def _decision() -> SelectionDecision:
    return SelectionDecision.create(
        decision_id="decision-outer-1",
        fold_id="outer-1",
        candidates=(
            SelectionCandidate(
                candidate_id="candidate-a",
                configuration={"epoch": 2},
            ),
        ),
        candidate_scores={"candidate-a": 0.5},
        source_sample_ids=("train-b",),
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate-id-ascending",
        source_prediction_hash="2" * 64,
        selected_candidate="candidate-a",
        frozen_utc=_FROZEN_AT,
    )


def _checkpoint(decision: SelectionDecision) -> FrozenOuterCheckpoint:
    return FrozenOuterCheckpoint(
        artifact=ArtifactReference(
            artifact_id="outer-checkpoint-1",
            relative_path="checkpoints/outer-1.canonical.json",
            content_hash="3" * 64,
        ),
        fold_id="outer-1",
        seed=42,
        inference_branch="primary",
        selection_decision_hash=decision.decision_hash,
        frozen_utc=_FROZEN_AT,
    )


def _fitted(protocol: ProtocolManifest) -> FittedArtifact:
    config = {"epsilon": 1e-12}
    state = {"mean": [0.5], "scale": [1.0]}
    return FittedArtifact(
        artifact_id="fitted-outer-1",
        kind=FittedArtifactKind.BAND_NORMALIZATION,
        dataset_hash=protocol.dataset_hash,
        protocol_hash=protocol.protocol_hash,
        fold_id="outer-1",
        boundary_scope=FitBoundaryScope.OUTER_TRAIN,
        fit_data_role=FitDataRole.TRAINING_VALUES,
        fit_sample_ids=("train-a", "train-b"),
        config=config,
        config_hash=hash_object(config),
        input_hash="4" * 64,
        state=state,
        state_hash=hash_object(state),
        output_shape=(2, 1),
    )


class _GeneratedLabelVault:
    def __init__(self) -> None:
        self.unlock_calls: list[tuple[object, tuple[str, ...]]] = []
        self.fail_next = False

    def unlock(self, lease: object, sample_ids: tuple[str, ...]) -> dict[str, int]:
        self.unlock_calls.append((lease, sample_ids))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("generated label unlock failure")
        return {"official-a": 1, "official-b": 0}


def _prediction_snapshot(root: Path) -> dict[str, bytes]:
    prediction_root = root / "predictions"
    if not prediction_root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(prediction_root.rglob("*"))
        if path.is_file()
    }


def _predict(_checkpoint: FrozenOuterCheckpoint, sample_id: str) -> dict[str, Any]:
    return {
        "y_pred": int(sample_id == "official-a"),
        "probability": 0.8 if sample_id == "official-a" else 0.2,
    }


def _evaluate(
    gate: OuterEvaluationGate,
    lease: object,
    protocol: ProtocolManifest,
    decision: SelectionDecision,
    *,
    predictor: Callable[[FrozenOuterCheckpoint, str], dict[str, Any]],
    artifact_id: str,
    run_id: str,
    relative_path: str,
):
    return gate.evaluate(
        lease,
        outer_fold=_fold(),
        checkpoint=_checkpoint(decision),
        selection_decision=decision,
        fitted_artifacts=(_fitted(protocol),),
        predictor=predictor,
        artifact_id=artifact_id,
        run_id=run_id,
        relative_path=relative_path,
    )


@settings(max_examples=100, deadline=None)
@given(sequence=_evaluation_sequences())
def test_generated_success_failure_retry_and_overbudget_sequences_are_one_shot(
    sequence: _EvaluationSequence,
) -> None:
    """Only successful publications consume budget, exactly once."""

    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "artifact root"
        protocol = _protocol(sequence.maximum_successes)
        decision = _decision()
        vault = _GeneratedLabelVault()
        gate = OuterEvaluationGate(
            protocol,
            vault,
            PredictionArtifactWriter(FailureAtomicArtifactStore(root)),
        )
        successful_count = 0

        def run_success(success_index: int) -> None:
            nonlocal successful_count
            before_predictions = _prediction_snapshot(root)
            before_unlocks = len(vault.unlock_calls)
            lease = gate.reserve(protocol.protocol_key)

            published = _evaluate(
                gate,
                lease,
                protocol,
                decision,
                predictor=_predict,
                artifact_id=f"prediction-success-{success_index}",
                run_id=f"run-success-{success_index}",
                relative_path=f"predictions/success-{success_index}.json",
            )
            successful_count += 1

            after_predictions = _prediction_snapshot(root)
            assert published.artifact.complete is True
            assert len(after_predictions) == len(before_predictions) + 1
            assert all(after_predictions[path] == content for path, content in before_predictions.items())
            assert len(vault.unlock_calls) == before_unlocks + 1
            assert gate.budget.consumed_successful_evaluations == successful_count
            assert gate.reserved_evaluations == 0

            for _ in range(sequence.duplicate_commit_retries[success_index]):
                budget = gate.commit_successful_evaluation(lease)
                assert budget.consumed_successful_evaluations == successful_count

            completed_before_retry = _prediction_snapshot(root)
            unlocks_before_retry = len(vault.unlock_calls)
            with pytest.raises(EvaluationLeaseError):
                _evaluate(
                    gate,
                    lease,
                    protocol,
                    decision,
                    predictor=_predict,
                    artifact_id=f"prediction-success-retry-{success_index}",
                    run_id=f"run-success-retry-{success_index}",
                    relative_path=f"predictions/success-retry-{success_index}.json",
                )
            assert gate.budget.consumed_successful_evaluations == successful_count
            assert len(vault.unlock_calls) == unlocks_before_retry
            assert _prediction_snapshot(root) == completed_before_retry

        def run_failure(failure_index: int, mode: str) -> None:
            before_predictions = _prediction_snapshot(root)
            before_unlocks = len(vault.unlock_calls)
            before_consumed = gate.budget.consumed_successful_evaluations
            lease = gate.reserve(protocol.protocol_key)
            predictor = _predict
            relative_path = f"predictions/failure-{failure_index}.json"
            expected_error: type[Exception] = RuntimeError

            if mode == "prediction":
                def fail_prediction(
                    _checkpoint: FrozenOuterCheckpoint,
                    _sample_id: str,
                ) -> dict[str, Any]:
                    raise RuntimeError("generated prediction failure")

                predictor = fail_prediction
            elif mode == "label_unlock":
                vault.fail_next = True
            else:
                relative_path = "predictions/success-0.json"
                expected_error = ArtifactStoreError

            with pytest.raises(expected_error):
                _evaluate(
                    gate,
                    lease,
                    protocol,
                    decision,
                    predictor=predictor,
                    artifact_id=f"prediction-failure-{failure_index}",
                    run_id=f"run-failure-{failure_index}",
                    relative_path=relative_path,
                )

            expected_unlocks = before_unlocks if mode == "prediction" else before_unlocks + 1
            assert len(vault.unlock_calls) == expected_unlocks
            assert gate.budget.consumed_successful_evaluations == before_consumed
            assert gate.reserved_evaluations == 0
            assert _prediction_snapshot(root) == before_predictions

            with pytest.raises(EvaluationLeaseError):
                _evaluate(
                    gate,
                    lease,
                    protocol,
                    decision,
                    predictor=_predict,
                    artifact_id=f"prediction-failure-retry-{failure_index}",
                    run_id=f"run-failure-retry-{failure_index}",
                    relative_path=f"predictions/failure-retry-{failure_index}.json",
                )
            assert len(vault.unlock_calls) == expected_unlocks
            assert gate.budget.consumed_successful_evaluations == before_consumed
            assert _prediction_snapshot(root) == before_predictions

        # Keep one successful slot available while generated failures and retries run.
        run_success(0)
        failure_index = 0
        for event, value in sequence.middle_events:
            if event == "success":
                run_success(int(value))
            else:
                run_failure(failure_index, str(value))
                failure_index += 1
        run_success(sequence.maximum_successes - 1)

        assert successful_count == sequence.maximum_successes
        assert gate.budget.consumed_successful_evaluations == sequence.maximum_successes
        assert gate.budget.remaining_successful_evaluations == 0

        completed_before_overbudget = _prediction_snapshot(root)
        unlocks_before_overbudget = len(vault.unlock_calls)
        with pytest.raises(EvaluationBudgetExceeded) as caught:
            gate.reserve(protocol.protocol_key, sequence.overbudget_request)

        error = caught.value.error_record
        assert error.error_code == "EVALUATION_BUDGET_EXCEEDED"
        assert error.affected_identifiers["protocol_key"] == protocol.protocol_key
        assert error.expected_value == {
            "protocol_key": protocol.protocol_key,
            "maximum_successful_evaluations": sequence.maximum_successes,
        }
        assert error.observed_value == {
            "consumed_successful_evaluations": sequence.maximum_successes,
            "reserved_evaluations": 0,
            "requested_count": sequence.overbudget_request,
        }
        assert len(vault.unlock_calls) == unlocks_before_overbudget
        assert gate.budget.consumed_successful_evaluations == sequence.maximum_successes
        assert gate.reserved_evaluations == 0
        assert _prediction_snapshot(root) == completed_before_overbudget
