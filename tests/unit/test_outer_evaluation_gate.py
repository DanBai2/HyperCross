"""Targeted synthetic coverage for task 9.3 outer evaluation leases."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from benchmark.engine import (
    EvaluationBudgetExceeded,
    EvaluationLeaseError,
    OuterEvaluationError,
    OuterEvaluationGate,
)
from benchmark.metrics import PredictionArtifactWriter
from benchmark.recovery import FailureAtomicArtifactStore
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


HASH = "a" * 64
FROZEN_AT = datetime(2027, 1, 2, 3, 4, tzinfo=timezone.utc)


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


def _protocol(maximum: int = 1) -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1",
        protocol_key="leaf.official.synthetic.v1",
        dataset_hash="1" * 64,
        folds=(_fold(),),
        selection_policy=SelectionPolicy(
            policy_key="inner-r2-v1",
            objective_metric="mean-r2",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule="candidate_id_utf8_ascending",
        ),
        seed_set=(42,),
        evaluation_budget=EvaluationBudget(
            maximum_successful_evaluations=maximum
        ),
    )


def _decision() -> SelectionDecision:
    return SelectionDecision.create(
        decision_id="decision-outer-1",
        fold_id="outer-1",
        candidates=(
            SelectionCandidate(candidate_id="candidate-a", configuration={"epoch": 2}),
        ),
        candidate_scores={"candidate-a": 0.5},
        source_sample_ids=("train-b",),
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
        source_prediction_hash="2" * 64,
        selected_candidate="candidate-a",
        frozen_utc=FROZEN_AT,
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
        frozen_utc=FROZEN_AT,
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


class _SyntheticLabelVault:
    def __init__(self) -> None:
        self.unlock_calls: list[tuple[object, tuple[str, ...]]] = []

    def unlock(self, lease, sample_ids):
        self.unlock_calls.append((lease, sample_ids))
        return {"official-a": 1, "official-b": 0}


def _gate(tmp_path: Path, maximum: int = 1):
    protocol = _protocol(maximum)
    vault = _SyntheticLabelVault()
    writer = PredictionArtifactWriter(
        FailureAtomicArtifactStore(tmp_path / "synthetic artifacts")
    )
    return protocol, vault, OuterEvaluationGate(protocol, vault, writer)


def test_frozen_checkpoint_evaluation_unlocks_labels_only_after_inference_and_commits_once(
    tmp_path: Path,
) -> None:
    """Validates Requirements 5.13, 5.19-5.21, 8.6-8.8, 9.7, 9.8."""

    protocol, vault, gate = _gate(tmp_path)
    decision = _decision()
    checkpoint = _checkpoint(decision)
    fitted = _fitted(protocol)
    observed: list[tuple[str, bool]] = []

    lease = gate.reserve(protocol.protocol_key, 1)
    assert vault.unlock_calls == []
    assert gate.budget.consumed_successful_evaluations == 0
    assert gate.reserved_evaluations == 1

    def predict(received_checkpoint, sample_id: str) -> dict[str, Any]:
        assert received_checkpoint is checkpoint
        assert vault.unlock_calls == []
        try:
            import torch
        except ImportError:
            gradients_enabled = False
        else:
            gradients_enabled = torch.is_grad_enabled()
        observed.append((sample_id, gradients_enabled))
        return {
            "y_pred": 1 if sample_id == "official-a" else 0,
            "probability": 0.8 if sample_id == "official-a" else 0.2,
        }

    published = gate.evaluate(
        lease,
        outer_fold=_fold(),
        checkpoint=checkpoint,
        selection_decision=decision,
        fitted_artifacts=(fitted,),
        predictor=predict,
        artifact_id="prediction-outer-1",
        run_id="run-outer-1",
    )

    assert observed == [("official-a", False), ("official-b", False)]
    assert len(vault.unlock_calls) == 1
    assert vault.unlock_calls[0][0] is lease
    assert published.artifact.complete is True
    assert tuple(record.y_true for record in published.artifact.records) == (1, 0)
    assert gate.budget.consumed_successful_evaluations == 1
    assert gate.reserved_evaluations == 0

    # Commit retries are safe and never consume a second successful evaluation.
    assert gate.commit_successful_evaluation(lease).consumed_successful_evaluations == 1
    with pytest.raises(EvaluationLeaseError):
        gate.evaluate(
            lease,
            outer_fold=_fold(),
            checkpoint=checkpoint,
            selection_decision=decision,
            fitted_artifacts=(fitted,),
            predictor=predict,
            artifact_id="prediction-outer-retry",
            run_id="run-outer-retry",
        )
    assert len(vault.unlock_calls) == 1
    assert gate.budget.consumed_successful_evaluations == 1


def test_failure_cancels_lease_without_labels_or_budget_consumption(
    tmp_path: Path,
) -> None:
    """Validates Requirements 5.19-5.21 through a synthetic failed inference."""

    protocol, vault, gate = _gate(tmp_path, maximum=2)
    decision = _decision()
    lease = gate.reserve(protocol.protocol_key)

    def fail_before_labels(checkpoint, sample_id):
        raise RuntimeError("synthetic inference failure")

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        gate.evaluate(
            lease,
            outer_fold=_fold(),
            checkpoint=_checkpoint(decision),
            selection_decision=decision,
            fitted_artifacts=(_fitted(protocol),),
            predictor=fail_before_labels,
            artifact_id="prediction-failed",
            run_id="run-failed",
        )

    assert vault.unlock_calls == []
    assert gate.budget.consumed_successful_evaluations == 0
    assert gate.reserved_evaluations == 0
    assert not (tmp_path / "synthetic artifacts" / "predictions").exists()


def test_exhausted_budget_rejects_before_vault_access_and_preserves_prediction(
    tmp_path: Path,
) -> None:
    """Validates Requirements 5.22-5.24 without reading any real labels."""

    protocol, vault, gate = _gate(tmp_path)
    decision = _decision()
    first = gate.reserve(protocol.protocol_key)
    gate.evaluate(
        first,
        outer_fold=_fold(),
        checkpoint=_checkpoint(decision),
        selection_decision=decision,
        fitted_artifacts=(_fitted(protocol),),
        predictor=lambda checkpoint, sample_id: {
            "y_pred": int(sample_id == "official-a"),
            "probability": 0.5,
        },
        artifact_id="prediction-complete",
        run_id="run-complete",
        relative_path="predictions/completed.json",
    )
    completed = tmp_path / "synthetic artifacts" / "predictions" / "completed.json"
    before = completed.read_bytes()
    unlock_count = len(vault.unlock_calls)

    with pytest.raises(EvaluationBudgetExceeded) as captured:
        gate.reserve(protocol.protocol_key)

    assert captured.value.error_record.error_code == "EVALUATION_BUDGET_EXCEEDED"
    assert captured.value.error_record.affected_identifiers["protocol_key"] == protocol.protocol_key
    assert captured.value.error_record.observed_value == {
        "consumed_successful_evaluations": 1,
        "reserved_evaluations": 0,
        "requested_count": 1,
    }
    assert len(vault.unlock_calls) == unlock_count
    assert completed.read_bytes() == before
    assert gate.budget.consumed_successful_evaluations == 1


def test_mutated_selection_is_rejected_before_outer_labels_become_accessible(
    tmp_path: Path,
) -> None:
    """Validates Requirements 5.13 and 5.18: freeze precedes label access."""

    protocol, vault, gate = _gate(tmp_path)
    decision = _decision()
    checkpoint = _checkpoint(decision)
    decision.candidate_scores["candidate-a"] = 0.25
    lease = gate.reserve(protocol.protocol_key)

    with pytest.raises(OuterEvaluationError) as captured:
        gate.evaluate(
            lease,
            outer_fold=_fold(),
            checkpoint=checkpoint,
            selection_decision=decision,
            fitted_artifacts=(_fitted(protocol),),
            predictor=lambda checkpoint, sample_id: {
                "y_pred": 0,
                "probability": 0.5,
            },
            artifact_id="prediction-unfrozen-selection",
            run_id="run-unfrozen-selection",
        )

    assert captured.value.error_record.error_code == "SELECTION_DECISION_NOT_FROZEN"
    assert vault.unlock_calls == []
    assert gate.budget.consumed_successful_evaluations == 0
    assert gate.reserved_evaluations == 0
    assert not (tmp_path / "synthetic artifacts" / "predictions").exists()
