"""Targeted unit coverage for task 3.1 protocol schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark.schema import (
    EvaluationBudget,
    InnerFold,
    MetricDirection,
    OuterFold,
    ProtocolManifest,
    SelectionPolicy,
)


DATASET_HASH = "a" * 64


def _selection_policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="grouped-inner-r2-v1",
        objective_metric="mean_r2",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def _outer_fold(fold_id: str = "outer-01") -> OuterFold:
    return OuterFold(
        fold_id=fold_id,
        outer_train_ids=("sample\\Café", "sample-B", "sample-A"),
        outer_test_ids=("sample-test-2", "sample-test-1"),
        inner_folds=(
            InnerFold(
                fold_id="inner-02",
                train_ids=("sample-B", "sample-A"),
                validation_ids=("sample\\Cafe\u0301",),
            ),
            InnerFold(
                fold_id="inner-01",
                train_ids=("sample\\Cafe\u0301", "sample-A"),
                validation_ids=("sample-B",),
            ),
        ),
    )


def test_fold_sample_sets_are_normalized_sorted_and_immutable() -> None:
    fold = _outer_fold()

    assert fold.outer_train_ids == ("sample-A", "sample-B", "sample/Café")
    assert fold.outer_test_ids == ("sample-test-1", "sample-test-2")
    assert tuple(inner.fold_id for inner in fold.inner_folds) == (
        "inner-01",
        "inner-02",
    )
    assert fold.inner_folds[0].train_ids == ("sample-A", "sample/Café")

    with pytest.raises(ValidationError, match="frozen"):
        fold.outer_test_ids = ("replacement",)


def test_fold_rejects_empty_duplicate_and_scalar_sample_sets() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        InnerFold(fold_id="inner", train_ids=(), validation_ids=("sample-1",))

    with pytest.raises(ValidationError, match="duplicate identifiers"):
        InnerFold(
            fold_id="inner",
            train_ids=("sample\\Cafe\u0301", "sample/Café"),
            validation_ids=("sample-1",),
        )

    with pytest.raises(ValidationError, match="collection of Sample_IDs"):
        OuterFold(
            fold_id="outer",
            outer_train_ids="sample-1",
            outer_test_ids=("sample-2",),
        )


def test_successful_evaluation_budget_tracks_only_valid_committed_counts() -> None:
    budget = EvaluationBudget(
        maximum_successful_evaluations=2,
        consumed_successful_evaluations=1,
    )
    assert budget.remaining_successful_evaluations == 1

    for values in (
        {"maximum_successful_evaluations": 0},
        {"maximum_successful_evaluations": True},
        {
            "maximum_successful_evaluations": 1,
            "consumed_successful_evaluations": 2,
        },
        {
            "maximum_successful_evaluations": 1,
            "consumed_successful_evaluations": -1,
        },
    ):
        with pytest.raises(ValidationError):
            EvaluationBudget.model_validate(values)


def test_protocol_manifest_canonicalizes_folds_and_seed_set_and_binds_hash() -> None:
    first = ProtocolManifest.create(
        schema_version="1",
        protocol_key="leaf.official.v1",
        dataset_hash=DATASET_HASH,
        folds=(_outer_fold("outer-02"), _outer_fold("outer-01")),
        selection_policy=_selection_policy(),
        seed_set=(42, -7, 7),
        evaluation_budget={"maximum_successful_evaluations": 1},
    )
    second = ProtocolManifest.create(
        schema_version="1",
        protocol_key="leaf.official.v1",
        dataset_hash=DATASET_HASH,
        folds=(_outer_fold("outer-01"), _outer_fold("outer-02")),
        selection_policy=_selection_policy(),
        seed_set=(7, 42, -7),
        evaluation_budget={"maximum_successful_evaluations": 1},
    )

    assert first.seed_set == (-7, 7, 42)
    assert tuple(fold.fold_id for fold in first.folds) == ("outer-01", "outer-02")
    assert first.protocol_hash == second.protocol_hash
    assert first == second

    tampered = first.model_dump(mode="python")
    tampered["protocol_key"] = "leaf.changed.v1"
    with pytest.raises(ValidationError, match="protocol_hash does not match"):
        ProtocolManifest.model_validate(tampered)


def test_protocol_manifest_rejects_duplicate_or_non_integer_seeds_and_fold_ids() -> None:
    common = {
        "schema_version": "1",
        "protocol_key": "hist.lopo-fixed.v1",
        "dataset_hash": DATASET_HASH,
        "selection_policy": _selection_policy(),
        "evaluation_budget": {"maximum_successful_evaluations": 1},
    }

    with pytest.raises(ValidationError, match="duplicate seeds"):
        ProtocolManifest.create(**common, folds=(_outer_fold(),), seed_set=(7, 7))

    with pytest.raises(ValidationError, match="strict integers"):
        ProtocolManifest.create(**common, folds=(_outer_fold(),), seed_set=(7, True))

    with pytest.raises(ValidationError, match="duplicate Fold_IDs"):
        ProtocolManifest.create(
            **common,
            folds=(_outer_fold("outer-01"), _outer_fold("outer-01")),
            seed_set=(7,),
        )


def test_protocol_models_reject_unknown_fields_and_invalid_selection_direction() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationBudget.model_validate(
            {"maximum_successful_evaluations": 1, "requested": 1}
        )

    with pytest.raises(ValidationError):
        SelectionPolicy.model_validate(
            {
                "policy_key": "policy-v1",
                "objective_metric": "accuracy",
                "metric_direction": "sideways",
                "tie_break_rule": "candidate_id_utf8_ascending",
            }
        )
