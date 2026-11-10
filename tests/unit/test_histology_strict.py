from __future__ import annotations

from benchmark.runners.histology_strict import choose_outer_decisions, freeze_lopo_policy


def test_outer_decisions_are_independent_and_tie_break_by_epoch_then_candidate() -> None:
    evidence = {
        "candidate-b": {
            "outer-01": {"score": 0.8, "epoch": 8, "source_prediction_hash": "b1", "source_sample_ids": ("P1/a",)},
            "outer-02": {"score": 0.7, "epoch": 4, "source_prediction_hash": "b2", "source_sample_ids": ("P2/a",)},
        },
        "candidate-a": {
            "outer-01": {"score": 0.8, "epoch": 5, "source_prediction_hash": "a1", "source_sample_ids": ("P1/a",)},
            "outer-02": {"score": 0.9, "epoch": 9, "source_prediction_hash": "a2", "source_sample_ids": ("P2/a",)},
        },
    }

    decisions = choose_outer_decisions(evidence)

    assert decisions["outer-01"]["selected_candidate"] == "candidate-a"
    assert decisions["outer-01"]["selected_epoch"] == 5
    assert decisions["outer-01"]["source_sample_ids"] == ("P1/a",)
    assert decisions["outer-02"]["selected_candidate"] == "candidate-a"
    assert decisions["outer-02"]["selected_epoch"] == 9


def test_lopo_policy_uses_only_nested_inner_scores_and_median_selected_epoch() -> None:
    decisions = {
        "outer-01": {"selected_epoch": 3, "candidate_scores": {"a": 0.7, "b": 0.8}},
        "outer-02": {"selected_epoch": 9, "candidate_scores": {"a": 0.9, "b": 0.7}},
        "outer-03": {"selected_epoch": 5, "candidate_scores": {"a": 0.8, "b": 0.6}},
        "outer-04": {"selected_epoch": 7, "candidate_scores": {"a": 0.8, "b": 0.7}},
        "outer-05": {"selected_epoch": 1, "candidate_scores": {"a": 0.9, "b": 0.8}},
    }

    policy = freeze_lopo_policy(decisions)

    assert policy["selected_candidate"] == "a"
    assert policy["fixed_epoch"] == 5
    assert policy["outer_labels_used"] is False
    assert len(policy["policy_hash"]) == 64
