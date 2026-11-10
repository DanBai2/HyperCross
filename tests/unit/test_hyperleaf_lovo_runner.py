"""Unit coverage for task 14.3 HyperLeaf nested LOVO planning and wiring.

Validates Requirements 8.14-8.23 and 8.28-8.30.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS, HYPERLEAF_VARIETIES
from benchmark.models.hyperleaf_route_a import GLOBAL_FUSION_MODEL_KEY, TOKEN_MODEL_KEY
from benchmark.protocols import (
    HYPERLEAF_LOVO_PROTOCOL_KEY,
    HyperLeafLovoPlanningError,
    ProtocolPlanner,
    ProtocolReloadError,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.runners.hyperleaf_lovo import (
    HyperLeafLovoRunner,
    HyperLeafLovoRunnerError,
    LovoRouteEpochOption,
)
from benchmark.schema import (
    EnvironmentLock,
    PredictionKey,
    PredictionRecord,
    SampleManifest,
    SampleRecord,
    canonical_sample_order,
    canonical_serialize,
    hash_object,
)


NOW = datetime(2027, 1, 1, tzinfo=timezone.utc)
INPUT_HASH = "a" * 64


def _manifest(*, official: bool = False) -> SampleManifest:
    records = tuple(
        SampleRecord(
            sample_id=f"{variety.casefold()}-{index}",
            group_id=variety,
            label_reference=(
                f"withheld://hyperleaf2024/official-test/{variety}-{index}"
                if official and variety == HYPERLEAF_VARIETIES[0]
                else f"labels://hyperleaf2024/train/{variety}-{index}"
            ),
            input_references={"full_cube": f"images/{variety}-{index}.tiff"},
            input_content_hashes={"full_cube": INPUT_HASH},
            label_schema_metadata={
                "dimension": 3,
                "partition": "official_test"
                if official and variety == HYPERLEAF_VARIETIES[0]
                else "train",
                "labels_available": not (official and variety == HYPERLEAF_VARIETIES[0]),
            },
        )
        for variety in HYPERLEAF_VARIETIES
        for index in range(2)
    )
    order = canonical_sample_order(record.sample_id for record in records)
    return SampleManifest(
        schema_version="1.0",
        dataset_key="hyperleaf2024",
        records=records,
        canonical_order=order,
        dataset_hash=hash_object(
            tuple((record.sample_id, record.group_id) for record in sorted(records, key=lambda item: item.sample_id))
        ),
        created_utc=NOW,
    )


def _runner() -> HyperLeafLovoRunner:
    return HyperLeafLovoRunner(r"G:\Datasets\HyperCross")


def test_planner_holds_out_every_complete_variety_once_with_nested_variety_folds(
    tmp_path: Path,
) -> None:
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    planner = ProtocolPlanner(store)
    published = _runner().plan_protocol(planner, _manifest())
    protocol = published.manifest

    assert protocol.protocol_key == HYPERLEAF_LOVO_PROTOCOL_KEY
    assert len(protocol.folds) == len(HYPERLEAF_VARIETIES)
    assert protocol.evaluation_budget.maximum_successful_evaluations == 12
    group_by_sample = {
        record.sample_id: record.group_id for record in _manifest().records
    }
    held_out = []
    for fold in protocol.folds:
        test_groups = {group_by_sample[sample_id] for sample_id in fold.outer_test_ids}
        assert len(test_groups) == 1
        held_out.extend(test_groups)
        assert set(fold.outer_train_ids).isdisjoint(fold.outer_test_ids)
        assert set(fold.outer_train_ids) | set(fold.outer_test_ids) == set(
            _manifest().canonical_order
        )
        assert len(fold.inner_folds) == 3
        inner_validation_groups = []
        for inner in fold.inner_folds:
            train_groups = {group_by_sample[sample_id] for sample_id in inner.train_ids}
            validation_groups = {
                group_by_sample[sample_id] for sample_id in inner.validation_ids
            }
            assert len(validation_groups) == 1
            assert train_groups.isdisjoint(validation_groups)
            assert set(inner.train_ids) | set(inner.validation_ids) == set(
                fold.outer_train_ids
            )
            inner_validation_groups.extend(validation_groups)
        assert set(inner_validation_groups) == set(HYPERLEAF_VARIETIES) - test_groups

    assert held_out == list(HYPERLEAF_VARIETIES)
    assert published.artifact.relative_path.startswith(
        "runs_v2/strict/hyperleaf/lovo/protocols/"
    )
    content = (store.root / published.artifact.relative_path).read_bytes()
    assert ProtocolPlanner.reload(content) == protocol
    assert canonical_serialize(protocol) == content
    with pytest.raises(ProtocolReloadError):
        ProtocolPlanner.reload(content + b" ")


def test_planner_rejects_official_test_records_before_publication(tmp_path: Path) -> None:
    store = FailureAtomicArtifactStore(tmp_path / "artifacts")
    planner = ProtocolPlanner(store)

    with pytest.raises(HyperLeafLovoPlanningError) as caught:
        _runner().plan_protocol(planner, _manifest(official=True))

    assert caught.value.error_record.error_code == "HYPERLEAF_LOVO_OFFICIAL_TEST_FORBIDDEN"
    assert caught.value.error_record.invariant_checks["official_test_accessed"] is False
    assert not tuple(store.root.glob("runs_v2/**/*.json"))


def test_route_and_epoch_selection_requires_complete_inner_oof_only(tmp_path: Path) -> None:
    planner = ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts"))
    protocol = _runner().plan_protocol(planner, _manifest()).manifest
    fold = protocol.folds[0]
    runner = _runner()
    options = (
        LovoRouteEpochOption(GLOBAL_FUSION_MODEL_KEY, 10),
        LovoRouteEpochOption(TOKEN_MODEL_KEY, 20),
    )
    candidates = tuple(runner.selection_candidate(option) for option in options)
    decision = runner.freeze_route_and_epoch(
        fold,
        options=options,
        candidate_scores={candidates[0].candidate_id: 0.2, candidates[1].candidate_id: 0.3},
        inner_source_sample_ids=fold.outer_train_ids,
        inner_prediction_hash="b" * 64,
        decision_id="outer-variety-route-epoch",
        frozen_utc=NOW,
    )

    assert decision.fold_id == fold.fold_id
    assert decision.source_sample_ids == fold.outer_train_ids
    assert decision.selected_candidate == candidates[1].candidate_id
    assert decision.candidates[1].configuration["epoch"] == 20

    with pytest.raises(HyperLeafLovoRunnerError, match="complete inner-variety OOF"):
        runner.freeze_route_and_epoch(
            fold,
            options=options,
            candidate_scores={candidates[0].candidate_id: 0.2, candidates[1].candidate_id: 0.3},
            inner_source_sample_ids=fold.outer_train_ids[:-1],
            inner_prediction_hash="c" * 64,
        )
    with pytest.raises(HyperLeafLovoRunnerError, match="complete inner-variety OOF"):
        runner.freeze_route_and_epoch(
            fold,
            options=options,
            candidate_scores={candidates[0].candidate_id: 0.2, candidates[1].candidate_id: 0.3},
            inner_source_sample_ids=fold.outer_train_ids + (fold.outer_test_ids[0],),
            inner_prediction_hash="d" * 64,
        )


def test_metric_wiring_declares_and_computes_all_lovo_output_families(tmp_path: Path) -> None:
    runner = _runner()
    protocol = runner.plan_protocol(
        ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts")),
        _manifest(),
        seed_set=(42,),
    ).manifest
    records = []
    counter = 0
    for fold in protocol.folds:
        for sample_id in fold.outer_test_ids:
            counter += 1
            truth = tuple(float(counter + index) for index in range(3))
            records.append(
                PredictionRecord(
                    key=PredictionKey(
                        sample_id=sample_id,
                        fold_id=fold.fold_id,
                        seed=42,
                        inference_branch="selected-route",
                    ),
                    y_true=truth,
                    y_pred=truth,
                )
            )

    metrics = runner.metric_computer(protocol).compute(
        tuple(records), runner.metric_schema()
    )
    policy = runner.aggregation_policy(protocol)

    assert policy.fold_ids == tuple(fold.fold_id for fold in protocol.folds)
    assert policy.seed_set == (42,)
    assert policy.variance_metric == "pooled_oof.mean_mae"
    for fold in protocol.folds:
        assert metrics[f"per_fold.{fold.fold_id}.mean_r2"] == pytest.approx(1.0)
    for target in HYPERLEAF_TARGETS:
        assert metrics[f"per_target.r2.{target}"] == pytest.approx(1.0)
        assert metrics[f"fold_macro.r2.{target}"] == pytest.approx(1.0)
        assert metrics[f"pooled_oof.r2.{target}"] == pytest.approx(1.0)
    assert metrics["fold_macro.mean_r2"] == pytest.approx(1.0)
    assert metrics["pooled_oof.mean_r2"] == pytest.approx(1.0)

    incomplete = tuple(records[:-1])
    with pytest.raises(HyperLeafLovoRunnerError, match="exact one-prediction OOF coverage"):
        runner.metric_computer(protocol).compute(incomplete, runner.metric_schema())


def test_runner_contract_is_relative_single_process_tcfm_rtx5080_and_train_only() -> None:
    runner = _runner()
    metadata = runner.metadata()
    environment = EnvironmentLock.create(
        python_version="3.10",
        pytorch_version="2.7",
        numpy_version="2.0",
        pandas_version="2.0",
        pillow_version="11",
        tifffile_version="2025",
        sklearn_version="1.6",
        scipy_version="1.15",
        cuda_version="12.8",
        cuda_available=True,
        cuda_device_count=1,
        device_identity="cuda:0:NVIDIA GeForce RTX 5080",
    )

    runner.validate_launch(environment, conda_environment="TCFM", process_count=1)
    assert not Path(metadata["output_root"]).is_absolute()
    assert metadata["official_test_access"] is False
    assert metadata["execution"]["dataset_partition"] == "train"
    assert metadata["execution"]["process_count"] == 1
    assert metadata["recovery_boundary"] == "outer_fold"

    with pytest.raises(HyperLeafLovoRunnerError, match="exactly G"):
        HyperLeafLovoRunner(r"F:\HyperCross")
    with pytest.raises(HyperLeafLovoRunnerError, match="execution contract"):
        runner.validate_launch(environment, conda_environment="base", process_count=2)
