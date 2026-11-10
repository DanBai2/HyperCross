"""Property 4: protocol-before-model and canonical round trip.

**Validates: Requirements 3.1, 3.2, 3.19-3.21**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from hypothesis import given, settings, strategies as st

from benchmark.protocols import ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    EvaluationBudget,
    InnerFold,
    MetricDirection,
    OuterFold,
    ProtocolManifest,
    SampleManifest,
    SampleRecord,
    SelectionPolicy,
    canonical_sample_order,
    canonical_serialize,
)


_DATASET_HASH = "a" * 64
_INPUT_HASH = "b" * 64
_GROUPS = ("patient-a", "patient-b", "patient-c", "patient-d")
_SAMPLES_BY_GROUP = {
    group_id: (f"sample-{index}-1", f"sample-{index}-2")
    for index, group_id in enumerate(_GROUPS, start=1)
}


@dataclass(frozen=True)
class _EquivalentFoldOrder:
    folds: tuple[dict[str, Any], ...]
    seed_set: tuple[int, ...]


def _sample_manifest() -> SampleManifest:
    records = tuple(
        SampleRecord(
            sample_id=sample_id,
            group_id=group_id,
            label_reference=f"labels/{sample_id}.json",
            input_references={"cube": f"cubes/{sample_id}.tiff"},
            input_content_hashes={"cube": _INPUT_HASH},
            label_schema_metadata={"dimension": 1},
        )
        for group_id in _GROUPS
        for sample_id in _SAMPLES_BY_GROUP[group_id]
    )
    return SampleManifest(
        schema_version="1",
        dataset_key="property4.synthetic-histology",
        records=records,
        canonical_order=canonical_sample_order(
            tuple(record.sample_id for record in records)
        ),
        dataset_hash=_DATASET_HASH,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_key="patient-inner-v1",
        objective_metric="balanced_accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
    )


def _canonical_fold_payloads() -> tuple[dict[str, Any], ...]:
    folds: list[dict[str, Any]] = []
    for held_out_group in _GROUPS:
        training_groups = tuple(
            group_id for group_id in _GROUPS if group_id != held_out_group
        )
        outer_train_ids = tuple(
            sample_id
            for group_id in training_groups
            for sample_id in _SAMPLES_BY_GROUP[group_id]
        )
        inner_folds = tuple(
            {
                "fold_id": f"inner-{held_out_group}-{validation_group}",
                "train_ids": tuple(
                    sample_id
                    for group_id in training_groups
                    if group_id != validation_group
                    for sample_id in _SAMPLES_BY_GROUP[group_id]
                ),
                "validation_ids": _SAMPLES_BY_GROUP[validation_group],
            }
            for validation_group in training_groups
        )
        folds.append(
            {
                "fold_id": f"outer-{held_out_group}",
                "outer_train_ids": outer_train_ids,
                "outer_test_ids": _SAMPLES_BY_GROUP[held_out_group],
                "inner_folds": inner_folds,
            }
        )
    return tuple(folds)


@st.composite
def _equivalent_fold_orders(draw: st.DrawFn) -> _EquivalentFoldOrder:
    permuted_folds: list[dict[str, Any]] = []
    for fold in _canonical_fold_payloads():
        permuted_inner_folds: list[dict[str, Any]] = []
        for inner_fold in fold["inner_folds"]:
            permuted_inner_folds.append(
                {
                    "fold_id": inner_fold["fold_id"],
                    "train_ids": draw(st.permutations(inner_fold["train_ids"])),
                    "validation_ids": draw(
                        st.permutations(inner_fold["validation_ids"])
                    ),
                }
            )
        permuted_folds.append(
            {
                "fold_id": fold["fold_id"],
                "outer_train_ids": draw(st.permutations(fold["outer_train_ids"])),
                "outer_test_ids": draw(st.permutations(fold["outer_test_ids"])),
                "inner_folds": draw(st.permutations(tuple(permuted_inner_folds))),
            }
        )

    return _EquivalentFoldOrder(
        folds=draw(st.permutations(tuple(permuted_folds))),
        seed_set=draw(st.permutations((7, 42, 123))),
    )


def _make_folds(payloads: tuple[dict[str, Any], ...]) -> tuple[OuterFold, ...]:
    return tuple(
        OuterFold(
            fold_id=payload["fold_id"],
            outer_train_ids=payload["outer_train_ids"],
            outer_test_ids=payload["outer_test_ids"],
            inner_folds=tuple(
                InnerFold.model_validate(inner_payload)
                for inner_payload in payload["inner_folds"]
            ),
        )
        for payload in payloads
    )


def _protocol(folds: tuple[OuterFold, ...], seed_set: tuple[int, ...]) -> ProtocolManifest:
    return ProtocolManifest.create(
        schema_version="1",
        protocol_key="hist.patient-nested.property4.v1",
        dataset_hash=_DATASET_HASH,
        folds=folds,
        selection_policy=_policy(),
        seed_set=seed_set,
        evaluation_budget=EvaluationBudget(maximum_successful_evaluations=1),
    )


@settings(max_examples=100, deadline=None)
@given(order=_equivalent_fold_orders())
def test_equivalent_fold_orders_publish_before_build_and_round_trip_stably(
    order: _EquivalentFoldOrder,
) -> None:
    """Every equivalent order has one canonical identity before model build."""

    reference = _protocol(
        _make_folds(_canonical_fold_payloads()),
        (7, 42, 123),
    )
    candidate_folds = _make_folds(order.folds)

    with TemporaryDirectory(prefix="property4-") as directory:
        artifact_root = Path(directory) / "artifacts"
        planner = ProtocolPlanner(FailureAtomicArtifactStore(artifact_root))
        build_observations: list[tuple[str, bytes]] = []

        def build_model(protocol: ProtocolManifest) -> str:
            artifact_path = artifact_root / Path(
                f"protocols/{protocol.protocol_hash}.canonical.json"
            )
            assert artifact_path.is_file()
            published_bytes = artifact_path.read_bytes()
            reloaded_during_build = planner.reload(published_bytes)
            assert reloaded_during_build == protocol
            assert reloaded_during_build.protocol_hash == protocol.protocol_hash
            build_observations.append((protocol.protocol_hash, published_bytes))
            return "built-after-publication"

        published, model = planner.plan_publish_and_build(
            _sample_manifest(),
            model_builder=build_model,
            schema_version="1",
            protocol_key="hist.patient-nested.property4.v1",
            folds=candidate_folds,
            selection_policy=_policy(),
            seed_set=order.seed_set,
            evaluation_budget=EvaluationBudget(
                maximum_successful_evaluations=1
            ),
        )

        artifact_path = artifact_root / Path(
            *published.artifact.relative_path.split("/")
        )
        artifact_bytes = artifact_path.read_bytes()
        reloaded = planner.reload(artifact_bytes)

        assert model == "built-after-publication"
        assert build_observations == [(published.manifest.protocol_hash, artifact_bytes)]
        assert published.manifest == reference
        assert published.manifest.folds == reference.folds
        assert published.manifest.protocol_hash == reference.protocol_hash
        assert canonical_serialize(published.manifest) == canonical_serialize(reference)
        assert reloaded == published.manifest
        assert reloaded.folds == reference.folds
        assert reloaded.protocol_hash == reference.protocol_hash
        assert canonical_serialize(reloaded) == artifact_bytes
