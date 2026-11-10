"""HyperLeaf nested leave-one-variety-out protocol planning.

The planner consumes only the governed training ``SampleManifest``.  It has no
runtime dataset adapter and therefore has no path through which it can open the
official-test partition or its labels.
"""

from __future__ import annotations

from typing import Iterable, Optional

from benchmark.catalog.hyperleaf import (
    HYPERLEAF_DATASET_KEY,
    HYPERLEAF_VARIETIES,
)
from benchmark.protocols.planner import ProtocolPlanner, PublishedProtocol
from benchmark.schema.base import MetricDirection
from benchmark.schema.datasets import SampleManifest
from benchmark.schema.protocol import (
    EvaluationBudget,
    InnerFold,
    OuterFold,
    ProtocolManifest,
    SelectionPolicy,
)
from benchmark.schema.records import ErrorRecord


HYPERLEAF_LOVO_PROTOCOL_KEY = "leaf.lovo.nested.v1"
HYPERLEAF_LOVO_SCHEMA_VERSION = "1.0"
HYPERLEAF_LOVO_SELECTION_POLICY = SelectionPolicy(
    policy_key="outer-train-inner-variety-mean-r2-v1",
    objective_metric="mean_r2",
    metric_direction=MetricDirection.MAXIMIZE,
    tie_break_rule="candidate_id_utf8_ascending",
)


class HyperLeafLovoPlanningError(ValueError):
    """The supplied manifest cannot define the strict HyperLeaf LOVO protocol."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        super().__init__(
            f"HyperLeaf LOVO planning failed: {error_record.error_code} "
            f"({error_record.failing_field})"
        )


def _ordered(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


def _reject(
    planner: ProtocolPlanner,
    *,
    code: str,
    field: str,
    expected: object,
    observed: object,
    missing_ids: tuple[str, ...] = (),
    unexpected_ids: tuple[str, ...] = (),
) -> None:
    record = ErrorRecord(
        error_code=code,
        failed_stage="hyperleaf_lovo_planning",
        affected_identifiers={"protocol_key": HYPERLEAF_LOVO_PROTOCOL_KEY},
        failing_field=field,
        expected_value=expected,
        observed_value=observed,
        missing_ids=missing_ids,
        unexpected_ids=unexpected_ids,
        invariant_checks={
            "protocol_artifact_published": False,
            "official_test_accessed": False,
            "completed_hyperleaf_artifacts_unchanged": True,
        },
    )
    planner.error_journal.append(record)
    raise HyperLeafLovoPlanningError(record)


def build_hyperleaf_lovo_folds(
    sample_manifest: SampleManifest,
    *,
    planner: ProtocolPlanner,
) -> tuple[OuterFold, ...]:
    """Create one outer test fold per complete wheat variety.

    Every remaining outer-train variety also appears exactly once as an inner
    validation variety.  Consequently route and epoch selection can aggregate
    complete inner OOF evidence without referencing the held-out variety.
    """

    if not isinstance(sample_manifest, SampleManifest):
        raise TypeError("sample_manifest must be a SampleManifest")
    if sample_manifest.dataset_key != HYPERLEAF_DATASET_KEY:
        _reject(
            planner,
            code="HYPERLEAF_LOVO_DATASET_MISMATCH",
            field="dataset_key",
            expected=HYPERLEAF_DATASET_KEY,
            observed=sample_manifest.dataset_key,
        )

    samples_by_variety: dict[str, list[str]] = {
        variety: [] for variety in HYPERLEAF_VARIETIES
    }
    unexpected_groups: set[str] = set()
    unsafe_samples: list[str] = []
    for record in sample_manifest.records:
        metadata = record.label_schema_metadata
        if (
            metadata.get("partition") == "official_test"
            or metadata.get("labels_available") is False
            or str(record.label_reference).startswith("withheld://")
        ):
            unsafe_samples.append(record.sample_id)
        if record.group_id in samples_by_variety:
            samples_by_variety[record.group_id].append(record.sample_id)
        else:
            unexpected_groups.add(record.group_id or "<missing>")

    if unsafe_samples:
        ordered = _ordered(unsafe_samples)
        _reject(
            planner,
            code="HYPERLEAF_LOVO_OFFICIAL_TEST_FORBIDDEN",
            field="records.label_schema_metadata.partition",
            expected="training records only",
            observed=list(ordered),
            unexpected_ids=ordered,
        )

    missing_varieties = tuple(
        variety for variety in HYPERLEAF_VARIETIES if not samples_by_variety[variety]
    )
    if missing_varieties or unexpected_groups:
        unexpected = _ordered(unexpected_groups)
        _reject(
            planner,
            code="HYPERLEAF_LOVO_VARIETY_SET_INVALID",
            field="records.group_id",
            expected=list(HYPERLEAF_VARIETIES),
            observed={
                "missing_varieties": list(missing_varieties),
                "unexpected_varieties": list(unexpected),
            },
            missing_ids=missing_varieties,
            unexpected_ids=unexpected,
        )

    all_ids = set(sample_manifest.canonical_order)
    folds: list[OuterFold] = []
    for held_out in HYPERLEAF_VARIETIES:
        outer_test_ids = _ordered(samples_by_variety[held_out])
        outer_train_ids = _ordered(all_ids - set(outer_test_ids))
        inner_folds = []
        for validation_variety in HYPERLEAF_VARIETIES:
            if validation_variety == held_out:
                continue
            validation_ids = _ordered(samples_by_variety[validation_variety])
            train_ids = _ordered(set(outer_train_ids) - set(validation_ids))
            inner_folds.append(
                InnerFold(
                    fold_id=f"inner-{validation_variety.casefold()}",
                    train_ids=train_ids,
                    validation_ids=validation_ids,
                )
            )
        folds.append(
            OuterFold(
                fold_id=f"outer-{held_out.casefold()}",
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,
                inner_folds=tuple(inner_folds),
            )
        )

    planned = tuple(folds)
    outer_test_groups = [
        {
            record.group_id
            for record in sample_manifest.records
            if record.sample_id in fold.outer_test_ids
        }
        for fold in planned
    ]
    if (
        any(len(groups) != 1 for groups in outer_test_groups)
        or len({next(iter(groups)) for groups in outer_test_groups}) != len(planned)
        or set().union(*outer_test_groups) != set(HYPERLEAF_VARIETIES)
    ):
        _reject(
            planner,
            code="HYPERLEAF_LOVO_OUTER_COVERAGE_INVALID",
            field="folds.outer_test_ids",
            expected="each complete wheat variety exactly once",
            observed=[sorted(groups) for groups in outer_test_groups],
        )
    return planned


def plan_hyperleaf_lovo(
    planner: ProtocolPlanner,
    sample_manifest: SampleManifest,
    *,
    seed_set: Iterable[int] = (7, 42, 123),
    publish: bool = True,
    relative_path: Optional[str] = None,
) -> ProtocolManifest | PublishedProtocol:
    """Plan and optionally publish the immutable strict nested LOVO manifest."""

    seeds = tuple(seed_set)
    folds = build_hyperleaf_lovo_folds(sample_manifest, planner=planner)
    budget = EvaluationBudget(
        maximum_successful_evaluations=len(folds) * len(seeds)
    )
    kwargs = dict(
        schema_version=HYPERLEAF_LOVO_SCHEMA_VERSION,
        protocol_key=HYPERLEAF_LOVO_PROTOCOL_KEY,
        folds=folds,
        selection_policy=HYPERLEAF_LOVO_SELECTION_POLICY,
        seed_set=seeds,
        evaluation_budget=budget,
    )
    if publish:
        return planner.plan_and_publish(
            sample_manifest,
            relative_path=relative_path,
            **kwargs,
        )
    return planner.plan(sample_manifest, **kwargs)


__all__ = [
    "HYPERLEAF_LOVO_PROTOCOL_KEY",
    "HYPERLEAF_LOVO_SCHEMA_VERSION",
    "HYPERLEAF_LOVO_SELECTION_POLICY",
    "HyperLeafLovoPlanningError",
    "build_hyperleaf_lovo_folds",
    "plan_hyperleaf_lovo",
]
