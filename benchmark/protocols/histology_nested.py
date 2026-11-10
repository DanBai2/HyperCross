"""Deterministic patient-isolated planning for ``hist.patient-nested5.v1``."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from benchmark.protocols.planner import ProtocolPlanner, PublishedProtocol
from benchmark.schema import (
    EvaluationBudget,
    InnerFold,
    MetricDirection,
    OuterFold,
    SampleManifest,
    SelectionPolicy,
)


HISTOLOGY_NESTED5_PROTOCOL_KEY = "hist.patient-nested5.v1"
HISTOLOGY_NESTED5_OUTER_FOLDS = 5
HISTOLOGY_NESTED5_INNER_FOLDS = 5
HISTOLOGY_NESTED5_SELECTION_POLICY = SelectionPolicy(
    policy_key="hist.inner-patient.config-epoch.v1",
    objective_metric="balanced_accuracy",
    metric_direction=MetricDirection.MAXIMIZE,
    tie_break_rule="candidate_id_utf8_ascending",
)


class HistologyNestedPlanningError(ValueError):
    """The sample manifest cannot support the frozen nested-patient protocol."""


@dataclass(frozen=True)
class PatientPartition:
    """Canonical patient groups assigned to one validation/test partition."""

    group_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]


class HistologyNested5Planner:
    """Build five deterministic outer folds and five inner folds per outer fold.

    Patients are indivisible assignment units. Assignment uses canonical patient
    IDs and a deterministic least-loaded bin rule, so record enumeration order
    cannot change memberships or the resulting Protocol_Hash.
    """

    def __init__(self, planner: ProtocolPlanner) -> None:
        if not isinstance(planner, ProtocolPlanner):
            raise TypeError("planner must be a ProtocolPlanner")
        self.planner = planner

    def build_folds(self, sample_manifest: SampleManifest) -> tuple[OuterFold, ...]:
        """Create model-independent patient-isolated nested fold memberships."""

        groups = self._samples_by_group(sample_manifest)
        if len(groups) < HISTOLOGY_NESTED5_OUTER_FOLDS:
            raise HistologyNestedPlanningError(
                "hist.patient-nested5.v1 requires at least five patient groups"
            )

        all_ids = tuple(sample_manifest.canonical_order)
        all_id_set = set(all_ids)
        outer_partitions = self._balanced_group_partitions(
            groups, HISTOLOGY_NESTED5_OUTER_FOLDS
        )
        folds: list[OuterFold] = []
        for outer_index, test_partition in enumerate(outer_partitions, start=1):
            outer_fold_id = f"outer-{outer_index:02d}"
            test_ids = test_partition.sample_ids
            train_ids = self._ordered(all_id_set - set(test_ids))
            outer_groups = {
                group_id: sample_ids
                for group_id, sample_ids in groups.items()
                if group_id not in set(test_partition.group_ids)
            }
            if len(outer_groups) < HISTOLOGY_NESTED5_INNER_FOLDS:
                raise HistologyNestedPlanningError(
                    f"{outer_fold_id} has fewer than five outer-train patient groups"
                )
            inner_partitions = self._balanced_group_partitions(
                outer_groups, HISTOLOGY_NESTED5_INNER_FOLDS
            )
            inner_folds = tuple(
                InnerFold(
                    fold_id=f"{outer_fold_id}.inner-{inner_index:02d}",
                    train_ids=self._ordered(set(train_ids) - set(validation.sample_ids)),
                    validation_ids=validation.sample_ids,
                )
                for inner_index, validation in enumerate(inner_partitions, start=1)
            )
            folds.append(
                OuterFold(
                    fold_id=outer_fold_id,
                    outer_train_ids=train_ids,
                    outer_test_ids=test_ids,
                    inner_folds=inner_folds,
                )
            )
        return tuple(folds)

    def plan(
        self,
        sample_manifest: SampleManifest,
        *,
        seed_set: Iterable[int],
        schema_version: str = "1.0",
    ):
        """Create and validate the immutable protocol without publishing it."""

        seeds = tuple(seed_set)
        folds = self.build_folds(sample_manifest)
        return self.planner.plan(
            sample_manifest,
            schema_version=schema_version,
            protocol_key=HISTOLOGY_NESTED5_PROTOCOL_KEY,
            folds=folds,
            selection_policy=HISTOLOGY_NESTED5_SELECTION_POLICY,
            seed_set=seeds,
            evaluation_budget=EvaluationBudget(
                maximum_successful_evaluations=len(folds) * len(seeds)
            ),
        )

    def plan_and_publish(
        self,
        sample_manifest: SampleManifest,
        *,
        seed_set: Iterable[int],
        schema_version: str = "1.0",
        relative_path: Optional[str] = None,
    ) -> PublishedProtocol:
        """Validate and atomically publish before any model construction."""

        seeds = tuple(seed_set)
        folds = self.build_folds(sample_manifest)
        return self.planner.plan_and_publish(
            sample_manifest,
            schema_version=schema_version,
            protocol_key=HISTOLOGY_NESTED5_PROTOCOL_KEY,
            folds=folds,
            selection_policy=HISTOLOGY_NESTED5_SELECTION_POLICY,
            seed_set=seeds,
            evaluation_budget=EvaluationBudget(
                maximum_successful_evaluations=len(folds) * len(seeds)
            ),
            relative_path=relative_path,
        )

    @classmethod
    def _samples_by_group(
        cls, sample_manifest: SampleManifest
    ) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        observed_ids: set[str] = set()
        for record in sample_manifest.records:
            if record.group_id is None or not record.group_id.strip():
                raise HistologyNestedPlanningError(
                    f"sample {record.sample_id!r} has no patient Group_ID"
                )
            if record.sample_id in observed_ids:
                raise HistologyNestedPlanningError(
                    f"duplicate Sample_ID in manifest: {record.sample_id}"
                )
            observed_ids.add(record.sample_id)
            grouped[record.group_id].append(record.sample_id)
        if observed_ids != set(sample_manifest.canonical_order):
            raise HistologyNestedPlanningError(
                "manifest records and Canonical_Order contain different Sample_ID sets"
            )
        return {
            group_id: cls._ordered(sample_ids)
            for group_id, sample_ids in sorted(
                grouped.items(), key=lambda item: item[0].encode("utf-8")
            )
        }

    @classmethod
    def _balanced_group_partitions(
        cls,
        groups: Mapping[str, tuple[str, ...]],
        partition_count: int,
    ) -> tuple[PatientPartition, ...]:
        if partition_count <= 0 or len(groups) < partition_count:
            raise HistologyNestedPlanningError(
                "patient partition count must be positive and no larger than group count"
            )
        bins: list[list[str]] = [[] for _ in range(partition_count)]
        loads = [0] * partition_count
        ordered_groups = sorted(
            groups,
            key=lambda group_id: (-len(groups[group_id]), group_id.encode("utf-8")),
        )
        for group_id in ordered_groups:
            destination = min(range(partition_count), key=lambda index: (loads[index], index))
            bins[destination].append(group_id)
            loads[destination] += len(groups[group_id])
        return tuple(
            PatientPartition(
                group_ids=tuple(sorted(group_ids, key=lambda value: value.encode("utf-8"))),
                sample_ids=cls._ordered(
                    sample_id
                    for group_id in group_ids
                    for sample_id in groups[group_id]
                ),
            )
            for group_ids in bins
        )

    @staticmethod
    def _ordered(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


__all__ = [
    "HISTOLOGY_NESTED5_INNER_FOLDS",
    "HISTOLOGY_NESTED5_OUTER_FOLDS",
    "HISTOLOGY_NESTED5_PROTOCOL_KEY",
    "HISTOLOGY_NESTED5_SELECTION_POLICY",
    "HistologyNested5Planner",
    "HistologyNestedPlanningError",
    "PatientPartition",
]
