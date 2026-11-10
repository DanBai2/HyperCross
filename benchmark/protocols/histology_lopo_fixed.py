"""Deterministic fixed-epoch leave-one-patient-out protocol planning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from benchmark.protocols.planner import ProtocolPlanner, PublishedProtocol
from benchmark.schema import (
    EvaluationBudget,
    MetricDirection,
    OuterFold,
    SampleManifest,
    SelectionPolicy,
    hash_object,
)


HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY = "hist.lopo-fixed.v1"
HISTOLOGY_LOPO_FIXED_DEFAULT_EPOCHS = 20


class HistologyLopoFixedPlanningError(ValueError):
    """The manifest or epoch policy cannot support strict fixed-epoch LOPO."""


@dataclass(frozen=True)
class HistologyFixedEpochPolicy:
    """A deterministic epoch policy frozen before any outer-patient access."""

    epochs: int = HISTOLOGY_LOPO_FIXED_DEFAULT_EPOCHS
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise HistologyLopoFixedPlanningError("epochs must be a positive integer")
        if self.policy_version != "v1":
            raise HistologyLopoFixedPlanningError("only fixed epoch policy version v1 is supported")

    @property
    def policy_key(self) -> str:
        """Encode the fixed epoch count into the protocol's hashed selection policy."""

        return f"hist.fixed-epoch-{self.epochs:04d}.{self.policy_version}"

    @property
    def policy_hash(self) -> str:
        return hash_object(
            {
                "epochs": self.epochs,
                "policy_key": self.policy_key,
                "policy_version": self.policy_version,
                "selection_source": "predeclared_before_outer_label_access",
            }
        )

    def selection_policy(self) -> SelectionPolicy:
        """Return the manifest field that binds this policy into Protocol_Hash."""

        return SelectionPolicy(
            policy_key=self.policy_key,
            objective_metric="predeclared_fixed_epoch",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule="candidate_id_utf8_ascending",
        )


class HistologyLopoFixedPlanner:
    """Plan one complete, patient-isolated outer test fold per patient."""

    def __init__(
        self,
        planner: ProtocolPlanner,
        *,
        epoch_policy: HistologyFixedEpochPolicy | int = HISTOLOGY_LOPO_FIXED_DEFAULT_EPOCHS,
    ) -> None:
        if not isinstance(planner, ProtocolPlanner):
            raise TypeError("planner must be a ProtocolPlanner")
        if isinstance(epoch_policy, int) and not isinstance(epoch_policy, bool):
            epoch_policy = HistologyFixedEpochPolicy(epoch_policy)
        if not isinstance(epoch_policy, HistologyFixedEpochPolicy):
            raise TypeError("epoch_policy must be a HistologyFixedEpochPolicy or integer")
        self.planner = planner
        self.epoch_policy = epoch_policy

    def build_folds(self, sample_manifest: SampleManifest) -> tuple[OuterFold, ...]:
        """Create canonical LOPO folds without reading labels or constructing a model."""

        groups = self._samples_by_group(sample_manifest)
        if len(groups) < 2:
            raise HistologyLopoFixedPlanningError(
                "hist.lopo-fixed.v1 requires at least two patient groups"
            )
        all_ids = set(sample_manifest.canonical_order)
        folds = tuple(
            OuterFold(
                fold_id=f"outer-patient-{index:03d}",
                outer_train_ids=self._ordered(all_ids - set(test_ids)),
                outer_test_ids=test_ids,
                inner_folds=(),
            )
            for index, (_, test_ids) in enumerate(groups.items(), start=1)
        )
        self._require_exact_oof_coverage(folds, all_ids)
        return folds

    def plan(
        self,
        sample_manifest: SampleManifest,
        *,
        seed_set: Iterable[int],
        schema_version: str = "1.0",
    ):
        """Create a hash-bound protocol with the epoch policy already frozen."""

        seeds = tuple(seed_set)
        folds = self.build_folds(sample_manifest)
        return self.planner.plan(
            sample_manifest,
            schema_version=schema_version,
            protocol_key=HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY,
            folds=folds,
            selection_policy=self.epoch_policy.selection_policy(),
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
        """Atomically publish the protocol before any patient job can be scheduled."""

        seeds = tuple(seed_set)
        folds = self.build_folds(sample_manifest)
        return self.planner.plan_and_publish(
            sample_manifest,
            schema_version=schema_version,
            protocol_key=HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY,
            folds=folds,
            selection_policy=self.epoch_policy.selection_policy(),
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
                raise HistologyLopoFixedPlanningError(
                    f"sample {record.sample_id!r} has no patient Group_ID"
                )
            if record.sample_id in observed_ids:
                raise HistologyLopoFixedPlanningError(
                    f"duplicate Sample_ID in manifest: {record.sample_id}"
                )
            observed_ids.add(record.sample_id)
            grouped[record.group_id].append(record.sample_id)
        if observed_ids != set(sample_manifest.canonical_order):
            raise HistologyLopoFixedPlanningError(
                "manifest records and Canonical_Order contain different Sample_ID sets"
            )
        return {
            group_id: cls._ordered(sample_ids)
            for group_id, sample_ids in sorted(
                grouped.items(), key=lambda item: item[0].encode("utf-8")
            )
        }

    @staticmethod
    def _require_exact_oof_coverage(
        folds: tuple[OuterFold, ...], expected_ids: set[str]
    ) -> None:
        observed: list[str] = [
            sample_id for fold in folds for sample_id in fold.outer_test_ids
        ]
        if len(observed) != len(set(observed)) or set(observed) != expected_ids:
            raise HistologyLopoFixedPlanningError(
                "LOPO outer tests must cover every Sample_ID exactly once"
            )

    @staticmethod
    def _ordered(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


# Concise aliases for callers that use the protocol name rather than its policy suffix.
HistologyLopoPlanner = HistologyLopoFixedPlanner
HistologyLopoPlanningError = HistologyLopoFixedPlanningError


__all__ = [
    "HISTOLOGY_LOPO_FIXED_DEFAULT_EPOCHS",
    "HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY",
    "HistologyFixedEpochPolicy",
    "HistologyLopoFixedPlanner",
    "HistologyLopoFixedPlanningError",
    "HistologyLopoPlanner",
    "HistologyLopoPlanningError",
]
