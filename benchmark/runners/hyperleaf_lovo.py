"""Planning-only runner wiring for ``leaf.lovo.nested.v1``.

This module freezes the local execution contract, delegates all split creation
to the HyperLeaf LOVO planner, limits route/epoch selection to complete inner
OOF evidence, declares prediction-first LOVO aggregation, and exposes the
existing failure-atomic recovery manager at an outer-fold boundary.  It does
not load data, construct an official-test adapter, or start training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Optional, Sequence

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS
from benchmark.engine.selection import SelectionController
from benchmark.protocols.hyperleaf_lovo import (
    HYPERLEAF_LOVO_PROTOCOL_KEY,
    plan_hyperleaf_lovo,
)
from benchmark.protocols.planner import ProtocolPlanner, PublishedProtocol
from benchmark.recovery.manager import RecoveryManager
from benchmark.registry.records import RecoveryArtifact
from benchmark.schema.aggregation import AggregationPolicy
from benchmark.schema.base import TaskKind
from benchmark.schema.canonical import normalize_id
from benchmark.schema.datasets import SampleManifest
from benchmark.schema.environment import EnvironmentLock
from benchmark.schema.metrics import MetricSchema
from benchmark.schema.predictions import PredictionRecord
from benchmark.schema.protocol import OuterFold, ProtocolManifest
from benchmark.schema.selection import SelectionCandidate, SelectionDecision

from .hyperleaf_route_a import HyperLeafRouteARunner


_EXPECTED_WORKSPACE = PureWindowsPath(r"G:\Datasets\HyperCross")
HYPERLEAF_LOVO_OUTPUT_ROOT = PurePosixPath("runs_v2/strict/hyperleaf/lovo")
HYPERLEAF_LOVO_AGGREGATION_POLICY_KEY = "hyperleaf-lovo-fold-and-pooled-oof.v1"


class HyperLeafLovoRunnerError(ValueError):
    """LOVO runner wiring or frozen execution policy was violated."""


@dataclass(frozen=True)
class HyperLeafLovoExecutionContract:
    """Non-overridable local execution and data-access contract."""

    conda_environment: str = "TCFM"
    device: str = "cuda:0"
    expected_device_name: str = "NVIDIA GeForce RTX 5080"
    visible_cuda_device_count: int = 1
    process_count: int = 1
    dataset_partition: str = "train"
    official_test_access: bool = False
    output_root: str = HYPERLEAF_LOVO_OUTPUT_ROOT.as_posix()
    checkpoint_boundary: str = "outer_fold"


@dataclass(frozen=True)
class LovoRouteEpochOption:
    """One route and epoch choice declared before outer evaluation."""

    model_key: str
    epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key.strip():
            raise ValueError("model_key must be a non-empty stable Model_Key")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch <= 0:
            raise ValueError("epoch must be a positive integer")


class HyperLeafLovoMetricComputer:
    """Compute explicitly named per-fold, per-target, macro, and pooled OOF metrics."""

    def __init__(self, protocol: ProtocolManifest) -> None:
        if protocol.protocol_key != HYPERLEAF_LOVO_PROTOCOL_KEY:
            raise HyperLeafLovoRunnerError("metric computer requires leaf.lovo.nested.v1")
        self.protocol = protocol
        self._fold_by_sample = {
            sample_id: fold.fold_id
            for fold in protocol.folds
            for sample_id in fold.outer_test_ids
        }
        if len(self._fold_by_sample) != sum(len(fold.outer_test_ids) for fold in protocol.folds):
            raise HyperLeafLovoRunnerError("LOVO outer-test samples must be globally unique")

    def compute(
        self,
        records: tuple[PredictionRecord, ...],
        metric_schema: MetricSchema,
    ) -> Mapping[str, Any]:
        if metric_schema.task_kind is not TaskKind.MULTI_TARGET_REGRESSION:
            raise HyperLeafLovoRunnerError("LOVO aggregation requires regression metrics")
        if metric_schema.target_names != HYPERLEAF_TARGETS:
            raise HyperLeafLovoRunnerError("LOVO target names differ from HyperLeaf targets")
        if not records:
            raise HyperLeafLovoRunnerError("LOVO aggregation requires OOF predictions")

        expected_ids = set(self._fold_by_sample)
        seeds = {record.key.seed for record in records}
        for seed in seeds:
            seed_records = tuple(record for record in records if record.key.seed == seed)
            observed_ids = [record.key.sample_id for record in seed_records]
            if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
                raise HyperLeafLovoRunnerError(
                    f"seed {seed} does not provide exact one-prediction OOF coverage"
                )
            mismatched = tuple(
                record.key.sample_id
                for record in seed_records
                if self._fold_by_sample[record.key.sample_id] != record.key.fold_id
            )
            if mismatched:
                raise HyperLeafLovoRunnerError(
                    "OOF predictions use a fold other than the sample's held-out variety fold"
                )

        per_fold: dict[str, dict[str, tuple[Optional[float], float]]] = {}
        metrics: dict[str, Any] = {}
        for fold in self.protocol.folds:
            fold_records = tuple(record for record in records if record.key.fold_id == fold.fold_id)
            target_metrics = self._target_metrics(fold_records, metric_schema.target_names)
            per_fold[fold.fold_id] = target_metrics
            for target, (r2, mae) in target_metrics.items():
                prefix = f"per_fold.{fold.fold_id}"
                self._put_r2(metrics, f"{prefix}.r2.{target}", r2)
                metrics[f"{prefix}.mae.{target}"] = mae
            metrics[f"per_fold.{fold.fold_id}.mean_mae"] = self._mean(
                value[1] for value in target_metrics.values()
            )
            self._put_r2(
                metrics,
                f"per_fold.{fold.fold_id}.mean_r2",
                self._mean_optional(value[0] for value in target_metrics.values()),
            )

        pooled = self._target_metrics(records, metric_schema.target_names)
        for target, (r2, mae) in pooled.items():
            # The per-target namespace is intentionally separate from the
            # aggregation-policy namespace required by Requirements 8.21-8.23.
            self._put_r2(metrics, f"per_target.r2.{target}", r2)
            metrics[f"per_target.mae.{target}"] = mae
            self._put_r2(metrics, f"pooled_oof.r2.{target}", r2)
            metrics[f"pooled_oof.mae.{target}"] = mae

            fold_r2 = [per_fold[fold.fold_id][target][0] for fold in self.protocol.folds]
            fold_mae = [per_fold[fold.fold_id][target][1] for fold in self.protocol.folds]
            self._put_r2(metrics, f"fold_macro.r2.{target}", self._mean_optional(fold_r2))
            metrics[f"fold_macro.mae.{target}"] = self._mean(fold_mae)

        self._put_r2(
            metrics,
            "pooled_oof.mean_r2",
            self._mean_optional(value[0] for value in pooled.values()),
        )
        metrics["pooled_oof.mean_mae"] = self._mean(value[1] for value in pooled.values())
        macro_target_r2 = [metrics[f"fold_macro.r2.{target}"] for target in metric_schema.target_names]
        self._put_r2(metrics, "fold_macro.mean_r2", self._mean_optional(macro_target_r2))
        metrics["fold_macro.mean_mae"] = self._mean(
            metrics[f"fold_macro.mae.{target}"] for target in metric_schema.target_names
        )
        return metrics

    @classmethod
    def _target_metrics(
        cls,
        records: Sequence[PredictionRecord],
        target_names: Sequence[str],
    ) -> dict[str, tuple[Optional[float], float]]:
        truths = [[] for _ in target_names]
        predictions = [[] for _ in target_names]
        for record in records:
            if not isinstance(record.y_true, tuple) or not isinstance(record.y_pred, tuple):
                raise HyperLeafLovoRunnerError("HyperLeaf predictions must be target vectors")
            if len(record.y_true) != len(target_names) or len(record.y_pred) != len(target_names):
                raise HyperLeafLovoRunnerError("HyperLeaf prediction target dimension is invalid")
            for index, (truth, prediction) in enumerate(zip(record.y_true, record.y_pred)):
                if (
                    isinstance(truth, bool)
                    or isinstance(prediction, bool)
                    or not isinstance(truth, (int, float))
                    or not isinstance(prediction, (int, float))
                    or not math.isfinite(float(truth))
                    or not math.isfinite(float(prediction))
                ):
                    raise HyperLeafLovoRunnerError("HyperLeaf predictions must be finite numeric vectors")
                truths[index].append(float(truth))
                predictions[index].append(float(prediction))
        result = {}
        for index, name in enumerate(target_names):
            target_truth = truths[index]
            target_prediction = predictions[index]
            mean_truth = math.fsum(target_truth) / len(target_truth)
            total = math.fsum((value - mean_truth) ** 2 for value in target_truth)
            residual = math.fsum(
                (truth - prediction) ** 2
                for truth, prediction in zip(target_truth, target_prediction)
            )
            r2 = None if total == 0.0 else 1.0 - residual / total
            mae = math.fsum(
                abs(truth - prediction)
                for truth, prediction in zip(target_truth, target_prediction)
            ) / len(target_truth)
            result[name] = (r2, mae)
        return result

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        materialized = tuple(float(value) for value in values)
        return math.fsum(materialized) / len(materialized)

    @classmethod
    def _mean_optional(cls, values: Iterable[Optional[float]]) -> Optional[float]:
        materialized = tuple(values)
        if not materialized or any(value is None for value in materialized):
            return None
        return cls._mean(value for value in materialized if value is not None)

    @staticmethod
    def _put_r2(metrics: dict[str, Any], key: str, value: Optional[float]) -> None:
        metrics[key] = value
        if value is None:
            metrics[f"{key}.reason"] = "zero_variance_target"


class HyperLeafLovoRunner:
    """Strict LOVO planning and orchestration surface with no training entry point."""

    execution_contract = HyperLeafLovoExecutionContract()
    output_root = HYPERLEAF_LOVO_OUTPUT_ROOT

    def __init__(
        self,
        workspace_root: Path | str = Path(r"G:\Datasets\HyperCross"),
        *,
        selection_controller: Optional[SelectionController] = None,
        route_runner: Optional[HyperLeafRouteARunner] = None,
    ) -> None:
        windows_root = PureWindowsPath(str(workspace_root))
        if str(windows_root).casefold() != str(_EXPECTED_WORKSPACE).casefold():
            raise HyperLeafLovoRunnerError(
                r"LOVO workspace must be exactly G:\Datasets\HyperCross; F: and other roots are forbidden"
            )
        self.workspace_root = Path(str(windows_root)).resolve(strict=False)
        self.artifact_root = self.workspace_root / Path(*self.output_root.parts)
        self.selection_controller = selection_controller or SelectionController()
        self.route_runner = route_runner or HyperLeafRouteARunner(self.workspace_root)

    def validate_launch(
        self,
        environment: EnvironmentLock,
        *,
        conda_environment: str,
        process_count: int,
    ) -> None:
        """Reject any launch outside the frozen TCFM/one-RTX-5080 contract."""

        expected = self.execution_contract
        observed_device = environment.device_identity
        failures = []
        if conda_environment != expected.conda_environment:
            failures.append("conda_environment")
        if process_count != expected.process_count:
            failures.append("process_count")
        if not environment.cuda_available or environment.cuda_device_count != 1:
            failures.append("visible_cuda_device_count")
        if f"cuda:0:{expected.expected_device_name}" not in observed_device:
            failures.append("device_identity")
        if failures:
            raise HyperLeafLovoRunnerError(
                "LOVO launch differs from frozen execution contract: " + ", ".join(failures)
            )

    def plan_protocol(
        self,
        planner: ProtocolPlanner,
        training_manifest: SampleManifest,
        *,
        seed_set: Iterable[int] = (7, 42, 123),
    ) -> PublishedProtocol:
        """Publish a hash-addressed protocol under the workspace-relative output root."""

        seeds = tuple(seed_set)
        dry = plan_hyperleaf_lovo(
            planner,
            training_manifest,
            seed_set=seeds,
            publish=False,
        )
        assert isinstance(dry, ProtocolManifest)
        relative_path = (
            self.output_root
            / "protocols"
            / f"{dry.protocol_hash}.canonical.json"
        ).as_posix()
        published = plan_hyperleaf_lovo(
            planner,
            training_manifest,
            seed_set=seeds,
            publish=True,
            relative_path=relative_path,
        )
        assert isinstance(published, PublishedProtocol)
        return published

    def selection_candidate(self, option: LovoRouteEpochOption) -> SelectionCandidate:
        """Bind a stable Route A model key and positive epoch into one candidate."""

        route_spec, registration = self.route_runner.resolve(option.model_key)
        candidate_id = f"{option.model_key}::epoch-{option.epoch:04d}"
        return SelectionCandidate(
            candidate_id=candidate_id,
            configuration={
                "model_key": option.model_key,
                "route": route_spec.inference_branch,
                "epoch": option.epoch,
                "representation_key": registration.input_contract.representation_key,
            },
        )

    def freeze_route_and_epoch(
        self,
        outer_fold: OuterFold,
        *,
        options: Iterable[LovoRouteEpochOption],
        candidate_scores: Mapping[str, float],
        inner_source_sample_ids: Iterable[str],
        inner_prediction_hash: str,
        decision_id: Optional[str] = None,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Freeze route/epoch using complete outer-train inner-variety OOF only."""

        candidates = tuple(self.selection_candidate(option) for option in options)
        source_ids = tuple(inner_source_sample_ids)
        complete_inner_oof = {
            sample_id
            for inner_fold in outer_fold.inner_folds
            for sample_id in inner_fold.validation_ids
        }
        if set(source_ids) != complete_inner_oof or complete_inner_oof != set(
            outer_fold.outer_train_ids
        ):
            raise HyperLeafLovoRunnerError(
                "route/epoch selection requires exact complete inner-variety OOF evidence"
            )
        return self.selection_controller.freeze_selection(
            outer_fold,
            candidates=candidates,
            candidate_scores=candidate_scores,
            source_sample_ids=source_ids,
            source_prediction_hash=inner_prediction_hash,
            selection_policy=self._selection_policy(),
            decision_id=decision_id,
            frozen_utc=frozen_utc,
        )

    @staticmethod
    def _selection_policy():
        from benchmark.protocols.hyperleaf_lovo import HYPERLEAF_LOVO_SELECTION_POLICY

        return HYPERLEAF_LOVO_SELECTION_POLICY

    @staticmethod
    def metric_schema() -> MetricSchema:
        return MetricSchema(
            task_kind=TaskKind.MULTI_TARGET_REGRESSION,
            numeric_tolerance=1e-12,
            target_names=HYPERLEAF_TARGETS,
        )

    def aggregation_policy(self, protocol: ProtocolManifest) -> AggregationPolicy:
        """Declare exact fold/seed membership and all required LOVO output families."""

        if protocol.protocol_key != HYPERLEAF_LOVO_PROTOCOL_KEY:
            raise HyperLeafLovoRunnerError("aggregation requires leaf.lovo.nested.v1")
        return AggregationPolicy(
            policy_key=HYPERLEAF_LOVO_AGGREGATION_POLICY_KEY,
            metric_schema=self.metric_schema(),
            variance_metric="pooled_oof.mean_mae",
            seed_set=protocol.seed_set,
            fold_ids=tuple(fold.fold_id for fold in protocol.folds),
        )

    def metric_computer(self, protocol: ProtocolManifest) -> HyperLeafLovoMetricComputer:
        return HyperLeafLovoMetricComputer(protocol)

    def save_fold_boundary(
        self,
        recovery_manager: RecoveryManager,
        protocol: ProtocolManifest,
        *,
        run_id: str,
        fold_id: str,
        model_state: bytes,
        optimizer_state: bytes,
        scheduler_state: bytes,
        stage_state: bytes,
        created_utc: Optional[datetime] = None,
    ) -> RecoveryArtifact:
        """Atomically persist all resumable state after one complete outer fold."""

        if protocol.protocol_key != HYPERLEAF_LOVO_PROTOCOL_KEY:
            raise HyperLeafLovoRunnerError("fold recovery requires leaf.lovo.nested.v1")
        fold_ids = {fold.fold_id for fold in protocol.folds}
        normalized_fold = normalize_id(fold_id)
        if normalized_fold not in fold_ids:
            raise HyperLeafLovoRunnerError(f"unknown LOVO Fold_ID: {fold_id!r}")
        latest = recovery_manager.registry.latest_run(run_id)
        if latest.frozen_identity.protocol_hash != protocol.protocol_hash:
            raise HyperLeafLovoRunnerError("run Protocol_Hash differs from LOVO protocol")
        if latest.frozen_identity.fold_id != normalized_fold:
            raise HyperLeafLovoRunnerError("run Fold_ID differs from recovery fold")
        recovery_id = f"recovery-{run_id}-{normalized_fold}-complete"
        relative_path = (
            self.output_root / "recovery" / run_id / f"{normalized_fold}.canonical.json"
        ).as_posix()
        return recovery_manager.save_checkpoint_boundary(
            run_id=run_id,
            recovery_id=recovery_id,
            relative_path=relative_path,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            stage_state=stage_state,
            last_completed_stage="outer_fold_complete",
            completed_checkpoint_boundary=1,
            created_utc=created_utc,
        )

    def metadata(self) -> dict[str, Any]:
        """Return auditable planning metadata without absolute paths."""

        return {
            "protocol_key": HYPERLEAF_LOVO_PROTOCOL_KEY,
            "output_root": self.output_root.as_posix(),
            "execution": self.execution_contract.__dict__,
            "selection_evidence": "outer_train_inner_varieties_only",
            "aggregation_outputs": (
                "per_fold",
                "per_target",
                "fold_macro",
                "pooled_oof",
            ),
            "recovery_boundary": "outer_fold",
            "official_test_access": False,
        }


__all__ = [
    "HYPERLEAF_LOVO_AGGREGATION_POLICY_KEY",
    "HYPERLEAF_LOVO_OUTPUT_ROOT",
    "HyperLeafLovoExecutionContract",
    "HyperLeafLovoMetricComputer",
    "HyperLeafLovoRunner",
    "HyperLeafLovoRunnerError",
    "LovoRouteEpochOption",
]
