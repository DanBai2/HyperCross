"""Planning-only strict runner wiring for ``hist.lopo-fixed.v1``.

The runner never starts training. It freezes the epoch policy before publishing
or scheduling the protocol, declares exactly one outer evaluation per patient
and seed, validates exact pooled OOF coverage, and exposes a patient-completion
recovery boundary under the workspace-relative strict output root.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Optional

from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import (
    FastLaneConfigurationError,
    FastLaneStage,
    FastLaneStageTemplate,
    load_fast_lane_template,
)
from benchmark.metrics.engine import _binary_metric_components
from benchmark.protocols import ProtocolPlanner, PublishedProtocol
from benchmark.protocols.histology_lopo_fixed import (
    HISTOLOGY_LOPO_FIXED_DEFAULT_EPOCHS,
    HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY,
    HistologyFixedEpochPolicy,
    HistologyLopoFixedPlanner,
)
from benchmark.recovery import FailureAtomicArtifactStore, RecoveryManager
from benchmark.registry import RecoveryArtifact
from benchmark.schema import (
    AggregationPolicy,
    CheckpointRetentionPolicy,
    EnvironmentLock,
    MetricSchema,
    PredictionRecord,
    ProtocolManifest,
    SelectionCandidate,
    SelectionDecision,
    TaskKind,
)


_EXPECTED_WORKSPACE = PureWindowsPath(r"G:\Datasets\HyperCross")
HISTOLOGY_LOPO_RELATIVE_OUTPUT_ROOT = "runs_v2/strict/histology-lopo"
HISTOLOGY_LOPO_AGGREGATION_POLICY_KEY = "histology-lopo-pooled-oof.v1"
HISTOLOGY_LOPO_REQUIRED_METRICS = (
    "pooled_oof_accuracy",
    "pooled_oof_balanced_accuracy",
    "pooled_oof_auroc",
    "pooled_oof_sensitivity",
    "pooled_oof_specificity",
    "pooled_oof_confusion_matrix",
    "pooled_oof_ece",
)


class HistologyLopoFixedRunnerError(ValueError):
    """Fixed-epoch LOPO runner input violates its frozen contract."""


@dataclass(frozen=True)
class HistologyLopoExecutionContract:
    """Non-overridable local execution and write-boundary declaration."""

    conda_environment: str = "TCFM"
    device: str = "cuda:0"
    expected_device_name: str = "NVIDIA GeForce RTX 5080"
    visible_cuda_device_count: int = 1
    process_count: int = 1
    output_root: str = HISTOLOGY_LOPO_RELATIVE_OUTPUT_ROOT
    recovery_boundary: str = "patient_complete"
    read_only_roots: tuple[str, ...] = (
        "data_caches",
        "legacy_checkpoints",
        "runs",
        "src",
    )


@dataclass(frozen=True)
class LopoPatientJob:
    """One immutable held-out patient-fold/seed job."""

    fold_id: str
    seed: int
    protocol_hash: str
    epoch_policy_hash: str
    epochs: int
    outer_test_sample_ids: tuple[str, ...]
    successful_outer_evaluations: int = 1
    recovery_boundary: str = "patient_complete"


class HistologyLopoMetricComputer:
    """Compute fixed-threshold pooled OOF metrics after exact coverage checks."""

    def __init__(self, protocol: ProtocolManifest) -> None:
        if protocol.protocol_key != HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY:
            raise HistologyLopoFixedRunnerError(
                "metric computer requires hist.lopo-fixed.v1"
            )
        self.protocol = protocol
        self._fold_by_sample = {
            sample_id: fold.fold_id
            for fold in protocol.folds
            for sample_id in fold.outer_test_ids
        }
        expected_count = sum(len(fold.outer_test_ids) for fold in protocol.folds)
        if len(self._fold_by_sample) != expected_count:
            raise HistologyLopoFixedRunnerError(
                "LOPO outer-test Sample_IDs must be globally unique"
            )

    def compute(
        self,
        records: tuple[PredictionRecord, ...],
        metric_schema: MetricSchema,
    ) -> Mapping[str, Any]:
        if metric_schema.task_kind is not TaskKind.BINARY_CLASSIFICATION:
            raise HistologyLopoFixedRunnerError(
                "LOPO aggregation requires binary classification metrics"
            )
        if metric_schema.decision_threshold != 0.5:
            raise HistologyLopoFixedRunnerError(
                "LOPO primary aggregation requires threshold 0.5"
            )
        if not records:
            raise HistologyLopoFixedRunnerError(
                "LOPO aggregation requires OOF predictions"
            )

        expected_ids = set(self._fold_by_sample)
        observed_seeds = {record.key.seed for record in records}
        if observed_seeds != set(self.protocol.seed_set):
            raise HistologyLopoFixedRunnerError(
                "OOF predictions must cover the complete protocol Seed_Set"
            )
        for seed in self.protocol.seed_set:
            seed_records = tuple(record for record in records if record.key.seed == seed)
            sample_ids = [record.key.sample_id for record in seed_records]
            if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != expected_ids:
                raise HistologyLopoFixedRunnerError(
                    f"seed {seed} does not provide exact once-only OOF coverage"
                )
            mismatched = tuple(
                record.key.sample_id
                for record in seed_records
                if self._fold_by_sample.get(record.key.sample_id) != record.key.fold_id
            )
            if mismatched:
                raise HistologyLopoFixedRunnerError(
                    "OOF prediction uses a fold other than its held-out patient fold"
                )

        values, confusion_matrix = _binary_metric_components(records, metric_schema)
        result: dict[str, Any] = {}
        for name in (
            "accuracy",
            "balanced_accuracy",
            "auroc",
            "sensitivity",
            "specificity",
            "ece",
        ):
            metric = values[name]
            key = f"pooled_oof_{name}"
            result[key] = metric.value
            if metric.reason is not None:
                result[f"{key}_reason"] = metric.reason
        result["pooled_oof_confusion_matrix"] = confusion_matrix
        return result


class HistologyLopoFixedRunner:
    """Bind fixed-epoch LOPO planning to strict local execution components."""

    protocol_key = HISTOLOGY_LOPO_FIXED_PROTOCOL_KEY
    relative_output_root = HISTOLOGY_LOPO_RELATIVE_OUTPUT_ROOT
    required_metrics = HISTOLOGY_LOPO_REQUIRED_METRICS
    execution_contract = HistologyLopoExecutionContract()

    def __init__(
        self,
        workspace_root: str | Path = Path(r"G:\Datasets\HyperCross"),
        *,
        epoch_policy: HistologyFixedEpochPolicy | int = HISTOLOGY_LOPO_FIXED_DEFAULT_EPOCHS,
        template_path: Optional[str | Path] = None,
        template: Optional[FastLaneStageTemplate] = None,
        artifact_store: Optional[FailureAtomicArtifactStore] = None,
    ) -> None:
        if template is not None and template_path is not None:
            raise HistologyLopoFixedRunnerError(
                "provide template or template_path, not both"
            )
        loaded = template or load_fast_lane_template(
            template_path
            or Path(r"G:\Datasets\HyperCross\configs\fast_lane\strict.json")
        )
        try:
            workspace = loaded.validate_workspace_root(workspace_root)
            output_relative = loaded.validate_write_path(self.relative_output_root)
        except FastLaneConfigurationError as exc:
            raise HistologyLopoFixedRunnerError(str(exc)) from exc
        self._validate_template(loaded)
        if isinstance(epoch_policy, int) and not isinstance(epoch_policy, bool):
            epoch_policy = HistologyFixedEpochPolicy(epoch_policy)
        if not isinstance(epoch_policy, HistologyFixedEpochPolicy):
            raise TypeError("epoch_policy must be a HistologyFixedEpochPolicy or integer")

        self.workspace_root = workspace
        self.template = loaded
        self.epoch_policy = epoch_policy
        self.output_root = workspace / Path(*output_relative.split("/"))
        self.artifact_store = artifact_store or FailureAtomicArtifactStore(
            self.output_root
        )
        if self.artifact_store.root.resolve(strict=False) != self.output_root.resolve(
            strict=False
        ):
            raise HistologyLopoFixedRunnerError(
                "artifact store root must equal runs_v2/strict/histology-lopo"
            )
        self.protocol_planner = HistologyLopoFixedPlanner(
            ProtocolPlanner(
                self.artifact_store,
                error_journal=self.artifact_store.error_journal,
            ),
            epoch_policy=self.epoch_policy,
        )

    @property
    def seed_set(self) -> tuple[int, ...]:
        return tuple(self.template.seed_set)

    def validate_live_environment(self) -> EnvironmentLock:
        """Require the audited TCFM, one-process, local RTX 5080 environment."""

        return validate_live_environment(self.template)

    def plan_protocol(self, sample_manifest: Any) -> PublishedProtocol:
        """Publish the policy-bound protocol before any patient job exists."""

        dry = self.protocol_planner.plan(sample_manifest, seed_set=self.seed_set)
        relative_path = PurePosixPath(
            "protocols", f"{dry.protocol_hash}.canonical.json"
        ).as_posix()
        return self.protocol_planner.plan_and_publish(
            sample_manifest,
            seed_set=self.seed_set,
            relative_path=relative_path,
        )

    def schedule(self, protocol: ProtocolManifest) -> tuple[LopoPatientJob, ...]:
        """Declare one successful complete-patient evaluation per fold and seed."""

        self._require_protocol(protocol)
        expected_budget = len(protocol.folds) * len(protocol.seed_set)
        if protocol.evaluation_budget.maximum_successful_evaluations != expected_budget:
            raise HistologyLopoFixedRunnerError(
                "evaluation budget must allow exactly one evaluation per patient/seed job"
            )
        jobs = tuple(
            LopoPatientJob(
                fold_id=fold.fold_id,
                seed=seed,
                protocol_hash=protocol.protocol_hash,
                epoch_policy_hash=self.epoch_policy.policy_hash,
                epochs=self.epoch_policy.epochs,
                outer_test_sample_ids=fold.outer_test_ids,
            )
            for fold in protocol.folds
            for seed in protocol.seed_set
        )
        if len({(job.fold_id, job.seed) for job in jobs}) != len(jobs):
            raise HistologyLopoFixedRunnerError("patient/seed jobs must be unique")
        return jobs

    def freeze_epoch_decision(
        self,
        protocol: ProtocolManifest,
        *,
        fold_id: str,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Materialize the predeclared epoch as an immutable pre-evaluation decision."""

        self._require_protocol(protocol)
        try:
            fold = next(item for item in protocol.folds if item.fold_id == fold_id)
        except StopIteration as exc:
            raise HistologyLopoFixedRunnerError("fold_id is not declared by protocol") from exc
        candidate = SelectionCandidate(
            candidate_id=f"fixed-epoch-{self.epoch_policy.epochs:04d}",
            configuration={
                "epoch": self.epoch_policy.epochs,
                "epoch_policy_hash": self.epoch_policy.policy_hash,
                "selection_source": "predeclared_before_outer_label_access",
            },
        )
        values: dict[str, Any] = {
            "decision_id": f"{fold.fold_id}-fixed-epoch",
            "fold_id": fold.fold_id,
            "candidates": (candidate,),
            "candidate_scores": {candidate.candidate_id: 0.0},
            "source_sample_ids": fold.outer_train_ids,
            "metric_direction": protocol.selection_policy.metric_direction,
            "tie_break_rule": protocol.selection_policy.tie_break_rule,
            "source_prediction_hash": self.epoch_policy.policy_hash,
            "selected_candidate": candidate.candidate_id,
        }
        if frozen_utc is not None:
            values["frozen_utc"] = frozen_utc
        decision = SelectionDecision.create(**values)
        if set(decision.source_sample_ids) & set(fold.outer_test_ids):
            raise HistologyLopoFixedRunnerError(
                "frozen epoch decision must not reference held-out patient samples"
            )
        return decision

    @staticmethod
    def metric_schema() -> MetricSchema:
        return MetricSchema(
            task_kind=TaskKind.BINARY_CLASSIFICATION,
            numeric_tolerance=1e-12,
            decision_threshold=0.5,
            ece_bins=10,
        )

    def aggregation_policy(self, protocol: ProtocolManifest) -> AggregationPolicy:
        """Declare pooled OOF output over the exact fold/seed Cartesian product."""

        self._require_protocol(protocol)
        return AggregationPolicy(
            policy_key=HISTOLOGY_LOPO_AGGREGATION_POLICY_KEY,
            metric_schema=self.metric_schema(),
            variance_metric="pooled_oof_accuracy",
            seed_set=protocol.seed_set,
            fold_ids=tuple(fold.fold_id for fold in protocol.folds),
        )

    def metric_computer(self, protocol: ProtocolManifest) -> HistologyLopoMetricComputer:
        self._require_protocol(protocol)
        return HistologyLopoMetricComputer(protocol)

    def save_patient_boundary(
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
        """Atomically persist all state after one held-out patient is complete."""

        self._require_protocol(protocol)
        fold_ids = {fold.fold_id for fold in protocol.folds}
        if fold_id not in fold_ids:
            raise HistologyLopoFixedRunnerError(f"unknown patient Fold_ID: {fold_id!r}")
        latest = recovery_manager.registry.latest_run(run_id)
        if latest.frozen_identity.protocol_hash != protocol.protocol_hash:
            raise HistologyLopoFixedRunnerError(
                "run Protocol_Hash differs from LOPO protocol"
            )
        if latest.frozen_identity.fold_id != fold_id:
            raise HistologyLopoFixedRunnerError(
                "run Fold_ID differs from patient recovery fold"
            )
        relative_path = PurePosixPath(
            "recovery", run_id, f"{fold_id}.canonical.json"
        ).as_posix()
        return recovery_manager.save_checkpoint_boundary(
            run_id=run_id,
            recovery_id=f"recovery-{run_id}-{fold_id}-patient-complete",
            relative_path=relative_path,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            stage_state=stage_state,
            last_completed_stage="outer_patient_complete",
            completed_checkpoint_boundary=1,
            created_utc=created_utc,
        )

    def _require_protocol(self, protocol: ProtocolManifest) -> None:
        if protocol.protocol_key != self.protocol_key:
            raise HistologyLopoFixedRunnerError(
                f"runner requires protocol {self.protocol_key}"
            )
        if protocol.seed_set != self.seed_set:
            raise HistologyLopoFixedRunnerError(
                "protocol seed set differs from strict template"
            )
        if protocol.selection_policy != self.epoch_policy.selection_policy():
            raise HistologyLopoFixedRunnerError(
                "protocol does not bind the runner's frozen epoch policy"
            )
        test_ids = [sample_id for fold in protocol.folds for sample_id in fold.outer_test_ids]
        if len(test_ids) != len(set(test_ids)):
            raise HistologyLopoFixedRunnerError(
                "patient outer-test folds must provide exact once-only coverage"
            )
        protocol.__class__.model_validate(protocol.model_dump(mode="python"))

    @staticmethod
    def _validate_template(template: FastLaneStageTemplate) -> None:
        expected = {
            "stage": FastLaneStage.STRICT,
            "conda_environment": "TCFM",
            "device": "cuda:0",
            "expected_device_name": "NVIDIA GeForce RTX 5080",
            "visible_cuda_device_count": 1,
            "process_count": 1,
            "checkpoint_interval_epochs": 1,
            "checkpoint_retention": CheckpointRetentionPolicy(),
        }
        mismatches = [
            field for field, value in expected.items() if getattr(template, field) != value
        ]
        if mismatches:
            raise HistologyLopoFixedRunnerError(
                "strict runner template mismatch: " + ", ".join(mismatches)
            )


HistologyLopoRunner = HistologyLopoFixedRunner
HistologyLopoRunnerError = HistologyLopoFixedRunnerError


__all__ = [
    "HISTOLOGY_LOPO_AGGREGATION_POLICY_KEY",
    "HISTOLOGY_LOPO_RELATIVE_OUTPUT_ROOT",
    "HISTOLOGY_LOPO_REQUIRED_METRICS",
    "HistologyLopoExecutionContract",
    "HistologyLopoFixedRunner",
    "HistologyLopoFixedRunnerError",
    "HistologyLopoMetricComputer",
    "HistologyLopoRunner",
    "HistologyLopoRunnerError",
    "LopoPatientJob",
]
