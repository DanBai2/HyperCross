"""Strict runner wiring for ``hist.patient-nested5.v1``.

This module plans and validates work. It does not start training on import or
while constructing a runner. Actual fold execution delegates to the shared
recoverable orchestrator, which freezes selection before its single outer
one-shot evaluation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from benchmark.cli_runtime import validate_live_environment
from benchmark.engine import (
    OrchestrationCallbacks,
    OrchestrationRequest,
    OrchestrationResult,
    SelectionController,
    TrainingOrchestrator,
)
from benchmark.fast_lane import (
    FastLaneConfigurationError,
    FastLaneStage,
    FastLaneStageTemplate,
    load_fast_lane_template,
)
from benchmark.protocols import ProtocolPlanner, PublishedProtocol
from benchmark.protocols.histology_nested import (
    HISTOLOGY_NESTED5_PROTOCOL_KEY,
    HistologyNested5Planner,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.registry import FrozenRunIdentity
from benchmark.schema import (
    CheckpointRetentionPolicy,
    EnvironmentLock,
    FrozenRunConfig,
    OuterFold,
    ProtocolManifest,
    SelectionCandidate,
    SelectionDecision,
    hash_object,
)

from .histology_statistics import HistologyStatisticsWiring


HISTOLOGY_NESTED_RELATIVE_OUTPUT_ROOT = "runs_v2/strict/histology-nested"
HISTOLOGY_NESTED_REQUIRED_METRICS = (
    "fold_macro_accuracy",
    "pooled_oof_accuracy",
    "pooled_oof_balanced_accuracy",
    "pooled_oof_auroc",
    "pooled_oof_sensitivity",
    "pooled_oof_specificity",
    "pooled_oof_confusion_matrix",
    "pooled_oof_ece",
)


class HistologyNestedRunnerError(ValueError):
    """Strict nested runner input or wiring violates its frozen contract."""


@dataclass(frozen=True)
class NestedFoldJob:
    """One immutable fold/seed job with one permitted successful evaluation."""

    fold_id: str
    seed: int
    protocol_hash: str
    successful_outer_evaluations: int = 1
    recovery_boundaries: tuple[str, ...] = ("epoch", "fold")

    @property
    def job_hash(self) -> str:
        return hash_object(
            {
                "fold_id": self.fold_id,
                "seed": self.seed,
                "protocol_hash": self.protocol_hash,
                "successful_outer_evaluations": self.successful_outer_evaluations,
                "recovery_boundaries": self.recovery_boundaries,
            }
        )


class HistologyNestedRunner:
    """Bind the nested protocol to the audited strict execution components."""

    protocol_key = HISTOLOGY_NESTED5_PROTOCOL_KEY
    relative_output_root = HISTOLOGY_NESTED_RELATIVE_OUTPUT_ROOT
    required_metrics = HISTOLOGY_NESTED_REQUIRED_METRICS

    def __init__(
        self,
        workspace_root: str | Path = Path(r"G:\Datasets\HyperCross"),
        *,
        template_path: Optional[str | Path] = None,
        template: Optional[FastLaneStageTemplate] = None,
        artifact_store: Optional[FailureAtomicArtifactStore] = None,
        selection_controller: Optional[SelectionController] = None,
    ) -> None:
        if template is not None and template_path is not None:
            raise HistologyNestedRunnerError(
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
            raise HistologyNestedRunnerError(str(exc)) from exc
        self._validate_template(loaded)
        self.workspace_root = workspace
        self.template = loaded
        self.output_root = workspace / Path(*output_relative.split("/"))
        self.artifact_store = artifact_store or FailureAtomicArtifactStore(
            self.output_root
        )
        if self.artifact_store.root.resolve(strict=False) != self.output_root.resolve(
            strict=False
        ):
            raise HistologyNestedRunnerError(
                "artifact store root must equal runs_v2/strict/histology-nested"
            )
        self.selection_controller = selection_controller or SelectionController(
            error_journal=self.artifact_store.error_journal
        )
        self.statistics = HistologyStatisticsWiring()
        self.protocol_planner = HistologyNested5Planner(
            ProtocolPlanner(
                self.artifact_store,
                error_journal=self.artifact_store.error_journal,
            )
        )

    @property
    def seed_set(self) -> tuple[int, ...]:
        return tuple(self.template.seed_set)

    def validate_live_environment(self) -> EnvironmentLock:
        """Require the audited TCFM, one-process, local RTX 5080 environment."""

        return validate_live_environment(self.template)

    def plan_protocol(self, sample_manifest: Any) -> PublishedProtocol:
        """Atomically publish the nested protocol beneath the strict output root."""

        return self.protocol_planner.plan_and_publish(
            sample_manifest,
            seed_set=self.seed_set,
        )

    def schedule(self, protocol: ProtocolManifest) -> tuple[NestedFoldJob, ...]:
        """Declare exactly one one-shot outer evaluation per fold/seed run."""

        self._require_protocol(protocol)
        expected_budget = len(protocol.folds) * len(protocol.seed_set)
        if protocol.evaluation_budget.maximum_successful_evaluations != expected_budget:
            raise HistologyNestedRunnerError(
                "evaluation budget must allow exactly one evaluation per fold/seed job"
            )
        return tuple(
            NestedFoldJob(
                fold_id=fold.fold_id,
                seed=seed,
                protocol_hash=protocol.protocol_hash,
            )
            for fold in protocol.folds
            for seed in protocol.seed_set
        )

    @staticmethod
    def build_selection_space(
        configurations: Mapping[str, Mapping[str, Any]],
        epochs: Iterable[int],
    ) -> tuple[SelectionCandidate, ...]:
        """Freeze the Cartesian config/epoch candidates before outer access."""

        if not isinstance(configurations, Mapping) or not configurations:
            raise HistologyNestedRunnerError("configurations must not be empty")
        epoch_values = tuple(epochs)
        if (
            not epoch_values
            or any(isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0 for epoch in epoch_values)
            or len(set(epoch_values)) != len(epoch_values)
        ):
            raise HistologyNestedRunnerError("epochs must be unique positive integers")
        candidates = []
        for config_key in sorted(configurations, key=lambda value: value.encode("utf-8")):
            if not isinstance(config_key, str) or not config_key.strip():
                raise HistologyNestedRunnerError("configuration keys must be non-empty strings")
            parameters = dict(configurations[config_key])
            for epoch in sorted(epoch_values):
                candidates.append(
                    SelectionCandidate(
                        candidate_id=f"{config_key}.epoch-{epoch:04d}",
                        configuration={
                            "config_key": config_key,
                            "epoch": epoch,
                            "parameters": parameters,
                        },
                    )
                )
        return tuple(candidates)

    def freeze_config_epoch(
        self,
        outer_fold: OuterFold,
        *,
        candidates: Iterable[SelectionCandidate],
        candidate_scores: Mapping[str, float],
        inner_source_sample_ids: Iterable[str],
        inner_prediction_hash: str,
        decision_id: Optional[str] = None,
    ) -> SelectionDecision:
        """Select config and epoch exclusively from complete inner OOF evidence."""

        source_ids = tuple(inner_source_sample_ids)
        expected_inner_ids = tuple(
            sorted(
                {
                    sample_id
                    for inner_fold in outer_fold.inner_folds
                    for sample_id in inner_fold.validation_ids
                },
                key=lambda value: value.encode("utf-8"),
            )
        )
        if tuple(sorted(source_ids, key=lambda value: value.encode("utf-8"))) != expected_inner_ids:
            raise HistologyNestedRunnerError(
                "selection sources must equal the complete inner-validation OOF Sample_ID set"
            )
        candidate_values = tuple(candidates)
        for candidate in candidate_values:
            config = candidate.configuration
            if "config_key" not in config or "epoch" not in config:
                raise HistologyNestedRunnerError(
                    "every nested selection candidate must freeze config_key and epoch"
                )
        return self.selection_controller.freeze_selection(
            outer_fold,
            candidates=candidate_values,
            candidate_scores=candidate_scores,
            source_sample_ids=source_ids,
            source_prediction_hash=inner_prediction_hash,
            selection_policy=self._selection_policy(outer_fold),
            decision_id=decision_id,
        )

    def orchestration_request(
        self,
        *,
        run_id: str,
        published_protocol: PublishedProtocol,
        fold_id: str,
        frozen_config: FrozenRunConfig,
    ) -> OrchestrationRequest:
        """Build output-confined wiring for recoverable execution of one fold job."""

        protocol = published_protocol.manifest
        self._require_protocol(protocol)
        if fold_id not in {fold.fold_id for fold in protocol.folds}:
            raise HistologyNestedRunnerError("fold_id is not declared by the protocol")
        if frozen_config.protocol_key != self.protocol_key:
            raise HistologyNestedRunnerError("frozen config uses a different protocol")
        if frozen_config.checkpoint_retention != CheckpointRetentionPolicy():
            raise HistologyNestedRunnerError(
                "selected, last, and epoch recovery checkpoints must all be retained"
            )
        run_component = hash_object({"run_id": run_id})
        return OrchestrationRequest(
            run_id=run_id,
            published_protocol=published_protocol,
            outer_fold_id=fold_id,
            frozen_config=frozen_config,
            prediction_artifact_id=f"hist-nested-prediction-{run_component}",
            prediction_relative_path=(
                f"predictions/{fold_id}/{run_component}.canonical.json"
            ),
        )

    def execute_fold_once(
        self,
        orchestrator: TrainingOrchestrator,
        request: OrchestrationRequest,
        callbacks: OrchestrationCallbacks,
    ) -> OrchestrationResult:
        """Delegate one fresh job; the orchestrator owns its sole outer evaluation."""

        self._require_orchestrator(orchestrator)
        self._require_request(request)
        return orchestrator.execute(request, callbacks)

    def resume_fold(
        self,
        orchestrator: TrainingOrchestrator,
        request: OrchestrationRequest,
        callbacks: OrchestrationCallbacks,
        *,
        observed_identity: FrozenRunIdentity,
    ) -> OrchestrationResult:
        """Resume from a verified epoch/fold boundary without repeating evaluation."""

        self._require_orchestrator(orchestrator)
        self._require_request(request)
        return orchestrator.resume(
            request,
            callbacks,
            observed_identity=observed_identity,
        )

    def _require_orchestrator(self, orchestrator: TrainingOrchestrator) -> None:
        if not isinstance(orchestrator, TrainingOrchestrator):
            raise TypeError("orchestrator must be a TrainingOrchestrator")
        root = orchestrator.recovery_manager.artifact_store.root.resolve(strict=False)
        if root != self.output_root.resolve(strict=False):
            raise HistologyNestedRunnerError(
                "orchestrator artifacts must remain under the nested strict output root"
            )

    def _require_request(self, request: OrchestrationRequest) -> None:
        self._require_protocol(request.published_protocol.manifest)
        if request.prediction_relative_path is None or not request.prediction_relative_path.startswith(
            "predictions/"
        ):
            raise HistologyNestedRunnerError(
                "prediction path must be relative to the nested strict output root"
            )

    def _require_protocol(self, protocol: ProtocolManifest) -> None:
        if protocol.protocol_key != self.protocol_key:
            raise HistologyNestedRunnerError(
                f"runner requires protocol {self.protocol_key}"
            )
        if protocol.seed_set != self.seed_set:
            raise HistologyNestedRunnerError("protocol seed set differs from strict template")
        protocol.__class__.model_validate(protocol.model_dump(mode="python"))

    def _selection_policy(self, outer_fold: OuterFold):
        # Resolve through a planner-created manifest-independent constant only
        # after confirming the fold has the required nested structure.
        if len(outer_fold.inner_folds) != 5:
            raise HistologyNestedRunnerError("each outer fold must contain five inner folds")
        from benchmark.protocols.histology_nested import HISTOLOGY_NESTED5_SELECTION_POLICY

        return HISTOLOGY_NESTED5_SELECTION_POLICY

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
            raise HistologyNestedRunnerError(
                "strict runner template mismatch: " + ", ".join(mismatches)
            )


__all__ = [
    "HISTOLOGY_NESTED_RELATIVE_OUTPUT_ROOT",
    "HISTOLOGY_NESTED_REQUIRED_METRICS",
    "HistologyNestedRunner",
    "HistologyNestedRunnerError",
    "NestedFoldJob",
]
