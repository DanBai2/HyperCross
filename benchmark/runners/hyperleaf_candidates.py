"""Planning and preprocessing wiring for task 14.11 HyperLeaf candidates.

The runner is intentionally training-free.  It freezes candidate declarations,
fits target normalization and PCA only inside protocol-derived boundaries, and
publishes recoverable checkpoint bundles through the shared recovery manager.
It has no official-label reader and never imports the legacy ``src`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping, Optional

from benchmark.catalog.hyperleaf import (
    HyperLeafPartition,
    HyperLeafRepresentationBuilder,
)
from benchmark.engine.selection import SelectionController
from benchmark.models.hyperleaf_candidates import (
    AVERAGE_BRANCH,
    INDEPENDENT_TARGET_MODEL_KEY,
    PCA_BRANCH,
    PCA_INDEPENDENT_TARGET_MODEL_KEY,
    RGB_BRANCH,
    RGB_PCA_CONSISTENCY_MODEL_KEY,
    TARGET_ROUTED_BRANCH,
    create_hyperleaf_candidate_registry,
)
from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.recovery.manager import RecoveryManager, RestoredRecoveryState
from benchmark.registry import FrozenRunIdentity, RecoveryArtifact
from benchmark.schema import (
    CandidateOption,
    CandidateSpace,
    EnvironmentLock,
    FitBoundary,
    FitBoundaryScope,
    FittedArtifact,
    MetricDirection,
    OuterFold,
    SelectionCandidate,
    SelectionDecision,
    SelectionPolicy,
)
from benchmark.transforms.hyperleaf_candidates import (
    GswLossKind,
    GswTargetTransform,
    HYPERLEAF_STYLE_CONFIGS,
)


_EXPECTED_WORKSPACE = PureWindowsPath(r"G:\Datasets\HyperCross")
HYPERLEAF_CANDIDATE_OUTPUT_ROOT = PurePosixPath("runs_v2/candidates/hyperleaf")
GSW_SELECTION_POLICY = SelectionPolicy(
    policy_key="hyperleaf.gsw.inner-variety-candidate.v1",
    objective_metric="r2.Gsw",
    metric_direction=MetricDirection.MAXIMIZE,
    tie_break_rule="candidate_id_utf8_ascending",
)


class HyperLeafCandidateError(ValueError):
    """Candidate declaration, launch, fit, or recovery policy was violated."""


class HyperLeafCandidateBranch(str, Enum):
    INDEPENDENT_TARGET = "independent-target"
    STYLE_AUGMENTATION = "style-augmentation"
    TRAIN_FOLD_PCA = "train-fold-pca"
    RGB_PCA_CONSISTENCY = "rgb-pca-consistency"


class GswRoute(str, Enum):
    RGB = "rgb"
    PCA = "pca"
    RGB_PCA_CONSISTENCY_005 = "rgb-pca-consistency-0.05"
    RGB_PCA_CONSISTENCY_010 = "rgb-pca-consistency-0.10"


@dataclass(frozen=True)
class HyperLeafCandidateExecutionContract:
    """Non-overridable local execution contract for later candidate training."""

    conda_environment: str = "TCFM"
    device: str = "cuda:0"
    expected_device_name: str = "NVIDIA GeForce RTX 5080"
    visible_cuda_device_count: int = 1
    process_count: int = 1
    deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False
    dataset_partition: str = "train"
    official_image_labels_allowed: bool = False
    output_root: str = HYPERLEAF_CANDIDATE_OUTPUT_ROOT.as_posix()
    checkpoint_boundary: str = "epoch"


@dataclass(frozen=True)
class HyperLeafCandidateSpec:
    """One named branch rebuilt by the task-14.11 runner."""

    branch: HyperLeafCandidateBranch
    model_key: str
    inference_branch: str
    style_key: str
    requires_pca: bool
    consistency_weight: float


@dataclass(frozen=True)
class GswCandidateConfig:
    """Complete immutable Gsw route/loss/transform candidate declaration."""

    candidate_id: str
    route: GswRoute
    model_key: str
    inference_branch: str
    loss_kind: GswLossKind
    target_transform: GswTargetTransform
    consistency_weight: float
    style_key: str = "mild"


@dataclass(frozen=True)
class HyperLeafFoldPreprocessing:
    """Target and optional PCA states fitted from one exact train boundary."""

    boundary: FitBoundary
    target_normalization: FittedArtifact
    pca: Optional[FittedArtifact]


HYPERLEAF_CANDIDATE_SPECS: Mapping[HyperLeafCandidateBranch, HyperLeafCandidateSpec] = {
    HyperLeafCandidateBranch.INDEPENDENT_TARGET: HyperLeafCandidateSpec(
        HyperLeafCandidateBranch.INDEPENDENT_TARGET,
        INDEPENDENT_TARGET_MODEL_KEY,
        RGB_BRANCH,
        "none",
        False,
        0.0,
    ),
    HyperLeafCandidateBranch.STYLE_AUGMENTATION: HyperLeafCandidateSpec(
        HyperLeafCandidateBranch.STYLE_AUGMENTATION,
        INDEPENDENT_TARGET_MODEL_KEY,
        RGB_BRANCH,
        "mild",
        False,
        0.0,
    ),
    HyperLeafCandidateBranch.TRAIN_FOLD_PCA: HyperLeafCandidateSpec(
        HyperLeafCandidateBranch.TRAIN_FOLD_PCA,
        PCA_INDEPENDENT_TARGET_MODEL_KEY,
        PCA_BRANCH,
        "mild",
        True,
        0.0,
    ),
    HyperLeafCandidateBranch.RGB_PCA_CONSISTENCY: HyperLeafCandidateSpec(
        HyperLeafCandidateBranch.RGB_PCA_CONSISTENCY,
        RGB_PCA_CONSISTENCY_MODEL_KEY,
        TARGET_ROUTED_BRANCH,
        "mild",
        True,
        0.10,
    ),
}


def _gsw_route_details(route: GswRoute) -> tuple[str, str, float]:
    if route is GswRoute.RGB:
        return INDEPENDENT_TARGET_MODEL_KEY, RGB_BRANCH, 0.0
    if route is GswRoute.PCA:
        return PCA_INDEPENDENT_TARGET_MODEL_KEY, PCA_BRANCH, 0.0
    if route is GswRoute.RGB_PCA_CONSISTENCY_005:
        return RGB_PCA_CONSISTENCY_MODEL_KEY, AVERAGE_BRANCH, 0.05
    if route is GswRoute.RGB_PCA_CONSISTENCY_010:
        return RGB_PCA_CONSISTENCY_MODEL_KEY, AVERAGE_BRANCH, 0.10
    raise ValueError(f"unsupported Gsw route: {route!r}")


def _build_gsw_candidates() -> tuple[GswCandidateConfig, ...]:
    candidates = []
    for route in GswRoute:
        model_key, inference_branch, consistency_weight = _gsw_route_details(route)
        for loss_kind in GswLossKind:
            for target_transform in GswTargetTransform:
                candidate_id = (
                    f"gsw::{route.value}::{loss_kind.value}::{target_transform.value}"
                )
                candidates.append(
                    GswCandidateConfig(
                        candidate_id=candidate_id,
                        route=route,
                        model_key=model_key,
                        inference_branch=inference_branch,
                        loss_kind=loss_kind,
                        target_transform=target_transform,
                        consistency_weight=consistency_weight,
                    )
                )
    return tuple(sorted(candidates, key=lambda item: item.candidate_id.encode("utf-8")))


GSW_CANDIDATE_CONFIGS = _build_gsw_candidates()


def _candidate_option(key: str, **parameters: object) -> CandidateOption:
    return CandidateOption.create(key, parameters=parameters)


HYPERLEAF_GSW_CANDIDATE_SPACE = CandidateSpace.create(
    candidate_space_key="hyperleaf.gsw.route-loss-transform.v1",
    model_candidates=(
        _candidate_option(INDEPENDENT_TARGET_MODEL_KEY),
        _candidate_option(PCA_INDEPENDENT_TARGET_MODEL_KEY),
        _candidate_option(RGB_PCA_CONSISTENCY_MODEL_KEY),
    ),
    representation_candidates=(
        _candidate_option("visible-rgb-global-spectrum", pca_required=False),
        _candidate_option("pca-rgb-global-spectrum", pca_required=True),
        _candidate_option("rgb-pca-global-spectrum", pca_required=True),
    ),
    loss_candidates=tuple(_candidate_option(item.value) for item in GswLossKind),
    augmentation_candidates=tuple(
        _candidate_option(key, config=config.__dict__)
        for key, config in HYPERLEAF_STYLE_CONFIGS.items()
    ),
    target_route_candidates=tuple(
        _candidate_option(
            route.value,
            model_key=_gsw_route_details(route)[0],
            inference_branch=_gsw_route_details(route)[1],
            consistency_weight=_gsw_route_details(route)[2],
        )
        for route in GswRoute
    ),
    inference_branch_candidates=tuple(
        _candidate_option(branch)
        for branch in (RGB_BRANCH, PCA_BRANCH, AVERAGE_BRANCH, TARGET_ROUTED_BRANCH)
    ),
    threshold_candidates=(_candidate_option("not-applicable-regression"),),
    calibration_candidates=(_candidate_option("not-applicable-regression"),),
    objective_metric="r2.Gsw",
    metric_direction=MetricDirection.MAXIMIZE,
    frozen_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
)


class HyperLeafCandidateRunner:
    """G:-only integration surface with no screening or outer evaluation API."""

    execution_contract = HyperLeafCandidateExecutionContract()
    output_root = HYPERLEAF_CANDIDATE_OUTPUT_ROOT

    def __init__(
        self,
        workspace_root: Path | str = Path(r"G:\Datasets\HyperCross"),
        *,
        registry: Optional[ModelRegistry] = None,
        selection_controller: Optional[SelectionController] = None,
    ) -> None:
        windows_root = PureWindowsPath(str(workspace_root))
        if str(windows_root).casefold() != str(_EXPECTED_WORKSPACE).casefold():
            raise HyperLeafCandidateError(
                r"candidate workspace must be exactly G:\Datasets\HyperCross; F: and other roots are forbidden"
            )
        self.workspace_root = Path(str(windows_root)).resolve(strict=False)
        self.artifact_root = self.workspace_root / Path(*self.output_root.parts)
        self.registry = registry or create_hyperleaf_candidate_registry()
        self.selection_controller = selection_controller or SelectionController()
        self._validate_declarations()

    @property
    def model_keys(self) -> tuple[str, ...]:
        return self.registry.model_keys

    @property
    def candidate_space(self) -> CandidateSpace:
        """Return the hash-bound space frozen before any outer evaluation."""

        return HYPERLEAF_GSW_CANDIDATE_SPACE

    @property
    def gsw_candidates(self) -> tuple[GswCandidateConfig, ...]:
        return GSW_CANDIDATE_CONFIGS

    def resolve_branch(
        self, branch: HyperLeafCandidateBranch | str
    ) -> tuple[HyperLeafCandidateSpec, ModelRegistration]:
        try:
            normalized = HyperLeafCandidateBranch(branch)
            spec = HYPERLEAF_CANDIDATE_SPECS[normalized]
        except (KeyError, ValueError) as exc:
            raise HyperLeafCandidateError(f"unknown HyperLeaf candidate branch: {branch!r}") from exc
        return spec, self.registry.resolve(spec.model_key)

    def build_branch(self, branch: HyperLeafCandidateBranch | str):
        spec, _ = self.resolve_branch(branch)
        return self.registry.build(spec.model_key)

    def validate_launch(
        self,
        environment: EnvironmentLock,
        *,
        conda_environment: str,
        process_count: int,
        deterministic_algorithms: bool,
        cudnn_benchmark: bool,
    ) -> None:
        """Reject launches outside deterministic TCFM/one-RTX-5080 execution."""

        expected = self.execution_contract
        failures = []
        if conda_environment != expected.conda_environment:
            failures.append("conda_environment")
        if process_count != expected.process_count:
            failures.append("process_count")
        if not environment.cuda_available or environment.cuda_device_count != 1:
            failures.append("visible_cuda_device_count")
        if f"cuda:0:{expected.expected_device_name}" not in environment.device_identity:
            failures.append("device_identity")
        if deterministic_algorithms is not expected.deterministic_algorithms:
            failures.append("deterministic_algorithms")
        if cudnn_benchmark is not expected.cudnn_benchmark:
            failures.append("cudnn_benchmark")
        if failures:
            raise HyperLeafCandidateError(
                "candidate launch differs from frozen execution contract: "
                + ", ".join(failures)
            )

    @staticmethod
    def inner_fit_boundary(outer_fold: OuterFold, inner_fold_id: str) -> FitBoundary:
        try:
            inner = next(
                fold for fold in outer_fold.inner_folds if fold.fold_id == inner_fold_id
            )
        except StopIteration as exc:
            raise HyperLeafCandidateError(f"unknown Inner_Fold: {inner_fold_id!r}") from exc
        return FitBoundary(
            fold_id=outer_fold.fold_id,
            scope=FitBoundaryScope.INNER_TRAIN,
            inner_fold_id=inner.fold_id,
            allowed_fit_sample_ids=inner.train_ids,
            outer_test_sample_ids=outer_fold.outer_test_ids,
        )

    @staticmethod
    def outer_fit_boundary(outer_fold: OuterFold) -> FitBoundary:
        return FitBoundary(
            fold_id=outer_fold.fold_id,
            scope=FitBoundaryScope.OUTER_TRAIN,
            allowed_fit_sample_ids=outer_fold.outer_train_ids,
            outer_test_sample_ids=outer_fold.outer_test_ids,
        )

    def fit_preprocessing(
        self,
        representations: HyperLeafRepresentationBuilder,
        outer_fold: OuterFold,
        *,
        dataset_hash: str,
        protocol_hash: str,
        branch: HyperLeafCandidateBranch | str,
        inner_fold_id: Optional[str] = None,
    ) -> HyperLeafFoldPreprocessing:
        """Fit target normalization and optional PCA on exactly one train boundary."""

        spec, _ = self.resolve_branch(branch)
        if representations.adapter.partition is not HyperLeafPartition.TRAIN:
            raise HyperLeafCandidateError(
                "candidate preprocessing accepts only the labeled training partition"
            )
        boundary = (
            self.outer_fit_boundary(outer_fold)
            if inner_fold_id is None
            else self.inner_fit_boundary(outer_fold, inner_fold_id)
        )
        fit_ids = boundary.allowed_fit_sample_ids
        target = representations.fit_target_normalization(
            fit_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
        )
        pca = (
            representations.fit_pca_pseudo_rgb(
                fit_ids,
                boundary,
                dataset_hash=dataset_hash,
                protocol_hash=protocol_hash,
            )
            if spec.requires_pca
            else None
        )
        if target.fit_sample_ids != fit_ids or (pca is not None and pca.fit_sample_ids != fit_ids):
            raise HyperLeafCandidateError(
                "fitted preprocessing does not match the exact protocol train boundary"
            )
        return HyperLeafFoldPreprocessing(boundary, target, pca)

    @staticmethod
    def _selection_candidate(config: GswCandidateConfig) -> SelectionCandidate:
        return SelectionCandidate(
            candidate_id=config.candidate_id,
            configuration={
                "target": "Gsw",
                "route": config.route.value,
                "model_key": config.model_key,
                "inference_branch": config.inference_branch,
                "loss": config.loss_kind.value,
                "target_transform": config.target_transform.value,
                "consistency_weight": config.consistency_weight,
                "style_key": config.style_key,
                "selection_boundary": "inner_varieties_only",
            },
        )

    def freeze_gsw_selection(
        self,
        outer_fold: OuterFold,
        *,
        candidate_scores: Mapping[str, float],
        inner_source_sample_ids: Iterable[str],
        inner_prediction_hash: str,
        decision_id: Optional[str] = None,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Freeze the complete Gsw space from exact inner-variety OOF evidence."""

        expected_source = {
            sample_id
            for inner_fold in outer_fold.inner_folds
            for sample_id in inner_fold.validation_ids
        }
        supplied_source = set(inner_source_sample_ids)
        if not outer_fold.inner_folds or supplied_source != expected_source:
            raise HyperLeafCandidateError(
                "Gsw selection requires exact complete inner-variety OOF evidence"
            )
        candidates = tuple(self._selection_candidate(config) for config in self.gsw_candidates)
        return self.selection_controller.freeze_selection(
            outer_fold,
            candidates=candidates,
            candidate_scores=candidate_scores,
            source_sample_ids=supplied_source,
            source_prediction_hash=inner_prediction_hash,
            selection_policy=GSW_SELECTION_POLICY,
            decision_id=decision_id,
            frozen_utc=frozen_utc,
        )

    def prepare_outer_refit(
        self, outer_fold: OuterFold, decision: SelectionDecision
    ):
        """Bind the unchanged inner-frozen decision to all outer-train samples."""

        return self.selection_controller.prepare_refit(outer_fold, decision)

    def save_recovery_checkpoint(
        self,
        recovery_manager: RecoveryManager,
        *,
        run_id: str,
        epoch: int,
        model_state: bytes,
        optimizer_state: bytes,
        scheduler_state: bytes,
        stage_state: bytes,
        created_utc: Optional[datetime] = None,
    ) -> RecoveryArtifact:
        """Atomically save all resumable state at a monotonically increasing epoch."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise HyperLeafCandidateError("recovery epoch must be a positive integer")
        latest = recovery_manager.registry.latest_run(run_id)
        if latest.frozen_identity.model_key not in self.model_keys:
            raise HyperLeafCandidateError("run Model_Key is not a declared HyperLeaf candidate")
        recovery_id = f"recovery-{run_id}-epoch-{epoch:06d}"
        relative_path = (
            self.output_root
            / "recovery"
            / run_id
            / f"epoch-{epoch:06d}.canonical.json"
        ).as_posix()
        return recovery_manager.save_checkpoint_boundary(
            run_id=run_id,
            recovery_id=recovery_id,
            relative_path=relative_path,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            stage_state=stage_state,
            last_completed_stage=f"epoch-{epoch:06d}",
            completed_checkpoint_boundary=epoch,
            created_utc=created_utc,
        )

    @staticmethod
    def resume(
        recovery_manager: RecoveryManager,
        run_id: str,
        observed_identity: FrozenRunIdentity,
    ) -> RestoredRecoveryState:
        """Delegate exact frozen-identity verification before restoring state."""

        return recovery_manager.resume(run_id, observed_identity)

    def metadata(self) -> dict[str, object]:
        return {
            "output_root": self.output_root.as_posix(),
            "execution": self.execution_contract.__dict__,
            "branches": tuple(spec.branch.value for spec in HYPERLEAF_CANDIDATE_SPECS.values()),
            "model_keys": self.model_keys,
            "style_options": tuple(sorted(HYPERLEAF_STYLE_CONFIGS)),
            "gsw_candidate_ids": tuple(config.candidate_id for config in self.gsw_candidates),
            "candidate_space_key": self.candidate_space.candidate_space_key,
            "candidate_space_hash": self.candidate_space.candidate_space_hash,
            "gsw_selection_policy": GSW_SELECTION_POLICY.model_dump(mode="json"),
            "fit_policy": "inner-train for selection; complete outer-train for frozen refit",
            "official_test_labels_allowed": False,
            "legacy_src_imported": False,
        }

    def _validate_declarations(self) -> None:
        declared_keys = {spec.model_key for spec in HYPERLEAF_CANDIDATE_SPECS.values()}
        if declared_keys != set(self.registry.model_keys):
            raise HyperLeafCandidateError(
                "candidate branch declarations and model registry keys differ"
            )
        if any(spec.style_key not in HYPERLEAF_STYLE_CONFIGS for spec in HYPERLEAF_CANDIDATE_SPECS.values()):
            raise HyperLeafCandidateError("candidate branch references undeclared style config")
        expected_count = len(GswRoute) * len(GswLossKind) * len(GswTargetTransform)
        if len(GSW_CANDIDATE_CONFIGS) != expected_count or len(
            {item.candidate_id for item in GSW_CANDIDATE_CONFIGS}
        ) != expected_count:
            raise HyperLeafCandidateError("Gsw route/loss/transform space is incomplete")
        HYPERLEAF_GSW_CANDIDATE_SPACE.require_integrity()


__all__ = [
    "GSW_CANDIDATE_CONFIGS",
    "GSW_SELECTION_POLICY",
    "GswCandidateConfig",
    "GswRoute",
    "HYPERLEAF_CANDIDATE_OUTPUT_ROOT",
    "HYPERLEAF_CANDIDATE_SPECS",
    "HYPERLEAF_GSW_CANDIDATE_SPACE",
    "HyperLeafCandidateBranch",
    "HyperLeafCandidateError",
    "HyperLeafCandidateExecutionContract",
    "HyperLeafCandidateRunner",
    "HyperLeafCandidateSpec",
    "HyperLeafFoldPreprocessing",
]
