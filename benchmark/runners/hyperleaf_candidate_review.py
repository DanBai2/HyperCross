"""Immutable, planning-only HyperLeaf candidate review and control wiring.

The module freezes the task-17.6 review space before evaluation.  It has no
training or label-reading entry point.  Every candidate/control pair carries an
identical protocol-derived comparison context, while each run configuration
records its intentional model, route, consistency, and augmentation changes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Optional

from pydantic import Field, field_validator, model_validator

from benchmark.engine.selection import SelectionController
from benchmark.protocols.hyperleaf_lovo import HYPERLEAF_LOVO_PROTOCOL_KEY
from benchmark.registry.candidate_space import CandidateSpaceRegistry
from benchmark.schema import (
    CandidateOption,
    CandidateSpace,
    CheckpointRetentionPolicy,
    DETERMINISTIC_TIE_BREAK,
    EnvironmentLock,
    ExecutionMode,
    FrozenCandidateSpaceRegistry,
    FrozenRunConfig,
    FrameworkDeterminismSettings,
    MetricDirection,
    NumericPrecision,
    OuterFold,
    ProtocolManifest,
    ResourceBudget,
    SelectionCandidate,
    SelectionDecision,
    SelectionPolicy,
    hash_object,
)
from benchmark.schema.environment import FrozenSchemaModel


_EXPECTED_WORKSPACE = PureWindowsPath(r"G:\Datasets\HyperCross")
HYPERLEAF_REVIEW_OUTPUT_ROOT = PurePosixPath("runs_v2/candidates/hyperleaf/review")
HYPERLEAF_REVIEW_SPACE_KEY = "hyperleaf.candidate-review.v1"
HYPERLEAF_RGB_PCA_REPRESENTATION_KEY = "hyperleaf.visible-rgb-train-fold-pca-dual-view.v1"
HYPERLEAF_RGB_PCA_MODALITIES = ("visible_rgb", "train_fold_pca_pseudo_rgb")


class HyperLeafReviewRole(str, Enum):
    CANDIDATE = "candidate"
    MATCHED_INPUT_CONTROL = "matched_input_control"
    NEGATIVE_ABLATION = "negative_ablation"


class HyperLeafModelBranch(str, Enum):
    TARGET_ROUTED_RGB_PCA_CONSISTENCY = "target_routed_rgb_pca_consistency"
    INDEPENDENT_EXPERTS = "independent_target_experts"
    SHARED_HEADS = "shared_encoder_target_heads"
    GLOBAL_FUSION_CONTROL = "global_fusion_matched_control"


class HyperLeafAugmentation(str, Enum):
    STYLE = "style"
    STRONG = "strong"
    MIXED_P03 = "mixed_p03"
    MIXED_P01 = "mixed_p01"
    NONE = "none"


class HyperLeafReviewError(ValueError):
    """A review declaration, selection, launch, or comparison is invalid."""

    def __init__(self, message: str, *, mismatched_fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.mismatched_fields = mismatched_fields


class HyperLeafReviewExecutionContract(FrozenSchemaModel):
    """Non-overridable execution requirements for any later candidate run."""

    conda_environment: str = "TCFM"
    device: str = "cuda:0"
    expected_device_name: str = "NVIDIA GeForce RTX 5080"
    visible_cuda_device_count: int = 1
    process_count: int = 1
    deterministic_execution: bool = True
    official_label_access: bool = False
    output_root: str = HYPERLEAF_REVIEW_OUTPUT_ROOT.as_posix()
    checkpoint_boundary: str = "inner_fold_epoch"
    recovery_state_components: tuple[str, ...] = (
        "model",
        "optimizer",
        "scheduler",
        "stage",
    )

    @field_validator("output_root")
    @classmethod
    def require_relative_output(cls, value: str) -> str:
        windows = PureWindowsPath(value)
        posix = PurePosixPath(value)
        if windows.is_absolute() or windows.drive or posix.is_absolute() or ".." in posix.parts:
            raise ValueError("candidate review output_root must be workspace-relative")
        return value


class HyperLeafReviewContext(FrozenSchemaModel):
    """Complete identity that must match for candidate/control comparisons."""

    dataset_hash: str
    protocol_hash: str
    protocol_key: str
    seed_set: tuple[int, ...] = Field(min_length=1)
    fold_manifest_hashes: tuple[tuple[str, str], ...] = Field(min_length=1)
    metric_configuration_hash: str
    resource_budget: ResourceBudget
    context_hash: str

    @classmethod
    def create(
        cls,
        protocol: ProtocolManifest,
        *,
        metric_configuration: Mapping[str, Any],
        resource_budget: ResourceBudget,
    ) -> "HyperLeafReviewContext":
        if protocol.protocol_key != HYPERLEAF_LOVO_PROTOCOL_KEY:
            raise HyperLeafReviewError(
                "HyperLeaf candidate review requires the inner-variety LOVO protocol"
            )
        payload: dict[str, Any] = {
            "dataset_hash": protocol.dataset_hash,
            "protocol_hash": protocol.protocol_hash,
            "protocol_key": protocol.protocol_key,
            "seed_set": protocol.seed_set,
            "fold_manifest_hashes": tuple(
                (fold.fold_id, hash_object(fold)) for fold in protocol.folds
            ),
            "metric_configuration_hash": hash_object(dict(metric_configuration)),
            "resource_budget": resource_budget,
        }
        payload["context_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"context_hash"})

    def require_integrity(self) -> None:
        if self.context_hash != hash_object(self.hash_payload()):
            raise HyperLeafReviewError("review context hash does not match its identities")

    @model_validator(mode="after")
    def validate_context(self) -> "HyperLeafReviewContext":
        canonical_seeds = tuple(sorted(set(self.seed_set)))
        if canonical_seeds != self.seed_set:
            raise ValueError("seed_set must be sorted and unique")
        canonical_folds = tuple(
            sorted(self.fold_manifest_hashes, key=lambda item: item[0].encode("utf-8"))
        )
        if canonical_folds != self.fold_manifest_hashes:
            raise ValueError("fold_manifest_hashes must be in canonical Fold_ID order")
        if len({fold_id for fold_id, _ in canonical_folds}) != len(canonical_folds):
            raise ValueError("fold_manifest_hashes contains duplicate Fold_IDs")
        self.require_integrity()
        return self


class HyperLeafTargetRoute(FrozenSchemaModel):
    target_name: str
    inference_branch: str


class HyperLeafReviewSpec(FrozenSchemaModel):
    """One explicit candidate, matched control, or negative ablation."""

    review_key: str
    role: HyperLeafReviewRole
    model_branch: HyperLeafModelBranch
    augmentation: HyperLeafAugmentation
    target_routes: tuple[HyperLeafTargetRoute, ...]
    consistency_setting: str
    consistency_weight: float = Field(ge=0.0)
    mixed_augmentation_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    matched_control_key: Optional[str] = None
    evidence_disposition: str

    @model_validator(mode="after")
    def validate_spec(self) -> "HyperLeafReviewSpec":
        targets = tuple(route.target_name for route in self.target_routes)
        if targets != ("GrainWeight", "Gsw", "PhiPS2"):
            raise ValueError("target_routes must declare GrainWeight, Gsw, and PhiPS2 in order")
        if self.role is HyperLeafReviewRole.MATCHED_INPUT_CONTROL:
            if self.matched_control_key is not None:
                raise ValueError("a matched-input control cannot point to another control")
        elif not self.matched_control_key:
            raise ValueError("every candidate and negative ablation requires a matched control")
        mixed = self.augmentation in {
            HyperLeafAugmentation.MIXED_P03,
            HyperLeafAugmentation.MIXED_P01,
        }
        if mixed != (self.mixed_augmentation_probability is not None):
            raise ValueError("mixed augmentation requires exactly one declared probability")
        if self.role is HyperLeafReviewRole.NEGATIVE_ABLATION:
            if self.evidence_disposition != "append_only_negative":
                raise ValueError("negative ablations must use append_only_negative disposition")
        return self


class HyperLeafReviewBinding(FrozenSchemaModel):
    """A hash-bound run configuration attached to one shared review context."""

    spec: HyperLeafReviewSpec
    context: HyperLeafReviewContext
    representation_key: str = HYPERLEAF_RGB_PCA_REPRESENTATION_KEY
    input_modalities: tuple[str, ...] = HYPERLEAF_RGB_PCA_MODALITIES
    run_config: FrozenRunConfig


class HyperLeafReviewPlan(FrozenSchemaModel):
    """Complete immutable review plan sealed before any outer evaluation."""

    context: HyperLeafReviewContext
    execution_contract: HyperLeafReviewExecutionContract
    bindings: tuple[HyperLeafReviewBinding, ...]
    candidate_space: CandidateSpace
    frozen_candidate_registry: FrozenCandidateSpaceRegistry
    plan_hash: str

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"plan_hash"})

    @model_validator(mode="after")
    def validate_plan(self) -> "HyperLeafReviewPlan":
        if self.plan_hash != hash_object(self.hash_payload()):
            raise ValueError("plan_hash does not match the HyperLeaf review plan")
        return self


_TARGETS_RGB = tuple(
    HyperLeafTargetRoute(target_name=name, inference_branch="visible_rgb")
    for name in ("GrainWeight", "Gsw", "PhiPS2")
)
_TARGETS_PCA = tuple(
    HyperLeafTargetRoute(target_name=name, inference_branch="train_fold_pca")
    for name in ("GrainWeight", "Gsw", "PhiPS2")
)
_TARGET_ROUTED = (
    HyperLeafTargetRoute(target_name="GrainWeight", inference_branch="visible_rgb"),
    HyperLeafTargetRoute(target_name="Gsw", inference_branch="rgb_pca_consistency"),
    HyperLeafTargetRoute(target_name="PhiPS2", inference_branch="train_fold_pca"),
)


def _spec(
    review_key: str,
    role: HyperLeafReviewRole,
    branch: HyperLeafModelBranch,
    augmentation: HyperLeafAugmentation,
    *,
    routes: tuple[HyperLeafTargetRoute, ...] = _TARGETS_RGB,
    consistency: str = "task_only",
    consistency_weight: float = 0.0,
    mixed_probability: Optional[float] = None,
    control: Optional[str] = None,
    disposition: str = "review_candidate",
) -> HyperLeafReviewSpec:
    return HyperLeafReviewSpec(
        review_key=review_key,
        role=role,
        model_branch=branch,
        augmentation=augmentation,
        target_routes=routes,
        consistency_setting=consistency,
        consistency_weight=consistency_weight,
        mixed_augmentation_probability=mixed_probability,
        matched_control_key=control,
        evidence_disposition=disposition,
    )


# Controls are explicit dual-view controls.  They preserve RGB/PCA inputs and
# augmentation while removing target routing, expert separation, or consistency.
_CONTROL_BY_AUGMENTATION = {
    HyperLeafAugmentation.STYLE: "control.global-fusion.style",
    HyperLeafAugmentation.STRONG: "control.global-fusion.strong",
    HyperLeafAugmentation.MIXED_P03: "control.global-fusion.mixed-p03",
    HyperLeafAugmentation.MIXED_P01: "control.global-fusion.mixed-p01",
    HyperLeafAugmentation.NONE: "control.global-fusion.none",
}

HYPERLEAF_REVIEW_SPECS: tuple[HyperLeafReviewSpec, ...] = (
    _spec(
        "candidate.target-routed-rgb-pca-consistency.style",
        HyperLeafReviewRole.CANDIDATE,
        HyperLeafModelBranch.TARGET_ROUTED_RGB_PCA_CONSISTENCY,
        HyperLeafAugmentation.STYLE,
        routes=_TARGET_ROUTED,
        consistency="prediction_consistency",
        consistency_weight=0.1,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.STYLE],
    ),
    _spec(
        "candidate.independent-experts.style",
        HyperLeafReviewRole.CANDIDATE,
        HyperLeafModelBranch.INDEPENDENT_EXPERTS,
        HyperLeafAugmentation.STYLE,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.STYLE],
    ),
    _spec(
        "ablation.shared-head.style",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.SHARED_HEADS,
        HyperLeafAugmentation.STYLE,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.STYLE],
        disposition="append_only_negative",
    ),
    _spec(
        "ablation.shared-head.strong",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.SHARED_HEADS,
        HyperLeafAugmentation.STRONG,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.STRONG],
        disposition="append_only_negative",
    ),
    _spec(
        "ablation.shared-head.mixed-p03",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.SHARED_HEADS,
        HyperLeafAugmentation.MIXED_P03,
        mixed_probability=0.3,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.MIXED_P03],
        disposition="append_only_negative",
    ),
    _spec(
        "ablation.shared-head.mixed-p01",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.SHARED_HEADS,
        HyperLeafAugmentation.MIXED_P01,
        mixed_probability=0.1,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.MIXED_P01],
        disposition="append_only_negative",
    ),
    _spec(
        "ablation.independent-experts.no-augmentation",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.INDEPENDENT_EXPERTS,
        HyperLeafAugmentation.NONE,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.NONE],
        disposition="append_only_negative",
    ),
    _spec(
        "ablation.independent-experts.strong",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.INDEPENDENT_EXPERTS,
        HyperLeafAugmentation.STRONG,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.STRONG],
        disposition="append_only_negative",
    ),
    _spec(
        "ablation.independent-experts.pca-no-consistency",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.INDEPENDENT_EXPERTS,
        HyperLeafAugmentation.STYLE,
        routes=_TARGETS_PCA,
        consistency="alignment_and_nce_disabled",
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.STYLE],
        disposition="append_only_negative",
    ),
    _spec(
        "ablation.target-routed.consistency-005",
        HyperLeafReviewRole.NEGATIVE_ABLATION,
        HyperLeafModelBranch.TARGET_ROUTED_RGB_PCA_CONSISTENCY,
        HyperLeafAugmentation.STYLE,
        routes=_TARGET_ROUTED,
        consistency="prediction_consistency",
        consistency_weight=0.05,
        control=_CONTROL_BY_AUGMENTATION[HyperLeafAugmentation.STYLE],
        disposition="append_only_negative",
    ),
    *(
        _spec(
            control_key,
            HyperLeafReviewRole.MATCHED_INPUT_CONTROL,
            HyperLeafModelBranch.GLOBAL_FUSION_CONTROL,
            augmentation,
            mixed_probability=(
                0.3 if augmentation is HyperLeafAugmentation.MIXED_P03
                else 0.1 if augmentation is HyperLeafAugmentation.MIXED_P01
                else None
            ),
            disposition="matched_input_control",
        )
        for augmentation, control_key in _CONTROL_BY_AUGMENTATION.items()
    ),
)


def _candidate_space(frozen_utc: datetime) -> CandidateSpace:
    specs = HYPERLEAF_REVIEW_SPECS
    return CandidateSpace.create(
        candidate_space_key=HYPERLEAF_REVIEW_SPACE_KEY,
        model_candidates=tuple(
            CandidateOption.create(branch.value)
            for branch in HyperLeafModelBranch
        ),
        representation_candidates=(
            CandidateOption.create(
                HYPERLEAF_RGB_PCA_REPRESENTATION_KEY,
                parameters={
                    "modalities": HYPERLEAF_RGB_PCA_MODALITIES,
                    "pca_fit_boundary": "inner_train_for_selection_outer_train_for_refit",
                },
            ),
        ),
        loss_candidates=tuple(
            CandidateOption.create(key, parameters={"weight": weight})
            for key, weight in (
                ("alignment_and_nce_disabled", 0.0),
                ("prediction_consistency_005", 0.05),
                ("prediction_consistency_010", 0.1),
                ("task_only", 0.0),
            )
        ),
        augmentation_candidates=tuple(
            CandidateOption.create(
                augmentation.value,
                parameters={
                    "mixed_probability": (
                        0.3 if augmentation is HyperLeafAugmentation.MIXED_P03
                        else 0.1 if augmentation is HyperLeafAugmentation.MIXED_P01
                        else None
                    )
                },
            )
            for augmentation in HyperLeafAugmentation
        ),
        target_route_candidates=tuple(
            CandidateOption.create(
                spec.review_key,
                parameters={
                    route.target_name: route.inference_branch
                    for route in spec.target_routes
                },
            )
            for spec in specs
        ),
        inference_branch_candidates=tuple(
            CandidateOption.create(branch)
            for branch in ("rgb_pca_consistency", "train_fold_pca", "visible_rgb")
        ),
        threshold_candidates=(CandidateOption.create("not_applicable_regression"),),
        calibration_candidates=(CandidateOption.create("none"),),
        objective_metric="mean-r2",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule=DETERMINISTIC_TIE_BREAK,
        frozen_utc=frozen_utc,
    )


def build_hyperleaf_review_plan(
    protocol: ProtocolManifest,
    *,
    metric_configuration: Mapping[str, Any],
    resource_budget: ResourceBudget,
    frozen_utc: datetime,
) -> HyperLeafReviewPlan:
    """Build and seal all review candidates and controls without running them."""

    context = HyperLeafReviewContext.create(
        protocol,
        metric_configuration=metric_configuration,
        resource_budget=resource_budget,
    )
    execution = HyperLeafReviewExecutionContract()
    retention = CheckpointRetentionPolicy(
        preserve_selected=True,
        preserve_last=True,
        preserve_recovery_state=True,
    )
    settings = FrameworkDeterminismSettings.for_mode(ExecutionMode.DETERMINISTIC)
    bindings = tuple(
        HyperLeafReviewBinding(
            spec=spec,
            context=context,
            run_config=FrozenRunConfig.create(
                model_key=f"hyperleaf.review.{spec.model_branch.value}",
                protocol_key=protocol.protocol_key,
                execution_mode=ExecutionMode.DETERMINISTIC,
                numeric_precision=NumericPrecision.FLOAT32,
                framework_settings=settings,
                input_volume_bytes=0,
                checkpoint_retention=retention,
                resource_budget=resource_budget,
                parameters={
                    "review_key": spec.review_key,
                    "role": spec.role.value,
                    "representation_key": HYPERLEAF_RGB_PCA_REPRESENTATION_KEY,
                    "input_modalities": HYPERLEAF_RGB_PCA_MODALITIES,
                    "augmentation": spec.augmentation.value,
                    "mixed_augmentation_probability": spec.mixed_augmentation_probability,
                    "target_routes": tuple(
                        (route.target_name, route.inference_branch)
                        for route in spec.target_routes
                    ),
                    "consistency_setting": spec.consistency_setting,
                    "consistency_weight": spec.consistency_weight,
                    "normalization_fit_boundary": "inner_train_for_selection_outer_train_for_refit",
                    "pca_fit_boundary": "inner_train_for_selection_outer_train_for_refit",
                    "official_label_access": False,
                    "context_hash": context.context_hash,
                },
            ),
        )
        for spec in HYPERLEAF_REVIEW_SPECS
    )
    space = _candidate_space(frozen_utc)
    registry = CandidateSpaceRegistry()
    registry.register(space)
    frozen_registry = registry.freeze_before_evaluation(frozen_utc=frozen_utc)
    payload: dict[str, Any] = {
        "context": context,
        "execution_contract": execution,
        "bindings": bindings,
        "candidate_space": space,
        "frozen_candidate_registry": frozen_registry,
    }
    payload["plan_hash"] = hash_object(payload)
    plan = HyperLeafReviewPlan.model_validate(payload)
    _validate_review_declarations(plan)
    return plan


def _validate_review_declarations(plan: HyperLeafReviewPlan) -> None:
    by_key = {binding.spec.review_key: binding for binding in plan.bindings}
    if len(by_key) != len(plan.bindings):
        raise HyperLeafReviewError("review plan contains duplicate review keys")
    for binding in plan.bindings:
        if binding.context != plan.context:
            raise HyperLeafReviewError("all review bindings must share one context")
        if binding.run_config.resource_budget != plan.context.resource_budget:
            raise HyperLeafReviewError("all review bindings must share one resource budget")
        control_key = binding.spec.matched_control_key
        if control_key is not None:
            if control_key not in by_key:
                raise HyperLeafReviewError(f"undeclared matched control: {control_key}")
            validate_hyperleaf_matched_control(binding, by_key[control_key])


def validate_hyperleaf_matched_control(
    candidate: HyperLeafReviewBinding,
    control: HyperLeafReviewBinding,
) -> None:
    """Reject any candidate/control drift in required comparison identities."""

    if candidate.spec.role is HyperLeafReviewRole.MATCHED_INPUT_CONTROL:
        raise HyperLeafReviewError("candidate binding cannot itself be a matched-input control")
    if control.spec.role is not HyperLeafReviewRole.MATCHED_INPUT_CONTROL:
        raise HyperLeafReviewError("control binding is not declared as a matched-input control")
    if candidate.spec.matched_control_key != control.spec.review_key:
        raise HyperLeafReviewError("control key does not match the candidate declaration")

    fields = (
        "dataset_hash",
        "protocol_hash",
        "protocol_key",
        "seed_set",
        "fold_manifest_hashes",
        "metric_configuration_hash",
        "resource_budget",
    )
    mismatches = [
        field
        for field in fields
        if getattr(candidate.context, field) != getattr(control.context, field)
    ]
    if candidate.representation_key != control.representation_key:
        mismatches.append("representation_key")
    if candidate.input_modalities != control.input_modalities:
        mismatches.append("input_modalities")
    if candidate.run_config.resource_budget != control.run_config.resource_budget:
        mismatches.append("run_config.resource_budget")
    if mismatches:
        ordered = tuple(sorted(mismatches, key=lambda value: value.encode("utf-8")))
        raise HyperLeafReviewError(
            "candidate/control comparison fields do not match: " + ", ".join(ordered),
            mismatched_fields=ordered,
        )
    candidate.context.require_integrity()
    control.context.require_integrity()


class HyperLeafCandidateReviewRunner:
    """Selection and launch validation over one immutable review plan."""

    def __init__(
        self,
        plan: HyperLeafReviewPlan,
        workspace_root: Path | str = Path(r"G:\Datasets\HyperCross"),
        *,
        selection_controller: Optional[SelectionController] = None,
    ) -> None:
        windows_root = PureWindowsPath(str(workspace_root))
        if str(windows_root).casefold() != str(_EXPECTED_WORKSPACE).casefold():
            raise HyperLeafReviewError(
                r"candidate review workspace must be exactly G:\Datasets\HyperCross"
            )
        self.workspace_root = Path(str(windows_root)).resolve(strict=False)
        self.artifact_root = self.workspace_root / Path(*HYPERLEAF_REVIEW_OUTPUT_ROOT.parts)
        self.plan = plan
        self.selection_controller = selection_controller or SelectionController()
        _validate_review_declarations(plan)

    def binding(self, review_key: str) -> HyperLeafReviewBinding:
        for binding in self.plan.bindings:
            if binding.spec.review_key == review_key:
                return binding
        raise HyperLeafReviewError(f"unknown HyperLeaf review key: {review_key!r}")

    def validate_launch(
        self,
        environment: EnvironmentLock,
        *,
        conda_environment: str,
        process_count: int,
    ) -> None:
        expected = self.plan.execution_contract
        failures = []
        if conda_environment != expected.conda_environment:
            failures.append("conda_environment")
        if process_count != expected.process_count:
            failures.append("process_count")
        if not environment.cuda_available or environment.cuda_device_count != 1:
            failures.append("visible_cuda_device_count")
        if f"cuda:0:{expected.expected_device_name}" not in environment.device_identity:
            failures.append("device_identity")
        if failures:
            raise HyperLeafReviewError(
                "candidate launch differs from frozen execution contract: "
                + ", ".join(failures)
            )

    def freeze_route_and_consistency(
        self,
        outer_fold: OuterFold,
        *,
        review_keys: Iterable[str],
        candidate_scores: Mapping[str, float],
        inner_source_sample_ids: Iterable[str],
        inner_prediction_hash: str,
        selection_policy: SelectionPolicy,
        decision_id: Optional[str] = None,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Freeze route/consistency from exact complete inner-variety OOF only."""

        bindings = tuple(self.binding(key) for key in review_keys)
        if not bindings:
            raise HyperLeafReviewError("route/consistency review needs at least one candidate")
        source_ids = tuple(inner_source_sample_ids)
        complete_inner_oof = {
            sample_id
            for inner_fold in outer_fold.inner_folds
            for sample_id in inner_fold.validation_ids
        }
        if set(source_ids) != complete_inner_oof or complete_inner_oof != set(
            outer_fold.outer_train_ids
        ):
            raise HyperLeafReviewError(
                "route/consistency selection requires exact inner-variety OOF evidence"
            )
        candidates = tuple(
            SelectionCandidate(
                candidate_id=binding.spec.review_key,
                configuration={
                    "review_key": binding.spec.review_key,
                    "model_branch": binding.spec.model_branch.value,
                    "augmentation": binding.spec.augmentation.value,
                    "target_routes": tuple(
                        (route.target_name, route.inference_branch)
                        for route in binding.spec.target_routes
                    ),
                    "consistency_setting": binding.spec.consistency_setting,
                    "consistency_weight": binding.spec.consistency_weight,
                    "matched_control_key": binding.spec.matched_control_key,
                    "context_hash": binding.context.context_hash,
                },
            )
            for binding in bindings
        )
        return self.selection_controller.freeze_selection(
            outer_fold,
            candidates=candidates,
            candidate_scores=candidate_scores,
            source_sample_ids=source_ids,
            source_prediction_hash=inner_prediction_hash,
            selection_policy=selection_policy,
            decision_id=decision_id,
            frozen_utc=frozen_utc,
        )

    def metadata(self) -> dict[str, Any]:
        contract = self.plan.execution_contract
        return {
            "candidate_space_key": self.plan.candidate_space.candidate_space_key,
            "candidate_space_hash": self.plan.candidate_space.candidate_space_hash,
            "context_hash": self.plan.context.context_hash,
            "protocol_key": self.plan.context.protocol_key,
            "output_root": contract.output_root,
            "conda_environment": contract.conda_environment,
            "device": contract.device,
            "expected_device_name": contract.expected_device_name,
            "process_count": contract.process_count,
            "deterministic_execution": contract.deterministic_execution,
            "checkpoint_boundary": contract.checkpoint_boundary,
            "recovery_state_components": contract.recovery_state_components,
            "official_label_access": contract.official_label_access,
            "selection_evidence": "complete_outer_train_inner_variety_oof_only",
        }


__all__ = [
    "HYPERLEAF_REVIEW_OUTPUT_ROOT",
    "HYPERLEAF_REVIEW_SPACE_KEY",
    "HYPERLEAF_REVIEW_SPECS",
    "HYPERLEAF_RGB_PCA_MODALITIES",
    "HYPERLEAF_RGB_PCA_REPRESENTATION_KEY",
    "HyperLeafAugmentation",
    "HyperLeafCandidateReviewRunner",
    "HyperLeafModelBranch",
    "HyperLeafReviewBinding",
    "HyperLeafReviewContext",
    "HyperLeafReviewError",
    "HyperLeafReviewExecutionContract",
    "HyperLeafReviewPlan",
    "HyperLeafReviewRole",
    "HyperLeafReviewSpec",
    "HyperLeafTargetRoute",
    "build_hyperleaf_review_plan",
    "validate_hyperleaf_matched_control",
]
