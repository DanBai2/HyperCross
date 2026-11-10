"""Immutable task-24.3 HyperLeaf optimization registry and promotion guards.

This module performs no training or evaluation. It binds a computationally
limited candidate/control set to the corrected task-24.2 protocol identity and
exposes guards that accept nested inner-OOF evidence only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

from pydantic import Field, field_validator, model_validator

from benchmark.models.hyperleaf_candidates import (
    AVERAGE_BRANCH,
    INDEPENDENT_TARGET_MODEL_KEY,
    PCA_BRANCH,
    PCA_INDEPENDENT_TARGET_MODEL_KEY,
    RGB_BRANCH,
    RGB_PCA_CONSISTENCY_MODEL_KEY,
    RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
    RGB_PCA_SPECTRUM_REPRESENTATION_KEY,
    RGB_SPECTRUM_PREPROCESSING_KEY,
    RGB_SPECTRUM_REPRESENTATION_KEY,
    PCA_SPECTRUM_PREPROCESSING_KEY,
    PCA_SPECTRUM_REPRESENTATION_KEY,
)
from benchmark.models.hyperleaf_optimization import (
    HETEROSCEDASTIC_MODEL_KEY,
    LEARNED_GATED_BRANCH,
    LEARNED_GATED_MODEL_KEY,
    hyperleaf_optimization_registrations,
)
from benchmark.registry.candidate_space import CandidateSpaceRegistry
from benchmark.remediation import load_remediation_template
from benchmark.schema import CandidateOption, CandidateSpace, MetricDirection
from benchmark.schema.canonical import hash_object
from benchmark.schema.environment import FrozenSchemaModel
from benchmark.transforms.hyperleaf_optimization import (
    HETEROSCEDASTIC_LOSS,
    MSE_LOSS,
    STANDARD_SPECTRAL_NORMALIZATION,
    TARGET_BALANCED_LOSS,
    TARGET_NAMES,
    VARIETY_ROBUST_SPECTRAL_NORMALIZATION,
)


WORKSPACE = Path(r"G:\Datasets\HyperCross")
_EXPECTED_WORKSPACE = PureWindowsPath(r"G:\Datasets\HyperCross")
BASELINE_PLAN_PATH = PurePosixPath(
    "runs_v2/remediation/v1/deterministic-float32/hyperleaf-nested/"
    "executions/task-24-2-20261027T101633949873Z/frozen-plan.json"
)
BASELINE_SUMMARY_PATH = PurePosixPath(
    "runs_v2/remediation/v1/deterministic-float32/hyperleaf-nested/"
    "latest-task-24.2-summary.json"
)
BASELINE_PROTOCOL_ARTIFACT = PurePosixPath(
    "runs_v2/remediation/v1/deterministic-float32/hyperleaf-nested/"
    "executions/task-24-2-20261027T101633949873Z/protocol.canonical.json"
)
PRECISION_PROFILE_PATH = PurePosixPath("configs/remediation/v1/deterministic-float32.json")
OUTPUT_ROOT = PurePosixPath("runs_v2/optimization/hyperleaf/v1/deterministic-float32")
RECOVERY_ROOT = OUTPUT_ROOT / "recovery"
ARTIFACT_ROOT = PurePosixPath("artifacts/optimization/hyperleaf/v1/deterministic-float32")
CANDIDATE_SPACE_KEY = "hyperleaf.task-24.3.gsw-target-routing.v1"
FROZEN_UTC = datetime(2026, 10, 28, tzinfo=timezone.utc)
SEED_SET = (7, 42, 123)
FOLD_IDS = ("outer-heerup", "outer-kvium", "outer-rembrandt", "outer-sheriff")
TARGET_ROUTE_PCA_CONSISTENCY = (PCA_BRANCH, AVERAGE_BRANCH, PCA_BRANCH)
TARGET_ROUTE_EXISTING_FALLBACK = (RGB_BRANCH, AVERAGE_BRANCH, PCA_BRANCH)

CONTROL_RGB = "control::rgb-only::mse::standard-normalization"
CONTROL_PCA = "control::pca-only::mse::standard-normalization"
CONTROL_CONSISTENCY_005 = "control::pca-consistency-pca::consistency-005"
FALLBACK_EXISTING = "fallback::task-24.2::rgb-consistency-pca"
CANDIDATE_PCA_ROUTED = "candidate::pca-consistency-pca::consistency-010"
CANDIDATE_GATED = "candidate::pca-gated-pca::mse"
CANDIDATE_PHI_CONSISTENCY = "candidate::pca-consistency-consistency::mse"
CANDIDATE_ROBUST_NORMALIZATION = "candidate::pca-consistency-pca::variety-robust"
CANDIDATE_HETEROSCEDASTIC = "candidate::pca-consistency-pca::heteroscedastic"
CANDIDATE_TARGET_BALANCED = "candidate::pca-consistency-pca::target-balanced"


class HyperLeafOptimizationError(ValueError):
    """Raised when the frozen task-24.3 plan or evidence identity drifts."""


class CandidateRole(str, Enum):
    MATCHED_CONTROL = "matched_control"
    EXISTING_FALLBACK = "existing_leakage_safe_fallback"
    OPTIMIZATION = "optimization_candidate"


class HyperLeafOptimizationCandidate(FrozenSchemaModel):
    candidate_id: str = Field(min_length=1)
    role: CandidateRole
    model_key: str = Field(min_length=1)
    representation_key: str = Field(min_length=1)
    preprocessing_key: str = Field(min_length=1)
    spectral_normalization_key: str = Field(min_length=1)
    loss_key: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)
    target_routes: tuple[str, str, str]
    consistency_weight: float = Field(ge=0.0)
    changed_axis: str = Field(min_length=1)
    matched_guard_id: str | None = None
    matched_control_ids: tuple[str, ...] = ()
    comparison_context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> "HyperLeafOptimizationCandidate":
        payload = dict(values)
        payload["matched_control_ids"] = tuple(
            sorted(set(payload.get("matched_control_ids", ())), key=lambda item: item.encode("utf-8"))
        )
        payload["configuration_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_candidate(self) -> "HyperLeafOptimizationCandidate":
        if self.configuration_hash != hash_object(
            self.model_dump(mode="python", exclude={"configuration_hash"})
        ):
            raise ValueError("configuration_hash does not match candidate declaration")
        if self.role is CandidateRole.OPTIMIZATION and self.matched_guard_id is None:
            raise ValueError("optimization candidates require a matched promotion guard")
        if self.role is not CandidateRole.OPTIMIZATION and self.matched_guard_id is not None:
            raise ValueError("controls and fallback cannot be promoted against another guard")
        return self


class OptimizationComparisonContext(FrozenSchemaModel):
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_key: str
    protocol_artifact: str
    protocol_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_ids: tuple[str, ...]
    seed_set: tuple[int, ...]
    metric_configuration: Mapping[str, Any]
    metric_configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource_budget: Mapping[str, Any]
    resource_budget_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    precision_profile: str
    precision_profile_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    precision_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tie_break_rule: str
    epoch_tie_break_rule: str
    recovery_policy: Mapping[str, Any]
    recovery_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: str
    recovery_root: str
    artifact_root: str
    device_identity: str
    process_count: int
    official_labels_allowed: bool
    solution_csv_allowed: bool
    outer_metrics_allowed_for_selection: bool
    legacy_outer_metrics_allowed_for_selection: bool
    comparison_context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> "OptimizationComparisonContext":
        payload = dict(values)
        payload["metric_configuration_hash"] = hash_object(payload["metric_configuration"])
        payload["resource_budget_hash"] = hash_object(payload["resource_budget"])
        payload["recovery_policy_hash"] = hash_object(payload["recovery_policy"])
        precision_identity = {
            "precision_profile": payload["precision_profile"],
            "precision_profile_config_hash": payload["precision_profile_config_hash"],
            "device_identity": payload["device_identity"],
            "process_count": payload["process_count"],
        }
        payload["precision_identity_hash"] = hash_object(precision_identity)
        payload["comparison_context_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_context(self) -> "OptimizationComparisonContext":
        payload = self.model_dump(mode="python", exclude={"comparison_context_hash"})
        if self.comparison_context_hash != hash_object(payload):
            raise ValueError("comparison_context_hash does not match frozen context")
        if self.seed_set != SEED_SET or self.fold_ids != FOLD_IDS:
            raise ValueError("task 24.3 requires exact corrected folds and seeds 7/42/123")
        if self.official_labels_allowed or self.solution_csv_allowed:
            raise ValueError("official labels and solution.csv must remain sealed")
        if self.outer_metrics_allowed_for_selection or self.legacy_outer_metrics_allowed_for_selection:
            raise ValueError("outer metrics cannot be optimization selection scores")
        for value in (self.protocol_artifact, self.precision_profile, self.output_root, self.recovery_root, self.artifact_root):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or str(value).casefold().startswith("f:"):
                raise ValueError("all task 24.3 paths must be profile-bound G:-workspace relative paths")
        return self


class PromotionGuard(FrozenSchemaModel):
    evidence_scope: str = "nested_inner_oof_only"
    metric_scope: str = "per_seed_pooled_oof"
    gsw_mean_delta_r2_minimum: float = 0.0100
    gsw_each_seed_delta_strictly_positive: bool = True
    grainweight_mean_delta_r2_minimum: float = -0.0100
    phips2_mean_delta_r2_minimum: float = -0.0100
    required_seed_set: tuple[int, ...] = SEED_SET
    missing_or_nonfinite_metric_disposition: str = "reject_and_append_negative_evidence"
    failed_guard_disposition: str = "retain_as_negative_evidence_no_outer_revival"


class ScreeningStage(FrozenSchemaModel):
    stage_key: str
    candidate_epoch_cap: int = Field(gt=0)
    seed_set: tuple[int, ...]
    fold_scope: str
    equal_budget_per_candidate: bool
    maximum_new_candidate_families_advanced: int = Field(gt=0)
    pruning_rule: str
    tie_break_rule: str


class HyperLeafOptimizationPlan(FrozenSchemaModel):
    schema_version: str = "1.0"
    task_id: str = "24.3"
    context: OptimizationComparisonContext
    models: tuple[Mapping[str, Any], ...]
    model_registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[HyperLeafOptimizationCandidate, ...]
    candidate_space: CandidateSpace
    frozen_candidate_registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    promotion_guard: PromotionGuard
    screening_schedule: tuple[ScreeningStage, ...]
    fallback_candidate_id: str
    design_basis: Mapping[str, Any]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_plan(self) -> "HyperLeafOptimizationPlan":
        if self.plan_hash != hash_object(self.model_dump(mode="python", exclude={"plan_hash"})):
            raise ValueError("plan_hash does not match immutable task-24.3 registry")
        candidates = {item.candidate_id: item for item in self.candidates}
        if len(candidates) != len(self.candidates) or len(candidates) > 10:
            raise ValueError("candidate registry must be unique and capped at ten entries")
        if self.fallback_candidate_id not in candidates or candidates[self.fallback_candidate_id].role is not CandidateRole.EXISTING_FALLBACK:
            raise ValueError("fallback must reference the predeclared existing leakage-safe candidate")
        model_keys = {str(item["model_key"]) for item in self.models}
        for candidate in self.candidates:
            if candidate.comparison_context_hash != self.context.comparison_context_hash:
                raise ValueError("all candidates must share one comparison context")
            if candidate.model_key not in model_keys:
                raise ValueError(f"candidate model is absent from immutable registry: {candidate.model_key}")
            for control_id in candidate.matched_control_ids:
                if control_id not in candidates:
                    raise ValueError(f"undeclared matched control: {control_id}")
            if candidate.matched_guard_id is not None and candidate.matched_guard_id not in candidates:
                raise ValueError(f"undeclared matched guard: {candidate.matched_guard_id}")
            if candidate.role is CandidateRole.OPTIMIZATION and candidate.target_routes[0] != PCA_BRANCH:
                raise ValueError("every new candidate must route GrainWeight through PCA")
        if self.design_basis.get("selection_score_allowed") is not False:
            raise ValueError("historical design values must not become selection scores")
        return self


class SeedTargetR2(FrozenSchemaModel):
    seed: int
    grainweight: float
    gsw: float
    phips2: float

    @model_validator(mode="after")
    def require_finite(self) -> "SeedTargetR2":
        if not all(math.isfinite(value) for value in (self.grainweight, self.gsw, self.phips2)):
            raise ValueError("promotion metrics must be finite")
        return self


class InnerOofPromotionEvidence(FrozenSchemaModel):
    candidate_id: str
    comparison_context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_scope: str
    seed_metrics: tuple[SeedTargetR2, ...]
    source_prediction_hashes: tuple[str, ...]
    outer_metrics_used: bool
    official_labels_used: bool
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> "InnerOofPromotionEvidence":
        payload = dict(values)
        payload["seed_metrics"] = tuple(sorted(payload["seed_metrics"], key=lambda item: item.seed))
        payload["source_prediction_hashes"] = tuple(sorted(payload["source_prediction_hashes"]))
        payload["evidence_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_evidence(self) -> "InnerOofPromotionEvidence":
        if tuple(item.seed for item in self.seed_metrics) != SEED_SET:
            raise ValueError("promotion evidence requires complete seeds 7/42/123")
        if len(self.source_prediction_hashes) != len(SEED_SET) or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_prediction_hashes
        ):
            raise ValueError("one valid inner prediction hash is required per seed")
        if self.outer_metrics_used or self.official_labels_used or self.evidence_scope != "nested_inner_oof_only":
            raise ValueError("promotion evidence must be nested inner OOF only")
        if self.evidence_hash != hash_object(self.model_dump(mode="python", exclude={"evidence_hash"})):
            raise ValueError("evidence_hash does not match promotion evidence")
        return self


class PromotionDecision(FrozenSchemaModel):
    candidate_id: str
    matched_guard_id: str
    passed: bool
    mean_delta_r2: Mapping[str, float]
    gsw_seed_deltas: tuple[tuple[int, float], ...]
    failed_guards: tuple[str, ...]
    disposition: str
    candidate_evidence_hash: str
    guard_evidence_hash: str
    decision_hash: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option(key: str, **parameters: Any) -> CandidateOption:
    return CandidateOption.create(key, parameters=parameters)


def _candidate(
    context: OptimizationComparisonContext,
    candidate_id: str,
    role: CandidateRole,
    model_key: str,
    representation_key: str,
    preprocessing_key: str,
    normalization_key: str,
    loss_key: str,
    target_routes: tuple[str, str, str],
    consistency_weight: float,
    changed_axis: str,
    *,
    matched_guard_id: str | None = None,
    matched_control_ids: Sequence[str] = (),
    output_contract: str = "three_standardized_target_means",
) -> HyperLeafOptimizationCandidate:
    return HyperLeafOptimizationCandidate.create(
        candidate_id=candidate_id,
        role=role,
        model_key=model_key,
        representation_key=representation_key,
        preprocessing_key=preprocessing_key,
        spectral_normalization_key=normalization_key,
        loss_key=loss_key,
        output_contract=output_contract,
        target_routes=target_routes,
        consistency_weight=consistency_weight,
        changed_axis=changed_axis,
        matched_guard_id=matched_guard_id,
        matched_control_ids=tuple(matched_control_ids),
        comparison_context_hash=context.comparison_context_hash,
    )


def _build_candidates(context: OptimizationComparisonContext) -> tuple[HyperLeafOptimizationCandidate, ...]:
    shared_controls = (CONTROL_RGB, CONTROL_PCA, CONTROL_CONSISTENCY_005)
    values = (
        _candidate(context, CONTROL_RGB, CandidateRole.MATCHED_CONTROL, INDEPENDENT_TARGET_MODEL_KEY,
                   RGB_SPECTRUM_REPRESENTATION_KEY, RGB_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   (RGB_BRANCH, RGB_BRANCH, RGB_BRANCH), 0.0, "rgb_representation_ablation"),
        _candidate(context, CONTROL_PCA, CandidateRole.MATCHED_CONTROL, PCA_INDEPENDENT_TARGET_MODEL_KEY,
                   PCA_SPECTRUM_REPRESENTATION_KEY, PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   (PCA_BRANCH, PCA_BRANCH, PCA_BRANCH), 0.0, "pca_representation_ablation"),
        _candidate(context, CONTROL_CONSISTENCY_005, CandidateRole.MATCHED_CONTROL, RGB_PCA_CONSISTENCY_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   TARGET_ROUTE_PCA_CONSISTENCY, 0.05, "consistency_weight_control",
                   matched_control_ids=(CONTROL_RGB, CONTROL_PCA)),
        _candidate(context, FALLBACK_EXISTING, CandidateRole.EXISTING_FALLBACK, RGB_PCA_CONSISTENCY_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   TARGET_ROUTE_EXISTING_FALLBACK, 0.10, "preexisting_frozen_candidate",
                   matched_control_ids=shared_controls),
        _candidate(context, CANDIDATE_PCA_ROUTED, CandidateRole.OPTIMIZATION, RGB_PCA_CONSISTENCY_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   TARGET_ROUTE_PCA_CONSISTENCY, 0.10, "consistency_weight",
                   matched_guard_id=CONTROL_CONSISTENCY_005, matched_control_ids=shared_controls),
        _candidate(context, CANDIDATE_GATED, CandidateRole.OPTIMIZATION, LEARNED_GATED_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   (PCA_BRANCH, LEARNED_GATED_BRANCH, PCA_BRANCH), 0.10, "gsw_route",
                   matched_guard_id=CANDIDATE_PCA_ROUTED, matched_control_ids=(CANDIDATE_PCA_ROUTED, CONTROL_RGB, CONTROL_PCA)),
        _candidate(context, CANDIDATE_PHI_CONSISTENCY, CandidateRole.OPTIMIZATION, RGB_PCA_CONSISTENCY_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   (PCA_BRANCH, AVERAGE_BRANCH, AVERAGE_BRANCH), 0.10, "phips2_route",
                   matched_guard_id=CANDIDATE_PCA_ROUTED, matched_control_ids=(CANDIDATE_PCA_ROUTED, CONTROL_PCA)),
        _candidate(context, CANDIDATE_ROBUST_NORMALIZATION, CandidateRole.OPTIMIZATION, RGB_PCA_CONSISTENCY_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   VARIETY_ROBUST_SPECTRAL_NORMALIZATION, MSE_LOSS,
                   TARGET_ROUTE_PCA_CONSISTENCY, 0.10, "spectral_normalization",
                   matched_guard_id=CANDIDATE_PCA_ROUTED, matched_control_ids=(CANDIDATE_PCA_ROUTED,)),
        _candidate(context, CANDIDATE_HETEROSCEDASTIC, CandidateRole.OPTIMIZATION, HETEROSCEDASTIC_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, HETEROSCEDASTIC_LOSS,
                   TARGET_ROUTE_PCA_CONSISTENCY, 0.10, "loss_contract",
                   matched_guard_id=CANDIDATE_PCA_ROUTED, matched_control_ids=(CANDIDATE_PCA_ROUTED,),
                   output_contract="three_standardized_means_and_log_variances_clamped_minus8_plus8"),
        _candidate(context, CANDIDATE_TARGET_BALANCED, CandidateRole.OPTIMIZATION, RGB_PCA_CONSISTENCY_MODEL_KEY,
                   RGB_PCA_SPECTRUM_REPRESENTATION_KEY, RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
                   STANDARD_SPECTRAL_NORMALIZATION, TARGET_BALANCED_LOSS,
                   TARGET_ROUTE_PCA_CONSISTENCY, 0.10, "loss",
                   matched_guard_id=CANDIDATE_PCA_ROUTED, matched_control_ids=(CANDIDATE_PCA_ROUTED,)),
    )
    return tuple(sorted(values, key=lambda item: item.candidate_id.encode("utf-8")))


def _candidate_space(candidates: Sequence[HyperLeafOptimizationCandidate]) -> CandidateSpace:
    return CandidateSpace.create(
        candidate_space_key=CANDIDATE_SPACE_KEY,
        model_candidates=tuple(_option(value.model_key) for value in {item.model_key: item for item in candidates}.values()),
        representation_candidates=tuple(
            _option(value, fit_boundary="inner_train_for_screening_outer_train_for_refit")
            for value in sorted({item.representation_key for item in candidates}, key=str.encode)
        ),
        loss_candidates=tuple(
            _option(value, fit_boundary="inner_train_only_if_loss_has_learned_statistics")
            for value in sorted({item.loss_key for item in candidates}, key=str.encode)
        ),
        augmentation_candidates=(_option("mild-style-v1", frozen=True),),
        target_route_candidates=tuple(
            _option(item.candidate_id, target_routes=item.target_routes)
            for item in candidates
        ),
        inference_branch_candidates=tuple(
            _option(value)
            for value in (AVERAGE_BRANCH, LEARNED_GATED_BRANCH, PCA_BRANCH, RGB_BRANCH)
        ),
        threshold_candidates=(_option("not-applicable-regression"),),
        calibration_candidates=(_option("none"),),
        objective_metric="mean-over-seeds.inner-pooled-oof.r2.Gsw",
        metric_direction=MetricDirection.MAXIMIZE,
        tie_break_rule="candidate_id_utf8_ascending",
        frozen_utc=FROZEN_UTC,
    )


def build_hyperleaf_optimization_plan(
    workspace_root: Path | str = WORKSPACE,
) -> HyperLeafOptimizationPlan:
    """Build the hash-bound registry from task-24.2 identity files, not metrics."""

    windows_root = PureWindowsPath(str(workspace_root))
    if str(windows_root).casefold() != str(_EXPECTED_WORKSPACE).casefold():
        raise HyperLeafOptimizationError(r"task 24.3 workspace must be exactly G:\Datasets\HyperCross")
    root = Path(str(windows_root)).resolve(strict=False)
    baseline = json.loads((root / Path(*BASELINE_PLAN_PATH.parts)).read_text(encoding="utf-8"))
    summary = json.loads((root / Path(*BASELINE_SUMMARY_PATH.parts)).read_text(encoding="utf-8"))
    if summary.get("claim_status") != "primary_strict" or summary.get("execution_id") != baseline.get("execution_id"):
        raise HyperLeafOptimizationError("task-24.2 corrected primary-strict baseline is unavailable")
    if tuple(baseline.get("seed_set", ())) != SEED_SET:
        raise HyperLeafOptimizationError("task-24.2 baseline seed identity drift")
    template = load_remediation_template(root / Path(*PRECISION_PROFILE_PATH.parts))
    if template.config_hash != baseline["profile"]["config_hash"]:
        raise HyperLeafOptimizationError("task-24.2 precision profile identity drift")

    metric_configuration = {
        "schema_key": "hyperleaf-inner-pooled-oof-regression.v1",
        "targets": TARGET_NAMES,
        "primary_objective": "mean_over_seeds.inner_pooled_oof.r2.Gsw",
        "r2_definition": "coefficient_of_determination_from_complete_inner_oof_predictions",
        "seed_aggregation": "arithmetic_mean_of_per_seed_pooled_oof_r2",
        "undefined_r2": "null_zero_variance_target",
        "metric_direction": "maximize",
        "outer_metrics_permitted": False,
    }
    resource_budget = template.resource_budget.model_dump(mode="python")
    recovery_policy = {
        "checkpoint_interval_epochs": template.checkpoint_interval_epochs,
        "checkpoint_retention": template.checkpoint_retention.model_dump(mode="python"),
        "recovery_state_names": template.recovery_state_names,
        "persist_frozen_identity_before_training": template.persist_frozen_identity_before_training,
        "environment_compatibility_policy": template.environment_compatibility_policy,
        "checkpoint_boundary": "epoch",
    }
    context = OptimizationComparisonContext.create(
        dataset_hash=baseline["dataset_hash"],
        protocol_hash=baseline["protocol_hash"],
        protocol_key=baseline["protocol_key"],
        protocol_artifact=BASELINE_PROTOCOL_ARTIFACT.as_posix(),
        protocol_artifact_hash=_sha256_file(root / Path(*BASELINE_PROTOCOL_ARTIFACT.parts)),
        fold_ids=FOLD_IDS,
        seed_set=SEED_SET,
        metric_configuration=metric_configuration,
        resource_budget=resource_budget,
        precision_profile=PRECISION_PROFILE_PATH.as_posix(),
        precision_profile_config_hash=template.config_hash,
        tie_break_rule="candidate_id_utf8_ascending",
        epoch_tie_break_rule="earliest_epoch",
        recovery_policy=recovery_policy,
        output_root=OUTPUT_ROOT.as_posix(),
        recovery_root=RECOVERY_ROOT.as_posix(),
        artifact_root=ARTIFACT_ROOT.as_posix(),
        device_identity="cuda:0:NVIDIA GeForce RTX 5080",
        process_count=1,
        official_labels_allowed=False,
        solution_csv_allowed=False,
        outer_metrics_allowed_for_selection=False,
        legacy_outer_metrics_allowed_for_selection=False,
    )
    candidates = _build_candidates(context)
    space = _candidate_space(candidates)
    registry = CandidateSpaceRegistry()
    registry.register(space)
    frozen_registry = registry.freeze_before_evaluation(frozen_utc=FROZEN_UTC)
    models = tuple(
        {
            "model_key": registration.model_key,
            "class_fully_qualified_name": registration.identity.class_fully_qualified_name,
            "implementation_kind": registration.identity.implementation_kind.value,
            "source_revision": registration.identity.source_revision,
            "parameter_count": registration.identity.parameter_count,
            "representation_key": registration.input_contract.representation_key,
            "input_contract_hash": hash_object(registration.input_contract),
        }
        for registration in hyperleaf_optimization_registrations()
    )
    stages = (
        ScreeningStage(
            stage_key="paired-screen-20-epochs",
            candidate_epoch_cap=20,
            seed_set=SEED_SET,
            fold_scope="all_three_inner_variety_folds_in_each_outer_context",
            equal_budget_per_candidate=True,
            maximum_new_candidate_families_advanced=3,
            pruning_rule=(
                "rank optimization candidates by mean Gsw delta versus declared matched guard; "
                "require every seed delta > 0 and provisional GrainWeight/PhiPS2 mean deltas >= -0.0200; "
                "advance at most three candidates and every referenced guard"
            ),
            tie_break_rule="candidate_id_utf8_ascending",
        ),
        ScreeningStage(
            stage_key="promotion-screen-80-epochs",
            candidate_epoch_cap=80,
            seed_set=SEED_SET,
            fold_scope="all_three_inner_variety_folds_in_each_outer_context",
            equal_budget_per_candidate=True,
            maximum_new_candidate_families_advanced=3,
            pruning_rule="apply the exact immutable promotion guard; failed candidates remain negative evidence",
            tie_break_rule="candidate_id_utf8_ascending",
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "task_id": "24.3",
        "context": context,
        "models": models,
        "model_registry_hash": hash_object(models),
        "candidates": candidates,
        "candidate_space": space,
        "frozen_candidate_registry_hash": frozen_registry.registry_hash,
        "promotion_guard": PromotionGuard(),
        "screening_schedule": stages,
        "fallback_candidate_id": FALLBACK_EXISTING,
        "design_basis": {
            "source_kind": "historical_design_context_only",
            "gsw_r2_mean": 0.09112,
            "gsw_r2_standard_deviation": 0.01013,
            "delta_vs_consistency_005": 0.000776,
            "selection_score_allowed": False,
            "outer_metric_revival_allowed": False,
        },
    }
    payload["plan_hash"] = hash_object(payload)
    return HyperLeafOptimizationPlan.model_validate(payload)


def evaluate_promotion(
    plan: HyperLeafOptimizationPlan,
    candidate_evidence: InnerOofPromotionEvidence,
    guard_evidence: InnerOofPromotionEvidence,
) -> PromotionDecision:
    """Apply exact meaningful/stable/no-target-regression guards."""

    by_id = {item.candidate_id: item for item in plan.candidates}
    try:
        candidate = by_id[candidate_evidence.candidate_id]
    except KeyError as exc:
        raise HyperLeafOptimizationError("promotion candidate is not preregistered") from exc
    if candidate.role is not CandidateRole.OPTIMIZATION or candidate.matched_guard_id is None:
        raise HyperLeafOptimizationError("only preregistered optimization candidates can be promoted")
    if guard_evidence.candidate_id != candidate.matched_guard_id:
        raise HyperLeafOptimizationError("evidence does not use the candidate's declared matched guard")
    if candidate_evidence.comparison_context_hash != plan.context.comparison_context_hash or guard_evidence.comparison_context_hash != plan.context.comparison_context_hash:
        raise HyperLeafOptimizationError("candidate/control comparison identity mismatch")

    candidate_by_seed = {item.seed: item for item in candidate_evidence.seed_metrics}
    guard_by_seed = {item.seed: item for item in guard_evidence.seed_metrics}
    deltas = {
        target: tuple(
            getattr(candidate_by_seed[seed], target) - getattr(guard_by_seed[seed], target)
            for seed in SEED_SET
        )
        for target in ("grainweight", "gsw", "phips2")
    }
    means = {target: sum(values) / len(values) for target, values in deltas.items()}
    guard = plan.promotion_guard
    failed: list[str] = []
    if means["gsw"] < guard.gsw_mean_delta_r2_minimum:
        failed.append("gsw_mean_pooled_oof_delta_below_0.0100")
    if any(value <= 0.0 for value in deltas["gsw"]):
        failed.append("gsw_seed_direction_not_positive_for_all_7_42_123")
    if means["grainweight"] < guard.grainweight_mean_delta_r2_minimum:
        failed.append("grainweight_mean_pooled_oof_regression_exceeds_0.0100")
    if means["phips2"] < guard.phips2_mean_delta_r2_minimum:
        failed.append("phips2_mean_pooled_oof_regression_exceeds_0.0100")
    passed = not failed
    payload: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "matched_guard_id": candidate.matched_guard_id,
        "passed": passed,
        "mean_delta_r2": means,
        "gsw_seed_deltas": tuple((seed, deltas["gsw"][index]) for index, seed in enumerate(SEED_SET)),
        "failed_guards": tuple(failed),
        "disposition": "eligible_for_task_24.4_inner_selection" if passed else guard.failed_guard_disposition,
        "candidate_evidence_hash": candidate_evidence.evidence_hash,
        "guard_evidence_hash": guard_evidence.evidence_hash,
    }
    payload["decision_hash"] = hash_object(payload)
    return PromotionDecision.model_validate(payload)


def plan_as_json(plan: HyperLeafOptimizationPlan) -> str:
    """Render the authoritative frozen registry with stable field ordering."""

    return json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


__all__ = [
    "CANDIDATE_GATED",
    "CANDIDATE_HETEROSCEDASTIC",
    "CANDIDATE_PCA_ROUTED",
    "CANDIDATE_PHI_CONSISTENCY",
    "CANDIDATE_ROBUST_NORMALIZATION",
    "CANDIDATE_TARGET_BALANCED",
    "CONTROL_CONSISTENCY_005",
    "CONTROL_PCA",
    "CONTROL_RGB",
    "FALLBACK_EXISTING",
    "HyperLeafOptimizationError",
    "HyperLeafOptimizationPlan",
    "InnerOofPromotionEvidence",
    "PromotionDecision",
    "SeedTargetR2",
    "build_hyperleaf_optimization_plan",
    "evaluate_promotion",
    "plan_as_json",
]
