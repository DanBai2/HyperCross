"""Planning-only wiring for Histology candidates and task 17.7 review controls.

This module registers models, freezes the complete review configuration and
PCA-only inference branch from inner-patient evidence, validates matched-control
identities, and emits deterministic recoverable training-job declarations. It
exposes no outer-evaluation entry point and never reads labels, datasets, caches,
or legacy checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional

from benchmark.adapters import FULL_CUBE_SPATIAL_PCA_KEY
from benchmark.engine.selection import SelectionController
from benchmark.fast_lane import (
    FastLaneConfigurationError,
    FastLaneStage,
    FastLaneStageTemplate,
    FrozenFastLaneJob,
    load_fast_lane_template,
)
from benchmark.models.histology import (
    ATTENTION_DISABLED_INFERENCE_BRANCH,
    ATTENTION_DISABLED_MODEL_KEY,
    LOCAL_ROUTE_A_INFERENCE_BRANCH,
    LOCAL_ROUTE_A_MODEL_KEY,
    MIL_CONCAT_INFERENCE_BRANCH,
    MIL_CONCAT_MODEL_KEY,
    PCA_ONLY_INFERENCE_BRANCH,
    PCA_ONLY_MODEL_KEY,
    RF_INFERENCE_BRANCH,
    RF_MODEL_KEY,
    RGB_ONLY_INFERENCE_BRANCH,
    RGB_ONLY_MODEL_KEY,
    SVM_INFERENCE_BRANCH,
    SVM_MODEL_KEY,
    create_histology_model_registry,
)
from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.registry import CandidateSpaceRegistry
from benchmark.schema import (
    CandidateOption,
    CandidateSpace,
    DETERMINISTIC_TIE_BREAK,
    FrozenCandidateSpaceRegistry,
    MetricDirection,
    OuterFold,
    RuntimeMeasurements,
    SelectionCandidate,
    SelectionDecision,
    SelectionPolicy,
    hash_object,
)


class HistologyCandidateRole(str, Enum):
    CANDIDATE = "candidate"
    MATCHED_INPUT_CONTROL = "matched_input_control"
    ATTENTION_SPECIFIC_CONTROL = "attention_specific_control"
    MODALITY_CONTROL = "modality_control"
    CLASSIC_CONTROL = "classic_control"


@dataclass(frozen=True)
class HistologyCandidateSpec:
    """Portable declaration for one stable model and fixed inference branch."""

    model_key: str
    role: HistologyCandidateRole
    inference_branch: str
    input_resolution: str
    pretraining_data_status: str
    matched_control_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchedRunContext:
    """Fields that candidate and control runs must share before comparison."""

    dataset_hash: str
    protocol_hash: str
    seed_set: tuple[int, ...]
    fold_manifest_hash: str
    metric_configuration_hash: str
    resource_budget_hash: str

    def __post_init__(self) -> None:
        hash_fields = (
            "dataset_hash",
            "protocol_hash",
            "fold_manifest_hash",
            "metric_configuration_hash",
            "resource_budget_hash",
        )
        for field in hash_fields:
            value = getattr(self, field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if not self.seed_set or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seed_set
        ):
            raise ValueError("seed_set must contain integers")
        canonical = tuple(sorted(set(self.seed_set)))
        if len(canonical) != len(self.seed_set):
            raise ValueError("seed_set must not contain duplicates")
        object.__setattr__(self, "seed_set", canonical)


class HistologyCandidateError(ValueError):
    """Rejected candidate wiring or matched-control comparison."""

    def __init__(self, message: str, *, mismatched_fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.mismatched_fields = mismatched_fields


HISTOLOGY_CANDIDATE_SPECS: Mapping[str, HistologyCandidateSpec] = {
    PCA_ONLY_MODEL_KEY: HistologyCandidateSpec(
        model_key=PCA_ONLY_MODEL_KEY,
        role=HistologyCandidateRole.MODALITY_CONTROL,
        inference_branch=PCA_ONLY_INFERENCE_BRANCH,
        input_resolution="128 PCA components",
        pretraining_data_status="none",
    ),
    RGB_ONLY_MODEL_KEY: HistologyCandidateSpec(
        model_key=RGB_ONLY_MODEL_KEY,
        role=HistologyCandidateRole.MODALITY_CONTROL,
        inference_branch=RGB_ONLY_INFERENCE_BRANCH,
        input_resolution="3x64x64 RGB",
        pretraining_data_status="none",
    ),
    LOCAL_ROUTE_A_MODEL_KEY: HistologyCandidateSpec(
        model_key=LOCAL_ROUTE_A_MODEL_KEY,
        role=HistologyCandidateRole.CANDIDATE,
        inference_branch=LOCAL_ROUTE_A_INFERENCE_BRANCH,
        input_resolution="3x64x64 RGB + 16x128 local PCA tokens",
        pretraining_data_status="none",
        matched_control_keys=(
            ATTENTION_DISABLED_MODEL_KEY,
            MIL_CONCAT_MODEL_KEY,
        ),
    ),
    MIL_CONCAT_MODEL_KEY: HistologyCandidateSpec(
        model_key=MIL_CONCAT_MODEL_KEY,
        role=HistologyCandidateRole.MATCHED_INPUT_CONTROL,
        inference_branch=MIL_CONCAT_INFERENCE_BRANCH,
        input_resolution="3x64x64 RGB + 16x128 local PCA tokens",
        pretraining_data_status="none",
    ),
    ATTENTION_DISABLED_MODEL_KEY: HistologyCandidateSpec(
        model_key=ATTENTION_DISABLED_MODEL_KEY,
        role=HistologyCandidateRole.ATTENTION_SPECIFIC_CONTROL,
        inference_branch=ATTENTION_DISABLED_INFERENCE_BRANCH,
        input_resolution="3x64x64 RGB + 16x128 local PCA tokens",
        pretraining_data_status="none",
    ),
    SVM_MODEL_KEY: HistologyCandidateSpec(
        model_key=SVM_MODEL_KEY,
        role=HistologyCandidateRole.CLASSIC_CONTROL,
        inference_branch=SVM_INFERENCE_BRANCH,
        input_resolution="128 PCA components",
        pretraining_data_status="none",
    ),
    RF_MODEL_KEY: HistologyCandidateSpec(
        model_key=RF_MODEL_KEY,
        role=HistologyCandidateRole.CLASSIC_CONTROL,
        inference_branch=RF_INFERENCE_BRANCH,
        input_resolution="128 PCA components",
        pretraining_data_status="none",
    ),
}


class HistologyReviewRole(str, Enum):
    """Role of one complete option in the immutable task-17.7 review."""

    CANDIDATE = "candidate"
    MATCHED_INPUT_CONTROL = "matched_input_control"
    ATTENTION_SPECIFIC_CONTROL = "attention_specific_control"
    LOSS_CONTROL = "loss_control"
    NEGATIVE_CANDIDATE = "negative_candidate"


CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY = (
    "histology.consistency-trained-pca-only.v1"
)
PCA_TASK_ONLY_CONTROL_KEY = "histology.pca-only.task-only-control.v1"
LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY = (
    "histology.local-route-a.alignment-nce.v1"
)
MIL_CONCAT_CONTROL_KEY = "histology.mil-concat.control.v1"
ATTENTION_DISABLED_CONTROL_KEY = "histology.attention-disabled.control.v1"
LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY = "histology.local-route-a.task-only-control.v1"
LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY = (
    "histology.local-route-a.alignment-only-control.v1"
)
LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY = "histology.local-route-a.nce-only-control.v1"


@dataclass(frozen=True)
class HistologyReviewCandidateSpec:
    """One complete, non-Cartesian review option frozen before evaluation."""

    candidate_key: str
    model_key: str
    role: HistologyReviewRole
    inference_branch: str
    training_recipe_key: str
    loss_key: str
    normalization_key: str = "default-input-scaling"
    augmentation_key: str = "none"
    sampling_key: str = "sample-level"
    alignment_weight: float = 0.0
    nce_weight: float = 0.0
    consistency_weight: float = 0.0
    auxiliary_model_keys: tuple[str, ...] = ()
    matched_control_keys: tuple[str, ...] = ()
    negative_evidence: bool = False
    evidence_note: str = "pending-inner-patient-review"

    def __post_init__(self) -> None:
        text_fields = (
            "candidate_key",
            "model_key",
            "inference_branch",
            "training_recipe_key",
            "loss_key",
            "normalization_key",
            "augmentation_key",
            "sampling_key",
            "evidence_note",
        )
        for field_name in text_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a non-empty trimmed string")
        for field_name in ("alignment_weight", "nce_weight", "consistency_weight"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        for field_name in ("auxiliary_model_keys", "matched_control_keys"):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values), key=lambda item: item.encode("utf-8"))):
                raise ValueError(f"{field_name} must be unique and canonically sorted")
        if self.negative_evidence != (
            self.role is HistologyReviewRole.NEGATIVE_CANDIDATE
        ):
            raise ValueError(
                "negative_evidence must be true exactly for negative candidates"
            )

    def option_parameters(self) -> dict[str, object]:
        """Project the complete review option into canonical candidate-space data."""

        return {
            "alignment_weight": self.alignment_weight,
            "augmentation_key": self.augmentation_key,
            "auxiliary_model_keys": self.auxiliary_model_keys,
            "consistency_weight": self.consistency_weight,
            "evidence_note": self.evidence_note,
            "inference_branch": self.inference_branch,
            "loss_key": self.loss_key,
            "matched_control_keys": self.matched_control_keys,
            "model_key": self.model_key,
            "nce_weight": self.nce_weight,
            "negative_evidence": self.negative_evidence,
            "normalization_key": self.normalization_key,
            "role": self.role.value,
            "sampling_key": self.sampling_key,
            "training_recipe_key": self.training_recipe_key,
        }


def _review_spec(
    candidate_key: str,
    model_key: str,
    role: HistologyReviewRole,
    inference_branch: str,
    training_recipe_key: str,
    loss_key: str,
    **kwargs: object,
) -> HistologyReviewCandidateSpec:
    return HistologyReviewCandidateSpec(
        candidate_key=candidate_key,
        model_key=model_key,
        role=role,
        inference_branch=inference_branch,
        training_recipe_key=training_recipe_key,
        loss_key=loss_key,
        **kwargs,  # type: ignore[arg-type]
    )


# Complete review options are explicit so invalid cross-products (for example,
# MIL-concat with an attention-only ablation) cannot be scheduled accidentally.
HISTOLOGY_REVIEW_CANDIDATES: tuple[HistologyReviewCandidateSpec, ...] = tuple(
    sorted(
        (
            _review_spec(
                CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY,
                PCA_ONLY_MODEL_KEY,
                HistologyReviewRole.CANDIDATE,
                PCA_ONLY_INFERENCE_BRANCH,
                "paired-rgb-pca-consistency-training.v1",
                "classification-plus-symmetric-kl.v1",
                consistency_weight=0.1,
                auxiliary_model_keys=(RGB_ONLY_MODEL_KEY,),
                matched_control_keys=(PCA_TASK_ONLY_CONTROL_KEY,),
            ),
            _review_spec(
                PCA_TASK_ONLY_CONTROL_KEY,
                PCA_ONLY_MODEL_KEY,
                HistologyReviewRole.MATCHED_INPUT_CONTROL,
                PCA_ONLY_INFERENCE_BRANCH,
                "pca-only-supervised-training.v1",
                "classification-only.v1",
            ),
            _review_spec(
                LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.v1",
                "classification-alignment-nce.v1",
                alignment_weight=0.08,
                nce_weight=0.04,
                matched_control_keys=tuple(
                    sorted(
                        (
                            ATTENTION_DISABLED_CONTROL_KEY,
                            LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
                            LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
                            LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
                            MIL_CONCAT_CONTROL_KEY,
                        )
                    )
                ),
            ),
            _review_spec(
                MIL_CONCAT_CONTROL_KEY,
                MIL_CONCAT_MODEL_KEY,
                HistologyReviewRole.MATCHED_INPUT_CONTROL,
                MIL_CONCAT_INFERENCE_BRANCH,
                "mil-concat.v1",
                "classification-alignment-nce.v1",
                alignment_weight=0.08,
                nce_weight=0.04,
            ),
            _review_spec(
                ATTENTION_DISABLED_CONTROL_KEY,
                ATTENTION_DISABLED_MODEL_KEY,
                HistologyReviewRole.ATTENTION_SPECIFIC_CONTROL,
                ATTENTION_DISABLED_INFERENCE_BRANCH,
                "local-route-a-attention-disabled.v1",
                "classification-alignment-nce.v1",
                alignment_weight=0.08,
                nce_weight=0.04,
            ),
            _review_spec(
                LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.LOSS_CONTROL,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.v1",
                "classification-only.v1",
            ),
            _review_spec(
                LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.LOSS_CONTROL,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.v1",
                "classification-alignment-only.v1",
                alignment_weight=0.08,
            ),
            _review_spec(
                LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.LOSS_CONTROL,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.v1",
                "classification-nce-only.v1",
                nce_weight=0.04,
            ),
            _review_spec(
                "histology.negative.class-balanced-loss.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.class-balanced.v1",
                "class-balanced-classification.v1",
                negative_evidence=True,
                evidence_note="historical-negative-class-balanced-loss",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.instance-normalization.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.instance-normalization.v1",
                "classification-alignment-nce.v1",
                normalization_key="instance-normalized-visual-features.v1",
                alignment_weight=0.08,
                nce_weight=0.04,
                negative_evidence=True,
                evidence_note="historical-negative-instance-normalization",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.class-conditional-alignment.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.class-conditional-alignment.v1",
                "class-conditional-alignment.v1",
                alignment_weight=0.05,
                negative_evidence=True,
                evidence_note="historical-negative-class-conditional-alignment",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.entropy-confidence-fusion.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "entropy-confidence-fusion-temperature-0.5.v1",
                "classification-only.v1",
                negative_evidence=True,
                evidence_note="historical-negative-entropy-confidence-fusion",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.patient-domain-adversarial.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "gradient-reversal-patient-domain-lambda-0.1.v1",
                "classification-domain-adversarial.v1",
                negative_evidence=True,
                evidence_note="historical-negative-patient-domain-adversarial",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.patient-style-consistency.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "patient-style-consistency.v1",
                "classification-plus-symmetric-kl.v1",
                augmentation_key="rgb-affine-flip-spectral-noise-band-mask.v1",
                consistency_weight=0.1,
                negative_evidence=True,
                evidence_note="historical-negative-patient-style-consistency",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.train-patient-normalization.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.train-patient-normalization.v1",
                "classification-alignment-nce.v1",
                normalization_key="train-patient-rgb-local-spectrum.v1",
                alignment_weight=0.08,
                nce_weight=0.04,
                negative_evidence=True,
                evidence_note="historical-negative-train-patient-normalization",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.equal-patient-sampling.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "local-route-a.v1",
                "classification-alignment-nce.v1",
                sampling_key="equal-patient-probability.v1",
                alignment_weight=0.08,
                nce_weight=0.04,
                negative_evidence=True,
                evidence_note="historical-negative-equal-patient-sampling",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.groupdro.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "groupdro-patient-eta-0.1.v1",
                "groupdro-classification.v1",
                negative_evidence=True,
                evidence_note="historical-negative-groupdro",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
            _review_spec(
                "histology.negative.patient-episodic-prototype.v1",
                LOCAL_ROUTE_A_MODEL_KEY,
                HistologyReviewRole.NEGATIVE_CANDIDATE,
                LOCAL_ROUTE_A_INFERENCE_BRANCH,
                "patient-episodic-prototype-weight-0.2.v1",
                "classification-plus-prototype.v1",
                negative_evidence=True,
                evidence_note="historical-negative-patient-episodic-prototype",
                matched_control_keys=(LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,),
            ),
        ),
        key=lambda item: item.candidate_key.encode("utf-8"),
    )
)

_HISTOLOGY_REVIEW_BY_KEY: Mapping[str, HistologyReviewCandidateSpec] = {
    spec.candidate_key: spec for spec in HISTOLOGY_REVIEW_CANDIDATES
}
HISTOLOGY_REVIEW_SPACE_KEY = "histology.candidate-review.v1"
HISTOLOGY_REVIEW_RELATIVE_OUTPUT_ROOT = Path(
    "runs_v2/strict/histology-candidate-review"
)
# Training receives these immutable source classes as inputs only. ``runs`` also
# covers legacy/old checkpoints; new recovery checkpoints live under runs_v2.
HISTOLOGY_REVIEW_READ_ONLY_SOURCES = (
    "cache",
    "dataset",
    "old-checkpoint",
    "runs",
    "src",
)
HISTOLOGY_NORMALIZATION_KEYS = (
    "default-input-scaling",
    "instance-normalized-visual-features.v1",
    "train-patient-rgb-local-spectrum.v1",
)


@dataclass(frozen=True)
class HistologyReviewConfiguration:
    """Hash-bound review space plus its immutable later-run execution contract."""

    candidate_registry: FrozenCandidateSpaceRegistry
    candidate_space_hash: str
    normalization_candidates: tuple[CandidateOption, ...]
    negative_candidate_keys: tuple[str, ...]
    matched_control_pairs: tuple[tuple[str, str], ...]
    execution_template: FastLaneStageTemplate
    relative_output_root: str
    read_only_sources: tuple[str, ...] = HISTOLOGY_REVIEW_READ_ONLY_SOURCES
    outer_label_access_allowed: bool = False
    outer_evaluation_allowed: bool = False

    def __post_init__(self) -> None:
        space = self.candidate_registry.require_for_evaluation(
            HISTOLOGY_REVIEW_SPACE_KEY,
            expected_hash=self.candidate_space_hash,
        )
        expected_keys = tuple(spec.candidate_key for spec in HISTOLOGY_REVIEW_CANDIDATES)
        observed_keys = tuple(option.option_key for option in space.model_candidates)
        if observed_keys != expected_keys:
            raise ValueError("review candidate registry differs from complete declarations")
        normalization_keys = tuple(
            option.option_key for option in self.normalization_candidates
        )
        if normalization_keys != HISTOLOGY_NORMALIZATION_KEYS:
            raise ValueError("normalization candidates differ from frozen review policy")
        expected_negative = tuple(
            spec.candidate_key
            for spec in HISTOLOGY_REVIEW_CANDIDATES
            if spec.negative_evidence
        )
        if self.negative_candidate_keys != expected_negative:
            raise ValueError("negative candidate set differs from review declarations")
        expected_pairs = tuple(
            sorted(
                (
                    (spec.candidate_key, control_key)
                    for spec in HISTOLOGY_REVIEW_CANDIDATES
                    for control_key in spec.matched_control_keys
                ),
                key=lambda pair: (pair[0].encode("utf-8"), pair[1].encode("utf-8")),
            )
        )
        if self.matched_control_pairs != expected_pairs:
            raise ValueError("matched control links differ from review declarations")
        if self.execution_template.stage is not FastLaneStage.STRICT:
            raise ValueError("Histology review training must use the strict fast-lane stage")
        if self.relative_output_root != HISTOLOGY_REVIEW_RELATIVE_OUTPUT_ROOT.as_posix():
            raise ValueError("Histology review output root differs from frozen policy")
        if self.read_only_sources != HISTOLOGY_REVIEW_READ_ONLY_SOURCES:
            raise ValueError("Histology review input sources must remain read-only")
        if self.outer_label_access_allowed or self.outer_evaluation_allowed:
            raise ValueError(
                "candidate review cannot access outer labels or start outer evaluation"
            )
        if not {"runs", "src"}.issubset(self.execution_template.read_only_roots):
            raise ValueError("strict execution template must protect src and legacy runs")

    @property
    def candidate_space(self) -> CandidateSpace:
        return self.candidate_registry.require_for_evaluation(
            HISTOLOGY_REVIEW_SPACE_KEY,
            expected_hash=self.candidate_space_hash,
        )

    @property
    def review_hash(self) -> str:
        return hash_object(
            {
                "candidate_registry_hash": self.candidate_registry.registry_hash,
                "candidate_space_hash": self.candidate_space_hash,
                "execution_template_hash": self.execution_template.template_hash,
                "matched_control_pairs": self.matched_control_pairs,
                "negative_candidate_keys": self.negative_candidate_keys,
                "normalization_candidates": self.normalization_candidates,
                "outer_evaluation_allowed": self.outer_evaluation_allowed,
                "outer_label_access_allowed": self.outer_label_access_allowed,
                "read_only_sources": self.read_only_sources,
                "relative_output_root": self.relative_output_root,
            }
        )


class HistologyCandidateRunner:
    """Safe integration surface for Histology review planning and inner selection."""

    relative_artifact_root = HISTOLOGY_REVIEW_RELATIVE_OUTPUT_ROOT

    def __init__(
        self,
        workspace_root: Path | str = Path(r"G:\Datasets\HyperCross"),
        *,
        registry: Optional[ModelRegistry] = None,
        selection_controller: Optional[SelectionController] = None,
        template_path: Optional[Path | str] = None,
        template: Optional[FastLaneStageTemplate] = None,
    ) -> None:
        if template is not None and template_path is not None:
            raise HistologyCandidateError("provide template or template_path, not both")
        try:
            loaded = template or load_fast_lane_template(
                template_path
                or Path(r"G:\Datasets\HyperCross\configs\fast_lane\strict.json")
            )
            root = loaded.validate_workspace_root(workspace_root)
        except FastLaneConfigurationError as exc:
            if Path(workspace_root).drive.upper() == "F:":
                raise HistologyCandidateError(
                    "Histology candidate workspace must remain on G:"
                ) from exc
            raise HistologyCandidateError(str(exc)) from exc
        self._validate_template(loaded)
        self.workspace_root = root
        self.template = loaded
        self.artifact_root = root / self.relative_artifact_root
        self.registry = registry or create_histology_model_registry()
        self.selection_controller = selection_controller or SelectionController()
        self._validate_declarations()
        self._validate_review_declarations()

    @property
    def model_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(HISTOLOGY_CANDIDATE_SPECS, key=lambda value: value.encode("utf-8"))
        )

    @property
    def review_candidate_keys(self) -> tuple[str, ...]:
        return tuple(spec.candidate_key for spec in HISTOLOGY_REVIEW_CANDIDATES)

    @property
    def negative_candidate_keys(self) -> tuple[str, ...]:
        return tuple(
            spec.candidate_key
            for spec in HISTOLOGY_REVIEW_CANDIDATES
            if spec.negative_evidence
        )

    def resolve_review_candidate(
        self, candidate_key: str
    ) -> HistologyReviewCandidateSpec:
        try:
            return _HISTOLOGY_REVIEW_BY_KEY[candidate_key]
        except KeyError as exc:
            raise HistologyCandidateError(
                f"unknown Histology review candidate: {candidate_key!r}"
            ) from exc

    def freeze_review_configuration(
        self,
        *,
        frozen_utc: Optional[datetime] = None,
    ) -> HistologyReviewConfiguration:
        """Freeze all complete review options and later-run constraints.

        This method only constructs immutable schema objects. It neither creates
        output directories nor starts model construction, training, or evaluation.
        """

        model_options = tuple(
            CandidateOption.create(
                spec.candidate_key,
                parameters=spec.option_parameters(),
            )
            for spec in HISTOLOGY_REVIEW_CANDIDATES
        )
        representation_options = (
            CandidateOption.create(
                FULL_CUBE_SPATIAL_PCA_KEY,
                parameters={
                    "modalities": ("rgb", "local_pca_tokens"),
                    "normalization_candidates": HISTOLOGY_NORMALIZATION_KEYS,
                    "spatial_tokens": 16,
                    "pca_components": 128,
                },
            ),
            CandidateOption.create(
                self.registry.resolve(PCA_ONLY_MODEL_KEY).input_contract.representation_key,
                parameters={
                    "modalities": ("pca_spectrum",),
                    "pca_components": 128,
                    "inference_only_after_consistency_training": True,
                },
            ),
        )
        loss_signatures: dict[str, dict[str, float]] = {}
        for spec in HISTOLOGY_REVIEW_CANDIDATES:
            signature = {
                "alignment_weight": spec.alignment_weight,
                "consistency_weight": spec.consistency_weight,
                "nce_weight": spec.nce_weight,
            }
            existing = loss_signatures.setdefault(spec.loss_key, signature)
            if existing != signature:
                # A loss key must carry one exact setting. Distinct weights need
                # distinct keys so later jobs cannot silently reinterpret it.
                raise HistologyCandidateError(
                    f"loss key {spec.loss_key!r} has conflicting frozen weights"
                )
        loss_options = tuple(
            CandidateOption.create(key, parameters=loss_signatures[key])
            for key in sorted(loss_signatures, key=lambda item: item.encode("utf-8"))
        )
        augmentation_keys = tuple(
            sorted(
                {spec.augmentation_key for spec in HISTOLOGY_REVIEW_CANDIDATES},
                key=lambda item: item.encode("utf-8"),
            )
        )
        inference_branches = tuple(
            sorted(
                {spec.inference_branch for spec in HISTOLOGY_REVIEW_CANDIDATES},
                key=lambda item: item.encode("utf-8"),
            )
        )
        space = CandidateSpace.create(
            candidate_space_key=HISTOLOGY_REVIEW_SPACE_KEY,
            model_candidates=model_options,
            representation_candidates=representation_options,
            loss_candidates=loss_options,
            augmentation_candidates=tuple(
                CandidateOption.create(key) for key in augmentation_keys
            ),
            target_route_candidates=(
                CandidateOption.create("binary-classification-shared-route"),
            ),
            inference_branch_candidates=tuple(
                CandidateOption.create(key) for key in inference_branches
            ),
            threshold_candidates=(
                CandidateOption.create("fixed-0.5", parameters={"value": 0.5}),
            ),
            calibration_candidates=(
                CandidateOption.create("none"),
                CandidateOption.create(
                    "inner-oof-temperature",
                    parameters={"fit_data_role": "inner_oof_predictions"},
                ),
            ),
            objective_metric="pooled-oof-balanced-accuracy",
            metric_direction=MetricDirection.MAXIMIZE,
            tie_break_rule=DETERMINISTIC_TIE_BREAK,
            frozen_utc=frozen_utc,
        )
        registry = CandidateSpaceRegistry()
        registry.register(space)
        frozen_registry = registry.freeze_before_evaluation(frozen_utc=frozen_utc)
        normalization_options = tuple(
            CandidateOption.create(
                key,
                parameters={
                    "fit_scope": (
                        "inner_train_patients"
                        if key == "train-patient-rgb-local-spectrum.v1"
                        else "fixed_no_fit"
                    )
                },
            )
            for key in HISTOLOGY_NORMALIZATION_KEYS
        )
        matched_pairs = tuple(
            sorted(
                (
                    (spec.candidate_key, control_key)
                    for spec in HISTOLOGY_REVIEW_CANDIDATES
                    for control_key in spec.matched_control_keys
                ),
                key=lambda pair: (pair[0].encode("utf-8"), pair[1].encode("utf-8")),
            )
        )
        return HistologyReviewConfiguration(
            candidate_registry=frozen_registry,
            candidate_space_hash=space.candidate_space_hash,
            normalization_candidates=normalization_options,
            negative_candidate_keys=self.negative_candidate_keys,
            matched_control_pairs=matched_pairs,
            execution_template=self.template,
            relative_output_root=self.relative_artifact_root.as_posix(),
            read_only_sources=HISTOLOGY_REVIEW_READ_ONLY_SOURCES,
            outer_label_access_allowed=False,
            outer_evaluation_allowed=False,
        )

    def resolve(self, model_key: str) -> tuple[HistologyCandidateSpec, ModelRegistration]:
        try:
            spec = HISTOLOGY_CANDIDATE_SPECS[model_key]
        except KeyError as exc:
            raise HistologyCandidateError(
                f"unknown Histology candidate Model_Key: {model_key!r}"
            ) from exc
        return spec, self.registry.resolve(model_key)

    def freeze_inference_branch(
        self,
        outer_fold: OuterFold,
        *,
        model_key: str,
        inner_source_sample_ids: Iterable[str],
        inner_prediction_hash: str,
        inner_score: float,
        selection_policy: SelectionPolicy,
        decision_id: Optional[str] = None,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Freeze one declared branch using only inner-patient evidence.

        PCA-only is always represented by the literal ``pca-only`` candidate;
        callers cannot substitute or select another branch after this decision.
        The shared SelectionController rejects outer-patient or undeclared IDs.
        """

        spec, registration = self.resolve(model_key)
        branch = SelectionCandidate(
            candidate_id=spec.inference_branch,
            configuration={
                "model_key": spec.model_key,
                "representation_key": registration.input_contract.representation_key,
                "inference_branch": spec.inference_branch,
            },
        )
        return self.selection_controller.freeze_selection(
            outer_fold,
            candidates=(branch,),
            candidate_scores={spec.inference_branch: inner_score},
            source_sample_ids=inner_source_sample_ids,
            source_prediction_hash=inner_prediction_hash,
            selection_policy=selection_policy,
            decision_id=decision_id,
            frozen_utc=frozen_utc,
        )

    def freeze_pca_only_inference_branch(
        self,
        outer_fold: OuterFold,
        *,
        review_configuration: HistologyReviewConfiguration,
        inner_source_sample_ids: Iterable[str],
        inner_prediction_hash: str,
        inner_score: float,
        selection_policy: SelectionPolicy,
        outer_labels_accessible: bool = False,
        decision_id: Optional[str] = None,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Freeze the consistency-trained PCA-only branch before label access."""

        if outer_labels_accessible:
            raise HistologyCandidateError(
                "PCA-only inference branch must freeze before outer labels are accessible"
            )
        space = review_configuration.candidate_space
        options = {option.option_key: option for option in space.model_candidates}
        try:
            frozen_option = options[CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY]
        except KeyError as exc:
            raise HistologyCandidateError(
                "frozen review space lacks consistency-trained PCA-only"
            ) from exc
        spec = self.resolve_review_candidate(
            CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY
        )
        if frozen_option.option_hash != CandidateOption.create(
            spec.candidate_key, parameters=spec.option_parameters()
        ).option_hash:
            raise HistologyCandidateError(
                "consistency-trained PCA-only option differs from review declarations"
            )
        branch = SelectionCandidate(
            candidate_id=spec.candidate_key,
            configuration={
                **spec.option_parameters(),
                "candidate_space_hash": space.candidate_space_hash,
                "review_hash": review_configuration.review_hash,
            },
        )
        return self.selection_controller.freeze_selection(
            outer_fold,
            candidates=(branch,),
            candidate_scores={spec.candidate_key: inner_score},
            source_sample_ids=inner_source_sample_ids,
            source_prediction_hash=inner_prediction_hash,
            selection_policy=selection_policy,
            decision_id=decision_id,
            frozen_utc=frozen_utc,
        )

    def validate_review_matched_control(
        self,
        candidate_key: str,
        control_key: str,
        *,
        review_configuration: HistologyReviewConfiguration,
        candidate_context: MatchedRunContext,
        control_context: MatchedRunContext,
    ) -> None:
        """Validate a predeclared review control and every comparison identity."""

        space_keys = {
            option.option_key
            for option in review_configuration.candidate_space.model_candidates
        }
        if candidate_key not in space_keys or control_key not in space_keys:
            raise HistologyCandidateError(
                "candidate and control must both belong to the frozen review space"
            )
        candidate = self.resolve_review_candidate(candidate_key)
        control = self.resolve_review_candidate(control_key)
        if control_key not in candidate.matched_control_keys:
            raise HistologyCandidateError(
                f"{control_key!r} is not a declared matched control for {candidate_key!r}"
            )
        mismatches = self._comparison_mismatches(
            candidate_context, control_context
        )
        candidate_representation = self.registry.resolve(
            candidate.model_key
        ).input_contract.representation_key
        control_representation = self.registry.resolve(
            control.model_key
        ).input_contract.representation_key
        if candidate_representation != control_representation:
            mismatches.append("representation_key")
        if mismatches:
            ordered = tuple(sorted(set(mismatches), key=lambda value: value.encode("utf-8")))
            raise HistologyCandidateError(
                "candidate/control comparison fields do not match: " + ", ".join(ordered),
                mismatched_fields=ordered,
            )

    def freeze_training_job(
        self,
        *,
        review_configuration: HistologyReviewConfiguration,
        candidate_key: str,
        job_name: str,
        dataset_hash: str,
        protocol_hash: str,
        protocol_key: str,
        fold_id: str,
        seed: int,
        input_volume_bytes: int,
    ) -> FrozenFastLaneJob:
        """Declare a deterministic recoverable strict job without starting it."""

        spec = self.resolve_review_candidate(candidate_key)
        if candidate_key not in {
            option.option_key
            for option in review_configuration.candidate_space.model_candidates
        }:
            raise HistologyCandidateError(
                "candidate is absent from the frozen review configuration"
            )
        try:
            return review_configuration.execution_template.freeze_job(
                job_name=job_name,
                dataset_hash=dataset_hash,
                protocol_hash=protocol_hash,
                protocol_key=protocol_key,
                model_key=spec.model_key,
                fold_id=fold_id,
                seed=seed,
                input_volume_bytes=input_volume_bytes,
                selection_metric_direction=MetricDirection.MAXIMIZE,
                parameters={
                    "candidate_parameters": spec.option_parameters(),
                    "candidate_space_hash": review_configuration.candidate_space_hash,
                    "histology_review_candidate_key": spec.candidate_key,
                    "histology_review_hash": review_configuration.review_hash,
                    "inference_branch": spec.inference_branch,
                    "loss_key": spec.loss_key,
                    "normalization_key": spec.normalization_key,
                    "outer_evaluation_allowed": review_configuration.outer_evaluation_allowed,
                    "outer_label_access_allowed": review_configuration.outer_label_access_allowed,
                    "read_only_sources": review_configuration.read_only_sources,
                    "review_artifact_root": review_configuration.relative_output_root,
                    "training_recipe_key": spec.training_recipe_key,
                },
            )
        except FastLaneConfigurationError as exc:
            raise HistologyCandidateError(str(exc)) from exc

    def validate_matched_control(
        self,
        candidate_key: str,
        control_key: str,
        *,
        candidate_context: MatchedRunContext,
        control_context: MatchedRunContext,
    ) -> None:
        """Require matched representation, protocol, seeds, folds, metrics, and budget."""

        candidate, candidate_registration = self.resolve(candidate_key)
        control, control_registration = self.resolve(control_key)
        if control.model_key not in candidate.matched_control_keys:
            raise HistologyCandidateError(
                f"{control_key!r} is not a declared matched control for {candidate_key!r}"
            )

        mismatches = self._comparison_mismatches(
            candidate_context, control_context
        )
        if (
            candidate_registration.input_contract.representation_key
            != control_registration.input_contract.representation_key
        ):
            mismatches.append("representation_key")
        if mismatches:
            ordered = tuple(sorted(mismatches, key=lambda value: value.encode("utf-8")))
            raise HistologyCandidateError(
                "candidate/control comparison fields do not match: " + ", ".join(ordered),
                mismatched_fields=ordered,
            )

    def method_description(
        self,
        model_key: str,
        *,
        available_control_keys: Iterable[str] = (),
    ) -> str:
        """Return mechanism-safe wording based on available control evidence."""

        self.resolve(model_key)
        if model_key != LOCAL_ROUTE_A_MODEL_KEY:
            return HISTOLOGY_CANDIDATE_SPECS[model_key].role.value
        available = set(available_control_keys)
        if ATTENTION_DISABLED_MODEL_KEY not in available:
            return (
                "combined method: local tokenization, MIL pooling, and "
                "cross-modal interaction"
            )
        return "Local Route A cross-modal interaction with an attention-disabled control"

    def run_metadata(
        self,
        model_key: str,
        *,
        protocol_key: str,
        measurements: RuntimeMeasurements,
        available_control_keys: Iterable[str] = (),
    ) -> dict[str, object]:
        """Expose the required comparison columns from frozen and measured data."""

        spec, registration = self.resolve(model_key)
        expected_parameters = registration.identity.parameter_count
        if measurements.parameter_count != expected_parameters:
            raise HistologyCandidateError(
                "runtime parameter_count differs from the stable ModelIdentity"
            )
        return {
            "protocol_key": protocol_key,
            "model_key": model_key,
            "implementation_kind": registration.identity.implementation_kind.value,
            "representation_key": registration.input_contract.representation_key,
            "parameter_count": expected_parameters,
            "completed_training_steps": measurements.completed_training_steps,
            "input_resolution": spec.input_resolution,
            "pretraining_data_status": spec.pretraining_data_status,
            "wall_clock_duration_ms": measurements.wall_clock_duration_ms,
            "peak_device_memory_bytes": measurements.peak_device_memory_bytes,
            "inference_branch": spec.inference_branch,
            "method_description": self.method_description(
                model_key, available_control_keys=available_control_keys
            ),
        }

    @staticmethod
    def _comparison_mismatches(
        candidate_context: MatchedRunContext,
        control_context: MatchedRunContext,
    ) -> list[str]:
        fields = (
            "dataset_hash",
            "protocol_hash",
            "seed_set",
            "fold_manifest_hash",
            "metric_configuration_hash",
            "resource_budget_hash",
        )
        return [
            field
            for field in fields
            if getattr(candidate_context, field) != getattr(control_context, field)
        ]

    def _validate_review_declarations(self) -> None:
        keys = tuple(spec.candidate_key for spec in HISTOLOGY_REVIEW_CANDIDATES)
        if len(keys) != len(set(keys)):
            raise HistologyCandidateError("Histology review candidate keys are duplicated")
        registered_models = set(self.registry.model_keys)
        for spec in HISTOLOGY_REVIEW_CANDIDATES:
            missing_models = ({spec.model_key} | set(spec.auxiliary_model_keys)) - registered_models
            if missing_models:
                raise HistologyCandidateError(
                    f"review candidate {spec.candidate_key!r} uses unregistered models: "
                    f"{sorted(missing_models)}"
                )
            missing_controls = set(spec.matched_control_keys) - set(keys)
            if missing_controls:
                raise HistologyCandidateError(
                    f"review candidate {spec.candidate_key!r} has undeclared controls: "
                    f"{sorted(missing_controls)}"
                )
        consistency = self.resolve_review_candidate(
            CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY
        )
        if (
            consistency.inference_branch != PCA_ONLY_INFERENCE_BRANCH
            or consistency.model_key != PCA_ONLY_MODEL_KEY
            or RGB_ONLY_MODEL_KEY not in consistency.auxiliary_model_keys
        ):
            raise HistologyCandidateError(
                "consistency-trained candidate must freeze the PCA-only inference branch"
            )
        observed_normalizations = {
            spec.normalization_key for spec in HISTOLOGY_REVIEW_CANDIDATES
        }
        if not set(HISTOLOGY_NORMALIZATION_KEYS).issubset(observed_normalizations):
            raise HistologyCandidateError(
                "review declarations omit a required normalization candidate"
            )

    @staticmethod
    def _validate_template(template: FastLaneStageTemplate) -> None:
        exact = {
            "stage": (template.stage, FastLaneStage.STRICT),
            "conda_environment": (template.conda_environment, "TCFM"),
            "device": (template.device, "cuda:0"),
            "expected_device_name": (
                template.expected_device_name,
                "NVIDIA GeForce RTX 5080",
            ),
            "visible_cuda_device_count": (template.visible_cuda_device_count, 1),
            "process_count": (template.process_count, 1),
            "execution_mode": (template.execution_mode.value, "deterministic"),
            "recovery_state_names": (
                template.recovery_state_names,
                ("model", "optimizer", "scheduler", "stage"),
            ),
            "persist_frozen_identity_before_training": (
                template.persist_frozen_identity_before_training,
                True,
            ),
        }
        mismatches = [
            field for field, (observed, expected) in exact.items() if observed != expected
        ]
        if mismatches:
            raise HistologyCandidateError(
                "Histology review strict template mismatch: " + ", ".join(mismatches)
            )
        try:
            template.validate_write_path(
                HISTOLOGY_REVIEW_RELATIVE_OUTPUT_ROOT.as_posix()
            )
        except FastLaneConfigurationError as exc:
            raise HistologyCandidateError(str(exc)) from exc

    def _validate_declarations(self) -> None:
        declared = set(HISTOLOGY_CANDIDATE_SPECS)
        if declared != set(self.registry.model_keys):
            raise HistologyCandidateError(
                "Histology candidate declarations and registry keys differ"
            )
        if any(key != spec.model_key for key, spec in HISTOLOGY_CANDIDATE_SPECS.items()):
            raise HistologyCandidateError("candidate map key differs from stable Model_Key")
        for candidate in HISTOLOGY_CANDIDATE_SPECS.values():
            missing = set(candidate.matched_control_keys) - declared
            if missing:
                raise HistologyCandidateError(
                    f"candidate {candidate.model_key!r} has undeclared controls: {sorted(missing)}"
                )


__all__ = [
    "ATTENTION_DISABLED_CONTROL_KEY",
    "CONSISTENCY_TRAINED_PCA_ONLY_CANDIDATE_KEY",
    "HISTOLOGY_CANDIDATE_SPECS",
    "HISTOLOGY_NORMALIZATION_KEYS",
    "HISTOLOGY_REVIEW_CANDIDATES",
    "HISTOLOGY_REVIEW_READ_ONLY_SOURCES",
    "HISTOLOGY_REVIEW_RELATIVE_OUTPUT_ROOT",
    "HISTOLOGY_REVIEW_SPACE_KEY",
    "LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY",
    "LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY",
    "LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY",
    "LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY",
    "MIL_CONCAT_CONTROL_KEY",
    "PCA_TASK_ONLY_CONTROL_KEY",
    "HistologyCandidateError",
    "HistologyCandidateRole",
    "HistologyCandidateRunner",
    "HistologyCandidateSpec",
    "HistologyReviewCandidateSpec",
    "HistologyReviewConfiguration",
    "HistologyReviewRole",
    "MatchedRunContext",
]
