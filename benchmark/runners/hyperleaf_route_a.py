"""Planning-only integration for task-14.5 HyperLeaf Route A models.

This runner resolves only stable ``Model_Key`` values, builds only benchmark
adapters, and freezes route/consistency choices from inner-fold evidence. It
has no official-label reader and no outer-evaluation entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Optional

from benchmark.engine.selection import SelectionController
from benchmark.models.hyperleaf_route_a import (
    GLOBAL_FUSION_MODEL_KEY,
    LOCAL_MIL_MODEL_KEY,
    SHARED_HEADS_MODEL_KEY,
    TARGET_EXPERTS_MODEL_KEY,
    TOKEN_MODEL_KEY,
    create_hyperleaf_route_a_registry,
)
from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import OuterFold, SelectionCandidate, SelectionDecision, SelectionPolicy


class HyperLeafRouteARole(str, Enum):
    GLOBAL_FUSION_BASELINE = "global_fusion_baseline"
    TARGET_SPECIFIC_EXPERTS = "target_specific_experts"
    SHARED_ENCODER_HEADS = "shared_encoder_heads"
    TOKEN_ABLATION = "token_ablation"
    LOCAL_ABLATION = "local_ablation"


@dataclass(frozen=True)
class ConsistencySetting:
    """Predeclared loss setting eligible for inner-only selection."""

    setting_key: str
    alignment_weight: float
    contrastive_weight: float


@dataclass(frozen=True)
class HyperLeafRouteASpec:
    """Stable route declaration independent of legacy list positions."""

    model_key: str
    role: HyperLeafRouteARole
    inference_branch: str
    input_resolution: str
    consistency_settings: tuple[ConsistencySetting, ...]


DEFAULT_CONSISTENCY = ConsistencySetting("alignment-nce-v1", 0.08, 0.04)
TASK_ONLY = ConsistencySetting("task-only-v1", 0.0, 0.0)

HYPERLEAF_ROUTE_A_SPECS: Mapping[str, HyperLeafRouteASpec] = {
    GLOBAL_FUSION_MODEL_KEY: HyperLeafRouteASpec(
        GLOBAL_FUSION_MODEL_KEY,
        HyperLeafRouteARole.GLOBAL_FUSION_BASELINE,
        "global-fusion",
        "3x32x64 visible RGB + 128-bin global spectrum",
        (DEFAULT_CONSISTENCY, TASK_ONLY),
    ),
    TARGET_EXPERTS_MODEL_KEY: HyperLeafRouteASpec(
        TARGET_EXPERTS_MODEL_KEY,
        HyperLeafRouteARole.TARGET_SPECIFIC_EXPERTS,
        "target-specific-experts",
        "3x32x64 visible RGB + 128-bin global spectrum",
        (DEFAULT_CONSISTENCY, TASK_ONLY),
    ),
    SHARED_HEADS_MODEL_KEY: HyperLeafRouteASpec(
        SHARED_HEADS_MODEL_KEY,
        HyperLeafRouteARole.SHARED_ENCODER_HEADS,
        "shared-encoder-heads",
        "3x32x64 visible RGB + 128-bin global spectrum",
        (DEFAULT_CONSISTENCY, TASK_ONLY),
    ),
    TOKEN_MODEL_KEY: HyperLeafRouteASpec(
        TOKEN_MODEL_KEY,
        HyperLeafRouteARole.TOKEN_ABLATION,
        "token-ablation",
        "3x32x64 visible RGB + 128-bin global spectrum",
        (DEFAULT_CONSISTENCY, TASK_ONLY),
    ),
    LOCAL_MIL_MODEL_KEY: HyperLeafRouteASpec(
        LOCAL_MIL_MODEL_KEY,
        HyperLeafRouteARole.LOCAL_ABLATION,
        "local-mil-ablation",
        "3x32x64 visible RGB + 16x128 local spectral tokens",
        (DEFAULT_CONSISTENCY, TASK_ONLY),
    ),
}

# These legacy CLI selectors depended on list positions and are intentionally
# not aliases. Keeping them explicit produces a clear migration error.
_UNSTABLE_LEGACY_SELECTORS = {
    "route_only",
    "--route_only",
    "token_only",
    "--token_only",
    "crossmodal_only",
    "--crossmodal_only",
}


class HyperLeafRouteAError(ValueError):
    """A Route A adapter request violates stable-selection policy."""


class HyperLeafRouteARunner:
    """Safe build and inner-selection surface for HyperLeaf Route A candidates."""

    relative_artifact_root = Path("runs_v2/artifacts/hyperleaf-route-a")

    def __init__(
        self,
        workspace_root: Path | str = Path(r"G:\Datasets\HyperCross"),
        *,
        registry: Optional[ModelRegistry] = None,
        selection_controller: Optional[SelectionController] = None,
    ) -> None:
        root = Path(workspace_root).resolve(strict=False)
        if root.drive.upper() != "G:":
            raise HyperLeafRouteAError("HyperLeaf Route A workspace must remain on G:")
        self.workspace_root = root
        self.artifact_root = root / self.relative_artifact_root
        self.registry = registry or create_hyperleaf_route_a_registry()
        self.selection_controller = selection_controller or SelectionController()
        self._validate_declarations()

    @property
    def model_keys(self) -> tuple[str, ...]:
        return tuple(sorted(HYPERLEAF_ROUTE_A_SPECS, key=lambda value: value.encode("utf-8")))

    def resolve(self, model_key: str) -> tuple[HyperLeafRouteASpec, ModelRegistration]:
        if not isinstance(model_key, str) or model_key in _UNSTABLE_LEGACY_SELECTORS:
            raise HyperLeafRouteAError(
                "legacy positional/flag selectors are unsupported; use one exact stable Model_Key"
            )
        try:
            spec = HYPERLEAF_ROUTE_A_SPECS[model_key]
        except KeyError as exc:
            raise HyperLeafRouteAError(f"unknown HyperLeaf Route A Model_Key: {model_key!r}") from exc
        return spec, self.registry.resolve(model_key)

    def build(self, model_key: str):
        """Build a fresh adapter instance, never a legacy runner selection."""

        self.resolve(model_key)
        return self.registry.build(model_key)

    def selection_candidate(self, model_key: str, setting_key: str) -> SelectionCandidate:
        """Materialize one predeclared route/consistency option."""

        spec, registration = self.resolve(model_key)
        settings = {item.setting_key: item for item in spec.consistency_settings}
        try:
            setting = settings[setting_key]
        except KeyError as exc:
            raise HyperLeafRouteAError(
                f"consistency setting {setting_key!r} is not declared for {model_key!r}"
            ) from exc
        candidate_id = f"{model_key}::{setting.setting_key}"
        return SelectionCandidate(
            candidate_id=candidate_id,
            configuration={
                "model_key": model_key,
                "representation_key": registration.input_contract.representation_key,
                "inference_branch": spec.inference_branch,
                "consistency_setting": setting.setting_key,
                "alignment_weight": setting.alignment_weight,
                "contrastive_weight": setting.contrastive_weight,
            },
        )

    def freeze_route_and_consistency(
        self,
        outer_fold: OuterFold,
        *,
        options: Iterable[tuple[str, str]],
        candidate_scores: Mapping[str, float],
        inner_source_sample_ids: Iterable[str],
        inner_prediction_hash: str,
        selection_policy: SelectionPolicy,
        decision_id: Optional[str] = None,
        frozen_utc: Optional[datetime] = None,
    ) -> SelectionDecision:
        """Freeze route and consistency choices using declared inner evidence only."""

        candidates = tuple(
            self.selection_candidate(model_key, setting_key)
            for model_key, setting_key in options
        )
        if not candidates:
            raise HyperLeafRouteAError("at least one route/consistency option is required")
        return self.selection_controller.freeze_selection(
            outer_fold,
            candidates=candidates,
            candidate_scores=candidate_scores,
            source_sample_ids=inner_source_sample_ids,
            source_prediction_hash=inner_prediction_hash,
            selection_policy=selection_policy,
            decision_id=decision_id,
            frozen_utc=frozen_utc,
        )

    def method_description(self, model_key: str) -> str:
        """Return mechanism wording that does not overclaim global attention."""

        spec, _ = self.resolve(model_key)
        if spec.role is HyperLeafRouteARole.GLOBAL_FUSION_BASELINE:
            return (
                "Global Route A global fusion baseline; each modality is globally "
                "encoded before fusion, with no single-token correspondence claim"
            )
        return spec.role.value

    def metadata(self, model_key: str) -> dict[str, object]:
        """Expose stable identity and mechanism-safe planning metadata."""

        spec, registration = self.resolve(model_key)
        return {
            "model_key": model_key,
            "implementation_kind": registration.identity.implementation_kind.value,
            "representation_key": registration.input_contract.representation_key,
            "inference_branch": spec.inference_branch,
            "input_resolution": spec.input_resolution,
            "method_description": self.method_description(model_key),
            "artifact_root": self.relative_artifact_root.as_posix(),
        }

    def _validate_declarations(self) -> None:
        if set(HYPERLEAF_ROUTE_A_SPECS) != set(self.registry.model_keys):
            raise HyperLeafRouteAError("Route A declarations and registry keys differ")
        for key, spec in HYPERLEAF_ROUTE_A_SPECS.items():
            if key != spec.model_key:
                raise HyperLeafRouteAError("candidate map key differs from stable Model_Key")
            setting_keys = tuple(item.setting_key for item in spec.consistency_settings)
            if len(setting_keys) != len(set(setting_keys)):
                raise HyperLeafRouteAError(f"duplicate consistency setting for {key!r}")


__all__ = [
    "DEFAULT_CONSISTENCY",
    "HYPERLEAF_ROUTE_A_SPECS",
    "HyperLeafRouteAError",
    "HyperLeafRouteARole",
    "HyperLeafRouteARunner",
    "HyperLeafRouteASpec",
    "TASK_ONLY",
]
