"""Stable HyperLeaf Route A adapters over read-only in-repository models.

The classes in this module construct fresh networks from ``src.hypercross_routeA``.
They do not invoke the legacy CLI, read old checkpoints, or write to ``runs``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
from torch import Tensor, nn

from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import ImplementationKind, InputContract, ModelIdentity


SOURCE_REVISION = "task-14.5.v1"

GLOBAL_FUSION_MODEL_KEY = "hyperleaf.route-a.global-fusion.v1"
TARGET_EXPERTS_MODEL_KEY = "hyperleaf.route-a.target-experts.v1"
SHARED_HEADS_MODEL_KEY = "hyperleaf.route-a.shared-heads.v1"
TOKEN_MODEL_KEY = "hyperleaf.route-a.token.v1"
LOCAL_MIL_MODEL_KEY = "hyperleaf.route-a.local-mil.v1"

GLOBAL_FUSION_REPRESENTATION_KEY = "hyperleaf.visible-rgb-global-spectrum.v1"
LOCAL_TOKEN_REPRESENTATION_KEY = "hyperleaf.visible-rgb-local-spectrum.v1"
GLOBAL_PREPROCESSING_KEY = "hyperleaf.route-a.rgb-global-spectrum.fold-target-normalization.v1"
LOCAL_PREPROCESSING_KEY = "hyperleaf.route-a.rgb-local-spectrum.fold-target-normalization.v1"


class _LegacyRouteAAdapter(nn.Module):
    """Own a newly initialized legacy model while preserving legacy source."""

    legacy_class_name: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        legacy = import_module("src.hypercross_routeA")
        model_class = getattr(legacy, self.legacy_class_name)
        self.model = model_class(*args, **kwargs)

    def forward(self, *inputs: Tensor, **kwargs: Any) -> Any:
        return self.model(*inputs, **kwargs)


class HyperLeafGlobalFusionAdapter(_LegacyRouteAAdapter):
    """Global visual/spectral fusion baseline.

    Each modality is globally encoded before fusion. The adapter deliberately
    makes no single-token correspondence claim.
    """

    legacy_class_name = "CrossModalRouteA"
    __benchmark_model_key__ = GLOBAL_FUSION_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 3, embedding_dim: int = 128, heads: int = 4) -> None:
        super().__init__(task="reg", out_dim=out_dim, d=embedding_dim, heads=heads)


class HyperLeafTargetExpertsAdapter(nn.Module):
    """Three independent global-fusion experts, one per HyperLeaf target."""

    __benchmark_model_key__ = TARGET_EXPERTS_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 3, embedding_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        if out_dim != 3:
            raise ValueError("HyperLeaf target-specific experts require exactly three targets")
        legacy = import_module("src.hypercross_routeA")
        self.experts = nn.ModuleList(
            legacy.CrossModalRouteA(task="reg", out_dim=1, d=embedding_dim, heads=heads)
            for _ in range(out_dim)
        )
        self.task = "reg"

    def forward(
        self,
        visible_rgb: Tensor,
        global_spectrum: Tensor,
        *,
        return_aux: bool = False,
    ) -> Any:
        outputs = [
            expert(visible_rgb, global_spectrum, return_aux=return_aux)
            for expert in self.experts
        ]
        if not return_aux:
            return torch.cat(outputs, dim=1)
        predictions, visual, spectral = zip(*outputs)
        return (
            torch.cat(predictions, dim=1),
            torch.stack(visual, dim=1),
            torch.stack(spectral, dim=1),
        )


class HyperLeafSharedHeadsAdapter(_LegacyRouteAAdapter):
    """Shared visual/spectral encoder with one regression head per target."""

    legacy_class_name = "MultiHeadCrossModal"
    __benchmark_model_key__ = SHARED_HEADS_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 3, embedding_dim: int = 128, heads: int = 4) -> None:
        super().__init__(task="reg", out_dim=out_dim, d=embedding_dim, heads=heads)


class HyperLeafTokenRouteAAdapter(_LegacyRouteAAdapter):
    """Patch/band-token Route A ablation over the governed global inputs."""

    legacy_class_name = "TokenRouteA"
    __benchmark_model_key__ = TOKEN_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 3, embedding_dim: int = 128, heads: int = 4) -> None:
        super().__init__(task="reg", out_dim=out_dim, d=embedding_dim, heads=heads)


class HyperLeafLocalMILRouteAAdapter(_LegacyRouteAAdapter):
    """Local 4x4 spectral-token and visual-patch MIL Route A ablation."""

    legacy_class_name = "MILRouteA"
    __benchmark_model_key__ = LOCAL_MIL_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 3, embedding_dim: int = 64, heads: int = 4) -> None:
        super().__init__(out_dim=out_dim, d=embedding_dim, heads=heads)
        self.model.task = "reg"


GLOBAL_INPUT_CONTRACT = InputContract(
    representation_key=GLOBAL_FUSION_REPRESENTATION_KEY,
    modalities=("visible_rgb", "global_spectrum"),
    tensor_shapes={"visible_rgb": (3, 32, 64), "global_spectrum": (128,)},
    dtypes={"visible_rgb": "float32", "global_spectrum": "float32"},
    channel_or_band_order={
        "visible_rgb": ("red", "green", "blue"),
        "global_spectrum": ("wavelength_ascending",),
    },
    preprocessing_key=GLOBAL_PREPROCESSING_KEY,
)

LOCAL_INPUT_CONTRACT = InputContract(
    representation_key=LOCAL_TOKEN_REPRESENTATION_KEY,
    modalities=("visible_rgb", "local_spectral_tokens"),
    tensor_shapes={"visible_rgb": (3, 32, 64), "local_spectral_tokens": (16, 128)},
    dtypes={"visible_rgb": "float32", "local_spectral_tokens": "float32"},
    channel_or_band_order={
        "visible_rgb": ("red", "green", "blue"),
        "local_spectral_tokens": ("spatial_row_major", "wavelength_ascending"),
    },
    preprocessing_key=LOCAL_PREPROCESSING_KEY,
)


def _parameter_count(builder: type[nn.Module]) -> int:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = builder()
    return sum(parameter.numel() for parameter in model.parameters())


def _registration(
    model_key: str,
    display_name: str,
    builder: type[nn.Module],
    contract: InputContract,
) -> ModelRegistration:
    return ModelRegistration(
        model_key=model_key,
        identity=ModelIdentity(
            model_key=model_key,
            display_name=display_name,
            class_fully_qualified_name=f"{builder.__module__}.{builder.__qualname__}",
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=_parameter_count(builder),
            source_revision=SOURCE_REVISION,
        ),
        input_contract=contract,
        builder=builder,
        implementation_kind=ImplementationKind.IN_REPO,
        source_revision=SOURCE_REVISION,
    )


def hyperleaf_route_a_registrations() -> tuple[ModelRegistration, ...]:
    """Return each task-14.5 adapter registration in stable key order."""

    registrations = (
        _registration(
            GLOBAL_FUSION_MODEL_KEY,
            "HyperLeaf Route A global fusion baseline",
            HyperLeafGlobalFusionAdapter,
            GLOBAL_INPUT_CONTRACT,
        ),
        _registration(
            TARGET_EXPERTS_MODEL_KEY,
            "HyperLeaf Route A target-specific experts",
            HyperLeafTargetExpertsAdapter,
            GLOBAL_INPUT_CONTRACT,
        ),
        _registration(
            SHARED_HEADS_MODEL_KEY,
            "HyperLeaf Route A shared encoder and target heads",
            HyperLeafSharedHeadsAdapter,
            GLOBAL_INPUT_CONTRACT,
        ),
        _registration(
            TOKEN_MODEL_KEY,
            "HyperLeaf Route A token ablation",
            HyperLeafTokenRouteAAdapter,
            GLOBAL_INPUT_CONTRACT,
        ),
        _registration(
            LOCAL_MIL_MODEL_KEY,
            "HyperLeaf Route A local MIL ablation",
            HyperLeafLocalMILRouteAAdapter,
            LOCAL_INPUT_CONTRACT,
        ),
    )
    return tuple(sorted(registrations, key=lambda item: item.model_key.encode("utf-8")))


def create_hyperleaf_route_a_registry() -> ModelRegistry:
    """Create an isolated registry containing only task-14.5 adapters."""

    registry = ModelRegistry()
    for registration in hyperleaf_route_a_registrations():
        registry.register(registration)
    return registry


__all__ = [
    "GLOBAL_FUSION_MODEL_KEY",
    "GLOBAL_FUSION_REPRESENTATION_KEY",
    "GLOBAL_INPUT_CONTRACT",
    "GLOBAL_PREPROCESSING_KEY",
    "HyperLeafGlobalFusionAdapter",
    "HyperLeafLocalMILRouteAAdapter",
    "HyperLeafSharedHeadsAdapter",
    "HyperLeafTargetExpertsAdapter",
    "HyperLeafTokenRouteAAdapter",
    "LOCAL_INPUT_CONTRACT",
    "LOCAL_MIL_MODEL_KEY",
    "LOCAL_PREPROCESSING_KEY",
    "LOCAL_TOKEN_REPRESENTATION_KEY",
    "SHARED_HEADS_MODEL_KEY",
    "SOURCE_REVISION",
    "TARGET_EXPERTS_MODEL_KEY",
    "TOKEN_MODEL_KEY",
    "create_hyperleaf_route_a_registry",
    "hyperleaf_route_a_registrations",
]
