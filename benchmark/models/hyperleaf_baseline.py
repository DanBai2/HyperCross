"""Stable adapters for governed HyperLeaf baseline reproduction.

These adapters construct fresh legacy architectures without invoking a legacy
runner or writing to legacy ``src``/``runs`` locations.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from torch import Tensor, nn

from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import ImplementationKind, InputContract, ModelIdentity

SOURCE_REVISION = "task-14.10.v1"
VISUAL_MODEL_KEY = "hyperleaf.baseline.visual-only.v1"
SPECTRAL_MODEL_KEY = "hyperleaf.baseline.spectral-only.v1"
CONCAT_MODEL_KEY = "hyperleaf.baseline.concat.v1"

GLOBAL_REPRESENTATION_KEY = "hyperleaf.visible-rgb-global-spectrum.v1"
GLOBAL_PREPROCESSING_KEY = "hyperleaf.baseline.rgb-global-spectrum.fold-target-normalization.v1"
GLOBAL_INPUT_CONTRACT = InputContract(
    representation_key=GLOBAL_REPRESENTATION_KEY,
    modalities=("visible_rgb", "global_spectrum"),
    tensor_shapes={"visible_rgb": (3, 32, 64), "global_spectrum": (128,)},
    dtypes={"visible_rgb": "float32", "global_spectrum": "float32"},
    channel_or_band_order={
        "visible_rgb": ("red", "green", "blue"),
        "global_spectrum": ("wavelength_ascending",),
    },
    preprocessing_key=GLOBAL_PREPROCESSING_KEY,
)


class _BaselineAdapter(nn.Module):
    legacy_class_name: str
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 3, embedding_dim: int = 128) -> None:
        super().__init__()
        legacy = import_module("src.hypercross_routeA")
        self.model = getattr(legacy, self.legacy_class_name)(out_dim, d=embedding_dim)

    def forward(self, visible_rgb: Tensor, global_spectrum: Tensor, *, return_aux: bool = False) -> Any:
        return self.model(visible_rgb, global_spectrum, return_aux=return_aux)


class HyperLeafVisualOnlyAdapter(_BaselineAdapter):
    legacy_class_name = "VisualOnly"
    __benchmark_model_key__ = VISUAL_MODEL_KEY


class HyperLeafSpectralOnlyAdapter(_BaselineAdapter):
    legacy_class_name = "SpectralOnly"
    __benchmark_model_key__ = SPECTRAL_MODEL_KEY


class HyperLeafConcatAdapter(_BaselineAdapter):
    legacy_class_name = "ConcatModel"
    __benchmark_model_key__ = CONCAT_MODEL_KEY


def _registration(model_key: str, display_name: str, builder: type[nn.Module]) -> ModelRegistration:
    model = builder()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return ModelRegistration(
        model_key=model_key,
        identity=ModelIdentity(
            model_key=model_key,
            display_name=display_name,
            class_fully_qualified_name=f"{builder.__module__}.{builder.__qualname__}",
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=parameter_count,
            source_revision=SOURCE_REVISION,
        ),
        input_contract=GLOBAL_INPUT_CONTRACT,
        builder=builder,
        implementation_kind=ImplementationKind.IN_REPO,
        source_revision=SOURCE_REVISION,
    )


def hyperleaf_baseline_registrations() -> tuple[ModelRegistration, ...]:
    registrations = (
        _registration(VISUAL_MODEL_KEY, "HyperLeaf visible-only matched control", HyperLeafVisualOnlyAdapter),
        _registration(SPECTRAL_MODEL_KEY, "HyperLeaf spectral-only matched control", HyperLeafSpectralOnlyAdapter),
        _registration(CONCAT_MODEL_KEY, "HyperLeaf concatenation matched control", HyperLeafConcatAdapter),
    )
    return tuple(sorted(registrations, key=lambda item: item.model_key.encode("utf-8")))


def create_hyperleaf_baseline_registry() -> ModelRegistry:
    registry = ModelRegistry()
    for registration in hyperleaf_baseline_registrations():
        registry.register(registration)
    return registry


__all__ = [
    "CONCAT_MODEL_KEY",
    "GLOBAL_INPUT_CONTRACT",
    "GLOBAL_PREPROCESSING_KEY",
    "GLOBAL_REPRESENTATION_KEY",
    "HyperLeafConcatAdapter",
    "HyperLeafSpectralOnlyAdapter",
    "HyperLeafVisualOnlyAdapter",
    "SOURCE_REVISION",
    "SPECTRAL_MODEL_KEY",
    "VISUAL_MODEL_KEY",
    "create_hyperleaf_baseline_registry",
    "hyperleaf_baseline_registrations",
]
