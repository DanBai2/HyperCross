"""Stable adapter for the legacy Histology global Route A baseline."""
from __future__ import annotations

from importlib import import_module
from typing import Any

from torch import Tensor, nn

from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import ImplementationKind, InputContract, ModelIdentity

GLOBAL_ROUTE_A_MODEL_KEY = "histology.global-route-a.v1"
GLOBAL_ROUTE_A_INFERENCE_BRANCH = "global-route-a"
SOURCE_REVISION = "task-15.11.v1"


class HistologyGlobalRouteAAdapter(nn.Module):
    """Own a new legacy global-fusion Route A network."""

    __benchmark_model_key__ = GLOBAL_ROUTE_A_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 2, embedding_dim: int = 128) -> None:
        super().__init__()
        legacy = import_module("src.hypercross_routeA")
        self.model = legacy.CrossModalRouteA("cls", out_dim, d=embedding_dim)

    def forward(
        self, rgb: Tensor, global_pca_spectrum: Tensor, *, return_aux: bool = False
    ) -> Any:
        return self.model(rgb, global_pca_spectrum, return_aux=return_aux)


GLOBAL_ROUTE_A_CONTRACT = InputContract(
    representation_key="histology-hsi-gb.rgb-global-spectrum128.legacy",
    modalities=("rgb", "global_pca_spectrum"),
    tensor_shapes={"rgb": (3, 64, 64), "global_pca_spectrum": (128,)},
    dtypes={"rgb": "float32", "global_pca_spectrum": "float32"},
    channel_or_band_order={
        "rgb": ("red", "green", "blue"),
        "global_pca_spectrum": ("component_order_ascending",),
    },
    preprocessing_key="histology.cached-spatial-mean-pca.read-only.v1",
)


def create_histology_baseline_registry() -> ModelRegistry:
    """Create the isolated registry for task 15.11 global Route A."""

    with __import__("torch").random.fork_rng(devices=[]):
        __import__("torch").manual_seed(0)
        parameter_count = sum(
            parameter.numel() for parameter in HistologyGlobalRouteAAdapter().parameters()
        )
    registration = ModelRegistration(
        model_key=GLOBAL_ROUTE_A_MODEL_KEY,
        identity=ModelIdentity(
            model_key=GLOBAL_ROUTE_A_MODEL_KEY,
            display_name="Histology global Route A",
            class_fully_qualified_name=(
                f"{HistologyGlobalRouteAAdapter.__module__}."
                f"{HistologyGlobalRouteAAdapter.__qualname__}"
            ),
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=parameter_count,
            source_revision=SOURCE_REVISION,
        ),
        input_contract=GLOBAL_ROUTE_A_CONTRACT,
        builder=HistologyGlobalRouteAAdapter,
        implementation_kind=ImplementationKind.IN_REPO,
        source_revision=SOURCE_REVISION,
    )
    registry = ModelRegistry()
    registry.register(registration)
    return registry


__all__ = [
    "GLOBAL_ROUTE_A_CONTRACT",
    "GLOBAL_ROUTE_A_INFERENCE_BRANCH",
    "GLOBAL_ROUTE_A_MODEL_KEY",
    "HistologyGlobalRouteAAdapter",
    "SOURCE_REVISION",
    "create_histology_baseline_registry",
]
