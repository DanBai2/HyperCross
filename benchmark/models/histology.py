"""Stable Histology model adapters for the protocol-first benchmark runner.

The adapters reuse the legacy model definitions as read-only implementations.
They never load old checkpoints and all new state is owned by ``runs_v2``
runners. Classic estimators are constructed lazily so importing the benchmark
package does not make scikit-learn a core dependency.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from benchmark.adapters.histology import (
    CACHED_SPATIAL_MEAN_PCA_KEY,
    FULL_CUBE_SPATIAL_PCA_KEY,
    RGB_PREVIEW_KEY,
)
from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import ImplementationKind, InputContract, ModelIdentity


SOURCE_REVISION = "task-15.5.v1"

PCA_ONLY_MODEL_KEY = "histology.pca-only.v1"
RGB_ONLY_MODEL_KEY = "histology.rgb-only.v1"
LOCAL_ROUTE_A_MODEL_KEY = "histology.local-route-a.v1"
MIL_CONCAT_MODEL_KEY = "histology.mil-concat.v1"
ATTENTION_DISABLED_MODEL_KEY = "histology.local-route-a.attention-disabled.v1"
SVM_MODEL_KEY = "histology.classic.svm-rbf.v1"
RF_MODEL_KEY = "histology.classic.random-forest.v1"

PCA_ONLY_INFERENCE_BRANCH = "pca-only"
RGB_ONLY_INFERENCE_BRANCH = "rgb-only"
LOCAL_ROUTE_A_INFERENCE_BRANCH = "local-route-a"
MIL_CONCAT_INFERENCE_BRANCH = "mil-concat"
ATTENTION_DISABLED_INFERENCE_BRANCH = "local-route-a-attention-disabled"
SVM_INFERENCE_BRANCH = "svm-rbf"
RF_INFERENCE_BRANCH = "random-forest"


class _LegacyTorchAdapter(nn.Module):
    """Own a newly initialized legacy network without changing legacy source."""

    legacy_class_name: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        legacy = import_module("src.hypercross_routeA")
        model_class = getattr(legacy, self.legacy_class_name)
        self.model = model_class(*args, **kwargs)

    def forward(self, *inputs: Tensor, **kwargs: Any) -> Any:
        return self.model(*inputs, **kwargs)


class HistologyPCAOnlyAdapter(_LegacyTorchAdapter):
    """PCA/spectral-only matched modality control."""

    legacy_class_name = "SpectralOnly"
    __benchmark_model_key__ = PCA_ONLY_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 2, embedding_dim: int = 128) -> None:
        super().__init__(out_dim, d=embedding_dim)

    def forward(self, pca_spectrum: Tensor, *, return_aux: bool = False) -> Any:
        return self.model(None, pca_spectrum, return_aux=return_aux)


class HistologyRGBOnlyAdapter(_LegacyTorchAdapter):
    """RGB-only matched modality control."""

    legacy_class_name = "VisualOnly"
    __benchmark_model_key__ = RGB_ONLY_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 2, embedding_dim: int = 128) -> None:
        super().__init__(out_dim, d=embedding_dim)

    def forward(self, rgb: Tensor, *, return_aux: bool = False) -> Any:
        return self.model(rgb, return_aux=return_aux)


class HistologyLocalRouteAAdapter(_LegacyTorchAdapter):
    """Local 4x4 spatial-spectral Route A candidate."""

    legacy_class_name = "MILLocalRouteA"
    __benchmark_model_key__ = LOCAL_ROUTE_A_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(
        self,
        out_dim: int = 2,
        embedding_dim: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__(out_dim, d=embedding_dim, heads=attention_heads)

    def forward(
        self,
        rgb: Tensor,
        local_pca_tokens: Tensor,
        *,
        return_aux: bool = False,
    ) -> Any:
        return self.model(rgb, local_pca_tokens, return_aux=return_aux)


class HistologyMILConcatAdapter(_LegacyTorchAdapter):
    """MIL pooled concatenation matched-input control."""

    legacy_class_name = "MILConcat"
    __benchmark_model_key__ = MIL_CONCAT_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 2, embedding_dim: int = 64) -> None:
        super().__init__(out_dim, d=embedding_dim)

    def forward(
        self,
        rgb: Tensor,
        local_pca_tokens: Tensor,
        *,
        return_aux: bool = False,
    ) -> Any:
        return self.model(rgb, local_pca_tokens, return_aux=return_aux)


class HistologyAttentionDisabledControl(nn.Module):
    """Local Route A ablation preserving tokenization and MIL without attention.

    The visual and spectral branches keep the candidate's 4x4 token shape,
    positional parameter, normalization, learned MIL pooling, and classifier.
    Bidirectional cross-modal attention is the only removed mechanism.
    """

    __benchmark_model_key__ = ATTENTION_DISABLED_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, out_dim: int = 2, embedding_dim: int = 64) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, embedding_dim, 3, 2, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.spectral = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, 16, embedding_dim))
        nn.init.normal_(self.position, std=0.02)
        self.visual_norm = nn.LayerNorm(embedding_dim)
        self.spectral_norm = nn.LayerNorm(embedding_dim)
        self.visual_pool = nn.Linear(embedding_dim, 1)
        self.spectral_pool = nn.Linear(embedding_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, out_dim),
        )
        self.temp = nn.Parameter(torch.tensor(0.07).log())
        self.task = "cls"

    def forward(
        self,
        rgb: Tensor,
        local_pca_tokens: Tensor,
        *,
        return_aux: bool = False,
    ) -> Any:
        visual_tokens = self.visual(rgb).flatten(2).transpose(1, 2) + self.position
        spectral_tokens = self.spectral(local_pca_tokens)
        visual_features = self.visual_norm(visual_tokens)
        spectral_features = self.spectral_norm(spectral_tokens)
        visual_weights = F.softmax(self.visual_pool(visual_features).squeeze(-1), dim=1)
        spectral_weights = F.softmax(
            self.spectral_pool(spectral_features).squeeze(-1), dim=1
        )
        visual_embedding = (visual_weights.unsqueeze(-1) * visual_features).sum(1)
        spectral_embedding = (
            spectral_weights.unsqueeze(-1) * spectral_features
        ).sum(1)
        logits = self.head(torch.cat((visual_embedding, spectral_embedding), dim=1))
        if return_aux:
            return logits, visual_embedding, spectral_embedding
        return logits


class _ClassicEstimatorAdapter:
    """Expose sklearn classifiers through the registry's framework-neutral API."""

    estimator: Any

    def parameters(self) -> Iterator[Any]:
        return iter(())

    def fit(self, features: Any, labels: Any) -> "_ClassicEstimatorAdapter":
        self.estimator.fit(features, labels)
        return self

    def predict(self, features: Any) -> Any:
        return self.estimator.predict(features)

    def predict_proba(self, features: Any) -> Any:
        return self.estimator.predict_proba(features)


class HistologySVMAdapter(_ClassicEstimatorAdapter):
    """RBF SVM over the governed PCA representation."""

    __benchmark_model_key__ = SVM_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.CLASSIC
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, *, seed: int = 42, class_weight: str | None = "balanced") -> None:
        from sklearn.svm import SVC

        self.estimator = SVC(
            kernel="rbf",
            probability=True,
            class_weight=class_weight,
            random_state=seed,
        )


class HistologyRandomForestAdapter(_ClassicEstimatorAdapter):
    """Random Forest over the governed PCA representation."""

    __benchmark_model_key__ = RF_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.CLASSIC
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(
        self,
        *,
        seed: int = 42,
        estimators: int = 300,
        class_weight: str | None = "balanced",
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.estimator = RandomForestClassifier(
            n_estimators=estimators,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )


def _contract(
    representation_key: str,
    modalities: tuple[str, ...],
    shapes: dict[str, tuple[int, ...]],
    orders: dict[str, tuple[str, ...]],
    preprocessing_key: str,
) -> InputContract:
    return InputContract(
        representation_key=representation_key,
        modalities=modalities,
        tensor_shapes=shapes,
        dtypes={modality: "float32" for modality in modalities},
        channel_or_band_order=orders,
        preprocessing_key=preprocessing_key,
    )


PCA_CONTRACT = _contract(
    CACHED_SPATIAL_MEAN_PCA_KEY,
    ("pca_spectrum",),
    {"pca_spectrum": (128,)},
    {"pca_spectrum": ("component_order_ascending",)},
    "histology.cached-spatial-mean-pca.read-only.v1",
)
RGB_CONTRACT = _contract(
    RGB_PREVIEW_KEY,
    ("rgb",),
    {"rgb": (3, 64, 64)},
    {"rgb": ("red", "green", "blue")},
    "histology.rgb.unit-interval.v1",
)
LOCAL_CONTRACT = _contract(
    FULL_CUBE_SPATIAL_PCA_KEY,
    ("rgb", "local_pca_tokens"),
    {"rgb": (3, 64, 64), "local_pca_tokens": (16, 128)},
    {
        "rgb": ("red", "green", "blue"),
        "local_pca_tokens": ("spatial_row_major", "component_order_ascending"),
    },
    "histology.full-cube-spatial-pca.fold-scoped.v1",
)


def _parameter_count(builder: type[Any]) -> int:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        instance = builder()
    return sum(parameter.numel() for parameter in instance.parameters())


def _registration(
    model_key: str,
    display_name: str,
    builder: type[Any],
    implementation_kind: ImplementationKind,
    contract: InputContract,
) -> ModelRegistration:
    return ModelRegistration(
        model_key=model_key,
        identity=ModelIdentity(
            model_key=model_key,
            display_name=display_name,
            class_fully_qualified_name=f"{builder.__module__}.{builder.__qualname__}",
            implementation_kind=implementation_kind,
            parameter_count=_parameter_count(builder),
            source_revision=SOURCE_REVISION,
        ),
        input_contract=contract,
        builder=builder,
        implementation_kind=implementation_kind,
        source_revision=SOURCE_REVISION,
    )


def histology_model_registrations() -> tuple[ModelRegistration, ...]:
    """Return every task-15.5 model registration in stable key order."""

    declarations = (
        _registration(
            PCA_ONLY_MODEL_KEY,
            "Histology PCA-only",
            HistologyPCAOnlyAdapter,
            ImplementationKind.IN_REPO,
            PCA_CONTRACT,
        ),
        _registration(
            RGB_ONLY_MODEL_KEY,
            "Histology RGB-only",
            HistologyRGBOnlyAdapter,
            ImplementationKind.IN_REPO,
            RGB_CONTRACT,
        ),
        _registration(
            LOCAL_ROUTE_A_MODEL_KEY,
            "Histology Local Route A",
            HistologyLocalRouteAAdapter,
            ImplementationKind.IN_REPO,
            LOCAL_CONTRACT,
        ),
        _registration(
            MIL_CONCAT_MODEL_KEY,
            "Histology MIL-concat control",
            HistologyMILConcatAdapter,
            ImplementationKind.IN_REPO,
            LOCAL_CONTRACT,
        ),
        _registration(
            ATTENTION_DISABLED_MODEL_KEY,
            "Histology Local Route A attention-disabled control",
            HistologyAttentionDisabledControl,
            ImplementationKind.IN_REPO,
            LOCAL_CONTRACT,
        ),
        _registration(
            SVM_MODEL_KEY,
            "Histology SVM-RBF",
            HistologySVMAdapter,
            ImplementationKind.CLASSIC,
            PCA_CONTRACT,
        ),
        _registration(
            RF_MODEL_KEY,
            "Histology Random Forest",
            HistologyRandomForestAdapter,
            ImplementationKind.CLASSIC,
            PCA_CONTRACT,
        ),
    )
    return tuple(sorted(declarations, key=lambda item: item.model_key.encode("utf-8")))


def create_histology_model_registry() -> ModelRegistry:
    """Create an isolated registry containing only governed Histology models."""

    registry = ModelRegistry()
    for registration in histology_model_registrations():
        registry.register(registration)
    return registry


__all__ = [
    "ATTENTION_DISABLED_INFERENCE_BRANCH",
    "ATTENTION_DISABLED_MODEL_KEY",
    "HistologyAttentionDisabledControl",
    "HistologyLocalRouteAAdapter",
    "HistologyMILConcatAdapter",
    "HistologyPCAOnlyAdapter",
    "HistologyRGBOnlyAdapter",
    "HistologyRandomForestAdapter",
    "HistologySVMAdapter",
    "LOCAL_CONTRACT",
    "LOCAL_ROUTE_A_INFERENCE_BRANCH",
    "LOCAL_ROUTE_A_MODEL_KEY",
    "MIL_CONCAT_INFERENCE_BRANCH",
    "MIL_CONCAT_MODEL_KEY",
    "PCA_CONTRACT",
    "PCA_ONLY_INFERENCE_BRANCH",
    "PCA_ONLY_MODEL_KEY",
    "RF_INFERENCE_BRANCH",
    "RF_MODEL_KEY",
    "RGB_CONTRACT",
    "RGB_ONLY_INFERENCE_BRANCH",
    "RGB_ONLY_MODEL_KEY",
    "SOURCE_REVISION",
    "SVM_INFERENCE_BRANCH",
    "SVM_MODEL_KEY",
    "create_histology_model_registry",
    "histology_model_registrations",
]
