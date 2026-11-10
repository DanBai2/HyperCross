"""Native HyperLeaf candidate models for the governed G:-workspace runner.

These models rebuild the missing candidate branches without importing or
modifying the legacy ``src`` package.  They perform no data access and load no
historical checkpoints.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import ImplementationKind, InputContract, ModelIdentity


SOURCE_REVISION = "task-14.11.v1"

INDEPENDENT_TARGET_MODEL_KEY = "hyperleaf.candidate.independent-targets.v1"
PCA_INDEPENDENT_TARGET_MODEL_KEY = "hyperleaf.candidate.pca-independent-targets.v1"
RGB_PCA_CONSISTENCY_MODEL_KEY = "hyperleaf.candidate.rgb-pca-consistency.v1"

RGB_SPECTRUM_REPRESENTATION_KEY = "hyperleaf.candidate.rgb-global-spectrum.v1"
PCA_SPECTRUM_REPRESENTATION_KEY = "hyperleaf.candidate.pca-global-spectrum.fold-fitted.v1"
RGB_PCA_SPECTRUM_REPRESENTATION_KEY = (
    "hyperleaf.candidate.rgb-pca-global-spectrum.fold-fitted.v1"
)
RGB_SPECTRUM_PREPROCESSING_KEY = (
    "hyperleaf.candidate.rgb-spectrum.outer-train-target-normalization.v1"
)
PCA_SPECTRUM_PREPROCESSING_KEY = (
    "hyperleaf.candidate.pca-spectrum.outer-train-target-normalization-pca.v1"
)
RGB_PCA_SPECTRUM_PREPROCESSING_KEY = (
    "hyperleaf.candidate.rgb-pca-spectrum.outer-train-target-normalization-pca.v1"
)

RGB_BRANCH = "rgb"
PCA_BRANCH = "pca"
AVERAGE_BRANCH = "rgb-pca-average"
TARGET_ROUTED_BRANCH = "target-routed"


class _VisualEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, image: Tensor) -> Tensor:
        return self.projection(self.network(image).mean(dim=(2, 3)))


class _SpectrumEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, embedding_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.projection = nn.Linear(embedding_dim * 8, embedding_dim)

    def forward(self, spectrum: Tensor) -> Tensor:
        encoded = self.network(spectrum.unsqueeze(1)).flatten(1)
        return self.projection(encoded)


class _IndependentTargetExpert(nn.Module):
    """One complete visual/spectral regressor for exactly one target."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.visual = _VisualEncoder(embedding_dim)
        self.spectral = _SpectrumEncoder(embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, image: Tensor, spectrum: Tensor) -> Tensor:
        return self.head(torch.cat((self.visual(image), self.spectral(spectrum)), dim=1))


class HyperLeafIndependentTargets(nn.Module):
    """Three fully independent target regressors with no shared trainable state."""

    __benchmark_model_key__ = INDEPENDENT_TARGET_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def __init__(self, *, target_count: int = 3, embedding_dim: int = 64) -> None:
        super().__init__()
        if target_count != 3:
            raise ValueError("HyperLeaf candidates require exactly three targets")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.experts = nn.ModuleList(
            _IndependentTargetExpert(embedding_dim) for _ in range(target_count)
        )

    def forward(self, visible_rgb: Tensor, global_spectrum: Tensor) -> Tensor:
        return torch.cat(
            tuple(expert(visible_rgb, global_spectrum) for expert in self.experts),
            dim=1,
        )


class HyperLeafPcaIndependentTargets(HyperLeafIndependentTargets):
    """Independent target regressors driven by train-fold PCA pseudo-RGB."""

    __benchmark_model_key__ = PCA_INDEPENDENT_TARGET_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    def forward(self, pca_pseudo_rgb: Tensor, global_spectrum: Tensor) -> Tensor:
        return super().forward(pca_pseudo_rgb, global_spectrum)


class HyperLeafRgbPcaConsistency(nn.Module):
    """Independent RGB and train-fold-PCA views with fixed inference branches.

    The runner freezes branch routing from inner-variety evidence.  This model
    only executes that already-frozen route; it does not select routes itself.
    """

    __benchmark_model_key__ = RGB_PCA_CONSISTENCY_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = SOURCE_REVISION

    _VALID_ROUTES = frozenset({RGB_BRANCH, PCA_BRANCH, AVERAGE_BRANCH})

    def __init__(self, *, target_count: int = 3, embedding_dim: int = 64) -> None:
        super().__init__()
        self.rgb_model = HyperLeafIndependentTargets(
            target_count=target_count, embedding_dim=embedding_dim
        )
        self.pca_model = HyperLeafIndependentTargets(
            target_count=target_count, embedding_dim=embedding_dim
        )
        self.target_count = target_count

    def branch_predictions(
        self,
        visible_rgb: Tensor,
        pca_pseudo_rgb: Tensor,
        global_spectrum: Tensor,
    ) -> Mapping[str, Tensor]:
        rgb = self.rgb_model(visible_rgb, global_spectrum)
        pca = self.pca_model(pca_pseudo_rgb, global_spectrum)
        return {RGB_BRANCH: rgb, PCA_BRANCH: pca, AVERAGE_BRANCH: (rgb + pca) / 2.0}

    def forward(
        self,
        visible_rgb: Tensor,
        pca_pseudo_rgb: Tensor,
        global_spectrum: Tensor,
        *,
        inference_branch: str = AVERAGE_BRANCH,
        target_routes: Sequence[str] | None = None,
        return_branches: bool = False,
    ) -> Tensor | tuple[Tensor, Mapping[str, Tensor]]:
        branches = self.branch_predictions(visible_rgb, pca_pseudo_rgb, global_spectrum)
        if inference_branch == TARGET_ROUTED_BRANCH:
            routes = tuple(target_routes or (RGB_BRANCH, AVERAGE_BRANCH, PCA_BRANCH))
            if len(routes) != self.target_count or any(
                route not in self._VALID_ROUTES for route in routes
            ):
                raise ValueError(
                    "target-routed inference requires one declared RGB/PCA route per target"
                )
            prediction = torch.cat(
                tuple(branches[route][:, index : index + 1] for index, route in enumerate(routes)),
                dim=1,
            )
        else:
            try:
                prediction = branches[inference_branch]
            except KeyError as exc:
                raise ValueError(f"unknown RGB/PCA inference branch: {inference_branch!r}") from exc
        return (prediction, branches) if return_branches else prediction


def rgb_pca_consistency_loss(
    rgb_prediction: Tensor,
    pca_prediction: Tensor,
    *,
    weight: float,
) -> Tensor:
    """Return the symmetric prediction-consistency penalty for a frozen weight."""

    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError("consistency weight must be a finite non-negative number")
    if not torch.isfinite(torch.tensor(float(weight))):
        raise ValueError("consistency weight must be finite")
    if rgb_prediction.shape != pca_prediction.shape:
        raise ValueError("RGB and PCA predictions must have the same shape")
    return torch.mean((rgb_prediction - pca_prediction) ** 2) * float(weight)


RGB_INPUT_CONTRACT = InputContract(
    representation_key=RGB_SPECTRUM_REPRESENTATION_KEY,
    modalities=("visible_rgb", "global_spectrum"),
    tensor_shapes={"visible_rgb": (3, 32, 64), "global_spectrum": (128,)},
    dtypes={"visible_rgb": "float32", "global_spectrum": "float32"},
    channel_or_band_order={
        "visible_rgb": ("red", "green", "blue"),
        "global_spectrum": ("wavelength_ascending",),
    },
    preprocessing_key=RGB_SPECTRUM_PREPROCESSING_KEY,
)

PCA_INPUT_CONTRACT = InputContract(
    representation_key=PCA_SPECTRUM_REPRESENTATION_KEY,
    modalities=("pca_pseudo_rgb", "global_spectrum"),
    tensor_shapes={"pca_pseudo_rgb": (3, 32, 64), "global_spectrum": (128,)},
    dtypes={"pca_pseudo_rgb": "float32", "global_spectrum": "float32"},
    channel_or_band_order={
        "pca_pseudo_rgb": ("PC1", "PC2", "PC3"),
        "global_spectrum": ("wavelength_ascending",),
    },
    preprocessing_key=PCA_SPECTRUM_PREPROCESSING_KEY,
)

RGB_PCA_INPUT_CONTRACT = InputContract(
    representation_key=RGB_PCA_SPECTRUM_REPRESENTATION_KEY,
    modalities=("visible_rgb", "pca_pseudo_rgb", "global_spectrum"),
    tensor_shapes={
        "visible_rgb": (3, 32, 64),
        "pca_pseudo_rgb": (3, 32, 64),
        "global_spectrum": (128,),
    },
    dtypes={
        "visible_rgb": "float32",
        "pca_pseudo_rgb": "float32",
        "global_spectrum": "float32",
    },
    channel_or_band_order={
        "visible_rgb": ("red", "green", "blue"),
        "pca_pseudo_rgb": ("PC1", "PC2", "PC3"),
        "global_spectrum": ("wavelength_ascending",),
    },
    preprocessing_key=RGB_PCA_SPECTRUM_PREPROCESSING_KEY,
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


def hyperleaf_candidate_registrations() -> tuple[ModelRegistration, ...]:
    registrations = (
        _registration(
            INDEPENDENT_TARGET_MODEL_KEY,
            "HyperLeaf independent-target candidate",
            HyperLeafIndependentTargets,
            RGB_INPUT_CONTRACT,
        ),
        _registration(
            PCA_INDEPENDENT_TARGET_MODEL_KEY,
            "HyperLeaf train-fold PCA independent-target candidate",
            HyperLeafPcaIndependentTargets,
            PCA_INPUT_CONTRACT,
        ),
        _registration(
            RGB_PCA_CONSISTENCY_MODEL_KEY,
            "HyperLeaf target-routed RGB/PCA consistency candidate",
            HyperLeafRgbPcaConsistency,
            RGB_PCA_INPUT_CONTRACT,
        ),
    )
    return tuple(sorted(registrations, key=lambda item: item.model_key.encode("utf-8")))


def create_hyperleaf_candidate_registry() -> ModelRegistry:
    registry = ModelRegistry()
    for registration in hyperleaf_candidate_registrations():
        registry.register(registration)
    return registry


__all__ = [
    "AVERAGE_BRANCH",
    "INDEPENDENT_TARGET_MODEL_KEY",
    "PCA_BRANCH",
    "PCA_INDEPENDENT_TARGET_MODEL_KEY",
    "PCA_INPUT_CONTRACT",
    "PCA_SPECTRUM_PREPROCESSING_KEY",
    "PCA_SPECTRUM_REPRESENTATION_KEY",
    "RGB_BRANCH",
    "RGB_INPUT_CONTRACT",
    "RGB_PCA_CONSISTENCY_MODEL_KEY",
    "RGB_PCA_INPUT_CONTRACT",
    "RGB_PCA_SPECTRUM_PREPROCESSING_KEY",
    "RGB_PCA_SPECTRUM_REPRESENTATION_KEY",
    "RGB_SPECTRUM_PREPROCESSING_KEY",
    "RGB_SPECTRUM_REPRESENTATION_KEY",
    "SOURCE_REVISION",
    "TARGET_ROUTED_BRANCH",
    "HyperLeafIndependentTargets",
    "HyperLeafPcaIndependentTargets",
    "HyperLeafRgbPcaConsistency",
    "create_hyperleaf_candidate_registry",
    "hyperleaf_candidate_registrations",
    "rgb_pca_consistency_loss",
]
