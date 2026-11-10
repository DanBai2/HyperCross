"""Task-25 spectral nuisance-alignment candidates for the governed HyperLeaf runner.

Exploratory waves 1/6/7/9 established that the only reproducible positive
cross-variety mechanism on HyperLeaf is variety-adversarial training of the
SPECTRAL branch, that it must not be applied to a fused representation, and that
it trades a small GrainWeight loss for a large Gsw gain.  The frozen task-24.3
space could not express this mechanism at all: every optimization candidate in it
was a variant inside one architecture, their inner-variety OOF scores spanned
0.0014, and task 24.4 consequently fell back in all four outer folds.

These models add the missing mechanism as first-class governed models.  They are
data-agnostic, perform no data access, select no route themselves, and read no
labels beyond the targets and group ids supplied by the runner.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from benchmark.models.hyperleaf_candidates import _SpectrumEncoder
from benchmark.models.hyperleaf_optimization import hyperleaf_optimization_registrations
from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import ImplementationKind, InputContract, ModelIdentity


NUISANCE_ALIGNMENT_SOURCE_REVISION = "task-25.1.v1"
SPECTRAL_ONLY_MODEL_KEY = "hyperleaf.task25.spectral-only.v1"
SPECTRAL_ADVERSARIAL_03_MODEL_KEY = "hyperleaf.task25.spectral-variety-adversarial-lam0.3.v1"
SPECTRAL_ADVERSARIAL_10_MODEL_KEY = "hyperleaf.task25.spectral-variety-adversarial-lam1.0.v1"
SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY = "hyperleaf.task25.spectral-routed-adversarial-lam1.0.v1"

SPECTRUM_ONLY_REPRESENTATION_KEY = "hyperleaf.task25.spectral-global-spectrum.v1"
SPECTRUM_ONLY_PREPROCESSING_KEY = (
    "hyperleaf.task25.spectral-spectrum.outer-train-target-normalization.v1"
)

SPECTRAL_CLEAN_BRANCH = "spectral-clean"
SPECTRAL_ADVERSARIAL_BRANCH = "spectral-adversarial"
SPECTRAL_ROUTES = (SPECTRAL_CLEAN_BRANCH, SPECTRAL_ADVERSARIAL_BRANCH)

VARIETY_COUNT = 4


class GradientReversal(torch.autograd.Function):
    """Identity forward with a sign-flipped, strength-scaled backward pass."""

    @staticmethod
    def forward(ctx, values: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.strength * gradient, None


def _regression_head(embedding_dim: int, target_count: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(embedding_dim, target_count),
    )


def _variety_head(embedding_dim: int, variety_count: int) -> nn.Sequential:
    if embedding_dim < 2:
        raise ValueError("embedding_dim must be at least two for a variety head")
    return nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim // 2),
        nn.GELU(),
        nn.Linear(embedding_dim // 2, variety_count),
    )


class HyperLeafSpectralOnly(nn.Module):
    """Matched control: the governed spectrum encoder with one three-target head."""

    __benchmark_model_key__ = SPECTRAL_ONLY_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = NUISANCE_ALIGNMENT_SOURCE_REVISION

    def __init__(self, *, target_count: int = 3, embedding_dim: int = 64) -> None:
        super().__init__()
        if target_count != 3:
            raise ValueError("HyperLeaf task-25 requires exactly three targets")
        self.target_count = target_count
        self.spectral = _SpectrumEncoder(embedding_dim)
        self.head = _regression_head(embedding_dim, target_count)

    def forward(self, global_spectrum: Tensor) -> Tensor:
        return self.head(self.spectral(global_spectrum))


class HyperLeafSpectralVarietyAdversarial(nn.Module):
    """Spectral regressor whose embedding is made variety-uninformative by a GRL."""

    __benchmark_model_key__ = SPECTRAL_ADVERSARIAL_10_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = NUISANCE_ALIGNMENT_SOURCE_REVISION

    def __init__(
        self,
        *,
        target_count: int = 3,
        embedding_dim: int = 64,
        variety_count: int = VARIETY_COUNT,
        strength: float = 1.0,
    ) -> None:
        super().__init__()
        if target_count != 3:
            raise ValueError("HyperLeaf task-25 requires exactly three targets")
        if variety_count < 2:
            raise ValueError("variety_count must be at least two")
        if not float(strength) >= 0.0:
            raise ValueError("adversarial strength must be non-negative")
        self.target_count = target_count
        self.variety_count = variety_count
        self.strength = float(strength)
        self.spectral = _SpectrumEncoder(embedding_dim)
        self.head = _regression_head(embedding_dim, target_count)
        self.variety = _variety_head(embedding_dim, variety_count)

    def forward(self, global_spectrum: Tensor) -> Tensor:
        return self.head(self.spectral(global_spectrum))

    def variety_logits(self, global_spectrum: Tensor) -> Tensor:
        embedding = self.spectral(global_spectrum)
        return self.variety(GradientReversal.apply(embedding, self.strength))


class HyperLeafSpectralAdversarial03(HyperLeafSpectralVarietyAdversarial):
    """Frozen lambda=0.3 arm of the same mechanism."""

    __benchmark_model_key__ = SPECTRAL_ADVERSARIAL_03_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = NUISANCE_ALIGNMENT_SOURCE_REVISION

    def __init__(self, *, target_count: int = 3, embedding_dim: int = 64,
                 variety_count: int = VARIETY_COUNT) -> None:
        super().__init__(target_count=target_count, embedding_dim=embedding_dim,
                         variety_count=variety_count, strength=0.3)


class HyperLeafSpectralRoutedAdversarial(nn.Module):
    """Clean and adversarial spectral experts combined by a declared per-target route.

    The runner supplies the route tuple; this model never selects one.  The
    exploratory evidence predicts (clean, adversarial, clean) so that the
    variety-confounded target is read from the aligned expert while the other two
    keep the unconstrained expert.
    """

    __benchmark_model_key__ = SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = NUISANCE_ALIGNMENT_SOURCE_REVISION

    _VALID_ROUTES = frozenset(SPECTRAL_ROUTES)

    def __init__(self, *, target_count: int = 3, embedding_dim: int = 64,
                 variety_count: int = VARIETY_COUNT, strength: float = 1.0) -> None:
        super().__init__()
        if target_count != 3:
            raise ValueError("HyperLeaf task-25 requires exactly three targets")
        self.target_count = target_count
        self.clean = HyperLeafSpectralOnly(target_count=target_count, embedding_dim=embedding_dim)
        self.adversarial = HyperLeafSpectralVarietyAdversarial(
            target_count=target_count, embedding_dim=embedding_dim,
            variety_count=variety_count, strength=strength,
        )

    def branch_predictions(self, global_spectrum: Tensor) -> Mapping[str, Tensor]:
        return {
            SPECTRAL_CLEAN_BRANCH: self.clean(global_spectrum),
            SPECTRAL_ADVERSARIAL_BRANCH: self.adversarial(global_spectrum),
        }

    def forward(
        self,
        global_spectrum: Tensor,
        *,
        target_routes: Sequence[str] = (
            SPECTRAL_CLEAN_BRANCH, SPECTRAL_ADVERSARIAL_BRANCH, SPECTRAL_CLEAN_BRANCH,
        ),
        return_branches: bool = False,
    ) -> Tensor | tuple[Tensor, Mapping[str, Tensor]]:
        branches = self.branch_predictions(global_spectrum)
        routes = tuple(target_routes)
        if len(routes) != self.target_count or any(route not in self._VALID_ROUTES for route in routes):
            raise ValueError("one declared spectral route is required per target")
        prediction = torch.cat(
            tuple(branches[route][:, index:index + 1] for index, route in enumerate(routes)), dim=1
        )
        return (prediction, branches) if return_branches else prediction

    def variety_logits(self, global_spectrum: Tensor) -> Tensor:
        return self.adversarial.variety_logits(global_spectrum)


SPECTRUM_ONLY_INPUT_CONTRACT = InputContract(
    representation_key=SPECTRUM_ONLY_REPRESENTATION_KEY,
    modalities=("global_spectrum",),
    tensor_shapes={"global_spectrum": (128,)},
    dtypes={"global_spectrum": "float32"},
    channel_or_band_order={"global_spectrum": ("wavelength_ascending",)},
    preprocessing_key=SPECTRUM_ONLY_PREPROCESSING_KEY,
)


def _parameter_count(builder: type[nn.Module]) -> int:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = builder()
    return sum(parameter.numel() for parameter in model.parameters())


def _registration(model_key: str, display_name: str, builder: type[nn.Module]) -> ModelRegistration:
    return ModelRegistration(
        model_key=model_key,
        identity=ModelIdentity(
            model_key=model_key,
            display_name=display_name,
            class_fully_qualified_name=f"{builder.__module__}.{builder.__qualname__}",
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=_parameter_count(builder),
            source_revision=NUISANCE_ALIGNMENT_SOURCE_REVISION,
        ),
        input_contract=SPECTRUM_ONLY_INPUT_CONTRACT,
        builder=builder,
        implementation_kind=ImplementationKind.IN_REPO,
        source_revision=NUISANCE_ALIGNMENT_SOURCE_REVISION,
    )


def hyperleaf_nuisance_alignment_registrations() -> tuple[ModelRegistration, ...]:
    """Return the task-25 arms plus every baseline identity they are compared with."""

    registrations = (
        *hyperleaf_optimization_registrations(),
        _registration(SPECTRAL_ONLY_MODEL_KEY, "HyperLeaf task-25 spectral-only control",
                      HyperLeafSpectralOnly),
        _registration(SPECTRAL_ADVERSARIAL_03_MODEL_KEY,
                      "HyperLeaf task-25 spectral variety-adversarial lambda 0.3",
                      HyperLeafSpectralAdversarial03),
        _registration(SPECTRAL_ADVERSARIAL_10_MODEL_KEY,
                      "HyperLeaf task-25 spectral variety-adversarial lambda 1.0",
                      HyperLeafSpectralVarietyAdversarial),
        _registration(SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY,
                      "HyperLeaf task-25 target-routed spectral adversarial",
                      HyperLeafSpectralRoutedAdversarial),
    )
    return tuple(sorted(registrations, key=lambda item: item.model_key.encode("utf-8")))


def create_hyperleaf_nuisance_alignment_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    for registration in hyperleaf_nuisance_alignment_registrations():
        registry.register(registration)
    return registry


__all__ = [
    "GradientReversal",
    "HyperLeafSpectralAdversarial03",
    "HyperLeafSpectralOnly",
    "HyperLeafSpectralRoutedAdversarial",
    "HyperLeafSpectralVarietyAdversarial",
    "NUISANCE_ALIGNMENT_SOURCE_REVISION",
    "SPECTRAL_ADVERSARIAL_03_MODEL_KEY",
    "SPECTRAL_ADVERSARIAL_10_MODEL_KEY",
    "SPECTRAL_ADVERSARIAL_BRANCH",
    "SPECTRAL_CLEAN_BRANCH",
    "SPECTRAL_ONLY_MODEL_KEY",
    "SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY",
    "SPECTRUM_ONLY_INPUT_CONTRACT",
    "SPECTRUM_ONLY_PREPROCESSING_KEY",
    "SPECTRUM_ONLY_REPRESENTATION_KEY",
    "VARIETY_COUNT",
    "create_hyperleaf_nuisance_alignment_model_registry",
    "hyperleaf_nuisance_alignment_registrations",
]
