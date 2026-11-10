"""Implementable HyperLeaf models predeclared for task 24.3 optimization.

The models are data-agnostic and never select a route. Task 24.4 must supply an
already frozen target route from inner-variety evidence.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from benchmark.models.hyperleaf_candidates import (
    AVERAGE_BRANCH,
    PCA_BRANCH,
    RGB_BRANCH,
    RGB_PCA_INPUT_CONTRACT,
    HyperLeafRgbPcaConsistency,
    hyperleaf_candidate_registrations,
)
from benchmark.models.registry import ModelRegistration, ModelRegistry
from benchmark.schema import ImplementationKind, ModelIdentity


OPTIMIZATION_SOURCE_REVISION = "task-24.3.v1"
LEARNED_GATED_BRANCH = "learned-gated-fusion"
LEARNED_GATED_MODEL_KEY = "hyperleaf.optimization.learned-gated-fusion.v1"
HETEROSCEDASTIC_MODEL_KEY = "hyperleaf.optimization.heteroscedastic-rgb-pca.v1"


class HyperLeafLearnedGatedFusion(nn.Module):
    """Fuse independently predicted RGB/PCA targets with learned bounded gates."""

    __benchmark_model_key__ = LEARNED_GATED_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = OPTIMIZATION_SOURCE_REVISION

    _VALID_ROUTES = frozenset(
        {RGB_BRANCH, PCA_BRANCH, AVERAGE_BRANCH, LEARNED_GATED_BRANCH}
    )

    def __init__(
        self,
        *,
        target_count: int = 3,
        embedding_dim: int = 64,
        gate_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if target_count != 3:
            raise ValueError("HyperLeaf optimization requires exactly three targets")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        self.base = HyperLeafRgbPcaConsistency(
            target_count=target_count, embedding_dim=embedding_dim
        )
        self.gate = nn.Sequential(
            nn.Linear(target_count * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, target_count),
            nn.Sigmoid(),
        )
        self.target_count = target_count

    def branch_predictions(
        self,
        visible_rgb: Tensor,
        pca_pseudo_rgb: Tensor,
        global_spectrum: Tensor,
    ) -> Mapping[str, Tensor]:
        branches = dict(
            self.base.branch_predictions(visible_rgb, pca_pseudo_rgb, global_spectrum)
        )
        gate = self.gate(torch.cat((branches[RGB_BRANCH], branches[PCA_BRANCH]), dim=1))
        branches[LEARNED_GATED_BRANCH] = (
            gate * branches[RGB_BRANCH] + (1.0 - gate) * branches[PCA_BRANCH]
        )
        return branches

    def forward(
        self,
        visible_rgb: Tensor,
        pca_pseudo_rgb: Tensor,
        global_spectrum: Tensor,
        *,
        target_routes: Sequence[str] = (
            PCA_BRANCH,
            LEARNED_GATED_BRANCH,
            PCA_BRANCH,
        ),
        return_branches: bool = False,
    ) -> Tensor | tuple[Tensor, Mapping[str, Tensor]]:
        routes = tuple(target_routes)
        if len(routes) != self.target_count or any(
            route not in self._VALID_ROUTES for route in routes
        ):
            raise ValueError("one declared RGB/PCA/gated route is required per target")
        branches = self.branch_predictions(
            visible_rgb, pca_pseudo_rgb, global_spectrum
        )
        prediction = torch.cat(
            tuple(branches[route][:, index : index + 1] for index, route in enumerate(routes)),
            dim=1,
        )
        return (prediction, branches) if return_branches else prediction


class HyperLeafHeteroscedasticRgbPca(nn.Module):
    """Return target-routed means and bounded input-dependent log variances."""

    __benchmark_model_key__ = HETEROSCEDASTIC_MODEL_KEY
    __benchmark_implementation_kind__ = ImplementationKind.IN_REPO
    __benchmark_source_revision__ = OPTIMIZATION_SOURCE_REVISION

    def __init__(self, *, target_count: int = 3, embedding_dim: int = 64) -> None:
        super().__init__()
        if target_count != 3:
            raise ValueError("HyperLeaf optimization requires exactly three targets")
        self.mean_model = HyperLeafRgbPcaConsistency(
            target_count=target_count, embedding_dim=embedding_dim
        )
        self.log_variance_head = nn.Sequential(
            nn.Linear(target_count * 2, 16),
            nn.GELU(),
            nn.Linear(16, target_count),
        )
        self.target_count = target_count

    def forward(
        self,
        visible_rgb: Tensor,
        pca_pseudo_rgb: Tensor,
        global_spectrum: Tensor,
        *,
        target_routes: Sequence[str] = (PCA_BRANCH, AVERAGE_BRANCH, PCA_BRANCH),
    ) -> tuple[Tensor, Tensor]:
        branches = self.mean_model.branch_predictions(
            visible_rgb, pca_pseudo_rgb, global_spectrum
        )
        routes = tuple(target_routes)
        if len(routes) != self.target_count or any(
            route not in {RGB_BRANCH, PCA_BRANCH, AVERAGE_BRANCH} for route in routes
        ):
            raise ValueError("heteroscedastic means require one fixed route per target")
        mean = torch.cat(
            tuple(branches[route][:, index : index + 1] for index, route in enumerate(routes)),
            dim=1,
        )
        log_variance = self.log_variance_head(
            torch.cat((branches[RGB_BRANCH], branches[PCA_BRANCH]), dim=1)
        ).clamp(-8.0, 8.0)
        return mean, log_variance


def _parameter_count(builder: type[nn.Module]) -> int:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = builder()
    return sum(parameter.numel() for parameter in model.parameters())


def _optimization_registration(
    model_key: str,
    display_name: str,
    builder: type[nn.Module],
) -> ModelRegistration:
    return ModelRegistration(
        model_key=model_key,
        identity=ModelIdentity(
            model_key=model_key,
            display_name=display_name,
            class_fully_qualified_name=f"{builder.__module__}.{builder.__qualname__}",
            implementation_kind=ImplementationKind.IN_REPO,
            parameter_count=_parameter_count(builder),
            source_revision=OPTIMIZATION_SOURCE_REVISION,
        ),
        input_contract=RGB_PCA_INPUT_CONTRACT,
        builder=builder,
        implementation_kind=ImplementationKind.IN_REPO,
        source_revision=OPTIMIZATION_SOURCE_REVISION,
    )


def hyperleaf_optimization_registrations() -> tuple[ModelRegistration, ...]:
    """Return baseline and new model identities needed by the frozen space."""

    registrations = (
        *hyperleaf_candidate_registrations(),
        _optimization_registration(
            LEARNED_GATED_MODEL_KEY,
            "HyperLeaf learned RGB/PCA gated-fusion optimization",
            HyperLeafLearnedGatedFusion,
        ),
        _optimization_registration(
            HETEROSCEDASTIC_MODEL_KEY,
            "HyperLeaf heteroscedastic RGB/PCA optimization",
            HyperLeafHeteroscedasticRgbPca,
        ),
    )
    return tuple(sorted(registrations, key=lambda item: item.model_key.encode("utf-8")))


def create_hyperleaf_optimization_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    for registration in hyperleaf_optimization_registrations():
        registry.register(registration)
    return registry


__all__ = [
    "HETEROSCEDASTIC_MODEL_KEY",
    "LEARNED_GATED_BRANCH",
    "LEARNED_GATED_MODEL_KEY",
    "OPTIMIZATION_SOURCE_REVISION",
    "HyperLeafHeteroscedasticRgbPca",
    "HyperLeafLearnedGatedFusion",
    "create_hyperleaf_optimization_model_registry",
    "hyperleaf_optimization_registrations",
]
