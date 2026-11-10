"""HyperLeaf classic-regression adapters and matched-input execution glue.

The runner in this module never imports or calls legacy ``src`` runners.  It
builds scikit-learn estimators only through registry-managed adapters, fits all
learned transforms inside a protocol boundary, and delegates label access and
prediction publication to :class:`OuterEvaluationGate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.catalog.hyperleaf import (
    FULL_CUBE_KEY,
    GLOBAL_SPECTRUM_KEY,
    PCA_PSEUDO_RGB_KEY,
    VISIBLE_RGB_KEY,
    HyperLeafRepresentationBuilder,
)
from benchmark.engine.evaluation import OuterEvaluationGate
from benchmark.metrics.predictions import PublishedPredictionArtifact
from benchmark.models import ModelRegistration, ModelRegistry
from benchmark.schema import (
    FitBoundary,
    FitBoundaryScope,
    FittedArtifact,
    FrozenOuterCheckpoint,
    ImplementationKind,
    InputContract,
    ModelIdentity,
    OuterFold,
    OuterPrediction,
    ProtocolManifest,
    ResourceBudget,
    SelectionDecision,
    hash_object,
)


CLASSIC_PREPROCESSING_KEY = "hyperleaf.classic.fold-impute-band-target.v1"
CLASSIC_SOURCE_REVISION = "hyperleaf-sklearn-adapter.v1"


class HyperLeafClassicKind(str, Enum):
    """Supported classic multi-target regression implementations."""

    PLSR = "plsr"
    RANDOM_FOREST = "rf"


_REPRESENTATION_SLUGS = {
    VISIBLE_RGB_KEY: "visible-rgb",
    GLOBAL_SPECTRUM_KEY: "global-spectrum",
    PCA_PSEUDO_RGB_KEY: "pca-pseudo-rgb",
    FULL_CUBE_KEY: "full-cube",
}


class HyperLeafClassicError(RuntimeError):
    """A classic adapter request violates its frozen execution contract."""


@dataclass(frozen=True)
class MatchedRunContext:
    """Fields that must be identical for a candidate/control comparison.

    ``fold_manifest_hashes`` records each complete fold, rather than relying on
    fold names alone.  ``resource_budget`` is shared as an immutable schema
    object, and the metric configuration is represented by its canonical hash.
    """

    dataset_hash: str
    protocol_hash: str
    protocol_key: str
    fold_manifest_hashes: tuple[tuple[str, str], ...]
    seed_set: tuple[int, ...]
    metric_configuration_hash: str
    resource_budget: ResourceBudget
    representation_key: str

    @classmethod
    def from_protocol(
        cls,
        protocol: ProtocolManifest,
        *,
        metric_configuration: Mapping[str, Any],
        resource_budget: ResourceBudget,
        representation_key: str,
    ) -> "MatchedRunContext":
        """Freeze all comparison fields from one already-published protocol."""

        if not isinstance(protocol, ProtocolManifest):
            raise TypeError("protocol must be a ProtocolManifest")
        if representation_key not in _REPRESENTATION_SLUGS:
            raise ValueError(f"unsupported HyperLeaf Representation_Key: {representation_key}")
        return cls(
            dataset_hash=protocol.dataset_hash,
            protocol_hash=protocol.protocol_hash,
            protocol_key=protocol.protocol_key,
            fold_manifest_hashes=tuple(
                (fold.fold_id, hash_object(fold)) for fold in protocol.folds
            ),
            seed_set=protocol.seed_set,
            metric_configuration_hash=hash_object(dict(metric_configuration)),
            resource_budget=resource_budget,
            representation_key=representation_key,
        )

    @property
    def context_hash(self) -> str:
        """Return a canonical identity for the complete matched-run context."""

        return hash_object(
            {
                "dataset_hash": self.dataset_hash,
                "protocol_hash": self.protocol_hash,
                "protocol_key": self.protocol_key,
                "fold_manifest_hashes": self.fold_manifest_hashes,
                "seed_set": self.seed_set,
                "metric_configuration_hash": self.metric_configuration_hash,
                "resource_budget": self.resource_budget,
                "representation_key": self.representation_key,
            }
        )

    def require_matched(self, other: "MatchedRunContext") -> None:
        """Reject candidate/control drift before either result is compared."""

        if not isinstance(other, MatchedRunContext):
            raise TypeError("other must be a MatchedRunContext")
        fields = (
            "dataset_hash",
            "protocol_hash",
            "protocol_key",
            "fold_manifest_hashes",
            "seed_set",
            "metric_configuration_hash",
            "resource_budget",
            "representation_key",
        )
        mismatches = tuple(
            field for field in fields if getattr(self, field) != getattr(other, field)
        )
        if mismatches:
            raise HyperLeafClassicError(
                "candidate/control contexts differ: " + ", ".join(mismatches)
            )


class HyperLeafClassicEstimatorAdapter:
    """Registry-visible wrapper around one existing scikit-learn estimator."""

    __benchmark_implementation_kind__ = ImplementationKind.CLASSIC
    __benchmark_source_revision__ = CLASSIC_SOURCE_REVISION

    def __init__(
        self,
        *,
        model_key: str,
        kind: HyperLeafClassicKind,
        seed: int,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.__benchmark_model_key__ = model_key
        self.kind = HyperLeafClassicKind(kind)
        self.seed = seed
        self.parameters_config = dict(parameters or {})
        self._estimator: Any | None = None
        self._target_count: int | None = None

    def parameters(self):
        """Expose zero gradient parameters for registry identity validation."""

        return iter(())

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Fit the wrapped implementation on already-governed numeric arrays."""

        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("features and targets must be aligned two-dimensional arrays")
        if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1:
            raise ValueError("classic regression requires at least two samples and one feature")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("classic regression inputs must be finite after preprocessing")

        estimator_parameters = dict(self.parameters_config)
        if self.kind is HyperLeafClassicKind.PLSR:
            from sklearn.cross_decomposition import PLSRegression

            requested = estimator_parameters.pop("n_components", 8)
            if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
                raise ValueError("PLSR n_components must be a positive integer")
            maximum = min(x.shape[1], x.shape[0] - 1)
            estimator_parameters.setdefault("scale", False)
            estimator = PLSRegression(
                n_components=min(requested, maximum),
                **estimator_parameters,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor

            estimator_parameters.setdefault("n_estimators", 256)
            estimator_parameters.setdefault("random_state", self.seed)
            estimator_parameters.setdefault("n_jobs", 1)
            estimator = RandomForestRegressor(**estimator_parameters)
        estimator.fit(x, y)
        self._estimator = estimator
        self._target_count = y.shape[1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict a finite target matrix through the fitted adapter."""

        if self._estimator is None or self._target_count is None:
            raise HyperLeafClassicError("classic estimator has not been fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        prediction = np.asarray(self._estimator.predict(x), dtype=np.float64)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, self._target_count)
        if prediction.shape != (x.shape[0], self._target_count):
            raise HyperLeafClassicError("classic estimator returned an unexpected prediction shape")
        if not np.isfinite(prediction).all():
            raise HyperLeafClassicError("classic estimator returned non-finite predictions")
        return prediction


@dataclass(frozen=True)
class FittedHyperLeafClassicModel:
    """One in-memory fold model plus exact fold-scoped transform lineage."""

    model_key: str
    seed: int
    context: MatchedRunContext
    boundary: FitBoundary
    estimator: HyperLeafClassicEstimatorAdapter
    fitted_artifacts: tuple[FittedArtifact, ...]
    feature_shape: tuple[int, ...]
    representation_artifact: FittedArtifact | None
    imputation_artifact: FittedArtifact
    band_normalization_artifact: FittedArtifact
    target_normalization_artifact: FittedArtifact


class HyperLeafClassicRunner:
    """Fit classic controls and connect them to the unified outer gate.

    Construction and fitting are label-capability free outside permitted
    training IDs.  ``evaluate`` does not accept labels; only the supplied
    :class:`OuterEvaluationGate` may unlock labels and publish the canonical
    prediction artifact.
    """

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        representations: HyperLeafRepresentationBuilder,
        protocol: ProtocolManifest,
        context: MatchedRunContext,
    ) -> None:
        if not isinstance(registry, ModelRegistry):
            raise TypeError("registry must be a ModelRegistry")
        expected = MatchedRunContext.from_protocol(
            protocol,
            metric_configuration={"hash": context.metric_configuration_hash},
            resource_budget=context.resource_budget,
            representation_key=context.representation_key,
        )
        protocol_fields_match = (
            context.dataset_hash == expected.dataset_hash
            and context.protocol_hash == expected.protocol_hash
            and context.protocol_key == expected.protocol_key
            and context.fold_manifest_hashes == expected.fold_manifest_hashes
            and context.seed_set == expected.seed_set
        )
        if not protocol_fields_match:
            raise HyperLeafClassicError("matched-run context does not describe the protocol")
        self.registry = registry
        self.representations = representations
        self.protocol = protocol
        self.context = context

    def fit(
        self,
        model_key: str,
        *,
        seed: int,
        boundary: FitBoundary,
        estimator_parameters: Mapping[str, Any] | None = None,
    ) -> FittedHyperLeafClassicModel:
        """Fit one PLSR/RF control using only boundary-permitted samples."""

        if seed not in self.context.seed_set:
            raise HyperLeafClassicError("seed is outside the frozen Seed_Set")
        self._require_protocol_boundary(boundary)
        registration = self.registry.resolve(model_key)
        if registration.identity.implementation_kind is not ImplementationKind.CLASSIC:
            raise HyperLeafClassicError("model registration is not a Classic_Model")
        if registration.input_contract.representation_key != self.context.representation_key:
            raise HyperLeafClassicError("model and run context use different Representation_Key values")

        supplied_contract = registration.input_contract.model_copy(deep=True)
        estimator = self.registry.build(
            model_key,
            seed=seed,
            parameters=dict(estimator_parameters or {}),
        )
        self.registry.run_metadata(
            model_key,
            estimator,
            config_hash=self.context.context_hash,
            supplied_contract=supplied_contract,
        )

        sample_ids = boundary.allowed_fit_sample_ids
        target_artifact = self.representations.fit_target_normalization(
            sample_ids,
            boundary,
            dataset_hash=self.context.dataset_hash,
            protocol_hash=self.context.protocol_hash,
        )
        representation_artifact: FittedArtifact | None = None
        if self.context.representation_key == PCA_PSEUDO_RGB_KEY:
            representation_artifact = self.representations.fit_pca_pseudo_rgb(
                sample_ids,
                boundary,
                dataset_hash=self.context.dataset_hash,
                protocol_hash=self.context.protocol_hash,
            )

        raw_features, feature_shape = self._feature_matrix(
            sample_ids, representation_artifact, boundary
        )
        self._require_contract_shape(registration.input_contract, feature_shape)
        preprocessing = self.representations.preprocessing
        imputation = preprocessing.fit_imputation(
            raw_features,
            sample_ids,
            boundary,
            dataset_hash=self.context.dataset_hash,
            protocol_hash=self.context.protocol_hash,
        )
        imputed = preprocessing.apply(imputation, raw_features, boundary)
        band_normalization = preprocessing.fit_band_normalization(
            imputed,
            sample_ids,
            boundary,
            dataset_hash=self.context.dataset_hash,
            protocol_hash=self.context.protocol_hash,
        )
        normalized_features = preprocessing.apply(
            band_normalization, imputed, boundary
        )
        raw_targets = self.representations.adapter.training_targets(
            sample_ids, boundary, require_complete_boundary=True
        )
        normalized_targets = preprocessing.apply(
            target_artifact, raw_targets, boundary
        )
        estimator.fit(normalized_features, normalized_targets)

        artifacts = tuple(
            artifact
            for artifact in (
                representation_artifact,
                imputation,
                band_normalization,
                target_artifact,
            )
            if artifact is not None
        )
        return FittedHyperLeafClassicModel(
            model_key=model_key,
            seed=seed,
            context=self.context,
            boundary=boundary,
            estimator=estimator,
            fitted_artifacts=artifacts,
            feature_shape=feature_shape,
            representation_artifact=representation_artifact,
            imputation_artifact=imputation,
            band_normalization_artifact=band_normalization,
            target_normalization_artifact=target_artifact,
        )

    def predict(
        self,
        fitted: FittedHyperLeafClassicModel,
        sample_id: str,
    ) -> OuterPrediction:
        """Produce a label-free prediction from one fitted fold model."""

        if fitted.context != self.context:
            raise HyperLeafClassicError("fitted model belongs to a different run context")
        features, feature_shape = self._feature_matrix(
            (sample_id,), fitted.representation_artifact, fitted.boundary
        )
        if feature_shape != fitted.feature_shape:
            raise HyperLeafClassicError("prediction representation shape differs from training")
        preprocessing = self.representations.preprocessing
        imputed = preprocessing.apply(
            fitted.imputation_artifact, features, fitted.boundary
        )
        normalized = preprocessing.apply(
            fitted.band_normalization_artifact, imputed, fitted.boundary
        )
        normalized_prediction = fitted.estimator.predict(normalized)[0]
        target_state = fitted.target_normalization_artifact.state
        prediction = normalized_prediction * np.asarray(target_state["scale"]) + np.asarray(
            target_state["mean"]
        )
        return OuterPrediction(y_pred=tuple(float(value) for value in prediction))

    def evaluate(
        self,
        fitted: FittedHyperLeafClassicModel,
        *,
        gate: OuterEvaluationGate,
        outer_fold: OuterFold,
        checkpoint: FrozenOuterCheckpoint,
        selection_decision: SelectionDecision,
        artifact_id: str,
        run_id: str,
        relative_path: str | None = None,
    ) -> PublishedPredictionArtifact:
        """Delegate one outer evaluation and unified artifact write to the gate."""

        if gate.protocol.protocol_hash != self.context.protocol_hash:
            raise HyperLeafClassicError("evaluation gate uses a different Protocol_Hash")
        if outer_fold.fold_id != fitted.boundary.fold_id or checkpoint.fold_id != outer_fold.fold_id:
            raise HyperLeafClassicError("fold identity changed between fit and evaluation")
        if checkpoint.seed != fitted.seed:
            raise HyperLeafClassicError("checkpoint seed differs from fitted control seed")
        if relative_path is not None:
            path = PurePosixPath(relative_path.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
                raise HyperLeafClassicError("prediction path must be workspace-relative")

        lease = gate.reserve(self.context.protocol_key, 1)
        return gate.evaluate(
            lease,
            outer_fold=outer_fold,
            checkpoint=checkpoint,
            selection_decision=selection_decision,
            fitted_artifacts=fitted.fitted_artifacts,
            predictor=lambda received, sample_id: self.predict(fitted, sample_id),
            artifact_id=artifact_id,
            run_id=run_id,
            relative_path=relative_path,
        )

    def _feature_matrix(
        self,
        sample_ids: Sequence[str],
        representation_artifact: FittedArtifact | None,
        boundary: FitBoundary,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        arrays = tuple(
            self._representation(sample_id, representation_artifact, boundary)
            for sample_id in sample_ids
        )
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise HyperLeafClassicError("classic controls require one fixed representation shape")
        shape = arrays[0].shape
        matrix = np.stack([array.reshape(-1) for array in arrays]).astype(
            np.float64, copy=False
        )
        return matrix, shape

    def _representation(
        self,
        sample_id: str,
        representation_artifact: FittedArtifact | None,
        boundary: FitBoundary,
    ) -> np.ndarray:
        key = self.context.representation_key
        if key == VISIBLE_RGB_KEY:
            value = self.representations.visible_rgb(sample_id)
        elif key == GLOBAL_SPECTRUM_KEY:
            value = self.representations.global_spectrum(sample_id)
        elif key == PCA_PSEUDO_RGB_KEY:
            if representation_artifact is None:
                raise HyperLeafClassicError("PCA representation requires fold-fitted state")
            value = self.representations.pca_pseudo_rgb(
                sample_id, representation_artifact, boundary
            )
        elif key == FULL_CUBE_KEY:
            with self.representations.open_full_cube(sample_id) as reader:
                value = np.asarray(reader.memmap_view(), dtype=np.float32)
        else:
            raise HyperLeafClassicError(f"unsupported Representation_Key: {key}")
        array = np.asarray(value)
        if array.size == 0 or np.any(np.isinf(array)):
            raise HyperLeafClassicError("representation must be non-empty and contain no infinities")
        return array

    def _require_protocol_boundary(self, boundary: FitBoundary) -> None:
        try:
            outer = next(fold for fold in self.protocol.folds if fold.fold_id == boundary.fold_id)
        except StopIteration as exc:
            raise HyperLeafClassicError("fit boundary fold is absent from protocol") from exc
        if boundary.scope is FitBoundaryScope.OUTER_TRAIN:
            expected = outer.outer_train_ids
        else:
            try:
                inner = next(
                    fold for fold in outer.inner_folds if fold.fold_id == boundary.inner_fold_id
                )
            except StopIteration as exc:
                raise HyperLeafClassicError("inner fit boundary is absent from protocol") from exc
            expected = inner.train_ids
        if boundary.allowed_fit_sample_ids != expected:
            raise HyperLeafClassicError("fit boundary IDs differ from the frozen fold manifest")
        if boundary.outer_test_sample_ids != outer.outer_test_ids:
            raise HyperLeafClassicError("fit boundary outer-test IDs differ from the protocol")

    @staticmethod
    def _require_contract_shape(contract: InputContract, shape: tuple[int, ...]) -> None:
        modality = contract.modalities[0]
        if contract.tensor_shapes[modality] != shape:
            raise HyperLeafClassicError(
                "materialized representation shape differs from the Input_Contract"
            )


def classic_model_key(kind: HyperLeafClassicKind, representation_key: str) -> str:
    """Return the stable registry key for one classic matched-input control."""

    try:
        slug = _REPRESENTATION_SLUGS[representation_key]
    except KeyError as exc:
        raise ValueError(f"unsupported HyperLeaf Representation_Key: {representation_key}") from exc
    return f"hyperleaf.classic.{HyperLeafClassicKind(kind).value}.{slug}.v1"


def register_hyperleaf_classic_controls(
    registry: ModelRegistry,
    representations: HyperLeafRepresentationBuilder,
    *,
    representation_key: str,
    tensor_shape: Sequence[int] | None = None,
) -> tuple[ModelRegistration, ModelRegistration]:
    """Register PLSR and RF controls with the exact same Representation_Key."""

    descriptor = representations.descriptor(representation_key)
    shape = tuple(tensor_shape) if tensor_shape is not None else _default_shape(
        representations, representation_key
    )
    if not shape or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("tensor_shape must contain positive integers")
    modality = descriptor.modalities[0]
    contract = InputContract(
        representation_key=representation_key,
        modalities=(modality,),
        tensor_shapes={modality: shape},
        dtypes={modality: descriptor.dtype},
        channel_or_band_order={modality: descriptor.channel_or_band_order},
        preprocessing_key=CLASSIC_PREPROCESSING_KEY,
    )

    registrations = []
    for kind, display_name in (
        (HyperLeafClassicKind.PLSR, "HyperLeaf PLSR"),
        (HyperLeafClassicKind.RANDOM_FOREST, "HyperLeaf Random Forest"),
    ):
        model_key = classic_model_key(kind, representation_key)

        def build(
            *,
            seed: int,
            parameters: Mapping[str, Any] | None = None,
            _kind: HyperLeafClassicKind = kind,
            _model_key: str = model_key,
        ) -> HyperLeafClassicEstimatorAdapter:
            return HyperLeafClassicEstimatorAdapter(
                model_key=_model_key,
                kind=_kind,
                seed=seed,
                parameters=parameters,
            )

        identity = ModelIdentity(
            model_key=model_key,
            display_name=f"{display_name} ({representation_key})",
            class_fully_qualified_name=(
                f"{HyperLeafClassicEstimatorAdapter.__module__}."
                f"{HyperLeafClassicEstimatorAdapter.__qualname__}"
            ),
            implementation_kind=ImplementationKind.CLASSIC,
            parameter_count=0,
            source_revision=CLASSIC_SOURCE_REVISION,
        )
        registration = ModelRegistration(
            model_key=model_key,
            identity=identity,
            input_contract=contract,
            builder=build,
            implementation_kind=ImplementationKind.CLASSIC,
            source_revision=CLASSIC_SOURCE_REVISION,
        )
        registry.register(registration)
        registrations.append(registration)
    return registrations[0], registrations[1]


def _default_shape(
    representations: HyperLeafRepresentationBuilder,
    representation_key: str,
) -> tuple[int, ...]:
    if representation_key == GLOBAL_SPECTRUM_KEY:
        return (representations.spectrum_length,)
    if representation_key == VISIBLE_RGB_KEY:
        height, width = representations.visible_size
        return (3, height, width)
    raise ValueError(
        "tensor_shape is required for PCA pseudo-RGB and full-cube controls"
    )


__all__ = [
    "CLASSIC_PREPROCESSING_KEY",
    "CLASSIC_SOURCE_REVISION",
    "FittedHyperLeafClassicModel",
    "HyperLeafClassicError",
    "HyperLeafClassicEstimatorAdapter",
    "HyperLeafClassicKind",
    "HyperLeafClassicRunner",
    "MatchedRunContext",
    "classic_model_key",
    "register_hyperleaf_classic_controls",
]
