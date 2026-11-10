"""Fold-scoped fitting and application of learned preprocessing state."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema.base import (
    ClaimStatus,
    FittedArtifactKind,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from benchmark.schema.canonical import hash_object, normalize_id
from benchmark.schema.preprocessing import (
    FitBoundary,
    FitBoundaryScope,
    FitDataRole,
    FittedArtifact,
    canonical_sample_ids,
)
from benchmark.schema.records import ErrorRecord


class PreprocessingValidationError(ValueError):
    """Rejected fit or apply operation with a machine-readable error record."""

    def __init__(self, error_record: ErrorRecord, report: ValidationReport) -> None:
        self.error_record = error_record
        self.report = report
        self.claim_status = ClaimStatus.INVALID_OR_LEAKY
        super().__init__(
            f"{error_record.error_code}: preprocessing rejected at "
            f"{error_record.failed_stage} ({error_record.failing_field})"
        )


class PreprocessingManager:
    """Fit immutable preprocessing artifacts and apply them after boundary checks.

    The caller supplies a protocol-derived :class:`FitBoundary`. Boundary and
    source-role checks run before input values are converted or inspected. This
    keeps validation/outer targets unavailable to target normalization and
    restricts threshold and temperature fitting to inner out-of-fold evidence.
    """

    def __init__(self, *, error_journal: Optional[ErrorJournal] = None) -> None:
        self.error_journal = error_journal

    def fit(
        self,
        kind: FittedArtifactKind,
        values: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
        config: Optional[Mapping[str, Any]] = None,
        fit_data_role: FitDataRole = FitDataRole.TRAINING_VALUES,
    ) -> FittedArtifact:
        """Fit one supported transform and bind it to exact Fit_Sample_IDs."""

        if not isinstance(kind, FittedArtifactKind):
            try:
                kind = FittedArtifactKind(kind)
            except (TypeError, ValueError) as exc:
                raise TypeError("kind must be a supported FittedArtifactKind") from exc
        try:
            fit_ids = canonical_sample_ids(sample_ids, field_name="fit_sample_ids")
        except (TypeError, ValueError) as exc:
            self._reject(
                code="FIT_SAMPLE_IDS_INVALID",
                stage="preprocessing_fit_validation",
                field="fit_sample_ids",
                expected="a non-empty unique Sample_ID set",
                observed=str(exc),
                boundary=boundary,
                artifact_id=f"pending-{kind.value}",
            )

        self._require_fit_boundary(kind, fit_ids, boundary, fit_data_role)
        normalized_config = self._normalize_config(kind, config or {})

        try:
            state, output_shape, hash_values = self._fit_state(
                kind, values, normalized_config
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self._reject(
                code="PREPROCESSING_FIT_FAILED",
                stage="preprocessing_fit",
                field="values",
                expected=f"valid values for {kind.value}",
                observed=str(exc),
                boundary=boundary,
                artifact_id=f"pending-{kind.value}",
            )

        observed_sample_count = (
            len(hash_values[0])
            if kind in {FittedArtifactKind.THRESHOLD, FittedArtifactKind.CALIBRATION}
            else len(hash_values)
        )
        if observed_sample_count != len(fit_ids):
            self._reject(
                code="FIT_SAMPLE_COUNT_MISMATCH",
                stage="preprocessing_fit_validation",
                field="values",
                expected=len(fit_ids),
                observed=observed_sample_count,
                boundary=boundary,
                artifact_id=f"pending-{kind.value}",
            )

        config_dict = dict(normalized_config)
        state_dict = dict(state)
        config_hash = hash_object(config_dict)
        state_hash = hash_object(state_dict)
        input_hash = hash_object(
            {
                "fit_sample_ids": fit_ids,
                "fit_data_role": fit_data_role,
                "values": hash_values,
            }
        )
        identity_payload = {
            "kind": kind,
            "dataset_hash": dataset_hash,
            "protocol_hash": protocol_hash,
            "fold_id": boundary.fold_id,
            "boundary_scope": boundary.scope,
            "inner_fold_id": boundary.inner_fold_id,
            "fit_data_role": fit_data_role,
            "fit_sample_ids": fit_ids,
            "config_hash": config_hash,
            "input_hash": input_hash,
            "state_hash": state_hash,
        }
        artifact_id = f"fitted-{kind.value}-{hash_object(identity_payload)}"
        return FittedArtifact(
            artifact_id=artifact_id,
            kind=kind,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            fold_id=boundary.fold_id,
            boundary_scope=boundary.scope,
            inner_fold_id=boundary.inner_fold_id,
            fit_data_role=fit_data_role,
            fit_sample_ids=fit_ids,
            config=config_dict,
            config_hash=config_hash,
            input_hash=input_hash,
            state=state_dict,
            state_hash=state_hash,
            output_shape=output_shape,
        )

    def apply(
        self,
        artifact: FittedArtifact,
        values: Any,
        boundary: FitBoundary,
    ) -> np.ndarray:
        """Validate the fitted boundary, then apply immutable learned state."""

        self.require_boundary(artifact, boundary)
        try:
            return self._apply_state(artifact, values)
        except (TypeError, ValueError) as exc:
            self._reject(
                code="PREPROCESSING_APPLY_FAILED",
                stage="preprocessing_apply",
                field="values",
                expected=f"values compatible with {artifact.kind.value}",
                observed=str(exc),
                boundary=boundary,
                artifact_id=artifact.artifact_id,
            )

    def validate_boundary(
        self,
        artifact: FittedArtifact,
        boundary: FitBoundary,
    ) -> ValidationReport:
        """Report fold, scope, and Fit_Sample_ID boundary violations."""

        issues: list[ValidationIssue] = []
        if artifact.fold_id != boundary.fold_id:
            issues.append(
                self._issue(
                    "FITTED_ARTIFACT_FOLD_MISMATCH",
                    ("fold_id",),
                    boundary.fold_id,
                    artifact.fold_id,
                )
            )
        if artifact.boundary_scope is not boundary.scope:
            issues.append(
                self._issue(
                    "FITTED_ARTIFACT_SCOPE_MISMATCH",
                    ("boundary_scope",),
                    boundary.scope.value,
                    artifact.boundary_scope.value,
                )
            )
        if artifact.inner_fold_id != boundary.inner_fold_id:
            issues.append(
                self._issue(
                    "FITTED_ARTIFACT_INNER_FOLD_MISMATCH",
                    ("inner_fold_id",),
                    boundary.inner_fold_id,
                    artifact.inner_fold_id,
                )
            )

        fit_ids = set(artifact.fit_sample_ids)
        allowed_ids = set(boundary.allowed_fit_sample_ids)
        outer_test_ids = set(boundary.outer_test_sample_ids)
        outside = self._ordered(fit_ids - allowed_ids)
        if outside:
            issues.append(
                self._issue(
                    "FIT_SAMPLE_IDS_OUTSIDE_BOUNDARY",
                    ("fit_sample_ids",),
                    list(boundary.allowed_fit_sample_ids),
                    outside,
                )
            )
        leakage = self._ordered(fit_ids & outer_test_ids)
        if leakage:
            issues.append(
                self._issue(
                    "OUTER_TEST_FITTED_STATE_LEAKAGE",
                    ("fit_sample_ids",),
                    [],
                    leakage,
                )
            )
        return ValidationReport(issues=tuple(issues))

    def require_boundary(
        self,
        artifact: FittedArtifact,
        boundary: FitBoundary,
    ) -> None:
        """Raise before application when a fitted artifact crosses its boundary."""

        report = self.validate_boundary(artifact, boundary)
        if report.passes:
            return
        issue = report.errors[0]
        self._reject_issue(issue, artifact.artifact_id, boundary, report)

    def fit_target_normalization(
        self,
        targets: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
        epsilon: float = 1e-12,
    ) -> FittedArtifact:
        return self.fit(
            FittedArtifactKind.TARGET_NORMALIZATION,
            targets,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            config={"epsilon": epsilon},
        )

    def fit_pca(
        self,
        features: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
        n_components: int,
        whiten: bool = False,
        epsilon: float = 1e-12,
    ) -> FittedArtifact:
        return self.fit(
            FittedArtifactKind.PCA,
            features,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            config={
                "n_components": n_components,
                "whiten": whiten,
                "epsilon": epsilon,
            },
        )

    def fit_band_normalization(
        self,
        features: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
        epsilon: float = 1e-12,
    ) -> FittedArtifact:
        return self.fit(
            FittedArtifactKind.BAND_NORMALIZATION,
            features,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            config={"epsilon": epsilon},
        )

    def fit_imputation(
        self,
        features: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
        strategy: str = "mean",
        fill_value: float = 0.0,
    ) -> FittedArtifact:
        return self.fit(
            FittedArtifactKind.IMPUTATION,
            features,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            config={"strategy": strategy, "fill_value": fill_value},
        )

    def fit_class_weights(
        self,
        labels: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
    ) -> FittedArtifact:
        return self.fit(
            FittedArtifactKind.CLASS_WEIGHTS,
            labels,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            config={"strategy": "balanced"},
        )

    def fit_threshold(
        self,
        probabilities: Any,
        labels: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
        objective: str = "balanced_accuracy",
    ) -> FittedArtifact:
        paired = self._pair_without_numeric_read(probabilities, labels)
        return self.fit(
            FittedArtifactKind.THRESHOLD,
            paired,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            config={"objective": objective},
            fit_data_role=FitDataRole.INNER_OOF_PREDICTIONS,
        )

    def fit_calibration(
        self,
        scores: Any,
        labels: Any,
        sample_ids: Iterable[str],
        boundary: FitBoundary,
        *,
        dataset_hash: str,
        protocol_hash: str,
        input_kind: str = "probability",
        min_temperature: float = 0.05,
        max_temperature: float = 20.0,
        iterations: int = 80,
        epsilon: float = 1e-7,
    ) -> FittedArtifact:
        paired = self._pair_without_numeric_read(scores, labels)
        return self.fit(
            FittedArtifactKind.CALIBRATION,
            paired,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            config={
                "input_kind": input_kind,
                "min_temperature": min_temperature,
                "max_temperature": max_temperature,
                "iterations": iterations,
                "epsilon": epsilon,
            },
            fit_data_role=FitDataRole.INNER_OOF_PREDICTIONS,
        )

    @staticmethod
    def _pair_without_numeric_read(first: Any, second: Any) -> tuple[Any, Any]:
        """Keep two inputs opaque until fit-ID boundary validation has passed."""

        return first, second

    def _require_fit_boundary(
        self,
        kind: FittedArtifactKind,
        fit_ids: Tuple[str, ...],
        boundary: FitBoundary,
        fit_data_role: FitDataRole,
    ) -> None:
        allowed = set(boundary.allowed_fit_sample_ids)
        supplied = set(fit_ids)
        outside = self._ordered(supplied - allowed)
        leakage = self._ordered(supplied & set(boundary.outer_test_sample_ids))
        if outside:
            self._reject(
                code="FIT_SAMPLE_IDS_OUTSIDE_BOUNDARY",
                stage="preprocessing_fit_validation",
                field="fit_sample_ids",
                expected=list(boundary.allowed_fit_sample_ids),
                observed=outside,
                boundary=boundary,
                artifact_id=f"pending-{kind.value}",
                unexpected_ids=outside,
            )
        if leakage:
            self._reject(
                code="OUTER_TEST_FITTED_STATE_LEAKAGE",
                stage="preprocessing_fit_validation",
                field="fit_sample_ids",
                expected=[],
                observed=leakage,
                boundary=boundary,
                artifact_id=f"pending-{kind.value}",
                unexpected_ids=leakage,
            )
        if kind is FittedArtifactKind.TARGET_NORMALIZATION and supplied != allowed:
            missing = self._ordered(allowed - supplied)
            self._reject(
                code="TARGET_NORMALIZATION_INCOMPLETE_TRAINING_SET",
                stage="preprocessing_fit_validation",
                field="fit_sample_ids",
                expected=list(boundary.allowed_fit_sample_ids),
                observed=list(fit_ids),
                boundary=boundary,
                artifact_id=f"pending-{kind.value}",
                missing_ids=missing,
            )
        if kind in {FittedArtifactKind.THRESHOLD, FittedArtifactKind.CALIBRATION}:
            if fit_data_role is not FitDataRole.INNER_OOF_PREDICTIONS:
                self._reject(
                    code="INNER_OOF_EVIDENCE_REQUIRED",
                    stage="preprocessing_fit_validation",
                    field="fit_data_role",
                    expected=FitDataRole.INNER_OOF_PREDICTIONS.value,
                    observed=getattr(fit_data_role, "value", str(fit_data_role)),
                    boundary=boundary,
                    artifact_id=f"pending-{kind.value}",
                )

    def _normalize_config(
        self,
        kind: FittedArtifactKind,
        supplied: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        defaults: dict[FittedArtifactKind, dict[str, Any]] = {
            FittedArtifactKind.TARGET_NORMALIZATION: {"epsilon": 1e-12},
            FittedArtifactKind.PCA: {
                "n_components": 2,
                "whiten": False,
                "epsilon": 1e-12,
            },
            FittedArtifactKind.BAND_NORMALIZATION: {"epsilon": 1e-12},
            FittedArtifactKind.IMPUTATION: {
                "strategy": "mean",
                "fill_value": 0.0,
            },
            FittedArtifactKind.CLASS_WEIGHTS: {"strategy": "balanced"},
            FittedArtifactKind.THRESHOLD: {"objective": "balanced_accuracy"},
            FittedArtifactKind.CALIBRATION: {
                "input_kind": "probability",
                "min_temperature": 0.05,
                "max_temperature": 20.0,
                "iterations": 80,
                "epsilon": 1e-7,
            },
        }
        unknown = set(supplied) - set(defaults[kind])
        if unknown:
            raise ValueError(f"unknown {kind.value} config fields: {sorted(unknown)}")
        config = {**defaults[kind], **dict(supplied)}
        for key, value in config.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"config field {key!r} must be finite")
        if kind in {
            FittedArtifactKind.TARGET_NORMALIZATION,
            FittedArtifactKind.PCA,
            FittedArtifactKind.BAND_NORMALIZATION,
        } and (isinstance(config["epsilon"], bool) or config["epsilon"] <= 0):
            raise ValueError("epsilon must be positive")
        if kind is FittedArtifactKind.PCA:
            if (
                isinstance(config["n_components"], bool)
                or not isinstance(config["n_components"], int)
                or config["n_components"] <= 0
            ):
                raise ValueError("n_components must be a positive integer")
            if not isinstance(config["whiten"], bool):
                raise ValueError("whiten must be boolean")
        if kind is FittedArtifactKind.IMPUTATION:
            if config["strategy"] not in {"mean", "median", "constant"}:
                raise ValueError("imputation strategy must be mean, median, or constant")
            if isinstance(config["fill_value"], bool) or not isinstance(
                config["fill_value"], (int, float)
            ):
                raise ValueError("fill_value must be numeric")
        if kind is FittedArtifactKind.CLASS_WEIGHTS and config["strategy"] != "balanced":
            raise ValueError("class weight strategy must be balanced")
        if kind is FittedArtifactKind.THRESHOLD and config["objective"] not in {
            "accuracy",
            "balanced_accuracy",
        }:
            raise ValueError("threshold objective must be accuracy or balanced_accuracy")
        if kind is FittedArtifactKind.CALIBRATION:
            if config["input_kind"] not in {"probability", "logit"}:
                raise ValueError("calibration input_kind must be probability or logit")
            if not 0 < config["min_temperature"] < config["max_temperature"]:
                raise ValueError("temperature bounds must satisfy 0 < min < max")
            if (
                isinstance(config["iterations"], bool)
                or not isinstance(config["iterations"], int)
                or config["iterations"] <= 0
            ):
                raise ValueError("iterations must be a positive integer")
            if not 0 < config["epsilon"] < 0.5:
                raise ValueError("calibration epsilon must be between 0 and 0.5")
        return config

    def _fit_state(
        self,
        kind: FittedArtifactKind,
        values: Any,
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Tuple[int, ...], Any]:
        if kind is FittedArtifactKind.TARGET_NORMALIZATION:
            array, input_ndim = self._numeric_features(values)
            mean = array.mean(axis=0)
            scale = array.std(axis=0)
            scale = np.where(scale < config["epsilon"], 1.0, scale)
            state = self._location_scale_state(mean, scale, input_ndim)
            return state, (array.shape[1],), self._hashable_array(array)

        if kind is FittedArtifactKind.BAND_NORMALIZATION:
            array, input_ndim = self._numeric_features(values)
            mean = array.mean(axis=0)
            scale = array.std(axis=0)
            scale = np.where(scale < config["epsilon"], 1.0, scale)
            state = self._location_scale_state(mean, scale, input_ndim)
            return state, (array.shape[1],), self._hashable_array(array)

        if kind is FittedArtifactKind.PCA:
            array, _ = self._numeric_features(values, require_matrix=True)
            components_count = config["n_components"]
            if components_count > min(array.shape):
                raise ValueError("n_components exceeds min(sample_count, feature_count)")
            mean = array.mean(axis=0)
            centered = array - mean
            _, singular_values, right_vectors = np.linalg.svd(
                centered, full_matrices=False
            )
            components = right_vectors[:components_count].copy()
            for index, component in enumerate(components):
                pivot = int(np.argmax(np.abs(component)))
                if component[pivot] < 0:
                    components[index] *= -1.0
            denominator = max(array.shape[0] - 1, 1)
            explained_variance = (singular_values[:components_count] ** 2) / denominator
            whitening_scale = np.sqrt(explained_variance + config["epsilon"])
            state = {
                "mean": self._float_tuple(mean),
                "components": tuple(self._float_tuple(row) for row in components),
                "explained_variance": self._float_tuple(explained_variance),
                "whitening_scale": self._float_tuple(whitening_scale),
                "whiten": config["whiten"],
                "feature_count": int(array.shape[1]),
            }
            return state, (components_count,), self._hashable_array(array)

        if kind is FittedArtifactKind.IMPUTATION:
            array, input_ndim = self._numeric_features(values, allow_nan=True)
            strategy = config["strategy"]
            if strategy == "constant":
                fill = np.full(array.shape[1], float(config["fill_value"]))
            elif strategy == "mean":
                fill = np.nanmean(array, axis=0)
            else:
                fill = np.nanmedian(array, axis=0)
            if not np.all(np.isfinite(fill)):
                raise ValueError("mean/median imputation cannot fit an all-missing feature")
            state = {
                "fill_values": self._float_tuple(fill),
                "input_ndim": input_ndim,
                "feature_count": int(array.shape[1]),
                "strategy": strategy,
            }
            return state, (array.shape[1],), self._hashable_array(array, allow_nan=True)

        if kind is FittedArtifactKind.CLASS_WEIGHTS:
            labels = np.asarray(values)
            if labels.ndim != 1 or labels.size == 0:
                raise ValueError("class weight labels must be a non-empty 1-D array")
            native_labels = [self._native_scalar(value) for value in labels.tolist()]
            if any(
                isinstance(value, bool) or not isinstance(value, (str, int))
                for value in native_labels
            ):
                raise ValueError("class labels must be all strings or all integers")
            if len({type(value) for value in native_labels}) != 1:
                raise ValueError("class labels must use one consistent scalar type")
            classes = sorted(set(native_labels), key=lambda value: str(value).encode("utf-8"))
            counts = [native_labels.count(label) for label in classes]
            weights = [len(native_labels) / (len(classes) * count) for count in counts]
            state = {
                "classes": tuple(classes),
                "weights": tuple(float(weight) for weight in weights),
                "strategy": "balanced",
            }
            return state, (len(classes),), tuple(native_labels)

        first, second = self._paired_numeric_values(values)
        if first.shape != second.shape or first.ndim != 1 or first.size == 0:
            raise ValueError("prediction scores and labels must be equal non-empty 1-D arrays")
        labels = self._binary_labels(second)
        if kind is FittedArtifactKind.THRESHOLD:
            probabilities = self._probabilities(first)
            threshold = self._select_threshold(
                probabilities, labels, config["objective"]
            )
            state = {"threshold": threshold, "objective": config["objective"]}
            return state, (1,), (
                self._hashable_array(probabilities),
                self._hashable_array(labels),
            )

        scores = first.astype(np.float64, copy=False)
        if not np.all(np.isfinite(scores)):
            raise ValueError("calibration scores must be finite")
        if len(np.unique(labels)) != 2:
            raise ValueError("temperature calibration requires both binary classes")
        if config["input_kind"] == "probability":
            probabilities = self._probabilities(scores)
            epsilon = config["epsilon"]
            clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
            logits = np.log(clipped / (1.0 - clipped))
        else:
            logits = scores
        temperature = self._fit_temperature(logits, labels, config)
        state = {
            "temperature": temperature,
            "input_kind": config["input_kind"],
            "epsilon": config["epsilon"],
        }
        return state, (1,), (
            self._hashable_array(scores),
            self._hashable_array(labels),
        )

    def _apply_state(self, artifact: FittedArtifact, values: Any) -> np.ndarray:
        kind = artifact.kind
        state = artifact.state
        if kind in {
            FittedArtifactKind.TARGET_NORMALIZATION,
            FittedArtifactKind.BAND_NORMALIZATION,
        }:
            array, input_ndim = self._numeric_features(values)
            self._require_feature_count(array, state)
            result = (array - np.asarray(state["mean"])) / np.asarray(state["scale"])
            return result[:, 0] if input_ndim == 1 else result

        if kind is FittedArtifactKind.PCA:
            array, _ = self._numeric_features(values, require_matrix=True)
            self._require_feature_count(array, state)
            result = (array - np.asarray(state["mean"])) @ np.asarray(
                state["components"]
            ).T
            if state["whiten"]:
                result = result / np.asarray(state["whitening_scale"])
            return result

        if kind is FittedArtifactKind.IMPUTATION:
            array, input_ndim = self._numeric_features(values, allow_nan=True)
            self._require_feature_count(array, state)
            result = array.copy()
            missing_rows, missing_columns = np.where(np.isnan(result))
            result[missing_rows, missing_columns] = np.asarray(state["fill_values"])[
                missing_columns
            ]
            return result[:, 0] if input_ndim == 1 else result

        if kind is FittedArtifactKind.CLASS_WEIGHTS:
            labels = np.asarray(values)
            original_shape = labels.shape
            lookup = dict(zip(state["classes"], state["weights"]))
            native = [self._native_scalar(value) for value in labels.reshape(-1).tolist()]
            unknown = sorted(
                {value for value in native if value not in lookup}, key=str
            )
            if unknown:
                raise ValueError(f"labels contain classes absent from fitted state: {unknown}")
            return np.asarray([lookup[value] for value in native], dtype=np.float64).reshape(
                original_shape
            )

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            array = array.reshape(1)
        if not np.all(np.isfinite(array)):
            raise ValueError("prediction values must be finite")
        if kind is FittedArtifactKind.THRESHOLD:
            probabilities = self._probabilities(array)
            return (probabilities >= state["threshold"]).astype(np.int64)

        if state["input_kind"] == "probability":
            probabilities = self._probabilities(array)
            epsilon = state["epsilon"]
            clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
            logits = np.log(clipped / (1.0 - clipped))
        else:
            logits = array
        return self._sigmoid(logits / state["temperature"])

    @staticmethod
    def _numeric_features(
        values: Any,
        *,
        allow_nan: bool = False,
        require_matrix: bool = False,
    ) -> tuple[np.ndarray, int]:
        array = np.asarray(values, dtype=np.float64)
        input_ndim = array.ndim
        if require_matrix and array.ndim != 2:
            raise ValueError("values must be a two-dimensional matrix")
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("values must be a non-empty one- or two-dimensional array")
        if np.any(np.isinf(array)) or (not allow_nan and np.any(np.isnan(array))):
            raise ValueError("values contain non-finite entries")
        return array, input_ndim

    @staticmethod
    def _paired_numeric_values(values: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("threshold/calibration values must be a (scores, labels) pair")
        return (
            np.asarray(values[0], dtype=np.float64),
            np.asarray(values[1], dtype=np.float64),
        )

    @staticmethod
    def _binary_labels(values: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0.0, 1.0))):
            raise ValueError("binary labels must contain only 0 and 1")
        return values.astype(np.int64)

    @staticmethod
    def _probabilities(values: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("probabilities must be finite values in [0, 1]")
        return values.astype(np.float64, copy=False)

    @classmethod
    def _select_threshold(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        objective: str,
    ) -> float:
        if len(np.unique(labels)) != 2:
            raise ValueError("threshold fitting requires both binary classes")
        candidates = sorted(
            set(float(value) for value in probabilities.tolist()) | {0.0, 0.5, 1.0}
        )
        best_threshold = candidates[0]
        best_score = -math.inf
        for threshold in candidates:
            predicted = probabilities >= threshold
            if objective == "accuracy":
                score = float(np.mean(predicted == labels))
            else:
                sensitivity = float(np.mean(predicted[labels == 1] == 1))
                specificity = float(np.mean(predicted[labels == 0] == 0))
                score = (sensitivity + specificity) / 2.0
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return float(best_threshold)

    @classmethod
    def _fit_temperature(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        config: Mapping[str, Any],
    ) -> float:
        lower = math.log(config["min_temperature"])
        upper = math.log(config["max_temperature"])
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        left = upper - ratio * (upper - lower)
        right = lower + ratio * (upper - lower)

        def loss(log_temperature: float) -> float:
            scaled = logits / math.exp(log_temperature)
            terms = np.maximum(scaled, 0.0) - scaled * labels + np.log1p(
                np.exp(-np.abs(scaled))
            )
            return float(np.mean(terms))

        left_loss = loss(left)
        right_loss = loss(right)
        for _ in range(config["iterations"]):
            if left_loss <= right_loss:
                upper, right, right_loss = right, left, left_loss
                left = upper - ratio * (upper - lower)
                left_loss = loss(left)
            else:
                lower, left, left_loss = left, right, right_loss
                right = lower + ratio * (upper - lower)
                right_loss = loss(right)
        return float(math.exp((lower + upper) / 2.0))

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        result = np.empty_like(values, dtype=np.float64)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result

    @staticmethod
    def _location_scale_state(
        mean: np.ndarray,
        scale: np.ndarray,
        input_ndim: int,
    ) -> Mapping[str, Any]:
        return {
            "mean": PreprocessingManager._float_tuple(mean),
            "scale": PreprocessingManager._float_tuple(scale),
            "input_ndim": input_ndim,
            "feature_count": int(mean.size),
        }

    @staticmethod
    def _require_feature_count(array: np.ndarray, state: Mapping[str, Any]) -> None:
        if array.shape[1] != state["feature_count"]:
            raise ValueError(
                f"feature count mismatch: expected {state['feature_count']}, "
                f"observed {array.shape[1]}"
            )

    @staticmethod
    def _float_tuple(values: np.ndarray) -> Tuple[float, ...]:
        return tuple(float(value) for value in np.asarray(values).reshape(-1))

    @classmethod
    def _hashable_array(cls, values: np.ndarray, *, allow_nan: bool = False) -> Any:
        def convert(value: Any) -> Any:
            native = cls._native_scalar(value)
            if isinstance(native, float) and math.isnan(native):
                if allow_nan:
                    return None
                raise ValueError("NaN cannot enter a fitted artifact input hash")
            return native

        array = np.asarray(values)
        if array.ndim == 0:
            return convert(array.item())
        return tuple(
            tuple(convert(value) for value in row)
            for row in array.reshape(array.shape[0], -1).tolist()
        )

    @staticmethod
    def _native_scalar(value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    @staticmethod
    def _ordered(values: Iterable[str]) -> list[str]:
        return sorted(
            (normalize_id(value) for value in values),
            key=lambda value: value.encode("utf-8"),
        )

    @staticmethod
    def _issue(
        code: str,
        field_path: Tuple[str, ...],
        expected: Any,
        observed: Any,
    ) -> ValidationIssue:
        return ValidationIssue(
            code=code,
            severity=ValidationSeverity.ERROR,
            field_path=field_path,
            message=code,
            expected_value=expected,
            observed_value=observed,
        )

    def _reject_issue(
        self,
        issue: ValidationIssue,
        artifact_id: str,
        boundary: FitBoundary,
        report: ValidationReport,
    ) -> None:
        observed = issue.observed_value
        unexpected = tuple(observed) if isinstance(observed, list) else ()
        record = self._error_record(
            code=issue.code,
            stage="preprocessing_apply_validation",
            field=".".join(issue.field_path),
            expected=issue.expected_value,
            observed=observed,
            boundary=boundary,
            artifact_id=artifact_id,
            unexpected_ids=unexpected,
        )
        if self.error_journal is not None:
            self.error_journal.append(record)
        raise PreprocessingValidationError(record, report)

    def _reject(
        self,
        *,
        code: str,
        stage: str,
        field: str,
        expected: Any,
        observed: Any,
        boundary: FitBoundary,
        artifact_id: str,
        missing_ids: Sequence[str] = (),
        unexpected_ids: Sequence[str] = (),
    ) -> None:
        issue = self._issue(code, (field,), expected, observed)
        report = ValidationReport(issues=(issue,))
        record = self._error_record(
            code=code,
            stage=stage,
            field=field,
            expected=expected,
            observed=observed,
            boundary=boundary,
            artifact_id=artifact_id,
            missing_ids=missing_ids,
            unexpected_ids=unexpected_ids,
        )
        if self.error_journal is not None:
            self.error_journal.append(record)
        raise PreprocessingValidationError(record, report)

    @staticmethod
    def _error_record(
        *,
        code: str,
        stage: str,
        field: str,
        expected: Any,
        observed: Any,
        boundary: FitBoundary,
        artifact_id: str,
        missing_ids: Sequence[str] = (),
        unexpected_ids: Sequence[str] = (),
    ) -> ErrorRecord:
        affected = {
            "fitted_artifact_id": artifact_id,
            "fold_id": boundary.fold_id,
        }
        if boundary.inner_fold_id is not None:
            affected["inner_fold_id"] = boundary.inner_fold_id
        return ErrorRecord(
            error_code=code,
            failed_stage=stage,
            affected_identifiers=affected,
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            missing_ids=tuple(missing_ids),
            unexpected_ids=tuple(unexpected_ids),
            invariant_checks={
                "checkpoint_unchanged": True,
                "prediction_artifact_unchanged": True,
                "completed_artifacts_unchanged": True,
            },
        )


__all__ = ["PreprocessingManager", "PreprocessingValidationError"]
