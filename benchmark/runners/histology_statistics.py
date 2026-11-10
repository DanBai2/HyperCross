"""Leakage-safe Histology threshold, calibration, and patient statistics wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from benchmark.metrics import binary_metrics, paired_patient_compare, patient_bootstrap
from benchmark.schema import (
    ConfidenceInterval,
    FitBoundary,
    FitBoundaryScope,
    FitDataRole,
    FittedArtifact,
    FittedArtifactKind,
    MetricSchema,
    MetricSet,
    OuterFold,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    StatisticalComparison,
    TaskKind,
)
from benchmark.transforms import PreprocessingManager


HISTOLOGY_PRIMARY_THRESHOLD = 0.5
HISTOLOGY_PRIMARY_RESULT_LABEL = "primary.fixed-threshold-0.5"
HISTOLOGY_FITTED_THRESHOLD_RESULT_LABEL = "secondary.inner-oof-fitted-threshold"
HISTOLOGY_TEMPERATURE_RESULT_LABEL = "secondary.inner-oof-fitted-temperature"


class HistologyStatisticsWiringError(ValueError):
    """Histology statistical wiring violated its frozen evidence boundary."""


@dataclass(frozen=True)
class HistologyThresholdCalibration:
    """Inner-OOF fitted secondary artifacts, separate from the primary policy."""

    threshold: FittedArtifact
    temperature: FittedArtifact
    threshold_result_label: str = HISTOLOGY_FITTED_THRESHOLD_RESULT_LABEL
    temperature_result_label: str = HISTOLOGY_TEMPERATURE_RESULT_LABEL

    def __post_init__(self) -> None:
        if self.threshold.kind is not FittedArtifactKind.THRESHOLD:
            raise HistologyStatisticsWiringError("threshold artifact has the wrong kind")
        if self.temperature.kind is not FittedArtifactKind.CALIBRATION:
            raise HistologyStatisticsWiringError("temperature artifact has the wrong kind")
        for artifact in (self.threshold, self.temperature):
            if artifact.fit_data_role is not FitDataRole.INNER_OOF_PREDICTIONS:
                raise HistologyStatisticsWiringError(
                    "threshold and temperature must be fitted from inner OOF predictions"
                )
        identity = (
            self.threshold.dataset_hash,
            self.threshold.protocol_hash,
            self.threshold.fold_id,
            self.threshold.fit_sample_ids,
        )
        temperature_identity = (
            self.temperature.dataset_hash,
            self.temperature.protocol_hash,
            self.temperature.fold_id,
            self.temperature.fit_sample_ids,
        )
        if identity != temperature_identity:
            raise HistologyStatisticsWiringError(
                "threshold and temperature artifacts must share one evidence identity"
            )


@dataclass(frozen=True)
class HistologyMetricVariant:
    """One explicitly labeled prediction/metric projection."""

    result_label: str
    prediction_artifact: PredictionArtifact
    metric_schema: MetricSchema
    metrics: MetricSet
    primary: bool = False

    def __post_init__(self) -> None:
        if self.primary and (
            self.result_label != HISTOLOGY_PRIMARY_RESULT_LABEL
            or self.metric_schema.decision_threshold != HISTOLOGY_PRIMARY_THRESHOLD
        ):
            raise HistologyStatisticsWiringError(
                "primary Histology metrics must use the immutable threshold 0.5 label"
            )
        if self.metrics.metric_schema != self.metric_schema:
            raise HistologyStatisticsWiringError("metric variant schema does not match metrics")


@dataclass(frozen=True)
class HistologyMetricVariants:
    """Primary and separately labeled inner-OOF-derived metric projections."""

    primary: HistologyMetricVariant
    fitted_threshold: HistologyMetricVariant
    fitted_temperature: HistologyMetricVariant


@dataclass(frozen=True)
class LabeledPatientStatistic:
    """Patient statistic retaining the metric variant label used to compute it."""

    result_label: str
    statistic: ConfidenceInterval | StatisticalComparison


class HistologyStatisticsWiring:
    """Connect strict Histology outputs to fitted secondary and patient statistics."""

    def __init__(self, preprocessing_manager: PreprocessingManager | None = None) -> None:
        self.preprocessing_manager = preprocessing_manager or PreprocessingManager()

    @staticmethod
    def primary_metric_schema(*, ece_bins: int = 10) -> MetricSchema:
        """Return the fixed primary metric schema; callers cannot override 0.5."""

        return MetricSchema(
            task_kind=TaskKind.BINARY_CLASSIFICATION,
            numeric_tolerance=1e-12,
            decision_threshold=HISTOLOGY_PRIMARY_THRESHOLD,
            ece_bins=ece_bins,
        )

    def fit_inner_oof(
        self,
        outer_fold: OuterFold,
        inner_oof_predictions: PredictionArtifact,
        *,
        dataset_hash: str,
        protocol_hash: str,
        threshold_objective: str = "balanced_accuracy",
    ) -> HistologyThresholdCalibration:
        """Fit threshold and temperature only from complete inner-patient OOF evidence."""

        expected_ids = set(outer_fold.outer_train_ids)
        observed_ids = {record.key.sample_id for record in inner_oof_predictions.records}
        if (
            not inner_oof_predictions.complete
            or set(inner_oof_predictions.expected_sample_ids) != expected_ids
            or observed_ids != expected_ids
        ):
            raise HistologyStatisticsWiringError(
                "fitting requires exact complete inner OOF coverage of outer-train samples"
            )
        forbidden = observed_ids.intersection(outer_fold.outer_test_ids)
        if forbidden:
            raise HistologyStatisticsWiringError(
                "inner OOF fitting evidence contains outer-test Sample_IDs"
            )

        # Validate labels and probabilities before handing the same immutable
        # inner-OOF records to both fitters.
        binary_metrics(inner_oof_predictions, self.primary_metric_schema())
        sample_ids = tuple(record.key.sample_id for record in inner_oof_predictions.records)
        probabilities = tuple(self._scalar_probability(record) for record in inner_oof_predictions.records)
        labels = tuple(record.y_true for record in inner_oof_predictions.records)
        boundary = self._outer_train_boundary(outer_fold)
        threshold = self.preprocessing_manager.fit_threshold(
            probabilities,
            labels,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            objective=threshold_objective,
        )
        temperature = self.preprocessing_manager.fit_calibration(
            probabilities,
            labels,
            sample_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            input_kind="probability",
        )
        return HistologyThresholdCalibration(
            threshold=threshold,
            temperature=temperature,
        )

    def evaluate_variants(
        self,
        outer_fold: OuterFold,
        outer_predictions: PredictionArtifact,
        fitted: HistologyThresholdCalibration,
        *,
        ece_bins: int = 10,
    ) -> HistologyMetricVariants:
        """Evaluate immutable primary 0.5 and separately labeled secondary variants."""

        self._require_outer_predictions(outer_fold, outer_predictions)
        self._require_fitted_identity(outer_fold, fitted)
        primary_schema = self.primary_metric_schema(ece_bins=ece_bins)
        primary = self._variant(
            HISTOLOGY_PRIMARY_RESULT_LABEL,
            outer_predictions,
            primary_schema,
            primary=True,
        )

        fitted_threshold_schema = primary_schema.model_copy(
            update={"decision_threshold": float(fitted.threshold.state["threshold"])}
        )
        fitted_threshold = self._variant(
            fitted.threshold_result_label,
            outer_predictions,
            fitted_threshold_schema,
        )

        calibrated_predictions = self._calibrated_predictions(
            outer_fold, outer_predictions, fitted.temperature
        )
        fitted_temperature = self._variant(
            fitted.temperature_result_label,
            calibrated_predictions,
            primary_schema,
        )
        return HistologyMetricVariants(
            primary=primary,
            fitted_threshold=fitted_threshold,
            fitted_temperature=fitted_temperature,
        )

    def patient_bootstrap(
        self,
        variant: HistologyMetricVariant,
        metric_name: str,
        *,
        group_by_sample_id: Mapping[str, str],
        seed: int,
    ) -> LabeledPatientStatistic:
        """Run the fixed 10,000-draw patient bootstrap for one labeled variant."""

        interval = patient_bootstrap(
            variant.prediction_artifact,
            metric_name,
            group_by_sample_id=group_by_sample_id,
            metric_schema=variant.metric_schema,
            seed=seed,
        )
        return LabeledPatientStatistic(variant.result_label, interval)

    def paired_patient_compare(
        self,
        candidate: HistologyMetricVariant,
        control: HistologyMetricVariant,
        metric_name: str,
        *,
        candidate_group_by_sample_id: Mapping[str, str],
        control_group_by_sample_id: Mapping[str, str],
        seed: int,
    ) -> LabeledPatientStatistic:
        """Run a paired patient comparison for like-labeled metric variants."""

        if candidate.result_label != control.result_label:
            raise HistologyStatisticsWiringError(
                "paired comparison requires the same threshold/calibration result label"
            )
        if candidate.metric_schema != control.metric_schema:
            raise HistologyStatisticsWiringError(
                "paired comparison requires identical metric schemas"
            )
        comparison = paired_patient_compare(
            candidate.prediction_artifact,
            control.prediction_artifact,
            metric_name,
            candidate_group_by_sample_id=candidate_group_by_sample_id,
            control_group_by_sample_id=control_group_by_sample_id,
            metric_schema=candidate.metric_schema,
            seed=seed,
        )
        return LabeledPatientStatistic(candidate.result_label, comparison)

    def _calibrated_predictions(
        self,
        outer_fold: OuterFold,
        source: PredictionArtifact,
        temperature: FittedArtifact,
    ) -> PredictionArtifact:
        probabilities = tuple(self._scalar_probability(record) for record in source.records)
        calibrated = self.preprocessing_manager.apply(
            temperature,
            probabilities,
            self._outer_train_boundary(outer_fold),
        )
        records = tuple(
            PredictionRecord(
                key=PredictionKey(
                    sample_id=record.key.sample_id,
                    fold_id=record.key.fold_id,
                    seed=record.key.seed,
                    inference_branch=(
                        f"{record.key.inference_branch}.inner-oof-temperature"
                    ),
                ),
                target_name=record.target_name,
                y_true=record.y_true,
                y_pred=(1 if probability >= HISTOLOGY_PRIMARY_THRESHOLD else 0),
                probability=float(probability),
            )
            for record, probability in zip(source.records, calibrated)
        )
        return PredictionArtifact.create(
            artifact_id=f"{source.artifact_id}-inner-oof-temperature",
            run_id=source.run_id,
            records=records,
            expected_sample_ids=source.expected_sample_ids,
            complete=True,
        )

    @staticmethod
    def _variant(
        result_label: str,
        artifact: PredictionArtifact,
        schema: MetricSchema,
        *,
        primary: bool = False,
    ) -> HistologyMetricVariant:
        return HistologyMetricVariant(
            result_label=result_label,
            prediction_artifact=artifact,
            metric_schema=schema,
            metrics=binary_metrics(artifact, schema),
            primary=primary,
        )

    @staticmethod
    def _outer_train_boundary(outer_fold: OuterFold) -> FitBoundary:
        return FitBoundary(
            fold_id=outer_fold.fold_id,
            scope=FitBoundaryScope.OUTER_TRAIN,
            allowed_fit_sample_ids=outer_fold.outer_train_ids,
            outer_test_sample_ids=outer_fold.outer_test_ids,
        )

    @staticmethod
    def _require_fitted_identity(
        outer_fold: OuterFold,
        fitted: HistologyThresholdCalibration,
    ) -> None:
        expected = set(outer_fold.outer_train_ids)
        for artifact in (fitted.threshold, fitted.temperature):
            if artifact.fold_id != outer_fold.fold_id or set(artifact.fit_sample_ids) != expected:
                raise HistologyStatisticsWiringError(
                    "fitted threshold/calibration identity differs from the outer fold"
                )
            if set(artifact.fit_sample_ids).intersection(outer_fold.outer_test_ids):
                raise HistologyStatisticsWiringError(
                    "fitted threshold/calibration contains outer-test Sample_IDs"
                )

    @staticmethod
    def _require_outer_predictions(
        outer_fold: OuterFold,
        artifact: PredictionArtifact,
    ) -> None:
        expected = set(outer_fold.outer_test_ids)
        observed = {record.key.sample_id for record in artifact.records}
        if not artifact.complete or observed != expected:
            raise HistologyStatisticsWiringError(
                "evaluation requires exact complete outer-test predictions"
            )
        if any(record.key.fold_id != outer_fold.fold_id for record in artifact.records):
            raise HistologyStatisticsWiringError(
                "outer prediction Fold_ID differs from the requested outer fold"
            )

    @staticmethod
    def _scalar_probability(record: PredictionRecord) -> float:
        if isinstance(record.probability, tuple) or record.probability is None:
            raise HistologyStatisticsWiringError(
                "Histology wiring requires one scalar positive-class probability"
            )
        return float(record.probability)


__all__ = [
    "HISTOLOGY_FITTED_THRESHOLD_RESULT_LABEL",
    "HISTOLOGY_PRIMARY_RESULT_LABEL",
    "HISTOLOGY_PRIMARY_THRESHOLD",
    "HISTOLOGY_TEMPERATURE_RESULT_LABEL",
    "HistologyMetricVariant",
    "HistologyMetricVariants",
    "HistologyStatisticsWiring",
    "HistologyStatisticsWiringError",
    "HistologyThresholdCalibration",
    "LabeledPatientStatistic",
]
