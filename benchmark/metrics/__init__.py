"""Prediction-first metrics and aggregate interfaces."""

from benchmark.metrics.aggregate import (
    AggregateBuildError,
    AggregateBuilder,
    PredictionMetricComputer,
    merge_prediction_artifacts,
)
from benchmark.metrics.engine import (
    MetricEngine,
    MetricInputError,
    binary_metrics,
    regression_metrics,
    remote_metrics,
)
from benchmark.metrics.statistics import (
    PATIENT_CONFIDENCE_LEVEL,
    PATIENT_RESAMPLES,
    PATIENT_RESAMPLING_UNIT,
    PATIENT_SIDEDNESS,
    paired_patient_compare,
    patient_bootstrap,
)
from benchmark.metrics.predictions import (
    PredictionArtifactReloadError,
    PredictionArtifactValidationError,
    PredictionArtifactWriter,
    PublishedPredictionArtifact,
    validate_prediction_records,
)

__all__ = [
    "AggregateBuildError",
    "AggregateBuilder",
    "MetricEngine",
    "MetricInputError",
    "PATIENT_CONFIDENCE_LEVEL",
    "PATIENT_RESAMPLES",
    "PATIENT_RESAMPLING_UNIT",
    "PATIENT_SIDEDNESS",
    "PredictionArtifactReloadError",
    "PredictionArtifactValidationError",
    "PredictionArtifactWriter",
    "PredictionMetricComputer",
    "PublishedPredictionArtifact",
    "binary_metrics",
    "paired_patient_compare",
    "patient_bootstrap",
    "regression_metrics",
    "remote_metrics",
    "merge_prediction_artifacts",
    "validate_prediction_records",
]
