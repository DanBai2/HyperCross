"""Deterministic metrics computed only from complete prediction artifacts."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence, Tuple

from benchmark.metrics.predictions import validate_prediction_records
from benchmark.schema.base import TaskKind
from benchmark.schema.canonical import canonical_serialize, hash_object
from benchmark.schema.metrics import (
    MetricSchema,
    MetricSet,
    MetricValue,
    UNDEFINED_SINGLE_CLASS_AUROC,
    UNDEFINED_ZERO_VARIANCE_R2,
)
from benchmark.schema.predictions import PredictionArtifact, PredictionRecord


class MetricInputError(ValueError):
    """Prediction content or metric configuration is invalid for computation."""


def _label_identity(value: Any) -> bytes:
    return canonical_serialize(value)


def _labels_equal(left: Any, right: Any) -> bool:
    return _label_identity(left) == _label_identity(right)


def _require_artifact(artifact: PredictionArtifact) -> Tuple[PredictionRecord, ...]:
    if not isinstance(artifact, PredictionArtifact):
        raise TypeError("metrics require a validated PredictionArtifact")
    if not artifact.complete:
        raise MetricInputError("metrics require a complete PredictionArtifact")
    if artifact.content_hash != hash_object(artifact.hash_payload()):
        raise MetricInputError("prediction artifact content_hash is invalid")
    report = validate_prediction_records(artifact.records, artifact.expected_sample_ids)
    if not report.passes:
        codes = ", ".join(issue.code for issue in report.errors)
        raise MetricInputError(f"prediction artifact validation failed: {codes}")
    return artifact.records


def _number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricInputError(f"{field_name} must contain numeric values")
    result = float(value)
    if not math.isfinite(result):
        raise MetricInputError(f"{field_name} must contain finite values")
    return result


def _defined(value: float) -> MetricValue:
    return MetricValue(value=float(value))


def _undefined(reason: str) -> MetricValue:
    return MetricValue(reason=reason)


def _r2(y_true: Sequence[float], y_pred: Sequence[float]) -> MetricValue:
    mean_true = math.fsum(y_true) / len(y_true)
    total = math.fsum((value - mean_true) ** 2 for value in y_true)
    if total == 0.0:
        return _undefined(UNDEFINED_ZERO_VARIANCE_R2)
    residual = math.fsum(
        (truth - prediction) ** 2 for truth, prediction in zip(y_true, y_pred)
    )
    return _defined(1.0 - residual / total)


def _mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.fsum(
        abs(truth - prediction) for truth, prediction in zip(y_true, y_pred)
    ) / len(y_true)


def _regression_columns(
    records: Sequence[PredictionRecord], schema: MetricSchema
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    first_true = records[0].y_true
    vector = isinstance(first_true, tuple)
    width = len(first_true) if vector else 1
    if len(schema.target_names) != width:
        raise MetricInputError(
            "target_names count must equal the prediction target dimension"
        )

    truths = [[] for _ in range(width)]
    predictions = [[] for _ in range(width)]
    for record in records:
        if isinstance(record.y_true, tuple) != vector or isinstance(record.y_pred, tuple) != vector:
            raise MetricInputError("all regression records must have the same target shape")
        true_values = record.y_true if vector else (record.y_true,)
        pred_values = record.y_pred if vector else (record.y_pred,)
        if len(true_values) != width or len(pred_values) != width:
            raise MetricInputError("all regression records must have the same target dimension")
        for index, (truth, prediction) in enumerate(zip(true_values, pred_values)):
            truths[index].append(_number(truth, field_name="y_true"))
            predictions[index].append(_number(prediction, field_name="y_pred"))
    return tuple(tuple(column) for column in truths), tuple(
        tuple(column) for column in predictions
    )


def _positive_probability(record: PredictionRecord, schema: MetricSchema) -> float:
    probability = record.probability
    if probability is None:
        raise MetricInputError("binary metrics require probability for every record")
    if isinstance(probability, tuple):
        if len(probability) != 2:
            raise MetricInputError("binary probability vectors must have length two")
        order = schema.probability_class_order or (
            schema.negative_label,
            schema.positive_label,
        )
        positive_index = next(
            index
            for index, label in enumerate(order)
            if _labels_equal(label, schema.positive_label)
        )
        return float(probability[positive_index])
    return float(probability)


def _binary_truth(value: Any, schema: MetricSchema) -> bool:
    if _labels_equal(value, schema.positive_label):
        return True
    if _labels_equal(value, schema.negative_label):
        return False
    raise MetricInputError("binary y_true contains a label outside the metric schema")


def _auroc(truths: Sequence[bool], probabilities: Sequence[float]) -> MetricValue:
    positives = sum(truths)
    negatives = len(truths) - positives
    if positives == 0 or negatives == 0:
        return _undefined(UNDEFINED_SINGLE_CLASS_AUROC)

    ranked = sorted(zip(probabilities, truths), key=lambda item: item[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            1 for _, is_positive in ranked[start:end] if is_positive
        )
        start = end

    value = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return _defined(value)


def _ece(
    truths: Sequence[bool], probabilities: Sequence[float], bin_count: int
) -> float:
    counts = [0] * bin_count
    probability_sums = [0.0] * bin_count
    positive_counts = [0] * bin_count
    for truth, probability in zip(truths, probabilities):
        # Bins are [lower, upper), except that probability 1 belongs to the last bin.
        index = min(int(probability * bin_count), bin_count - 1)
        counts[index] += 1
        probability_sums[index] += probability
        positive_counts[index] += int(truth)

    sample_count = len(truths)
    return math.fsum(
        (count / sample_count)
        * abs(probability_sums[index] / count - positive_counts[index] / count)
        for index, count in enumerate(counts)
        if count
    )


def _validate_schema(schema: MetricSchema, expected_kind: TaskKind) -> None:
    if not isinstance(schema, MetricSchema):
        raise TypeError("schema must be a MetricSchema")
    if schema.task_kind is not expected_kind:
        raise MetricInputError(
            f"metric schema task_kind must be {expected_kind.value}"
        )


def regression_metrics(
    artifact: PredictionArtifact, schema: MetricSchema
) -> MetricSet:
    """Compute per-target R²/MAE and mean R² from validated predictions."""

    _validate_schema(schema, TaskKind.MULTI_TARGET_REGRESSION)
    records = _require_artifact(artifact)
    truths, predictions = _regression_columns(records, schema)

    metrics = {}
    r2_values = []
    has_undefined_r2 = False
    for name, target_truths, target_predictions in zip(
        schema.target_names, truths, predictions
    ):
        r2 = _r2(target_truths, target_predictions)
        metrics[f"r2.{name}"] = r2
        metrics[f"mae.{name}"] = _defined(_mae(target_truths, target_predictions))
        if r2.value is None:
            has_undefined_r2 = True
        else:
            r2_values.append(r2.value)

    metrics["mean_r2"] = (
        _undefined(UNDEFINED_ZERO_VARIANCE_R2)
        if has_undefined_r2
        else _defined(math.fsum(r2_values) / len(r2_values))
    )
    metrics["mean_mae"] = _defined(
        math.fsum(metrics[f"mae.{name}"].value for name in schema.target_names)  # type: ignore[arg-type]
        / len(schema.target_names)
    )
    return MetricSet(
        metric_schema=schema,
        source_prediction_hash=artifact.content_hash,
        metrics=metrics,
    )


def _binary_metric_components(
    records: Sequence[PredictionRecord], schema: MetricSchema
) -> Tuple[dict[str, MetricValue], Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Compute binary scalar metrics from records already inside a trusted artifact.

    This lower-level helper intentionally accepts repeated records so cluster
    bootstrap samples can include the same complete patient more than once.
    """

    if not records:
        raise MetricInputError("binary metrics require at least one prediction record")
    truths = tuple(_binary_truth(record.y_true, schema) for record in records)
    probabilities = tuple(_positive_probability(record, schema) for record in records)
    predictions = tuple(
        probability >= schema.decision_threshold for probability in probabilities
    )

    tp = sum(truth and prediction for truth, prediction in zip(truths, predictions))
    tn = sum(not truth and not prediction for truth, prediction in zip(truths, predictions))
    fp = sum(not truth and prediction for truth, prediction in zip(truths, predictions))
    fn = sum(truth and not prediction for truth, prediction in zip(truths, predictions))
    positives = tp + fn
    negatives = tn + fp

    sensitivity = (
        _defined(tp / positives)
        if positives
        else _undefined("no_positive_samples")
    )
    specificity = (
        _defined(tn / negatives)
        if negatives
        else _undefined("no_negative_samples")
    )
    if sensitivity.value is None or specificity.value is None:
        balanced_accuracy = _undefined(UNDEFINED_SINGLE_CLASS_AUROC)
    else:
        balanced_accuracy = _defined((sensitivity.value + specificity.value) / 2.0)

    metrics = {
        "accuracy": _defined((tp + tn) / len(records)),
        "balanced_accuracy": balanced_accuracy,
        "auroc": _auroc(truths, probabilities),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ece": _defined(_ece(truths, probabilities, schema.ece_bins)),
    }
    return metrics, ((tn, fp), (fn, tp))


def binary_metrics(
    artifact: PredictionArtifact, schema: MetricSchema
) -> MetricSet:
    """Compute fixed-threshold binary discrimination and calibration metrics."""

    _validate_schema(schema, TaskKind.BINARY_CLASSIFICATION)
    records = _require_artifact(artifact)
    metrics, confusion_matrix = _binary_metric_components(records, schema)

    return MetricSet(
        metric_schema=schema,
        source_prediction_hash=artifact.content_hash,
        metrics=metrics,
        confusion_matrices={"confusion_matrix": confusion_matrix},
    )


def remote_metrics(
    artifact: PredictionArtifact, schema: MetricSchema
) -> MetricSet:
    """Compute overall accuracy, average class accuracy, and Cohen's kappa."""

    _validate_schema(schema, TaskKind.REMOTE_CLASSIFICATION)
    records = _require_artifact(artifact)
    labels = schema.class_labels
    label_indices = {_label_identity(label): index for index, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    for record in records:
        truth_index = label_indices.get(_label_identity(record.y_true))
        prediction_index = label_indices.get(_label_identity(record.y_pred))
        if truth_index is None or prediction_index is None:
            raise MetricInputError(
                "remote prediction contains a label outside class_labels"
            )
        matrix[truth_index][prediction_index] += 1

    total = len(records)
    correct = sum(matrix[index][index] for index in range(size))
    row_totals = [sum(row) for row in matrix]
    column_totals = [sum(matrix[row][column] for row in range(size)) for column in range(size)]
    supported_recalls = [
        matrix[index][index] / row_totals[index]
        for index in range(size)
        if row_totals[index]
    ]
    oa = correct / total
    aa = math.fsum(supported_recalls) / len(supported_recalls)
    expected_agreement = math.fsum(
        row_totals[index] * column_totals[index] for index in range(size)
    ) / (total * total)
    kappa = (
        _undefined("zero_expected_disagreement")
        if expected_agreement == 1.0
        else _defined((oa - expected_agreement) / (1.0 - expected_agreement))
    )

    return MetricSet(
        metric_schema=schema,
        source_prediction_hash=artifact.content_hash,
        metrics={"oa": _defined(oa), "aa": _defined(aa), "kappa": kappa},
    )


class MetricEngine:
    """Namespace matching the benchmark design's metric-engine interface."""

    regression_metrics = staticmethod(regression_metrics)
    binary_metrics = staticmethod(binary_metrics)
    remote_metrics = staticmethod(remote_metrics)

    @staticmethod
    def patient_bootstrap(*args: Any, **kwargs: Any) -> Any:
        """Dispatch to patient statistics without creating an import cycle."""

        from benchmark.metrics.statistics import patient_bootstrap

        return patient_bootstrap(*args, **kwargs)

    @staticmethod
    def paired_patient_compare(*args: Any, **kwargs: Any) -> Any:
        """Dispatch to paired patient statistics without an import cycle."""

        from benchmark.metrics.statistics import paired_patient_compare

        return paired_patient_compare(*args, **kwargs)


__all__ = [
    "MetricEngine",
    "MetricInputError",
    "binary_metrics",
    "regression_metrics",
    "remote_metrics",
]
