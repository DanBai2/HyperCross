"""Deterministic patient-cluster bootstrap and paired comparisons."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence, Tuple

from benchmark.metrics.engine import (
    MetricInputError,
    _binary_metric_components,
    binary_metrics,
)
from benchmark.schema.base import TaskKind
from benchmark.schema.canonical import normalize_id
from benchmark.schema.metrics import (
    ConfidenceInterval,
    MetricSchema,
    StatisticalComparison,
)
from benchmark.schema.predictions import PredictionArtifact, PredictionRecord


PATIENT_RESAMPLES = 10_000
PATIENT_CONFIDENCE_LEVEL = 0.95
PATIENT_SIDEDNESS = "two_sided"
PATIENT_RESAMPLING_UNIT = "patient_group_id"
_LOWER_PERCENTILE = 0.025
_UPPER_PERCENTILE = 0.975


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


def _group_records(
    artifact: PredictionArtifact,
    group_by_sample_id: Mapping[str, str],
) -> dict[str, Tuple[PredictionRecord, ...]]:
    if not isinstance(group_by_sample_id, Mapping):
        raise TypeError("group_by_sample_id must be a mapping")

    normalized: dict[str, str] = {}
    for raw_sample_id, raw_group_id in group_by_sample_id.items():
        if not isinstance(raw_sample_id, str) or not isinstance(raw_group_id, str):
            raise MetricInputError("sample and patient Group_ID values must be strings")
        sample_id = normalize_id(raw_sample_id)
        group_id = normalize_id(raw_group_id)
        if (
            not group_id
            or group_id != group_id.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in group_id)
        ):
            raise MetricInputError(
                f"invalid patient Group_ID for sample {sample_id!r}"
            )
        previous = normalized.get(sample_id)
        if previous is not None and previous != group_id:
            raise MetricInputError(
                f"sample {sample_id!r} maps to multiple patient Group_ID values"
            )
        normalized[sample_id] = group_id

    missing = tuple(
        sorted(
            {
                record.key.sample_id
                for record in artifact.records
                if record.key.sample_id not in normalized
            },
            key=lambda value: value.encode("utf-8"),
        )
    )
    if missing:
        raise MetricInputError(
            "patient Group_ID mapping is missing prediction samples: "
            + ", ".join(missing)
        )

    grouped: dict[str, list[PredictionRecord]] = {}
    for record in artifact.records:
        grouped.setdefault(normalized[record.key.sample_id], []).append(record)
    return {
        group_id: tuple(records)
        for group_id, records in sorted(
            grouped.items(), key=lambda item: item[0].encode("utf-8")
        )
    }


def _metric_value(
    records: Sequence[PredictionRecord],
    metric_schema: MetricSchema,
    metric_name: str,
) -> float:
    metrics, _ = _binary_metric_components(records, metric_schema)
    try:
        result = metrics[metric_name]
    except KeyError as exc:
        supported = ", ".join(sorted(metrics))
        raise MetricInputError(
            f"unsupported patient bootstrap metric {metric_name!r}; supported: {supported}"
        ) from exc
    if result.value is None:
        raise MetricInputError(
            f"metric {metric_name!r} is undefined for a patient resample: {result.reason}"
        )
    return result.value


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile using rank ``(n - 1) * p``."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = rank - lower_index
    return ordered[lower_index] + weight * (
        ordered[upper_index] - ordered[lower_index]
    )


def _resampled_records(
    grouped: Mapping[str, Tuple[PredictionRecord, ...]],
    group_ids: Sequence[str],
    rng: random.Random,
) -> Tuple[PredictionRecord, ...]:
    selected = (group_ids[rng.randrange(len(group_ids))] for _ in group_ids)
    return tuple(
        record
        for group_id in selected
        for record in grouped[group_id]
    )


def patient_bootstrap(
    artifact: PredictionArtifact,
    metric_name: str,
    *,
    group_by_sample_id: Mapping[str, str],
    metric_schema: MetricSchema,
    seed: int,
) -> ConfidenceInterval:
    """Compute a fixed 10,000-draw two-sided 95% patient-cluster CI.

    Every draw samples the original number of patient Group_IDs with
    replacement and carries all ROI prediction records for each selected
    patient into the metric computation.
    """

    seed = _validate_seed(seed)
    if not isinstance(metric_name, str):
        raise TypeError("metric_name must be a string")
    if metric_schema.task_kind is not TaskKind.BINARY_CLASSIFICATION:
        raise MetricInputError("patient bootstrap requires a binary metric schema")

    # This validates completeness, content hash, schema compatibility, labels,
    # and probabilities before any resampling work begins.
    full_metrics = binary_metrics(artifact, metric_schema)
    try:
        point_metric = full_metrics.metric(metric_name)
    except KeyError as exc:
        raise MetricInputError(
            f"unsupported patient bootstrap metric {metric_name!r}"
        ) from exc
    if point_metric.value is None:
        raise MetricInputError(
            f"metric {metric_name!r} is undefined for the source artifact: "
            f"{point_metric.reason}"
        )

    grouped = _group_records(artifact, group_by_sample_id)
    group_ids = tuple(grouped)
    rng = random.Random(seed)
    estimates = tuple(
        _metric_value(
            _resampled_records(grouped, group_ids, rng),
            metric_schema,
            metric_name,
        )
        for _ in range(PATIENT_RESAMPLES)
    )
    return ConfidenceInterval(
        metric_name=metric_name,
        source_prediction_hash=artifact.content_hash,
        point_estimate=point_metric.value,
        lower_bound=_percentile(estimates, _LOWER_PERCENTILE),
        upper_bound=_percentile(estimates, _UPPER_PERCENTILE),
        resampling_unit=PATIENT_RESAMPLING_UNIT,
        resamples=PATIENT_RESAMPLES,
        confidence_level=PATIENT_CONFIDENCE_LEVEL,
        sidedness=PATIENT_SIDEDNESS,
        seed=seed,
    )


def paired_patient_compare(
    candidate: PredictionArtifact,
    control: PredictionArtifact,
    metric_name: str,
    *,
    candidate_group_by_sample_id: Mapping[str, str],
    control_group_by_sample_id: Mapping[str, str],
    metric_schema: MetricSchema,
    seed: int,
) -> StatisticalComparison:
    """Compare candidate minus control using only common patient clusters.

    Each bootstrap draw selects one shared sequence of patient Group_IDs and
    applies it to both artifacts, preserving pairing while retaining every ROI
    belonging to each selected patient.
    """

    seed = _validate_seed(seed)
    if not isinstance(metric_name, str):
        raise TypeError("metric_name must be a string")
    if metric_schema.task_kind is not TaskKind.BINARY_CLASSIFICATION:
        raise MetricInputError("paired patient comparison requires a binary metric schema")

    # Validate each source independently before restricting to common patients.
    binary_metrics(candidate, metric_schema)
    binary_metrics(control, metric_schema)
    candidate_groups = _group_records(candidate, candidate_group_by_sample_id)
    control_groups = _group_records(control, control_group_by_sample_id)
    common_group_ids = tuple(
        sorted(
            set(candidate_groups).intersection(control_groups),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if not common_group_ids:
        raise MetricInputError(
            "paired patient comparison requires at least one common patient Group_ID"
        )

    candidate_common = tuple(
        record
        for group_id in common_group_ids
        for record in candidate_groups[group_id]
    )
    control_common = tuple(
        record
        for group_id in common_group_ids
        for record in control_groups[group_id]
    )
    candidate_estimate = _metric_value(
        candidate_common, metric_schema, metric_name
    )
    control_estimate = _metric_value(control_common, metric_schema, metric_name)

    rng = random.Random(seed)
    differences = []
    for _ in range(PATIENT_RESAMPLES):
        selected = tuple(
            common_group_ids[rng.randrange(len(common_group_ids))]
            for _ in common_group_ids
        )
        candidate_records = tuple(
            record
            for group_id in selected
            for record in candidate_groups[group_id]
        )
        control_records = tuple(
            record
            for group_id in selected
            for record in control_groups[group_id]
        )
        differences.append(
            _metric_value(candidate_records, metric_schema, metric_name)
            - _metric_value(control_records, metric_schema, metric_name)
        )

    return StatisticalComparison(
        metric_name=metric_name,
        candidate_prediction_hash=candidate.content_hash,
        control_prediction_hash=control.content_hash,
        candidate_estimate=candidate_estimate,
        control_estimate=control_estimate,
        estimated_difference=candidate_estimate - control_estimate,
        lower_bound=_percentile(differences, _LOWER_PERCENTILE),
        upper_bound=_percentile(differences, _UPPER_PERCENTILE),
        common_group_ids=common_group_ids,
        resampling_unit=PATIENT_RESAMPLING_UNIT,
        resamples=PATIENT_RESAMPLES,
        confidence_level=PATIENT_CONFIDENCE_LEVEL,
        sidedness=PATIENT_SIDEDNESS,
        seed=seed,
    )


__all__ = [
    "PATIENT_CONFIDENCE_LEVEL",
    "PATIENT_RESAMPLES",
    "PATIENT_RESAMPLING_UNIT",
    "PATIENT_SIDEDNESS",
    "paired_patient_compare",
    "patient_bootstrap",
]
