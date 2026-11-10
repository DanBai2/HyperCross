"""Canonical prediction merge and exact-membership aggregate construction."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, NoReturn, Optional, Protocol, Sequence, Tuple, Union

from benchmark.metrics.predictions import (
    PredictionArtifactReloadError,
    PredictionArtifactWriter,
)
from benchmark.registry.error_journal import ErrorJournal
from benchmark.registry.records import AggregateRecord, Provenance, RunRecord
from benchmark.registry.run_registry import RunRegistry
from benchmark.schema.aggregation import AggregationPolicy
from benchmark.schema.base import RunStatus, ensure_machine_value
from benchmark.schema.metrics import MetricSchema
from benchmark.schema.canonical import hash_file, hash_object, normalize_id
from benchmark.schema.predictions import (
    PredictionArtifact,
    PredictionRecord,
    canonical_prediction_key,
    prediction_key_text,
)
from benchmark.schema.records import ArtifactReference, ErrorRecord


class PredictionMetricComputer(Protocol):
    """Metric engine surface required by the aggregate builder."""

    def compute(
        self,
        records: Tuple[PredictionRecord, ...],
        metric_schema: MetricSchema,
    ) -> Mapping[str, Any]:
        """Compute metrics solely from validated prediction records."""


class AggregateBuildError(ValueError):
    """A validation failure raised before an aggregate registry append."""

    def __init__(self, error_record: ErrorRecord) -> None:
        self.error_record = error_record
        super().__init__(
            f"aggregate build failed: {error_record.error_code} "
            f"({error_record.failing_field})"
        )


def merge_prediction_artifacts(
    artifacts: Iterable[PredictionArtifact],
) -> Tuple[PredictionRecord, ...]:
    """Merge complete artifacts in canonical Prediction_Key order.

    Artifact enumeration order never affects the result. Duplicate keys across
    artifacts are rejected before any metric computer receives records.
    """

    records = []
    for artifact in artifacts:
        if not isinstance(artifact, PredictionArtifact):
            raise TypeError("artifacts must contain PredictionArtifact values")
        if not artifact.complete:
            raise ValueError("all prediction artifacts must be complete")
        records.extend(artifact.records)

    ordered = tuple(sorted(records, key=lambda record: canonical_prediction_key(record.key)))
    duplicate_keys = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous.key == current.key:
            duplicate_keys.append(prediction_key_text(current.key))
    if duplicate_keys:
        raise ValueError(
            "duplicate Prediction_Keys across prediction artifacts: "
            + "; ".join(dict.fromkeys(duplicate_keys))
        )
    return ordered


class AggregateBuilder:
    """Load explicit completed runs and append one prediction-first aggregate.

    The builder never enumerates registry runs. It resolves only the supplied
    member IDs, validates every referenced canonical prediction file and hash,
    computes all metrics from those records, then performs the sole aggregate
    registry mutation as its final step.
    """

    _RESERVED_VARIANCE_KEYS = frozenset({"seed_variance", "fold_variance"})

    def __init__(
        self,
        registry: RunRegistry,
        artifact_root: Union[str, Path],
        metric_computer: PredictionMetricComputer,
        *,
        error_journal: Optional[ErrorJournal] = None,
    ) -> None:
        self.registry = registry
        self.artifact_root = Path(artifact_root).resolve()
        self.metric_computer = metric_computer
        self.error_journal = error_journal or ErrorJournal(
            self.artifact_root / ".errors" / "error-journal.jsonl"
        )

    def build(
        self,
        *,
        aggregate_id: str,
        member_run_ids: Iterable[str],
        aggregation_policy: AggregationPolicy,
        created_utc: Optional[datetime] = None,
    ) -> AggregateRecord:
        """Build and append an aggregate from exactly ``member_run_ids``."""

        if not isinstance(aggregation_policy, AggregationPolicy):
            raise TypeError("aggregation_policy must be an AggregationPolicy")
        members = tuple(member_run_ids)
        if not members:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_MEMBERS_EMPTY",
                field="member_run_ids",
                expected="at least one explicit completed Run_ID",
                observed=[],
            )
        if len(set(members)) != len(members):
            duplicates = sorted(
                {run_id for run_id in members if members.count(run_id) > 1},
                key=lambda run_id: normalize_id(run_id).encode("utf-8"),
            )
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_MEMBER_DUPLICATE",
                field="member_run_ids",
                expected="unique explicit Run_IDs",
                observed=duplicates,
            )
        members = tuple(
            sorted(members, key=lambda run_id: normalize_id(run_id).encode("utf-8"))
        )

        runs = self._load_exact_runs(aggregate_id, members)
        self._validate_run_compatibility(aggregate_id, members, runs)
        self._validate_declared_membership(
            aggregate_id, members, runs, aggregation_policy
        )

        artifacts = tuple(
            self._load_prediction(aggregate_id, members, run) for run in runs
        )
        try:
            merged = merge_prediction_artifacts(artifacts)
        except (TypeError, ValueError) as exc:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_KEY_INVALID",
                field="prediction_keys",
                expected="globally unique canonical Prediction_Keys",
                observed=str(exc),
            )

        for run, artifact in zip(runs, artifacts):
            contexts = {
                (record.key.fold_id, record.key.seed) for record in artifact.records
            }
            expected_context = (run.frozen_identity.fold_id, run.frozen_identity.seed)
            if contexts != {expected_context}:
                self._reject(
                    aggregate_id,
                    members,
                    code="AGGREGATE_RUN_PREDICTION_CONTEXT_MISMATCH",
                    field="prediction_artifact.records.key",
                    expected={
                        "fold_id": expected_context[0],
                        "seed": expected_context[1],
                    },
                    observed=[
                        {"fold_id": fold_id, "seed": seed}
                        for fold_id, seed in sorted(contexts)
                    ],
                )

        metrics = self._compute_metrics(
            aggregate_id, members, merged, aggregation_policy
        )
        prediction_hashes = {
            run.run_id: run.prediction_artifact_ref.content_hash  # type: ignore[union-attr]
            for run in runs
        }
        references = tuple(
            sorted(
                (run.prediction_artifact_ref for run in runs),
                key=lambda reference: normalize_id(reference.artifact_id).encode("utf-8"),  # type: ignore[union-attr]
            )
        )
        provenance = self._aggregate_provenance(
            aggregate_id, members, runs, references
        )

        record = AggregateRecord(
            aggregate_id=aggregate_id,
            member_run_ids=members,
            prediction_hashes=prediction_hashes,
            dataset_hash=runs[0].frozen_identity.dataset_hash,
            protocol_hash=runs[0].frozen_identity.protocol_hash,
            model_key=runs[0].frozen_identity.model_key,
            representation_key=runs[0].input_contract.representation_key,
            aggregation_policy=aggregation_policy,
            aggregation_policy_hash=hash_object(aggregation_policy),
            metrics=metrics,
            artifact_refs=references,
            claim_decision=runs[0].claim_decision,
            provenance=provenance,
            created_utc=created_utc or datetime.now(timezone.utc),
        )
        self.registry.append_aggregate(record)
        return record

    def _load_exact_runs(
        self, aggregate_id: str, members: Tuple[str, ...]
    ) -> Tuple[RunRecord, ...]:
        runs = []
        for run_id in members:
            try:
                run = self.registry.latest_run(run_id)
            except KeyError:
                self._reject(
                    aggregate_id,
                    members,
                    code="AGGREGATE_MEMBER_NOT_FOUND",
                    field="member_run_ids",
                    expected="registered Run_ID",
                    observed=run_id,
                )
            if run.status is not RunStatus.COMPLETE:
                self._reject(
                    aggregate_id,
                    members,
                    code="AGGREGATE_MEMBER_INCOMPLETE",
                    field="run.status",
                    expected=RunStatus.COMPLETE.value,
                    observed={"run_id": run_id, "status": run.status.value},
                )
            if run.prediction_artifact_ref is None:
                self._reject(
                    aggregate_id,
                    members,
                    code="AGGREGATE_PREDICTION_REFERENCE_MISSING",
                    field="prediction_artifact_ref",
                    expected="complete prediction artifact reference",
                    observed=run_id,
                )
            runs.append(run)
        return tuple(runs)

    def _validate_run_compatibility(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        runs: Tuple[RunRecord, ...],
    ) -> None:
        fields = {
            "dataset_hash": [run.frozen_identity.dataset_hash for run in runs],
            "protocol_hash": [run.frozen_identity.protocol_hash for run in runs],
            "model_key": [run.frozen_identity.model_key for run in runs],
            "representation_key": [run.input_contract.representation_key for run in runs],
            "claim_decision": [run.claim_decision.model_dump(mode="json") for run in runs],
        }
        for field, values in fields.items():
            if any(value != values[0] for value in values[1:]):
                self._reject(
                    aggregate_id,
                    members,
                    code="AGGREGATE_MEMBER_IDENTITY_MISMATCH",
                    field=field,
                    expected=values[0],
                    observed=values,
                )

    def _validate_declared_membership(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        runs: Tuple[RunRecord, ...],
        policy: AggregationPolicy,
    ) -> None:
        expected = {
            (fold_id, seed)
            for fold_id in policy.fold_ids
            for seed in policy.seed_set
        }
        observed_pairs = [
            (run.frozen_identity.fold_id, run.frozen_identity.seed) for run in runs
        ]
        observed = set(observed_pairs)
        if len(observed) != len(observed_pairs):
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_JOB_DUPLICATE",
                field="member_run_ids",
                expected="one member per declared Fold_ID and seed",
                observed=[{"fold_id": fold_id, "seed": seed} for fold_id, seed in observed_pairs],
            )
        if observed != expected:
            key = lambda item: (item[0].encode("utf-8"), item[1])
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_DECLARED_MEMBERSHIP_MISMATCH",
                field="member_run_ids",
                expected={
                    "fold_seed_pairs": [
                        {"fold_id": fold_id, "seed": seed}
                        for fold_id, seed in sorted(expected, key=key)
                    ]
                },
                observed={
                    "fold_seed_pairs": [
                        {"fold_id": fold_id, "seed": seed}
                        for fold_id, seed in sorted(observed, key=key)
                    ]
                },
                missing_ids=tuple(
                    f"{fold_id}|{seed}" for fold_id, seed in sorted(expected - observed, key=key)
                ),
                unexpected_ids=tuple(
                    f"{fold_id}|{seed}" for fold_id, seed in sorted(observed - expected, key=key)
                ),
            )

    def _load_prediction(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        run: RunRecord,
    ) -> PredictionArtifact:
        reference = run.prediction_artifact_ref
        assert reference is not None
        try:
            registered = self.registry.get_artifact(reference.artifact_id)
        except KeyError:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_NOT_REGISTERED",
                field="prediction_artifact_ref.artifact_id",
                expected="registered prediction artifact",
                observed=reference.artifact_id,
            )
        if (
            registered.artifact_kind != "prediction"
            or registered.run_id != run.run_id
            or registered.artifact != reference
        ):
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_REGISTRY_MISMATCH",
                field="prediction_artifact_ref",
                expected={"run_id": run.run_id, "artifact_kind": "prediction"},
                observed={
                    "run_id": registered.run_id,
                    "artifact_kind": registered.artifact_kind,
                },
            )

        try:
            path = self._resolve_artifact_path(reference)
        except ValueError as exc:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_PATH_INVALID",
                field="prediction_artifact_ref.relative_path",
                expected="path contained by the configured artifact root",
                observed=str(exc),
            )
        if not path.is_file():
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_FILE_MISSING",
                field="prediction_artifact_ref.relative_path",
                expected="readable prediction artifact file",
                observed=reference.relative_path,
            )
        observed_hash = hash_file(path)
        if observed_hash != reference.content_hash:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_HASH_MISMATCH",
                field="prediction_artifact_ref.content_hash",
                expected=reference.content_hash,
                observed=observed_hash,
            )
        try:
            artifact = PredictionArtifactWriter.reload(path.read_bytes())
        except (OSError, PredictionArtifactReloadError) as exc:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_INVALID",
                field="prediction_artifact_ref",
                expected="complete canonical Prediction_Artifact",
                observed=str(exc),
            )
        if artifact.run_id != run.run_id or artifact.artifact_id != reference.artifact_id:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PREDICTION_IDENTITY_MISMATCH",
                field="prediction_artifact",
                expected={
                    "run_id": run.run_id,
                    "artifact_id": reference.artifact_id,
                },
                observed={
                    "run_id": artifact.run_id,
                    "artifact_id": artifact.artifact_id,
                },
            )
        return artifact

    def _resolve_artifact_path(self, reference: ArtifactReference) -> Path:
        relative = PurePosixPath(reference.relative_path.replace("\\", "/"))
        path = (self.artifact_root / Path(*relative.parts)).resolve(strict=False)
        try:
            path.relative_to(self.artifact_root)
        except ValueError as exc:
            raise ValueError("prediction artifact resolves outside artifact root") from exc
        return path

    def _compute_metrics(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        merged: Tuple[PredictionRecord, ...],
        policy: AggregationPolicy,
    ) -> Dict[str, Any]:
        metrics = self._metric_result(aggregate_id, members, merged, policy)
        conflict = self._RESERVED_VARIANCE_KEYS.intersection(metrics)
        if conflict:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_RESERVED_METRIC_NAME",
                field="metrics",
                expected="metric engine output without builder-owned variance names",
                observed=sorted(conflict),
            )

        seed_values = [
            self._variance_metric(
                aggregate_id,
                members,
                tuple(record for record in merged if record.key.seed == seed),
                policy,
                {"seed": seed},
            )
            for seed in policy.seed_set
        ]
        fold_values = [
            self._variance_metric(
                aggregate_id,
                members,
                tuple(record for record in merged if record.key.fold_id == fold_id),
                policy,
                {"fold_id": fold_id},
            )
            for fold_id in policy.fold_ids
        ]
        return {
            **metrics,
            "seed_variance": statistics.pvariance(seed_values),
            "fold_variance": statistics.pvariance(fold_values),
        }

    def _metric_result(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        records: Tuple[PredictionRecord, ...],
        policy: AggregationPolicy,
    ) -> Dict[str, Any]:
        try:
            result = dict(self.metric_computer.compute(records, policy.metric_schema))
            ensure_machine_value(result, field_name="metrics")
        except Exception as exc:
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_METRIC_COMPUTATION_FAILED",
                field="metrics",
                expected="finite machine-readable prediction-first metrics",
                observed=f"{type(exc).__name__}: {exc}",
            )
        return result

    def _variance_metric(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        records: Tuple[PredictionRecord, ...],
        policy: AggregationPolicy,
        context: Dict[str, Any],
    ) -> float:
        result = self._metric_result(aggregate_id, members, records, policy)
        value = result.get(policy.variance_metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_VARIANCE_METRIC_INVALID",
                field=f"metrics.{policy.variance_metric}",
                expected="finite numeric metric for every declared seed and fold",
                observed={**context, "value": value},
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_VARIANCE_METRIC_INVALID",
                field=f"metrics.{policy.variance_metric}",
                expected="finite numeric metric for every declared seed and fold",
                observed={**context, "value": repr(value)},
            )
        return numeric

    def _aggregate_provenance(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        runs: Tuple[RunRecord, ...],
        references: Sequence[ArtifactReference],
    ) -> Provenance:
        comparable = [
            run.provenance.model_dump(mode="python", exclude={"artifact_hashes"})
            for run in runs
        ]
        if any(value != comparable[0] for value in comparable[1:]):
            self._reject(
                aggregate_id,
                members,
                code="AGGREGATE_PROVENANCE_MISMATCH",
                field="provenance",
                expected=comparable[0],
                observed=comparable,
            )
        artifact_hashes: Dict[str, str] = {}
        for run in runs:
            for artifact_id, content_hash in run.provenance.artifact_hashes.items():
                previous = artifact_hashes.get(artifact_id)
                if previous is not None and previous != content_hash:
                    self._reject(
                        aggregate_id,
                        members,
                        code="AGGREGATE_PROVENANCE_HASH_CONFLICT",
                        field="provenance.artifact_hashes",
                        expected=previous,
                        observed={"artifact_id": artifact_id, "content_hash": content_hash},
                    )
                artifact_hashes[artifact_id] = content_hash
        for reference in references:
            artifact_hashes[reference.artifact_id] = reference.content_hash
        return runs[0].provenance.model_copy(
            update={"artifact_hashes": artifact_hashes}
        )

    def _reject(
        self,
        aggregate_id: str,
        members: Tuple[str, ...],
        *,
        code: str,
        field: str,
        expected: Any,
        observed: Any,
        missing_ids: Tuple[str, ...] = (),
        unexpected_ids: Tuple[str, ...] = (),
    ) -> NoReturn:
        record = ErrorRecord(
            error_code=code,
            failed_stage="aggregate_build",
            affected_identifiers={
                "aggregate_id": str(aggregate_id),
                "member_run_ids": ",".join(str(member) for member in members),
            },
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            missing_ids=missing_ids,
            unexpected_ids=unexpected_ids,
            invariant_checks={
                "completed_predictions_unchanged": True,
                "aggregate_registry_unchanged": True,
            },
        )
        self.error_journal.append(record)
        raise AggregateBuildError(record)


__all__ = [
    "AggregateBuildError",
    "AggregateBuilder",
    "PredictionMetricComputer",
    "merge_prediction_artifacts",
]
