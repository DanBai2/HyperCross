"""Upstream reproduction gate for formal official-model adaptations."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ConfigDict

from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema import (
    ErrorRecord,
    ImplementationKind,
    ModelProvenance,
    StrictSchemaModel,
    UpstreamReproductionRecord,
    UpstreamReproductionStatus,
)


class ReproductionGateDecision(StrictSchemaModel):
    """Immutable approval and the exact checks supporting it."""

    model_key: str
    approved: bool
    formal_project_comparison_allowed: bool
    prerequisite_checks: dict[str, bool]

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
        frozen=True,
    )


class ReproductionGateError(RuntimeError):
    """A blocked formal adaptation with a machine-readable error record."""

    def __init__(self, message: str, error_record: ErrorRecord) -> None:
        super().__init__(message)
        self.error_record = error_record


class ReproductionGate:
    """Require passing, source-bound upstream evidence before adaptation."""

    def __init__(self, error_journal: Optional[ErrorJournal] = None) -> None:
        self._error_journal = error_journal

    def validate_external_provenance(self, registration: Any) -> None:
        """Validate any supplied reproduction record against pinned provenance."""

        identity = getattr(registration, "identity", None)
        model_key = getattr(registration, "model_key", "<unknown>")
        kind = getattr(identity, "implementation_kind", None)
        if kind not in {
            ImplementationKind.OFFICIAL,
            ImplementationKind.ADAPTED_OFFICIAL,
        }:
            self._fail(
                model_key=model_key,
                field="implementation_kind",
                expected="official or adapted_official",
                observed=getattr(kind, "value", kind),
                code="EXTERNAL_PROVENANCE_KIND_INVALID",
                message="external provenance validation requires an official model",
            )

        provenance = getattr(registration, "provenance", None)
        if not isinstance(provenance, ModelProvenance):
            self._fail(
                model_key=model_key,
                field="provenance",
                expected="complete ModelProvenance",
                observed=None if provenance is None else type(provenance).__name__,
                code="OFFICIAL_PROVENANCE_MISSING",
                message="official model is missing external provenance",
            )

        record = provenance.upstream_reproduction_record
        if record is None:
            if (
                provenance.upstream_reproduction_status
                is UpstreamReproductionStatus.PASSED
            ):
                self._fail(
                    model_key=model_key,
                    field="upstream_reproduction_record",
                    expected="record bound to a passed reproduction status",
                    observed=None,
                    code="UPSTREAM_REPRODUCTION_RECORD_MISSING",
                    message="passed reproduction status requires its evidence record",
                )
            return

        bindings = {
            "model_key": (model_key, record.model_key),
            "upstream_repository": (
                provenance.upstream_repository,
                record.upstream_repository,
            ),
            "upstream_commit": (provenance.upstream_commit, record.upstream_commit),
            "license_reference": (
                provenance.license_reference,
                record.license_reference,
            ),
            "source_hash": (provenance.source_hash, record.source_hash),
            "dependency_environment_hash": (
                provenance.dependency_environment_hash,
                record.dependency_environment_hash,
            ),
            "upstream_reproduction_status": (
                provenance.upstream_reproduction_status.value,
                record.status.value,
            ),
        }
        mismatch = next(
            (
                (field, expected, observed)
                for field, (expected, observed) in bindings.items()
                if expected != observed
            ),
            None,
        )
        if mismatch is not None:
            field, expected, observed = mismatch
            self._fail(
                model_key=model_key,
                field=field,
                expected=expected,
                observed=observed,
                code="UPSTREAM_REPRODUCTION_RECORD_INVALID",
                message="reproduction record does not match pinned model provenance",
            )

    def approve_adaptation(
        self,
        registration: Any,
        upstream_record: Optional[UpstreamReproductionRecord] = None,
    ) -> ReproductionGateDecision:
        """Approve an adapted official model or fail before registry mutation.

        The loose registration annotation avoids a module cycle while runtime
        checks still require the ModelRegistration interface.
        """

        identity = getattr(registration, "identity", None)
        model_key = getattr(registration, "model_key", "<unknown>")
        implementation_kind = getattr(registration, "implementation_kind", None)
        if (
            implementation_kind is not ImplementationKind.ADAPTED_OFFICIAL
            or identity is None
            or identity.implementation_kind is not ImplementationKind.ADAPTED_OFFICIAL
        ):
            self._fail(
                model_key=model_key,
                field="implementation_kind",
                expected=ImplementationKind.ADAPTED_OFFICIAL.value,
                observed={
                    "registration": getattr(implementation_kind, "value", implementation_kind),
                    "identity": getattr(
                        getattr(identity, "implementation_kind", None),
                        "value",
                        getattr(identity, "implementation_kind", None),
                    ),
                },
                code="REPRODUCTION_GATE_NOT_ADAPTED_OFFICIAL",
                message="reproduction approval applies only to adapted official models",
            )

        self.validate_external_provenance(registration)
        provenance = getattr(registration, "provenance", None)
        if not isinstance(provenance, ModelProvenance):
            self._fail(
                model_key=model_key,
                field="provenance",
                expected="complete ModelProvenance",
                observed=None if provenance is None else type(provenance).__name__,
                code="OFFICIAL_PROVENANCE_MISSING",
                message="adapted official model is missing external provenance",
            )

        record = upstream_record or provenance.upstream_reproduction_record
        if record is None:
            self._fail(
                model_key=model_key,
                field="upstream_reproduction_record",
                expected="a passing upstream reproduction record",
                observed=None,
                code="UPSTREAM_REPRODUCTION_RECORD_MISSING",
                message="formal adaptation requires a passing upstream reproduction record",
            )

        checks = {
            "model_key": record.model_key == model_key,
            "upstream_repository": (
                record.upstream_repository == provenance.upstream_repository
            ),
            "upstream_commit": record.upstream_commit == provenance.upstream_commit,
            "license_reference": (
                record.license_reference == provenance.license_reference
            ),
            "source_hash": record.source_hash == provenance.source_hash,
            "dependency_environment_hash": (
                record.dependency_environment_hash
                == provenance.dependency_environment_hash
            ),
            "record_status_passed": (
                record.status is UpstreamReproductionStatus.PASSED
            ),
            "provenance_status_passed": (
                provenance.upstream_reproduction_status
                is UpstreamReproductionStatus.PASSED
            ),
            "status_equality": (
                provenance.upstream_reproduction_status is record.status
            ),
        }
        failed_field = next((name for name, passed in checks.items() if not passed), None)
        if failed_field is not None:
            self._fail(
                model_key=model_key,
                field=(
                    "upstream_reproduction_record"
                    if failed_field.endswith("status_passed")
                    else failed_field
                ),
                expected=self._expected_value(failed_field, model_key, provenance),
                observed=self._observed_value(failed_field, record, provenance),
                code="UPSTREAM_REPRODUCTION_RECORD_INVALID",
                message=(
                    "upstream reproduction record is not passing or does not match "
                    "the pinned model provenance"
                ),
            )

        return ReproductionGateDecision(
            model_key=model_key,
            approved=True,
            formal_project_comparison_allowed=True,
            prerequisite_checks=checks,
        )

    @staticmethod
    def _expected_value(
        field: str, model_key: str, provenance: ModelProvenance
    ) -> Any:
        expected: dict[str, Any] = {
            "model_key": model_key,
            "upstream_repository": provenance.upstream_repository,
            "upstream_commit": provenance.upstream_commit,
            "license_reference": provenance.license_reference,
            "source_hash": provenance.source_hash,
            "dependency_environment_hash": provenance.dependency_environment_hash,
            "record_status_passed": UpstreamReproductionStatus.PASSED.value,
            "provenance_status_passed": UpstreamReproductionStatus.PASSED.value,
            "status_equality": provenance.upstream_reproduction_status.value,
        }
        return expected[field]

    @staticmethod
    def _observed_value(
        field: str,
        record: UpstreamReproductionRecord,
        provenance: ModelProvenance,
    ) -> Any:
        observed: dict[str, Any] = {
            "model_key": record.model_key,
            "upstream_repository": record.upstream_repository,
            "upstream_commit": record.upstream_commit,
            "license_reference": record.license_reference,
            "source_hash": record.source_hash,
            "dependency_environment_hash": record.dependency_environment_hash,
            "record_status_passed": record.status.value,
            "provenance_status_passed": provenance.upstream_reproduction_status.value,
            "status_equality": record.status.value,
        }
        return observed[field]

    def _fail(
        self,
        *,
        model_key: str,
        field: str,
        expected: Any,
        observed: Any,
        code: str,
        message: str,
    ) -> None:
        error = ErrorRecord(
            error_code=code,
            failed_stage="reproduction_gate",
            affected_identifiers={"model_key": model_key},
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            invariant_checks={
                "adaptation_not_approved": True,
                "training_not_started": True,
                "registered_upstream_artifacts_unchanged": True,
                "completed_runs_unchanged": True,
            },
        )
        if self._error_journal is not None:
            self._error_journal.append(error)
        raise ReproductionGateError(message, error)
