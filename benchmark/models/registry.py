"""Stable-key model registry with post-build identity validation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import re
from typing import Any, Optional

from benchmark.models.reproduction import ReproductionGate, ReproductionGateDecision
from benchmark.registry.error_journal import ErrorJournal
from benchmark.schema import (
    ClaimStatus,
    ErrorRecord,
    ImplementationKind,
    InputContract,
    ModelIdentity,
    ModelProvenance,
    ModelRunMetadata,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


ModelBuilder = Callable[..., Any]


@dataclass(frozen=True)
class ModelRegistration:
    """One independently declared stable-key model factory.

    ``implementation_kind`` and ``source_revision`` describe the resolved
    factory and are intentionally separate from ``identity`` so a mismatch is
    detectable. A built instance may expose stricter runtime metadata through
    ``__benchmark_model_key__``, ``__benchmark_implementation_kind__``, and
    ``__benchmark_source_revision__``.
    """

    model_key: str
    identity: ModelIdentity
    input_contract: InputContract
    builder: ModelBuilder
    implementation_kind: ImplementationKind
    source_revision: str
    provenance: Optional[ModelProvenance] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key.strip():
            raise ValueError("model_key must be a non-empty string")
        if self.model_key != self.model_key.strip():
            raise ValueError("model_key must not have surrounding whitespace")
        if not callable(self.builder):
            raise TypeError("builder must be callable")
        if not isinstance(self.implementation_kind, ImplementationKind):
            try:
                object.__setattr__(
                    self,
                    "implementation_kind",
                    ImplementationKind(self.implementation_kind),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("implementation_kind is invalid") from exc
        if not isinstance(self.source_revision, str) or not self.source_revision.strip():
            raise ValueError("source_revision must be a non-empty string")
        if self.source_revision != self.source_revision.strip():
            raise ValueError("source_revision must not have surrounding whitespace")
        if self.provenance is not None and not isinstance(
            self.provenance, ModelProvenance
        ):
            raise TypeError("provenance must be ModelProvenance or None")


class ModelRegistryError(RuntimeError):
    """Registry failure carrying its machine-readable error record."""

    def __init__(self, message: str, error_record: ErrorRecord) -> None:
        super().__init__(message)
        self.error_record = error_record


class ModelRegistry(Mapping[str, ModelRegistration]):
    """Resolve one immutable registration for each stable ``Model_Key``."""

    def __init__(self, error_journal: Optional[ErrorJournal] = None) -> None:
        self._registrations: dict[str, ModelRegistration] = {}
        self._error_journal = error_journal
        self._reproduction_gate = ReproductionGate(error_journal)

    def __getitem__(self, model_key: str) -> ModelRegistration:
        return self.resolve(model_key)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._registrations))

    def __len__(self) -> int:
        return len(self._registrations)

    @property
    def model_keys(self) -> tuple[str, ...]:
        """Return stable keys in deterministic bytewise order."""

        return tuple(sorted(self._registrations, key=lambda value: value.encode("utf-8")))

    def snapshot(self) -> tuple[tuple[str, int], ...]:
        """Return a process-local identity snapshot useful for atomicity checks."""

        return tuple((key, id(self._registrations[key])) for key in self.model_keys)

    def register(self, registration: ModelRegistration) -> None:
        """Register exactly one identity/contract pair without replacement."""

        if not isinstance(registration, ModelRegistration):
            raise TypeError("registration must be a ModelRegistration")
        key = registration.model_key
        if key in self._registrations:
            self._fail(
                code="MODEL_KEY_ALREADY_REGISTERED",
                model_key=key,
                field="model_key",
                expected="an unregistered stable Model_Key",
                observed=key,
                message=f"Model_Key {key!r} is already registered",
            )
        if registration.identity.model_key != key:
            self._fail(
                code="MODEL_KEY_IDENTITY_MISMATCH",
                model_key=key,
                field="model_key",
                expected=key,
                observed=registration.identity.model_key,
                message="registration Model_Key disagrees with ModelIdentity",
            )
        self._validate_provenance(registration)
        self._registrations[key] = registration

    def _validate_provenance(self, registration: ModelRegistration) -> None:
        """Enforce tier-specific provenance before mutating the registry."""

        kind = registration.identity.implementation_kind
        display_name = registration.identity.display_name
        if kind is ImplementationKind.PROXY and re.search(
            r"\bproxy\b", display_name, flags=re.IGNORECASE
        ) is None:
            self._fail(
                code="PROXY_DISPLAY_NAME_MISSING_LABEL",
                model_key=registration.model_key,
                field="display_name",
                expected="a display name containing the word 'proxy'",
                observed=display_name,
                message="proxy model display name must contain the word 'proxy'",
            )

        if kind in {
            ImplementationKind.OFFICIAL,
            ImplementationKind.ADAPTED_OFFICIAL,
        } and registration.provenance is None:
            self._fail(
                code="OFFICIAL_PROVENANCE_MISSING",
                model_key=registration.model_key,
                field="provenance",
                expected="complete external ModelProvenance",
                observed=None,
                message="official model registration requires complete provenance",
            )

        if kind in {
            ImplementationKind.OFFICIAL,
            ImplementationKind.ADAPTED_OFFICIAL,
        } and registration.provenance is not None:
            self._reproduction_gate.validate_external_provenance(registration)

        if kind is ImplementationKind.ADAPTED_OFFICIAL:
            assert registration.provenance is not None
            if not registration.provenance.adaptation_notes:
                self._fail(
                    code="ADAPTATION_NOTES_MISSING",
                    model_key=registration.model_key,
                    field="adaptation_notes",
                    expected="at least one recorded input adaptation",
                    observed=[],
                    message="adapted official model must record every input adaptation",
                )
            self._reproduction_gate.approve_adaptation(registration)

    def resolve(self, model_key: str) -> ModelRegistration:
        """Resolve exactly one registration by stable key, never by position."""

        if not isinstance(model_key, str):
            self._fail(
                code="MODEL_SELECTOR_NOT_STABLE_KEY",
                model_key=str(model_key),
                field="model_key",
                expected="stable string Model_Key",
                observed=type(model_key).__name__,
                message="models must be selected by stable string Model_Key",
            )
        try:
            return self._registrations[model_key]
        except KeyError:
            self._fail(
                code="MODEL_KEY_NOT_REGISTERED",
                model_key=model_key,
                field="model_key",
                expected=list(self.model_keys),
                observed=model_key,
                message=f"Model_Key {model_key!r} is not registered",
            )
        raise AssertionError("unreachable")

    def approve_adaptation(self, model_key: str) -> ReproductionGateDecision:
        """Re-evaluate the reproduction gate for one registered adaptation."""

        return self._reproduction_gate.approve_adaptation(self.resolve(model_key))

    def claim_status_ceiling(
        self,
        model_key: str,
        proposed_status: ClaimStatus,
    ) -> ClaimStatus:
        """Apply the provenance ceiling implied by a registered model tier.

        Proxy provenance cannot support a claim above ``proxy_only``. A lower
        status, such as ``invalid_or_leaky``, remains lower. Other model tiers
        are unchanged here because their additional ceilings belong to the
        full Claim Gate.
        """

        if not isinstance(proposed_status, ClaimStatus):
            try:
                proposed_status = ClaimStatus(proposed_status)
            except (TypeError, ValueError) as exc:
                raise ValueError("proposed_status is invalid") from exc

        registration = self.resolve(model_key)
        if registration.identity.implementation_kind is not ImplementationKind.PROXY:
            return proposed_status

        precedence = tuple(ClaimStatus)
        proxy_index = precedence.index(ClaimStatus.PROXY_ONLY)
        proposed_index = precedence.index(proposed_status)
        if proposed_index < proxy_index:
            return ClaimStatus.PROXY_ONLY
        return proposed_status

    def build(self, model_key: str, /, *args: Any, **kwargs: Any) -> Any:
        """Construct a model and reject every identity mismatch before return."""

        registration = self.resolve(model_key)
        try:
            instance = registration.builder(*args, **kwargs)
        except Exception as exc:
            self._fail(
                code="MODEL_BUILD_FAILED",
                model_key=model_key,
                field="builder",
                expected="a successfully constructed model instance",
                observed=f"{type(exc).__name__}: {exc}",
                message=f"builder for Model_Key {model_key!r} failed",
            )

        report = self.validate_instance(registration, instance)
        if not report.passes:
            issue = report.errors[0]
            self._fail(
                code=issue.code,
                model_key=model_key,
                field=issue.field_path[-1],
                expected=issue.expected_value,
                observed=issue.observed_value,
                message=issue.message,
            )
        return instance

    @classmethod
    def validate_instance(
        cls,
        registration: ModelRegistration,
        instance: Any,
    ) -> ValidationReport:
        """Compare built class, kind, parameter count, and source revision."""

        expected = registration.identity
        actual_class = cls.fully_qualified_class_name(instance)
        actual_key = getattr(instance, "__benchmark_model_key__", registration.model_key)
        resolved_kind_value = registration.implementation_kind.value
        runtime_kind = getattr(instance, "__benchmark_implementation_kind__", None)
        if isinstance(runtime_kind, ImplementationKind):
            runtime_kind_value: Any = runtime_kind.value
        else:
            runtime_kind_value = runtime_kind
        resolved_revision = registration.source_revision
        runtime_revision = getattr(instance, "__benchmark_source_revision__", None)

        issues: list[ValidationIssue] = []
        cls._append_mismatch(
            issues,
            code="MODEL_KEY_INSTANCE_MISMATCH",
            field="model_key",
            expected=expected.model_key,
            observed=actual_key,
        )
        cls._append_mismatch(
            issues,
            code="MODEL_CLASS_MISMATCH",
            field="class_fully_qualified_name",
            expected=expected.class_fully_qualified_name,
            observed=actual_class,
        )
        cls._append_mismatch(
            issues,
            code="MODEL_IMPLEMENTATION_KIND_MISMATCH",
            field="implementation_kind",
            expected=expected.implementation_kind.value,
            observed=resolved_kind_value,
        )
        if runtime_kind is not None:
            cls._append_mismatch(
                issues,
                code="MODEL_IMPLEMENTATION_KIND_MISMATCH",
                field="implementation_kind",
                expected=expected.implementation_kind.value,
                observed=runtime_kind_value,
            )

        try:
            actual_parameter_count: Any = cls.parameter_count(instance)
        except (TypeError, ValueError) as exc:
            issues.append(
                cls._issue(
                    code="MODEL_PARAMETER_COUNT_UNAVAILABLE",
                    field="parameter_count",
                    expected=expected.parameter_count,
                    observed=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            cls._append_mismatch(
                issues,
                code="MODEL_PARAMETER_COUNT_MISMATCH",
                field="parameter_count",
                expected=expected.parameter_count,
                observed=actual_parameter_count,
            )

        cls._append_mismatch(
            issues,
            code="MODEL_SOURCE_REVISION_MISMATCH",
            field="source_revision",
            expected=expected.source_revision,
            observed=resolved_revision,
        )
        if runtime_revision is not None:
            cls._append_mismatch(
                issues,
                code="MODEL_SOURCE_REVISION_MISMATCH",
                field="source_revision",
                expected=expected.source_revision,
                observed=runtime_revision,
            )
        return ValidationReport(issues=tuple(issues))

    def run_metadata(
        self,
        model_key: str,
        instance: Any,
        *,
        config_hash: str,
        supplied_contract: InputContract,
    ) -> ModelRunMetadata:
        """Validate the model and supplied inputs before starting training.

        The caller describes the materialized inputs with ``supplied_contract``.
        Returning metadata is the run-start gate: no training caller should
        proceed unless both model identity and every input-contract field match.
        """

        registration = self.resolve(model_key)
        report = self.validate_instance(registration, instance)
        if not report.passes:
            issue = report.errors[0]
            self._fail(
                code=issue.code,
                model_key=model_key,
                field=issue.field_path[-1],
                expected=issue.expected_value,
                observed=issue.observed_value,
                message=issue.message,
            )

        contract_report = self.validate_input_contract(
            registration.input_contract,
            supplied_contract,
        )
        if not contract_report.passes:
            issue = contract_report.errors[0]
            self._fail(
                code=issue.code,
                model_key=model_key,
                field=issue.field_path[-1],
                expected=issue.expected_value,
                observed=issue.observed_value,
                message=issue.message,
                invariant_checks={
                    "training_not_started": True,
                    "checkpoint_unchanged": True,
                    "prediction_artifact_unchanged": True,
                    "completed_run_records_unchanged": True,
                },
            )

        return ModelRunMetadata(
            model_key=model_key,
            source_revision=registration.identity.source_revision,
            config_hash=config_hash,
            parameter_count=registration.identity.parameter_count,
            representation_key=registration.input_contract.representation_key,
            preprocessing_key=registration.input_contract.preprocessing_key,
        )

    @classmethod
    def validate_input_contract(
        cls,
        expected: InputContract,
        supplied: InputContract,
    ) -> ValidationReport:
        """Compare all representation and materialized tensor declarations.

        Modalities have set semantics. Tensor dimensions and channel/band order
        remain ordered because changing either changes the model input meaning.
        No tensor or training operation occurs in this validation path.
        """

        if not isinstance(expected, InputContract):
            raise TypeError("expected must be an InputContract")
        if not isinstance(supplied, InputContract):
            raise TypeError("supplied must be an InputContract")

        issues: list[ValidationIssue] = []
        comparisons = (
            (
                "representation_key",
                "INPUT_CONTRACT_REPRESENTATION_KEY_MISMATCH",
                expected.representation_key,
                supplied.representation_key,
            ),
            (
                "modalities",
                "INPUT_CONTRACT_MODALITIES_MISMATCH",
                cls._canonical_modalities(expected.modalities),
                cls._canonical_modalities(supplied.modalities),
            ),
            (
                "tensor_shapes",
                "INPUT_CONTRACT_TENSOR_SHAPES_MISMATCH",
                expected.tensor_shapes,
                supplied.tensor_shapes,
            ),
            (
                "dtypes",
                "INPUT_CONTRACT_DTYPES_MISMATCH",
                expected.dtypes,
                supplied.dtypes,
            ),
            (
                "channel_or_band_order",
                "INPUT_CONTRACT_CHANNEL_OR_BAND_ORDER_MISMATCH",
                expected.channel_or_band_order,
                supplied.channel_or_band_order,
            ),
            (
                "preprocessing_key",
                "INPUT_CONTRACT_PREPROCESSING_KEY_MISMATCH",
                expected.preprocessing_key,
                supplied.preprocessing_key,
            ),
        )
        for field, code, expected_value, observed_value in comparisons:
            if observed_value != expected_value:
                issues.append(
                    ValidationIssue(
                        code=code,
                        severity=ValidationSeverity.ERROR,
                        field_path=(field,),
                        message=f"input contract field {field!r} does not match",
                        expected_value=expected_value,
                        observed_value=observed_value,
                    )
                )
        return ValidationReport(issues=tuple(issues))

    @staticmethod
    def _canonical_modalities(modalities: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(modalities, key=lambda value: value.encode("utf-8")))

    @staticmethod
    def fully_qualified_class_name(instance: Any) -> str:
        model_class = type(instance)
        return f"{model_class.__module__}.{model_class.__qualname__}"

    @staticmethod
    def parameter_count(instance: Any) -> int:
        """Count all model parameters without importing a framework."""

        parameters = getattr(instance, "parameters", None)
        if not callable(parameters):
            raise TypeError("model instance does not expose parameters()")
        total = 0
        for parameter in parameters():
            numel = getattr(parameter, "numel", None)
            if not callable(numel):
                raise TypeError("model parameter does not expose numel()")
            count = numel()
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("model parameter numel() must return a non-negative integer")
            total += count
        return total

    @classmethod
    def _append_mismatch(
        cls,
        issues: list[ValidationIssue],
        *,
        code: str,
        field: str,
        expected: Any,
        observed: Any,
    ) -> None:
        if observed != expected:
            issues.append(
                cls._issue(
                    code=code,
                    field=field,
                    expected=expected,
                    observed=observed,
                )
            )

    @staticmethod
    def _issue(
        *,
        code: str,
        field: str,
        expected: Any,
        observed: Any,
    ) -> ValidationIssue:
        return ValidationIssue(
            code=code,
            severity=ValidationSeverity.ERROR,
            field_path=(field,),
            message=f"model identity field {field!r} does not match",
            expected_value=expected,
            observed_value=observed,
        )

    def _fail(
        self,
        *,
        code: str,
        model_key: str,
        field: str,
        expected: Any,
        observed: Any,
        message: str,
        invariant_checks: Optional[dict[str, bool]] = None,
    ) -> None:
        error = ErrorRecord(
            error_code=code,
            failed_stage="model_validation",
            affected_identifiers={"model_key": model_key},
            failing_field=field,
            expected_value=expected,
            observed_value=observed,
            invariant_checks=(
                invariant_checks
                if invariant_checks is not None
                else {
                    "model_not_returned": True,
                    "training_not_started": True,
                    "completed_artifacts_unchanged": True,
                }
            ),
        )
        if self._error_journal is not None:
            self._error_journal.append(error)
        raise ModelRegistryError(message, error)
