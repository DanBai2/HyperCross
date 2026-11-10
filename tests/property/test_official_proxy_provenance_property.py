"""Property 14: Official and proxy provenance safety.

**Validates: Requirements 6.14-6.23, 7.21-7.24**
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.models import (
    ModelRegistration,
    ModelRegistry,
    ModelRegistryError,
    ReproductionGateError,
)
from benchmark.schema import (
    ClaimStatus,
    ImplementationKind,
    InputContract,
    ModelIdentity,
    ModelProvenance,
    UpstreamReproductionRecord,
    UpstreamReproductionStatus,
)


_UTC = datetime(2027, 1, 1, tzinfo=timezone.utc)
_IDENTIFIERS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=24,
).filter(lambda value: value[0] != "-" and value[-1] != "-")
_NON_PROXY_LABELS = _IDENTIFIERS.filter(lambda value: "proxy" not in value.lower())


class ProvenanceToyModel:
    def parameters(self):
        return iter(())


def _hash(namespace: str, token: str) -> str:
    return hashlib.sha256(f"{namespace}:{token}".encode("utf-8")).hexdigest()


def _contract(token: str) -> InputContract:
    return InputContract(
        representation_key=f"representation-{token}",
        modalities=("spectrum",),
        tensor_shapes={"spectrum": (128,)},
        dtypes={"spectrum": "float32"},
        channel_or_band_order={"spectrum": ("wavelength-ascending",)},
        preprocessing_key=f"preprocessing-{token}",
    )


def _registration(
    *,
    token: str,
    kind: ImplementationKind,
    display_name: str,
    provenance: ModelProvenance | None,
) -> ModelRegistration:
    model_key = f"model-{token}"
    revision = f"revision-{token}"
    identity = ModelIdentity(
        model_key=model_key,
        display_name=display_name,
        class_fully_qualified_name=(
            f"{ProvenanceToyModel.__module__}.{ProvenanceToyModel.__qualname__}"
        ),
        implementation_kind=kind,
        parameter_count=0,
        source_revision=revision,
    )
    return ModelRegistration(
        model_key=model_key,
        identity=identity,
        input_contract=_contract(token),
        builder=ProvenanceToyModel,
        implementation_kind=kind,
        source_revision=revision,
        provenance=provenance,
    )


def _provenance(
    *,
    token: str,
    model_key: str,
    status: UpstreamReproductionStatus,
    adaptation_notes: tuple[str, ...],
    ignored_modalities: tuple[str, ...],
    include_record: bool = True,
) -> ModelProvenance:
    repository = f"https://example.org/upstream/{token}"
    commit = f"commit-{token}"
    license_reference = f"license-{token}"
    source_hash = _hash("source", token)
    environment_hash = _hash("environment", token)
    record = None
    if include_record:
        record = UpstreamReproductionRecord(
            model_key=model_key,
            upstream_repository=repository,
            upstream_commit=commit,
            license_reference=license_reference,
            source_hash=source_hash,
            dependency_environment_hash=environment_hash,
            result_artifact_hash=_hash("result", token),
            status=status,
            completed_utc=_UTC,
        )
    return ModelProvenance(
        upstream_repository=repository,
        upstream_commit=commit,
        license_reference=license_reference,
        downloaded_utc=_UTC,
        source_hash=source_hash,
        local_patch_hash=_hash("patch", token),
        dependency_environment_hash=environment_hash,
        upstream_reproduction_status=status,
        upstream_reproduction_record=record,
        adaptation_notes=adaptation_notes,
        ignored_modalities=ignored_modalities,
    )


@st.composite
def _official_cases(draw: st.DrawFn) -> tuple[ModelRegistration, ModelProvenance]:
    token = draw(_IDENTIFIERS)
    kind = draw(
        st.sampled_from((ImplementationKind.OFFICIAL, ImplementationKind.ADAPTED_OFFICIAL))
    )
    status = (
        UpstreamReproductionStatus.PASSED
        if kind is ImplementationKind.ADAPTED_OFFICIAL
        else draw(st.sampled_from(tuple(UpstreamReproductionStatus)))
    )
    generated_notes = draw(
        st.lists(_IDENTIFIERS, min_size=1, max_size=4, unique=True)
    )
    adaptation_notes = tuple(f"adapt-{item}" for item in generated_notes)
    ignored_modalities = tuple(
        sorted(
            draw(st.lists(_IDENTIFIERS, max_size=4, unique=True)),
            key=lambda value: value.encode("utf-8"),
        )
    )
    model_key = f"model-{token}"
    provenance = _provenance(
        token=token,
        model_key=model_key,
        status=status,
        adaptation_notes=adaptation_notes,
        ignored_modalities=ignored_modalities,
    )
    registration = _registration(
        token=token,
        kind=kind,
        display_name=(
            "Generated adapted official model"
            if kind is ImplementationKind.ADAPTED_OFFICIAL
            else "Generated official model"
        ),
        provenance=provenance,
    )
    return registration, provenance


@settings(max_examples=100, deadline=None)
@given(case=_official_cases())
def test_official_and_adapted_registrations_preserve_complete_provenance(
    case: tuple[ModelRegistration, ModelProvenance],
) -> None:
    """Every required upstream, reproduction, and adaptation field survives registration."""

    registration, provenance = case
    registry = ModelRegistry()
    registry.register(registration)

    stored = registry.resolve(registration.model_key).provenance
    assert stored is provenance
    assert stored is not None
    assert stored.upstream_repository == provenance.upstream_repository
    assert stored.upstream_commit == provenance.upstream_commit
    assert stored.license_reference == provenance.license_reference
    assert stored.source_hash == provenance.source_hash
    assert stored.local_patch_hash == provenance.local_patch_hash
    assert stored.dependency_environment_hash == provenance.dependency_environment_hash
    assert stored.upstream_reproduction_status is provenance.upstream_reproduction_status
    assert stored.upstream_reproduction_record == provenance.upstream_reproduction_record
    assert stored.adaptation_notes == provenance.adaptation_notes
    assert stored.ignored_modalities == provenance.ignored_modalities

    if registration.implementation_kind is ImplementationKind.ADAPTED_OFFICIAL:
        decision = registry.approve_adaptation(registration.model_key)
        assert decision.approved is True
        assert decision.formal_project_comparison_allowed is True
        assert all(decision.prerequisite_checks.values())


@settings(max_examples=100, deadline=None)
@given(token=_IDENTIFIERS, ignored=st.lists(_IDENTIFIERS, max_size=4, unique=True))
def test_adaptation_without_passing_record_is_blocked_atomically(
    token: str,
    ignored: list[str],
) -> None:
    """Missing reproduction evidence blocks adaptation and preserves registry state."""

    model_key = f"model-{token}"
    provenance = _provenance(
        token=token,
        model_key=model_key,
        status=UpstreamReproductionStatus.NOT_RUN,
        adaptation_notes=(f"adapt-{token}",),
        ignored_modalities=tuple(sorted(ignored, key=lambda value: value.encode("utf-8"))),
        include_record=False,
    )
    registration = _registration(
        token=token,
        kind=ImplementationKind.ADAPTED_OFFICIAL,
        display_name="Generated adapted official model",
        provenance=provenance,
    )
    registry = ModelRegistry()
    before = registry.snapshot()

    with pytest.raises(ReproductionGateError) as caught:
        registry.register(registration)

    error = caught.value.error_record
    assert error.error_code == "UPSTREAM_REPRODUCTION_RECORD_MISSING"
    assert error.affected_identifiers == {"model_key": model_key}
    assert error.failing_field == "upstream_reproduction_record"
    assert error.invariant_checks["registered_upstream_artifacts_unchanged"] is True
    assert error.invariant_checks["completed_runs_unchanged"] is True
    assert registry.snapshot() == before


@settings(max_examples=100, deadline=None)
@given(
    token=_IDENTIFIERS,
    proposed_status=st.sampled_from(tuple(ClaimStatus)),
    attach_metadata=st.booleans(),
    label_prefix=st.sampled_from(("compact", "inspired", "third-party", "generated")),
)
def test_proxy_metadata_never_raises_claim_above_proxy_only(
    token: str,
    proposed_status: ClaimStatus,
    attach_metadata: bool,
    label_prefix: str,
) -> None:
    """Proxy identity controls the ceiling regardless of attached metadata or score status."""

    model_key = f"model-{token}"
    provenance = None
    if attach_metadata:
        provenance = _provenance(
            token=token,
            model_key=model_key,
            status=UpstreamReproductionStatus.PASSED,
            adaptation_notes=(f"metadata-{token}",),
            ignored_modalities=(),
        )
    registration = _registration(
        token=token,
        kind=ImplementationKind.PROXY,
        display_name=f"{label_prefix} proxy baseline",
        provenance=provenance,
    )
    registry = ModelRegistry()
    registry.register(registration)

    observed = registry.claim_status_ceiling(model_key, proposed_status)
    precedence = tuple(ClaimStatus)
    expected = (
        ClaimStatus.PROXY_ONLY
        if precedence.index(proposed_status) < precedence.index(ClaimStatus.PROXY_ONLY)
        else proposed_status
    )
    assert observed is expected
    assert precedence.index(observed) >= precedence.index(ClaimStatus.PROXY_ONLY)
    assert registry.resolve(model_key).identity.implementation_kind is ImplementationKind.PROXY


@settings(max_examples=100, deadline=None)
@given(token=_IDENTIFIERS, display_name=_NON_PROXY_LABELS)
def test_proxy_display_name_requires_proxy_as_a_word(
    token: str,
    display_name: str,
) -> None:
    """Arbitrary proxy metadata cannot conceal the required proxy label."""

    registration = _registration(
        token=token,
        kind=ImplementationKind.PROXY,
        display_name=display_name,
        provenance=None,
    )
    registry = ModelRegistry()

    with pytest.raises(ModelRegistryError) as caught:
        registry.register(registration)

    assert caught.value.error_record.error_code == "PROXY_DISPLAY_NAME_MISSING_LABEL"
    assert caught.value.error_record.failing_field == "display_name"
    assert registry.snapshot() == ()
