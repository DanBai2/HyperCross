"""Property 26: Four-decimal cross-format consistency.

**Validates: Requirements 12.1-12.10**
"""

from __future__ import annotations

import csv
import io
import json
import re
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

from hypothesis import given, settings, strategies as st

from benchmark.registry import (
    AggregateRecord,
    ArtifactRecord,
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    ModelIdentity,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.reporting import PublicationSource, ReportGenerator
from benchmark.schema import (
    ArtifactReference,
    ClaimStatus,
    ImplementationKind,
    LimitEnforcement,
    ResourceBudget,
    ResourceLimit,
    RunStatus,
    RuntimeMeasurements,
)


_NOW = datetime(2027, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
_FOUR_DECIMALS = re.compile(r"^-?\d+\.\d{4}$")
_BOUNDARY_FLOATS = st.one_of(
    st.sampled_from(
        (
            -0.0,
            0.0,
            -0.00005,
            0.00005,
            -0.00015,
            0.00015,
            1.23445,
            1.23455,
            -1.23445,
            -1.23455,
            999_999_999.99995,
            -999_999_999.99995,
        )
    ),
    st.integers(min_value=-10**10, max_value=10**10).map(
        lambda numerator: float(Decimal(numerator) / Decimal("100000"))
    ),
    st.floats(
        min_value=-1_000_000_000,
        max_value=1_000_000_000,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
_SAFE_TOKEN = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=12,
)


def _hash(number: int) -> str:
    return f"{number:064x}"


def _expected_decimal4(value: float | int) -> str:
    rounded = Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    if rounded == 0:
        rounded = abs(rounded)
    return format(rounded, ".4f")


def _append_publication_records(
    registry: RunRegistry,
    *,
    token: str,
    metric_value: float,
    parameter_count: int,
    protocol_hash: str,
    claim_status: ClaimStatus,
) -> tuple[str, str, str]:
    model_key = f"model.{token}.v1"
    protocol_name = f"protocol.{token}.v1"
    identity = ModelIdentity(
        model_key=model_key,
        display_name=f"Model {token}",
        class_fully_qualified_name=f"tests.models.Model{token}",
        implementation_kind=ImplementationKind.IN_REPO,
        parameter_count=parameter_count,
        source_revision=f"revision-{token}",
    )
    contract = InputContract(
        representation_key=f"representation.{token}.v1",
        modalities=("image",),
        tensor_shapes={"image": (3, 8, 8)},
        dtypes={"image": "float32"},
        channel_or_band_order={"image": ("r", "g", "b")},
        preprocessing_key="normalize.v1",
    )
    claim = ClaimDecision(
        status=claim_status,
        supporting_reasons=(f"support-{token}",),
        prerequisite_checks={f"check-{token}": True},
    )
    frozen = FrozenRunIdentity(
        dataset_hash=_hash(1),
        protocol_hash=protocol_hash,
        model_key=model_key,
        config_hash=_hash(2),
        fold_id=f"fold-{token}",
        seed=7,
        environment_lock_hash=_hash(3),
        environment_compatibility_policy="exact",
    )
    provenance = Provenance(
        code_revision=f"revision-{token}",
        source_hash=_hash(4),
        environment_hash=frozen.environment_lock_hash,
        config_hash=frozen.config_hash,
        dataset_hash=frozen.dataset_hash,
    )
    budget = ResourceBudget(
        limits=(
            ResourceLimit(
                field="completed_epochs",
                value=1,
                unit="count",
                enforcement=LimitEnforcement.HARD,
            ),
        )
    )
    planned = registry.allocate_run(
        frozen_identity=frozen,
        model_identity=identity,
        input_contract=contract,
        provenance=provenance,
        resource_budget=budget,
        claim_decision=claim,
        started_utc=_NOW,
    )
    checkpoint = ArtifactReference(
        artifact_id=f"checkpoint-{token}",
        relative_path=f"checkpoints/{token}.bin",
        content_hash=_hash(5),
    )
    prediction = ArtifactReference(
        artifact_id=f"prediction-{token}",
        relative_path=f"predictions/{token}.json",
        content_hash=_hash(6),
    )
    for artifact, kind, offset in (
        (checkpoint, "checkpoint", 1),
        (prediction, "prediction", 2),
    ):
        registry.append_artifact(
            ArtifactRecord(
                artifact=artifact,
                artifact_kind=kind,
                run_id=planned.run_id,
                created_utc=_NOW + timedelta(seconds=offset),
            )
        )
    complete = RunRecord.model_validate(
        {
            **planned.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "checkpoint_ref": checkpoint,
            "prediction_artifact_ref": prediction,
            "runtime_measurements": RuntimeMeasurements(
                completed_epochs=1,
                completed_training_steps=1,
                wall_clock_duration_ms=1,
                peak_device_memory_bytes=1,
                device_identity="cpu",
                numeric_precision="float32",
                parameter_count=parameter_count,
                input_volume_bytes=1,
            ),
            "completed_utc": _NOW + timedelta(seconds=3),
        }
    )
    registry.append_run(complete)

    aggregate_id = f"aggregate-{token}"
    registry.append_aggregate(
        AggregateRecord(
            aggregate_id=aggregate_id,
            member_run_ids=(complete.run_id,),
            prediction_hashes={complete.run_id: prediction.content_hash},
            dataset_hash=frozen.dataset_hash,
            protocol_hash=protocol_hash,
            model_key=model_key,
            representation_key=contract.representation_key,
            metrics={
                "boundary": metric_value,
                "nested": {"count": parameter_count},
            },
            artifact_refs=(prediction,),
            claim_decision=claim,
            provenance=provenance,
            created_utc=_NOW + timedelta(minutes=1),
        )
    )
    return aggregate_id, protocol_name, model_key


def _parse_markdown(payload: bytes, columns: tuple[str, ...]) -> dict[str, str]:
    lines = payload.decode("utf-8").splitlines()
    header = tuple(cell.strip() for cell in lines[0][1:-1].split("|"))
    values = tuple(cell.strip() for cell in lines[2][1:-1].split("|"))
    assert header == columns
    return dict(zip(header, values))


@settings(max_examples=100, deadline=None)
@given(
    metric_value=_BOUNDARY_FLOATS,
    parameter_count=st.integers(min_value=0, max_value=10**9),
    token=_SAFE_TOKEN,
    protocol_number=st.integers(min_value=10, max_value=10**6),
    claim_status=st.sampled_from(tuple(ClaimStatus)),
)
def test_registry_backed_publication_cells_are_identical_and_fixed_to_four_decimals(
    metric_value: float,
    parameter_count: int,
    token: str,
    protocol_number: int,
    claim_status: ClaimStatus,
) -> None:
    with tempfile.TemporaryDirectory(prefix="property26-") as temporary:
        registry = RunRegistry(Path(temporary) / "registry.jsonl")
        protocol_hash = _hash(protocol_number)
        aggregate_id, protocol_name, model_key = _append_publication_records(
            registry,
            token=token,
            metric_value=metric_value,
            parameter_count=parameter_count,
            protocol_hash=protocol_hash,
            claim_status=claim_status,
        )
        generator = ReportGenerator(registry)
        matrix = generator.build_matrix(
            PublicationSource(
                aggregate_id=aggregate_id,
                protocol_name=protocol_name,
            )
        )

        json_bytes = generator.render_json(matrix)
        csv_bytes = generator.render_csv(matrix)
        markdown_bytes = generator.render_markdown(matrix)

        assert generator.validate_cross_format(
            json_bytes, csv_bytes, markdown_bytes
        ).passes
        json_row: dict[str, Any] = json.loads(json_bytes)["rows"][0]
        csv_row = next(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))
        markdown_row = _parse_markdown(markdown_bytes, matrix.columns)

        expected_row = matrix.rows[0].cells
        assert json_row == expected_row
        assert csv_row == expected_row
        assert markdown_row == expected_row
        assert json_row["model_key"] == model_key
        assert json_row["protocol_name"] == protocol_name
        assert json_row["protocol_hash"] == protocol_hash
        assert json_row["claim_status"] == claim_status.value

        numeric_columns = (
            "parameter_count",
            "metric.boundary",
            "metric.nested.count",
        )
        assert all(_FOUR_DECIMALS.fullmatch(json_row[column]) for column in numeric_columns)
        assert json_row["parameter_count"] == _expected_decimal4(parameter_count)
        assert json_row["metric.boundary"] == _expected_decimal4(metric_value)
        assert json_row["metric.nested.count"] == _expected_decimal4(parameter_count)
