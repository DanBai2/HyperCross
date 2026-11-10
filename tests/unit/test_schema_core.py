"""Focused unit coverage for task 1.2 shared schemas."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from benchmark.schema import (
    ArtifactReference,
    ContentHash,
    ErrorRecord,
    LimitEnforcement,
    ResourceBudget,
    ResourceLimit,
    SampleId,
    ValidationSeverity,
    validate_schema,
)


HASH = "a" * 64


def test_identifier_and_hash_primitives_validate_and_normalize_hash_case() -> None:
    assert TypeAdapter(SampleId).validate_python("sample-01") == "sample-01"
    assert TypeAdapter(ContentHash).validate_python("A" * 64) == HASH

    with pytest.raises(ValidationError):
        TypeAdapter(SampleId).validate_python("  ")
    with pytest.raises(ValidationError):
        TypeAdapter(ContentHash).validate_python("not-a-sha256")


def test_error_record_rejects_unknown_fields_and_non_finite_details() -> None:
    common = {
        "error_code": "sample_count_mismatch",
        "failed_stage": "manifest_validation",
        "failing_field": "sample_count",
    }

    with pytest.raises(ValidationError) as unknown:
        ErrorRecord.model_validate({**common, "undeclared": True})
    assert unknown.value.errors()[0]["type"] == "extra_forbidden"

    with pytest.raises(ValidationError) as non_finite:
        ErrorRecord.model_validate({**common, "observed_value": {"count": float("nan")}})
    assert non_finite.value.errors()[0]["loc"] == ("observed_value",)


@pytest.mark.parametrize("value", [0, -1, 1.5, float("inf"), True])
def test_resource_budget_rejects_non_positive_or_non_integer_values(value: object) -> None:
    candidate = {
        "limits": [
            {
                "field": "training_steps",
                "value": value,
                "unit": "steps",
                "enforcement": "hard",
            }
        ]
    }

    model, report = validate_schema(ResourceBudget, candidate)

    assert model is None
    assert not report.passes
    assert report.errors
    assert report.errors[0].field_path[:3] == ("limits", "0", "value")


def test_resource_budget_rejects_empty_and_duplicate_limit_sets() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        ResourceBudget(limits=())

    duplicate = ResourceLimit(
        field="completed_epochs",
        value=10,
        unit="count",
        enforcement=LimitEnforcement.HARD,
    )
    with pytest.raises(ValidationError, match="duplicate fields: completed_epochs"):
        ResourceBudget(limits=(duplicate, duplicate))


def test_resource_budget_accepts_explicit_limit_kind_and_rejects_illegal_enum() -> None:
    budget = ResourceBudget.model_validate(
        {
            "limits": [
                {
                    "field": "peak_device_memory",
                    "value": 16,
                    "unit": "GiB",
                    "enforcement": "advisory",
                }
            ]
        }
    )
    assert budget.limits[0].enforcement is LimitEnforcement.ADVISORY

    model, report = validate_schema(
        ResourceLimit,
        {
            "field": "epochs",
            "value": 10,
            "unit": "epochs",
            "enforcement": "sometimes",
        },
    )
    assert model is None
    assert report.errors[0].field_path == ("enforcement",)
    assert report.errors[0].code == "enum"


def test_validation_report_exposes_machine_readable_details() -> None:
    _, report = validate_schema(
        ResourceLimit,
        {
            "field": "epochs",
            "value": 0,
            "unit": "epochs",
            "enforcement": "hard",
            "unexpected": "blocked",
        },
    )

    details = report.machine_details()
    assert details["passes"] is False
    assert all(issue["severity"] == ValidationSeverity.ERROR.value for issue in details["issues"])
    assert {tuple(issue["field_path"]) for issue in details["issues"]} == {
        ("value",),
        ("unexpected",),
    }
    assert all({"code", "message", "field_path", "observed_value"} <= issue.keys() for issue in details["issues"])


def test_error_timestamps_must_be_utc_and_artifact_references_are_portable() -> None:
    with pytest.raises(ValidationError):
        ErrorRecord(
            error_code="checkpoint_hash_mismatch",
            failed_stage="checkpoint_validation",
            failing_field="content_hash",
            created_utc=datetime(2027, 1, 1, tzinfo=timezone(timedelta(hours=1))),
        )

    reference = ArtifactReference(
        artifact_id="prediction-fold-1",
        relative_path="predictions/fold 1.json",
        content_hash=HASH,
        media_type="application/json",
    )
    assert reference.relative_path == "predictions/fold 1.json"

    with pytest.raises(ValidationError):
        ArtifactReference(
            artifact_id="prediction-fold-1",
            relative_path=r"C:\\private\\prediction.json",
            content_hash=HASH,
        )
