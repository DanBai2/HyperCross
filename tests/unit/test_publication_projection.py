"""Unit coverage for safe publication projection and claim audit output."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from benchmark.reporting import (
    BenchmarkTrack,
    PublicationProjectionPolicy,
    PublicationProjector,
    PublicationSafetyError,
)


def _hash(number: int) -> str:
    return f"{number:064x}"


def _claim(
    claim_id: str,
    track: str,
    dataset_key: str,
    protocol_name: str,
    **extra: object,
) -> dict[str, object]:
    source: dict[str, object] = {
        "claim_id": claim_id,
        "track": track,
        "dataset_key": dataset_key,
        "protocol_name": protocol_name,
        "conclusion": f"Task-scoped conclusion for {dataset_key}",
        "uncertainty_estimate": "two-sided 95% CI [0.7000, 0.9000]",
        "evidence_limitation": "Limited to the declared protocol and seed set.",
        "claim_status": "primary_strict",
        "artifact_references": (
            {
                "artifact_id": f"prediction-{claim_id}",
                "relative_path": f"predictions\\{claim_id}.json",
                "content_hash": _hash(len(claim_id)),
                "media_type": "application/json",
            },
        ),
    }
    source.update(extra)
    return source


def _conclusion(track: str, dataset_key: str) -> dict[str, str]:
    return {
        "track": track,
        "dataset_key": dataset_key,
        "conclusion": f"Preserved conclusion for {dataset_key}",
    }


def test_projection_preserves_track_conclusions_and_required_claim_context() -> None:
    projector = PublicationProjector(
        PublicationProjectionPolicy(
            approved_label_names={"HistologyHSI-GB": ("diagnosis_schema",)}
        )
    )
    claims = (
        _claim(
            "leaf-claim",
            "HyperLeaf2024",
            "HyperLeaf2024",
            "leaf.lovo.nested.v1",
            destinations=("headline",),
        ),
        _claim(
            "hist-claim",
            "HistologyHSI-GB",
            "HistologyHSI-GB",
            "hist.patient-nested5.v1",
            destinations=("abstract",),
            labels={"diagnosis_schema": "binary-v1", "private_diagnosis": 1},
            raw_cube=b"private cube bytes",
            patient_mapping={"patient-1": "group-1"},
            access_token="token-value",
            client_secret="secret-value",
            environment_variables={"PATH": "private"},
            overall_score=0.99,
            local_data_path=r"G:\Datasets\PKG - HistologyHSI-GB",
        ),
        _claim(
            "remote-claim",
            "Remote_Sensing_Sanity_Track",
            "IndianPines",
            "remote.indian-pines.sanity.v1",
            claim_status="sanity_only",
        ),
    )

    output = projector.project(
        claims=claims,
        task_conclusions=(
            _conclusion("HyperLeaf2024", "HyperLeaf2024"),
            _conclusion("HistologyHSI-GB", "HistologyHSI-GB"),
            _conclusion("Remote_Sensing_Sanity_Track", "IndianPines"),
        ),
    )
    public = output.model_dump(mode="json")

    assert {item["track"] for item in public["task_conclusions"]} == {
        "HyperLeaf2024",
        "HistologyHSI-GB",
        "Remote_Sensing_Sanity_Track",
    }
    assert {item["claim_id"] for item in public["headline_claims"]} == {
        "leaf-claim",
        "remote-claim",
    }
    assert {item["claim_id"] for item in public["abstract_evidence"]} == {
        "hist-claim",
        "remote-claim",
    }
    for item in public["headline_claims"] + public["abstract_evidence"]:
        assert item["protocol_name"]
        assert item["uncertainty_estimate"]
        assert item["evidence_limitation"]
        assert item["claim_status"]
        assert all(
            not reference["relative_path"].startswith(("/", "\\"))
            and ":" not in reference["relative_path"]
            for reference in item["artifact_references"]
        )

    hist = public["abstract_evidence"][0]
    assert hist["dataset_key"] == "HistologyHSI-GB"
    assert hist["approved_labels"] == {"diagnosis_schema": "binary-v1"}
    assert hist["artifact_references"][0]["relative_path"] == (
        "predictions/hist-claim.json"
    )
    assert "overall_score" not in public
    assert public["redaction_summary"] == {
        "raw_content": 1,
        "patient_mapping": 1,
        "unapproved_labels": 1,
        "tokens": 1,
        "secrets": 1,
        "environment_dumps": 1,
        "local_absolute_paths": 1,
        "combined_overall_scores": 1,
    }


def test_audit_event_recursively_removes_sensitive_values_and_absolute_paths() -> None:
    output = PublicationProjector().project(
        claims=(
            _claim(
                "leaf-claim",
                "HyperLeaf2024",
                "HyperLeaf2024",
                "leaf.official.v1",
            ),
        ),
        task_conclusions=(
            _conclusion("HyperLeaf2024", "HyperLeaf2024"),
        ),
        audit_events=(
            {
                "event_code": "publication-created",
                "details": {
                    "dataset_key": "HyperLeaf2024",
                    "artifact_reference": "predictions/leaf.json",
                    "environment_hash": _hash(99),
                    "environment_variables": {"PATH": r"C:\private"},
                    "raw_sample_content": [1, 2, 3],
                    "access_token": "do-not-persist",
                    "secret": "do-not-persist",
                    "labels": {"target": 1.0},
                    "nested": {
                        "source": r"G:\private\cube.tif",
                        "message": "safe",
                    },
                },
            },
        ),
    )

    event = output.model_dump(mode="json")["audit_events"][0]
    assert event["details"] == {
        "dataset_key": "HyperLeaf2024",
        "artifact_reference": "predictions/leaf.json",
        "environment_hash": _hash(99),
        "nested": {
            "source": "[REDACTED_LOCAL_ABSOLUTE_PATH]",
            "message": "safe",
        },
    }
    serialized = json.dumps(event)
    assert "do-not-persist" not in serialized
    assert r"G:\\private" not in serialized
    assert output.redaction_summary.environment_dumps == 1
    assert output.redaction_summary.raw_content == 1
    assert output.redaction_summary.tokens == 1
    assert output.redaction_summary.secrets == 1
    assert output.redaction_summary.unapproved_labels == 1
    assert output.redaction_summary.local_absolute_paths == 1


@pytest.mark.parametrize(
    "dataset_key",
    (r"G:\Datasets\HyperLeaf2024", "/srv/data/histology", "file:///private/data"),
)
def test_dataset_identity_rejects_local_paths_and_uris(dataset_key: str) -> None:
    with pytest.raises(ValidationError, match="stable key, not a path or URI"):
        PublicationProjector().project(
            claims=(
                _claim(
                    "unsafe-claim",
                    "HyperLeaf2024",
                    dataset_key,
                    "leaf.official.v1",
                ),
            ),
            task_conclusions=(
                _conclusion("HyperLeaf2024", dataset_key),
            ),
        )


def test_projection_rejects_absolute_artifact_reference() -> None:
    claim = _claim(
        "unsafe-artifact",
        "HyperLeaf2024",
        "HyperLeaf2024",
        "leaf.official.v1",
    )
    claim["artifact_references"] = (
        {
            "artifact_id": "private-prediction",
            "relative_path": r"C:\private\prediction.json",
            "content_hash": _hash(5),
        },
    )

    with pytest.raises(ValidationError, match="artifact path must be relative"):
        PublicationProjector().project(
            claims=(claim,),
            task_conclusions=(
                _conclusion("HyperLeaf2024", "HyperLeaf2024"),
            ),
        )


def test_every_published_claim_requires_matching_task_conclusion() -> None:
    with pytest.raises(ValidationError, match="task-level conclusion"):
        PublicationProjector().project(
            claims=(
                _claim(
                    "hist-claim",
                    "HistologyHSI-GB",
                    "HistologyHSI-GB",
                    "hist.patient-nested5.v1",
                ),
            ),
            task_conclusions=(),
        )


def test_missing_required_claim_field_has_explicit_error() -> None:
    claim = _claim(
        "leaf-claim",
        "HyperLeaf2024",
        "HyperLeaf2024",
        "leaf.official.v1",
    )
    del claim["uncertainty_estimate"]

    with pytest.raises(PublicationSafetyError, match="uncertainty_estimate"):
        PublicationProjector().project(
            claims=(claim,),
            task_conclusions=(
                _conclusion("HyperLeaf2024", "HyperLeaf2024"),
            ),
        )
