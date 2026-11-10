"""Property 2: manifest completeness and failure atomicity.

**Validates: Requirements 2.1-2.5, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20**
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pytest
from hypothesis import given, strategies as st

from benchmark.catalog import DatasetCatalog, ManifestMaterializationError
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    DatasetSpec,
    LabelSchema,
    LabelValueKind,
    SampleRecord,
    TaskKind,
)


_FAILURE_KINDS = st.sampled_from(
    ("duplicate", "missing_group", "bad_labels", "count_mismatch", "unreadable_input")
)
_PAYLOADS = st.lists(st.binary(min_size=1, max_size=128), min_size=2, max_size=6)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class _Adapter:
    spec: DatasetSpec
    records: tuple[SampleRecord, ...]

    def iter_samples(self) -> Iterable[SampleRecord]:
        yield from self.records


class _MemoryInputHasher:
    def __init__(self, inputs: dict[str, bytes]) -> None:
        self._inputs = inputs

    def __call__(self, reference: str) -> str:
        try:
            content = self._inputs[reference]
        except KeyError as exc:
            raise FileNotFoundError(reference) from exc
        return _sha256(content)


def _spec(sample_count: int) -> DatasetSpec:
    return DatasetSpec(
        dataset_key="property2-synthetic-leaf",
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        expected_sample_count=sample_count,
        label_schema=LabelSchema(
            schema_key="leaf.targets.v1",
            value_kind=LabelValueKind.CONTINUOUS,
            dimension=3,
            target_names=("GrainWeight", "Gsw", "PhiPS2"),
        ),
        grouped=True,
    )


def _records(payloads: list[bytes]) -> tuple[tuple[SampleRecord, ...], dict[str, bytes]]:
    inputs = {
        f"inputs/sample-{index}.bin": payload
        for index, payload in enumerate(payloads)
    }
    records = tuple(
        SampleRecord(
            sample_id=f"sample-{index}",
            group_id=f"variety-{index % 3}",
            label_reference=f"labels/sample-{index}.json",
            input_references={"cube": reference},
            input_content_hashes={"cube": _sha256(inputs[reference])},
            label_schema_metadata={"schema_key": "leaf.targets.v1", "dimension": 3},
        )
        for index, reference in enumerate(inputs)
    )
    return records, inputs


def _inject_failure(
    records: tuple[SampleRecord, ...], failure_kind: str
) -> tuple[tuple[SampleRecord, ...], str, str]:
    invalid = list(records)
    affected_id = records[-1].sample_id

    if failure_kind == "duplicate":
        affected_id = records[0].sample_id
        invalid[-1] = invalid[-1].model_copy(update={"sample_id": affected_id})
        return tuple(invalid), "MANIFEST_DUPLICATE_SAMPLE_ID", "sample_id"
    if failure_kind == "missing_group":
        invalid[-1] = invalid[-1].model_copy(update={"group_id": None})
        return tuple(invalid), "MANIFEST_GROUP_ID_MISSING", "group_id"
    if failure_kind == "bad_labels":
        invalid[-1] = invalid[-1].model_copy(
            update={
                "label_schema_metadata": {
                    "schema_key": "leaf.targets.v1",
                    "dimension": 2,
                }
            }
        )
        return tuple(invalid), "MANIFEST_LABEL_DIMENSION_INVALID", "label_dimension"
    if failure_kind == "count_mismatch":
        return tuple(invalid[:-1]), "MANIFEST_SAMPLE_COUNT_MISMATCH", "sample_count"

    invalid[-1] = invalid[-1].model_copy(
        update={
            "input_references": {"cube": "inputs/does-not-exist.bin"},
            "input_content_hashes": {"cube": _sha256(b"unavailable")},
        }
    )
    return tuple(invalid), "MANIFEST_REQUIRED_INPUT_UNREADABLE", "input_reference"


def _completed_artifact_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".errors" not in path.parts
    }


@given(failure_kind=_FAILURE_KINDS, payloads=_PAYLOADS)
def test_generated_invalid_manifests_block_protocol_and_preserve_completed_artifacts(
    failure_kind: str,
    payloads: list[bytes],
) -> None:
    """All completeness failures are rejected before protocol creation or mutation."""

    with TemporaryDirectory(prefix="property2-") as directory:
        root = Path(directory) / "artifact root"
        records, inputs = _records(payloads)
        spec = _spec(len(records))
        store = FailureAtomicArtifactStore(root)
        catalog = DatasetCatalog(store, input_hasher=_MemoryInputHasher(inputs))

        baseline = catalog.materialize_manifest(
            _Adapter(spec, records),
            created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        )
        store.publish_bytes(
            artifact_id="property2-existing-protocol",
            relative_path="protocols/existing.json",
            content=b'{"status":"complete"}',
            media_type="application/json",
        )
        completed_before = _completed_artifact_snapshot(root)
        baseline_path = (
            root
            / "manifests"
            / spec.dataset_key
            / f"{baseline.dataset_hash}.json"
        )
        baseline_bytes = baseline_path.read_bytes()

        invalid_records, expected_code, expected_field = _inject_failure(
            records, failure_kind
        )
        protocol_creation_calls = 0

        def materialize_then_create_protocol() -> None:
            nonlocal protocol_creation_calls
            catalog.materialize_manifest(_Adapter(spec, invalid_records))
            protocol_creation_calls += 1

        with pytest.raises(ManifestMaterializationError) as caught:
            materialize_then_create_protocol()

        failure = caught.value
        assert failure.protocol_creation_allowed is False
        assert protocol_creation_calls == 0
        assert len(failure.error_records) == 1

        error = failure.error_records[0]
        assert error.error_code == expected_code
        assert error.failing_field == expected_field
        assert error.invariant_checks == {
            "protocol_creation_blocked": True,
            "previous_manifests_unchanged": True,
            "new_manifest_unpublished": True,
        }

        if failure_kind == "duplicate":
            assert error.unexpected_ids == (records[0].sample_id,)
        elif failure_kind == "missing_group":
            assert error.missing_ids == (records[-1].sample_id,)
        elif failure_kind == "bad_labels":
            assert error.expected_value == 3
            assert error.observed_value == {
                "sample_dimensions": {records[-1].sample_id: 2}
            }
        elif failure_kind == "count_mismatch":
            assert error.expected_value == len(records)
            assert error.observed_value == len(records) - 1
        else:
            unreadable = error.observed_value["inputs"]
            assert unreadable == [
                {
                    "sample_id": records[-1].sample_id,
                    "input_name": "cube",
                    "input_reference": "inputs/does-not-exist.bin",
                    "read_error_code": "FileNotFoundError",
                }
            ]

        assert baseline_path.read_bytes() == baseline_bytes
        assert _completed_artifact_snapshot(root) == completed_before
        assert not list(root.rglob("*.tmp"))
