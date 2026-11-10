"""Property 1: canonical dataset identity.

**Validates: Requirements 2.6-2.9, 2.21-2.23**
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from hypothesis import given, settings, strategies as st

from benchmark.catalog import DatasetCatalog
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import (
    DatasetSpec,
    LabelSchema,
    LabelValueKind,
    SampleRecord,
    TaskKind,
    canonical_serialize,
    hash_object,
    normalize_id,
)


_CREATED_UTC = datetime(2027, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _Adapter:
    spec: DatasetSpec
    records: tuple[SampleRecord, ...]

    def iter_samples(self) -> Iterable[SampleRecord]:
        yield from self.records


@dataclass(frozen=True)
class _ManifestCase:
    spec: DatasetSpec
    records: tuple[SampleRecord, ...]
    permutation: tuple[SampleRecord, ...]
    input_hashes: dict[str, str]
    mutation_index: int


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@st.composite
def _manifest_cases(draw: st.DrawFn) -> _ManifestCase:
    sample_numbers = draw(
        st.lists(
            st.integers(min_value=0, max_value=999_999),
            min_size=2,
            max_size=7,
            unique=True,
        )
    )
    payloads = draw(
        st.lists(
            st.binary(min_size=1, max_size=48),
            min_size=len(sample_numbers),
            max_size=len(sample_numbers),
        )
    )
    separators = draw(
        st.lists(
            st.sampled_from(("/", "\\")),
            min_size=len(sample_numbers),
            max_size=len(sample_numbers),
        )
    )

    records: list[SampleRecord] = []
    input_hashes: dict[str, str] = {}
    for position, (number, payload, separator) in enumerate(
        zip(sample_numbers, payloads, separators)
    ):
        sample_id = f"batch{separator}sample-{number:06d}"
        input_reference = f"inputs/cube-{position}.bin"
        content_hash = _sha256(payload)
        input_hashes[input_reference] = content_hash
        records.append(
            SampleRecord(
                sample_id=sample_id,
                group_id=f"variety-{number % 5}",
                label_reference=f"labels/target-{number:06d}.json",
                input_references={"cube": input_reference},
                input_content_hashes={"cube": content_hash},
                label_schema_metadata={
                    "schema_key": "leaf.targets.v1",
                    "dimension": 3,
                },
            )
        )

    record_tuple = tuple(records)
    permutation = draw(
        st.permutations(record_tuple).filter(lambda candidate: candidate != record_tuple)
    )
    spec = DatasetSpec(
        dataset_key="property1.synthetic-leaf",
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        expected_sample_count=len(record_tuple),
        label_schema=LabelSchema(
            schema_key="leaf.targets.v1",
            value_kind=LabelValueKind.CONTINUOUS,
            dimension=3,
            target_names=("GrainWeight", "Gsw", "PhiPS2"),
        ),
        grouped=True,
    )
    return _ManifestCase(
        spec=spec,
        records=record_tuple,
        permutation=permutation,
        input_hashes=input_hashes,
        mutation_index=draw(st.integers(min_value=0, max_value=len(record_tuple) - 1)),
    )


def _manifest_path(catalog: DatasetCatalog, dataset_hash: str) -> Path:
    return (
        catalog.artifact_store.root
        / "manifests"
        / "property1.synthetic-leaf"
        / f"{dataset_hash}.json"
    )


def _identity(manifest: object, spec: DatasetSpec) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "dataset_spec": spec,
        "records": manifest.records,
        "canonical_order": manifest.canonical_order,
    }


@settings(max_examples=100, deadline=None)
@given(case=_manifest_cases())
def test_permutations_and_version_changes_preserve_canonical_dataset_identity(
    case: _ManifestCase,
) -> None:
    """Enumeration order is irrelevant and prior manifest bytes are immutable."""

    with TemporaryDirectory(prefix="property1-") as directory:
        root = Path(directory)
        hashes = dict(case.input_hashes)
        input_hasher = lambda reference: hashes[reference]  # noqa: E731
        first_catalog = DatasetCatalog(
            FailureAtomicArtifactStore(root / "first"),
            input_hasher=input_hasher,
        )
        permuted_catalog = DatasetCatalog(
            FailureAtomicArtifactStore(root / "permuted"),
            input_hasher=input_hasher,
        )

        first = first_catalog.materialize_manifest(
            _Adapter(case.spec, case.records),
            created_utc=_CREATED_UTC,
        )
        permuted = permuted_catalog.materialize_manifest(
            _Adapter(case.spec, case.permutation),
            created_utc=_CREATED_UTC,
        )
        first_path = _manifest_path(first_catalog, first.dataset_hash)
        permuted_path = _manifest_path(permuted_catalog, permuted.dataset_hash)
        first_bytes = first_path.read_bytes()
        first_identity = _identity(first, case.spec)
        permuted_identity = _identity(permuted, case.spec)

        assert case.permutation != case.records
        assert first.canonical_order == tuple(
            sorted(
                (normalize_id(record.sample_id) for record in case.records),
                key=lambda sample_id: sample_id.encode("utf-8"),
            )
        )
        assert permuted.canonical_order == first.canonical_order
        assert canonical_serialize(permuted_identity) == canonical_serialize(first_identity)
        assert hash_object(first_identity) == hash_object(first_identity)
        assert hash_object(first_identity) == hash_object(permuted_identity)
        assert first.dataset_hash == permuted.dataset_hash == hash_object(first_identity)
        assert permuted_path.read_bytes() == first_bytes

        changed_input_records = list(case.records)
        original = changed_input_records[case.mutation_index]
        input_reference = original.input_references["cube"]
        changed_hash = _sha256(bytes.fromhex(original.input_content_hashes["cube"]) + b"changed")
        hashes[input_reference] = changed_hash
        changed_input_records[case.mutation_index] = original.model_copy(
            update={"input_content_hashes": {"cube": changed_hash}}
        )
        changed_content = first_catalog.materialize_manifest(
            _Adapter(case.spec, tuple(changed_input_records)),
            created_utc=_CREATED_UTC,
        )

        changed_label_records = list(changed_input_records)
        content_version = changed_label_records[case.mutation_index]
        changed_label_records[case.mutation_index] = content_version.model_copy(
            update={"label_reference": f"{content_version.label_reference}.v2"}
        )
        changed_label = first_catalog.materialize_manifest(
            _Adapter(case.spec, tuple(changed_label_records)),
            created_utc=_CREATED_UTC,
        )

        assert len(
            {first.dataset_hash, changed_content.dataset_hash, changed_label.dataset_hash}
        ) == 3
        assert first_path.read_bytes() == first_bytes
        assert _manifest_path(first_catalog, changed_content.dataset_hash).exists()
        assert _manifest_path(first_catalog, changed_label.dataset_hash).exists()
