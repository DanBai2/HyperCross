"""Targeted unit coverage for task 2.1 dataset schemas and adapters."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from benchmark.catalog import (
    DatasetAdapter,
    HistologyDatasetAdapter,
    HyperLeafDatasetAdapter,
    RemoteSanityDatasetAdapter,
)
from benchmark.schema import (
    DatasetSpec,
    LabelSchema,
    LabelValueKind,
    SampleManifest,
    SampleRecord,
    TaskKind,
    canonical_sample_order,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def regression_schema() -> LabelSchema:
    return LabelSchema(
        schema_key="hyperleaf.targets.v1",
        value_kind=LabelValueKind.CONTINUOUS,
        dimension=3,
        target_names=("GrainWeight", "Gsw", "PhiPS2"),
    )


def binary_schema() -> LabelSchema:
    return LabelSchema(
        schema_key="histology.binary.v1",
        value_kind=LabelValueKind.BINARY,
        dimension=1,
        target_names=("diagnosis",),
        class_names=("negative", "positive"),
    )


def categorical_schema() -> LabelSchema:
    return LabelSchema(
        schema_key="remote.class-index.v1",
        value_kind=LabelValueKind.CATEGORICAL,
        dimension=1,
        target_names=("class_index",),
        class_names=("class_1", "class_2", "class_3"),
    )


def sample(sample_id: str, content_hash: str, *, group_id: str = "group-1") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        group_id=group_id,
        label_reference=f"labels/{sample_id}.json",
        input_references={"cube": f"inputs/{sample_id}.tif"},
        input_content_hashes={"cube": content_hash},
        label_schema_metadata={"schema_key": "hyperleaf.targets.v1", "dimension": 3},
    )


def test_dataset_specs_cover_each_track_without_storage_paths() -> None:
    leaf = DatasetSpec(
        dataset_key="hyperleaf2024",
        task_kind=TaskKind.MULTI_TARGET_REGRESSION,
        expected_sample_count=2410,
        label_schema=regression_schema(),
        grouped=True,
    )
    histology = DatasetSpec(
        dataset_key="histology-hsi-gb",
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        expected_sample_count=200,
        label_schema=binary_schema(),
        grouped=True,
    )
    remote = DatasetSpec(
        dataset_key="indian-pines-sanity",
        task_kind=TaskKind.REMOTE_CLASSIFICATION,
        expected_sample_count=10249,
        label_schema=categorical_schema(),
        grouped=False,
    )

    assert {leaf.task_kind, histology.task_kind, remote.task_kind} == set(TaskKind)
    assert "path" not in DatasetSpec.model_fields
    assert "root" not in DatasetSpec.model_fields


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_dataset_spec_requires_a_positive_integer_expected_count(value: object) -> None:
    with pytest.raises(ValidationError):
        DatasetSpec.model_validate(
            {
                "dataset_key": "hyperleaf2024",
                "task_kind": "multi_target_regression",
                "expected_sample_count": value,
                "label_schema": regression_schema(),
                "grouped": True,
            }
        )


def test_dataset_spec_rejects_task_and_label_semantic_mismatch() -> None:
    with pytest.raises(ValidationError, match="requires label value_kind"):
        DatasetSpec(
            dataset_key="bad-histology",
            task_kind=TaskKind.BINARY_CLASSIFICATION,
            expected_sample_count=10,
            label_schema=regression_schema(),
            grouped=True,
        )


def test_label_schema_requires_declared_dimension_and_valid_classes() -> None:
    with pytest.raises(ValidationError, match="target_names length"):
        LabelSchema(
            schema_key="bad-regression",
            value_kind=LabelValueKind.CONTINUOUS,
            dimension=2,
            target_names=("only_one",),
        )

    with pytest.raises(ValidationError, match="exactly two classes"):
        LabelSchema(
            schema_key="bad-binary",
            value_kind=LabelValueKind.BINARY,
            dimension=1,
            target_names=("diagnosis",),
            class_names=("positive",),
        )


def test_sample_record_requires_matching_input_references_and_hashes() -> None:
    valid = sample("sample-01", HASH_A)
    assert valid.input_content_hashes["cube"] == HASH_A
    assert valid.label_schema_metadata["dimension"] == 3

    with pytest.raises(ValidationError, match="exactly cover"):
        SampleRecord(
            sample_id="sample-01",
            group_id="group-1",
            label_reference="labels/sample-01.json",
            input_references={"cube": "inputs/sample-01.tif", "rgb": "inputs/sample-01.png"},
            input_content_hashes={"cube": HASH_A},
            label_schema_metadata={"schema_key": "hyperleaf.targets.v1", "dimension": 3},
        )


def test_manifest_enforces_complete_normalized_canonical_sample_order() -> None:
    records = (sample("z\\sample", HASH_A), sample("é/sample", HASH_B))
    order = canonical_sample_order(tuple(record.sample_id for record in records))
    manifest = SampleManifest(
        schema_version="1.0",
        dataset_key="hyperleaf2024",
        records=records,
        canonical_order=order,
        dataset_hash=HASH_A,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert manifest.canonical_order == tuple(
        sorted(("z/sample", "é/sample"), key=lambda item: item.encode("utf-8"))
    )

    with pytest.raises(ValidationError, match="canonical_order"):
        SampleManifest(
            schema_version="1.0",
            dataset_key="hyperleaf2024",
            records=records,
            canonical_order=tuple(reversed(order)),
            dataset_hash=HASH_A,
            created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        )


def test_manifest_rejects_duplicate_ids_after_normalization_and_is_frozen() -> None:
    duplicate_records = (
        sample("folder\\sample", HASH_A),
        sample("folder/sample", HASH_B),
    )
    with pytest.raises(ValidationError, match="unique after canonical normalization"):
        SampleManifest(
            schema_version="1.0",
            dataset_key="hyperleaf2024",
            records=duplicate_records,
            canonical_order=("folder/sample", "folder/sample"),
            dataset_hash=HASH_A,
            created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        )

    manifest = SampleManifest(
        schema_version="1.0",
        dataset_key="hyperleaf2024",
        records=(sample("sample-01", HASH_A),),
        canonical_order=("sample-01",),
        dataset_hash=HASH_A,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(ValidationError, match="frozen"):
        manifest.dataset_key = "changed"  # type: ignore[misc]


def test_track_adapter_protocols_are_interface_only_and_runtime_checkable() -> None:
    class HistologyAdapter:
        @property
        def spec(self) -> DatasetSpec:
            return DatasetSpec(
                dataset_key="histology-hsi-gb",
                task_kind=TaskKind.BINARY_CLASSIFICATION,
                expected_sample_count=1,
                label_schema=binary_schema(),
                grouped=True,
            )

        def iter_samples(self):
            return iter(())

        def iter_patient_grouped_samples(self):
            return self.iter_samples()

    class HyperLeafAdapter(HistologyAdapter):
        def iter_training_samples(self):
            return iter(())

        def iter_official_test_samples(self):
            return iter(())

    class RemoteAdapter(HistologyAdapter):
        def iter_remote_scene_samples(self):
            return self.iter_samples()

    histology = HistologyAdapter()
    leaf = HyperLeafAdapter()
    remote = RemoteAdapter()
    assert isinstance(histology, DatasetAdapter)
    assert isinstance(histology, HistologyDatasetAdapter)
    assert not isinstance(histology, RemoteSanityDatasetAdapter)
    assert isinstance(leaf, HyperLeafDatasetAdapter)
    assert isinstance(remote, RemoteSanityDatasetAdapter)
