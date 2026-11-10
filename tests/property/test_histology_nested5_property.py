"""Property 17 coverage for task 15.2 nested patient isolation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from benchmark.protocols import HistologyNested5Planner, ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import SampleManifest, SampleRecord, canonical_sample_order


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(roi_counts=st.lists(st.integers(min_value=1, max_value=8), min_size=13, max_size=13))
def test_generated_nested_folds_never_split_a_patient(
    roi_counts: list[int], tmp_path: Path
) -> None:
    """**Validates: Requirements 9.1, 9.2, 9.3, 9.4**"""

    records = tuple(
        SampleRecord(
            sample_id=f"patient-{patient:02d}/roi-{roi:03d}",
            group_id=f"patient-{patient:02d}",
            label_reference="synthetic-label",
            input_references={"cube": f"patient-{patient:02d}/roi-{roi:03d}.bin"},
            input_content_hashes={"cube": f"{patient * 1000 + roi:064x}"},
            label_schema_metadata={"dimension": 1, "label_value": roi % 2},
        )
        for patient, count in enumerate(roi_counts, start=1)
        for roi in range(1, count + 1)
    )
    manifest = SampleManifest(
        schema_version="1.0",
        dataset_key="histology-hsi-gb.generated",
        records=records,
        canonical_order=canonical_sample_order(tuple(item.sample_id for item in records)),
        dataset_hash="e" * 64,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    planner = HistologyNested5Planner(
        ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts"))
    )
    folds = planner.build_folds(manifest)
    expected_ids = set(manifest.canonical_order)
    all_outer_test_ids: set[str] = set()

    for outer in folds:
        outer_train = set(outer.outer_train_ids)
        outer_test = set(outer.outer_test_ids)
        outer_train_groups = {item.split("/", 1)[0] for item in outer_train}
        outer_test_groups = {item.split("/", 1)[0] for item in outer_test}
        assert outer_train.isdisjoint(outer_test)
        assert outer_train | outer_test == expected_ids
        assert outer_train_groups.isdisjoint(outer_test_groups)
        assert all_outer_test_ids.isdisjoint(outer_test)
        all_outer_test_ids.update(outer_test)

        all_inner_validation_ids: set[str] = set()
        for inner in outer.inner_folds:
            inner_train = set(inner.train_ids)
            inner_validation = set(inner.validation_ids)
            inner_train_groups = {item.split("/", 1)[0] for item in inner_train}
            inner_validation_groups = {
                item.split("/", 1)[0] for item in inner_validation
            }
            assert inner_train.isdisjoint(inner_validation)
            assert inner_train | inner_validation == outer_train
            assert inner_train_groups.isdisjoint(inner_validation_groups)
            assert all_inner_validation_ids.isdisjoint(inner_validation)
            all_inner_validation_ids.update(inner_validation)
        assert all_inner_validation_ids == outer_train

    assert all_outer_test_ids == expected_ids
