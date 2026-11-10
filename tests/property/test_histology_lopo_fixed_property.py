"""Property 17 coverage for fixed-epoch patient LOPO isolation and coverage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from benchmark.protocols import HistologyLopoFixedPlanner, ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import SampleManifest, SampleRecord, canonical_sample_order


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    roi_counts=st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=16),
    epochs=st.integers(min_value=1, max_value=400),
)
def test_generated_lopo_folds_hold_out_every_patient_exactly_once(
    roi_counts: list[int], epochs: int, tmp_path: Path
) -> None:
    """**Validates: Requirements 9.17, 9.18, 9.19, 9.39, 9.40, 9.41**"""

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
        dataset_key="histology.generated",
        records=records,
        canonical_order=canonical_sample_order(record.sample_id for record in records),
        dataset_hash="e" * 64,
        created_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    planner = HistologyLopoFixedPlanner(
        ProtocolPlanner(FailureAtomicArtifactStore(tmp_path / "artifacts")),
        epoch_policy=epochs,
    )
    protocol = planner.plan(manifest, seed_set=(7, 42, 123))

    expected_ids = set(manifest.canonical_order)
    held_out_ids: set[str] = set()
    held_out_groups: set[str] = set()
    for fold in protocol.folds:
        train_ids = set(fold.outer_train_ids)
        test_ids = set(fold.outer_test_ids)
        train_groups = {sample_id.split("/", 1)[0] for sample_id in train_ids}
        test_groups = {sample_id.split("/", 1)[0] for sample_id in test_ids}
        assert len(test_groups) == 1
        assert train_ids.isdisjoint(test_ids)
        assert train_ids | test_ids == expected_ids
        assert train_groups.isdisjoint(test_groups)
        assert held_out_ids.isdisjoint(test_ids)
        assert held_out_groups.isdisjoint(test_groups)
        held_out_ids.update(test_ids)
        held_out_groups.update(test_groups)

    assert held_out_ids == expected_ids
    assert len(held_out_groups) == len(roi_counts)
    assert protocol.selection_policy.policy_key == f"hist.fixed-epoch-{epochs:04d}.v1"
