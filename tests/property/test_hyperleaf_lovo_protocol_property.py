"""Property 16: HyperLeaf protocol separation for nested LOVO.

**Validates: Requirements 8.14-8.23, 8.28-8.30**
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings, strategies as st

from benchmark.catalog.hyperleaf import HYPERLEAF_VARIETIES
from benchmark.protocols import ProtocolPlanner, plan_hyperleaf_lovo
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.schema import SampleManifest, SampleRecord, canonical_sample_order, hash_object


_CREATED = datetime(2027, 1, 1, tzinfo=timezone.utc)
_INPUT_HASH = "d" * 64


@st.composite
def _complete_variety_manifests(draw: st.DrawFn) -> SampleManifest:
    sizes = draw(
        st.lists(
            st.integers(min_value=1, max_value=6),
            min_size=len(HYPERLEAF_VARIETIES),
            max_size=len(HYPERLEAF_VARIETIES),
        )
    )
    records = tuple(
        SampleRecord(
            sample_id=f"{variety.casefold()}-{sample_index:02d}",
            group_id=variety,
            label_reference=f"labels://hyperleaf2024/train/{variety}-{sample_index}",
            input_references={"full_cube": f"images/{variety}-{sample_index}.tiff"},
            input_content_hashes={"full_cube": _INPUT_HASH},
            label_schema_metadata={
                "dimension": 3,
                "partition": "train",
                "labels_available": True,
            },
        )
        for variety, size in zip(HYPERLEAF_VARIETIES, sizes)
        for sample_index in range(size)
    )
    permuted = draw(st.permutations(records))
    order = canonical_sample_order(record.sample_id for record in records)
    return SampleManifest(
        schema_version="1.0",
        dataset_key="hyperleaf2024",
        records=tuple(permuted),
        canonical_order=order,
        dataset_hash=hash_object(
            tuple(sorted((record.sample_id, record.group_id) for record in records))
        ),
        created_utc=_CREATED,
    )


@settings(max_examples=100, deadline=None)
@given(manifest=_complete_variety_manifests())
def test_each_complete_variety_is_exactly_one_outer_test_and_inner_selection_is_outer_train_only(
    manifest: SampleManifest,
) -> None:
    with TemporaryDirectory(prefix="hyperleaf-lovo-property-") as directory:
        planner = ProtocolPlanner(
            FailureAtomicArtifactStore(Path(directory) / "artifacts")
        )
        protocol = plan_hyperleaf_lovo(
            planner,
            manifest,
            seed_set=(42,),
            publish=False,
        )

        group_by_sample = {
            record.sample_id: record.group_id for record in manifest.records
        }
        outer_test_membership: dict[str, int] = {
            variety: 0 for variety in HYPERLEAF_VARIETIES
        }
        covered_samples: set[str] = set()
        for outer in protocol.folds:
            outer_train = set(outer.outer_train_ids)
            outer_test = set(outer.outer_test_ids)
            test_groups = {group_by_sample[sample_id] for sample_id in outer_test}
            assert len(test_groups) == 1
            held_out = next(iter(test_groups))
            outer_test_membership[held_out] += 1
            assert outer_train.isdisjoint(outer_test)
            assert outer_train | outer_test == set(manifest.canonical_order)
            assert covered_samples.isdisjoint(outer_test)
            covered_samples.update(outer_test)

            inner_validation_groups = set()
            for inner in outer.inner_folds:
                inner_train = set(inner.train_ids)
                inner_validation = set(inner.validation_ids)
                validation_groups = {
                    group_by_sample[sample_id] for sample_id in inner_validation
                }
                assert len(validation_groups) == 1
                inner_validation_groups.update(validation_groups)
                assert inner_train.isdisjoint(inner_validation)
                assert inner_train | inner_validation == outer_train
                assert held_out not in {
                    group_by_sample[sample_id]
                    for sample_id in inner_train | inner_validation
                }
            assert inner_validation_groups == set(HYPERLEAF_VARIETIES) - {held_out}

        assert outer_test_membership == {
            variety: 1 for variety in HYPERLEAF_VARIETIES
        }
        assert covered_samples == set(manifest.canonical_order)
