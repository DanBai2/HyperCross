"""Property coverage for task-14.12 exact grouped screening evidence."""
from __future__ import annotations

import numpy as np
from hypothesis import assume, given, strategies as st

from benchmark.runners.hyperleaf_screening import compute_grouped_metrics


@given(
    values=st.lists(
        st.tuples(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        ),
        min_size=8,
        max_size=24,
    )
)
def test_exact_oof_identity_predictions_have_unit_r2_and_zero_mae(
    values: list[tuple[float, float, float]],
) -> None:
    """Exact OOF metric computation preserves every target without outer evidence.

    **Validates: Requirements 5.1, 5.7-5.12, 8.17-8.23, 10.8, 10.9**
    """

    truth = np.asarray(values, dtype=np.float64)
    assume(all(float(np.var(truth[:, index])) > 1e-12 for index in range(3)))
    sample_ids = tuple(f"sample-{index:03d}" for index in range(len(truth)))
    fold_ids = tuple(f"inner-{index % 4}" for index in range(len(truth)))

    metrics = compute_grouped_metrics(sample_ids, fold_ids, truth, truth.copy())

    for target in ("GrainWeight", "Gsw", "PhiPS2"):
        assert metrics["pooled_oof"]["targets"][target] == {"r2": 1.0, "mae": 0.0}
    assert metrics["pooled_oof"]["mean_r2"] == 1.0
    assert metrics["pooled_oof"]["mean_mae"] == 0.0
