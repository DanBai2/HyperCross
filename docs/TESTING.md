# Testing

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Marker taxonomy (see `tests/pytest_policy.py`): `unit`, `property`, `integration`,
`regression`, `smoke`, `audit`, `gpu`, `slow`, `strict`. The default run applies
`-m "not gpu and not slow and not strict"`.

`tests/pytest_policy.py` compares the active process against the audited environment
lock in `artifacts/audit/training_fast_lane_audit.json`. Pass
`--require-audited-environment` to turn the resulting GPU skips into failures.

## Reference result

Measured on a machine where the repository was **not** on the audited drive
(`--basetemp` pointed inside the checkout, since the sandbox blocked the default
temporary directory):

```
441 passed, 14 failed, 79 warnings in 367s
```

## The 14 host-bound failures are intentional

Every one of them fails because the checkout was not at `G:\Datasets\HyperCross`.
This is a deliberate guard, not a defect: the audited pipeline refuses to read
HyperLeaf data or import legacy sources from anywhere else, so that a result can never
be produced silently in an unaudited location.

Two mechanisms enforce it:

| mechanism | where | affects |
|---|---|---|
| `HyperLeaf2024Adapter._reject_disallowed_drive` — `HyperLeafDataError: HyperLeaf runtime data must be read from the G: drive` | `benchmark/catalog/hyperleaf.py` | 5 tests in `tests/unit/test_hyperleaf_adapter.py`, 3 in `tests/unit/test_hyperleaf_classic_adapter.py`, 1 property test |
| `_require_g_drive` / `g_tmp_path` fixtures assert `Path(...).resolve().drive == "G:"` | `tests/unit/test_legacy_importer.py` | 4 tests |

The fixtures materialise synthetic data under `<repo>/.tmp/...`, so their drive is
whatever drive the checkout is on. The workspace itself is additionally constrained by
`benchmark/cli.py`, which rejects any workspace root other than `G:\Datasets\HyperCross`.

If you want a green run elsewhere you have three options:

1. Put the checkout at `G:\Datasets\HyperCross` (the audited configuration).
2. Run `tools/relocate_data_root.ps1 -NewRoot <drive>:\datasets` — this rewrites the
   `G:\Datasets` literal everywhere. Note that it retargets the *data root*, and the
   drive guards reject any drive other than `G:`, so you additionally need to keep the
   data on `G:` or adapt the guard. Treat any result obtained this way as
   non-comparable with the numbers in `RESULTS.md`.
3. Deselect the host-bound files:

```bash
python -m pytest -q --deselect tests/unit/test_legacy_importer.py \
  --deselect tests/unit/test_hyperleaf_adapter.py \
  --deselect tests/unit/test_hyperleaf_classic_adapter.py
```

## GPU tests

The `gpu` and `strict` marker groups need the audited TCFM CUDA environment
(Python 3.10, torch 2.11.0+cu128, exactly one visible RTX 5080) and are excluded from
the default run. They are the tests that actually verify the reported protocol
behaviour end to end.
