# Reproducing the results

## 1. Datasets (not redistributed)

| dataset | what is needed | where to get it |
|---|---|---|
| HyperLeaf2024 | `train.csv` + `images/<id>.tiff` hyperspectral cubes (204 x 48 x 352) | the HyperLeaf2024 benchmark release |
| HistologyHSI-GB | 13-patient ROI crops: `rgb.png` + 826-band ENVI cube per ROI | the HistologyHSI-GB release |

Expected layout on the audited machine (see the portability note below):

```
G:\Datasets\HyperLeaf2024\
    train.csv
    images\00000.tiff ...
G:\Datasets\PKG - HistologyHSI-GB\
    ... per-ROI rgb.png / ENVI cube pairs ...
G:\Datasets\HyperCross\        <- the writable benchmark workspace
```

## 2. Portability note (important)

The audited revision hard-codes `G:\Datasets` in the source. Before running on your
own machine, retarget it:

```powershell
powershell -ExecutionPolicy Bypass -File tools/relocate_data_root.ps1 -NewRoot D:\datasets
```

This rewrites the `G:\Datasets` literal in every `*.py` outside `external/` and prints
what it touched. The archived artefacts under `results/governed/` still record the
original absolute paths, and `benchmark/cli.py` enforces that the workspace directory
is named `HyperCross`.

## 3. Environment

```bash
python -m pip install -e ".[dev]"
```

Reported numbers were produced on Python 3.10.20 with `torch 2.11.0+cu128` on a single
NVIDIA RTX 5080 (16 GB). The exact lock of the governed runs is archived at
`results/governed/runs_v2/remediation/v1/deterministic-float32/hyperleaf-nested/executions/task-24-2-20261027T101633949873Z/environment-lock.json`.

All experiments are deterministic in float32 and tiny by modern standards: the
exploratory waves train a few-hundred-parameter MLP on a 204-dimensional mean spectrum,
so a complete wave finishes in seconds on a GPU and remains feasible on CPU.

## 4. Rebuilding the feature cache

Waves 10-12 read a cached mean spectrum + RGB tensor. It is not committed (it is
derived from the licensed datasets). Rebuild it by running wave 1 once:

```bash
python experiments/wave1/hyperleaf_lovo_screen.py --out experiments/wave1/results
```

The first call builds `experiments/wave1/cache_leaf_spec_rgb.npz`
(1590 x 204 mean spectrum + 3 x 32 x 64 RGB, ~4.4 MB). Every subsequent wave reads it.

## 5. Command reference

Exploratory waves (outputs land in `results/exploratory/runs_v3/...` equivalents):

```bash
python experiments/wave1/hyperleaf_lovo_screen.py --out <dir>          # screen + cache
python experiments/wave3/histology_lopo_screen.py                      # Histology LOPO
python experiments/wave6/hyperleaf_dann_sweep.py                       # lambda sweep
python experiments/wave10/hyperleaf_representation_sensitivity.py      # 4-variant grid
python experiments/wave11/hyperleaf_spectral_representation_grid.py    # 8 representations
python experiments/wave12/hyperleaf_representation_routing.py          # routing + budget
```

Defaults are `seeds = 7, 42, 123` unless overridden.

Governed pipelines:

```bash
python -m benchmark.cli --help
python -m benchmark.runners.hyperleaf_nuisance_alignment --freeze     # task 25.x
python -m benchmark.runners.hyperleaf_nuisance_alignment --execute \
    --epochs 200 --seeds 7 42 123 \
    --arms control::spectral-only::mse,arm::spectral-dann-lam1.0
```

Tests:

```bash
python -m pytest -q                    # GPU-free suite (default markers)
python -m pytest -q -m "gpu or slow"   # requires the audited CUDA environment
```

## 6. Mapping results to commands

| reported result | artefact | produced by |
|---|---|---|
| HyperLeaf task-24.2 nested LOVO | `results/governed/.../task-24.2-report.json` | `benchmark.runners.hyperleaf_candidates` / `hyperleaf_nested_remediation` |
| Histology task-15.13 LOPO | `results/governed/.../task-15.13-report.json` | `benchmark.runners.histology_strict` |
| task-25.1 spectral arms | `results/governed/.../task-25.1-report.json` | `benchmark.runners.hyperleaf_nuisance_alignment` |
| representation grids (waves 10-12) | `results/exploratory/runs_v3/wave1{0,1,2}/*` | `experiments/wave1{0,1,2}/*.py` |
| Histology calibration, patient DANN | `results/exploratory/runs_v3/wave{4,5,8}/*` | `experiments/wave{4,5,8}/*.py` |

## 7. Reporting discipline

- `benchmark/` output is the evidence tier; `experiments/wave*` output is development
  tier and must be re-computed through the benchmark before it is claimed in a paper.
- Never mix `results/legacy/` numbers (random / patient splits) with the governed
  leave-one-variety-out or leave-one-patient-out numbers.
- The official HyperLeaf2024 test budget (`official_evaluation_budget_consumed`) is 0
  and must stay 0 until a single final one-shot evaluation is intended.
