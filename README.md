# HyperCross — Scientific Hyperspectral Representation Benchmark

A protocol-first benchmark and experiment suite for **small-sample scientific
hyperspectral imaging**, covering two independent scientific domains:

- **HyperLeaf2024** — whole-leaf hyperspectral cubes (204 bands x 48 x 352) + RGB,
  4 wheat varieties, 3 regression targets (GrainWeight, Gsw, PhiPS2).
- **HistologyHSI-GB** — 13-patient glioma ROI classification (826-band ENVI cube + RGB).

The suite exists to answer one question honestly: **what actually makes scientific
HSI transfer across groups (varieties / patients)?** Everything here is evaluated
under group-disjoint protocols (leave-one-variety-out, leave-one-patient-out), never
random splits, with all fitted state (normalization, PCA, target scaling, thresholds,
temperature) estimated from training folds only.

---

## Headline findings

### 1. The dominant mechanism is spectral representation design, not fusion or alignment

Leave-one-variety-out on HyperLeaf2024, 3 seeds, pooled out-of-fold R^2 in original
target units (`runs_v3/wave11`, `runs_v3/wave12`):

| spectral representation | GrainWeight | Gsw | PhiPS2 | delta(DANN - clean) on Gsw |
|---|---:|---:|---:|---:|
| per-fold per-band z-score (`fold-z`) | 0.6375 | **-0.1191** | 0.4144 | **+0.1714** |
| **per-sample SNV (`snv`)** | 0.6252 | **+0.1075** | **0.4800** | **+0.0062** |
| SNV + first derivative | 0.5714 | -0.1644 | 0.4875 | +0.0845 |
| SNV + continuum removal | 0.4982 | -0.1651 | 0.4535 | +0.1435 |
| MSC (raw-spectrum scatter correction) | 0.6034 | -0.0843 | 0.4539 | +0.0781 |
| SNV + 32-D PCA | 0.2557 | -0.3088 | 0.4284 | — |

**Per-sample SNV moves cross-variety Gsw from -0.1191 to +0.1075 (+0.2266 R^2)** with
3-seed standard deviation <= 0.0016. It dominates every other representation tested,
including MSC, derivatives, continuum removal, and PCA latents.

### 2. A spectrum-only baseline beats the entire multimodal candidate family

| model | protocol | GrainWeight | Gsw | PhiPS2 |
|---|---|---:|---:|---:|
| governed task-24.2 winner `rgb-pca-consistency` | nested LOVO, 3 seeds | 0.5070 | 0.0756 | 0.4670 |
| best governed candidate per target | nested LOVO, 3 seeds | 0.6227 | 0.1013 | 0.4737 |
| **spectrum-only SNV MLP, 200 epochs** | LOVO, 3 seeds | 0.6252 | **0.1075** | **0.4800** |
| **spectrum-only SNV MLP, 400 epochs** | LOVO, 3 seeds | **0.6415** | 0.0958 | 0.4759 |

The SNV result replicates across **two independent code paths** — the governed
runner (128 bands, `_SpectrumEncoder(64)`, AdamW lr 2e-4: Gsw 0.1016 / PhiPS2 0.4781)
and the exploratory wave scripts (204 bands, MLP(256), AdamW lr 1e-3: 0.1075 / 0.4800).

### 3. Domain-adversarial alignment is conditionally useful at best

The sign of the DANN gain flips when **only the normalization switch changes**: it is
+0.1714 where the representation still carries a removable amplitude nuisance
(`fold-z`) and +0.0062 once per-sample SNV removes it. In Histology, where patient
identity is entangled with the label, alignment is actively harmful
(BA 0.8286 clean -> 0.7909 with DANN; 0.4590 for full-fusion DANN).

Working rule: **delta(DANN - clean) scales with how weak the clean model is.**
Adversarial alignment backfills nuisance that a better representation removes for free.

### 4. Reproducible negative results

| mechanism | verdict | evidence location |
|---|---|---|
| real token-level cross-attention | no gain vs matched `value-only` control | `runs_v3/wave1` |
| dual-view consistency fusion | collapses (GW 0.0737, Gsw -0.3427) | `runs_v3/wave2` |
| hand-crafted physics features | worse (Gsw -0.2306) | `runs_v3/wave1` |
| post-hoc calibration (temperature / Platt) | improves ECE only, never BA | `runs_v3/wave4`, `wave5` |
| per-target representation routing | worse than a fixed representation | `runs_v3/wave12` |
| full-fusion DANN | catastrophic (GW -0.4822) | `runs_v3/wave6` |
| patient-adversarial Histology | harmful (BA -0.038) | `runs_v3/wave8` |
| the legacy cross-attention module | degenerate: softmax == 1, Q/K gradients are 0 | `legacy/src/hypercross_routeA.py` |

---

## Repository layout

```
benchmark/      protocol-first governed benchmark (schemas, catalog, protocols,
                training/evaluation/selection engine, metrics, run registry, runners)
experiments/    wave1..wave12 exploratory studies (representation grids, DANN sweeps,
                routing, calibration, instrument-validity)
configs/        frozen pre-registration configs
results/
  governed/     claim-gate artefacts: task reports, promotion records, pooled OOF
  exploratory/  wave summary JSON + Markdown (runs_v3)
  legacy/       superseded results - NOT protocol-comparable, see results/legacy/README
specs/          design / requirements / task breakdown for the benchmark itself
docs/           MAINLINE_PLAN.md, EXPERIMENT_LOG.md, STATUS_FOR_CVPR2027.md
legacy/src/     pre-benchmark scripts, superseded, kept for provenance only
tests/          pytest suite (unit / property / integration / audit markers)
external/       third-party source used for reimplementation reference (see provenance)
tools/          release packaging helper
```

## Quickstart

```bash
python -m pip install -e ".[dev]"
python -m pytest -q            # GPU-free suite

# governed HyperLeaf nested-LOVO pipeline
python -m benchmark.cli --help

# exploratory representation grid (needs the cached spectra, see docs/REPRODUCING.md)
python experiments/wave11/hyperleaf_spectral_representation_grid.py
```

Datasets are **not** included. See `docs/REPRODUCING.md` for dataset access, the
expected directory layout, and how to rebuild the feature caches.

## Honest limitations

- Gsw is genuinely hard; the best cross-variety R^2 is about 0.11 in absolute terms.
- The "alignment sign is decided by the representation" rule is currently supported by
  two domains in one dataset. A synthetic entanglement sweep and a third dataset are
  required before it can be stated as a general law.
- Modern baselines (SpectralFormer / SSFTT / HybridSN / S2Mamba) have **not** yet been
  rerun under these protocols; classical spectral baselines (Ridge / PLS / RF) are
  partially covered.
- Statistical strength is 3 seeds; >= 5 seeds plus held-out-group bootstrap intervals
  are still outstanding.
- `results/legacy/` contains numbers produced under random or patient splits. They are
  **not** comparable to the governed LOVO / LOPO numbers and must never be placed
  side by side.

## Citation

If you use this benchmark, please cite the repository (see `CITATION.cff`).

## License

Code is released under the MIT License (`LICENSE`). Third-party components under
`external/` retain their upstream licenses. Datasets are governed by their own terms.
