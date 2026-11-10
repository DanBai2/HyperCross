# Results

All numbers below are **pooled out-of-fold (OOF) R^2 in original target units** for
HyperLeaf2024 and **pooled OOF accuracy / balanced accuracy / AUROC** for
HistologyHSI-GB. Seeds are `7, 42, 123`. "Governed" = produced through `benchmark/`
and gated by the claim-gate; "exploratory" = produced by `experiments/wave*` and
still development tier.

Protocols: `leaf.lovo.nested.v1` (outer 4-fold leave-one-variety-out, inner variety
selection) and `hist.lopo-fixed.v1` (13-fold leave-one-patient-out).

---

## 1. Governed anchors (claim-gate artefacts)

| task | protocol | model | metric | value | status |
|---|---|---|---|---:|---|
| 15.13 | LOPO, 13 patients, 3 seeds | `local-route-a.task-only-control.v1` | Acc / BA / AUROC | 0.8948 / 0.8714 / 0.8677 | `strict_candidate` (incomplete provenance) |
| 24.2 | nested LOVO, 3 seeds | `branch::rgb-pca-consistency` (frozen winner) | GW / Gsw / PhiPS2 | 0.5070 / 0.0756 / 0.4670 | `primary_strict`, promotion permitted |
| 24.2 | nested LOVO, 3 seeds | `train-fold-pca` | GW / Gsw / PhiPS2 | 0.6227 / 0.0203 / 0.4626 | per-target best |
| 24.2 | nested LOVO, 3 seeds | `branch::rgb-pca-consistency-0.05` | GW / Gsw / PhiPS2 | 0.5921 / 0.1013 / 0.4737 | per-target best |
| 24.4 | inner-variety only | all 8 optimisation candidates | — | `all_contexts_fell_back` | negative |
| 25.1 | nested LOVO, 60 ep, 6 arms | pre-registered criteria | C1 NO / C2 NO / C3 NO / C4 YES | — | `development` |
| 14.2 | official test (820 samples) | — | — | **not consumed** | sealed |

C4 = YES is the harness-validity control: inside the same runner, `rgb-pca-consistency-010`
reproduced Gsw 0.0961 and `train-fold-pca` 0.0617, i.e. inside the task-24.2 range
(0.1013 / 0.0756). The task-25.1 spectral-DANN failure is therefore a mechanism
failure, not a harness defect.

---

## 2. Governed task-25.1 spectral nuisance-alignment arms

60 epochs, seeds 7/42/123, pooled OOF R^2 (mean +/- std over seeds):

| arm | role | GrainWeight | Gsw | PhiPS2 |
|---|---|---:|---:|---:|
| `control::spectral-only::mse` | matched control | 0.5313 +/- 0.0012 | -0.0200 +/- 0.0166 | 0.1017 +/- 0.0350 |
| `arm::spectral-dann-lam0.3` | optimisation | 0.5310 +/- 0.0131 | -0.0331 +/- 0.0008 | 0.0595 +/- 0.0027 |
| `arm::spectral-dann-lam1.0` | optimisation | 0.4975 +/- 0.0073 | -0.0343 +/- 0.0054 | 0.0551 +/- 0.0111 |
| `arm::spectral-routed-dann-lam1.0` | optimisation | 0.5298 +/- 0.0012 | -0.0411 +/- 0.0185 | 0.0622 +/- 0.0072 |
| `reference::rgb-pca-consistency-010` | external reference | 0.4991 +/- 0.0080 | 0.0961 +/- 0.0030 | 0.4660 +/- 0.0064 |
| `reference::train-fold-pca` | external reference | 0.6195 +/- 0.0026 | 0.0617 +/- 0.0151 | 0.4524 +/- 0.0027 |

200-epoch diagnostic (same runner, 2 arms): `control::spectral-only::mse` reaches
**GW 0.5230 / Gsw 0.1016 / PhiPS2 0.4781**; `arm::spectral-dann-lam1.0` collapses to
GW 0.3474 / Gsw -0.0386 / PhiPS2 0.0535.

**The spectral control at 200 epochs is the strongest governed HyperLeaf result on Gsw
and PhiPS2 in this repository, and it is a pure-spectrum, single-branch model.**

---

## 3. Exploratory wave results

### 3.1 Wave-1 LOVO screen (200 epochs, pooled OOF R^2)

| candidate | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `dann-spec` (on fold-z) | 0.6442 | -0.0200 | 0.4366 |
| `mlp-spec` (fold-z) | 0.6375 | -0.1191 | 0.4144 |
| `fusion-valueonly` | 0.5861 | -0.1425 | 0.4159 |
| `fusion-crossattn` | 0.5743 | -0.1529 | 0.4223 |
| `fusion-concat` | 0.5759 | -0.1628 | 0.4104 |
| `ridge-spec` | 0.3439 | -0.1654 | 0.4800 |
| `mlp-phys` (hand-crafted physics) | 0.4851 | -0.2306 | 0.5074 |
| `cnn-rgb` | -0.0387 | -0.4339 | -0.0564 |
| `ridge-phys` | 0.0794 | -1.2088 | 0.3143 |

True token-level cross-attention gives no gain over the matched value-only control.
`ridge-spec` ties the strongest neural PhiPS2 (0.4800) — classical baselines matter.

### 3.2 Wave-2 dual-view consistency

| candidate | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `mlp-spec` | 0.6375 | -0.1191 | 0.4144 |
| `dual-consistency` | 0.0737 | -0.3427 | 0.0090 |
| `dual-noconsistency` | 0.0780 | -0.3435 | 0.0146 |
| `dual-consistency-dann` | 0.0486 | -0.3863 | -0.0118 |
| `pca-cnn` | -0.0126 | -0.4214 | -0.1186 |

### 3.3 Wave-3 Histology LOPO (120 epochs, 469 ROIs / 13 patients)

| candidate | Acc | BA | AUROC | ECE |
|---|---:|---:|---:|---:|
| `mlp-spec` | 0.8472 | 0.8286 | 0.8606 | 0.1267 |
| `fusion-concat` | 0.8152 | 0.8229 | 0.9190 | 0.1658 |
| `fusion-valueonly` | 0.8173 | 0.8200 | 0.9125 | 0.1557 |
| `dann-spec` | 0.8088 | 0.8094 | 0.8935 | 0.1008 |
| `cnn-rgb` | 0.7989 | 0.8071 | 0.9155 | 0.1805 |
| `fusion-crossattn` | 0.7918 | 0.7993 | 0.9053 | 0.1951 |

Governed reference (task 15.13): Acc 0.8948 / BA 0.8714.

### 3.4 Wave-4 / Wave-5 calibration

Temperature and Platt scaling cut ECE from 0.13-0.20 to 0.035-0.073 but never improve
balanced accuracy (e.g. `fusion-crossattn` BA 0.7993 -> 0.7679 with fitted threshold +
temperature). Calibration is a trustworthiness result, not an accuracy result.

### 3.5 Wave-6 adversarial-strength sweep (200 epochs, fold-z representation)

| candidate | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `dann-spec-lam1.0` | 0.6057 | 0.0523 | 0.4308 |
| `dann-spec-lam0.3` | 0.6442 | -0.0200 | 0.4366 |
| `dann-spec-lam0.1` | 0.6428 | -0.0732 | 0.4171 |
| `mlp-spec` | 0.6375 | -0.1191 | 0.4144 |
| `dann-fusion-lam1.0` | -0.4822 | -0.0754 | -0.0674 |
| `dann-fusion-lam0.3` | -0.0875 | -0.3999 | -0.1129 |

### 3.6 Wave-7 routed / ensemble adversarial

| candidate | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `routed` | 0.6504 | 0.0523 | 0.4149 |
| `ensemble` | 0.6468 | -0.0051 | 0.4394 |
| `dann1.0` | 0.6057 | 0.0523 | 0.4308 |
| `clean` | 0.6375 | -0.1191 | 0.4144 |

### 3.7 Wave-8 patient-adversarial Histology (120 epochs)

| candidate | Acc | BA | AUROC |
|---|---:|---:|---:|
| `mlp-spec` | 0.8472 | 0.8286 | 0.8606 |
| `fusion-concat` | 0.8159 | 0.8258 | 0.9183 |
| `dann-spec-0.3` | 0.8088 | 0.8094 | 0.8935 |
| `dann-spec-1.0` | 0.7868 | 0.7909 | 0.8745 |
| `fusion-dann-all-1.0` | 0.5842 | 0.4590 | 0.4986 |

### 3.8 Wave-9 instrument-validity study

The governed inner instrument (train on 2 varieties, validate on the 3rd) is *harder*
than the outer estimand and can even invert the sign of the effect:

| design | delta(DANN - clean) on Gsw |
|---|---:|
| outer LOVO (reported estimand) | +0.1714 |
| inner LOVO (task-24.4 instrument) | +0.2014 |
| inner stratified (difficulty-matched) | **-0.0568** |

Per-held-out-variety Gsw (outer LOVO, clean MLP): Heerup 0.0521, Kvium 0.1122,
Rembrandt -0.1568, Sheriff -0.8762 — the pooled estimate is dominated by one variety.

### 3.9 Wave-10 / Wave-11 representation grids

Wave-10, 60 epochs, delta(DANN - clean) on pooled OOF Gsw:

| variant | delta Gsw | delta per seed |
|---|---:|---|
| `fold-z-204` | +0.0677 | 7: +0.0709, 42: +0.0540, 123: +0.0781 |
| `sample-snv-204` | -0.1066 | 7: -0.1176, 42: -0.1090, 123: -0.0931 |
| `fold-z-128` | +0.0739 | 7: +0.0588, 42: +0.0754, 123: +0.0873 |
| `sample-snv-128` | -0.0636 | 7: -0.0619, 42: -0.0621, 123: -0.0667 |

Wave-11, 200 epochs:

| representation | features | domain identifiability | clean GW | clean Gsw | clean PhiPS2 | Dann Gsw | delta Gsw |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fold-z-raw` | 204 | 0.8427 | 0.6375 | -0.1191 | 0.4144 | 0.0523 | +0.1714 |
| `snv-raw` | 204 | 0.7326 | 0.6252 | **0.1075** | **0.4800** | 0.1137 | +0.0062 |
| `snv-z` | 204 | 0.8325 | 0.6209 | -0.0531 | 0.4657 | 0.0286 | +0.0817 |
| `snv-d1-z` | 408 | 0.8099 | 0.5714 | -0.1644 | 0.4875 | -0.0799 | +0.0845 |
| `snv-cr-z` | 408 | 0.7919 | 0.4982 | -0.1651 | 0.4535 | -0.0216 | +0.1435 |
| `snv-d1-cr-z` | 612 | 0.8030 | 0.5027 | -0.1858 | 0.5020 | -0.0769 | +0.1089 |
| `msc-z` | 204 | 0.8211 | 0.6055 | -0.0821 | 0.4550 | -0.0040 | +0.0781 |

Note: in wave-11 the MSC transform was dispatched incorrectly (applied on top of SNV).
This is fixed in wave-12.

### 3.10 Wave-12 single representations, routing, and budget

200 epochs, mean +/- std over 3 seeds:

| representation | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `fold-z` | 0.6375 +/- 0.0058 | -0.1191 +/- 0.0142 | 0.4144 +/- 0.0085 |
| **`snv`** | 0.6252 +/- 0.0009 | **0.1075 +/- 0.0014** | **0.4800 +/- 0.0001** |
| `snv` (400 epochs) | **0.6415 +/- 0.0007** | 0.0958 +/- 0.0016 | 0.4759 +/- 0.0005 |
| `snv-pca32` | 0.2557 +/- 0.0839 | -0.3088 +/- 0.0372 | 0.4284 +/- 0.0292 |
| `msc` (raw-spectrum MSC) | 0.6034 +/- 0.0028 | -0.0843 +/- 0.0186 | 0.4539 +/- 0.0010 |
| `snv-d1-cr-z` | 0.5027 +/- 0.0277 | -0.1858 +/- 0.0298 | 0.5020 +/- 0.0052 |

Inner-variety selected per-target routing (no outer metric used):

| policy | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| inner-selected per-target route | 0.6046 | 0.1075 | 0.4442 |
| fixed `snv` | 0.6252 | 0.1075 | 0.4800 |
| fixed `fold-z` | 0.6375 | -0.1191 | 0.4144 |
| outer-oracle per target (reference only) | 0.6375 | 0.1075 | 0.5020 |

Routing is a **negative result**: the inner instrument cannot select representations
(it picks `snv` for Gsw in 4/4 folds, which is correct, but mis-selects for GW and PhiPS2).
Fix a principled single representation instead.

Budget is saturated and trades GW against Gsw: 200 -> 400 epochs moves GW 0.6252 ->
0.6415 while Gsw falls 0.1075 -> 0.0958. There is no "train 10x longer" headroom.

---

## 4. Known protocol-consistency issues

1. `benchmark/transforms/preprocessing.py` (~L397-400) applies per-sample SNV
   `((raw - mean) / std)` while the pre-registration declares
   `per-band-zscore-inner-train.v1`. Declaration and implementation must be reconciled.
2. The governed candidate space in `benchmark/runners/hyperleaf_optimization_registry.py`
   (`_build_candidates`, ~L377-430) contains **no spectrum-only candidate**, and
   representation was never an optimisation axis. This is why task-24.2/24.3 could not
   discover the SNV baseline.
3. Task-24.2/24.3 screen at 20 epochs and promote at 80 epochs
   (`SCREEN_CAP` / `PROMOTION_CAP`), which systematically penalises simple
   late-converging models. The governed spectral control improves from Gsw -0.0200
   (60 epochs) to +0.1016 (200 epochs) under the same runner.
4. Governed GW 0.5230 vs exploratory GW 0.6252 on the same SNV representation is an
   unresolved encoder/optimizer configuration gap.

---

## 5. Superseded (legacy) numbers — do not compare

`results/legacy/` holds pre-benchmark results computed under patient-grouped or random
splits (for example Histology Local Route A accuracy 0.9189 and an official-test
accuracy of 0.4550). Those are **not** comparable with the group-disjoint protocols
used here and must never be placed in the same table.
