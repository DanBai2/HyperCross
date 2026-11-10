# Task 25.1 - spectral nuisance-alignment arm family (development tier)

- execution_id: `task-25-1-20261108T130712621781Z`
- preregistration hash: `69783c2d5c392a41f1ad1ffa04963eb7f53b620e6ffa4c27b0f350826ce781e8`
- protocol: `leaf.lovo.nested.v1` hash `b22c46f6b44d2824f672b73aab331799642d776e5b7ce296614db2270586bcdf`
- budget: 60 epochs / seeds [7, 42, 123] / device `NVIDIA GeForce RTX 5080`
- claim status: `development`
- official evaluation budget consumed: 0

## Pooled out-of-fold R2 (mean +/- std over seeds)

| arm | role | GrainWeight | Gsw | PhiPS2 |
|---|---|---:|---:|---:|
| `control::spectral-only::mse` | matched_control | 0.5313 +/- 0.0012 | -0.0200 +/- 0.0166 | 0.1017 +/- 0.0350 |
| `arm::spectral-dann-lam0.3` | optimization | 0.5310 +/- 0.0131 | -0.0331 +/- 0.0008 | 0.0595 +/- 0.0027 |
| `arm::spectral-dann-lam1.0` | optimization | 0.4975 +/- 0.0073 | -0.0343 +/- 0.0054 | 0.0551 +/- 0.0111 |
| `arm::spectral-routed-dann-lam1.0` | optimization | 0.5298 +/- 0.0012 | -0.0411 +/- 0.0185 | 0.0622 +/- 0.0072 |
| `reference::rgb-pca-consistency-010` | external_reference | 0.4991 +/- 0.0080 | 0.0961 +/- 0.0030 | 0.4660 +/- 0.0064 |
| `reference::train-fold-pca` | external_reference | 0.6195 +/- 0.0026 | 0.0617 +/- 0.0151 | 0.4524 +/- 0.0027 |

## Pre-registered criteria

| criterion | met | evidence |
|---|---|---|
| C1 | NO | gsw_delta_mean=-0.0143, gsw_delta_per_seed={'7': -0.029027685637578404, '42': -0.004717739918575159, '123': -0.009187094140035601} |
| C2 | NO | gsw_delta_mean=-0.0212, grainweight_gap=0.0015, phips2_gap=0.0395 |
| C3 | NO | lam0.3_gsw_mean=-0.0331, lam1.0_gsw_mean=-0.0343 |
| C4 | YES | reference_gsw_means={'reference::rgb-pca-consistency-010': 0.09611757793824272, 'reference::train-fold-pca': 0.0617042688545277} |

## Declared criteria text

- **C1**: spectral-dann-lam1.0 pooled-OOF Gsw R2 exceeds the matched spectral-only control by at least 0.03 in the seed mean and is strictly positive in every seed.
- **C2**: spectral-routed-dann-lam1.0 keeps the Gsw gain (>= control + 0.03) while staying within 0.02 of the control on GrainWeight and PhiPS2.
- **C3**: Gsw R2 is monotone in the adversarial strength: lam0.3 <= lam1.0.
- **C4**: The in-run RGB/PCA and train-fold-PCA references land in the task-24.2 range on Gsw, confirming harness comparability.

