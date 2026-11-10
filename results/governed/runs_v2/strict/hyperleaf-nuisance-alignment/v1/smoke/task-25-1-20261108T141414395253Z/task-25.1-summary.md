# Task 25.1 - spectral nuisance-alignment arm family (development tier)

- execution_id: `task-25-1-20261108T141414395253Z`
- preregistration hash: `69783c2d5c392a41f1ad1ffa04963eb7f53b620e6ffa4c27b0f350826ce781e8`
- protocol: `leaf.lovo.nested.v1` hash `b22c46f6b44d2824f672b73aab331799642d776e5b7ce296614db2270586bcdf`
- budget: 200 epochs / seeds [7, 42, 123] / device `NVIDIA GeForce RTX 5080`
- claim status: `development`
- official evaluation budget consumed: 0

## Pooled out-of-fold R2 (mean +/- std over seeds)

| arm | role | GrainWeight | Gsw | PhiPS2 |
|---|---|---:|---:|---:|
| `control::spectral-only::mse` | matched_control | 0.5230 +/- 0.0111 | 0.1016 +/- 0.0049 | 0.4781 +/- 0.0042 |
| `arm::spectral-dann-lam1.0` | optimization | 0.3474 +/- 0.0953 | -0.0386 +/- 0.0053 | 0.0535 +/- 0.0202 |

## Pre-registered criteria

| criterion | met | evidence |
|---|---|---|
| C1 | n/a | not evaluated in this partial run |
| C2 | n/a | not evaluated in this partial run |
| C3 | n/a | not evaluated in this partial run |
| C4 | n/a | not evaluated in this partial run |

## Declared criteria text

- **C1**: spectral-dann-lam1.0 pooled-OOF Gsw R2 exceeds the matched spectral-only control by at least 0.03 in the seed mean and is strictly positive in every seed.
- **C2**: spectral-routed-dann-lam1.0 keeps the Gsw gain (>= control + 0.03) while staying within 0.02 of the control on GrainWeight and PhiPS2.
- **C3**: Gsw R2 is monotone in the adversarial strength: lam0.3 <= lam1.0.
- **C4**: The in-run RGB/PCA and train-fold-PCA references land in the task-24.2 range on Gsw, confirming harness comparability.

