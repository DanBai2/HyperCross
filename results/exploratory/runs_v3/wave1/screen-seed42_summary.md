# Wave-1 LOVO screen (screen-seed42)

- device: `cuda`, seeds: [42], epochs: 200, n=1590
- metric: pooled OOF (all folds concatenated), R2 in original target units

| candidate | GrainWeight | Gsw | PhiPS2 | min(Gsw) |
|---|---|---|---|---|
| `ridge-spec` | 0.3439 | -0.1654 | 0.48 | -0.1654 |
| `ridge-phys` | nan | nan | nan | nan |
| `mlp-spec` | 0.6413 | -0.1124 | 0.4142 | -0.1124 |
| `mlp-phys` | nan | nan | nan | nan |
| `cnn-rgb` | -0.0902 | -0.5367 | -0.0419 | -0.5367 |
| `fusion-concat` | nan | nan | nan | nan |
| `fusion-valueonly` | nan | nan | nan | nan |
| `fusion-crossattn` | nan | nan | nan | nan |
| `dann-spec` | 0.6463 | -0.0215 | 0.4332 | -0.0215 |
