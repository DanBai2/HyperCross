# Wave-1 LOVO screen (screen-3seed)

- device: `cuda`, seeds: [7, 42, 123], epochs: 200, n=1590
- metric: pooled OOF (all folds concatenated), R2 in original target units

| candidate | GrainWeight | Gsw | PhiPS2 | min(Gsw) |
|---|---|---|---|---|
| `dann-spec` | 0.6442 | -0.02 | 0.4366 | -0.0215 |
| `mlp-spec` | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `fusion-valueonly` | 0.5861 | -0.1425 | 0.4159 | -0.1551 |
| `fusion-crossattn` | 0.5743 | -0.1529 | 0.4223 | -0.1931 |
| `fusion-concat` | 0.5759 | -0.1628 | 0.4104 | -0.1747 |
| `ridge-spec` | 0.3439 | -0.1654 | 0.48 | -0.1654 |
| `mlp-phys` | 0.4851 | -0.2306 | 0.5074 | -0.2787 |
| `cnn-rgb` | -0.0387 | -0.4339 | -0.0564 | -0.5369 |
| `ridge-phys` | 0.0794 | -1.2088 | 0.3143 | -1.2164 |
