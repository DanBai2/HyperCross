# Wave-2 dual-view LOVO (dualview-3seed)

- device: `cuda`, seeds: [7, 42, 123], epochs: 200, consistency: 0.05, n=1590
- metric: pooled OOF R2 (all folds concatenated, original target units)

| candidate | GrainWeight | Gsw | PhiPS2 | min(Gsw) |
|---|---|---|---|---|
| `mlp-spec` | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `dann-spec` | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `dual-consistency` | 0.0737 | -0.3427 | 0.009 | -0.3914 |
| `dual-noconsistency` | 0.078 | -0.3435 | 0.0146 | -0.3875 |
| `dual-consistency-dann` | 0.0486 | -0.3863 | -0.0118 | -0.4137 |
| `pca-cnn` | -0.0126 | -0.4214 | -0.1186 | -0.4573 |
