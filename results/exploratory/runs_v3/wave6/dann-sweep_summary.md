# Wave-6 adversarial sweep (dann-sweep)

- device: `cuda`, seeds: [7, 42, 123], lams: [0.1, 0.3, 1.0], epochs: 200
- metric: pooled OOF R2, mean/min over seeds

| candidate | GW | Gsw | PhiPS2 | min(Gsw) |
|---|---|---|---|---|
| `dann-spec-lam1.0` | 0.6057 | 0.0523 | 0.4308 | 0.0308 |
| `dann-spec-lam0.3` | 0.6442 | -0.02 | 0.4366 | -0.0215 |
| `dann-spec-lam0.1` | 0.6428 | -0.0732 | 0.4171 | -0.0855 |
| `dann-fusion-lam1.0` | -0.4822 | -0.0754 | -0.0674 | -0.1393 |
| `mlp-spec` | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `fusion-concat` | 0.576 | -0.1628 | 0.4104 | -0.1747 |
| `dann-fusion-spec-lam0.3` | -0.1156 | -0.3669 | -0.056 | -0.4569 |
| `dann-fusion-lam0.3` | -0.0875 | -0.3999 | -0.1129 | -0.5066 |
| `dann-fusion-lam0.1` | -0.0996 | -0.4811 | -0.0743 | -0.5034 |
