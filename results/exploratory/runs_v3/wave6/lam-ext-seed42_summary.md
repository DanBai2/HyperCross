# Wave-6 adversarial sweep (lam-ext-seed42)

- device: `cuda`, seeds: [42], lams: [2.0, 3.0], epochs: 200
- metric: pooled OOF R2, mean/min over seeds

| candidate | GW | Gsw | PhiPS2 | min(Gsw) |
|---|---|---|---|---|
| `dann-spec-lam2.0` | 0.5694 | 0.0497 | 0.3604 | 0.0497 |
| `dann-spec-lam3.0` | 0.519 | 0.0044 | 0.2631 | 0.0044 |
| `dann-fusion-lam2.0` | -0.1268 | -0.0387 | -0.0172 | -0.0387 |
| `dann-fusion-lam3.0` | -0.1256 | -0.0468 | -0.0137 | -0.0468 |
| `mlp-spec` | 0.6413 | -0.1124 | 0.4142 | -0.1124 |
| `fusion-concat` | 0.5781 | -0.1414 | 0.4044 | -0.1414 |
| `dann-fusion-spec-lam0.3` | -0.0992 | -0.333 | -0.0847 | -0.333 |
