# Wave-3 Histology LOPO (lopo-3seed)

- device: `cuda`, seeds: [7, 42, 123], epochs: 120, n=469 ROIs / 13 patients
- metric: pooled OOF; BA/AUROC use sklearn; ECE 15 bins

| candidate | Acc | BA | AUROC | ECE |
|---|---|---|---|---|
| `mlp-spec` | 0.8472 | 0.8286 | 0.8606 | 0.1267 |
| `fusion-concat` | 0.8152 | 0.8229 | 0.919 | 0.1658 |
| `fusion-valueonly` | 0.8173 | 0.82 | 0.9125 | 0.1557 |
| `dann-spec` | 0.8088 | 0.8094 | 0.8935 | 0.1008 |
| `cnn-rgb` | 0.7989 | 0.8071 | 0.9155 | 0.1805 |
| `fusion-crossattn` | 0.7918 | 0.7993 | 0.9053 | 0.1951 |

Governed reference (task 15.13, pooled OOF): `local-route-a.task-only-control.v1` Acc 0.8948 / BA 0.8714 / AUROC 0.8677
