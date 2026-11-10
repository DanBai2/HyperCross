# Wave-8 patient-adversarial Histology LOPO (patient-dann)

- device: `cuda`, seeds: [7, 42, 123], epochs: 120, n=469 ROIs / 13 patients
- reference (governed task 15.13): BA 0.8714 / AUROC 0.8677

| candidate | Acc | BA | AUROC | ECE |
|---|---|---|---|---|
| `mlp-spec` | 0.8472 | 0.8286 | 0.8606 | 0.1267 |
| `fusion-concat` | 0.8159 | 0.8258 | 0.9183 | 0.1688 |
| `dann-spec-0.3` | 0.8088 | 0.8094 | 0.8935 | 0.1008 |
| `dann-spec-1.0` | 0.7868 | 0.7909 | 0.8745 | 0.0805 |
| `fusion-dann-spec-1.0` | 0.7754 | 0.7785 | 0.8675 | 0.1778 |
| `fusion-dann-all-1.0` | 0.5842 | 0.459 | 0.4986 | 0.1324 |
