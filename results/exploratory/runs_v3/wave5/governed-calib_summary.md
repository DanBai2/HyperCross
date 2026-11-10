# Wave-5 governed-baseline calibration (governed-calib)

- source: task-15.13 seed-consensus OOF artifacts (n=469 ROIs, 13 patients)
- Platt + threshold fitted leave-one-patient-out; no retraining

| candidate | AUROC | Acc@0.5 | BA@0.5 | BA@tau(raw) | BA@0.5(platt) | BA@tau(platt) | ECE raw | ECE platt |
|---|---|---|---|---|---|---|---|---|
| `alignment-nce` | 0.8936 | 0.8934 | 0.8685 | 0.8616 | 0.8459 | 0.83 | 0.0748 | 0.0785 |
| `alignment-only-control` | 0.8929 | 0.8913 | 0.8682 | 0.858 | 0.8443 | 0.8254 | 0.0725 | 0.0747 |
| `mil-concat` | 0.8845 | 0.8955 | 0.866 | 0.838 | 0.8416 | 0.8245 | 0.0981 | 0.0866 |
| `attention-disabled` | 0.8786 | 0.8934 | 0.8644 | 0.8185 | 0.8657 | 0.8127 | 0.0853 | 0.1052 |
| `pca-only` | 0.8564 | 0.8678 | 0.8323 | 0.8363 | 0.8126 | 0.8079 | 0.1008 | 0.0991 |
| `nce-only-control` | 0.8865 | 0.8913 | 0.8682 | 0.8416 | 0.8426 | 0.7993 | 0.0672 | 0.0905 |
| `task-only-control` | 0.8621 | 0.8913 | 0.8682 | 0.8369 | 0.8459 | 0.7977 | 0.0636 | 0.0912 |
| `rgb-only` | 0.877 | 0.7719 | 0.7662 | 0.7875 | 0.762 | 0.7624 | 0.1043 | 0.0626 |
