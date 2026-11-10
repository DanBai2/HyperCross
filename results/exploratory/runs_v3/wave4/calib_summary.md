# Wave-4 calibration study (calib)

- input: `lopo-3seed_oof_logits.npz`, n=469 ROIs, 13 patients
- temperature fitted leave-one-patient-out; metrics are 3-seed means

| candidate | AUROC | BA@0.5 | BA@fittedTau | BA@0.5+temp | BA@fittedTau+temp | ECE raw | ECE temp |
|---|---|---|---|---|---|---|---|
| `mlp-spec` | 0.8605 | 0.8286 | 0.8405 | 0.8286 | 0.8433 | 0.1268 | 0.0568 |
| `fusion-concat` | 0.9198 | 0.8229 | 0.8125 | 0.8229 | 0.825 | 0.1658 | 0.0354 |
| `fusion-valueonly` | 0.9125 | 0.82 | 0.8079 | 0.82 | 0.8023 | 0.1557 | 0.0548 |
| `dann-spec` | 0.8935 | 0.8094 | 0.8188 | 0.8094 | 0.8178 | 0.1008 | 0.0692 |
| `cnn-rgb` | 0.915 | 0.8071 | 0.8082 | 0.8071 | 0.808 | 0.1805 | 0.0552 |
| `fusion-crossattn` | 0.9053 | 0.7993 | 0.7971 | 0.7993 | 0.7679 | 0.1951 | 0.0731 |

Governed reference pooled OOF BA = 0.8714 (task 15.13)
