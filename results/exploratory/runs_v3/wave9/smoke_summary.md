# Wave-9 instrument-validity study (smoke)

- device `cuda`, seeds [42], epochs 20, n=1590
- metric: pooled OOF R2 in original target units (mean over seeds)
- A `outer-lovo` = reported estimand (3 train varieties); B `inner-lovo` = governed task-24.4 instrument (2 train varieties); C `inner-stratified` = difficulty-matched replacement

## Pooled R2 by design

| family | A outer-lovo | B inner-lovo | C inner-stratified |
|---|---|---|---|
| `clean-mlp` Gsw | 0.0076 | -0.0406 | 0.1534 |
| `dann1.0` Gsw | -0.0265 | -0.055 | 0.0593 |
| `clean-mlp` GrainWeight | 0.5621 | 0.5512 | 0.6765 |
| `dann1.0` GrainWeight | 0.5021 | 0.4749 | 0.6119 |
| `clean-mlp` Gsw (per-outer mean) | -0.0661 | -0.048 | 0.1487 |
| `dann1.0` Gsw (per-outer mean) | -0.0926 | -0.0616 | 0.0538 |

## DANN-minus-clean delta on Gsw

| design | GrainWeight | Gsw | PhiPS2 |
|---|---|---|---|
| outer-lovo | -0.06 | -0.0341 | -0.2357 |
| inner-lovo | -0.0763 | -0.0144 | -0.1327 |
| inner-stratified | -0.0646 | -0.0941 | -0.1912 |

## Per-held-out-variety Gsw (design A, grading the outer estimand)

| family | Heerup | Kvium | Rembrandt | Sheriff |
|---|---|---|---|---|
| `clean-mlp` | 0.1045 | 0.1117 | -0.0694 | -0.4111 |
| `dann1.0` | 0.0141 | 0.0282 | -0.1382 | -0.2747 |
