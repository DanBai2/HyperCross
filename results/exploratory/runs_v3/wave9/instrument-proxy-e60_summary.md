# Wave-9 instrument-validity study (instrument-proxy-e60)

- device `cuda`, seeds [7, 42, 123], epochs 60, n=1590
- metric: pooled OOF R2 in original target units (mean over seeds)
- A `outer-lovo` = reported estimand (3 train varieties); B `inner-lovo` = governed task-24.4 instrument (2 train varieties); C `inner-stratified` = difficulty-matched replacement

## Pooled R2 by design

| family | A outer-lovo | B inner-lovo | C inner-stratified |
|---|---|---|---|
| `clean-mlp` Gsw | -0.0476 | n/a | n/a |
| `dann1.0` Gsw | 0.0201 | n/a | n/a |
| `clean-mlp` GrainWeight | 0.6148 | n/a | n/a |
| `dann1.0` GrainWeight | 0.5371 | n/a | n/a |
| `clean-mlp` Gsw (per-outer mean) | -0.1382 | n/a | n/a |
| `dann1.0` Gsw (per-outer mean) | -0.0535 | n/a | n/a |

## DANN-minus-clean delta on Gsw

| design | GrainWeight | Gsw | PhiPS2 |
|---|---|---|---|
| outer-lovo | -0.0777 | 0.0677 | -0.0516 |

## Per-held-out-variety Gsw (design A, grading the outer estimand)

| family | Heerup | Kvium | Rembrandt | Sheriff |
|---|---|---|---|---|
| `clean-mlp` | 0.1197 | 0.1329 | -0.0633 | -0.7423 |
| `dann1.0` | 0.1339 | 0.1147 | -0.0671 | -0.3954 |
