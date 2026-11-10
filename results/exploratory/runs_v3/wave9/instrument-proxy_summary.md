# Wave-9 instrument-validity study (instrument-proxy)

- device `cuda`, seeds [7, 42, 123], epochs 200, n=1590
- metric: pooled OOF R2 in original target units (mean over seeds)
- A `outer-lovo` = reported estimand (3 train varieties); B `inner-lovo` = governed task-24.4 instrument (2 train varieties); C `inner-stratified` = difficulty-matched replacement

## Pooled R2 by design

| family | A outer-lovo | B inner-lovo | C inner-stratified |
|---|---|---|---|
| `ridge-spec` Gsw | -0.1654 | -0.2405 | 0.2707 |
| `clean-mlp` Gsw | -0.1191 | -0.2195 | 0.2732 |
| `dann1.0` Gsw | 0.0523 | -0.0181 | 0.2164 |
| `ridge-spec` GrainWeight | 0.3439 | 0.2596 | 0.6614 |
| `clean-mlp` GrainWeight | 0.6375 | 0.5909 | 0.776 |
| `dann1.0` GrainWeight | 0.6057 | 0.5169 | 0.6739 |
| `ridge-spec` Gsw (per-outer mean) | -0.2477 | -0.2485 | 0.2665 |
| `clean-mlp` Gsw (per-outer mean) | -0.2172 | -0.2274 | 0.2693 |
| `dann1.0` Gsw (per-outer mean) | -0.0159 | -0.0254 | 0.2118 |

## DANN-minus-clean delta on Gsw

| design | GrainWeight | Gsw | PhiPS2 |
|---|---|---|---|
| outer-lovo | -0.0318 | 0.1714 | 0.0164 |
| inner-lovo | -0.074 | 0.2014 | -0.019 |
| inner-stratified | -0.1021 | -0.0568 | -0.0739 |

## Per-held-out-variety Gsw (design A, grading the outer estimand)

| family | Heerup | Kvium | Rembrandt | Sheriff |
|---|---|---|---|---|
| `ridge-spec` | -0.1935 | 0.0953 | -0.2789 | -0.6135 |
| `clean-mlp` | 0.0521 | 0.1122 | -0.1568 | -0.8762 |
| `dann1.0` | 0.1635 | 0.0911 | -0.0136 | -0.3046 |
