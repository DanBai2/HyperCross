# Wave-12 representation routing (snv-budget400)

- device `cuda`, seeds [7, 42, 123], epochs 400, n=1590, outer LOVO
- `msc` now applies scatter correction to the raw spectrum (wave-11 dispatch defect fixed); `snv-pca32` is new

## Single-representation pooled OOF R2 (mean +/- std over seeds)

| representation | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `snv` | 0.6415 +/- 0.0007 | 0.0958 +/- 0.0016 | 0.4759 +/- 0.0005 |

## Inner-variety selected per-target route (no outer metric used)

| target | selected representation per outer fold |
|---|---|
| GrainWeight | Heerup:`snv`, Kvium:`snv`, Rembrandt:`snv`, Sheriff:`snv` |
| Gsw | Heerup:`snv`, Kvium:`snv`, Rembrandt:`snv`, Sheriff:`snv` |
| PhiPS2 | Heerup:`snv`, Kvium:`snv`, Rembrandt:`snv`, Sheriff:`snv` |

## Routed policy vs a-priori single representation (outer LOVO pooled OOF R2)

| policy | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| inner-selected per-target route | 0.6415 | 0.0958 | 0.4759 |
| fixed `snv` | 0.6415 | 0.0958 | 0.4759 |
| outer-oracle per target (reference only) | 0.6415 | 0.0958 | 0.4759 |

