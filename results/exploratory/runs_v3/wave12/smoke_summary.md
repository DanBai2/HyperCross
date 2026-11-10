# Wave-12 representation routing (smoke)

- device `cuda`, seeds [42], epochs 5, n=1590, outer LOVO
- `msc` now applies scatter correction to the raw spectrum (wave-11 dispatch defect fixed); `snv-pca32` is new

## Single-representation pooled OOF R2 (mean +/- std over seeds)

| representation | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `msc` | 0.5096 +/- 0.0 | 0.1507 +/- 0.0 | 0.2046 +/- 0.0 |

## Inner-variety selected per-target route (no outer metric used)

| target | selected representation per outer fold |
|---|---|
| GrainWeight | Heerup:`msc` |
| Gsw | Heerup:`msc` |
| PhiPS2 | Heerup:`msc` |

## Routed policy vs a-priori single representation (outer LOVO pooled OOF R2)

| policy | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| inner-selected per-target route | 0.5096 | 0.1507 | 0.2046 |
| fixed `msc` | 0.5096 | 0.1507 | 0.2046 |
| outer-oracle per target (reference only) | 0.5096 | 0.1507 | 0.2046 |

