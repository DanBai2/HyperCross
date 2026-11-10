# Wave-12 representation routing (representation-routing)

- device `cuda`, seeds [7, 42, 123], epochs 200, n=1590, outer LOVO
- `msc` now applies scatter correction to the raw spectrum (wave-11 dispatch defect fixed); `snv-pca32` is new

## Single-representation pooled OOF R2 (mean +/- std over seeds)

| representation | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `fold-z` | 0.6375 +/- 0.0058 | -0.1191 +/- 0.0142 | 0.4144 +/- 0.0085 |
| `snv` | 0.6252 +/- 0.0009 | 0.1075 +/- 0.0014 | 0.48 +/- 0.0001 |
| `snv-pca32` | 0.2557 +/- 0.0839 | -0.3088 +/- 0.0372 | 0.4284 +/- 0.0292 |
| `msc` | 0.6034 +/- 0.0028 | -0.0843 +/- 0.0186 | 0.4539 +/- 0.001 |
| `snv-d1-cr-z` | 0.5027 +/- 0.0277 | -0.1858 +/- 0.0298 | 0.502 +/- 0.0052 |

## Inner-variety selected per-target route (no outer metric used)

| target | selected representation per outer fold |
|---|---|
| GrainWeight | Heerup:`msc`, Kvium:`fold-z`, Rembrandt:`fold-z`, Sheriff:`snv` |
| Gsw | Heerup:`snv`, Kvium:`snv`, Rembrandt:`snv`, Sheriff:`snv` |
| PhiPS2 | Heerup:`snv`, Kvium:`msc`, Rembrandt:`snv`, Sheriff:`snv` |

## Routed policy vs a-priori single representation (outer LOVO pooled OOF R2)

| policy | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| inner-selected per-target route | 0.6046 | 0.1075 | 0.4442 |
| fixed `fold-z` | 0.6375 | -0.1191 | 0.4144 |
| fixed `snv` | 0.6252 | 0.1075 | 0.48 |
| fixed `snv-pca32` | 0.2557 | -0.3088 | 0.4284 |
| fixed `msc` | 0.6034 | -0.0843 | 0.4539 |
| fixed `snv-d1-cr-z` | 0.5027 | -0.1858 | 0.502 |
| outer-oracle per target (reference only) | 0.6375 | 0.1075 | 0.502 |

