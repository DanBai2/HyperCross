# Wave-10 representation sensitivity (representation-sensitivity)

- device `cuda`, seeds [7, 42, 123], epochs 60, n=1590, outer LOVO
- `clean-mlp` = plain MLP on the spectrum; `dann1.0` = same trunk plus a variety-adversarial head (GRL lambda 1.0)

## Pooled OOF R2 (mean over seeds)

| variant | n_in | fold-fitted | family | GrainWeight | Gsw | PhiPS2 |
|---|---:|---|---|---:|---:|---:|
| `fold-z-204` | 204 | True | `clean-mlp` | 0.6148 | -0.0476 | 0.4051 |
| `fold-z-204` | 204 | True | `dann1.0` | 0.5371 | 0.0201 | 0.3535 |
| `sample-snv-204` | 204 | False | `clean-mlp` | 0.5222 | 0.1154 | 0.4758 |
| `sample-snv-204` | 204 | False | `dann1.0` | 0.5253 | 0.0088 | 0.1468 |
| `fold-z-128` | 128 | True | `clean-mlp` | 0.5889 | -0.0295 | 0.41 |
| `fold-z-128` | 128 | True | `dann1.0` | 0.52 | 0.0444 | 0.3758 |
| `sample-snv-128` | 128 | False | `clean-mlp` | 0.5209 | 0.1165 | 0.4699 |
| `sample-snv-128` | 128 | False | `dann1.0` | 0.5351 | 0.0529 | 0.2593 |

## delta(DANN - clean) on pooled OOF R2

| variant | GrainWeight | Gsw | PhiPS2 | Gsw per seed |
|---|---:|---:|---:|---|
| `fold-z-204` | -0.0777 | 0.0677 | -0.0516 | 7:0.0709, 42:0.054, 123:0.0781 |
| `sample-snv-204` | 0.0031 | -0.1066 | -0.329 | 7:-0.1176, 42:-0.109, 123:-0.0931 |
| `fold-z-128` | -0.0689 | 0.0739 | -0.0342 | 7:0.0588, 42:0.0754, 123:0.0873 |
| `sample-snv-128` | 0.0142 | -0.0636 | -0.2106 | 7:-0.0619, 42:-0.0621, 123:-0.0667 |

## Pre-registered predictions

- Q1 R1 fold-z-204 reproduces the wave-9 delta, positive in every seed
- Q2 R4 sample-snv-128 reproduces the task-25.1 delta, negative in every seed
- Q3 sign follows the normalization (SNV vs fold-fitted), not the band count

