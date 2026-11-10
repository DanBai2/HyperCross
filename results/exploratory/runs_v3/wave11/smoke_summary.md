# Wave-11 spectral representation grid (smoke)

- device `cuda`, seeds [42], epochs 10, n=1590, outer LOVO
- `clean-mlp` = plain MLP; `dann1.0` = same trunk plus variety-adversarial head (GRL lambda 1.0)

## Pooled OOF R2 (mean over seeds)

| representation | features | domain-ident. | linear Gsw | clean GrainWeight | clean Gsw | clean PhiPS2 | dann Gsw |
|---|---:|---:|---:|---:|---:|---:|---:|
| `snv-d1-cr-z` | 612 | 0.803 | -0.4055 | 0.5991 | 0.0868 | 0.5308 | n/a |
| `msc-z` | 204 | 0.8211 | -0.3779 | 0.6217 | 0.1129 | 0.5101 | n/a |

## delta(DANN - clean) on pooled OOF R2

| representation | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|

## Pre-registered predictions

- Q1 some representation other than fold-z-raw beats it on pooled-OOF Gsw by at least +0.03 with a positive GrainWeight delta
- Q2 delta(DANN - clean) is positive exactly where in-domain variety identifiability is high and the clean target readout is weak

