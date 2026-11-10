# Wave-11 spectral representation grid (spectral-representation-grid)

- device `cuda`, seeds [7, 42, 123], epochs 200, n=1590, outer LOVO
- `clean-mlp` = plain MLP; `dann1.0` = same trunk plus variety-adversarial head (GRL lambda 1.0)

## Pooled OOF R2 (mean over seeds)

| representation | features | domain-ident. | linear Gsw | clean GrainWeight | clean Gsw | clean PhiPS2 | dann Gsw |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fold-z-raw` | 204 | 0.8427 | -0.1655 | 0.6375 | -0.1191 | 0.4144 | 0.0523 |
| `snv-raw` | 204 | 0.7326 | -0.0166 | 0.6252 | 0.1075 | 0.48 | 0.1137 |
| `snv-z` | 204 | 0.8325 | -0.2917 | 0.6209 | -0.0531 | 0.4657 | 0.0286 |
| `snv-d1-z` | 408 | 0.8099 | -0.8739 | 0.5714 | -0.1644 | 0.4875 | -0.0799 |
| `snv-cr-z` | 408 | 0.7919 | -0.6812 | 0.4982 | -0.1651 | 0.4535 | -0.0216 |
| `snv-d1-cr-z` | 612 | 0.803 | -0.4055 | 0.5027 | -0.1858 | 0.502 | -0.0769 |
| `msc-z` | 204 | 0.8211 | -0.3779 | 0.6055 | -0.0821 | 0.455 | -0.004 |
| `snv-msc-z` | 204 | 0.8211 | -0.3779 | 0.6055 | -0.0821 | 0.455 | -0.004 |

## delta(DANN - clean) on pooled OOF R2

| representation | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `fold-z-raw` | -0.0318 | 0.1714 | 0.0164 |
| `snv-raw` | -0.0417 | 0.0062 | -0.0103 |
| `snv-z` | -0.0617 | 0.0817 | 0.013 |
| `snv-d1-z` | -0.0756 | 0.0845 | 0.0205 |
| `snv-cr-z` | -0.0517 | 0.1435 | 0.0464 |
| `snv-d1-cr-z` | -0.0156 | 0.1089 | 0.0011 |
| `msc-z` | -0.0473 | 0.0781 | 0.0285 |
| `snv-msc-z` | -0.0473 | 0.0781 | 0.0285 |

## Pre-registered predictions

- Q1 some representation other than fold-z-raw beats it on pooled-OOF Gsw by at least +0.03 with a positive GrainWeight delta
- Q2 delta(DANN - clean) is positive exactly where in-domain variety identifiability is high and the clean target readout is weak

