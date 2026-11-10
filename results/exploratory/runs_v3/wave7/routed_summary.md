# Wave-7 routed / ensembled adversarial models (routed)

- device: `cuda`, seeds: [7, 42, 123], epochs: 200
- metric: pooled OOF R2 (mean / min over seeds); routing chosen by inner LOVO only

| candidate | GW | Gsw | PhiPS2 | min GW | min Gsw |
|---|---|---|---|---|---|
| `dann1.0` | 0.6057 | 0.0523 | 0.4308 | 0.5907 | 0.0308 |
| `routed` | 0.6504 | 0.0523 | 0.4149 | 0.6386 | 0.0308 |
| `ensemble` | 0.6468 | -0.0051 | 0.4394 | 0.6405 | -0.0189 |
| `dann0.3` | 0.6442 | -0.02 | 0.4366 | 0.6383 | -0.0215 |
| `clean` | 0.6375 | -0.1191 | 0.4144 | 0.6293 | -0.1389 |
