# Exploratory waves

`wave1` .. `wave12` are self-contained exploratory studies. Their outputs live in
`results/exploratory/runs_v3/`. **Everything here is development tier**: it is valid
for hypothesis generation, but any claim in a paper must be re-computed through
`benchmark/` (the claim gate).

| wave | script | topic | outcome |
|---|---|---|---|
| 1 | `wave1/hyperleaf_lovo_screen.py` | HyperLeaf LOVO screen: physics features, real token cross-attention vs matched value-only control, adversarial | negative for attention |
| 2 | `wave2/hyperleaf_lovo_dualview.py` | dual-view consistency fusion | collapses |
| 3 | `wave3/histology_lopo_screen.py` | Histology LOPO screen | below governed anchor |
| 4 | `wave4/histology_calibration.py` | temperature scaling | ECE only |
| 5 | `wave5/governed_histology_calibration.py` | Platt + threshold on governed OOF | ECE only |
| 6 | `wave6/hyperleaf_dann_sweep.py` | adversarial strength sweep | fusion DANN catastrophic |
| 7 | `wave7/hyperleaf_routed_adversarial.py` | per-target routing + ensembling | no gain beyond representation |
| 8 | `wave8/histology_patient_dann.py` | patient-adversarial Histology | harmful |
| 9 | `wave9/hyperleaf_instrument_proxy_study.py` | validity of the governed selection instrument | inner instrument unreliable |
| 10 | `wave10/hyperleaf_representation_sensitivity.py` | normalization sensitivity grid | sign of DANN gain follows normalization |
| 11 | `wave11/hyperleaf_spectral_representation_grid.py` | 8 representations x {clean, DANN} + domain-identifiability probe | **per-sample SNV is the dominant gain** |
| 12 | `wave12/hyperleaf_representation_routing.py` | corrected MSC, chemometric latents, honest per-target routing, budget curve | routing is a negative result; budget saturated |

## Feature cache

Waves 10-12 read a cached mean spectrum + RGB tensor
(`experiments/wave1/cache_leaf_spec_rgb.npz`, 1590 x 204 spectrum + 3 x 32 x 64 RGB),
which makes every wave rerun take seconds. The cache is **not** committed because it
is derived from the licensed datasets; see `docs/REPRODUCING.md` to rebuild it.
