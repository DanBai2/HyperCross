# Contributing

Thanks for wanting to build on this. The point of the repository is that a number can be
traced from a claim back to a frozen protocol, a seed set, a fitted-artefact lineage and
an append-only run record. Contributions that preserve that are welcome.

## Before you start

Read, in this order:

1. `docs/ARCHITECTURE.md` — module map, the six invariants, and the extension cookbook.
2. `RESULTS.md` — what is already established, and what has already been falsified.
3. `docs/TESTING.md` — how to run the suite, and which tests are bound to the audited host.
4. `docs/REPRODUCING.md` — datasets, layout, feature cache.

## The two rules that matter most

**Do not mix protocol tiers.** A number computed under a random or patient-grouped split
(`results/legacy/`) cannot appear next to a governed LOVO / LOPO number. If you publish a
comparison, bind dataset version, protocol manifest, representation, selection rule, seed
set and metric definition.

**Pre-register before you execute.** State the hypothesis, the mechanism, the decision
criterion (including the exact delta and the sign requirement per seed) and the controls
*before* running. `configs/optimization/v2/hyperleaf-task-25-preregistration.json` is a
frozen example; `benchmark/runners/hyperleaf_nuisance_alignment.py` evaluates the criteria
automatically and writes `n/a` for any criterion a partial run did not reach.

## Contribution types

### A new representation or baseline

Highest-value area. The open questions are listed in `RESULTS.md` section 4 and the
repository `README.md` limitations section; the largest are modern spectral baselines
(SpectralFormer, SSFTT, HybridSN, S2Mamba) and classical chemometric baselines
(PLS, RF, Ridge) under the same protocols and seeds. Note that a plain Ridge on the
fold-z representation already matches the strongest neural PhiPS2 number, so classical
baselines are not a formality — they are a threat to any claim about the neural design.

### A new mechanism

Follow the negative-result discipline this repository already uses: include a matched
control that differs in exactly one factor (`fusion-valueonly` next to
`fusion-crossattn`, `clean-mlp` next to `dann1.0`), report every seed, and report the
budget curve. A mechanism that only helps inside one normalization is a representation
result, not a mechanism result.

### Bug fixes, docs, tests

Please do.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -q                 # see docs/TESTING.md for the host-bound exceptions
& ./scripts/Invoke-Pytest.ps1 -Suite Unit
```

Style: keep the existing shape of the code — `from __future__ import annotations`,
dataclasses or pydantic for schema objects, canonical serialization for anything hashed,
and no absolute paths in configuration files.

## Reporting results

When you report a result, include:

- protocol key and hash, seed set, and the exact representation key;
- the budget in epochs as a curve, not a single point;
- pooled OOF and fold-macro, with standard deviation across seeds;
- the per-group decomposition (per variety / per patient), since pooled numbers can be
  dominated by a single held-out group;
- for anything classified as a claim, the artefacts that back it.

## What not to send

- Results produced outside the governed pipeline presented as evidence. Mark them
  `exploratory`, exactly as `experiments/` does.
- Redistributed dataset files. The datasets are licensed separately and are not part of
  this repository.
- Code that silently widens the evaluation budget, opens an outer-test lease early, or
  writes fitted state outside its fold.
