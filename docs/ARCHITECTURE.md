# Architecture

This document is the shortest path to understanding how the system is put together and
where to make a change. The authoritative but much longer (Chinese-language) design
rationale lives in `specs/design.md`, `specs/requirements.md` and `specs/tasks.md`.

## 1. What the system guarantees

Everything else in the repository exists to make these six properties hold. If you
change code, keep them true.

| # | Invariant | Enforced by |
|---|---|---|
| 1 | **Group-disjoint evaluation only.** HyperLeaf is always leave-one-variety-out, Histology always leave-one-patient-out. Random splits are never used for a claim. | `benchmark/protocols/` |
| 2 | **Fold-scoped fitting.** Normalization, PCA, target scaling, thresholds and temperature are estimated from training-fold samples only and recorded with their `Fit_Sample_IDs`. | `benchmark/transforms/preprocessing.py`, `benchmark/schema/preprocessing.py` |
| 3 | **Nested selection.** Candidate, route and epoch selection see inner folds only; outer labels are released once, behind an evaluation lease. | `benchmark/engine/selection.py`, `benchmark/engine/evaluation.py` |
| 4 | **Bound execution.** Runs are frozen identities: seed set, budget, precision, determinism, device, resource limits and output roots are hashed before training starts and cannot drift on resume. | `benchmark/fast_lane.py`, `benchmark/remediation.py`, `benchmark/schema/configuration.py` |
| 5 | **Append-only evidence.** Runs, artefacts and aggregates are never mutated in place; a later run supersedes an earlier one explicitly. | `benchmark/registry/run_registry.py`, `benchmark/recovery/artifact_store.py` |
| 6 | **Explicit claim tier.** Nothing is evidence until the claim gate classifies it. | `benchmark/reporting/claim_gate.py`, `benchmark/reporting/promotion.py` |

Two consequences worth knowing up front:

- The workspace is pinned to `G:\Datasets\HyperCross`. The HyperLeaf adapter rejects any
  other drive (`benchmark/catalog/hyperleaf.py`), and `benchmark/cli.py` validates the
  workspace root. See `docs/TESTING.md` and `tools/relocate_data_root.ps1`.
- The official HyperLeaf test split has an **evaluation budget**, currently 0. It is meant
  to be spent once, at the end, not during development.

## 2. Module map

```
benchmark/
  schema/       canonical, hashable data models: protocol, predictions, selection,
                metrics, aggregation, preprocessing, configuration, environment,
                resources, candidate_space, records, datasets, evaluation, models, base
  catalog/      dataset semantics and immutable sample manifests
                hyperleaf / materialization / raw_cube (chunked, memory-mapped)
                deidentification / adapters
  protocols/    planner + the three protocol keys
                leaf.lovo.nested.v1 | hist.lopo-fixed.v1 | hist.patient-nested5.v1
  transforms/   fold-scoped preprocessing and the lineage-checked feature cache
  models/       stable Model_Key -> implementation + Input_Contract + provenance
                registry / hyperleaf_* / histology* / route_a / checkpoint /
                reproduction / nuisance_alignment
  engine/       training | selection | evaluation | orchestration
  metrics/      predictions | engine | aggregate | statistics
  registry/     run_registry (append-only) | records | candidate_space | error_journal
  recovery/     artifact_store (failure-atomic) | manager (resume, missing jobs)
  reporting/    claim_gate | promotion | matrix | publication
  runners/      per-task entry points (see below)
  fast_lane.py  frozen stage templates: smoke, baseline, screening, strict, official
  remediation.py  separate immutable precision-namespace templates (float32 / bf16)
  cli.py        plan | run | resume | smoke | delegated | model_resolve  (+dry-run)
  cli_runtime.py / audit/ / legacy/ / adapters/
```

Runner families: `hyperleaf_{smoke,screening,strict,baseline,lovo,candidates,`
`candidate_review,inner_optimization,optimization_registry,nested_remediation,`
`nuisance_alignment,route_a}` and the `histology_*` equivalents.

## 3. End-to-end flow

```
plan ─────► protocol manifest + frozen run identity (hashed, no files written)
  │
  ├─ run ──► inner training on outer-train only ──► recovery checkpoints (per epoch)
  │            │
  │            └─► selection freeze: candidate / route / epoch chosen on inner folds
  │
  ├─ evaluate ─► evaluation lease opens ─► one-shot outer evaluation ─► predictions
  │
  ├─ aggregate ► canonical shard merge ─► pooled OOF, fold-macro, bootstrap CI
  │
  ├─ claim gate ► tier assigned (development | strict_candidate | primary_strict ...)
  │
  └─ publish ───► single publication matrix rendered to several formats,
                  cross-format validated, transactional write
```

Every stage writes under the stage's own namespace (`runs_v2/<stage>/...`,
`artifacts/...`) and resumes only from a workspace-relative recovery source.

## 4. Where results live

| tier | location | may be cited as evidence? |
|---|---|---|
| governed | `runs_v2/`, `results/governed/` | yes, at the tier the claim gate assigned |
| exploratory | `experiments/wave*`, `results/exploratory/` | no — hypothesis generation only |
| legacy | `results/legacy/`, `src/` | no — random/patient splits, not comparable |

## 5. Cookbook: extending the system

**Add a spectral representation.** Representations are built in
`benchmark/transforms/preprocessing.py` and referenced by a `Representation_Key`
declared in `benchmark/catalog/hyperleaf.py`. Declare the key, implement it fold-scoped
if it has fitted state, and add it to the candidate space. Then re-run the grid style of
`experiments/wave11/hyperleaf_spectral_representation_grid.py` to see where it lands.

**Add a candidate.** The HyperLeaf candidate space is built in
`benchmark/runners/hyperleaf_optimization_registry.py` (`_build_candidates`, roughly
lines 377-430); the registry surface is `benchmark/registry/candidate_space.py` plus
`benchmark/schema/candidate_space.py`. Pre-register the criteria before executing; see
`configs/optimization/v2/hyperleaf-task-25-preregistration.json` for the shape of a
frozen pre-registration, and `benchmark/runners/hyperleaf_nuisance_alignment.py` (842
lines) for a self-contained modern runner with freeze / execute / resume / smoke / arm
selection and automatic criterion verdicts. That pair is the best template to copy.

**Add a model.** Register a `Model_Key` in `benchmark/models/registry.py` with its class,
input contract and provenance (`official` / `adapted` / `proxy`). The adapter tests in
`tests/unit/test_hyperleaf_route_a_adapter.py` show the expectations.

**Add a metric.** `benchmark/metrics/engine.py` for the definition,
`benchmark/metrics/aggregate.py` for pooling, `benchmark/schema/metrics.py` for the
schema, `benchmark/reporting/matrix.py` for exposure in the publication matrix.

### Three lessons this repository paid for

1. **Declaration must equal implementation.** A pre-registration once declared
   `per-band-zscore-inner-train.v1` while the runner actually applied per-sample SNV.
   Two "successful" conclusions rested on that mismatch. Assert the key you register
   against the array you feed the model.
2. **Screen caps are a hypothesis.** Task 24.2/24.3 screened at 20 epochs and promoted at
   80, which systematically removed simple late-converging models — including the
   per-sample SNV baseline that turned out to be the strongest configuration in the whole
   project. Always report a budget curve.
3. **Representation must be an explicit axis.** For three tasks the candidate space varied
   fusion, routing, loss and normalization but never the input representation, so the
   largest effect in the data was invisible to the search.

## 6. Glossary

| term | meaning |
|---|---|
| **Group_ID** | the isolation unit: wheat variety, or patient |
| **Outer fold** | one held-out group; used once, after everything is frozen |
| **Inner fold** | a split inside outer-train, used for selection only |
| **LOVO / LOPO** | leave-one-variety-out / leave-one-patient-out |
| **Representation_Key** | name of an input representation (RGB, global spectrum, PCA pseudo-RGB, full cube, ...) |
| **Fitted_Artifact** | anything estimated from data (normalization, PCA, threshold, temperature), with its Fit_Sample_IDs |
| **Selection_Decision** | the frozen record of candidate / route / epoch / threshold choice |
| **Evaluation_Budget** | declared maximum number of outer-test evaluations |
| **Claim tier** | required evidence strength before a number may be presented |
| **Pooled OOF** | all outer folds concatenated, then scored once |
| **Fold-macro** | per-fold metric averaged over folds |
