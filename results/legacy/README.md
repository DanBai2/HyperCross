# Legacy results — do NOT compare with the governed numbers

These files predate the governed benchmark and were computed under **patient-grouped
or random splits**. They are kept only for provenance.

In particular:

- `CCF_A_COMPARISON.md` reports Histology Local Route A accuracy 0.9189 and an
  official-test accuracy of 0.4550. Those come from patient-grouped and random splits
  respectively and are **not** comparable with the 13-fold LOPO protocol used in
  `RESULTS.md`.
- `benchmark_summary.json`, `benchmark_table.csv`, `patient_cv_metrics.json`,
  `balanced_cv_metrics.json`, `hist_ensemble_results.json`, `leaf_multiseed.json`,
  `mil_eval.json`, `modern_best_summary.json` are pre-benchmark aggregations with the
  same caveat.
- `EXPERIMENT_DECISION_LOG.md` and `top_models_research.md` record the reasoning of the
  earlier phase and include the remote-sensing scene numbers that were later dropped.

Never place these numbers side by side with the governed LOVO / LOPO results.
