# Results layout

```
governed/     claim-gate artefacts copied from the private workspace runs_v2/
exploratory/  wave summaries copied from runs_v3/
legacy/       superseded pre-benchmark results - NOT protocol-comparable
```

`governed/` contains task reports, promotion / claim records, protocol manifests,
environment locks, and pooled out-of-fold predictions for the two headline anchored
tasks. Bulk per-fold job trees, recovery journals and model checkpoints (~1.4 GB) are
deliberately excluded; they remain in the private workspace.

`exploratory/` contains the complete wave summary set (JSON + Markdown).

Every number in `README.md` and `RESULTS.md` can be traced to a file here.

## Privacy note

The governed Histology artefacts (`results/governed/.../task-15-1*/...`) contain
pseudonymous sample identifiers of the form `P1/ROI_01_C01_T`, exactly as defined by
the public HistologyHSI-GB release (13 patients P1..P13, tumour / non-tumour ROI
labels). No names, dates, or other direct identifiers are present. If your institution
requires stricter handling, remove those files before publishing.
