# Legacy code (superseded)

`legacy/src/` contains the pre-benchmark research scripts, including the original
paper model definition (`hypercross_routeA.py`) and the remote-sensing scene
experiments (Indian Pines / Salinas / Pavia / Houston2013, S2Mamba comparisons,
pseudo-labelling, test-time adaptation, and so on).

Status: **superseded**. The governed pipeline in `benchmark/` replaced these scripts,
and the remote-sensing track was dropped as unsuitable for the scientific-HSI
research question.

Two important caveats if you read this code:

1. The legacy cross-attention module is **degenerate**: each branch produces a single
   token, so the attention matrix is identically 1 and the Q/K gradients are exactly 0.
   The module reduces to a value projection. Do not cite it as bidirectional attention.
2. Results produced by these scripts use patient-grouped or random splits and are
   **not** comparable to the group-disjoint LOVO / LOPO numbers in `RESULTS.md`.

Kept in the release only for provenance and for anyone who wants to trace the history.
