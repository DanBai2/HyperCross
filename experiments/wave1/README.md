# Wave-1: cross-variety (LOVO) HyperLeaf screening

Exploratory lane. Not governed benchmark evidence.

`hyperleaf_lovo_screen.py` trains a small candidate zoo under the governed LOVO
split definition (leave one cultivar out; 3 train varieties -> 1 held-out
variety). Fitted state uses train-fold rows only; the official test partition
and `solution.csv` are never opened.

Run:

    conda run --no-capture-output -n TCFM python hyperleaf_lovo_screen.py --seeds 42 --epochs 200 --out G:\Datasets\HyperCross\runs_v3\wave1 --tag screen-seed42

Outputs: `<out>/<tag>_summary.json` and `<out>/<tag>_summary.md`.

Candidate roles:

- `ridge-spec`, `mlp-spec`   : spectral-only reference
- `ridge-phys`, `mlp-phys`   : physics-informed spectral features
- `cnn-rgb`                  : morphology-only reference
- `fusion-concat`            : plain concatenation control
- `fusion-valueonly`         : attention replaced by a linear value path (control)
- `fusion-crossattn`         : genuine patch-token x band-token cross-attention
- `dann-spec`                : variety-adversarial domain generalisation

The `fusion-*` triple is the pre-registered causal test of the paper''s
headline mechanism.
