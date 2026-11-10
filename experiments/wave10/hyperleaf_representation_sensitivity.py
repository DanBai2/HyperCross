"""Wave-10: is the sign of variety-adversarial spectral alignment set by the spectral representation?

EXPLORATORY LANE -- not governed benchmark evidence.

Established facts:
  * governed task-25.1 pipeline (128-band per-sample-SNV mean spectrum, 60 epochs,
    outer LOVO, seeds 7/42/123): delta(DANN 1.0 - clean MLP) on pooled-OOF Gsw =
    -0.0143, negative in all three seeds; the in-run RGB/PCA and train-fold-PCA
    references reproduce the governed task-24.2 Gsw range, so the harness is sound.
  * exploratory wave-9 harness (204-band fold-fitted per-band z-scored mean
    spectrum, same folds/seeds): delta = +0.0677 at 60 epochs and +0.1714 at
    200 epochs, positive in all three seeds at both budgets.

Changing the budget therefore changes the magnitude but not the sign; changing the
representation changes the sign.  Wave-10 holds the model, folds, seeds and budget
fixed and varies only the spectral normalization.

Variants:
  R1 `fold-z-204`     per-band z-score fitted on the training fold, 204 bands
                      (the wave-1 / wave-9 representation)
  R2 `sample-snv-204` per-sample SNV (subtract the sample mean, divide by its
                      standard deviation) over the 204 bands
  R3 `fold-z-128`     interpolate to 128 bands, then fold-fitted per-band z-score
  R4 `sample-snv-128` interpolate to 128 bands, then per-sample SNV
                      (the governed task-25.1 representation)

Pre-registered predictions (recorded before execution):
  Q1 R1 reproduces the wave-9 delta (positive in every seed).
  Q2 R4 reproduces the task-25.1 delta (negative in every seed).
  Q3 if R2 and R4 share a sign while R1 and R3 share the opposite sign, the
     sign of the mechanism is a property of the normalization, not of the band
     count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wave1"))
sys.path.insert(0, str(HERE.parent / "wave6"))

from hyperleaf_lovo_screen import (  # noqa: E402
    LEAF_ROOT,
    TARGETS,
    VAR_COLS,
    WORK_ROOT,
    build_cache,
    per_target_r2,
    standardize,
    Mlp,
)
from hyperleaf_dann_sweep import DannSpectral, fit_torch2  # noqa: E402

FAMILIES = ("clean-mlp", "dann1.0")
VARIANTS = ("fold-z-204", "sample-snv-204", "fold-z-128", "sample-snv-128")


def interpolate_to_bands(spec: np.ndarray, bands: int) -> np.ndarray:
    """Resample every row of `spec` onto `bands` positions in [0, 1]."""
    if spec.shape[1] == bands:
        return np.asarray(spec, dtype=np.float32)
    source = np.linspace(0.0, 1.0, spec.shape[1])
    target = np.linspace(0.0, 1.0, bands)
    return np.stack([np.interp(target, source, row) for row in spec]).astype(np.float32)


def sample_snv(values: np.ndarray) -> np.ndarray:
    """Per-sample standard-normal-variate normalization."""
    mean = values.mean(1, keepdims=True)
    scale = values.std(1, keepdims=True)
    return ((values - mean) / np.where(scale > 1e-12, scale, 1.0)).astype(np.float32)


def build_variant(spec: np.ndarray, variant: str) -> tuple[np.ndarray, bool]:
    """Return (matrix, is_fold_fitted) for one representation variant."""
    bands = 128 if variant.endswith("128") else 204
    matrix = interpolate_to_bands(spec, bands)
    if variant.startswith("fold-z"):
        return matrix, True
    if variant.startswith("sample-snv"):
        return sample_snv(matrix), False
    raise ValueError(f"unknown variant: {variant!r}")


def prepare(matrix: np.ndarray, fold_fitted: bool, tr_idx: np.ndarray, te_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if fold_fitted:
        fitted, applied = standardize({"s": matrix[tr_idx]}, {"s": matrix[te_idx]}, ["s"])
        return fitted["s"], applied["s"]
    return matrix[tr_idx], matrix[te_idx]


def fit_family(family: str, train: np.ndarray, test: np.ndarray, targets: np.ndarray, domains: np.ndarray, seed: int, device, epochs: int) -> np.ndarray:
    if family == "clean-mlp":
        return fit_torch2(
            lambda: Mlp(train.shape[1], 3),
            {"spec": train, "y": targets},
            {"spec": test},
            seed, device, epochs=epochs,
        )
    if family == "dann1.0":
        return fit_torch2(
            lambda: DannSpectral(n_in=train.shape[1], lam=1.0),
            {"x": train, "y": targets, "dom": domains},
            {"x": test},
            seed, device, epochs=epochs, dann=True,
        )
    raise ValueError(f"unknown family: {family!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave10")
    ap.add_argument("--tag", default="representation-sensitivity")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    families = [x for x in args.families.split(",") if x]
    variants = [x for x in args.variants.split(",") if x]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} seeds {seeds} epochs {args.epochs} variants {variants}", flush=True)

    spec, _rgb = build_cache(WORK_ROOT / "cache_leaf_spec_rgb.npz")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    y = df[list(TARGETS)].to_numpy(np.float32)
    dom = df[list(VAR_COLS)].to_numpy(np.float32).argmax(1).astype(np.int64)
    print(f"n={len(y)} spec={spec.shape}", flush=True)

    folds = []
    for variety in VAR_COLS:
        held = df[variety].astype(bool).to_numpy()
        folds.append((variety, np.where(~held)[0], np.where(held)[0]))

    store: dict[str, dict[str, dict[str, object]]] = {}
    for variant in variants:
        matrix, fold_fitted = build_variant(spec, variant)
        store[variant] = {}
        for family in families:
            per_seed_pooled: list[dict[str, float]] = []
            for seed in seeds:
                predictions: list[np.ndarray] = []
                order: list[np.ndarray] = []
                for _variety, tr_idx, te_idx in folds:
                    train, test = prepare(matrix, fold_fitted, tr_idx, te_idx)
                    mean = y[tr_idx].mean(0, keepdims=True)
                    scale = y[tr_idx].std(0, keepdims=True) + 1e-6
                    normalized = ((y[tr_idx] - mean) / scale).astype(np.float32)
                    predicted = fit_family(family, train, test, normalized, dom[tr_idx], seed, device, args.epochs)
                    predictions.append(predicted * scale + mean)
                    order.append(te_idx)
                pooled = per_target_r2(y[np.concatenate(order)], np.concatenate(predictions))
                per_seed_pooled.append(pooled)
                print(f"  [{variant}/{family}/seed{seed}] pooled Gsw={pooled['Gsw']}", flush=True)
            store[variant][family] = {
                "pooled_r2_mean_over_seeds": {t: round(float(np.mean([row[t] for row in per_seed_pooled])), 4) for t in TARGETS},
                "per_seed_pooled": per_seed_pooled,
                "n_in": int(matrix.shape[1]),
                "fold_fitted": bool(fold_fitted),
            }

    deltas: dict[str, dict[str, float]] = {}
    per_seed_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for variant in variants:
        if "clean-mlp" not in store[variant] or "dann1.0" not in store[variant]:
            continue
        clean = store[variant]["clean-mlp"]
        dann = store[variant]["dann1.0"]
        deltas[variant] = {
            t: round(dann["pooled_r2_mean_over_seeds"][t] - clean["pooled_r2_mean_over_seeds"][t], 4) for t in TARGETS
        }
        per_seed_deltas[variant] = {
            t: {str(seeds[i]): round(dann["per_seed_pooled"][i][t] - clean["per_seed_pooled"][i][t], 4) for i in range(len(seeds))}
            for t in TARGETS
        }

    summary = {
        "device": str(device), "seeds": seeds, "epochs": args.epochs,
        "families": families, "variants": variants, "n_samples": int(len(y)),
        "tag": args.tag, "evidence_grade": "exploratory",
        "results": store, "deltas": deltas, "per_seed_deltas": per_seed_deltas,
        "pre_registered_predictions": [
            "Q1 R1 fold-z-204 reproduces the wave-9 delta, positive in every seed",
            "Q2 R4 sample-snv-128 reproduces the task-25.1 delta, negative in every seed",
            "Q3 sign follows the normalization (SNV vs fold-fitted), not the band count",
        ],
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Wave-10 representation sensitivity ({args.tag})",
        "",
        f"- device `{device}`, seeds {seeds}, epochs {args.epochs}, n={len(y)}, outer LOVO",
        "- `clean-mlp` = plain MLP on the spectrum; `dann1.0` = same trunk plus a variety-adversarial head (GRL lambda 1.0)",
        "",
        "## Pooled OOF R2 (mean over seeds)",
        "",
        "| variant | n_in | fold-fitted | family | GrainWeight | Gsw | PhiPS2 |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for variant in variants:
        for family in families:
            if family not in store[variant]:
                continue
            row = store[variant][family]["pooled_r2_mean_over_seeds"]
            lines.append(
                f"| `{variant}` | {store[variant][family]['n_in']} | {store[variant][family]['fold_fitted']} | "
                f"`{family}` | {row['GrainWeight']} | {row['Gsw']} | {row['PhiPS2']} |"
            )
    lines += ["", "## delta(DANN - clean) on pooled OOF R2", "", "| variant | GrainWeight | Gsw | PhiPS2 | Gsw per seed |", "|---|---:|---:|---:|---|"]
    for variant in variants:
        if variant not in deltas:
            continue
        row = deltas[variant]
        gsw_seeds = ", ".join(f"{k}:{v}" for k, v in per_seed_deltas[variant]["Gsw"].items())
        lines.append(f"| `{variant}` | {row['GrainWeight']} | {row['Gsw']} | {row['PhiPS2']} | {gsw_seeds} |")
    lines += ["", "## Pre-registered predictions", ""]
    for prediction in summary["pre_registered_predictions"]:
        lines.append(f"- {prediction}")
    lines.append("")
    (out_dir / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
