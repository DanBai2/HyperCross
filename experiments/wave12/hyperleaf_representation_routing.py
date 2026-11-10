"""Wave-12: corrected MSC, a chemometric latent representation, and honest
per-target spectral-representation routing.

EXPLORATORY LANE.

Wave-11 established (outer LOVO, 200 epochs, 3 seeds):
  * per-sample SNV (`snv`, 204 bands) gives pooled-OOF Gsw +0.1075 / GrainWeight
    0.6252 / PhiPS2 0.4800, versus `fold-z` -0.1191 / 0.6375 / 0.4144 — the
    largest single gain of the campaign;
  * delta(DANN - clean) is large exactly where the clean model is poor and
    collapses to ~0 once the removable level nuisance is removed by SNV;
  * its `msc-z` and `snv-msc-z` rows were identical, which is a defect of the
    wave-11 transform dispatch (MSC was applied to already-SNV'd spectra).

Wave-12 therefore (a) corrects MSC, (b) adds a fold-fitted PCA latent
representation, and (c) asks whether the per-target representation route that
wave-11 picks using OUTER metrics can be recovered honestly using inner-variety
evidence only.

Representations (clean MLP, 200 epochs, seeds 7/42/123):
  `fold-z`      fold-fitted per-band z-score of the raw 204-band mean spectrum
  `snv`         per-sample SNV
  `snv-pca32`   per-sample SNV then fold-fitted 32-component PCA, then fold z-score
  `msc`         multiplicative scatter correction against the train-mean spectrum
  `snv-d1-cr-z` per-sample SNV with first derivative and continuum removal

Routing protocol (no outer metric is used for selection):
  for each outer fold (held-out variety V) the inner instrument trains on TWO of
  the remaining three varieties and validates on the third; the per-target
  representation argmax over the three inner folds is then evaluated on V.
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
    LEAF_ROOT, TARGETS, VAR_COLS, WORK_ROOT, build_cache, per_target_r2, standardize, Mlp,
)
from hyperleaf_dann_sweep import fit_torch2  # noqa: E402

VARIANTS = ("fold-z", "snv", "snv-pca32", "msc", "snv-d1-cr-z")
PCA_COMPONENTS = 32


def sample_snv(values):
    mean = values.mean(1, keepdims=True)
    scale = values.std(1, keepdims=True)
    return ((values - mean) / np.where(scale > 1e-12, scale, 1.0)).astype(np.float32)


def moving_average(values, window=21):
    pad = np.pad(values, ((0, 0), (window // 2, window // 2)), mode="edge")
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, pad)


def msc_apply(values, reference):
    centered = reference - reference.mean()
    denominator = float((centered ** 2).sum()) + 1e-12
    out = np.empty_like(values, dtype=np.float64)
    for index, row in enumerate(values):
        row_centered = row - row.mean()
        slope = float((row_centered * centered).sum()) / denominator
        intercept = float(row.mean() - slope * reference.mean())
        out[index] = (row - intercept) / (slope if abs(slope) > 1e-6 else 1.0)
    return out.astype(np.float32)


def pca_fit_project(train, test, components=PCA_COMPONENTS):
    mean = train.mean(0, keepdims=True)
    centered = train - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: int(components)]
    return ((train - mean) @ basis.T).astype(np.float32), ((test - mean) @ basis.T).astype(np.float32)


def prepare(variant, spec, tr_idx, te_idx):
    """Fold-safe features for one representation."""
    if variant == "fold-z":
        fitted, applied = standardize({"s": spec[tr_idx]}, {"s": spec[te_idx]}, ["s"])
        return fitted["s"], applied["s"]
    if variant == "snv":
        return sample_snv(spec[tr_idx]), sample_snv(spec[te_idx])
    if variant == "msc":
        reference = spec[tr_idx].mean(0)
        train = msc_apply(spec[tr_idx], reference)
        test = msc_apply(spec[te_idx], reference)
        fitted, applied = standardize({"s": train}, {"s": test}, ["s"])
        return fitted["s"], applied["s"]
    if variant == "snv-pca32":
        train = sample_snv(spec[tr_idx])
        test = sample_snv(spec[te_idx])
        train, test = pca_fit_project(train, test)
        fitted, applied = standardize({"s": train}, {"s": test}, ["s"])
        return fitted["s"], applied["s"]
    if variant == "snv-d1-cr-z":
        train = sample_snv(spec[tr_idx])
        test = sample_snv(spec[te_idx])
        train = np.concatenate([train, np.gradient(train, axis=1), train - moving_average(train)], 1).astype(np.float32)
        test = np.concatenate([test, np.gradient(test, axis=1), test - moving_average(test)], 1).astype(np.float32)
        fitted, applied = standardize({"s": train}, {"s": test}, ["s"])
        return fitted["s"], applied["s"]
    raise ValueError(f"unknown variant: {variant!r}")


def fit_clean(train, test, targets, seed, device, epochs):
    return fit_torch2(lambda: Mlp(train.shape[1], 3), {"spec": train, "y": targets}, {"spec": test}, seed, device, epochs=epochs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int, default=0, help="debug: restrict to the first N outer folds")
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave12")
    ap.add_argument("--tag", default="representation-routing")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    variants = [x for x in args.variants.split(",") if x]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} seeds {seeds} epochs {args.epochs} variants {variants}", flush=True)

    spec, _rgb = build_cache(WORK_ROOT / "cache_leaf_spec_rgb.npz")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    y = df[list(TARGETS)].to_numpy(np.float32)
    n = len(y)
    masks = {v: df[v].astype(bool).to_numpy() for v in VAR_COLS}
    all_idx = np.arange(n)
    print(f"n={n} spec={spec.shape}", flush=True)

    outer_folds = [(v, np.where(~masks[v])[0], np.where(masks[v])[0]) for v in VAR_COLS]
    if args.limit:
        outer_folds = outer_folds[: args.limit]

    outer_pred = {v: {s: np.full((n, 3), np.nan) for s in seeds} for v in variants}
    inner_pred = {v: {s: {} for s in seeds} for v in variants}
    outer_metrics = {v: {} for v in variants}

    for variant in variants:
        per_seed_outer = []
        for seed in seeds:
            for _held, tr_idx, te_idx in outer_folds:
                train, test = prepare(variant, spec, tr_idx, te_idx)
                mean = y[tr_idx].mean(0, keepdims=True)
                scale = y[tr_idx].std(0, keepdims=True) + 1e-6
                normalized = ((y[tr_idx] - mean) / scale).astype(np.float32)
                predicted = fit_clean(train, test, normalized, seed, device, args.epochs)
                outer_pred[variant][seed][te_idx] = predicted * scale + mean
            valid = ~np.isnan(outer_pred[variant][seed][:, 0])
            pooled = per_target_r2(y[valid], outer_pred[variant][seed][valid])
            per_seed_outer.append(pooled)
            print(f"  [{variant}/seed{seed}] outer pooled Gsw={pooled['Gsw']} GrainWeight={pooled['GrainWeight']}", flush=True)
            # inner instrument: for every outer fold train on two of the other three varieties
            for held_index, (held, pool_idx, _te) in enumerate(outer_folds):
                pool_varieties = [v for v in VAR_COLS if v != held]
                for inner in pool_varieties:
                    inner_te = np.where(masks[inner])[0]
                    inner_tr = np.asarray([i for i in pool_idx if i not in set(inner_te.tolist())], dtype=np.int64)
                    train, test = prepare(variant, spec, inner_tr, inner_te)
                    mean = y[inner_tr].mean(0, keepdims=True)
                    scale = y[inner_tr].std(0, keepdims=True) + 1e-6
                    normalized = ((y[inner_tr] - mean) / scale).astype(np.float32)
                    predicted = fit_clean(train, test, normalized, seed, device, args.epochs)
                    inner_pred[variant][seed][(held, inner)] = (inner_te, predicted * scale + mean)
        outer_metrics[variant] = {
            "pooled_r2_mean_over_seeds": {t: round(float(np.mean([row[t] for row in per_seed_outer])), 4) for t in TARGETS},
            "pooled_r2_std_over_seeds": {t: round(float(np.std([row[t] for row in per_seed_outer])), 4) for t in TARGETS},
            "per_seed_pooled": per_seed_outer,
        }

    # inner instrument score per variant / target / outer fold (mean over seeds)
    inner_score = {t: {v: {} for v in variants} for t in TARGETS}
    for held, _pool, _te in outer_folds:
        pool_varieties = [v for v in VAR_COLS if v != held]
        for variant in variants:
            per_seed = []
            for seed in seeds:
                truth = np.concatenate([y[inner_pred[variant][seed][(held, w)][0]] for w in pool_varieties])
                pred = np.concatenate([inner_pred[variant][seed][(held, w)][1] for w in pool_varieties])
                per_seed.append(per_target_r2(truth, pred))
            for t in TARGETS:
                inner_score[t][variant][held] = round(float(np.mean([row[t] for row in per_seed])), 4)

    route = {t: {held: max(variants, key=lambda v: inner_score[t][v][held]) for held, _p, _t in outer_folds} for t in TARGETS}

    routed_per_seed = []
    for seed in seeds:
        routed = np.full((n, 3), np.nan)
        for held, _pool, te_idx in outer_folds:
            for target_index, t in enumerate(TARGETS):
                chosen = route[t][held]
                routed[te_idx, target_index] = outer_pred[chosen][seed][te_idx, target_index]
        valid = ~np.isnan(routed[:, 0])
        routed_per_seed.append(per_target_r2(y[valid], routed[valid]))
    routed_summary = {
        "pooled_r2_mean_over_seeds": {t: round(float(np.mean([row[t] for row in routed_per_seed])), 4) for t in TARGETS},
        "per_seed_pooled": routed_per_seed,
    }

    best_single = max(variants, key=lambda v: outer_metrics[v]["pooled_r2_mean_over_seeds"]["Gsw"])
    oracle = {
        t: round(max(outer_metrics[v]["pooled_r2_mean_over_seeds"][t] for v in variants), 4) for t in TARGETS
    }
    summary = {
        "device": str(device), "seeds": seeds, "epochs": args.epochs, "variants": variants,
        "n_samples": int(n), "tag": args.tag, "evidence_grade": "exploratory",
        "outer_metrics": outer_metrics, "inner_score": inner_score, "selected_route": route,
        "routed_policy": routed_summary, "best_single_variant_for_gsw": best_single,
        "outer_metric_oracle_reference_only": oracle,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Wave-12 representation routing ({args.tag})",
        "",
        f"- device `{device}`, seeds {seeds}, epochs {args.epochs}, n={n}, outer LOVO",
        "- `msc` now applies scatter correction to the raw spectrum (wave-11 dispatch defect fixed); `snv-pca32` is new",
        "",
        "## Single-representation pooled OOF R2 (mean +/- std over seeds)",
        "",
        "| representation | GrainWeight | Gsw | PhiPS2 |",
        "|---|---:|---:|---:|",
    ]
    for variant in variants:
        row = outer_metrics[variant]["pooled_r2_mean_over_seeds"]
        std = outer_metrics[variant]["pooled_r2_std_over_seeds"]
        lines.append(
            f"| `{variant}` | {row['GrainWeight']} +/- {std['GrainWeight']} | {row['Gsw']} +/- {std['Gsw']} | {row['PhiPS2']} +/- {std['PhiPS2']} |"
        )
    lines += ["", "## Inner-variety selected per-target route (no outer metric used)", "", "| target | selected representation per outer fold |", "|---|---|"]
    for t in TARGETS:
        cells = ", ".join(f"{held}:`{route[t][held]}`" for held, _p, _t in outer_folds)
        lines.append(f"| {t} | {cells} |")
    lines += [
        "", "## Routed policy vs a-priori single representation (outer LOVO pooled OOF R2)", "",
        "| policy | GrainWeight | Gsw | PhiPS2 |",
        "|---|---:|---:|---:|",
    ]
    routed_row = routed_summary["pooled_r2_mean_over_seeds"]
    lines.append(f"| inner-selected per-target route | {routed_row['GrainWeight']} | {routed_row['Gsw']} | {routed_row['PhiPS2']} |")
    for variant in variants:
        row = outer_metrics[variant]["pooled_r2_mean_over_seeds"]
        lines.append(f"| fixed `{variant}` | {row['GrainWeight']} | {row['Gsw']} | {row['PhiPS2']} |")
    lines += [
        f"| outer-oracle per target (reference only) | {oracle['GrainWeight']} | {oracle['Gsw']} | {oracle['PhiPS2']} |",
        "",
    ]
    (out_dir / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
