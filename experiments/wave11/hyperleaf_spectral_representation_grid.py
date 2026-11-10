"""Wave-11: which spectral representation carries cross-variety phenotypic signal,
and does domain identifiability predict the sign of adversarial alignment?

EXPLORATORY LANE -- not governed benchmark evidence.

Motivation (2026-11-08):
  * governed task 24.2/24.3 candidate spaces contain NO spectrum-only candidate and
    NEVER treat the spectral representation as an optimization axis; the 24.3
    screening schedule is 20 epochs (promotion 80), and the governed spectrum-only
    arm still gains +0.12 pooled-OOF Gsw R2 between 60 and 200 epochs;
  * wave-10 showed the sign of variety-adversarial alignment flips with the
    spectral normalization (fold-fitted per-band z-score -> helps, per-sample SNV
    -> hurts), independent of the band count, and that per-sample SNV alone is
    worth ~+0.16 Gsw R2.

Wave-11 therefore optimises the axis the governed stack never optimised and, in the
same pass, measures a candidate criterion for the entanglement claim.

Representations (all from the cached 204-band mean spectrum, outer LOVO):
  `fold-z-raw`   fold-fitted per-band z-score (wave-1/9 reference)
  `snv-raw`      per-sample SNV (wave-10 / governed reference)
  `snv-z`        per-sample SNV, then fold-fitted per-feature z-score
  `snv-d1-z`     + first derivative along bands
  `snv-cr-z`     + continuum-removal proxy (SNV minus 21-band moving average)
  `snv-d1-cr-z`  + derivative and continuum removal
  `msc-z`        multiplicative scatter correction against the train-mean spectrum
  `snv-msc-z`    SNV then MSC

For every representation wave-11 reports
  (a) a ridge linear readout score (no neural net, fold-fitted alpha),
  (b) an MLP score, and
  (c) a variety-adversarial (DANN lambda 1.0) score,
plus one diagnostic: in-domain variety identifiability from the representation
(ridge one-hot decoding on a held-out split of the training pool).

Pre-registered predictions (recorded before execution):
  Q1 some representation other than `fold-z-raw` beats it on pooled-OOF Gsw by
     at least +0.03 with a positive GrainWeight delta.
  Q2 delta(DANN - clean) is positive exactly where in-domain variety
     identifiability is high and the clean linear readout of the target is weak.
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
    ridge_fit_predict,
    standardize,
    Mlp,
)
from hyperleaf_dann_sweep import DannSpectral, fit_torch2  # noqa: E402

FAMILIES = ("clean-mlp", "dann1.0")
VARIANTS = (
    "fold-z-raw",
    "snv-raw",
    "snv-z",
    "snv-d1-z",
    "snv-cr-z",
    "snv-d1-cr-z",
    "msc-z",
    "snv-msc-z",
)


def sample_snv(values: np.ndarray) -> np.ndarray:
    mean = values.mean(1, keepdims=True)
    scale = values.std(1, keepdims=True)
    return ((values - mean) / np.where(scale > 1e-12, scale, 1.0)).astype(np.float32)


def moving_average(values: np.ndarray, window: int = 21) -> np.ndarray:
    pad = np.pad(values, ((0, 0), (window // 2, window // 2)), mode="edge")
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, pad)


def msc_apply(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Multiplicative scatter correction against a reference spectrum."""
    centered = reference - reference.mean()
    denominator = float((centered ** 2).sum()) + 1e-12
    out = np.empty_like(values, dtype=np.float64)
    for index, row in enumerate(values):
        row_centered = row - row.mean()
        slope = float((row_centered * centered).sum()) / denominator
        intercept = float(row.mean() - slope * reference.mean())
        out[index] = (row - intercept) / (slope if abs(slope) > 1e-6 else 1.0)
    return out.astype(np.float32)


def per_sample_transform(spec: np.ndarray, variant: str) -> np.ndarray:
    """Fold-independent part of one representation."""
    if variant.startswith("snv") or variant.startswith("msc") or variant == "snv-msc-z":
        base = sample_snv(spec)
    else:
        base = np.asarray(spec, dtype=np.float32)
    blocks = [base]
    if "d1" in variant:
        blocks.append(np.gradient(base, axis=1).astype(np.float32))
    if "cr" in variant:
        blocks.append((base - moving_average(base)).astype(np.float32))
    return np.concatenate(blocks, 1) if len(blocks) > 1 else base


def prepare(variant: str, spec: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test) features for one representation, fitted on the train fold only."""
    if variant == "fold-z-raw":
        fitted, applied = standardize({"s": spec[tr_idx]}, {"s": spec[te_idx]}, ["s"])
        return fitted["s"], applied["s"]
    base = per_sample_transform(spec, variant)
    train, test = base[tr_idx], base[te_idx]
    if variant == "msc-z":
        reference = train.mean(0)
        train = msc_apply(train, reference)
        test = msc_apply(test, reference)
    elif variant == "snv-msc-z":
        reference = train.mean(0)
        train = msc_apply(train, reference)
        test = msc_apply(test, reference)
    if variant.endswith("-z") and variant != "fold-z-raw":
        fitted, applied = standardize({"s": train}, {"s": test}, ["s"])
        train, test = fitted["s"], applied["s"]
    return train.astype(np.float32), test.astype(np.float32)


def fit_family(family, train, test, targets, domains, seed, device, epochs):
    if family == "clean-mlp":
        return fit_torch2(lambda: Mlp(train.shape[1], 3), {"spec": train, "y": targets}, {"spec": test}, seed, device, epochs=epochs)
    if family == "dann1.0":
        return fit_torch2(
            lambda: DannSpectral(n_in=train.shape[1], lam=1.0),
            {"x": train, "y": targets, "dom": domains}, {"x": test}, seed, device, epochs=epochs, dann=True,
        )
    raise ValueError(f"unknown family: {family!r}")


def domain_identifiability(train: np.ndarray, domains: np.ndarray, seed: int = 0) -> float:
    """Out-of-sample accuracy of a ridge one-hot variety decoder on the training pool."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    cut = int(0.8 * len(train))
    fit_idx, hold_idx = order[:cut], order[cut:]
    if len(np.unique(domains[fit_idx])) < 2 or len(hold_idx) == 0:
        return float("nan")
    onehot = np.zeros((len(domains), int(domains.max()) + 1), dtype=np.float64)
    onehot[np.arange(len(domains)), domains] = 1.0
    predicted, _alpha = ridge_fit_predict(train[fit_idx], onehot[fit_idx], train[hold_idx], seed=seed)
    return float((predicted.argmax(1) == domains[hold_idx]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave11")
    ap.add_argument("--tag", default="spectral-representation-grid")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    families = [x for x in args.families.split(",") if x]
    variants = [x for x in args.variants.split(",") if x]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} seeds {seeds} epochs {args.epochs} variants {len(variants)}", flush=True)

    spec, _rgb = build_cache(WORK_ROOT / "cache_leaf_spec_rgb.npz")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    y = df[list(TARGETS)].to_numpy(np.float32)
    dom = df[list(VAR_COLS)].to_numpy(np.float32).argmax(1).astype(np.int64)
    folds = []
    for variety in VAR_COLS:
        held = df[variety].astype(bool).to_numpy()
        folds.append((np.where(~held)[0], np.where(held)[0]))
    print(f"n={len(y)} spec={spec.shape} folds={len(folds)}", flush=True)

    store: dict[str, dict[str, object]] = {}
    for variant in variants:
        entry: dict[str, object] = {"n_features": None, "domain_identifiability": None, "linear_readout": None, "families": {}}
        identifiability: list[float] = []
        linear_predictions: list[np.ndarray] = []
        linear_order: list[np.ndarray] = []
        for tr_idx, te_idx in folds:
            train, test = prepare(variant, spec, tr_idx, te_idx)
            entry["n_features"] = int(train.shape[1])
            identifiability.append(domain_identifiability(train, dom[tr_idx]))
            mean = y[tr_idx].mean(0, keepdims=True)
            scale = y[tr_idx].std(0, keepdims=True) + 1e-6
            normalized = ((y[tr_idx] - mean) / scale).astype(np.float64)
            predicted, _alpha = ridge_fit_predict(train.astype(np.float64), normalized, test.astype(np.float64), seed=0)
            linear_predictions.append(predicted * scale + mean)
            linear_order.append(te_idx)
        entry["domain_identifiability"] = round(float(np.mean(identifiability)), 4)
        entry["linear_readout"] = per_target_r2(y[np.concatenate(linear_order)], np.concatenate(linear_predictions))
        print(f"  [{variant}] linear readout Gsw={entry['linear_readout']['Gsw']} domain-identifiability={entry['domain_identifiability']}", flush=True)

        for family in families:
            per_seed: list[dict[str, float]] = []
            for seed in seeds:
                predictions: list[np.ndarray] = []
                order: list[np.ndarray] = []
                for tr_idx, te_idx in folds:
                    train, test = prepare(variant, spec, tr_idx, te_idx)
                    mean = y[tr_idx].mean(0, keepdims=True)
                    scale = y[tr_idx].std(0, keepdims=True) + 1e-6
                    normalized = ((y[tr_idx] - mean) / scale).astype(np.float32)
                    predicted = fit_family(family, train, test, normalized, dom[tr_idx], seed, device, args.epochs)
                    predictions.append(predicted * scale + mean)
                    order.append(te_idx)
                pooled = per_target_r2(y[np.concatenate(order)], np.concatenate(predictions))
                per_seed.append(pooled)
                print(f"  [{variant}/{family}/seed{seed}] pooled Gsw={pooled['Gsw']} GrainWeight={pooled['GrainWeight']}", flush=True)
            entry["families"][family] = {
                "pooled_r2_mean_over_seeds": {t: round(float(np.mean([row[t] for row in per_seed])), 4) for t in TARGETS},
                "pooled_r2_std_over_seeds": {t: round(float(np.std([row[t] for row in per_seed])), 4) for t in TARGETS},
                "per_seed_pooled": per_seed,
            }
        store[variant] = entry

    deltas: dict[str, dict[str, float]] = {}
    for variant in variants:
        families_here = store[variant]["families"]
        if "clean-mlp" in families_here and "dann1.0" in families_here:
            deltas[variant] = {
                t: round(families_here["dann1.0"]["pooled_r2_mean_over_seeds"][t] - families_here["clean-mlp"]["pooled_r2_mean_over_seeds"][t], 4)
                for t in TARGETS
            }

    summary = {
        "device": str(device), "seeds": seeds, "epochs": args.epochs, "families": families,
        "variants": variants, "n_samples": int(len(y)), "tag": args.tag, "evidence_grade": "exploratory",
        "results": store, "dann_minus_clean": deltas,
        "pre_registered_predictions": [
            "Q1 some representation other than fold-z-raw beats it on pooled-OOF Gsw by at least +0.03 with a positive GrainWeight delta",
            "Q2 delta(DANN - clean) is positive exactly where in-domain variety identifiability is high and the clean target readout is weak",
        ],
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Wave-11 spectral representation grid ({args.tag})",
        "",
        f"- device `{device}`, seeds {seeds}, epochs {args.epochs}, n={len(y)}, outer LOVO",
        "- `clean-mlp` = plain MLP; `dann1.0` = same trunk plus variety-adversarial head (GRL lambda 1.0)",
        "",
        "## Pooled OOF R2 (mean over seeds)",
        "",
        "| representation | features | domain-ident. | linear Gsw | clean GrainWeight | clean Gsw | clean PhiPS2 | dann Gsw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        entry = store[variant]
        clean = entry["families"].get("clean-mlp", {}).get("pooled_r2_mean_over_seeds")
        dann = entry["families"].get("dann1.0", {}).get("pooled_r2_mean_over_seeds")
        lines.append(
            f"| `{variant}` | {entry['n_features']} | {entry['domain_identifiability']} | {entry['linear_readout']['Gsw']} | "
            f"{clean['GrainWeight'] if clean else 'n/a'} | {clean['Gsw'] if clean else 'n/a'} | {clean['PhiPS2'] if clean else 'n/a'} | "
            f"{dann['Gsw'] if dann else 'n/a'} |"
        )
    lines += ["", "## delta(DANN - clean) on pooled OOF R2", "", "| representation | GrainWeight | Gsw | PhiPS2 |", "|---|---:|---:|---:|"]
    for variant in variants:
        if variant in deltas:
            row = deltas[variant]
            lines.append(f"| `{variant}` | {row['GrainWeight']} | {row['Gsw']} | {row['PhiPS2']} |")
    lines += ["", "## Pre-registered predictions", ""]
    for prediction in summary["pre_registered_predictions"]:
        lines.append(f"- {prediction}")
    lines.append("")
    (out_dir / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
