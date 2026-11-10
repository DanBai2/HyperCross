"""Wave-7: honest target-routed and ensembled adversarial spectral models.

EXPLORATORY LANE -- not governed benchmark evidence.

Motivation (waves 1-6):
  * genuine cross-attention never helps (3 independent waves);
  * raw hand-crafted physics features hurt;
  * post-hoc calibration is useless on the governed baseline;
  * the ONLY reproducible positive mechanism is variety-adversarial training of
    the SPECTRAL branch, and Gsw is monotone in the gradient-reversal strength
    (lambda 0.1 -> -0.073, 0.3 -> -0.020, 1.0 -> +0.052, all seeds positive at 1.0);
  * adversarial training on the FUSED representation destroys the model.

Wave-7 asks the practical question: can a per-target ROUTE, chosen honestly from
inner varieties only, combine the clean branch (best GW/PhiPS2) with the
adversarial branch (best Gsw) and beat every fixed configuration?

  * ensemble  : mean of {clean, dann-0.3, dann-1.0}
  * routed    : per outer fold, pick the best family PER TARGET by inner LOVO
                over the three training varieties (held-out variety never used)

Pre-registered criterion: GW >= 0.62 AND Gsw >= 0.05 simultaneously with the
routed or ensemble configuration, in all three seeds.
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
    LEAF_ROOT, TARGETS, VAR_COLS, build_cache, folds, per_target_r2, pooled_r2,
    standardize, Mlp)
from hyperleaf_dann_sweep import DannSpectral, fit_torch2  # noqa: E402


FAMILIES = {
    "clean": lambda: Mlp(204, 3),
    "dann0.3": lambda: DannSpectral(lam=0.3),
    "dann1.0": lambda: DannSpectral(lam=1.0),
}
DANN_FAMILIES = {"dann0.3", "dann1.0"}


def fit_family(fam, spec, y, dom, tr_idx, te_idx, seed, device, epochs):
    mu = spec[tr_idx].mean(0, keepdims=True)
    sd = spec[tr_idx].std(0, keepdims=True) + 1e-6
    y_mu = y[tr_idx].mean(0, keepdims=True)
    y_sd = y[tr_idx].std(0, keepdims=True) + 1e-6
    keys = ["x"]
    tr = {"x": ((spec[tr_idx] - mu) / sd).astype(np.float32),
          "y": ((y[tr_idx] - y_mu) / y_sd).astype(np.float32),
          "dom": dom[tr_idx]}
    te = {"x": ((spec[te_idx] - mu) / sd).astype(np.float32)}
    pred = fit_torch2(FAMILIES[fam], tr, te, seed, device, epochs=epochs,
                      dann=fam in DANN_FAMILIES)
    return pred * y_sd + y_mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave7")
    ap.add_argument("--tag", default="routed")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "seeds", seeds, flush=True)

    spec, _rgb = build_cache(HERE.parent / "wave1" / "cache_leaf_spec_rgb.npz")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    y = df[list(TARGETS)].to_numpy(np.float32)
    dom = df[list(VAR_COLS)].to_numpy(np.float32).argmax(1).astype(np.int64)
    varieties = list(VAR_COLS)

    agg = {}
    for seed in seeds:
        oof = {k: np.zeros_like(y) for k in ("clean", "dann0.3", "dann1.0", "ensemble", "routed")}
        routes = {}
        for held in varieties:
            te_idx = np.where(df[held].astype(bool).to_numpy())[0]
            tr_idx = np.where(~df[held].astype(bool).to_numpy())[0]
            preds = {fam: fit_family(fam, spec, y, dom, tr_idx, te_idx, seed, device, args.epochs)
                     for fam in FAMILIES}
            # honest per-target route from inner LOVO over the training varieties
            inner_oof = {fam: {} for fam in FAMILIES}
            for inner in varieties:
                if inner == held:
                    continue
                i_te = np.where(df[inner].astype(bool).to_numpy())[0]
                i_tr = np.where((~df[inner].astype(bool).to_numpy()) & (~df[held].astype(bool).to_numpy()))[0]
                for fam in FAMILIES:
                    inner_oof[fam][inner] = fit_family(fam, spec, y, dom, i_tr, i_te, seed, device, args.epochs)
            best = {}
            for ti, t in enumerate(TARGETS):
                scores = {}
                for fam in FAMILIES:
                    yy = np.concatenate([y[np.where(df[v].astype(bool).to_numpy())[0]][:, ti] for v in inner_oof[fam]])
                    pp = np.concatenate([inner_oof[fam][v][:, ti] for v in inner_oof[fam]])
                    scores[fam] = 1.0 - float(((yy - pp) ** 2).sum()) / (float(((yy - yy.mean()) ** 2).sum()) + 1e-12)
                best[t] = max(scores, key=scores.get)
            routes[held] = best
            for fam in FAMILIES:
                oof[fam][te_idx] = preds[fam]
            oof["ensemble"][te_idx] = np.mean([preds[f] for f in FAMILIES], 0)
            routed = np.zeros_like(preds["clean"])
            for ti, t in enumerate(TARGETS):
                routed[:, ti] = preds[best[t]][:, ti]
            oof["routed"][te_idx] = routed
            print(f"  seed{seed} held {held}: route {best}", flush=True)
        for k, p in oof.items():
            agg.setdefault(k, []).append(per_target_r2(y, p))
        agg.setdefault("_routes", []).append(routes)

    summary = {"device": str(device), "seeds": seeds, "epochs": args.epochs, "candidates": {}}
    for k, lst in agg.items():
        if k == "_routes":
            continue
        summary["candidates"][k] = {
            "per_target_r2_mean": {t: round(float(np.mean([d[t] for d in lst])), 4) for t in TARGETS},
            "per_target_r2_min": {t: round(float(np.min([d[t] for d in lst])), 4) for t in TARGETS},
        }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Wave-7 routed / ensembled adversarial models ({args.tag})", "",
             f"- device: `{device}`, seeds: {seeds}, epochs: {args.epochs}",
             "- metric: pooled OOF R2 (mean / min over seeds); routing chosen by inner LOVO only", "",
             "| candidate | GW | Gsw | PhiPS2 | min GW | min Gsw |", "|---|---|---|---|---|---|"]
    for k, v in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["per_target_r2_mean"]["Gsw"]):
        pt, mn = v["per_target_r2_mean"], v["per_target_r2_min"]
        lines.append(f"| `{k}` | {pt['GrainWeight']} | {pt['Gsw']} | {pt['PhiPS2']} | {mn['GrainWeight']} | {mn['Gsw']} |")
    (out / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
