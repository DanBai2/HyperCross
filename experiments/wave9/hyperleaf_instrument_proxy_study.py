"""Wave-9: is the governed inner-variety instrument a valid proxy for LOVO?

EXPLORATORY LANE -- not governed benchmark evidence.

Motivation (task-24.4 audit, 2026-11-06):
  * task 24.4 already executed; it fell back to the frozen 24.2 candidate in ALL
    four outer folds ("all_contexts_fell_back": true, no promotion);
  * every candidate's inner-variety-OOF Gsw landed in [-0.081, -0.079] (spread
    0.0014) with the best epoch at 4 of 20 -- flat, noise-dominated, and worse
    with more training;
  * on that instrument the eight candidates were statistically indistinguishable,
    so promotion could not happen regardless of mechanism quality.

Wave-9 tests the mechanical explanation.  Governed inner folds train on TWO
varieties and validate on a third, whereas both the outer LOVO evaluation and the
reported pooled-OOF metric train on THREE varieties.  The selection instrument is
therefore a strictly harder estimand than the thing it selects for, which pushes
selection towards the least-trained (most conservative) configuration.

Designs compared on identical samples, families, and seeds:
  A. outer-lovo       train 3 varieties -> test 1 fully held-out variety
                      (the estimand of the reported claim)
  B. inner-lovo       train 2 varieties -> test the third (governed 24.4 instrument)
  C. inner-stratified train 3 varieties (2/3 of each) -> test the held-out thirds
                      (candidate replacement instrument, difficulty-matched to A)

Pre-registered predictions:
  P1  B's per-family Gsw R2 is far below A's.
  P2  the DANN-minus-clean Gsw delta is ~0 under B and positive under A.
  P3  C tracks A better than B does, both in absolute level and in family delta.
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
    pooled_r2,
    standardize,
    Mlp,
    ridge_fit_predict,
)
from hyperleaf_dann_sweep import DannSpectral, fit_torch2  # noqa: E402

FAMILIES = ("ridge-spec", "clean-mlp", "dann1.0")
DESIGNS = ("outer-lovo", "inner-lovo", "inner-stratified")


# ----------------------------------------------------------------- design ---
def make_fits(design: str, df: pd.DataFrame, parts: int) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Return (fit_id, outer_label, train_idx, test_idx) for one design."""
    masks = {v: df[v].astype(bool).to_numpy() for v in VAR_COLS}
    out: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    if design == "outer-lovo":
        for v in VAR_COLS:
            tr = np.where(~masks[v])[0]
            te = np.where(masks[v])[0]
            out.append((f"hold={v}", v, tr, te))
        return out

    if design == "inner-lovo":
        for outer in VAR_COLS:
            pool = np.where(~masks[outer])[0]
            for inner in VAR_COLS:
                if inner == outer:
                    continue
                tr = np.asarray([i for i in pool if not masks[inner][i]], dtype=np.int64)
                te = np.where(masks[inner])[0]
                out.append((f"{outer}/hold={inner}", outer, tr, te))
        return out

    if design == "inner-stratified":
        for outer in VAR_COLS:
            pool = np.where(~masks[outer])[0]
            buckets: dict[int, list[np.ndarray]] = {k: [] for k in range(parts)}
            for w in VAR_COLS:
                if w == outer:
                    continue
                idx_w = np.where(masks[w])[0]
                rng = np.random.default_rng(1000 + VAR_COLS.index(w))
                perm = rng.permutation(len(idx_w))
                for k in range(parts):
                    buckets[k].append(idx_w[perm[k::parts]])
            for k in range(parts):
                te = np.concatenate(buckets[k])
                tr = np.asarray([i for i in pool if i not in set(te.tolist())], dtype=np.int64)
                out.append((f"{outer}/part{k}", outer, tr, te))
        return out

    raise ValueError(f"unknown design: {design!r}")


# ----------------------------------------------------------------- family ---
def fit_family(family: str, tr_raw, te_raw, seed, device, epochs, y_mu, y_sd) -> np.ndarray:
    y_tr = ((tr_raw["y"] - y_mu) / y_sd).astype(np.float32)
    if family == "ridge-spec":
        a, b = standardize({"spec": tr_raw["spec"]}, {"spec": te_raw["spec"]}, ["spec"])
        pred, _alpha = ridge_fit_predict(a["spec"], y_tr, b["spec"], seed=seed)
    elif family == "clean-mlp":
        a, b = standardize({"spec": tr_raw["spec"]}, {"spec": te_raw["spec"]}, ["spec"])
        pred = fit_torch2(
            lambda: Mlp(204, 3), {"spec": a["spec"], "y": y_tr}, {"spec": b["spec"]},
            seed, device, epochs=epochs,
        )
    elif family == "dann1.0":
        a, b = standardize({"x": tr_raw["x"]}, {"x": te_raw["x"]}, ["x"])
        pred = fit_torch2(
            lambda: DannSpectral(lam=1.0), {"x": a["x"], "y": y_tr, "dom": tr_raw["dom"]},
            {"x": b["x"]}, seed, device, epochs=epochs, dann=True,
        )
    else:
        raise ValueError(f"unknown family: {family!r}")
    return pred * y_sd + y_mu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave9")
    ap.add_argument("--tag", default="instrument-proxy")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    designs = [x for x in args.designs.split(",") if x]
    families = [x for x in args.families.split(",") if x]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} seeds {seeds} epochs {args.epochs}", flush=True)

    spec, _rgb = build_cache(WORK_ROOT / "cache_leaf_spec_rgb.npz")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    y = df[list(TARGETS)].to_numpy(np.float32)
    dom = df[list(VAR_COLS)].to_numpy(np.float32).argmax(1).astype(np.int64)
    n = len(y)
    print(f"n={n} spec={spec.shape}", flush=True)

    store: dict[str, dict[str, dict[str, object]]] = {}
    for design in designs:
        fits = make_fits(design, df, args.parts)
        print(f"[{design}] {len(fits)} fits", flush=True)
        store[design] = {}
        for family in families:
            per_seed_pooled: list[dict[str, float]] = []
            per_seed_outer: list[dict[str, dict[str, float]]] = []
            for seed in seeds:
                preds_all: list[np.ndarray] = []
                order_all: list[np.ndarray] = []
                per_outer: dict[str, list[int]] = {}
                for fit_id, outer, tr_idx, te_idx in fits:
                    y_mu = y[tr_idx].mean(0, keepdims=True)
                    y_sd = y[tr_idx].std(0, keepdims=True) + 1e-6
                    tr_raw = {"spec": spec[tr_idx], "x": spec[tr_idx], "y": y[tr_idx], "dom": dom[tr_idx]}
                    te_raw = {"spec": spec[te_idx], "x": spec[te_idx]}
                    preds_all.append(fit_family(family, tr_raw, te_raw, seed, device, args.epochs, y_mu, y_sd))
                    order_all.append(te_idx)
                    per_outer.setdefault(outer, []).extend(range(len(preds_all) - 1, len(preds_all)))
                p_all = np.concatenate(preds_all, 0)
                i_all = np.concatenate(order_all, 0)
                per_seed_pooled.append(per_target_r2(y[i_all], p_all))
                outer_scores: dict[str, dict[str, float]] = {}
                for outer, positions in per_outer.items():
                    pp = np.concatenate([preds_all[pos] for pos in positions], 0)
                    ii = np.concatenate([order_all[pos] for pos in positions], 0)
                    outer_scores[outer] = per_target_r2(y[ii], pp)
                per_seed_outer.append(outer_scores)
                print(f"  [{design}/{family}/seed{seed}] pooled Gsw={per_seed_pooled[-1]['Gsw']}", flush=True)
            mean_pooled = {
                t: round(float(np.mean([row[t] for row in per_seed_pooled])), 4) for t in TARGETS
            }
            mean_outer = {
                t: round(
                    float(np.mean([np.mean([row[o][t] for o in row]) for row in per_seed_outer])), 4
                )
                for t in TARGETS
            }
            per_outer_seed_mean = {}
            for outer in per_seed_outer[0]:
                per_outer_seed_mean[outer] = {
                    t: round(float(np.mean([row[outer][t] for row in per_seed_outer])), 4) for t in TARGETS
                }
            store[design][family] = {
                "pooled_r2_mean_over_seeds": mean_pooled,
                "per_outer_mean_r2_mean_over_seeds": mean_outer,
                "per_outer_seed_mean": per_outer_seed_mean,
                "per_seed_pooled": per_seed_pooled,
            }

    summary: dict[str, object] = {
        "device": str(device), "seeds": seeds, "epochs": args.epochs,
        "designs": designs, "families": families, "parts": args.parts,
        "n_samples": int(n), "tag": args.tag, "evidence_grade": "exploratory",
        "results": store,
    }
    gsw_table = {d: {f: store[d][f]["pooled_r2_mean_over_seeds"]["Gsw"] for f in families} for d in designs}
    summary["gsw_pooled_table"] = gsw_table
    dels = {}
    for d in designs:
        if "clean-mlp" in families and "dann1.0" in families:
            dels[d] = {
                "dann1.0_minus_clean-mlp": {
                    t: round(
                        store[d]["dann1.0"]["pooled_r2_mean_over_seeds"][t]
                        - store[d]["clean-mlp"]["pooled_r2_mean_over_seeds"][t], 4
                    ) for t in TARGETS
                }
            }
    summary["deltas"] = dels

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Wave-9 instrument-validity study ({args.tag})",
        "",
        f"- device `{device}`, seeds {seeds}, epochs {args.epochs}, n={n}",
        "- metric: pooled OOF R2 in original target units (mean over seeds)",
        "- A `outer-lovo` = reported estimand (3 train varieties); B `inner-lovo` = governed task-24.4 instrument (2 train varieties); C `inner-stratified` = difficulty-matched replacement",
        "",
        "## Pooled R2 by design",
        "",
        "| family | A outer-lovo | B inner-lovo | C inner-stratified |",
        "|---|---|---|---|",
    ]
    for f in families:
        cells = []
        for d in DESIGNS:
            if d in gsw_table and f in gsw_table[d]:
                cells.append(str(gsw_table[d][f]))
            else:
                cells.append("n/a")
        lines.append(f"| `{f}` Gsw | {cells[0]} | {cells[1]} | {cells[2]} |")
    for f in families:
        cells = []
        for d in DESIGNS:
            if d in store and f in store[d]:
                cells.append(str(store[d][f]["pooled_r2_mean_over_seeds"]["GrainWeight"]))
            else:
                cells.append("n/a")
        lines.append(f"| `{f}` GrainWeight | {cells[0]} | {cells[1]} | {cells[2]} |")
    for f in families:
        cells = []
        for d in DESIGNS:
            if d in store and f in store[d]:
                cells.append(str(store[d][f]["per_outer_mean_r2_mean_over_seeds"]["Gsw"]))
            else:
                cells.append("n/a")
        lines.append(f"| `{f}` Gsw (per-outer mean) | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += ["", "## DANN-minus-clean delta on Gsw", "", "| design | GrainWeight | Gsw | PhiPS2 |", "|---|---|---|---|"]
    for d, payload in dels.items():
        row = payload["dann1.0_minus_clean-mlp"]
        lines.append(f"| {d} | {row['GrainWeight']} | {row['Gsw']} | {row['PhiPS2']} |")

    lines += ["", "## Per-held-out-variety Gsw (design A, grading the outer estimand)", "",
              "| family | " + " | ".join(VAR_COLS) + " |", "|---|" + "---|" * len(VAR_COLS)]
    if "outer-lovo" in store:
        for f in families:
            row = store["outer-lovo"][f]["per_outer_seed_mean"]
            lines.append(f"| `{f}` | " + " | ".join(str(row.get(v, {}).get("Gsw", "n/a")) for v in VAR_COLS) + " |")

    (out_dir / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
