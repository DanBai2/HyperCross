"""Wave-5: calibration and decision layer on the GOVERNED Histology LOPO baseline.

EXPLORATORY LANE -- analysis of already-published governed OOF artifacts.

Wave-3/4 showed the lightweight local harness is weaker than the governed
pipeline (best pooled BA 0.8433 vs 0.8714), so new mechanisms must be tested on
top of the governed baseline, not beside it.

This wave asks a single question: can an honest, leave-one-patient-out Platt
scaling + decision-threshold layer improve the governed Histology baseline's
decision quality and calibration *without retraining anything*?

Honesty: for each held-out patient, the affine logit transform and the decision
threshold are fitted only on the other 12 patients' out-of-fold scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(r"G:\Datasets\HyperCross\runs_v2\strict\histology\executions"
            r"\task-15-13-20261026T134933767825Z\lopo\aggregates")
CANDIDATES = {
    "task-only-control": "5c4f34f232a00982",
    "alignment-nce": "548ee96053241ce6",
    "alignment-only-control": "a77598ef0f67263a",
    "nce-only-control": "7b601033d09f99b7",
    "attention-disabled": "2705358c514eaf80",
    "mil-concat": "9d2347fe8233a5f3",
    "pca-only": "e55861aa4f9abf01",
    "rgb-only": "6625d5651d7b8c25",
}


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def ece(p, y, bins=15):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(np.int64)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += float(m.sum()) / len(y) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return e


def ba(p, y, t=0.5):
    pred = (p >= t).astype(np.int64)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))


def fit_threshold(p, y, grid=201):
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, grid):
        v = ba(p, y, t)
        if not np.isnan(v) and v > best:
            best, best_t = v, float(t)
    return best_t


def fit_platt(z, y):
    """Affine logit transform fitted by logistic regression on 1-D logits."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return a, b


def load_candidate(hash_id: str):
    p = BASE / hash_id / "seed-consensus-oof.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    recs = d["records_extended"]
    prob = np.array([r["probability"] for r in recs], np.float64)
    y = np.array([int(r["y_true"]) for r in recs], np.int64)
    g = np.array([str(r["group_id"]) for r in recs])
    return prob, y, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave5")
    ap.add_argument("--tag", default="governed-calib")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"source": str(BASE), "candidates": {}}
    for name, h in CANDIDATES.items():
        loaded = load_candidate(h)
        if loaded is None:
            print("missing", name, h, flush=True)
            continue
        prob, y, g = loaded
        patients = sorted(set(g.tolist()))
        from sklearn.metrics import roc_auc_score
        raw = np.zeros_like(prob)
        platt = np.zeros_like(prob)
        pred_platt = np.zeros(len(y), np.int64)
        pred_raw_tau = np.zeros(len(y), np.int64)
        for p in patients:
            te = g == p
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                a, b, tau = 1.0, 0.0, 0.5
            else:
                a, b = fit_platt(logit(prob[tr]), y[tr])
                tau = fit_threshold(sigmoid(a * logit(prob[tr]) + b), y[tr])
            raw[te] = prob[te]
            platt[te] = sigmoid(a * logit(prob[te]) + b)
            pred_platt[te] = (platt[te] >= tau).astype(np.int64)
            pred_raw_tau[te] = (prob[te] >= tau).astype(np.int64)

        def bap(pred):
            pos, neg = y == 1, y == 0
            return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))

        summary["candidates"][name] = {
            "auroc": round(float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"), 4),
            "acc_raw50": round(float(((prob >= 0.5).astype(int) == y).mean()), 4),
            "ba_raw50": round(ba(prob, y, 0.5), 4),
            "ba_raw_tau": round(bap(pred_raw_tau), 4),
            "acc_platt": round(float(((platt >= 0.5).astype(int) == y).mean()), 4),
            "ba_platt50": round(ba(platt, y, 0.5), 4),
            "ba_platt_tau": round(bap(pred_platt), 4),
            "ece_raw": round(ece(prob, y), 4),
            "ece_platt": round(ece(platt, y), 4),
        }
        print(f"{name}: {summary['candidates'][name]}", flush=True)

    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Wave-5 governed-baseline calibration ({args.tag})", "",
             "- source: task-15.13 seed-consensus OOF artifacts (n=469 ROIs, 13 patients)",
             "- Platt + threshold fitted leave-one-patient-out; no retraining", "",
             "| candidate | AUROC | Acc@0.5 | BA@0.5 | BA@tau(raw) | BA@0.5(platt) | BA@tau(platt) | ECE raw | ECE platt |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, d in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["ba_platt_tau"]):
        lines.append(f"| `{name}` | {d['auroc']} | {d['acc_raw50']} | {d['ba_raw50']} | {d['ba_raw_tau']} | "
                     f"{d['ba_platt50']} | {d['ba_platt_tau']} | {d['ece_raw']} | {d['ece_platt']} |")
    (out / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
