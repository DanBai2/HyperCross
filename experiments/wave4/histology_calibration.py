"""Wave-4: leave-one-patient-out calibration and decision-rule study.

EXPLORATORY LANE -- analysis of wave-3 OOF logits, no retraining.

Wave-3 found a consistent pattern on HistologyHSI-GB: fusion models rank ROIs
well (AUROC up to 0.9185) but threshold badly (BA ~0.82) and are poorly
calibrated (ECE 0.10-0.20).  This wave asks whether an honest, label-free-at-test
post-hoc temperature + threshold rule recovers the gap.

Honesty: for each held-out patient, both the temperature and the decision
threshold are fitted ONLY on the other patients' out-of-fold scores.  Nothing
about the held-out patient is used except final scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TARGET_BA = 0.8714  # governed task 15.13 pooled OOF reference


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += float(m.sum()) / len(y) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return e


def fit_temperature(logits, y):
    z = torch.as_tensor(logits, dtype=torch.float32)
    yy = torch.as_tensor(y, dtype=torch.long)
    t = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / t.clamp(min=1e-2), yy)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.clamp(min=1e-2).item())


def fit_threshold(scores, y, grid=201):
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, grid):
        pred = (scores >= t).astype(np.int64)
        pos, neg = y == 1, y == 0
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        ba = 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t


def ba_at(scores, y, t):
    pred = (scores >= t).astype(np.int64)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))


def ba_pred(pred, y):
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))


def acc_at(scores, y, t):
    return float(((scores >= t).astype(np.int64) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=r"G:\Datasets\HyperCross\runs_v3\wave3\lopo-3seed_oof_logits.npz")
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave4")
    ap.add_argument("--tag", default="calib")
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = z["y"].astype(np.int64)
    pat = z["pat"].astype(str)
    patients = sorted(set(pat.tolist()))
    keys = [k for k in z.files if k not in ("y", "pat")]
    print("candidates:", keys, "n:", len(y), "patients:", len(patients), flush=True)

    rows = {}
    for k in keys:
        lg = z[k].astype(np.float64)
        cand, seed = k.rsplit("__seed", 1)
        raw_s = np.zeros(len(y))
        tmp_s = np.zeros(len(y))
        pred_raw = np.zeros(len(y), np.int64)
        pred_tmp = np.zeros(len(y), np.int64)
        for p in patients:
            te = pat == p
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                T = 1.0
                tau_raw = tau_tmp = 0.5
            else:
                T = fit_temperature(lg[tr], y[tr])
                pr_tr = softmax(lg[tr] / T)[:, 1]
                tau_tmp = fit_threshold(pr_tr, y[tr])
                tau_raw = fit_threshold(softmax(lg[tr])[:, 1], y[tr])
            raw_s[te] = softmax(lg[te])[:, 1]
            tmp_s[te] = softmax(lg[te] / T)[:, 1]
            pred_raw[te] = (raw_s[te] >= tau_raw).astype(np.int64)
            pred_tmp[te] = (tmp_s[te] >= tau_tmp).astype(np.int64)
        # pooled metrics with per-fold-fitted thresholds are approximated by the
        # global rule below; report the robust global variants too.
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        pr_raw = np.stack([1 - raw_s, raw_s], 1)
        pr_tmp = np.stack([1 - tmp_s, tmp_s], 1)
        r = {
            "auroc": round(float(roc_auc_score(y, raw_s)) if len(np.unique(y)) > 1 else float("nan"), 4),
            "acc_raw50": round(acc_at(raw_s, y, 0.5), 4),
            "ba_raw50": round(balanced_accuracy_score(y, (raw_s >= 0.5).astype(int)), 4),
            "acc_tmp50": round(acc_at(tmp_s, y, 0.5), 4),
            "ba_tmp50": round(balanced_accuracy_score(y, (tmp_s >= 0.5).astype(int)), 4),
            "ba_rawTau": round(ba_pred(pred_raw, y), 4),
            "ba_tmpTau": round(ba_pred(pred_tmp, y), 4),
            "ece_raw": round(ece(pr_raw, y), 4),
            "ece_tmp": round(ece(pr_tmp, y), 4),
        }
        rows.setdefault(cand, []).append(r)

    summary = {"reference_ba": TARGET_BA, "n_rois": int(len(y)), "candidates": {}}
    for cand, rs in rows.items():
        summary["candidates"][cand] = {m: round(float(np.nanmean([r[m] for r in rs])), 4) for m in rs[0]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Wave-4 calibration study ({args.tag})", "",
             f"- input: `{Path(args.oof).name}`, n={len(y)} ROIs, {len(patients)} patients",
             "- temperature fitted leave-one-patient-out; metrics are 3-seed means", "",
             "| candidate | AUROC | BA@0.5 | BA@fittedTau | BA@0.5+temp | BA@fittedTau+temp | ECE raw | ECE temp |",
             "|---|---|---|---|---|---|---|---|"]
    for cand, d in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["ba_tmp50"]):
        lines.append(f"| `{cand}` | {d['auroc']} | {d['ba_raw50']} | {d['ba_rawTau']} | {d['ba_tmp50']} | "
                     f"{d['ba_tmpTau']} | {d['ece_raw']} | {d['ece_tmp']} |")
    lines += ["", f"Governed reference pooled OOF BA = {TARGET_BA} (task 15.13)"]
    (out / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
