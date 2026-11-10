"""Wave-3: HistologyHSI-GB leave-one-patient-out (LOPO) screening.

EXPLORATORY LANE -- not governed benchmark evidence.

Mirrors the governed `hist.lopo-fixed.v1` protocol (13 outer patient folds,
leave one patient out) but with a lightweight local harness.  Fitted state
(normalizers, temperature, class weights) uses train-fold rows only; the
held-out patient's labels are never used for anything but scoring.

Governed reference to beat (task 15.13, pooled OOF, 3 seeds):
    local-route-a.task-only-control.v1 : Acc 0.8948 / BA 0.8714 / AUROC 0.8677
Wave-1 found variety-adversarial training was the only cross-domain signal for
HyperLeaf, so wave-3 tests the patient-adversarial analogue here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wave1"))
from hyperleaf_lovo_screen import (  # noqa: E402
    RgbNet, SpectraNet, GradReverse, _head)

HIST_ROOT = Path(r"G:\Datasets\PKG - HistologyHSI-GB")
CACHE = HIST_ROOT / "cache_routeA_v2"


def load_histology():
    files = sorted(CACHE.glob("*.npz"))
    v = np.zeros((len(files), 3, 64, 64), np.float32)
    s = np.zeros((len(files), 128), np.float32)
    y = np.zeros(len(files), np.int64)
    pat = []
    for i, f in enumerate(files):
        z = np.load(f)
        v[i], s[i] = z["v"], z["s"]
        y[i] = 1 if f.stem.endswith("_T") else 0
        pat.append(f.stem.split("_")[0])
    return v, s, y, np.array(pat)


class DannSpecCls(nn.Module):
    key = "dann-spec"
    def __init__(self, n_in=128, n_out=2, n_dom=13, hidden=256, drop=0.2, lam=0.3):
        super().__init__()
        self.lam = lam
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        self.reg = nn.Linear(hidden, n_out)
        self.dom = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_dom))
    def forward(self, x=None, spec=None, **kw):
        return self.reg(self.trunk(x if x is not None else spec))
    def domain_logits(self, x):
        return self.dom(GradReverse.apply(self.trunk(x), self.lam))


class MlpCls(nn.Module):
    key = "mlp-spec"
    def __init__(self, n_in=128, n_out=2, hidden=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, n_out))
    def forward(self, x=None, spec=None, **kw):
        return self.net(x if x is not None else spec)


class CnnRgbCls(nn.Module):
    key = "cnn-rgb"
    def __init__(self, d=128, n_out=2):
        super().__init__()
        self.rgb = RgbNet(d)
        self.head = _head(d, n_out)
    def forward(self, rgb=None, **kw):
        return self.head(self.rgb(rgb))


class ConcatCls(nn.Module):
    key = "fusion-concat"
    def __init__(self, d=128, n_out=2):
        super().__init__()
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.head = _head(2 * d, n_out)
    def forward(self, rgb=None, spec=None, **kw):
        return self.head(torch.cat([self.rgb(rgb), self.spec(spec)], 1))


class ValueOnlyCls(nn.Module):
    key = "fusion-valueonly"
    def __init__(self, d=128, n_out=2):
        super().__init__()
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.v2s, self.s2v = nn.Linear(d, d), nn.Linear(d, d)
        self.vn, self.sn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = _head(2 * d, n_out)
    def forward(self, rgb=None, spec=None, **kw):
        vt, st = self.rgb.tokens(rgb), self.spec.tokens(spec)
        vf = self.vn(vt + self.v2s(st).mean(1, keepdim=True))
        sf = self.sn(st + self.s2v(vt).mean(1, keepdim=True))
        return self.head(torch.cat([vf.mean(1), sf.mean(1)], 1))


class CrossAttnCls(nn.Module):
    key = "fusion-crossattn"
    def __init__(self, d=128, n_out=2, heads=4):
        super().__init__()
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.v2s = nn.MultiheadAttention(d, heads, batch_first=True)
        self.s2v = nn.MultiheadAttention(d, heads, batch_first=True)
        self.vn, self.sn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = _head(2 * d, n_out)
    def forward(self, rgb=None, spec=None, **kw):
        vt, st = self.rgb.tokens(rgb), self.spec.tokens(spec)
        vf = self.vn(vt + self.v2s(vt, st, st)[0])
        sf = self.sn(st + self.s2v(st, vt, vt)[0])
        return self.head(torch.cat([vf.mean(1), sf.mean(1)], 1))


def standardize(tr, te, keys):
    a, b = {}, {}
    for k in keys:
        mu = tr[k].mean(0, keepdims=True)
        sd = tr[k].std(0, keepdims=True) + 1e-6
        a[k] = ((tr[k] - mu) / sd).astype(np.float32)
        b[k] = ((te[k] - mu) / sd).astype(np.float32)
    return a, b


def fit_cls(build, tr, te, seed, device, epochs=120, lr=1e-3, wd=1e-4, dann=False, cls_w=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    keys = [k for k in ("spec", "rgb", "x") if k in tr]
    Xtr = {k: torch.as_tensor(np.ascontiguousarray(tr[k]), dtype=torch.float32, device=device) for k in keys}
    Xte = {k: torch.as_tensor(np.ascontiguousarray(te[k]), dtype=torch.float32, device=device) for k in keys}
    ytr = torch.as_tensor(tr["y"], dtype=torch.long, device=device)
    dtr = torch.as_tensor(tr["dom"], dtype=torch.long, device=device) if dann else None
    w = None if cls_w is None else torch.as_tensor(cls_w, dtype=torch.float32, device=device)
    n = ytr.shape[0]
    bs = min(64, n)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = {k: v[idx] for k, v in Xtr.items()}
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(**b), ytr[idx], weight=w)
            if dann:
                src = b.get("x", b.get("spec"))
                loss = loss + nn.functional.cross_entropy(model.domain_logits(src), dtr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, Xte[keys[0]].shape[0], 512):
            logits.append(model(**{k: v[i:i + 512] for k, v in Xte.items()}).cpu())
    return torch.cat(logits, 0).numpy()


def _ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += float(m.sum()) / len(y) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return e


def _fit_temperature(logits, y, iters=200):
    t = torch.ones(1, requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=iters)
    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / t.clamp(min=1e-2), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.clamp(min=1e-2).item())


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _auc(scores, y):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _ba(pred, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    pos, neg = y == 1, y == 0
    return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))


def candidates_cls():
    return [
        ("mlp-spec", {"keys": ["spec"], "build": lambda: MlpCls()}),
        ("dann-spec", {"keys": ["x"], "build": lambda: DannSpecCls(), "dann": True}),
        ("cnn-rgb", {"keys": ["rgb"], "build": lambda: CnnRgbCls()}),
        ("fusion-concat", {"keys": ["spec", "rgb"], "build": lambda: ConcatCls()}),
        ("fusion-valueonly", {"keys": ["spec", "rgb"], "build": lambda: ValueOnlyCls()}),
        ("fusion-crossattn", {"keys": ["spec", "rgb"], "build": lambda: CrossAttnCls()}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave3")
    ap.add_argument("--tag", default="lopo")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "seeds", seeds, flush=True)

    v, spec, y, pat = load_histology()
    patients = sorted(set(pat.tolist()))
    print("rois", len(y), "patients", len(patients), "pos", int(y.sum()), flush=True)

    agg = {}
    dump = {}
    for seed in seeds:
        oof_logits = {}
        fold_rows = {k: [] for k, _ in candidates_cls()}
        for held in patients:
            te_idx = np.where(pat == held)[0]
            tr_idx = np.where(pat != held)[0]
            mu, sd = spec[tr_idx].mean(0, keepdims=True), spec[tr_idx].std(0, keepdims=True) + 1e-6
            tr_raw = {"spec": ((spec[tr_idx] - mu) / sd).astype(np.float32), "rgb": v[tr_idx],
                      "x": ((spec[tr_idx] - mu) / sd).astype(np.float32), "y": y[tr_idx],
                      "dom": np.searchsorted(patients, pat[tr_idx]).astype(np.int64)}
            te_raw = {"spec": ((spec[te_idx] - mu) / sd).astype(np.float32), "rgb": v[te_idx],
                      "x": ((spec[te_idx] - mu) / sd).astype(np.float32)}
            cnt = np.bincount(y[tr_idx], minlength=2).astype(np.float32)
            cls_w = (cnt.sum() / (2.0 * np.maximum(cnt, 1.0))).tolist()
            for key, cfg in candidates_cls():
                A, B = standardize(tr_raw, te_raw, cfg["keys"])
                A["y"] = y[tr_idx]
                if cfg.get("dann"):
                    A["dom"] = tr_raw["dom"]
                lg = fit_cls(cfg["build"], A, B, seed, device, epochs=args.epochs,
                             dann=bool(cfg.get("dann")), cls_w=cls_w)
                oof_logits.setdefault(key, np.zeros((len(y), 2), np.float32))
                oof_logits[key][te_idx] = lg
            print(f"  seed{seed} held-out {held}: done", flush=True)
        for key, lg in oof_logits.items():
            dump[f"{key}__seed{seed}"] = lg
            p = _softmax(lg)
            pred = p.argmax(1)
            agg.setdefault(key, {"acc": [], "ba": [], "auroc": [], "ece": []})
            agg[key]["acc"].append(float((pred == y).mean()))
            agg[key]["ba"].append(_ba(pred, y))
            agg[key]["auroc"].append(_auc(p[:, 1], y))
            agg[key]["ece"].append(_ece(p, y))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{args.tag}_oof_logits.npz", y=y, pat=pat.astype("U8"), **dump)
    summary = {"device": str(device), "seeds": seeds, "epochs": args.epochs, "tag": args.tag,
               "n_rois": int(len(y)), "n_patients": len(patients), "pos": int(y.sum()), "candidates": {}}
    for k, d in agg.items():
        summary["candidates"][k] = {m: round(float(np.nanmean(d[m])), 4) for m in ("acc", "ba", "auroc", "ece")}
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Wave-3 Histology LOPO ({args.tag})", "",
             f"- device: `{device}`, seeds: {seeds}, epochs: {args.epochs}, n={len(y)} ROIs / {len(patients)} patients",
             "- metric: pooled OOF; BA/AUROC use sklearn; ECE 15 bins", "",
             "| candidate | Acc | BA | AUROC | ECE |", "|---|---|---|---|---|"]
    for k, d in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["ba"]):
        lines.append(f"| `{k}` | {d['acc']} | {d['ba']} | {d['auroc']} | {d['ece']} |")
    lines += ["", "Governed reference (task 15.13, pooled OOF): "
              "`local-route-a.task-only-control.v1` Acc 0.8948 / BA 0.8714 / AUROC 0.8677"]
    (out_dir / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
