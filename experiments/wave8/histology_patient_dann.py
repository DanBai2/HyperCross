"""Wave-8: patient-adversarial spectral representation for Histology LOPO.

EXPLORATORY LANE -- not governed benchmark evidence.

Parallel to wave-6/7 on HyperLeaf.  Wave-6 established that variety-adversarial
training of the SPECTRAL branch is the only reproducible cross-domain gain, and
that applying the adversarial objective to a fused representation destroys the
model.  Wave-8 tests the same two hypotheses on the patient domain:

  H1  patient-adversarial training of the spectral branch improves cross-patient
      balanced accuracy relative to the matched clean model;
  H2  applying it to the fused RGB+spectral representation is harmful.

Reference: governed task 15.13 pooled OOF BA 0.8714 / AUROC 0.8677
(seed-consensus OOF of `histology.local-route-a.task-only-control.v1`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wave3"))
sys.path.insert(0, str(HERE.parent / "wave1"))
from histology_lopo_screen import (  # noqa: E402
    load_histology, standardize, _ece, _softmax, _auc, _ba, MlpCls, CnnRgbCls, ConcatCls)
from hyperleaf_lovo_screen import RgbNet, SpectraNet, _head, GradReverse  # noqa: E402


class DannSpecCls(nn.Module):
    def __init__(self, n_in=128, n_out=2, n_dom=13, hidden=256, drop=0.2, lam=0.3):
        super().__init__()
        self.lam = lam
        self.trunk = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        self.reg = nn.Linear(hidden, n_out)
        self.dom = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_dom))
    def forward(self, x=None, spec=None, **kw):
        return self.reg(self.trunk(x if x is not None else spec))
    def domain_logits(self, x=None, spec=None, **kw):
        return self.dom(GradReverse.apply(self.trunk(x if x is not None else spec), self.lam))


class FusionDannSpec(nn.Module):
    """Fusion model whose SPECTRAL embedding is made patient-invariant (H1/H2)."""
    def __init__(self, d=128, n_out=2, n_dom=13, lam=0.3):
        super().__init__()
        self.lam = lam
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.head = _head(2 * d, n_out)
        self.dom = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, n_dom))
    def forward(self, rgb=None, spec=None, **kw):
        return self.head(torch.cat([self.rgb(rgb), self.spec(spec)], 1))
    def domain_logits(self, rgb=None, spec=None, **kw):
        return self.dom(GradReverse.apply(self.spec(spec), self.lam))


class FusionDannAll(nn.Module):
    """Control for H2: adversarial objective on the whole fused representation."""
    def __init__(self, d=128, n_out=2, n_dom=13, lam=0.3):
        super().__init__()
        self.lam = lam
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.head = _head(2 * d, n_out)
        self.dom = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_dom))
    def fused(self, rgb, spec):
        return torch.cat([self.rgb(rgb), self.spec(spec)], 1)
    def forward(self, rgb=None, spec=None, **kw):
        return self.head(self.fused(rgb, spec))
    def domain_logits(self, rgb=None, spec=None, **kw):
        return self.dom(GradReverse.apply(self.fused(rgb, spec), self.lam))


def fit_cls2(build, tr, te, seed, device, epochs, dann, cls_w):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    keys = [k for k in ("spec", "rgb", "x") if k in tr]
    Xtr = {k: torch.as_tensor(np.ascontiguousarray(tr[k]), dtype=torch.float32, device=device) for k in keys}
    Xte = {k: torch.as_tensor(np.ascontiguousarray(te[k]), dtype=torch.float32, device=device) for k in keys}
    ytr = torch.as_tensor(tr["y"], dtype=torch.long, device=device)
    dtr = torch.as_tensor(tr["dom"], dtype=torch.long, device=device)
    w = torch.as_tensor(cls_w, dtype=torch.float32, device=device)
    n = ytr.shape[0]; bs = min(64, n)
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
                loss = loss + nn.functional.cross_entropy(model.domain_logits(**b), dtr[idx])
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, Xte[keys[0]].shape[0], 512):
            chunks.append(model(**{k: v[i:i + 512] for k, v in Xte.items()}).cpu().numpy())
    return np.concatenate(chunks, 0)


def families():
    return [
        ("mlp-spec", {"keys": ["spec"], "build": lambda: MlpCls(), "dann": False}),
        ("dann-spec-0.3", {"keys": ["x"], "build": lambda: DannSpecCls(lam=0.3), "dann": True}),
        ("dann-spec-1.0", {"keys": ["x"], "build": lambda: DannSpecCls(lam=1.0), "dann": True}),
        ("fusion-concat", {"keys": ["spec", "rgb"], "build": lambda: ConcatCls(), "dann": False}),
        ("fusion-dann-spec-1.0", {"keys": ["spec", "rgb"], "build": lambda: FusionDannSpec(lam=1.0), "dann": True}),
        ("fusion-dann-all-1.0", {"keys": ["spec", "rgb"], "build": lambda: FusionDannAll(lam=1.0), "dann": True}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave8")
    ap.add_argument("--tag", default="patient-dann")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "seeds", seeds, flush=True)
    v, spec, y, pat = load_histology()
    patients = sorted(set(pat.tolist()))
    print("rois", len(y), "patients", len(patients), "pos", int(y.sum()), flush=True)

    agg = {}
    for seed in seeds:
        oof = {}
        for held in patients:
            te_idx = np.where(pat == held)[0]
            tr_idx = np.where(pat != held)[0]
            mu, sd = spec[tr_idx].mean(0, keepdims=True), spec[tr_idx].std(0, keepdims=True) + 1e-6
            s_tr = ((spec[tr_idx] - mu) / sd).astype(np.float32)
            s_te = ((spec[te_idx] - mu) / sd).astype(np.float32)
            tr_raw = {"spec": s_tr, "rgb": v[tr_idx], "x": s_tr}
            te_raw = {"spec": s_te, "rgb": v[te_idx], "x": s_te}
            cnt = np.bincount(y[tr_idx], minlength=2).astype(np.float32)
            cls_w = (cnt.sum() / (2.0 * np.maximum(cnt, 1.0))).tolist()
            for key, cfg in families():
                A, B = standardize(tr_raw, te_raw, cfg["keys"])
                A["y"] = y[tr_idx]
                A["dom"] = np.searchsorted(patients, pat[tr_idx]).astype(np.int64)
                lg = fit_cls2(cfg["build"], A, B, seed, device, args.epochs, cfg["dann"], cls_w)
                oof.setdefault(key, np.zeros((len(y), 2), np.float32))
                oof[key][te_idx] = lg
            print(f"  seed{seed} held {held}: done", flush=True)
        for key, lg in oof.items():
            p = _softmax(lg); pred = p.argmax(1)
            agg.setdefault(key, {"acc": [], "ba": [], "auroc": [], "ece": []})
            agg[key]["acc"].append(float((pred == y).mean()))
            agg[key]["ba"].append(_ba(pred, y))
            agg[key]["auroc"].append(_auc(p[:, 1], y))
            agg[key]["ece"].append(_ece(p, y))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(device), "seeds": seeds, "epochs": args.epochs, "tag": args.tag,
               "reference": {"ba": 0.8714, "auroc": 0.8677}, "candidates": {}}
    for k, d in agg.items():
        summary["candidates"][k] = {m: round(float(np.nanmean(d[m])), 4) for m in ("acc", "ba", "auroc", "ece")}
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Wave-8 patient-adversarial Histology LOPO ({args.tag})", "",
             f"- device: `{device}`, seeds: {seeds}, epochs: {args.epochs}, n={len(y)} ROIs / {len(patients)} patients",
             "- reference (governed task 15.13): BA 0.8714 / AUROC 0.8677", "",
             "| candidate | Acc | BA | AUROC | ECE |", "|---|---|---|---|---|"]
    for k, d in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["ba"]):
        lines.append(f"| `{k}` | {d['acc']} | {d['ba']} | {d['auroc']} | {d['ece']} |")
    (out / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
