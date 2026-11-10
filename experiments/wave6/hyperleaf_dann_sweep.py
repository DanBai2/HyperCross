"""Wave-6: adversarial-strength sweep for the only positive wave-1 lead.

EXPLORATORY LANE -- not governed benchmark evidence.

Wave-1 found variety-adversarial spectral training ("dann-spec") was the only
candidate with a consistent cross-variety gain on HyperLeaf LOVO
(Gsw -0.0200 vs -0.1191 for the matched non-adversarial MLP, same sign in all
three seeds).  Wave-6 asks whether that lead is real by sweeping the gradient
reversal strength and by testing the adversarial objective on the fused model.

Pre-registered criterion: Gsw pooled OOF R2 >= 0.10 with GW >= 0.60 would make
the mechanism worth promoting to the governed runner.
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
from hyperleaf_lovo_screen import (  # noqa: E402
    LEAF_ROOT, TARGETS, VAR_COLS, build_cache, folds, physics_features,
    per_target_r2, pooled_r2, fit_torch, standardize, DannSpec, Mlp,
    ConcatFusion, ValueOnlyFusion, CrossAttnFusion, RgbNet, SpectraNet, _head,
    GradReverse)
import torch.nn as nn  # noqa: E402


class DannSpectral(nn.Module):
    def __init__(self, n_in=204, n_out=3, n_var=4, hidden=256, drop=0.2, lam=0.3):
        super().__init__()
        self.lam = lam
        self.trunk = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        self.reg = nn.Linear(hidden, n_out)
        self.dom = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_var))
    def forward(self, x=None, **kw):
        return self.reg(self.trunk(x))
    def domain_logits(self, x=None, **kw):
        return self.dom(GradReverse.apply(self.trunk(x), self.lam))


class DannFusion(nn.Module):
    """Adversarial objective on the fused spectral+RGB representation."""
    def __init__(self, d=128, n_out=3, n_var=4, drop=0.2, lam=0.3):
        super().__init__()
        self.lam = lam
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.head = _head(2 * d, n_out)
        self.dom = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_var))
    def forward(self, rgb=None, spec=None, **kw):
        return self.head(self.fused(rgb, spec))
    def fused(self, rgb, spec):
        return torch.cat([self.rgb(rgb), self.spec(spec)], 1)
    def domain_logits(self, rgb=None, spec=None, **kw):
        return self.dom(GradReverse.apply(self.fused(rgb, spec), self.lam))


class DannFusionSpec(DannFusion):
    """Adversarial on the spectral embedding only, RGB kept clean."""
    def domain_logits(self, rgb=None, spec=None, **kw):
        return self.dom(GradReverse.apply(self.spec(spec), self.lam))
    def __init__(self, d=128, n_out=3, n_var=4, drop=0.2, lam=0.3):
        super().__init__(d=d, n_out=n_out, n_var=n_var, drop=drop, lam=lam)
        self.dom = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, n_var))


def candidates(lam_list):
    out = [("mlp-spec", {"keys": ["spec"], "build": lambda: Mlp(204, 3)})]
    for lam in lam_list:
        out.append((f"dann-spec-lam{lam}", {"keys": ["x"], "build": lambda l=lam: DannSpectral(lam=l), "dann": True}))
    for lam in lam_list:
        out.append((f"dann-fusion-lam{lam}", {"keys": ["x", "spec", "rgb"], "build": lambda l=lam: DannFusion(lam=l), "dann": True}))
    out.append(("dann-fusion-spec-lam0.3", {"keys": ["x", "spec", "rgb"], "build": lambda: DannFusionSpec(lam=0.3), "dann": True}))
    out.append(("fusion-concat", {"keys": ["spec", "rgb", "phys"], "build": lambda: ConcatFusion(840)}))
    return out



def fit_torch2(build, tr, te, seed, device, epochs=200, lr=1e-3, wd=1e-4, dann=False):
    """Batch-aware trainer: adversarial head receives the whole input dict."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    keys = [k for k in ("spec", "phys", "rgb", "x") if k in tr]
    Xtr = {k: torch.as_tensor(np.ascontiguousarray(tr[k]), dtype=torch.float32, device=device) for k in keys}
    Xte = {k: torch.as_tensor(np.ascontiguousarray(te[k]), dtype=torch.float32, device=device) for k in keys}
    ytr = torch.as_tensor(tr["y"], dtype=torch.float32, device=device)
    dtr = torch.as_tensor(tr["dom"], dtype=torch.long, device=device) if dann else None
    n = ytr.shape[0]
    bs = min(128, n)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = {k: v[idx] for k, v in Xtr.items()}
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(**b), ytr[idx])
            if dann:
                loss = loss + nn.functional.cross_entropy(model.domain_logits(**b), dtr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, Xte[keys[0]].shape[0], 512):
            chunks.append(model(**{k: v[i:i + 512] for k, v in Xte.items()}).cpu().numpy())
    return np.concatenate(chunks, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lams", default="0.1,0.3,1.0")
    ap.add_argument("--out", default=r"G:\Datasets\HyperCross\runs_v3\wave6")
    ap.add_argument("--tag", default="dann-sweep")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "seeds", seeds, "lams", lams, flush=True)

    spec, rgb = build_cache(HERE.parent / "wave1" / "cache_leaf_spec_rgb.npz")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    phys = physics_features(spec)
    y = df[list(TARGETS)].to_numpy(np.float32)
    dom = df[list(VAR_COLS)].to_numpy(np.float32).argmax(1).astype(np.int64)

    agg = {}
    for seed in seeds:
        oof = {}
        for fold_name, tr_idx, te_idx in folds(df):
            y_mu = y[tr_idx].mean(0, keepdims=True)
            y_sd = y[tr_idx].std(0, keepdims=True) + 1e-6
            tr_raw = {"spec": spec[tr_idx], "phys": phys[tr_idx], "rgb": rgb[tr_idx], "x": spec[tr_idx],
                      "y": y[tr_idx], "dom": dom[tr_idx]}
            te_raw = {"spec": spec[te_idx], "phys": phys[te_idx], "rgb": rgb[te_idx], "x": spec[te_idx]}
            for key, cfg in candidates(lams):
                A, B = standardize(tr_raw, te_raw, cfg["keys"])
                A["y"] = ((tr_raw["y"] - y_mu) / y_sd).astype(np.float32)
                if cfg.get("dann"):
                    A["dom"] = tr_raw["dom"]
                pred = fit_torch2(cfg["build"], A, B, seed, device, epochs=args.epochs, dann=bool(cfg.get("dann")))
                oof.setdefault(key, np.zeros_like(y))
                oof[key][te_idx] = pred * y_sd + y_mu
            print(f"  seed{seed} fold {fold_name}: done", flush=True)
        for k, p in oof.items():
            agg.setdefault(k, []).append(per_target_r2(y, p))

    summary = {"device": str(device), "seeds": seeds, "lams": lams, "tag": args.tag, "candidates": {}}
    for k, lst in agg.items():
        pt = {t: round(float(np.mean([d[t] for d in lst])), 4) for t in TARGETS}
        mn = {t: round(float(np.min([d[t] for d in lst])), 4) for t in TARGETS}
        summary["candidates"][k] = {"per_target_r2_mean": pt, "per_target_r2_min": mn}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Wave-6 adversarial sweep ({args.tag})", "",
             f"- device: `{device}`, seeds: {seeds}, lams: {lams}, epochs: {args.epochs}",
             "- metric: pooled OOF R2, mean/min over seeds", "",
             "| candidate | GW | Gsw | PhiPS2 | min(Gsw) |", "|---|---|---|---|---|"]
    for k, v in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["per_target_r2_mean"]["Gsw"]):
        pt, mn = v["per_target_r2_mean"], v["per_target_r2_min"]
        lines.append(f"| `{k}` | {pt['GrainWeight']} | {pt['Gsw']} | {pt['PhiPS2']} | {mn['Gsw']} |")
    (out / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
