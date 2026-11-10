"""Wave-2: fold-fitted PCA pseudo-RGB view + cross-view consistency.

EXPLORATORY LANE -- not governed benchmark evidence.

Wave-1 showed (a) genuine cross-attention adds nothing, (b) variety-adversarial
training is the only cross-variety signal, (c) every candidate has negative Gsw
R2, and (d) the governed benchmark reaches Gsw ~0.10 using a fold-fitted PCA
pseudo-RGB view plus a consistency regulariser and per-target routing.

Wave-2 adds exactly those missing ingredients, honestly:
  * PCA is fitted on TRAIN-FOLD cubes only (never the held-out variety);
  * consistency is an unlabelled objective between the RGB and PCA views;
  * the held-out variety label is never used for anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wave1"))
from hyperleaf_lovo_screen import (  # noqa: E402
    LEAF_ROOT, TARGETS, VAR_COLS, WORK_ROOT, build_cache, folds,
    per_target_mae, per_target_r2, pooled_r2, RgbNet, SpectraNet, DannSpec,
    _head, standardize, Mlp)


def build_cube_cache(path: Path) -> np.ndarray:
    """Cache a 4x-strided cube (N, 204, 12, 88) as float16."""
    if path.exists():
        return np.load(path, mmap_mode="r")
    import tifffile

    df = pd.read_csv(LEAF_ROOT / "train.csv")
    ids = df["ImageId"].astype(int).to_numpy()
    cube = np.zeros((len(ids), 204, 12, 88), np.float16)
    t0 = time.time()
    for i, fid in enumerate(ids):
        a = tifffile.imread(LEAF_ROOT / "images" / f"{fid:05d}.tiff").astype(np.float32) / 65535.0
        cube[i] = np.nan_to_num(a)[:, ::4, ::4]
        if (i + 1) % 400 == 0:
            print(f"  cube {i + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(path, cube)
    return np.load(path, mmap_mode="r")


def fit_pca_view(cube: np.ndarray, tr_idx: np.ndarray, n_pc: int = 3, max_pixels: int = 20000, seed: int = 0):
    """Fold-fitted PCA pseudo-RGB. Fit uses train-fold pixels only."""
    rng = np.random.default_rng(seed)
    n = cube.shape[0]
    pixels = np.asarray(cube[tr_idx], np.float32).transpose(0, 2, 3, 1).reshape(-1, cube.shape[1])
    if pixels.shape[0] > max_pixels:
        pixels = pixels[rng.choice(pixels.shape[0], max_pixels, replace=False)]
    mu = pixels.mean(0, keepdims=True)
    sd = pixels.std(0, keepdims=True) + 1e-6
    _, _, vt = np.linalg.svd((pixels - mu) / sd, full_matrices=False)
    comps = vt[:n_pc].astype(np.float32)
    out = np.zeros((n, n_pc, 32, 64), np.float32)
    from PIL import Image
    for i in range(n):
        y = (np.asarray(cube[i], np.float32).reshape(cube.shape[1], -1).T - mu) / sd
        z = (y @ comps.T).reshape(12, 88, n_pc).transpose(2, 0, 1)
        lo = z.min(axis=(1, 2), keepdims=True)
        hi = z.max(axis=(1, 2), keepdims=True)
        z = (z - lo) / (hi - lo + 1e-6)
        out[i] = np.asarray(
            Image.fromarray((z.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((64, 32), Image.BILINEAR),
            np.float32).transpose(2, 0, 1) / 255.0
    return out


class DualViewConsistency(nn.Module):
    """RGB view + PCA view + spectral view with an unlabelled consistency term."""
    key = "dual-consistency"

    def __init__(self, d=128, n_out=3, cons_w=0.05, dann=False, n_var=4, lam=0.3):
        super().__init__()
        self.rgb_view, self.pca_view = RgbNet(d), RgbNet(d)
        self.spec = SpectraNet(d)
        self.head = _head(3 * d, n_out)
        self.cons_w = cons_w
        self.dann = dann
        if dann:
            self.dom = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, n_var))
            self.lam = lam

    def embeddings(self, rgb, pca, spec):
        return self.rgb_view(rgb), self.pca_view(pca), self.spec(spec)

    def forward(self, rgb=None, pca=None, spec=None, **kw):
        zr, zp, zs = self.embeddings(rgb, pca, spec)
        return self.head(torch.cat([zr, zp, zs], 1))

    def consistency(self, rgb, pca):
        zr = F.normalize(self.rgb_view(rgb), dim=1)
        zp = F.normalize(self.pca_view(pca), dim=1)
        return ((zr - zp) ** 2).mean()

    def domain_logits(self, spec):
        z = self.spec(spec)
        return self.dom(-self.lam * z + z.detach() * (1 + self.lam))


class PcaOnly(nn.Module):
    key = "pca-cnn"
    def __init__(self, d=128, n_out=3):
        super().__init__()
        self.pca = RgbNet(d)
        self.head = _head(d, n_out)
    def forward(self, pca=None, **kw):
        return self.head(self.pca(pca))


class DualNoConsistency(DualViewConsistency):
    """Control: identical architecture, consistency weight zero."""
    key = "dual-noconsistency"
    def __init__(self, d=128, n_out=3, **kw):
        super().__init__(d=d, n_out=n_out, cons_w=0.0, dann=False)


def fit_w2(build, tr, te, seed, device, epochs=200, lr=1e-3, wd=1e-4, cons_w=0.0, dann=False):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    keys = [k for k in ("spec", "rgb", "pca", "x") if k in tr]
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
            if cons_w > 0 and "rgb" in b and "pca" in b:
                loss = loss + cons_w * model.consistency(b["rgb"], b["pca"])
            if dann:
                src = b.get("x", b.get("spec"))
                loss = loss + nn.functional.cross_entropy(model.domain_logits(src), dtr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, Xte[keys[0]].shape[0], 512):
            chunks.append(model(**{k: v[i:i + 512] for k, v in Xte.items()}).cpu().numpy())
    return np.concatenate(chunks, 0)


def candidates_w2(cons_w: float):
    return [
        ("mlp-spec", "torch", {"keys": ["spec"], "build": lambda: Mlp(204, 3)}),
        ("dann-spec", "torch", {"keys": ["x"], "build": lambda: DannSpec(204), "dann": True}),
        ("pca-cnn", "torch", {"keys": ["pca"], "build": lambda: PcaOnly()}),
        ("dual-noconsistency", "torch", {"keys": ["rgb", "pca", "spec"], "build": lambda: DualNoConsistency(), "cons": 0.0}),
        ("dual-consistency", "torch", {"keys": ["rgb", "pca", "spec"], "build": lambda: DualViewConsistency(cons_w=cons_w), "cons": cons_w}),
        ("dual-consistency-dann", "torch", {"keys": ["rgb", "pca", "spec", "x"], "build": lambda: DualViewConsistency(cons_w=cons_w, dann=True), "cons": cons_w, "dann": True}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--consistency", type=float, default=0.05)
    ap.add_argument("--out", default=str(WORK_ROOT.parent / "runs_v3" / "wave2"))
    ap.add_argument("--tag", default="dualview")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "seeds", seeds, "consistency", args.consistency, flush=True)

    spec, rgb = build_cache(HERE.parent / "wave1" / "cache_leaf_spec_rgb.npz")
    cube = build_cube_cache(HERE.parent / "wave1" / "cache_leaf_cube_small.npy")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    y = df[list(TARGETS)].to_numpy(np.float32)
    dom = df[list(VAR_COLS)].to_numpy(np.float32).argmax(1).astype(np.int64)
    print("spec", spec.shape, "rgb", rgb.shape, "cube", cube.shape, flush=True)

    agg: dict[str, dict[str, list]] = {}
    for seed in seeds:
        oof: dict[str, np.ndarray] = {}
        for fold_name, tr_idx, te_idx in folds(df):
            pca = fit_pca_view(cube, tr_idx, seed=seed)
            y_mu = y[tr_idx].mean(0, keepdims=True)
            y_sd = y[tr_idx].std(0, keepdims=True) + 1e-6
            tr_raw = {"spec": spec[tr_idx], "rgb": rgb[tr_idx], "pca": pca[tr_idx], "x": spec[tr_idx],
                      "y": y[tr_idx], "dom": dom[tr_idx]}
            te_raw = {"spec": spec[te_idx], "rgb": rgb[te_idx], "pca": pca[te_idx], "x": spec[te_idx]}
            for key, kind, cfg in candidates_w2(args.consistency):
                keys = cfg["keys"]
                a, b = standardize(tr_raw, te_raw, keys)
                A, B = dict(a), dict(b)
                A["y"] = ((tr_raw["y"] - y_mu) / y_sd).astype(np.float32)
                if cfg.get("dann"):
                    A["dom"] = tr_raw["dom"]
                pred = fit_w2(cfg["build"], A, B, seed, device, epochs=args.epochs,
                              cons_w=cfg.get("cons", 0.0), dann=bool(cfg.get("dann")))
                p = pred * y_sd + y_mu
                oof.setdefault(key, np.zeros_like(y))
                oof[key][te_idx] = p
            print(f"  seed{seed} fold {fold_name}: done", flush=True)
        for k, p in oof.items():
            agg.setdefault(k, {"r2": [], "per_target_r2": []})
            agg[k]["r2"].append(round(pooled_r2(y, p), 4))
            agg[k]["per_target_r2"].append(per_target_r2(y, p))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(device), "seeds": seeds, "epochs": args.epochs,
               "consistency": args.consistency, "tag": args.tag, "n_samples": int(len(y)), "candidates": {}}
    for k, v in agg.items():
        pt = {t: round(float(np.mean([d[t] for d in v["per_target_r2"]])), 4) for t in TARGETS}
        summary["candidates"][k] = {
            "pooled_r2_mean_over_seeds": round(float(np.mean(v["r2"])), 4),
            "per_target_r2_mean": pt,
            "per_target_r2_min": {t: round(float(np.min([d[t] for d in v["per_target_r2"]])), 4) for t in TARGETS},
        }
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"# Wave-2 dual-view LOVO ({args.tag})", "",
             f"- device: `{device}`, seeds: {seeds}, epochs: {args.epochs}, consistency: {args.consistency}, n={len(y)}",
             "- metric: pooled OOF R2 (all folds concatenated, original target units)", "",
             "| candidate | GrainWeight | Gsw | PhiPS2 | min(Gsw) |", "|---|---|---|---|---|"]
    for k, v in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["per_target_r2_mean"]["Gsw"]):
        pt = v["per_target_r2_mean"]
        lines.append(f"| `{k}` | {pt['GrainWeight']} | {pt['Gsw']} | {pt['PhiPS2']} | {v['per_target_r2_min']['Gsw']} |")
    (out_dir / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
