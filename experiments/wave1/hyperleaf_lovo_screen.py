"""Wave-1 exploratory harness: cross-variety HyperLeaf regression (LOVO).

EXPLORATORY LANE -- not governed benchmark evidence.
Split definition matches the governed LOVO protocol (leave one cultivar out,
3 train varieties -> 1 fully held-out variety).  All fitted state (scalers,
target normalization, feature statistics) uses train-fold rows only.  The
official HyperLeaf test partition and its labels are never opened.

Any winner must be re-run through benchmark/ before it is reported as evidence.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAF_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
WORK_ROOT = Path(r"G:\Datasets\HyperCross\experiments\wave1")
VAR_COLS = ("Heerup", "Kvium", "Rembrandt", "Sheriff")
TARGETS = ("GrainWeight", "Gsw", "PhiPS2")
SEEDS = (7, 42, 123)


def build_cache(cache_path: Path, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (spectrum[N,204], rgb[N,3,32,64]) for the 1590 training leaves."""
    if cache_path.exists():
        z = np.load(cache_path)
        return z["spec"], z["rgb"]
    import tifffile
    from PIL import Image

    df = pd.read_csv(LEAF_ROOT / "train.csv")
    ids = df["ImageId"].astype(int).to_numpy()
    spec = np.zeros((len(ids), 204), np.float32)
    rgb = np.zeros((len(ids), 3, 32, 64), np.float32)
    t0 = time.time()
    for i, fid in enumerate(ids):
        a = tifffile.imread(LEAF_ROOT / "images" / f"{fid:05d}.tiff").astype(np.float32) / 65535.0
        a = np.nan_to_num(a)
        spec[i] = a.reshape(a.shape[0], -1).mean(1)
        bands = a.shape[0]
        sl = np.stack([a[int(bands * 0.20)], a[int(bands * 0.48)], a[int(bands * 0.72)]], 0)
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
        rgb[i] = np.asarray(
            Image.fromarray((sl * 255).astype(np.uint8).transpose(1, 2, 0)).resize((64, 32), Image.BILINEAR),
            np.float32,
        ).transpose(2, 0, 1) / 255.0
        if verbose and (i + 1) % 200 == 0:
            print(f"  cached {i + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, spec=spec, rgb=rgb)
    return spec, rgb


def physics_features(spec: np.ndarray) -> np.ndarray:
    """Physics-informed spectral descriptors (no label information used)."""
    x = np.asarray(spec, np.float64)
    snv = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    d1 = np.gradient(snv, axis=1)
    d2 = np.gradient(d1, axis=1)
    # smoothed baseline; deviation from it is a cheap continuum-removal proxy
    k = 21
    pad = np.pad(snv, ((0, 0), (k // 2, k // 2)), mode="edge")
    kernel = np.ones(k) / k
    sm = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, pad)
    cr = snv - sm
    # narrow-band ratios on a coarse grid (position-robust, log domain)
    idx = np.linspace(2, x.shape[1] - 3, 24).astype(int)
    pos = snv - snv.min(1, keepdims=True) + 1.0
    ratios = np.log(pos[:, idx] / (pos[:, idx - 2] + 1e-6) + 1e-6)
    out = np.concatenate([snv, d1, d2, cr, ratios], 1).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def folds(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for v in VAR_COLS:
        held = df[v].astype(bool).to_numpy()
        out.append((v, np.where(~held)[0], np.where(held)[0]))
    return out


def pooled_r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def per_target_r2(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {t: round(1.0 - float(((y[:, i] - p[:, i]) ** 2).sum()) /
                     (float(((y[:, i] - y[:, i].mean()) ** 2).sum()) + 1e-12), 4)
            for i, t in enumerate(TARGETS)}


def per_target_mae(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {t: round(float(np.abs(y[:, i] - p[:, i]).mean()), 4) for i, t in enumerate(TARGETS)}

# ---------------------------------------------------------------- models ----
class Mlp(nn.Module):
    def __init__(self, n_in, n_out, hidden=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, n_out))

    def forward(self, **kw):
        x = kw.get("x")
        if x is None:
            x = next(iter(kw.values()))
        return self.net(x)


class SpectraNet(nn.Module):
    """Shared 1-D spectral trunk; also emits band-group tokens."""
    def __init__(self, d=128, n_tokens=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3), nn.GELU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.GELU(),
            nn.Conv1d(64, d, 5, 2, 2), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.proj = nn.Linear(d, d)

    def tokens(self, spec):
        h = self.net(spec[:, None])
        return self.proj(self.pool(h).transpose(1, 2))

    def forward(self, spec):
        return self.tokens(spec).mean(1)


class RgbNet(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, d, 3, 2, 1), nn.GELU())
        self.proj = nn.Linear(d, d)

    def tokens(self, rgb):
        return self.proj(self.net(rgb).flatten(2).transpose(1, 2))

    def forward(self, rgb):
        return self.tokens(rgb).mean(1)


class SpecMlp(nn.Module):
    def __init__(self, n_in, n_out, hidden=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def _head(n_in, n_out, hidden=256, drop=0.2):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, n_out))


class ConcatFusion(nn.Module):
    key = "fusion-concat"
    def __init__(self, n_phys, d=128, n_out=3):
        super().__init__()
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.phys = SpecMlp(n_phys, d)
        self.head = _head(3 * d, n_out)
    def forward(self, spec=None, rgb=None, phys=None):
        return self.head(torch.cat([self.rgb(rgb), self.spec(spec), F.gelu(self.phys(phys))], 1))


class ValueOnlyFusion(nn.Module):
    """Control: same token budgets and capacity, attention replaced by a plain
    linear value projection (no softmax, no Q/K) -- isolates attention itself."""
    key = "fusion-valueonly"
    def __init__(self, n_phys, d=128, n_out=3):
        super().__init__()
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.phys = SpecMlp(n_phys, d)
        self.v2s, self.s2v = nn.Linear(d, d), nn.Linear(d, d)
        self.vn, self.sn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = _head(3 * d, n_out)
    def forward(self, spec=None, rgb=None, phys=None):
        vt, st = self.rgb.tokens(rgb), self.spec.tokens(spec)
        vf = self.vn(vt + self.v2s(st).mean(1, keepdim=True))
        sf = self.sn(st + self.s2v(vt).mean(1, keepdim=True))
        return self.head(torch.cat([vf.mean(1), sf.mean(1), F.gelu(self.phys(phys))], 1))


class CrossAttnFusion(nn.Module):
    """Genuine patch-token x band-token bidirectional cross-attention."""
    key = "fusion-crossattn"
    def __init__(self, n_phys, d=128, n_out=3, heads=4):
        super().__init__()
        self.rgb, self.spec = RgbNet(d), SpectraNet(d)
        self.phys = SpecMlp(n_phys, d)
        self.v2s = nn.MultiheadAttention(d, heads, batch_first=True)
        self.s2v = nn.MultiheadAttention(d, heads, batch_first=True)
        self.vn, self.sn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = _head(3 * d, n_out)
    def forward(self, spec=None, rgb=None, phys=None):
        vt, st = self.rgb.tokens(rgb), self.spec.tokens(spec)
        vf = self.vn(vt + self.v2s(vt, st, st)[0])
        sf = self.sn(st + self.s2v(st, vt, vt)[0])
        return self.head(torch.cat([vf.mean(1), sf.mean(1), F.gelu(self.phys(phys))], 1))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class DannSpec(nn.Module):
    """Variety-adversarial spectral regressor (domain generalisation)."""
    key = "dann-spec"
    def __init__(self, n_in, n_out=3, n_var=4, hidden=256, drop=0.2, lam=0.3):
        super().__init__()
        self.lam = lam
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        self.reg = nn.Linear(hidden, n_out)
        self.dom = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_var))
    def forward(self, x=None, **kw):
        h = self.trunk(x)
        return self.reg(h)
    def domain_logits(self, x):
        h = self.trunk(x)
        return self.dom(GradReverse.apply(h, self.lam))

# ------------------------------------------------------------- training ----
def fit_torch(build, tr, te, seed, device, epochs=200, lr=1e-3, wd=1e-4, dann=False):
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
            opt.zero_grad(set_to_none=True)
            batch = {k: v[idx] for k, v in Xtr.items()}
            loss = nn.functional.mse_loss(model(**batch), ytr[idx])
            if dann:
                loss = loss + nn.functional.cross_entropy(model.domain_logits(batch["x"]), dtr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, Xte[keys[0]].shape[0], 512):
            chunks.append(model(**{k: v[i:i + 512] for k, v in Xte.items()}).cpu().numpy())
    return np.concatenate(chunks, 0)


def standardize(tr, te, keys):
    a, b = {}, {}
    for k in keys:
        mu = tr[k].mean(0, keepdims=True)
        sd = tr[k].std(0, keepdims=True) + 1e-6
        a[k] = ((tr[k] - mu) / sd).astype(np.float32)
        b[k] = ((te[k] - mu) / sd).astype(np.float32)
    return a, b


def ridge_fit_predict(Xtr, ytr, Xte, alphas=(0.1, 1.0, 10.0, 100.0), inner=4, seed=0):
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    order = rng.permutation(n)
    parts = np.array_split(order, inner)
    best, best_err = alphas[0], np.inf
    for a in alphas:
        errs = []
        for p in parts:
            m = np.ones(n, bool)
            m[p] = False
            w = np.linalg.solve(Xtr[m].T @ Xtr[m] + a * np.eye(Xtr.shape[1]), Xtr[m].T @ ytr[m])
            errs.append(float(((ytr[p] - Xtr[p] @ w) ** 2).mean()))
        if np.mean(errs) < best_err:
            best, best_err = a, float(np.mean(errs))
    w = np.linalg.solve(Xtr.T @ Xtr + best * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    return Xte @ w, best


def candidates(n_phys: int):
    """Ordered candidate registry: key -> (kind, spec)."""
    return [
        ("ridge-spec", "ridge", {"keys": ["spec"]}),
        ("ridge-phys", "ridge", {"keys": ["phys"]}),
        ("mlp-spec", "torch", {"keys": ["spec"], "build": lambda: Mlp(204, 3)}),
        ("mlp-phys", "torch", {"keys": ["phys"], "build": lambda: Mlp(n_phys, 3)}),
        ("cnn-rgb", "torch", {"keys": ["rgb"], "build": lambda: _RgbOnly(3)}),
        ("fusion-concat", "torch", {"keys": ["spec", "rgb", "phys"], "build": lambda: ConcatFusion(n_phys)}),
        ("fusion-valueonly", "torch", {"keys": ["spec", "rgb", "phys"], "build": lambda: ValueOnlyFusion(n_phys)}),
        ("fusion-crossattn", "torch", {"keys": ["spec", "rgb", "phys"], "build": lambda: CrossAttnFusion(n_phys)}),
        ("dann-spec", "torch", {"keys": ["x"], "build": lambda: DannSpec(204), "dann": True}),
    ]


class _RgbOnly(nn.Module):
    def __init__(self, n_out=3, d=128):
        super().__init__()
        self.rgb = RgbNet(d)
        self.head = _head(d, n_out)
    def forward(self, rgb=None, **kw):
        return self.head(self.rgb(rgb))


def run_fold(tr_raw, te_raw, y_mu, y_sd, n_phys, device, seed, epochs):
    tr_t = {k: v for k, v in tr_raw.items()}
    te_t = {k: v for k, v in te_raw.items()}
    tr_t["y"] = ((tr_raw["y"] - y_mu) / y_sd).astype(np.float32)
    tr_t["dom"] = tr_raw["dom"]
    out = {}
    for key, kind, cfg in candidates(n_phys):
        keys = cfg["keys"]
        a, b = standardize(tr_raw, te_raw, keys)
        A, B = dict(a), dict(b)
        A["y"] = tr_t["y"]
        if kind == "ridge":
            pred, alpha = ridge_fit_predict(A[keys[0]], A["y"], B[keys[0]], seed=seed)
        else:
            if cfg.get("dann"):
                A["dom"] = tr_raw["dom"]
                pred = fit_torch(cfg["build"], A, B, seed, device, epochs=epochs, dann=True)
            else:
                pred = fit_torch(cfg["build"], A, B, seed, device, epochs=epochs)
        out[key] = pred * y_sd + y_mu
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default=str(WORK_ROOT / "results"))
    ap.add_argument("--tag", default="screen")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "seeds", seeds, flush=True)

    spec, rgb = build_cache(WORK_ROOT / "cache_leaf_spec_rgb.npz")
    df = pd.read_csv(LEAF_ROOT / "train.csv")
    print("spec", spec.shape, "rgb", rgb.shape, flush=True)
    phys = physics_features(spec)
    print("phys", phys.shape, flush=True)
    y = df[list(TARGETS)].to_numpy(np.float32)
    dom = df[list(VAR_COLS)].to_numpy(np.float32).argmax(1).astype(np.int64)

    n_phys = phys.shape[1]
    agg: dict[str, dict[str, list]] = {}
    for seed in seeds:
        oof = {}
        for fold_name, tr_idx, te_idx in folds(df):
            y_mu = y[tr_idx].mean(0, keepdims=True)
            y_sd = y[tr_idx].std(0, keepdims=True) + 1e-6
            tr_raw = {"spec": spec[tr_idx], "phys": phys[tr_idx], "rgb": rgb[tr_idx], "x": spec[tr_idx],
                      "y": y[tr_idx], "dom": dom[tr_idx]}
            te_raw = {"spec": spec[te_idx], "phys": phys[te_idx], "rgb": rgb[te_idx], "x": spec[te_idx]}
            preds = run_fold(tr_raw, te_raw, y_mu, y_sd, n_phys, device, seed, args.epochs)
            for k, p in preds.items():
                oof.setdefault(k, np.zeros_like(y))
                oof[k][te_idx] = p
            print(f"  seed{seed} fold {fold_name}: done", flush=True)
        for k, p in oof.items():
            agg.setdefault(k, {"r2": [], "mae": [], "per_target_r2": []})
            agg[k]["r2"].append(round(pooled_r2(y, p), 4))
            agg[k]["mae"].append(per_target_mae(y, p))
            agg[k]["per_target_r2"].append(per_target_r2(y, p))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(device), "seeds": seeds, "epochs": args.epochs, "tag": args.tag,
               "n_samples": int(len(y)), "candidates": {}}
    for k, v in agg.items():
        pt = {t: round(float(np.mean([d[t] for d in v["per_target_r2"]])), 4) for t in TARGETS}
        summary["candidates"][k] = {
            "pooled_r2_mean_over_seeds": round(float(np.mean(v["r2"])), 4),
            "per_target_r2_mean": pt,
            "per_target_r2_min": {t: round(float(np.min([d[t] for d in v["per_target_r2"]])), 4) for t in TARGETS},
        }
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"# Wave-1 LOVO screen ({args.tag})", "",
             f"- device: `{device}`, seeds: {seeds}, epochs: {args.epochs}, n={len(y)}",
             "- metric: pooled OOF (all folds concatenated), R2 in original target units", "",
             "| candidate | GrainWeight | Gsw | PhiPS2 | min(Gsw) |", "|---|---|---|---|---|"]
    for k, v in sorted(summary["candidates"].items(), key=lambda kv: -kv[1]["per_target_r2_mean"]["Gsw"]):
        pt = v["per_target_r2_mean"]
        lines.append(f"| `{k}` | {pt['GrainWeight']} | {pt['Gsw']} | {pt['PhiPS2']} | {v['per_target_r2_min']['Gsw']} |")
    (out_dir / f"{args.tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
