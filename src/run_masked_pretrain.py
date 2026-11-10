"""Masked local spectral-spatial pretraining for the Route A MIL encoder.

The pretext task masks local visual and spectral tokens and reconstructs their
low-dimensional visual patch averages and 128-bin spectra.  The same encoder
is then fine-tuned with the existing supervised objective on the fixed splits.
"""
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hypercross_routeA as h


class MaskedMILRouteA(nn.Module):
    """MILRouteA encoder with masked local token reconstruction heads."""

    def __init__(self, out_dim=2, d=64, heads=4):
        super().__init__()
        self.v = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, d, 3, 2, 1), nn.GELU()
        )
        self.s = nn.Sequential(nn.Linear(128, d), nn.GELU(), nn.LayerNorm(d))
        self.v2s = nn.MultiheadAttention(d, heads, batch_first=True)
        self.s2v = nn.MultiheadAttention(d, heads, batch_first=True)
        self.vn = nn.LayerNorm(d)
        self.sn = nn.LayerNorm(d)
        self.vpool = nn.Linear(d, 1)
        self.spool = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(.3), nn.Linear(d, out_dim))
        self.temp = nn.Parameter(torch.tensor(.07).log())
        self.mask_v = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_s = nn.Parameter(torch.zeros(1, 1, d))
        self.v_recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.s_recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))
        nn.init.normal_(self.mask_v, std=.02)
        nn.init.normal_(self.mask_s, std=.02)
        self.task = 'cls'

    def _tokens(self, v, s, mask_v=None, mask_s=None):
        raw_v = self.v(v)
        hh, ww = raw_v.shape[-2:]
        vt = raw_v.flatten(2).transpose(1, 2)
        st = self.s(s)
        if mask_v is not None:
            mv = self.mask_v.expand(vt.shape[0], vt.shape[1], -1)
            vt = torch.where(mask_v.unsqueeze(-1), mv, vt)
        if mask_s is not None:
            ms = self.mask_s.expand(st.shape[0], st.shape[1], -1)
            st = torch.where(mask_s.unsqueeze(-1), ms, st)
        av = self.v2s(vt, st, st)[0]
        ass = self.s2v(st, vt, vt)[0]
        vf = self.vn(vt + av)
        sf = self.sn(st + ass)
        return vf, sf, hh, ww

    def forward(self, v, s, return_aux=False):
        vf, sf, _, _ = self._tokens(v, s)
        wa = F.softmax(self.vpool(vf).squeeze(-1), 1)
        wb = F.softmax(self.spool(sf).squeeze(-1), 1)
        zv = (wa.unsqueeze(-1) * vf).sum(1)
        zs = (wb.unsqueeze(-1) * sf).sum(1)
        out = self.head(torch.cat([zv, zs], 1))
        return (out, zv, zs) if return_aux else out

    def masked_loss(self, v, s, mask_ratio=.35):
        with torch.no_grad():
            # The convolutional token grid is the reconstruction target grid.
            _, _, hh, ww = self._tokens(v, s)
            tv = F.adaptive_avg_pool2d(v, (hh, ww)).flatten(2).transpose(1, 2)
        ts = s
        vf_mask = torch.rand(v.shape[0], hh * ww, device=v.device) < mask_ratio
        sf_mask = torch.rand(v.shape[0], s.shape[1], device=s.device) < mask_ratio
        # Ensure every sample contributes to both masked reconstruction losses.
        vf_mask[:, 0] = True
        sf_mask[:, 0] = True
        vf, sf, _, _ = self._tokens(v, s, vf_mask, sf_mask)
        pv = self.v_recon(vf)
        ps = self.s_recon(sf)
        lv = F.mse_loss(pv[vf_mask], tv[vf_mask])
        ls = F.mse_loss(ps[sf_mask], ts[sf_mask])
        return lv + ls, {'visual': float(lv.detach()), 'spectral': float(ls.detach()),
                         'masked_visual': float(vf_mask.float().mean()),
                         'masked_spectral': float(sf_mask.float().mean())}


def split_leaf(root):
    df = pd.read_csv(Path(root) / 'train.csv')
    idx = np.random.default_rng(42).permutation(len(df))
    cut = int(.8 * len(df))
    return idx[:cut], idx[cut:]


def split_hist(root):
    rows = h.build_hist_rows(root)
    patients = sorted({r[0] for r in rows})
    rng = np.random.default_rng(42)
    rng.shuffle(patients)
    cut = int(.7 * len(patients))
    trp, vap = set(patients[:cut]), set(patients[cut:])
    return [r for r in rows if r[0] in trp], [r for r in rows if r[0] in vap], sorted(trp), sorted(vap)


def pretrain(model, loader, epochs, device, outdir, name):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); total = 0.; n = 0; lv = 0.; ls = 0.
        for v, s, _ in loader:
            v, s = v.to(device), s.to(device)
            opt.zero_grad(set_to_none=True)
            loss, aux = model.masked_loss(v, s)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total += float(loss.item()) * len(v); n += len(v)
            lv += aux['visual'] * len(v); ls += aux['spectral'] * len(v)
        sch.step()
        row = {'epoch': ep, 'loss': total / n, 'visual_loss': lv / n, 'spectral_loss': ls / n}
        hist.append(row)
        if ep == 1 or ep % 5 == 0:
            print(name, 'pretrain', ep, row, flush=True)
    torch.save({'model': model.state_dict(), 'epoch': epochs, 'metrics': hist[-1]}, outdir / f'{name}_pretrained.pt')
    (outdir / f'{name}_pretrain_history.json').write_text(json.dumps(hist, indent=2))
    return hist[-1]


def run_dataset(kind, out, device, pre_epochs=20, fine_epochs=30):
    if kind == 'leaf':
        root = r'G:\Datasets\HyperLeaf2024'
        tr, va = split_leaf(root)
        train = h.HyperLeafMILDataset(root, tr, cache_dir=Path(root) / 'cache_routeA_leaf_mil', augment=False)
        val = h.HyperLeafMILDataset(root, va, cache_dir=Path(root) / 'cache_routeA_leaf_mil', mu=train.mu, sd=train.sd)
        out_dim, task, batch = 3, 'reg', 64
    else:
        root = r'G:\Datasets\PKG - HistologyHSI-GB'
        tr, va, trp, vap = split_hist(root)
        train = h.HistologyMILDataset(root, tr, cache_dir=Path(root) / 'cache_routeA_hist_mil', augment=False)
        val = h.HistologyMILDataset(root, va, cache_dir=Path(root) / 'cache_routeA_hist_mil')
        out_dim, task, batch = 2, 'cls', 16
    tl = DataLoader(train, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val, batch_size=batch * 2, shuffle=False, num_workers=0, pin_memory=True)
    pre_dir, scratch_dir, ft_dir = out / kind / 'pretrain', out / kind / 'scratch', out / kind / 'finetune'
    for d in [pre_dir, scratch_dir, ft_dir]: d.mkdir(parents=True, exist_ok=True)

    h.seed_everything(42)
    pre = MaskedMILRouteA(out_dim)
    pre_metrics = pretrain(pre, tl, pre_epochs, device, pre_dir, 'masked_routeA')

    h.seed_everything(42)
    scratch = MaskedMILRouteA(out_dim)
    h.GLOBAL_ALIGN_W, h.GLOBAL_NCE_W, h.GLOBAL_BALANCE, h.GLOBAL_LR = .08, .04, False, 2e-4
    h.train_one('masked_scratch', scratch, tl, vl, task, fine_epochs, device, scratch_dir)

    h.seed_everything(42)
    finetune = MaskedMILRouteA(out_dim)
    finetune.load_state_dict(pre.state_dict(), strict=False)
    h.train_one('masked_pretrained', finetune, tl, vl, task, fine_epochs, device, ft_dir)
    result = {'pretrain': pre_metrics, 'split': {'train': len(train), 'val': len(val)}}
    if kind == 'hist': result['split'].update({'train_patients': trp, 'val_patients': vap})
    return result


def main():
    h.seed_everything(42)
    out = Path(r'G:\Datasets\HyperCross\runs') / ('masked_pretrain_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device, flush=True)
    results = {'leaf': run_dataset('leaf', out, device), 'hist': run_dataset('hist', out, device)}
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
