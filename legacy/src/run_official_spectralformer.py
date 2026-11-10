"""Run the upstream SpectralFormer ViT/CAF implementation on our fixed splits.

The upstream model is used unchanged.  We only adapt a 128-bin spectrum to
the official band-neighbour input convention (three neighbouring bands per
token); morphology is ignored for this spectral-only benchmark.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hypercross_routeA as h

UPSTREAM = Path(r'G:\Datasets\HyperCross\external\spectralformer')
sys.path.insert(0, str(UPSTREAM))
from vit_pytorch import ViT  # noqa: E402


class OfficialSpectralFormer(nn.Module):
    def __init__(self, task='cls', out_dim=2, mode='CAF'):
        super().__init__()
        self.net = ViT(image_size=1, near_band=3, num_patches=128,
                       num_classes=out_dim, dim=64, depth=5, heads=4,
                       mlp_dim=8, dropout=.1, emb_dropout=.1, mode=mode)
        self.task = task
        self.temp = nn.Parameter(torch.tensor(.07).log())

    @staticmethod
    def neighbours(s):
        # [B,128] -> [B,128,3], matching official patch_to_embedding input.
        p = F.pad(s.unsqueeze(1), (1, 1), mode='reflect').squeeze(1)
        return p.unfold(1, 3, 1)

    def forward(self, v, s, return_aux=False):
        if s.dim() == 3:
            b, n, bands = s.shape
            q = self.net(self.neighbours(s.reshape(b*n, bands))).reshape(b, n, -1).mean(1)
        else:
            q = self.net(self.neighbours(s))
        # Upstream ViT does not expose a pooled feature; disable auxiliary
        # alignment/NCE weights for this spectral-only benchmark.
        return (q, q, q) if return_aux else q


def leaf_loaders():
    root = r'G:\Datasets\HyperLeaf2024'; df = pd.read_csv(Path(root)/'train.csv')
    idx = np.random.default_rng(42).permutation(len(df)); cut = int(.8 * len(df)); tr, va = idx[:cut], idx[cut:]
    train = h.HyperLeafDataset(root, indices=tr, cache_dir=Path(root)/'cache_routeA_leaf')
    val = h.HyperLeafDataset(root, indices=va, cache_dir=Path(root)/'cache_routeA_leaf')
    val.mu, val.sd = train.mu, train.sd
    return DataLoader(train, 64, shuffle=True), DataLoader(val, 128), 3, 'reg'


def hist_loaders():
    root = r'G:\Datasets\PKG - HistologyHSI-GB'; rows = h.build_hist_rows(root); pats = sorted({r[0] for r in rows})
    rng = np.random.default_rng(42); rng.shuffle(pats); cut = int(.7 * len(pats)); trp, vap = set(pats[:cut]), set(pats[cut:])
    tr = [r for r in rows if r[0] in trp]; va = [r for r in rows if r[0] in vap]
    train = h.HistologyDataset(root, tr, cache_dir=Path(root) / 'cache_routeA_v2')
    val = h.HistologyDataset(root, va, cache_dir=Path(root) / 'cache_routeA_v2')
    return DataLoader(train, 16, shuffle=True), DataLoader(val, 32), 2, 'cls'


def main():
    h.seed_everything(42); out = Path(r'G:\Datasets\HyperCross\runs') / ('official_spectralformer_' + time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); results = {}
    for kind, fn in [('leaf', leaf_loaders), ('hist', hist_loaders)]:
        tl, vl, out_dim, task = fn(); od = out / kind; od.mkdir()
        h.GLOBAL_ALIGN_W, h.GLOBAL_NCE_W, h.GLOBAL_BALANCE, h.GLOBAL_LR = 0., 0., False, 2e-4
        h.train_one('spectralformer_official_caf', OfficialSpectralFormer(task, out_dim), tl, vl, task, 30, device, od)
        results[kind] = {'task': task, 'out_dim': out_dim}
    (out / 'results.json').write_text(json.dumps(results, indent=2)); print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__': main()
