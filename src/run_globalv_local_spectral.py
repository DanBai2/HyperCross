"""Global visual + local spectral Transformer fusion benchmark."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hypercross_routeA as h


class GlobalVLocalSpectral(nn.Module):
    """MamTrans-style fusion with global morphology and local spectral tokens."""
    def __init__(self, task='cls', out_dim=2, d=96, heads=4, depth=2):
        super().__init__()
        self.v = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, d, 3, 2, 1), nn.GELU())
        self.s = nn.Sequential(nn.Linear(128, d), nn.GELU(), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, 16, d)); nn.init.normal_(self.pos, std=.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4*d, dropout=.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(.2), nn.Linear(d, out_dim))
        self.temp = nn.Parameter(torch.tensor(.07).log()); self.task = task

    def forward(self, v, s, return_aux=False):
        zv = self.v(v).mean((2, 3))
        zs = self.norm(self.enc(self.s(s) + self.pos).mean(1))
        g = self.gate(torch.cat([zv, zs], 1)); z = g*zv + (1-g)*zs
        out = self.head(z)
        return (out, zv, zs) if return_aux else out


def loaders(kind):
    if kind == 'leaf':
        root = r'G:\Datasets\HyperLeaf2024'; df = pd.read_csv(Path(root)/'train.csv')
        idx = np.random.default_rng(42).permutation(len(df)); cut = int(.8*len(df)); tr, va = idx[:cut], idx[cut:]
        train = h.HyperLeafMILDataset(root, tr, cache_dir=Path(root)/'cache_routeA_leaf_mil')
        val = h.HyperLeafMILDataset(root, va, cache_dir=Path(root)/'cache_routeA_leaf_mil', mu=train.mu, sd=train.sd)
        return DataLoader(train,64,shuffle=True), DataLoader(val,128), 3, 'reg'
    root = r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(.7*len(pats)); trp=set(pats[:cut]); vap=set(pats[cut:])
    tr=[r for r in rows if r[0] in trp]; va=[r for r in rows if r[0] in vap]
    train=h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'); val=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil')
    return DataLoader(train,16,shuffle=True), DataLoader(val,32), 2, 'cls'


def main():
    h.seed_everything(42); out=Path(r'G:\Datasets\HyperCross\runs')/('globalv_local_spectral_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); results={}
    for kind in ('leaf','hist'):
        tl,vl,out_dim,task=loaders(kind); od=out/kind; od.mkdir(); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4
        h.train_one('globalv_local_spectral',GlobalVLocalSpectral(task,out_dim),tl,vl,task,30,device,od); results[kind]={'task':task,'out_dim':out_dim}
    (out/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()
