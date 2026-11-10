import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h
from modern_hsi_models import MamTransLite


def main():
    root = r'G:\Datasets\PKG - HistologyHSI-GB'
    rows = h.build_hist_rows(root); pats = sorted({r[0] for r in rows}); rng = np.random.default_rng(42); rng.shuffle(pats); cut = int(.7 * len(pats)); trp=set(pats[:cut]); vap=set(pats[cut:])
    tr=[r for r in rows if r[0] in trp]; va=[r for r in rows if r[0] in vap]
    train=h.HistologyDataset(root,tr,cache_dir=Path(root)/'cache_routeA_v2'); val=h.HistologyDataset(root,va,cache_dir=Path(root)/'cache_routeA_v2')
    tl=DataLoader(train,16,shuffle=True); vl=DataLoader(val,32); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(r'G:\Datasets\HyperCross\runs')/('mamtrans_hist_multiseed_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); results={}
    for seed in [42,123,7]:
        h.seed_everything(seed); od=out/f'seed{seed}'; od.mkdir(); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4
        h.train_one('mamtrans_lite',MamTransLite(2),tl,vl,'cls',30,device,od); results[str(seed)]={'outdir':str(od)}
    (out/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))

if __name__=='__main__': main()
