import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import torch, numpy as np
import hypercross_routeA as h
from torch.utils.data import DataLoader
root=r'G:\Datasets\PKG - HistologyHSI-GB'; out=Path(r'G:\Datasets\HyperCross\runs')/'hist_finetune_residual'; out.mkdir(parents=True,exist_ok=True)
h.seed_everything(42); h.GLOBAL_ALIGN_W=0.; h.GLOBAL_NCE_W=0.; h.GLOBAL_BALANCE=False; h.GLOBAL_LR=5e-5
rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(.7*len(pats)); tr=[r for r in rows if r[0] in set(pats[:cut])]; va=[r for r in rows if r[0] in set(pats[cut:])]
tl=DataLoader(h.HistologyDataset(root,tr,cache_dir=Path(root)/'cache_routeA_v2'),batch_size=16,shuffle=True); vl=DataLoader(h.HistologyDataset(root,va,cache_dir=Path(root)/'cache_routeA_v2'),batch_size=32)
base=h.ConcatModel(2); base.load_state_dict(torch.load(r'G:\Datasets\HyperCross\runs\hist_20261005_213113\concat_best.pt',map_location='cpu')['model']); m=h.ResidualRouteA('cls',2)
with torch.no_grad():
    m.ve.net.load_state_dict(base.v.net.state_dict()); m.ve.proj.load_state_dict(base.v.proj.state_dict()); m.se.conv.load_state_dict(base.s.conv.state_dict()); m.se.proj.load_state_dict(base.s.proj.state_dict()); m.base[0].load_state_dict(base.h[0].state_dict()); m.base[2].load_state_dict(base.h[2].state_dict())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); h.train_one('routeA_residual_finetune',m,tl,vl,'cls',60,device,out)
(out/'meta.json').write_text(json.dumps({'split_patients_train':pats[:cut],'split_patients_val':pats[cut:],'init':'concat_best','residual_init':'zero','lr':5e-5},indent=2))
