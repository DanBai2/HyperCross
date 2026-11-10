import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h
from modern_hsi_models import MamTransLite


def collect(model_cls, paths, loader, device, kind):
    out=[]
    for p in paths:
        model=model_cls(2).to(device); ck=torch.load(p,map_location=device); model.load_state_dict(ck['model']); model.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for v,s,y in loader:
                q=torch.softmax(model(v.to(device),s.to(device)),1).cpu(); ps.append(q); ys.append(y)
        out.append(torch.cat(ps)); y=torch.cat(ys); assert len(out[-1])==len(y)
    return out, y


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(.7*len(pats)); vap=set(pats[cut:]); va=[r for r in rows if r[0] in vap]
    ds_local=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'); ds_global=h.HistologyDataset(root,va,cache_dir=Path(root)/'cache_routeA_v2'); loader_local=DataLoader(ds_local,32); loader_global=DataLoader(ds_global,32); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rr=[Path(r'G:\Datasets\HyperCross\runs\hist_20261005_230107\routeA_mil_best.pt'),Path(r'G:\Datasets\HyperCross\runs\hist_20261005_230212\routeA_mil_best.pt'),Path(r'G:\Datasets\HyperCross\runs\hist_20261005_230301\routeA_mil_best.pt')]
    mm=[Path(r'G:\Datasets\HyperCross\runs\mamtrans_hist_multiseed_20261006_033503\seed42\mamtrans_lite_best.pt'),Path(r'G:\Datasets\HyperCross\runs\mamtrans_hist_multiseed_20261006_033503\seed123\mamtrans_lite_best.pt'),Path(r'G:\Datasets\HyperCross\runs\mamtrans_hist_multiseed_20261006_033503\seed7\mamtrans_lite_best.pt')]
    rp,y=collect(h.MILRouteA,rr,loader_local,device,'route'); mp,_=collect(MamTransLite,mm,loader_global,device,'mam'); y=y.numpy(); result={}
    def acc(p): return float((p.argmax(1).numpy()==y).mean())
    result['route_mean']=acc(sum(rp)/len(rp)); result['mam_mean']=acc(sum(mp)/len(mp)); result['route_each']=[acc(p) for p in rp]; result['mam_each']=[acc(p) for p in mp]
    for a in np.linspace(0,1,21): result[f'route_weight_{a:.2f}']=acc(a*sum(rp)/len(rp)+(1-a)*sum(mp)/len(mp))
    for name,p in [('route_seed42',rp[0]),('route_seed123',rp[1]),('route_seed7',rp[2]),('mam_seed42',mp[0]),('mam_seed123',mp[1]),('mam_seed7',mp[2])]: result[name]=acc(p)
    print(json.dumps(result,indent=2)); Path(r'G:\Datasets\HyperCross\hist_ensemble_results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
