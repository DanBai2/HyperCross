import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import hypercross_routeA as h
from torch.utils.data import DataLoader

def hist_eval():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); import numpy as np; rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(len(pats)*.7); va=[r for r in rows if r[0] in set(pats[cut:])]; ds=h.HistologyDataset(root,va,cache_dir=Path(root)/'cache_routeA_v2'); dl=DataLoader(ds,batch_size=32)
    specs=[(h.ConcatModel(2),r'G:\Datasets\HyperCross\runs\hist_20261005_213113\concat_best.pt'),(h.CrossModalRouteA('cls',2),r'G:\Datasets\HyperCross\runs\hist_20261005_213829\routeA_best.pt'),(h.HybridRouteA('cls',2),r'G:\Datasets\HyperCross\runs\hist_20261005_214401\routeA_hybrid_best.pt')]
    models=[]
    for m,p in specs: m.load_state_dict(torch.load(p,map_location='cpu')['model']); models.append(m.cuda().eval())
    ys=[]; outs=[[] for _ in models]
    with torch.no_grad():
        for v,s,y in dl:
            v,s=v.cuda(),s.cuda(); ys.append(y)
            for j,m in enumerate(models): outs[j].append(m(v,s).cpu())
    y=torch.cat(ys); logits=[torch.cat(x) for x in outs]
    res={}
    for name,z in zip(['concat','routeA','hybrid'],logits): res[name]=float((z.argmax(1)==y).float().mean())
    for a in [(0,1),(0,2),(0,1,2)]: res['ensemble_'+'_'.join(map(str,a))]=float((sum(logits[i] for i in a).argmax(1)==y).float().mean())
    return res

def leaf_eval():
    root=r'G:\Datasets\HyperLeaf2024'; import pandas as pd, numpy as np; df=pd.read_csv(Path(root)/'train.csv'); idx=np.random.default_rng(42).permutation(len(df)); va=idx[int(.8*len(df)):]; ds=h.HyperLeafDataset(root,indices=va,cache_dir=Path(root)/'cache_routeA_leaf'); dl=DataLoader(ds,batch_size=64); specs=[(h.ConcatModel(3),r'G:\Datasets\HyperCross\runs\leaf_20261005_200532\concat_best.pt'),(h.CrossModalRouteA('reg',3),r'G:\Datasets\HyperCross\runs\leaf_20261005_214059\routeA_best.pt')]; models=[]
    for m,p in specs: m.load_state_dict(torch.load(p,map_location='cpu')['model']); models.append(m.cuda().eval())
    ys=[]; outs=[[] for _ in models]
    with torch.no_grad():
        for v,s,y in dl:
            v,s=v.cuda(),s.cuda(); ys.append(y)
            for j,m in enumerate(models): outs[j].append(m(v,s).cpu())
    y=torch.cat(ys); p=[torch.cat(x) for x in outs]; res={}
    for name,z in zip(['concat','routeA'],p): res[name]=float((1-((z-y)**2).mean(0)/(((y-y.mean(0))**2).mean(0)+1e-8)).mean())
    z=(p[0]+p[1])/2; res['ensemble_mean']=float((1-((z-y)**2).mean(0)/(((y-y.mean(0))**2).mean(0)+1e-8)).mean()); return res

if __name__=='__main__':
    out={'hist_patient_split':hist_eval(),'leaf_random_split':leaf_eval()}; print(json.dumps(out,indent=2)); Path(r'G:\Datasets\HyperCross\benchmark_summary.json').write_text(json.dumps(out,indent=2))
