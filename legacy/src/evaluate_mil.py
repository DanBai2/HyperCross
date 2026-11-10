import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import torch,numpy as np
from torch.utils.data import DataLoader
import hypercross_routeA as h
root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(.7*len(pats)); va=[r for r in rows if r[0] in set(pats[cut:])]; ds=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'); dl=DataLoader(ds,batch_size=32); m=h.MILRouteA(2); ck=torch.load(r'G:\Datasets\HyperCross\runs\hist_20261005_230107\routeA_mil_best.pt',map_location='cpu'); m.load_state_dict(ck['model']); m.cuda().eval(); ys=[]; ps=[]
with torch.no_grad():
    for v,s,y in dl: ps.append(m(v.cuda(),s.cuda()).argmax(1).cpu()); ys.append(y)
y=torch.cat(ys); p=torch.cat(ps); cm=torch.zeros(2,2,dtype=torch.long)
for a,b in zip(y,p): cm[a,b]+=1
out={'checkpoint_epoch':ck['epoch'],'accuracy':float((p==y).float().mean()),'confusion_matrix':cm.tolist(),'val_patients':pats[cut:],'n':len(y)}; print(json.dumps(out,indent=2)); Path(r'G:\Datasets\HyperCross\mil_eval.json').write_text(json.dumps(out,indent=2))
