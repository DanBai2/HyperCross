"""Five-fold CV for a domain-invariant Local Route A visual encoder."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hypercross_routeA as h


class InvariantMILRouteA(h.MILRouteA):
    def __init__(self, out_dim=2, d=64, heads=4):
        super().__init__(out_dim, d, heads)
        self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.InstanceNorm2d(32,affine=True),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.InstanceNorm2d(d,affine=True),nn.GELU())


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_invariant_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        val_patients=set(val_list); tr=[r for r in rows if r[0] not in val_patients]; va=[r for r in rows if r[0] in val_patients]; tl=DataLoader(h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'),16,shuffle=True); vl=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; h.train_one('invariant_routeA',InvariantMILRouteA(2),tl,vl,'cls',30,device,od); hist=json.loads((od/'invariant_routeA_history.json').read_text()); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
