"""Five-fold CV with equal patient probability sampling."""
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import hypercross_routeA as h


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_patientbalanced_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; ds=h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'); dv=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'); counts={p:sum(r[0]==p for r in tr) for p in sorted({r[0] for r in tr})}; weights=torch.DoubleTensor([1/counts[r[0]] for r in tr]); sampler=WeightedRandomSampler(weights,len(weights),replacement=True); tl=DataLoader(ds,16,sampler=sampler); vl=DataLoader(dv,32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; h.train_one('patientbalanced_routeA',h.MILRouteA(2),tl,vl,'cls',30,device,od); hist=json.loads((od/'patientbalanced_routeA_history.json').read_text()); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
