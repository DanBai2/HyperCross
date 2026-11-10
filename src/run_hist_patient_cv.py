"""Five-fold patient-level CV for Local Route A and matched MIL-concat."""
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows})
    folds=[patients[i::5] for i in range(5)]
    out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); all_results=[]
    for fi,val_list in enumerate(folds):
        val_patients=set(val_list); tr=[r for r in rows if r[0] not in val_patients]; va=[r for r in rows if r[0] in val_patients]
        tl=DataLoader(h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'),16,shuffle=True); vl=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); fd=out/f'fold{fi}'; fd.mkdir()
        h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4
        h.train_one('routeA',h.MILRouteA(2),tl,vl,'cls',30,device,fd)
        h.seed_everything(42+fi); h.train_one('mil_concat',h.MILConcat(2),tl,vl,'cls',30,device,fd)
        def best(name):
            hist=json.loads((fd/f'{name}_history.json').read_text()); return max(hist,key=lambda x:x['acc'])
        row={'fold':fi,'val_patients':val_list,'n_val':len(va),'routeA':best('routeA'),'mil_concat':best('mil_concat')}; all_results.append(row); print(json.dumps(row),flush=True)
    def stats(name):
        x=np.array([r[name]['acc'] for r in all_results],float); return {'mean':float(x.mean()),'std':float(x.std()),'values':x.tolist()}
    result={'folds':all_results,'summary':{'routeA':stats('routeA'),'mil_concat':stats('mil_concat')}}; (out/'results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
