"""Five-fold CV with train-patient-only RGB/spectral normalization."""
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import hypercross_routeA as h


class TrainNormDataset(Dataset):
    def __init__(self, base, mean_v, std_v, mean_s, std_s): self.base=base; self.mv=mean_v; self.sv=std_v; self.ms=mean_s; self.ss=std_s
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        v,s,y=self.base[i]; v=(v-self.mv[:,None,None])/self.sv[:,None,None]; s=(s-self.ms[None,:])/self.ss[None,:]; return v,s,y


def stats(ds):
    sv=ss=nv=ns=0.
    for i in range(len(ds)):
        v,s,_=ds[i]; sv+=v.double().sum((1,2)); ss+=s.double().sum((0,)); nv+=v.shape[1]*v.shape[2]; ns+=s.shape[0]
    mv=sv/nv; ms=ss/ns; vv=ssv=0.
    for i in range(len(ds)):
        v,s,_=ds[i]; vv+=(v.double()-mv[:,None,None]).pow(2).sum((1,2)); ssv+=(s.double()-ms[None,:]).pow(2).sum(0)
    return mv.float(),(vv/nv).sqrt().clamp_min(1e-3).float(),ms.float(),(ssv/ns).sqrt().clamp_min(1e-3).float()


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_trainnorm_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; raw_tr=h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'); raw_va=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'); mv,sv,ms,ss=stats(raw_tr); tl=DataLoader(TrainNormDataset(raw_tr,mv,sv,ms,ss),16,shuffle=True); vl=DataLoader(TrainNormDataset(raw_va,mv,sv,ms,ss),32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; h.train_one('trainnorm_routeA',h.MILRouteA(2),tl,vl,'cls',30,device,od); hist=json.loads((od/'trainnorm_routeA_history.json').read_text()); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
