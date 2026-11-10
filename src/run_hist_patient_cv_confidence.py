"""Five-fold CV for entropy-weighted confidence-aware modality fusion."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hypercross_routeA as h


class ConfidenceMILRouteA(h.MILRouteA):
    def __init__(self, out_dim=2, d=64, heads=4, temp=.5):
        super().__init__(out_dim,d,heads); self.vhead=nn.Linear(d,out_dim); self.shead=nn.Linear(d,out_dim); self.conf_temp=temp

    def forward(self,v,s,return_aux=False):
        vt=self.v(v).flatten(2).transpose(1,2); st=self.s(s); av=self.v2s(vt,st,st)[0]; ass=self.s2v(st,vt,vt)[0]; vf=self.vn(vt+av); sf=self.sn(st+ass); wa=F.softmax(self.vpool(vf).squeeze(-1),1); wb=F.softmax(self.spool(sf).squeeze(-1),1); zv=(wa.unsqueeze(-1)*vf).sum(1); zs=(wb.unsqueeze(-1)*sf).sum(1); lv=self.vhead(zv); ls=self.shead(zs); pv=F.softmax(lv,1); ps=F.softmax(ls,1); logc=np.log(lv.shape[1]); cv=1-( -(pv*(pv.clamp_min(1e-8).log())).sum(1)/logc); cs=1-( -(ps*(ps.clamp_min(1e-8).log())).sum(1)/logc); w=F.softmax(torch.stack([cv,cs],1)/self.conf_temp,1); out=w[:,0:1]*lv+w[:,1:2]*ls; return (out,zv,zs) if return_aux else out


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_confidence_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; tl=DataLoader(h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'),16,shuffle=True); vl=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; h.train_one('confidence_routeA',ConfidenceMILRouteA(2),tl,vl,'cls',30,device,od); hist=json.loads((od/'confidence_routeA_history.json').read_text()); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'temperature':.5,'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
