"""Five-fold patient CV with gradient-reversal patient-domain adversarial training."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hypercross_routeA as h


class PatientDataset(h.HistologyMILDataset):
    def __init__(self, root, rows, cache_dir, patient_map, augment=False):
        super().__init__(root, rows, cache_dir, augment); self.patient_map=patient_map
    def __getitem__(self,i):
        v,s,y=super().__getitem__(i); return v,s,y,torch.tensor(self.patient_map[self.rows[i][0]],dtype=torch.long)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,alpha): ctx.alpha=alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx,grad): return -ctx.alpha*grad,None


class DomainAdvMILRouteA(h.MILRouteA):
    def __init__(self,out_dim=2,d=64,heads=4,n_domains=9,alpha=.1):
        super().__init__(out_dim,d,heads); self.domain=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,n_domains)); self.alpha=alpha
    def forward(self,v,s,return_aux=False,domain=False):
        vt=self.v(v).flatten(2).transpose(1,2); st=self.s(s); av=self.v2s(vt,st,st)[0]; ass=self.s2v(st,vt,vt)[0]; vf=self.vn(vt+av); sf=self.sn(st+ass); wa=F.softmax(self.vpool(vf).squeeze(-1),1); wb=F.softmax(self.spool(sf).squeeze(-1),1); zv=(wa.unsqueeze(-1)*vf).sum(1); zs=(wb.unsqueeze(-1)*sf).sum(1); out=self.head(torch.cat([zv,zs],1)); dom=self.domain(GRL.apply(torch.cat([zv,zs],1),self.alpha)); return (out,zv,zs,dom) if domain else ((out,zv,zs) if return_aux else out)


@torch.no_grad()
def eval_acc(model,loader,device):
    model.eval(); c=0;n=0
    for v,s,y,_ in loader:
        p=model(v.to(device),s.to(device)); c+=int((p.argmax(1).cpu()==y).sum()); n+=len(y)
    return c/n


def fit(model,tl,vl,device,epochs=30,lam=.1):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best=-1; hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.;n=0
        for v,s,y,pid in tl:
            v,s,y,pid=v.to(device),s.to(device),y.to(device),pid.to(device); opt.zero_grad(set_to_none=True); out,zv,zs,dom=model(v,s,domain=True); loss=h.loss_fn('cls',out,y,zv,zs,model)+lam*F.cross_entropy(dom,pid); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss.item())*len(y); n+=len(y)
        sch.step(); acc=eval_acc(model,vl,device); hist.append({'epoch':ep,'train_loss':total/n,'acc':acc});
        if acc>best: best=acc; torch.save({'model':model.state_dict(),'epoch':ep,'metrics':hist[-1]},model._out/'best.pt')
        if ep==1 or ep%5==0: print('domainadv',ep,round(total/n,4),acc,flush=True)
    return hist


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_domainadv_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); train_patients=sorted({r[0] for r in rows if r[0] not in vp}); pmap={p:i for i,p in enumerate(train_patients)}; tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; tl=DataLoader(PatientDataset(root,tr,Path(root)/'cache_routeA_hist_mil',pmap),16,shuffle=True); vl=DataLoader(PatientDataset(root,va,Path(root)/'cache_routeA_hist_mil',{p:i for i,p in enumerate(sorted(vp))}),32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; m=DomainAdvMILRouteA(2,n_domains=len(train_patients),alpha=.1); m._out=od; hist=fit(m,tl,vl,device); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); (od/'history.json').write_text(json.dumps(hist,indent=2)); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'lambda':.1,'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
