"""Five-fold CV with GroupDRO-style dynamic patient loss reweighting."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import hypercross_routeA as h


class PatientDataset(h.HistologyMILDataset):
    def __init__(self,root,rows,cache_dir,pmap): super().__init__(root,rows,cache_dir); self.pmap=pmap
    def __getitem__(self,i):
        v,s,y=super().__getitem__(i); return v,s,y,torch.tensor(self.pmap[self.rows[i][0]],dtype=torch.long)


@torch.no_grad()
def eval_acc(model,loader,device):
    model.eval(); c=n=0
    for v,s,y,_ in loader:
        p=model(v.to(device),s.to(device)); c+=int((p.argmax(1).cpu()==y).sum()); n+=len(y)
    return c/n


def fit(model,loader,val,device,n_groups,epochs=30,eta=.1):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); q=torch.ones(n_groups,device=device); best=-1; hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.; n=0
        for v,s,y,g in loader:
            v,s,y,g=v.to(device),s.to(device),y.to(device),g.to(device); opt.zero_grad(set_to_none=True); p,zv,zs=model(v,s,True); ce=F.cross_entropy(p,y,reduction='none');
            with torch.no_grad():
                for gi in g.unique(): q[gi]*=torch.exp(eta*ce[g==gi].mean().detach()); q/=q.mean().clamp_min(1e-6)
            base=h.loss_fn('cls',p,y,zv,zs,model); extra=base-F.cross_entropy(p,y); loss=(q[g]*ce).mean()+extra; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss.item())*len(y); n+=len(y)
        sch.step(); acc=eval_acc(model,val,device); hist.append({'epoch':ep,'train_loss':total/n,'acc':acc});
        if acc>best: best=acc; torch.save({'model':model.state_dict(),'epoch':ep,'metrics':hist[-1]},model._out/'best.pt')
        if ep==1 or ep%5==0: print('groupdro',ep,round(total/n,4),acc,flush=True)
    return hist


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_groupdro_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); train_patients=sorted({r[0] for r in rows if r[0] not in vp}); pmap={p:i for i,p in enumerate(train_patients)}; tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; tl=DataLoader(PatientDataset(root,tr,Path(root)/'cache_routeA_hist_mil',pmap),16,shuffle=True); vl=DataLoader(PatientDataset(root,va,Path(root)/'cache_routeA_hist_mil',{p:i for i,p in enumerate(sorted(vp))}),32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; m=h.MILRouteA(2); m._out=od; hist=fit(m,tl,vl,device,len(train_patients)); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); (od/'history.json').write_text(json.dumps(hist,indent=2)); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'eta':.1,'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
