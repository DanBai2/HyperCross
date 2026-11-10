"""Five-fold CV for class-conditional cross-modal alignment."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hypercross_routeA as h


def supervised_crossmodal(zv, zs, y, temp=.1):
    z=F.normalize(torch.cat([zv,zs],0),dim=1); lab=torch.cat([y,y],0); sim=z@z.t()/temp; n=len(z); eye=torch.eye(n,dtype=torch.bool,device=z.device); sim=sim.masked_fill(eye,-1e4); pos=(lab[:,None]==lab[None,:])&(~eye); logp=sim-torch.logsumexp(sim,dim=1,keepdim=True); denom=pos.sum(1).clamp_min(1); return -(logp.masked_fill(~pos,0).sum(1)/denom).mean()


@torch.no_grad()
def eval_acc(model,loader,device):
    model.eval(); n=0; c=0
    for v,s,y in loader:
        p=model(v.to(device),s.to(device)); c+=int((p.argmax(1).cpu()==y).sum()); n+=len(y)
    return c/n


def fit(model,tl,vl,device,epochs=30,lam=.05):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best=-1; hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.; n=0
        for v,s,y in tl:
            v,s,y=v.to(device),s.to(device),y.to(device); opt.zero_grad(set_to_none=True); p,zv,zs=model(v,s,True); loss=h.loss_fn('cls',p,y,zv,zs,model)+lam*supervised_crossmodal(zv,zs,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss.item())*len(y); n+=len(y)
        sch.step(); acc=eval_acc(model,vl,device); hist.append({'epoch':ep,'train_loss':total/n,'acc':acc});
        if acc>best: best=acc; torch.save({'model':model.state_dict(),'epoch':ep,'metrics':hist[-1]},model._out/'best.pt')
        if ep==1 or ep%5==0: print('classalign',ep,round(total/n,4),acc,flush=True)
    return hist


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_classalign_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; tl=DataLoader(h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'),16,shuffle=True); vl=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; m=h.MILRouteA(2); m._out=od; hist=fit(m,tl,vl,device); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); (od/'history.json').write_text(json.dumps(hist,indent=2)); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'lambda':.05,'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
