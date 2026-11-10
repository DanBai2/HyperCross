"""Five-fold CV for patient-style augmentation with prediction consistency."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hypercross_routeA as h


def style_view(v,s):
    b=len(v); gain=.8+.4*torch.rand(b,1,1,1,device=v.device); bias=(torch.rand(b,3,1,1,device=v.device)-.5)*.12; vv=(v*gain+bias).clamp(0,1); flip=torch.rand(b,device=v.device)<.5; vv[flip]=vv[flip].flip(-1); ss=s*(.95+.1*torch.rand(b,1,1,device=s.device))+torch.randn_like(s)*.02; drop=torch.rand(b,s.shape[1],1,device=s.device)<.05; ss=ss.masked_fill(drop,0); return vv,ss


def sym_kl(a,b):
    pa=F.softmax(a,1); pb=F.softmax(b,1); return .5*(F.kl_div(F.log_softmax(a,1),pb,reduction='batchmean')+F.kl_div(F.log_softmax(b,1),pa,reduction='batchmean'))


@torch.no_grad()
def eval_acc(model,loader,device):
    model.eval(); c=n=0
    for v,s,y in loader:
        p=model(v.to(device),s.to(device)); c+=int((p.argmax(1).cpu()==y).sum()); n+=len(y)
    return c/n


def fit(model,tl,vl,device,epochs=30,lam=.1):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best=-1; hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.; n=0
        for v,s,y in tl:
            v,s,y=v.to(device),s.to(device),y.to(device); v2,s2=style_view(v,s); opt.zero_grad(set_to_none=True); p,zv,zs=model(v,s,True); q,zv2,zs2=model(v2,s2,True); loss=.5*(h.loss_fn('cls',p,y,zv,zs,model)+h.loss_fn('cls',q,y,zv2,zs2,model))+lam*sym_kl(p,q); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss.item())*len(y); n+=len(y)
        sch.step(); acc=eval_acc(model,vl,device); hist.append({'epoch':ep,'train_loss':total/n,'acc':acc});
        if acc>best: best=acc; torch.save({'model':model.state_dict(),'epoch':ep,'metrics':hist[-1]},model._out/'best.pt')
        if ep==1 or ep%5==0: print('consistency',ep,round(total/n,4),acc,flush=True)
    return hist


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_consistency_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; tl=DataLoader(h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'),16,shuffle=True); vl=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; m=h.MILRouteA(2); m._out=od; hist=fit(m,tl,vl,device); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); (od/'history.json').write_text(json.dumps(hist,indent=2)); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'lambda':.1,'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
