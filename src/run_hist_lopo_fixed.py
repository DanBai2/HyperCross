"""Strict leave-one-patient-out evaluation with fixed-epoch training."""
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h
from evaluate_patient_cv_metrics import binary_metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for v,s,y in loader:
        ps.append(torch.softmax(model(v.to(device),s.to(device)),1)[:,1].cpu().numpy()); ys.append(y.numpy())
    prob=np.concatenate(ps); y=np.concatenate(ys); return y,(prob>=.5).astype(int),prob


def fit_fixed(model, loader, epochs, device):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.; n=0
        for v,s,y in loader:
            v,s,y=v.to(device),s.to(device),y.to(device); opt.zero_grad(set_to_none=True); p,zv,zs=model(v,s,True); loss=h.loss_fn('cls',p,y,zv,zs,model); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss.item())*len(y); n+=len(y)
        sch.step(); hist.append({'epoch':ep,'train_loss':total/n})
        if ep==1 or ep%5==0: print('epoch',ep,'loss',round(total/n,4),flush=True)
    return hist


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); out=Path(r'G:\Datasets\HyperCross\runs')/('hist_lopo_fixed_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); oof={n:{'y':[],'p':[],'prob':[]} for n in ['routeA','mil_concat']}; folds=[]
    for fi,patient in enumerate(patients):
        tr=[r for r in rows if r[0]!=patient]; va=[r for r in rows if r[0]==patient]; tl=DataLoader(h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'),16,shuffle=True); vl=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); fd=out/f'patient_{patient}'; fd.mkdir(); fold={'patient':patient,'n_val':len(va)}
        for name,cls in [('routeA',h.MILRouteA),('mil_concat',h.MILConcat)]:
            h.seed_everything(100+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; model=cls(2); hist=fit_fixed(model,tl,20,device); y,p,prob=predict(model,vl,device); fold[name]=binary_metrics(y,p,prob); (fd/f'{name}_history.json').write_text(json.dumps(hist,indent=2)); torch.save({'model':model.state_dict(),'epochs':20},fd/f'{name}_final.pt'); oof[name]['y'].append(y); oof[name]['p'].append(p); oof[name]['prob'].append(prob)
        folds.append(fold); print(json.dumps(fold),flush=True)
    result={'folds':folds}
    for name,a in oof.items(): result[name+'_aggregate']=binary_metrics(np.concatenate(a['y']),np.concatenate(a['p']),np.concatenate(a['prob']))
    (out/'results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
