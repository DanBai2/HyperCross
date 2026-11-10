"""Patient-episodic prototype regularization for Local Route A."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hypercross_routeA as h


def stack_sample(ds, idx):
    xs=[ds[int(i)] for i in idx]; return torch.stack([x[0] for x in xs]),torch.stack([x[1] for x in xs]),torch.stack([x[2] for x in xs])


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); patients=sorted({r[0] for r in rows}); folds=[patients[i::5] for i in range(5)]; out=Path(r'G:\Datasets\HyperCross\runs')/('hist_patient_cv_protoepisodic_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result=[]
    for fi,val_list in enumerate(folds):
        vp=set(val_list); train_patients=sorted({r[0] for r in rows if r[0] not in vp}); tr=[r for r in rows if r[0] not in vp]; va=[r for r in rows if r[0] in vp]; all_ds=h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'); val_ds=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'); patient_ds={p:h.HistologyMILDataset(root,[r for r in tr if r[0]==p],cache_dir=Path(root)/'cache_routeA_hist_mil') for p in train_patients}; vl=DataLoader(val_ds,32); od=out/f'fold{fi}'; od.mkdir(); h.seed_everything(42+fi); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; m=h.MILRouteA(2).to(device); opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30); best=-1; hist=[]; rng=np.random.default_rng(42+fi)
        for ep in range(1,31):
            m.train(); total=0.
            for _ in range(4):
                qp=str(rng.choice(train_patients)); qds=patient_ds[qp]; qidx=rng.choice(len(qds),min(16,len(qds)),replace=len(qds)<16); support_idx=rng.choice(len(all_ds),min(48,len(all_ds)),replace=False); vq,sq,yq=stack_sample(qds,qidx); vs,ss,ys=stack_sample(all_ds,support_idx); vq,sq,yq,vs,ss,ys=vq.to(device),sq.to(device),yq.to(device),vs.to(device),ss.to(device),ys.to(device); opt.zero_grad(set_to_none=True); pq,zvq,zsq=m(vq,sq,True); ps,zvs,zss=m(vs,ss,True); base=h.loss_fn('cls',pq,yq,zvq,zsq,m); zsup=torch.cat([zvs,zss],1); zq=torch.cat([zvq,zsq],1); protos=torch.stack([zsup[ys==c].mean(0) if (ys==c).any() else zsup.mean(0) for c in [0,1]]); proto_logits=-torch.cdist(zq,protos).pow(2)/.1; loss=base+.2*F.cross_entropy(proto_logits,yq); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.); opt.step(); total+=float(loss.item())
            sch.step(); met=h.evaluate(m,vl,'cls',device); hist.append({'epoch':ep,'train_loss':total/4,**met}); score=met['acc'];
            if score>best: best=score; torch.save({'model':m.state_dict(),'epoch':ep,'metrics':met},od/'best.pt')
            if ep==1 or ep%5==0: print('protoepisodic',ep,round(total/4,4),met,flush=True)
        (od/'history.json').write_text(json.dumps(hist,indent=2)); result.append({'fold':fi,'val_patients':val_list,'n_val':len(va),'best':max(hist,key=lambda x:x['acc'])}); print(json.dumps(result[-1]),flush=True)
    x=np.array([r['best']['acc'] for r in result]); summary={'prototype_weight':.2,'mean_accuracy':float(x.mean()),'std_accuracy':float(x.std()),'folds':result}; (out/'results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
