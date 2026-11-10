"""Train Local Route A with modality dropout and evaluate missing-modality robustness."""
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h


@torch.no_grad()
def eval_mode(model, loader, device, mode):
    model.eval(); ys=[]; ps=[]
    for v,s,y in loader:
        if mode == 'visual_only': s = torch.zeros_like(s)
        elif mode == 'spectral_only': v = torch.zeros_like(v)
        p=model(v.to(device),s.to(device)); ys.append(y); ps.append(p.cpu())
    y=torch.cat(ys); p=torch.cat(ps); return float((p.argmax(1)==y).float().mean())


def train_dropout(p_drop, out, epochs=30):
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(.7*len(pats)); trp=set(pats[:cut]); vap=set(pats[cut:]); tr=[r for r in rows if r[0] in trp]; va=[r for r in rows if r[0] in vap]
    ds=h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'); dv=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'); tl=DataLoader(ds,16,shuffle=True); vl=DataLoader(dv,32); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=h.MILRouteA(2).to(device); h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4; opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best=-1; hist=[]
    for ep in range(1,epochs+1):
        model.train(); trloss=0.; n=0
        for v,s,y in tl:
            v,s,y=v.to(device),s.to(device),y.to(device); rv=torch.rand(len(y),device=device); rs=torch.rand(len(y),device=device); vm=(rv<p_drop)&(rs>=p_drop/2); sm=(rs<p_drop)&(rv>=p_drop/2); v=v.masked_fill(vm[:,None,None,None],0); s=s.masked_fill(sm[:,None,None],0)
            opt.zero_grad(set_to_none=True); p,zv,zs=model(v,s,True); loss=h.loss_fn('cls',p,y,zv,zs,model); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); trloss+=float(loss.item())*len(y); n+=len(y)
        sch.step(); full=eval_mode(model,vl,device,'full'); hist.append({'epoch':ep,'train_loss':trloss/n,'full':full});
        if full>best: best=full; torch.save({'model':model.state_dict(),'epoch':ep,'metrics':hist[-1]},out/'best.pt')
        if ep==1 or ep%5==0: print('drop',p_drop,ep,round(trloss/n,4),hist[-1],flush=True)
    ck=torch.load(out/'best.pt',map_location=device); model.load_state_dict(ck['model']); result={'p_drop':p_drop,'best_epoch':int(ck['epoch']),'full':eval_mode(model,vl,device,'full'),'visual_only':eval_mode(model,vl,device,'visual_only'),'spectral_only':eval_mode(model,vl,device,'spectral_only'),'train_patients':sorted(trp),'val_patients':sorted(vap)}; (out/'history.json').write_text(json.dumps(hist,indent=2)); (out/'result.json').write_text(json.dumps(result,indent=2)); return result


def main():
    h.seed_everything(42); out=Path(r'G:\Datasets\HyperCross\runs')/('hist_modality_dropout_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); result=train_dropout(.2,out); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
