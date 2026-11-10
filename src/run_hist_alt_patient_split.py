"""Independent patient split check for the two leading Histology models."""
import json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h
from modern_hsi_models import MamTransLite


@torch.no_grad()
def probabilities(model, loader, device):
    model.eval(); ps=[]; ys=[]
    for v,s,y in loader:
        ps.append(torch.softmax(model(v.to(device),s.to(device)),1).cpu()); ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); val_patients={'P1','P4','P6','P9'}; tr=[r for r in rows if r[0] not in val_patients]; va=[r for r in rows if r[0] in val_patients]
    local_tr=h.HistologyMILDataset(root,tr,cache_dir=Path(root)/'cache_routeA_hist_mil'); local_va=h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'); global_tr=h.HistologyDataset(root,tr,cache_dir=Path(root)/'cache_routeA_v2'); global_va=h.HistologyDataset(root,va,cache_dir=Path(root)/'cache_routeA_v2')
    tl=DataLoader(local_tr,16,shuffle=True); vl=DataLoader(local_va,32); gt=DataLoader(global_tr,16,shuffle=True); gv=DataLoader(global_va,32); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=Path(r'G:\Datasets\HyperCross\runs')/('hist_alt_patient_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True)
    h.GLOBAL_ALIGN_W,h.GLOBAL_NCE_W,h.GLOBAL_BALANCE,h.GLOBAL_LR=.08,.04,False,2e-4
    h.seed_everything(42); rd=out/'routeA'; rd.mkdir(); h.train_one('routeA_alt',h.MILRouteA(2),tl,vl,'cls',30,device,rd)
    h.seed_everything(42); md=out/'mamtrans'; md.mkdir(); h.train_one('mamtrans_alt',MamTransLite(2),gt,gv,'cls',30,device,md)
    rm=h.MILRouteA(2).to(device); rm.load_state_dict(torch.load(rd/'routeA_alt_best.pt',map_location=device)['model']); mm=MamTransLite(2).to(device); mm.load_state_dict(torch.load(md/'mamtrans_alt_best.pt',map_location=device)['model']); rp,y=probabilities(rm,vl,device); mp,yg=probabilities(mm,gv,device); assert torch.equal(y,yg)
    acc=lambda p: float((p.argmax(1)==y).float().mean()); result={'val_patients':sorted(val_patients),'n_val':len(va),'routeA':acc(rp),'mamtrans':acc(mp),'equal_ensemble':acc((rp+mp)/2),'train_patients':sorted({r[0] for r in tr})}; (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
