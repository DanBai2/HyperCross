"""Evaluate saved 5-fold patient CV checkpoints with medical classification metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h


def binary_metrics(y, p, prob):
    y=np.asarray(y); p=np.asarray(p); prob=np.asarray(prob); tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum()); tp=int(((y==1)&(p==1)).sum())
    sens=tp/(tp+fn) if tp+fn else float('nan'); spec=tn/(tn+fp) if tn+fp else float('nan'); bal=np.nanmean([sens,spec]); acc=float((y==p).mean());
    pos=prob[y==1]; neg=prob[y==0]; auc=float(((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else float('nan')
    ece=0.; conf=np.maximum(prob,1-prob); 
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        m=(conf>=lo)&(conf<hi if hi<1 else conf<=hi)
        if m.any(): ece += float(m.mean())*abs(float((p[m]==y[m]).mean())-float(conf[m].mean()))
    return {'accuracy':acc,'balanced_accuracy':float(bal),'sensitivity':float(sens),'specificity':float(spec),'auroc':auc,'ece_10bin':ece,'confusion_matrix':[[tn,fp],[fn,tp]],'n':int(len(y))}


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys=[]; probs=[]
    for v,s,y in loader:
        probs.append(torch.softmax(model(v.to(device),s.to(device)),1)[:,1].cpu().numpy()); ys.append(y.numpy())
    prob=np.concatenate(probs); y=np.concatenate(ys); return y,(prob>=.5).astype(int),prob


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; run=Path(r'G:\Datasets\HyperCross\runs\hist_patient_cv_20261006_215300'); raw=json.loads((run/'results.json').read_text()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out={'folds':[]}; aggregate={'routeA':{'y':[],'p':[],'prob':[]},'mil_concat':{'y':[],'p':[],'prob':[]}}
    for item in raw['folds']:
        fold=item['fold']; val_patients=set(item['val_patients']); rows=h.build_hist_rows(root); va=[r for r in rows if r[0] in val_patients]; loader=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); row={'fold':fold,'val_patients':item['val_patients'],'n_val':len(va)}
        for name,cls in [('routeA',h.MILRouteA),('mil_concat',h.MILConcat)]:
            model=cls(2).to(device); model.load_state_dict(torch.load(run/f'fold{fold}'/f'{name}_best.pt',map_location=device)['model']); y,p,prob=predict(model,loader,device); row[name]=binary_metrics(y,p,prob); aggregate[name]['y'].append(y); aggregate[name]['p'].append(p); aggregate[name]['prob'].append(prob)
        out['folds'].append(row)
    for name,a in aggregate.items():
        yy=np.concatenate(a['y']); pp=np.concatenate(a['prob']); out[name+'_aggregate']=binary_metrics(yy,(pp>=.5).astype(int),pp)
        sweep=[]
        for t in np.linspace(.1,.9,17): sweep.append({'threshold':float(t),**binary_metrics(yy,(pp>=t).astype(int),pp)})
        out[name+'_threshold_sweep']=sweep
    path=Path(r'G:\Datasets\HyperCross\patient_cv_metrics.json'); path.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
