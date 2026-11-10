import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import hypercross_routeA as h
from evaluate_patient_cv_metrics import binary_metrics, predict


def main():
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; run=Path(r'G:\Datasets\HyperCross\runs\hist_patient_cv_balanced_20261006_220855'); raw=json.loads((run/'results.json').read_text()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ys=[]; ps=[]; probs=[]; folds=[]
    for item in raw['folds']:
        va_patients=set(item['val_patients']); rows=h.build_hist_rows(root); va=[r for r in rows if r[0] in va_patients]; loader=DataLoader(h.HistologyMILDataset(root,va,cache_dir=Path(root)/'cache_routeA_hist_mil'),32); m=h.MILRouteA(2).to(device); m.load_state_dict(torch.load(run/f"fold{item['fold']}"/'routeA_balanced_best.pt',map_location=device)['model']); y,p,prob=predict(m,loader,device); folds.append({'fold':item['fold'],'val_patients':item['val_patients'],'metrics':binary_metrics(y,p,prob)}); ys.append(y); ps.append(p); probs.append(prob)
    y=np.concatenate(ys); p=np.concatenate(ps); prob=np.concatenate(probs); out={'folds':folds,'aggregate':binary_metrics(y,p,prob)}; (Path(r'G:\Datasets\HyperCross')/'balanced_cv_metrics.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
