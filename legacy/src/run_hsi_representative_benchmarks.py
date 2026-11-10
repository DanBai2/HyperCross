import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
import hypercross_routeA as h
from modern_hsi_models import HybridSNLite,S2MambaLite,MamTransLite

def main():
    h.seed_everything(42); out=Path(r'G:\Datasets\HyperCross\runs')/('hsi_representative_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); (out/'leaf').mkdir(); (out/'hist').mkdir(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); results={}
    root=r'G:\Datasets\HyperLeaf2024'; df=pd.read_csv(Path(root)/'train.csv'); idx=np.random.default_rng(42).permutation(len(df)); cut=int(.8*len(df)); tr,va=idx[:cut],idx[cut:]; ds=h.HyperLeafDataset(root,indices=tr,cache_dir=Path(root)/'cache_routeA_leaf'); dv=h.HyperLeafDataset(root,indices=va,cache_dir=Path(root)/'cache_routeA_leaf'); tl=DataLoader(ds,batch_size=64,shuffle=True); vl=DataLoader(dv,batch_size=128); results['leaf']={}
    for name,cls in [('HybridSN_lite',HybridSNLite),('S2Mamba_lite',S2MambaLite),('MamTrans_lite',MamTransLite)]: results['leaf'][name]=h.train_one(name,cls(3),tl,vl,'reg',30,device,out/'leaf')
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(.7*len(pats)); tr=[r for r in rows if r[0] in set(pats[:cut])]; va=[r for r in rows if r[0] in set(pats[cut:])]; ds=h.HistologyDataset(root,tr,cache_dir=Path(root)/'cache_routeA_v2'); dv=h.HistologyDataset(root,va,cache_dir=Path(root)/'cache_routeA_v2'); tl=DataLoader(ds,batch_size=16,shuffle=True); vl=DataLoader(dv,batch_size=32); results['hist']={}
    for name,cls in [('HybridSN_lite',HybridSNLite),('S2Mamba_lite',S2MambaLite),('MamTrans_lite',MamTransLite)]: results['hist'][name]=h.train_one(name,cls(2),tl,vl,'cls',30,device,out/'hist')
    (out/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
