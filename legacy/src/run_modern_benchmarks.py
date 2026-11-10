import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
import hypercross_routeA as h

def run_leaf(out):
    root=r'G:\Datasets\HyperLeaf2024'; df=pd.read_csv(Path(root)/'train.csv'); idx=np.random.default_rng(42).permutation(len(df)); cut=int(.8*len(df)); tr,va=idx[:cut],idx[cut:]; ds=h.HyperLeafDataset(root,indices=tr,cache_dir=Path(root)/'cache_routeA_leaf'); dv=h.HyperLeafDataset(root,indices=va,cache_dir=Path(root)/'cache_routeA_leaf'); tl=DataLoader(ds,batch_size=64,shuffle=True); vl=DataLoader(dv,batch_size=128); specs=[('spectral_transformer',h.SpectralTransformer(3)),('tiny_vit',h.TinyViT(3)),('film',h.FiLMFusion(3))]; res={}
    for name,m in specs: res[name]=h.train_one(name,m,tl,vl,'reg',30,torch.device('cuda'),out)
    return res
def run_hist(out):
    root=r'G:\Datasets\PKG - HistologyHSI-GB'; rows=h.build_hist_rows(root); pats=sorted({r[0] for r in rows}); rng=np.random.default_rng(42); rng.shuffle(pats); cut=int(.7*len(pats)); tr=[r for r in rows if r[0] in set(pats[:cut])]; va=[r for r in rows if r[0] in set(pats[cut:])]; ds=h.HistologyDataset(root,tr,cache_dir=Path(root)/'cache_routeA_v2'); dv=h.HistologyDataset(root,va,cache_dir=Path(root)/'cache_routeA_v2'); tl=DataLoader(ds,batch_size=16,shuffle=True); vl=DataLoader(dv,batch_size=32); specs=[('spectral_transformer',h.SpectralTransformer(2)),('tiny_vit',h.TinyViT(2)),('film',h.FiLMFusion(2))]; res={}
    for name,m in specs: res[name]=h.train_one(name,m,tl,vl,'cls',30,torch.device('cuda'),out)
    return res
if __name__=='__main__':
    h.seed_everything(42); out=Path(r'G:\Datasets\HyperCross\runs')/('modern_benchmarks_'+time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True,exist_ok=True); (out/'leaf').mkdir(); (out/'hist').mkdir(); result={'leaf':run_leaf(out/'leaf'),'hist':run_hist(out/'hist')}; (out/'modern_results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
