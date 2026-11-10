from pathlib import Path
import numpy as np, concurrent.futures
from PIL import Image
ROOT=Path(r'G:\Datasets\PKG - HistologyHSI-GB'); OUT=ROOT/'cache_routeA_hist_mil'; OUT.mkdir(exist_ok=True)
def parse(h): return {x.split('=')[0].strip():int(x.split('=')[1].strip()) for x in h.splitlines() if x.split('=')[0].strip() in ['samples','lines','bands']}
items=[q for p in sorted(ROOT.glob('P*')) for q in sorted(p.glob('ROI_*'))]
def one(q):
    out=OUT/f'{q.parent.name}_{q.name}.npz'
    if out.exists(): return 'cached'
    h=parse((q/'raw.hdr').read_text(errors='ignore')); lines,samples,bands=h['lines'],h['samples'],h['bands']; bi=np.linspace(0,bands-1,128,dtype=int); yy=np.linspace(0,lines-1,16,dtype=int); xx=np.linspace(0,samples-1,32,dtype=int)
    raw=np.memmap(q/'raw',dtype=np.uint16,mode='r',shape=(lines,bands,samples)); x=np.asarray(raw[np.ix_(yy,bi,xx)],np.float32).transpose(0,2,1); x=x.reshape(4,4,4,8,128).mean((1,3)); s=x.reshape(16,128); s=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-6)
    v=np.asarray(Image.open(q/'rgb.png').convert('RGB').resize((64,64),Image.BILINEAR),np.float32).transpose(2,0,1)/255.; np.savez_compressed(out,v=v.astype('float32'),s=s.astype('float32')); return 'done'
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for i,r in enumerate(ex.map(one,items),1):
        if i%25==0: print(i,r,flush=True)
print('total',len(items),'files',len(list(OUT.glob('*.npz'))))
