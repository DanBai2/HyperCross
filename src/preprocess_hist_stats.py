from pathlib import Path
import numpy as np, re, concurrent.futures, os, json
ROOT=Path(r'G:\Datasets\PKG - HistologyHSI-GB'); OUT=ROOT/'cache_routeA_hist_stats'; OUT.mkdir(exist_ok=True)
def parse(h):
    return {x.split('=')[0].strip():int(x.split('=')[1].strip()) for x in h.splitlines() if x.split('=')[0].strip() in ['samples','lines','bands']}
items=[q for p in sorted(ROOT.glob('P*')) for q in sorted(p.glob('ROI_*'))]
def one(q):
    out=OUT/f'{q.parent.name}_{q.name}.npz'
    if out.exists(): return 'cached'
    hdr=parse((q/'raw.hdr').read_text(errors='ignore')); lines,samples,bands=hdr['lines'],hdr['samples'],hdr['bands']; bi=np.linspace(0,bands-1,64,dtype=int); yy=np.linspace(0,lines-1,12,dtype=int); xx=np.linspace(0,samples-1,16,dtype=int)
    raw=np.memmap(q/'raw',dtype=np.uint16,mode='r',shape=(lines,bands,samples)); dark=np.memmap(q/'darkReference',dtype=np.uint16,mode='r',shape=(1,bands,samples)); white=np.memmap(q/'whiteReference',dtype=np.uint16,mode='r',shape=(1,bands,samples));
    dv=np.asarray(dark[0,bi,:],np.float32).mean(1); wv=np.asarray(white[0,bi,:],np.float32).mean(1); x=np.asarray(raw[np.ix_(yy,bi,xx)],np.float32); x=(x-dv[None,:,None])/(wv[None,:,None]-dv[None,:,None]+1e-3); x=np.clip(x,0,2)
    mean=x.mean((0,2)); std=x.std((0,2)); s=np.concatenate([mean,std]).astype('float32'); s=(s-s.mean())/(s.std()+1e-6)
    from PIL import Image
    v=np.asarray(Image.open(q/'rgb.png').convert('RGB').resize((64,64),Image.BILINEAR),np.float32).transpose(2,0,1)/255.0; np.savez_compressed(out,v=v.astype('float32'),s=s); return 'done'
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for i,r in enumerate(ex.map(one,items),1):
        if i%25==0: print(i,r,flush=True)
print('total',len(items),'files',len(list(OUT.glob('*.npz'))))
