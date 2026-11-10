from pathlib import Path
import numpy as np, tifffile, concurrent.futures
from PIL import Image
ROOT=Path(r'G:\Datasets\HyperLeaf2024'); OUT=ROOT/'cache_routeA_leaf_mil'; OUT.mkdir(exist_ok=True)
files=sorted((ROOT/'images').glob('*.tiff'))
def one(p):
    out=OUT/f'{p.stem}.npz'
    if out.exists(): return 'cached'
    a=tifffile.imread(p).astype('float32')/65535.; bi=np.linspace(0,a.shape[0]-1,128,dtype=int); yy=np.linspace(0,a.shape[1]-1,16,dtype=int); xx=np.linspace(0,a.shape[2]-1,32,dtype=int); x=a[np.ix_(bi,yy,xx)].transpose(1,2,0).reshape(4,4,4,8,128).mean((1,3)); s=x.reshape(16,128); s=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-6)
    rgb=np.stack([a[int(a.shape[0]*.72)],a[int(a.shape[0]*.48)],a[int(a.shape[0]*.20)]],0); rgb=(rgb-rgb.min())/(rgb.max()-rgb.min()+1e-6); v=np.asarray(Image.fromarray(np.transpose((rgb*255).astype('uint8'),(1,2,0))).resize((64,32),Image.BILINEAR),np.float32).transpose(2,0,1)/255.; np.savez_compressed(out,v=v.astype('float32'),s=s.astype('float32')); return 'done'
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for i,r in enumerate(ex.map(one,files),1):
        if i%200==0: print(i,r,flush=True)
print('total',len(files),'files',len(list(OUT.glob('*.npz'))))
