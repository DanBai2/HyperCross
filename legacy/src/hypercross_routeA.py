import argparse, json, os, random, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED=42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=False; torch.backends.cudnn.benchmark=True

def zscore(x, axis=None):
    x=np.asarray(x, np.float32); m=x.mean(axis=axis, keepdims=True); s=x.std(axis=axis, keepdims=True)+1e-6; return (x-m)/s

class HyperLeafDataset(Dataset):
    def __init__(self, root, split='train', indices=None, cache_dir=None, augment=False):
        self.root=Path(root); self.df=pd.read_csv(self.root/'train.csv'); self.df=self.df.iloc[indices].reset_index(drop=True) if indices is not None else self.df
        self.split=split; self.augment=augment; self.cache_dir=Path(cache_dir) if cache_dir else self.root/'cache_routeA'; self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.targets=['GrainWeight','Gsw','PhiPS2']; self.y=self.df[self.targets].values.astype('float32'); self.mu=self.y.mean(0); self.sd=self.y.std(0)+1e-6
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; fid=f"{int(r.ImageId):05d}"; p=self.cache_dir/f'{fid}.npz'
        if p.exists(): d=np.load(p); v=d['v']; s=d['s']
        else:
            a=tifffile.imread(self.root/'images'/f'{fid}.tiff').astype('float32')/65535.0
            # (bands,H,W): visible RGB plus spatially pooled spectrum
            rgb=np.stack([a[int(a.shape[0]*.72)],a[int(a.shape[0]*.48)],a[int(a.shape[0]*.20)]],0)
            rgb=np.nan_to_num(rgb); rgb=(rgb-rgb.min())/(rgb.max()-rgb.min()+1e-6)
            v=np.asarray(Image.fromarray(np.transpose((rgb*255).astype('uint8'),(1,2,0))).resize((64,32),Image.BILINEAR)).transpose(2,0,1).astype('float32')/255.0
            s=a.reshape(a.shape[0],-1).mean(1); s=s.reshape(1,-1); s=np.asarray(Image.fromarray(s).resize((128,1),Image.BILINEAR)).reshape(-1); s=zscore(s)
            np.savez_compressed(p,v=v,s=s.astype('float32'))
        if self.augment:
            if random.random()<.5: v=v[:,:,::-1].copy()
            if random.random()<.5: v=v[:,::-1,:].copy()
            v=np.clip(v+np.random.normal(0,.015,v.shape).astype('float32'),0,1); s=(s+np.random.normal(0,.02,s.shape)).astype('float32')
        y=(self.y[i]-self.mu)/self.sd
        return torch.from_numpy(v),torch.from_numpy(s.astype('float32')),torch.from_numpy(y.astype('float32'))

class HyperLeafMILDataset(Dataset):
    def __init__(self, root, indices, cache_dir, mu=None, sd=None, augment=False):
        self.root=Path(root); self.df=pd.read_csv(self.root/'train.csv').iloc[indices].reset_index(drop=True); self.cache_dir=Path(cache_dir); self.augment=augment; self.targets=['GrainWeight','Gsw','PhiPS2']; self.y=self.df[self.targets].values.astype('float32'); self.mu=mu if mu is not None else self.y.mean(0); self.sd=sd if sd is not None else self.y.std(0)+1e-6
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        fid=f"{int(self.df.iloc[i].ImageId):05d}"; d=np.load(self.cache_dir/f'{fid}.npz'); v=d['v'].copy(); s=d['s'].copy()
        if self.augment:
            if random.random()<.5: v=v[:,:,::-1].copy()
            if random.random()<.5: v=v[:,::-1,:].copy()
            v=np.clip(v+np.random.normal(0,.015,v.shape).astype('float32'),0,1); s=(s+np.random.normal(0,.02,s.shape)).astype('float32')
        return torch.from_numpy(v),torch.from_numpy(s),torch.from_numpy(((self.y[i]-self.mu)/self.sd).astype('float32'))

class HistologyDataset(Dataset):
    def __init__(self, root, rows, cache_dir=None, augment=False):
        self.root=Path(root); self.rows=rows; self.augment=augment; self.cache_dir=Path(cache_dir) if cache_dir else self.root/'cache_routeA'; self.cache_dir.mkdir(parents=True,exist_ok=True)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        patient, roi, label=self.rows[i]; key=f'{patient}_{roi}'
        p=self.cache_dir/f'{key}.npz'
        if p.exists(): d=np.load(p); v=d['v']; s=d['s']
        else:
            q=self.root/patient/roi
            rgb=np.asarray(Image.open(q/'rgb.png').convert('RGB').resize((64,64),Image.BILINEAR),np.float32).transpose(2,0,1)/255.0
            hdr=(q/'raw.hdr').read_text(errors='ignore');
            import re
            lines=int(re.search(r'lines\s*=\s*(\d+)',hdr,re.I).group(1)); samples=int(re.search(r'samples\s*=\s*(\d+)',hdr,re.I).group(1)); bands=int(re.search(r'bands\s*=\s*(\d+)',hdr,re.I).group(1)); dtype=np.uint16 if 'data type = 12' in hdr.lower() else np.float32
            # ENVI BIL layout is [line, band, sample]. Read a sparse grid to avoid
            # materializing the ~1.3 GB cube for each ROI.
            raw=np.memmap(q/'raw',dtype=dtype,mode='r',shape=(lines,bands,samples))
            yy=np.linspace(0,lines-1,12,dtype=int); xx=np.linspace(0,samples-1,16,dtype=int); bi=np.linspace(0,bands-1,128,dtype=int)
            sm=np.asarray(raw[np.ix_(yy,bi,xx)],dtype=np.float32).mean((0,2)); sm=zscore(sm)
            v=rgb.astype('float32'); s=sm.astype('float32'); np.savez_compressed(p,v=v,s=s)
        if self.augment:
            if random.random()<.5: v=v[:,:,::-1].copy()
            if random.random()<.5: v=v[:,::-1,:].copy()
            v=np.clip(v+np.random.normal(0,.012,v.shape).astype('float32'),0,1); s=(s+np.random.normal(0,.015,s.shape)).astype('float32')
        return torch.from_numpy(v),torch.from_numpy(s),torch.tensor(label,dtype=torch.long)

class HistologyMILDataset(Dataset):
    def __init__(self, root, rows, cache_dir, augment=False):
        self.root=Path(root); self.rows=rows; self.cache_dir=Path(cache_dir); self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        patient,roi,label=self.rows[i]; d=np.load(self.cache_dir/f'{patient}_{roi}.npz'); v=d['v'].copy(); s=d['s'].copy()
        if self.augment:
            if random.random()<.5: v=v[:,:,::-1].copy()
            if random.random()<.5: v=v[:,::-1,:].copy()
            v=np.clip(v+np.random.normal(0,.012,v.shape).astype('float32'),0,1); s=(s+np.random.normal(0,.015,s.shape)).astype('float32')
        return torch.from_numpy(v),torch.from_numpy(s),torch.tensor(label,dtype=torch.long)

def build_hist_rows(root):
    rows=[]
    for p in sorted(Path(root).glob('P*')):
        for q in sorted(p.glob('ROI_*')):
            name=q.name; label=1 if name.endswith('_T') else 0
            rows.append((p.name,name,label))
    return rows

class ConvEncoder(nn.Module):
    def __init__(self,d=128,in_ch=3):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(in_ch,32,3,2,1),nn.BatchNorm2d(32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,d,3,2,1),nn.GELU()); self.proj=nn.Linear(d,d)
    def forward(self,x): return self.proj(self.net(x).mean((2,3)))

class GroupNormConvEncoder(nn.Module):
    """Visual encoder without running-statistics dependence across acquisitions."""
    def __init__(self,d=128,in_ch=3):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(in_ch,32,3,2,1),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,d,3,2,1),nn.GELU()); self.proj=nn.Linear(d,d)
    def forward(self,x): return self.proj(self.net(x).mean((2,3)))

class SpectralEncoder(nn.Module):
    def __init__(self,n=128,d=128):
        super().__init__(); self.conv=nn.Sequential(nn.Conv1d(1,32,7,2,3),nn.GELU(),nn.Conv1d(32,64,5,2,2),nn.GELU(),nn.Conv1d(64,d,5,2,2),nn.GELU()); self.pool=nn.AdaptiveAvgPool1d(16); self.proj=nn.Linear(d*16,d)
    def forward(self,x): return self.proj(self.pool(self.conv(x[:,None])).flatten(1))

class CrossModalRouteA(nn.Module):
    def __init__(self,task='reg',out_dim=3,d=128,heads=4):
        super().__init__(); self.ve=ConvEncoder(d); self.se=SpectralEncoder(128,d); self.vtok=nn.Linear(d,d); self.stok=nn.Linear(d,d); self.attn_vs=nn.MultiheadAttention(d,heads,batch_first=True); self.attn_sv=nn.MultiheadAttention(d,heads,batch_first=True); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,out_dim)); self.task=task; self.temp=nn.Parameter(torch.tensor(.07).log())
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); vt=self.vtok(zv).unsqueeze(1); st=self.stok(zs).unsqueeze(1)
        av,_=self.attn_vs(vt,st,st); ass,_=self.attn_sv(st,vt,vt); cv=self.norm(zv+av[:,0]); cs=self.norm(zs+ass[:,0]); out=self.head(torch.cat([cv,cs],1))
        if return_aux: return out,cv,cs
        return out

class DualVisualCrossModalRouteA(nn.Module):
    """Cross-modal Route A with visible-band RGB plus train-fitted PCA pseudo-RGB.

    The six channels are concatenated only at the first convolution.  This
    preserves the stable visible-band anchor while exposing a global spectral
    projection learned from the HSI cube; the spectral mean branch is kept
    unchanged for a controlled representation ablation.
    """
    def __init__(self,task='reg',out_dim=3,d=128,heads=4):
        super().__init__(); self.ve=ConvEncoder(d,in_ch=6); self.se=SpectralEncoder(128,d); self.vtok=nn.Linear(d,d); self.stok=nn.Linear(d,d); self.attn_vs=nn.MultiheadAttention(d,heads,batch_first=True); self.attn_sv=nn.MultiheadAttention(d,heads,batch_first=True); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,out_dim)); self.task=task; self.temp=nn.Parameter(torch.tensor(.07).log())
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); vt=zv[:,None]; st=zs[:,None]; cv=self.norm(zv+self.attn_vs(vt,st,st)[0][:,0]); cs=self.norm(zs+self.attn_sv(st,vt,vt)[0][:,0]); out=self.head(torch.cat([cv,cs],1)); return (out,cv,cs) if return_aux else out

class GroupNormCrossModalRouteA(nn.Module):
    """Route A with GroupNorm visual features for unseen acquisition domains."""
    def __init__(self,task='reg',out_dim=3,d=128,heads=4,in_ch=3):
        super().__init__(); self.ve=GroupNormConvEncoder(d,in_ch); self.se=SpectralEncoder(128,d); self.vtok=nn.Linear(d,d); self.stok=nn.Linear(d,d); self.attn_vs=nn.MultiheadAttention(d,heads,batch_first=True); self.attn_sv=nn.MultiheadAttention(d,heads,batch_first=True); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,out_dim)); self.task=task; self.temp=nn.Parameter(torch.tensor(.07).log())
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); vt=zv[:,None]; st=zs[:,None]; cv=self.norm(zv+self.attn_vs(vt,st,st)[0][:,0]); cs=self.norm(zs+self.attn_sv(st,vt,vt)[0][:,0]); out=self.head(torch.cat([cv,cs],1)); return (out,cv,cs) if return_aux else out

class MultiHeadCrossModal(nn.Module):
    """Shared multimodal encoder with target-specific regression heads.

    This isolates task-head negative transfer while keeping one shared visual
    and spectral representation, providing a parameter-efficient counterpart
    to training three completely independent target models.
    """
    def __init__(self,task='reg',out_dim=3,d=128,heads=4):
        super().__init__(); self.ve=ConvEncoder(d); self.se=SpectralEncoder(128,d); self.attn_vs=nn.MultiheadAttention(d,heads,batch_first=True); self.attn_sv=nn.MultiheadAttention(d,heads,batch_first=True); self.norm=nn.LayerNorm(d); self.heads=nn.ModuleList([nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,1)) for _ in range(out_dim)]); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); vt=zv[:,None]; st=zs[:,None]; cv=self.norm(zv+self.attn_vs(vt,st,st)[0][:,0]); cs=self.norm(zs+self.attn_sv(st,vt,vt)[0][:,0]); fused=torch.cat([cv,cs],1); out=torch.cat([head(fused) for head in self.heads],1); return (out,cv,cs) if return_aux else out

class VisualOnly(nn.Module):
    def __init__(self,out_dim,d=128): super().__init__(); self.e=ConvEncoder(d); self.h=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,out_dim))
    def forward(self,v,s=None,return_aux=False):
        z=self.e(v); o=self.h(z); return (o,z,z) if return_aux else o
class SpectralOnly(nn.Module):
    def __init__(self,out_dim,d=128): super().__init__(); self.e=SpectralEncoder(128,d); self.h=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,out_dim))
    def forward(self,v,s,return_aux=False):
        z=self.e(s); o=self.h(z); return (o,z,z) if return_aux else o
class ConcatModel(nn.Module):
    def __init__(self,out_dim,d=128): super().__init__(); self.v=ConvEncoder(d); self.s=SpectralEncoder(128,d); self.h=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,out_dim))
    def forward(self,v,s,return_aux=False):
        zv=self.v(v); zs=self.s(s); o=self.h(torch.cat([zv,zs],1)); return (o,zv,zs) if return_aux else o

class TokenVisualEncoder(nn.Module):
    def __init__(self,d=128):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),nn.Conv2d(64,d,3,2,1),nn.GELU()); self.proj=nn.Linear(d,d)
    def forward(self,x):
        t=self.net(x).flatten(2).transpose(1,2); return self.proj(t)
class TokenSpectralEncoder(nn.Module):
    def __init__(self,d=128):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,32,7,2,3),nn.GELU(),nn.Conv1d(32,64,5,2,2),nn.GELU(),nn.Conv1d(64,d,5,4,2),nn.GELU()); self.proj=nn.Linear(d,d)
    def forward(self,x): return self.proj(self.net(x[:,None]).transpose(1,2))
class TokenRouteA(nn.Module):
    """Route A with genuine patch-to-band token interaction and bidirectional fusion."""
    def __init__(self,task='reg',out_dim=3,d=128,heads=4):
        super().__init__(); self.ve=TokenVisualEncoder(d); self.se=TokenSpectralEncoder(d); self.v2s=nn.MultiheadAttention(d,heads,batch_first=True); self.s2v=nn.MultiheadAttention(d,heads,batch_first=True); self.vnorm=nn.LayerNorm(d); self.snorm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
    def forward(self,v,s,return_aux=False):
        vt=self.ve(v); st=self.se(s); av,_=self.v2s(vt,st,st); ass,_=self.s2v(st,vt,vt); vf=self.vnorm(vt+av).mean(1); sf=self.snorm(st+ass).mean(1); out=self.head(torch.cat([vf,sf],1));
        return (out,vf,sf) if return_aux else out

class CorrespondenceRouteA(nn.Module):
    """Token-level cross-modal fusion with positional and global pooling.

    The old global CrossModalRouteA feeds one visual token and one spectral
    token to MultiheadAttention, where attention is mathematically degenerate
    (the softmax is always one).  This variant keeps real patch/band tokens,
    adds positional information, and combines attention pooling with max
    pooling so the regression head retains both correspondence and global
    evidence.
    """
    def __init__(self,task='reg',out_dim=3,d=96,heads=4):
        super().__init__(); self.ve=TokenVisualEncoder(d); self.se=TokenSpectralEncoder(d)
        self.vpos=nn.Parameter(torch.zeros(1,64,d)); self.spos=nn.Parameter(torch.zeros(1,32,d)); nn.init.normal_(self.vpos,std=.02); nn.init.normal_(self.spos,std=.02)
        self.v2s=nn.MultiheadAttention(d,heads,batch_first=True); self.s2v=nn.MultiheadAttention(d,heads,batch_first=True)
        self.vn=nn.LayerNorm(d); self.sn=nn.LayerNorm(d); self.vpool=nn.Linear(d,1); self.spool=nn.Linear(d,1)
        self.head=nn.Sequential(nn.Linear(4*d,2*d),nn.GELU(),nn.Dropout(.2),nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
    def forward(self,v,s,return_aux=False):
        vt=self.ve(v); st=self.se(s); vt=vt+self.vpos[:,:vt.shape[1]]; st=st+self.spos[:,:st.shape[1]]
        vf=self.vn(vt+self.v2s(vt,st,st)[0]); sf=self.sn(st+self.s2v(st,vt,vt)[0])
        wa=F.softmax(self.vpool(vf).squeeze(-1),1); wb=F.softmax(self.spool(sf).squeeze(-1),1)
        vz=(wa.unsqueeze(-1)*vf).sum(1); sz=(wb.unsqueeze(-1)*sf).sum(1); fused=torch.cat([vz,sz,vz*sz,(vz-sz).abs()],1); out=self.head(fused)
        return (out,vz,sz) if return_aux else out

class StatsFusionRouteA(nn.Module):
    """Regression head that preserves low-order RGB and spectral statistics.

    This is a controlled test of the hypothesis that the convolutional branch
    was discarding the stable whole-leaf signal used by PLS/RF.  It is not
    presented as a spatial token model.
    """
    def __init__(self,task='reg',out_dim=3,d=128):
        super().__init__(); self.v=nn.Sequential(nn.Linear(6,64),nn.GELU(),nn.LayerNorm(64)); self.s=nn.Sequential(nn.Linear(128,d),nn.GELU(),nn.LayerNorm(d)); self.head=nn.Sequential(nn.Linear(64+d,192),nn.GELU(),nn.Dropout(.15),nn.Linear(192,d),nn.GELU(),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
    def forward(self,v,s,return_aux=False):
        rv=torch.cat([v.mean((2,3)),v.std((2,3))],1); zv=self.v(rv); zs=self.s(s); out=self.head(torch.cat([zv,zs],1)); return (out,zv,zs) if return_aux else out

class StatsResidualRouteA(nn.Module):
    """Global Route A augmented with an explicit low-order-statistics branch."""
    def __init__(self,task='reg',out_dim=3,d=128):
        super().__init__(); self.ve=ConvEncoder(d); self.se=SpectralEncoder(128,d); self.raw=nn.Sequential(nn.Linear(134,d),nn.GELU(),nn.LayerNorm(d)); self.head=nn.Sequential(nn.Linear(3*d,2*d),nn.GELU(),nn.Dropout(.2),nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); raw=self.raw(torch.cat([v.mean((2,3)),v.std((2,3)),s],1)); out=self.head(torch.cat([zv,zs,raw],1)); return (out,zv,zs) if return_aux else out

class HybridRouteA(nn.Module):
    """Cross-attention with a learned residual gate to preserve complementary modality evidence."""
    def __init__(self,task='reg',out_dim=3,d=128,heads=4):
        super().__init__(); self.ve=ConvEncoder(d); self.se=SpectralEncoder(128,d); self.v2s=nn.MultiheadAttention(d,heads,batch_first=True); self.s2v=nn.MultiheadAttention(d,heads,batch_first=True); self.norm=nn.LayerNorm(d); self.gate=nn.Sequential(nn.Linear(2*d,2*d),nn.Sigmoid()); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); vt=zv[:,None]; st=zs[:,None]; cv=self.norm(zv+self.v2s(vt,st,st)[0][:,0]); cs=self.norm(zs+self.s2v(st,vt,vt)[0][:,0]); raw=torch.cat([zv,zs],1); cross=torch.cat([cv,cs],1); fused=self.gate(raw)*cross+(1-self.gate(raw))*raw; out=self.head(fused)
        if return_aux: return out,cv,cs
        return out
class BilinearRouteA(nn.Module):
    def __init__(self,task='reg',out_dim=3,d=128):
        super().__init__(); self.ve=ConvEncoder(d); self.se=SpectralEncoder(128,d); self.head=nn.Sequential(nn.Linear(3*d,2*d),nn.GELU(),nn.Dropout(.2),nn.Linear(2*d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); out=self.head(torch.cat([zv,zs,zv*zs],1)); return (out,zv,zs) if return_aux else out
class ResidualRouteA(nn.Module):
    """Concat prediction plus a zero-initialized bidirectional cross-attention residual."""
    def __init__(self,task='reg',out_dim=3,d=128,heads=4):
        super().__init__(); self.ve=ConvEncoder(d); self.se=SpectralEncoder(128,d); self.v2s=nn.MultiheadAttention(d,heads,batch_first=True); self.s2v=nn.MultiheadAttention(d,heads,batch_first=True); self.norm=nn.LayerNorm(d); self.base=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,out_dim)); self.res=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task=task
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)
    def forward(self,v,s,return_aux=False):
        zv=self.ve(v); zs=self.se(s); vt=zv[:,None]; st=zs[:,None]; cv=self.norm(zv+self.v2s(vt,st,st)[0][:,0]); cs=self.norm(zs+self.s2v(st,vt,vt)[0][:,0]); raw=torch.cat([zv,zs],1); out=self.base(raw)+self.res(torch.cat([cv,cs],1)); return (out,cv,cs) if return_aux else out

class MILRouteA(nn.Module):
    """Local spatial-spectral MIL: patch tokens attend to local spectral tokens before pooling."""
    def __init__(self,out_dim=2,d=64,heads=4):
        super().__init__(); self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.GELU()); self.s=nn.Sequential(nn.Linear(128,d),nn.GELU(),nn.LayerNorm(d)); self.v2s=nn.MultiheadAttention(d,heads,batch_first=True); self.s2v=nn.MultiheadAttention(d,heads,batch_first=True); self.vn=nn.LayerNorm(d); self.sn=nn.LayerNorm(d); self.vpool=nn.Sequential(nn.Linear(d,1)); self.spool=nn.Sequential(nn.Linear(d,1)); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.3),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='cls'
    def forward(self,v,s,return_aux=False):
        vt=self.v(v).flatten(2).transpose(1,2); st=self.s(s); av=self.v2s(vt,st,st)[0]; ass=self.s2v(st,vt,vt)[0]; vf=self.vn(vt+av); sf=self.sn(st+ass); wa=F.softmax(self.vpool(vf).squeeze(-1),1); wb=F.softmax(self.spool(sf).squeeze(-1),1); zv=(wa.unsqueeze(-1)*vf).sum(1); zs=(wb.unsqueeze(-1)*sf).sum(1); out=self.head(torch.cat([zv,zs],1)); return (out,zv,zs) if return_aux else out

class MILConcat(nn.Module):
    def __init__(self,out_dim=2,d=64):
        super().__init__(); self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.GELU()); self.s=nn.Sequential(nn.Linear(128,d),nn.GELU(),nn.LayerNorm(d)); self.vpool=nn.Linear(d,1); self.spool=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.3),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='cls'
    def forward(self,v,s,return_aux=False):
        vt=self.v(v).flatten(2).transpose(1,2); st=self.s(s); wa=F.softmax(self.vpool(vt).squeeze(-1),1); wb=F.softmax(self.spool(st).squeeze(-1),1); zv=(wa.unsqueeze(-1)*vt).sum(1); zs=(wb.unsqueeze(-1)*st).sum(1); out=self.head(torch.cat([zv,zs],1)); return (out,zv,zs) if return_aux else out

class MILLocalRouteA(nn.Module):
    """Locally aligned 4x4 visual and spectral regions with bidirectional cross-attention."""
    def __init__(self,out_dim=2,d=64,heads=4):
        super().__init__(); self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d((4,4))); self.s=nn.Sequential(nn.Linear(128,d),nn.GELU(),nn.LayerNorm(d)); self.pos=nn.Parameter(torch.zeros(1,16,d)); nn.init.normal_(self.pos,std=.02); self.v2s=nn.MultiheadAttention(d,heads,batch_first=True); self.s2v=nn.MultiheadAttention(d,heads,batch_first=True); self.vn=nn.LayerNorm(d); self.sn=nn.LayerNorm(d); self.vpool=nn.Linear(d,1); self.spool=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.3),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='cls'
    def forward(self,v,s,return_aux=False):
        vt=self.v(v).flatten(2).transpose(1,2)+self.pos; st=self.s(s); av=self.v2s(vt,st,st)[0]; ass=self.s2v(st,vt,vt)[0]; vf=self.vn(vt+av); sf=self.sn(st+ass); wa=F.softmax(self.vpool(vf).squeeze(-1),1); wb=F.softmax(self.spool(sf).squeeze(-1),1); zv=(wa.unsqueeze(-1)*vf).sum(1); zs=(wb.unsqueeze(-1)*sf).sum(1); out=self.head(torch.cat([zv,zs],1)); return (out,zv,zs) if return_aux else out

class MILAlignedRouteA(nn.Module):
    """Local Route A with explicit spatial correspondence constraints.

    Visual and spectral tokens share a 4x4 ordering.  A local attention mask
    prevents arbitrary long-range token pairing, while the optional token NCE
    term aligns the same spatial bin across modalities before fusion.
    """
    def __init__(self,out_dim=2,d=64,heads=4,radius=1):
        super().__init__(); self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d((4,4))); self.s=nn.Sequential(nn.Linear(128,d),nn.GELU(),nn.LayerNorm(d)); self.pos=nn.Parameter(torch.zeros(1,16,d)); nn.init.normal_(self.pos,std=.02); self.v2s=nn.MultiheadAttention(d,heads,batch_first=True); self.s2v=nn.MultiheadAttention(d,heads,batch_first=True); self.vn=nn.LayerNorm(d); self.sn=nn.LayerNorm(d); self.vpool=nn.Linear(d,1); self.spool=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.3),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='cls'
        coords=torch.stack(torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij'),-1).reshape(16,2); dist=(coords[:,None]-coords[None,:]).abs().amax(-1); self.register_buffer('local_mask',torch.where(dist<=radius,torch.zeros_like(dist,dtype=torch.float32),torch.full_like(dist,float('-inf'),dtype=torch.float32)))
    def forward(self,v,s,return_aux=False):
        vt=self.v(v).flatten(2).transpose(1,2)+self.pos; st=self.s(s); av=self.v2s(vt,st,st,attn_mask=self.local_mask)[0]; ass=self.s2v(st,vt,vt,attn_mask=self.local_mask)[0]; vf=self.vn(vt+av); sf=self.sn(st+ass); wa=F.softmax(self.vpool(vf).squeeze(-1),1); wb=F.softmax(self.spool(sf).squeeze(-1),1); zv=(wa.unsqueeze(-1)*vf).sum(1); zs=(wb.unsqueeze(-1)*sf).sum(1); out=self.head(torch.cat([zv,zs],1)); return (out,zv,zs,vt,st) if return_aux else out

class SpectralTransformer(nn.Module):
    def __init__(self,out_dim=3,d=96,depth=2,heads=4):
        super().__init__(); self.patch=nn.Conv1d(1,d,8,8); self.pos=nn.Parameter(torch.zeros(1,16,d)); nn.init.normal_(self.pos,std=.02); layer=nn.TransformerEncoderLayer(d,heads,dim_feedforward=4*d,dropout=.1,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,depth); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,out_dim)); self.task='reg'
    def forward(self,v,s,return_aux=False):
        z=self.norm(self.enc(self.patch(s[:,None]).transpose(1,2)+self.pos).mean(1)); o=self.head(z); return (o,z,z) if return_aux else o

class TinyViT(nn.Module):
    def __init__(self,out_dim=3,d=96,depth=2,heads=4):
        super().__init__(); self.patch=nn.Conv2d(3,d,8,8); self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pos=nn.Parameter(torch.zeros(1,65,d)); nn.init.normal_(self.pos,std=.02); layer=nn.TransformerEncoderLayer(d,heads,dim_feedforward=4*d,dropout=.1,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,depth); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,out_dim)); self.task='reg'
    def forward(self,v,s,return_aux=False):
        q=self.patch(v); hh,ww=q.shape[-2:]; z=q.flatten(2).transpose(1,2); pos_patch=self.pos[:,1:].transpose(1,2).reshape(1,q.shape[1],8,8); pos_patch=F.interpolate(pos_patch,size=(hh,ww),mode='bicubic',align_corners=False).flatten(2).transpose(1,2); pos=torch.cat([self.pos[:,:1],pos_patch],1); z=torch.cat([self.cls.expand(len(v),-1,-1),z],1); z=self.norm(self.enc(z+pos)[:,0]); o=self.head(z); return (o,z,z) if return_aux else o

class FiLMFusion(nn.Module):
    def __init__(self,out_dim=3,d=128):
        super().__init__(); self.v=ConvEncoder(d); self.s=SpectralEncoder(128,d); self.film=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,2*d)); self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='reg'
    def forward(self,v,s,return_aux=False):
        zv=self.v(v); zs=self.s(s); g,b=self.film(zs).chunk(2,1); z=zv*(1+torch.tanh(g))+b; o=self.head(z); return (o,zv,zs) if return_aux else o

GLOBAL_ALIGN_W=0.08
GLOBAL_NCE_W=0.04
GLOBAL_TOKEN_NCE_W=0.0
GLOBAL_REG_WEIGHTS=None
GLOBAL_BALANCE=False
GLOBAL_LR=2e-4
def token_correspondence_nce(vt,st,temp):
    """Match visual and spectral tokens at the same spatial bin across samples.

    The MIL cache stores 16 spectral tokens in the same 4x4 spatial order as
    the visual tokens.  This loss uses other samples in the minibatch as
    negatives while preserving the within-sample spatial correspondence.
    """
    vt=F.normalize(vt,dim=-1); st=F.normalize(st,dim=-1)
    logits=torch.einsum('bld,cmd->blc',vt,st)/temp
    labels=torch.arange(vt.shape[0],device=vt.device)
    return 0.5*(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.repeat(vt.shape[1]))+
                F.cross_entropy(logits.transpose(0,2).reshape(-1,logits.shape[0]),labels.repeat(vt.shape[1])))

def loss_fn(task,pred,y,zv=None,zs=None,model=None,token_v=None,token_s=None):
    if task=='reg':
        err=(pred-y).pow(2)
        if GLOBAL_REG_WEIGHTS is not None:
            w=torch.as_tensor(GLOBAL_REG_WEIGHTS,device=pred.device,dtype=pred.dtype).view(1,-1)
            task_loss=(err*w).mean()
        else:
            task_loss=err.mean()
    else:
        weight=None
        if GLOBAL_BALANCE:
            cnt=torch.bincount(y,minlength=pred.shape[1]).float().clamp_min(1); weight=(cnt.sum()/(len(cnt)*cnt)).to(y.device)
        task_loss=F.cross_entropy(pred,y,weight=weight)
    if zv is None or (GLOBAL_ALIGN_W == 0 and GLOBAL_NCE_W == 0 and GLOBAL_TOKEN_NCE_W == 0): return task_loss
    # Route A: symmetric cross-modal alignment, smooth spectral embedding, and bounded contrastive term.
    zv=F.normalize(zv,dim=1); zs=F.normalize(zs,dim=1); align=(1-(zv*zs).sum(1)).mean(); temp=(model.temp.exp() if hasattr(model,'temp') else torch.tensor(.07,device=y.device)).clamp(.03,.5); logits=zv@zs.t()/temp; labels=torch.arange(len(y),device=y.device); nce=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.t(),labels))/2
    total=task_loss+GLOBAL_ALIGN_W*align+GLOBAL_NCE_W*nce
    if token_v is not None and token_s is not None and GLOBAL_TOKEN_NCE_W>0:
        total=total+GLOBAL_TOKEN_NCE_W*token_correspondence_nce(token_v,token_s,temp)
    return total

@torch.no_grad()
def evaluate(model,loader,task,device):
    model.eval(); ys=[]; ps=[]; total=0
    for v,s,y in loader:
        v,s,y=v.to(device),s.to(device),y.to(device); p=model(v,s); total += float(loss_fn(task,p,y).item())*len(y); ys.append(y.cpu()); ps.append(p.cpu())
    y=torch.cat(ys); p=torch.cat(ps); loss=total/len(y)
    if task=='reg':
        mse=((p-y)**2).mean(0); var=((y-y.mean(0))**2).mean(0)+1e-8; r2=1-mse/var; return {'loss':loss,'r2_mean':float(r2.mean()),'r2':r2.tolist(),'mae':float((p-y).abs().mean())}
    acc=float((p.argmax(1)==y).float().mean()); return {'loss':loss,'acc':acc,'n':len(y)}

def train_one(name,model,train_loader,val_loader,task,epochs,device,outdir):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=GLOBAL_LR,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best=-1e9; hist=[]
    for ep in range(1,epochs+1):
        model.train(); tr=0; n=0
        for v,s,y in train_loader:
            v,s,y=v.to(device),s.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            ret=model(v,s,True)
            if isinstance(ret,(tuple,list)) and len(ret)==5:
                p,zv,zs,token_v,token_s=ret
            else:
                p,zv,zs=ret; token_v=token_s=None
            loss=loss_fn(task,p,y,zv,zs,model,token_v,token_s); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); tr+=float(loss.item())*len(y); n+=len(y)
        sch.step(); met=evaluate(model,val_loader,task,device); score=met.get('r2_mean',met.get('acc',0)); hist.append({'epoch':ep,'train_loss':tr/n,**met});
        if score>best: best=score; torch.save({'model':model.state_dict(),'epoch':ep,'metrics':met},outdir/f'{name}_best.pt')
        if ep==1 or ep%5==0: print(name,ep,round(tr/n,4),met,flush=True)
    (outdir/f'{name}_history.json').write_text(json.dumps(hist,indent=2)); return hist[-1]

def make_leaf(args,outdir,device):
    df=pd.read_csv(Path(args.root)/'train.csv'); n=len(df); rng=np.random.default_rng(SEED); idx=rng.permutation(n); cut=int(n*.8); tr,va=idx[:cut],idx[cut:]
    leaf_cache=Path(args.leaf_cache); leaf_cache=leaf_cache if leaf_cache.is_absolute() else Path(args.root)/leaf_cache
    if args.leaf_mil_only:
        train=HyperLeafMILDataset(args.root,tr,cache_dir=Path(args.root)/'cache_routeA_leaf_mil',augment=args.augment); val=HyperLeafMILDataset(args.root,va,cache_dir=Path(args.root)/'cache_routeA_leaf_mil',mu=train.mu,sd=train.sd)
    else:
        train=HyperLeafDataset(args.root,indices=tr,cache_dir=leaf_cache,augment=args.augment); val=HyperLeafDataset(args.root,indices=va,cache_dir=leaf_cache,); val.mu,val.sd=train.mu,train.sd
    # share normalization across splits
    val.mu,val.sd=train.mu,train.sd
    tl=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=0,pin_memory=True); vl=DataLoader(val,batch_size=args.batch*2,num_workers=0,pin_memory=True)
    if args.leaf_mil_only:
        return train_one('routeA_leaf_mil',MILRouteA(3),tl,vl,'reg',args.epochs,device,outdir) or {'train':len(train),'val':len(val)}
    models=[('visual',VisualOnly(3)),('spectral',SpectralOnly(3)),('concat',ConcatModel(3)),('routeA',CrossModalRouteA('reg',3)),('routeA_token',TokenRouteA('reg',3)),('routeA_correspondence',CorrespondenceRouteA('reg',3)),('routeA_stats',StatsFusionRouteA('reg',3)),('routeA_stats_residual',StatsResidualRouteA('reg',3)),('routeA_hybrid',HybridRouteA('reg',3)),('routeA_bilinear',BilinearRouteA('reg',3)),('routeA_residual',ResidualRouteA('reg',3))];
    if args.crossmodal_only: models=[models[3]]
    if args.token_only: models=[models[4]]  # genuine token-to-token Route A
    if os.environ.get('HC_STATS_ONLY') == '1': models=[models[7]]
    elif args.correspondence_only: models=[models[5]]
    if args.route_only: models=[models[3]]  # backward-compatible alias for the true global CrossModalRouteA
    if args.hybrid_only: models=[models[-2]]
    if args.bilinear_only: models=[models[-1]]
    if args.concat_only: models=[models[2]]
    if args.residual_only: models=[models[-1]]
    for name,model in models: train_one(name,model,tl,vl,'reg',args.epochs,device,outdir)
    return {'train':len(train),'val':len(val),'targets':['GrainWeight','Gsw','PhiPS2']}

def make_hist(args,outdir,device):
    rows=build_hist_rows(args.root); patients=sorted({r[0] for r in rows}); rng=np.random.default_rng(SEED); rng.shuffle(patients); cut=int(len(patients)*.7); trp=set(patients[:cut]); vap=set(patients[cut:]); tr=[r for r in rows if r[0] in trp]; va=[r for r in rows if r[0] in vap]
    if args.mil_only or args.mil_local_only or args.mil_concat_only or args.mil_aligned_only:
        train=HistologyMILDataset(args.root,tr,cache_dir=Path(args.root)/'cache_routeA_hist_mil',augment=args.augment); val=HistologyMILDataset(args.root,va,cache_dir=Path(args.root)/'cache_routeA_hist_mil')
    else:
        train=HistologyDataset(args.root,tr,cache_dir=Path(args.root)/args.hist_cache,augment=args.augment); val=HistologyDataset(args.root,va,cache_dir=Path(args.root)/args.hist_cache)
    tl=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=0,pin_memory=True); vl=DataLoader(val,batch_size=args.batch*2,num_workers=0,pin_memory=True)
    if args.mil_only:
        return train_one('routeA_mil',MILRouteA(2),tl,vl,'cls',args.epochs,device,outdir) or {'train':len(train),'val':len(val),'train_patients':sorted(trp),'val_patients':sorted(vap)}
    if args.mil_local_only:
        return train_one('routeA_mil_local',MILLocalRouteA(2),tl,vl,'cls',args.epochs,device,outdir) or {'train':len(train),'val':len(val),'train_patients':sorted(trp),'val_patients':sorted(vap)}
    if args.mil_aligned_only:
        return train_one('routeA_mil_aligned',MILAlignedRouteA(2),tl,vl,'cls',args.epochs,device,outdir) or {'train':len(train),'val':len(val),'train_patients':sorted(trp),'val_patients':sorted(vap)}
    if args.mil_concat_only:
        return train_one('mil_concat',MILConcat(2),tl,vl,'cls',args.epochs,device,outdir) or {'train':len(train),'val':len(val),'train_patients':sorted(trp),'val_patients':sorted(vap)}
    models=[('visual',VisualOnly(2)),('spectral',SpectralOnly(2)),('concat',ConcatModel(2)),('routeA',CrossModalRouteA('cls',2)),('routeA_token',TokenRouteA('cls',2)),('routeA_correspondence',CorrespondenceRouteA('cls',2)),('routeA_stats',StatsFusionRouteA('cls',2)),('routeA_hybrid',HybridRouteA('cls',2)),('routeA_bilinear',BilinearRouteA('cls',2)),('routeA_residual',ResidualRouteA('cls',2))];
    if args.crossmodal_only: models=[models[3]]
    if args.token_only: models=[models[4]]  # genuine token-to-token Route A
    if args.correspondence_only: models=[models[5]]
    if args.route_only: models=[models[3]]  # backward-compatible alias for the true global CrossModalRouteA
    if args.hybrid_only: models=[models[-2]]
    if args.bilinear_only: models=[models[-1]]
    if args.concat_only: models=[models[2]]
    if args.residual_only: models=[models[-1]]
    for name,model in models: train_one(name,model,tl,vl,'cls',args.epochs,device,outdir)
    return {'train':len(train),'val':len(val),'train_patients':sorted(trp),'val_patients':sorted(vap),'class_counts':{str(k):sum(r[2]==k for r in rows) for k in [0,1]}}

def main():
    global GLOBAL_ALIGN_W, GLOBAL_NCE_W, GLOBAL_TOKEN_NCE_W, GLOBAL_REG_WEIGHTS, GLOBAL_BALANCE, GLOBAL_LR
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['leaf','hist'],required=True); ap.add_argument('--root',required=True); ap.add_argument('--outdir',default=r'G:\Datasets\HyperCross\runs'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--leaf_cache',default='cache_routeA_leaf'); ap.add_argument('--reg_weights',default='1,1,1'); ap.add_argument('--crossmodal_only',action='store_true'); ap.add_argument('--token_only',action='store_true'); ap.add_argument('--correspondence_only',action='store_true'); ap.add_argument('--route_only',action='store_true'); ap.add_argument('--hybrid_only',action='store_true'); ap.add_argument('--bilinear_only',action='store_true'); ap.add_argument('--concat_only',action='store_true'); ap.add_argument('--residual_only',action='store_true'); ap.add_argument('--mil_only',action='store_true'); ap.add_argument('--mil_local_only',action='store_true'); ap.add_argument('--mil_aligned_only',action='store_true'); ap.add_argument('--mil_concat_only',action='store_true'); ap.add_argument('--leaf_mil_only',action='store_true'); ap.add_argument('--align_w',type=float,default=.08); ap.add_argument('--nce_w',type=float,default=.04); ap.add_argument('--token_nce_w',type=float,default=0.0); ap.add_argument('--balanced',action='store_true'); ap.add_argument('--augment',action='store_true'); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--hist_cache',default='cache_routeA_v2'); args=ap.parse_args(); GLOBAL_ALIGN_W=args.align_w; GLOBAL_NCE_W=args.nce_w; GLOBAL_TOKEN_NCE_W=args.token_nce_w; GLOBAL_REG_WEIGHTS=[float(x) for x in args.reg_weights.split(',')] if args.dataset=='leaf' else None; GLOBAL_BALANCE=args.balanced; GLOBAL_LR=args.lr; seed_everything(args.seed); outdir=Path(args.outdir)/f'{args.dataset}_{time.strftime("%Y%m%d_%H%M%S")}'; outdir.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device,'align_w',GLOBAL_ALIGN_W,'nce_w',GLOBAL_NCE_W,'token_nce_w',GLOBAL_TOKEN_NCE_W,'reg_weights',GLOBAL_REG_WEIGHTS,'balanced',GLOBAL_BALANCE,'lr',GLOBAL_LR,'augment',args.augment,'leaf_cache',args.leaf_cache,'hist_cache',args.hist_cache,'correspondence_only',args.correspondence_only,'crossmodal_only',args.crossmodal_only,'mil_only',args.mil_only,'mil_local_only',args.mil_local_only,'mil_aligned_only',args.mil_aligned_only,'mil_concat_only',args.mil_concat_only,'leaf_mil_only',args.leaf_mil_only,flush=True)
    meta={'args':vars(args),'device':str(device)}; meta['split']=make_leaf(args,outdir,device) if args.dataset=='leaf' else make_hist(args,outdir,device); (outdir/'meta.json').write_text(json.dumps(meta,indent=2,default=str)); print('OUTDIR',outdir,flush=True)
if __name__=='__main__':
    if '--stats_only' in sys.argv:
        sys.argv.remove('--stats_only'); os.environ['HC_STATS_ONLY']='1'
    main()
