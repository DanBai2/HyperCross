import torch
import torch.nn as nn
import torch.nn.functional as F
from hypercross_routeA import ConvEncoder, SpectralEncoder

class HybridSNLite(nn.Module):
    """Compact 2D/1D HybridSN-style spectral-spatial baseline."""
    def __init__(self,out_dim=3,d=128):
        super().__init__(); self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.GELU()); self.s=nn.Sequential(nn.Conv1d(1,32,7,2,3),nn.GELU(),nn.Conv1d(32,d,5,2,2),nn.GELU(),nn.AdaptiveAvgPool1d(1)); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='reg'
    def forward(self,v,s,return_aux=False):
        zv=self.v(v).mean((2,3)); zs=self.s(s[:,None]).squeeze(-1); o=self.head(torch.cat([zv,zs],1)); return (o,zv,zs) if return_aux else o

class S2MambaLite(nn.Module):
    """Mamba-inspired linear state-space baseline using bidirectional GRU scans."""
    def __init__(self,out_dim=3,d=128):
        super().__init__(); self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.GELU()); self.gru=nn.GRU(1,d//2,batch_first=True,bidirectional=True); self.sproj=nn.Linear(d,d); self.gate=nn.Sequential(nn.Linear(2*d,d),nn.Sigmoid()); self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='reg'
    def forward(self,v,s,return_aux=False):
        zv=self.v(v).mean((2,3)); zs=self.sproj(self.gru(s[:,:,None])[0].mean(1)); g=self.gate(torch.cat([zv,zs],1)); z=g*zv+(1-g)*zs; o=self.head(z); return (o,zv,zs) if return_aux else o

class MamTransLite(nn.Module):
    """Hybrid spectral Transformer plus spatial convolution with gated fusion."""
    def __init__(self,out_dim=3,d=96,heads=4):
        super().__init__(); self.v=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,d,3,2,1),nn.GELU()); self.patch=nn.Conv1d(1,d,8,8); self.pos=nn.Parameter(torch.zeros(1,16,d)); nn.init.normal_(self.pos,std=.02); layer=nn.TransformerEncoderLayer(d,heads,dim_feedforward=4*d,dropout=.1,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,2); self.sn=nn.LayerNorm(d); self.gate=nn.Sequential(nn.Linear(2*d,d),nn.Sigmoid()); self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,out_dim)); self.temp=nn.Parameter(torch.tensor(.07).log()); self.task='reg'
    def forward(self,v,s,return_aux=False):
        zv=self.v(v).mean((2,3)); zs=self.sn(self.enc(self.patch(s[:,None]).transpose(1,2)+self.pos).mean(1)); g=self.gate(torch.cat([zv,zs],1)); z=g*zv+(1-g)*zs; o=self.head(z); return (o,zv,zs) if return_aux else o
