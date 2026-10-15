"""Independently written, explicitly adapted literature backbones for three tasks.

These are NOT official-repository checkpoints or full paper reproductions.
References and deviations are recorded in REFERENCES below. All inputs are
past-only windows and all encoders have separate 2/5/14-way supervised heads.
This research module is never imported by the production diagnosis runtime.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

REFERENCES = {
 'residual_tcn': {'origin':'existing exact AstroGraph shared residual TCN', 'adaptation':'unchanged architecture, retrained on common protocol'},
 'inception_time': {'url':'https://doi.org/10.1007/s10618-020-00710-y','origin':'existing local AstroGraph implementation', 'adaptation':'single network, not five-model ensemble; GELU and odd kernels 9/19/39; common three heads'},
 'timesnet_ref': {'url':'https://arxiv.org/abs/2210.02186','origin':'independent reference-oriented implementation','adaptation':'d_model32, d_ff64, 2 TimesBlocks, k3, 3 2D kernels; floor(T/frequency), convolutional token and fixed sinusoidal embeddings, flatten classification; no forecasting/unsupervised detection claim'},
 'moderntcn_ref': {'url':'https://openreview.net/forum?id=vpJMJerXHU','origin':'independent reference-oriented implementation','adaptation':'single stage, 2 blocks, per-variable width16, patch4/stride2, large13+small3 parallel depthwise, separate grouped feature and variable FFNs; no inference-time reparam or RevIN; three flattened heads'},
 'patchtst_ref': {'url':'https://arxiv.org/abs/2211.14730','origin':'independent classification adaptation','adaptation':'channel-independent shared patch encoder; patch4/stride2, width32, 4heads, 2layers, FF64, flatten-channel classification heads; train-only scaling, no RevIN; not a forecasting SOTA reproduction'},
 'hist_gradient_boosting': {'url':'https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting','origin':'installed sklearn implementation','adaptation':'three independent classifiers on same-window mean/std/last/last-first features; fixed160 iterations'},
}

class Heads(nn.Module):
    def __init__(self,encoder,width,components,classes):
        super().__init__();self.encoder=encoder
        self.detector=nn.Linear(width,2);self.localizer=nn.Linear(width,components);self.classifier=nn.Linear(width,classes)
    def forward(self,x):
        z=self.encoder(x);return self.detector(z),self.localizer(z),self.classifier(z)

def sinusoidal(length,width):
    p=torch.arange(length,dtype=torch.float32)[:,None]
    scale=torch.exp(torch.arange(0,width,2)*(-math.log(10000.)/width))
    out=torch.zeros(length,width);out[:,0::2]=torch.sin(p*scale);out[:,1::2]=torch.cos(p*scale)
    return out[None]

class Inception2D(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__();self.layers=nn.ModuleList([nn.Conv2d(c_in,c_out,k,padding=k//2) for k in (1,3,5)])
    def forward(self,x):return torch.stack([layer(x) for layer in self.layers],-1).mean(-1)

class PeriodBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.conv=nn.Sequential(Inception2D(width,width*2),nn.GELU(),Inception2D(width*2,width));self.norm=nn.LayerNorm(width)
    def forward(self,x):
        b,t,d=x.shape;spectrum=torch.fft.rfft(x,dim=1).abs()
        # Exclude DC even for an all-zero series. No division-by-zero period.
        freq=torch.topk(spectrum.mean((0,2))[1:],min(3,t//2)).indices+1
        weights=spectrum.mean(-1)[:,freq].softmax(-1);parts=[]
        for k in freq.tolist():
            period=max(1,t//k);length=math.ceil(t/period)*period
            z=F.pad(x,(0,0,0,length-t)).reshape(b,length//period,period,d).permute(0,3,1,2)
            z=self.conv(z).permute(0,2,3,1).reshape(b,length,d)[:,:t];parts.append(z)
        y=(torch.stack(parts,-1)*weights[:,None,None,:]).sum(-1)
        return self.norm(x+y)

class TimesEncoder(nn.Module):
    def __init__(self,c,t,d=32):
        super().__init__();self.token=nn.Conv1d(c,d,3,padding=1,padding_mode='circular',bias=False)
        self.register_buffer('position',sinusoidal(t,d));self.drop=nn.Dropout(.1)
        self.blocks=nn.Sequential(PeriodBlock(d),PeriodBlock(d));self.width=t*d
    def forward(self,x):
        z=self.drop(self.token(x.transpose(1,2)).transpose(1,2)+self.position)
        return self.drop(F.gelu(self.blocks(z))).flatten(1)

class ModernBlock(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.c=c;self.d=d
        self.large=nn.Sequential(nn.Conv1d(c*d,c*d,13,padding=6,groups=c*d,bias=False),nn.BatchNorm1d(c*d))
        self.small=nn.Sequential(nn.Conv1d(c*d,c*d,3,padding=1,groups=c*d,bias=False),nn.BatchNorm1d(c*d))
        self.norm=nn.BatchNorm1d(d)
        self.features=nn.Sequential(nn.Conv1d(c*d,c*d*2,1,groups=c),nn.GELU(),nn.Dropout(.1),nn.Conv1d(c*d*2,c*d,1,groups=c),nn.Dropout(.1))
        self.variables=nn.Sequential(nn.Conv1d(c*d,c*d*2,1,groups=d),nn.GELU(),nn.Dropout(.1),nn.Conv1d(c*d*2,c*d,1,groups=d),nn.Dropout(.1))
    def forward(self,x):
        b,c,d,n=x.shape;z=x.reshape(b,c*d,n)
        z=self.norm((self.large(z)+self.small(z)).reshape(b*c,d,n)).reshape(b,c*d,n)
        z=self.features(z).reshape(b,c,d,n).permute(0,2,1,3).reshape(b,d*c,n)
        z=self.variables(z).reshape(b,d,c,n).permute(0,2,1,3)
        return x+z

class ModernEncoder(nn.Module):
    def __init__(self,c,t,d=16):
        super().__init__();self.c=c;self.d=d;self.stem=nn.Sequential(nn.Conv1d(1,d,4,stride=2),nn.BatchNorm1d(d))
        self.blocks=nn.Sequential(ModernBlock(c,d),ModernBlock(c,d));self.drop=nn.Dropout(.1);self.width=c*d*(t//2)
    def forward(self,x):
        b,t,c=x.shape;z=F.pad(x.transpose(1,2).reshape(b*c,1,t),(0,2),mode='replicate')
        z=self.stem(z).reshape(b,c,self.d,-1)
        return self.drop(F.gelu(self.blocks(z))).flatten(1)

class PatchEncoder(nn.Module):
    def __init__(self,c,t,d=32):
        super().__init__();self.c=c;self.n=(t+2-4)//2+1;self.patch=nn.Linear(4,d,bias=False)
        self.register_buffer('position',sinusoidal(self.n,d))
        layer=nn.TransformerEncoderLayer(d,4,64,.1,activation='gelu',batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.norm=nn.LayerNorm(d)
        self.drop=nn.Dropout(.1);self.width=c*self.n*d
    def forward(self,x):
        b,t,c=x.shape;z=F.pad(x.transpose(1,2),(0,2),mode='replicate').unfold(-1,4,2)
        z=self.patch(z).reshape(b*c,self.n,-1)+self.position
        return self.drop(self.norm(self.encoder(z))).reshape(b,-1)

def make_reference_model(family,c,components,classes,t=20):
    if min(c,components,classes,t)<1:raise ValueError('positive dimensions required')
    if family=='residual_tcn':
        from astrograph.data_driven.three_task_tcn import make_shared_residual_tcn
        return make_shared_residual_tcn(c,components,classes)
    if family=='inception_time':
        from astrograph.data_driven.sota_backbones import make_sota_multitask_model
        return make_sota_multitask_model(family,c,components,classes,window_size=t)
    kind={'timesnet_ref':TimesEncoder,'moderntcn_ref':ModernEncoder,'patchtst_ref':PatchEncoder}
    if family not in kind:raise ValueError('unsupported benchmark family')
    enc=kind[family](c,t);return Heads(enc,enc.width,components,classes)
