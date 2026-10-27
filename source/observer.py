import torch
from torch import nn
import torch.nn.functional as F

class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_norm=nn.LayerNorm(1024)
        self.key=nn.Linear(1024,128)
        self.query=nn.Linear(768,128)
        self.attr=nn.Linear(2052,768)
        self.rel=nn.Linear(2052,768)
    def forward(self,patch,query,valid,coords):
        keys=self.key(self.patch_norm(patch))
        logits=torch.einsum('bsd,bnd->bsn',self.query(query),keys)/(128**.5)
        logits=logits+torch.log(valid.clamp_min(1e-12))[:,None,:]
        alpha=logits.softmax(-1)
        vector=pool_vector(patch,alpha,coords)
        return vector,alpha,F.normalize(self.attr(vector),dim=-1),F.normalize(self.rel(vector),dim=-1)

def pool_vector(patch,alpha,coords):
    slots=F.normalize(torch.einsum('bsn,bnd->bsd',alpha,patch),dim=-1)
    centers=torch.einsum('bsn,bnd->bsd',alpha,coords)
    return torch.cat([slots.reshape(len(patch),-1),(centers*.25).reshape(len(patch),-1)],dim=-1)

def loss_terms(output,batch):
    _,alpha,attr,rel=output
    masks=batch['teacher_masks'];den=masks.sum(-1);available=den>0
    target=masks/den.clamp_min(1e-12)[:,:,None]
    per=-(target*alpha.clamp_min(1e-12).log()).sum(-1)
    box=(per*available).sum()/available.sum().clamp_min(1)
    av=batch['attr_valid'];rv=batch['rel_valid']
    a=((1-(attr*batch['attr_target']).sum(-1))*av).sum()/av.sum().clamp_min(1)
    r=((1-(rel*batch['rel_target']).sum(-1))*rv).sum()/rv.sum().clamp_min(1)
    return box+.1*a+.1*r,{'box':box.detach(),'attr':a.detach(),'rel':r.detach()}
