from __future__ import annotations
import torch
from .geppetto_loss_v2 import GeppettoLossV2


def geppetto_train_step_v2(model,optimizer,conditioning,targets,*,loss_fn:GeppettoLossV2|None=None):
    loss_fn=loss_fn or GeppettoLossV2(support_topk=model.config.support_topk); f=torch.as_tensor(conditioning.features,device=next(model.parameters()).device,dtype=torch.float32); p=torch.as_tensor(conditioning.positions_normalized,device=f.device,dtype=torch.float32); m=torch.as_tensor(conditioning.valid_mask,device=f.device,dtype=torch.bool); max_j=max((len(t.positions_normalized) if t.valid else 0) for t in targets); steps=max(1,max_j+1); limit=int(m.sum(1).max().item()); steps=min(steps,limit); model.train(); optimizer.zero_grad(set_to_none=True); out=model(f,p,m,decode_steps=steps); losses=loss_fn(out,targets,p,m); losses["total"].backward(); optimizer.step(); return {k:float(v.detach().cpu()) for k,v in losses.items()}
