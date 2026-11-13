from __future__ import annotations

"""Deterministic parent-relative articulated probe bank for Arachne skin consequence.

The probe depends only on qualified skeleton geometry/tree and fixed world axes.
It uses no joint IDs, source bone names or serialized joint order, so consistent
joint permutation + tree remapping preserves the physical transform assignment.
"""
import math
import torch


def _skew(axis: torch.Tensor) -> torch.Tensor:
    x,y,z=axis.unbind(-1); O=torch.zeros_like(x)
    return torch.stack([O,-z,y,z,O,-x,-y,x,O],-1).reshape(*axis.shape[:-1],3,3)


def _rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis=axis/torch.linalg.norm(axis,dim=-1,keepdim=True).clamp_min(1e-8)
    K=_skew(axis); I=torch.eye(3,device=axis.device,dtype=axis.dtype).expand(*axis.shape[:-1],3,3)
    c=torch.cos(angle)[...,None,None]; s=torch.sin(angle)[...,None,None]
    aa=axis[..., :,None]*axis[...,None,:]
    return c*I+(1-c)*aa+s*K


def _T(v: torch.Tensor) -> torch.Tensor:
    out=torch.eye(4,device=v.device,dtype=v.dtype).expand(*v.shape[:-1],4,4).clone(); out[..., :3,3]=v; return out


def _R4(r: torch.Tensor) -> torch.Tensor:
    out=torch.eye(4,device=r.device,dtype=r.dtype).expand(*r.shape[:-2],4,4).clone(); out[..., :3,:3]=r; return out


def _depths(parent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B,J=parent.shape; out=torch.zeros((B,J),device=parent.device,dtype=torch.long)
    for b in range(B):
        for j in range(J):
            if not bool(mask[b,j]): continue
            seen=set(); cur=j; d=0
            while int(parent[b,cur])>=0:
                if cur in seen: raise ValueError("cycle in qualified tree")
                seen.add(cur); cur=int(parent[b,cur]); d+=1
                if d>J: raise ValueError("tree depth overflow")
            out[b,j]=d
    return out


def build_articulated_probe_transforms(joint_positions_world: torch.Tensor, parent_indices: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
    """Return [B,4,J,4,4] skinning matrices (posed_global @ inverse(bind_global))."""
    p=joint_positions_world.float(); parent=parent_indices.long(); mask=joint_mask.bool()
    if p.ndim!=3 or p.shape[-1]!=3 or parent.shape!=p.shape[:2] or mask.shape!=p.shape[:2]: raise ValueError("articulated probe shape drift")
    B,J,_=p.shape; depth=_depths(parent,mask)
    center=(p*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1).to(p.dtype)
    axis=torch.zeros_like(p); axis[...,2]=1.0
    for b in range(B):
        for j in range(J):
            if not bool(mask[b,j]): continue
            par=int(parent[b,j])
            if par<0: continue
            bone=p[b,j]-p[b,par]; bn=bone/torch.linalg.norm(bone).clamp_min(1e-8)
            ref=torch.tensor([0.,0.,1.],device=p.device,dtype=p.dtype)
            if abs(float(torch.dot(bn,ref)))>0.9: ref=torch.tensor([0.,1.,0.],device=p.device,dtype=p.dtype)
            a=torch.cross(bn,ref,dim=0); axis[b,j]=a/torch.linalg.norm(a).clamp_min(1e-8)
    deg=math.pi/180.0
    xsign=torch.where((p[...,0]-center[:,None,0])>=0,1.0,-1.0)
    zsign=torch.where((p[...,2]-center[:,None,2])>=0,1.0,-1.0)
    parity=torch.where((depth%2)==0,1.0,-1.0); dscale=1.0+0.08*depth.to(p.dtype)
    angles=torch.stack([torch.full_like(xsign,7.0*deg)*dscale,10.0*deg*xsign,-9.0*deg*zsign,8.0*deg*parity],dim=1)*mask[:,None].to(p.dtype)
    P=angles.shape[1]
    bind=torch.eye(4,device=p.device,dtype=p.dtype)[None,None].repeat(B,J,1,1)
    posed=torch.eye(4,device=p.device,dtype=p.dtype)[None,None,None].repeat(B,P,J,1,1)
    maxd=int(depth.max().item()) if bool(mask.any()) else 0
    for b in range(B):
        for d in range(maxd+1):
            rows=[j for j in range(J) if bool(mask[b,j]) and int(depth[b,j])==d]
            for j in rows:
                par=int(parent[b,j]); delta=p[b,j] if par<0 else p[b,j]-p[b,par]; local_bind=_T(delta)
                bind[b,j]=local_bind if par<0 else bind[b,par]@local_bind
                for pose in range(P):
                    local_pose=local_bind@_R4(_rotation(axis[b,j],angles[b,pose,j]))
                    posed[b,pose,j]=local_pose if par<0 else posed[b,pose,par]@local_pose
    skin=posed@torch.linalg.inv(bind)[:,None]
    skin=skin*mask[:,None,:,None,None].to(skin.dtype)+torch.eye(4,device=p.device,dtype=p.dtype)[None,None,None]*(~mask)[:,None,:,None,None].to(skin.dtype)
    return skin


__all__=["build_articulated_probe_transforms"]
