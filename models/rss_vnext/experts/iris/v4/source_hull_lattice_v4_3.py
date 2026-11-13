from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SourceHullLatticePolicyV43:
    resolution: int = 512
    tile_points_per_step: int = 32768
    replay_points_per_step: int = 32768
    replay_bank_size: int = 1048576
    full_scan_interval_steps: int = 100
    pixel_quantization_guard_px: float = math.sqrt(2.0)
    top_level_weight: float = 1.0
    huber_beta_normalized: float = 0.01
    band_edges_px: tuple[float, float, float] = (2.0, 8.0, 32.0)

    def validate(self) -> None:
        if int(self.resolution) < 4:
            raise ValueError("resolution must be >= 4")
        for name in ("tile_points_per_step","replay_points_per_step","replay_bank_size","full_scan_interval_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("pixel_quantization_guard_px","top_level_weight","huber_beta_normalized"):
            value=float(getattr(self,name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        e=tuple(float(x) for x in self.band_edges_px)
        if len(e)!=3 or not (0.0 < e[0] < e[1] < e[2]):
            raise ValueError("band_edges_px must be increasing positive 3-tuple")


def lattice_point_count_v43(resolution: int) -> int:
    r=int(resolution)
    if r < 4:
        raise ValueError("resolution must be >= 4")
    return r*r*r


def lattice_points_from_linear_indices_v43(
    linear_indices: torch.Tensor,
    *,
    resolution: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map canonical dense-grid linear indices to exact MC XYZ query points."""
    r=int(resolution); total=lattice_point_count_v43(r)
    idx=torch.as_tensor(linear_indices,dtype=torch.int64,device=device).reshape(-1)
    if idx.numel()==0 or torch.any(idx<0) or torch.any(idx>=total):
        raise ValueError("invalid lattice indices")
    rr=r*r
    z=torch.div(idx,rr,rounding_mode="floor")
    rem=idx-z*rr
    y=torch.div(rem,r,rounding_mode="floor")
    x=rem-y*r
    scale=2.0/float(r-1)
    out=torch.stack(
        (-1.0+x.float()*scale,-1.0+y.float()*scale,-1.0+z.float()*scale),
        dim=-1,
    )
    return out.to(dtype=dtype)


def deterministic_lattice_tile_indices_v43(
    *,
    training_step: int,
    fit_seed: int,
    policy: SourceHullLatticePolicyV43 = SourceHullLatticePolicyV43(),
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Deterministic permutation tiles; R512/32768 covers all corners in 4096 steps."""
    policy.validate()
    step=int(training_step)
    if step<=0:
        raise ValueError("training_step must be positive")
    total=lattice_point_count_v43(policy.resolution)
    n=int(policy.tile_points_per_step)
    base=((step-1)*n+torch.arange(n,dtype=torch.int64,device=device)) % total
    multiplier=747796405  # odd => bijection modulo 2^27
    offset=(2891336453+int(fit_seed)*277803737) % total
    return (base*multiplier+offset) % total


def source_hull_exterior_margin_v43(
    points_normalized: torch.Tensor,
    *,
    distance_fields_px: torch.Tensor,
    camera_origins_normalized: torch.Tensor,
    camera_right: torch.Tensor,
    camera_screen_up: torch.Tensor,
    camera_forward: torch.Tensor,
    camera_half_extent_normalized: torch.Tensor,
    pixel_quantization_guard_px: float = math.sqrt(2.0),
) -> dict[str, torch.Tensor]:
    """Source-only exterior lower bound; inside/uncertified points are masked out.

    Semantics intentionally match source_exterior_v4 except the historical 0.04
    target cap is absent. Out-of-frame views provide no negative evidence.
    """
    q=torch.as_tensor(points_normalized)
    if q.ndim==3:
        if q.shape[0]!=1: raise ValueError("B=1 required")
        q=q[0]
    if q.ndim!=2 or q.shape[1]!=3 or q.numel()==0:
        raise ValueError("points_normalized must be [N,3] or [1,N,3]")
    dev=q.device
    df=torch.as_tensor(distance_fields_px,dtype=torch.float32,device=dev)
    origins=torch.as_tensor(camera_origins_normalized,dtype=torch.float32,device=dev)
    right=torch.as_tensor(camera_right,dtype=torch.float32,device=dev)
    up=torch.as_tensor(camera_screen_up,dtype=torch.float32,device=dev)
    forward=torch.as_tensor(camera_forward,dtype=torch.float32,device=dev)
    half=torch.as_tensor(camera_half_extent_normalized,dtype=torch.float32,device=dev).reshape(-1)
    if df.ndim!=3 or df.shape[0]!=8 or origins.shape!=(8,3) or right.shape!=(8,3) or up.shape!=(8,3) or forward.shape!=(8,3) or half.shape!=(8,):
        raise ValueError("invalid source/camera shapes")
    if torch.any(half<=0): raise ValueError("camera half extent must be positive")
    h,w=int(df.shape[1]),int(df.shape[2])
    best=torch.zeros(q.shape[0],dtype=torch.float32,device=dev)
    best_px=torch.zeros_like(best)
    witness=torch.full((q.shape[0],),-1,dtype=torch.int64,device=dev)
    q=q.float()
    for vi in range(8):
        f=F.normalize(forward[vi],dim=0); r=F.normalize(right[vi],dim=0); u=F.normalize(up[vi],dim=0)
        rel=q-origins[vi].view(1,3)
        depth=rel@f
        gx=(rel@r)/half[vi]; gy=-(rel@u)/half[vi]
        px=(gx+1.0)*0.5*float(w); py=(gy+1.0)*0.5*float(h)
        valid=(depth>0)&(px>=0)&(px<float(w))&(py>=0)&(py<float(h))
        ix=torch.floor(px).long().clamp(0,w-1); iy=torch.floor(py).long().clamp(0,h-1)
        raw=df[vi,iy,ix]
        conservative=torch.clamp(raw-float(pixel_quantization_guard_px),min=0.0)
        margin=conservative*(2.0*half[vi]/float(w))
        margin=torch.where(valid,margin,torch.zeros_like(margin))
        improve=margin>best
        best=torch.where(improve,margin,best)
        best_px=torch.where(improve,conservative,best_px)
        witness=torch.where(improve,torch.full_like(witness,vi),witness)
    return {
        "certified":best>0.0,
        "margin_normalized":best,
        "conservative_projected_distance_px":best_px,
        "witness_view":witness,
    }


def balanced_exterior_lower_bound_loss_v43(
    sdf: torch.Tensor,
    margin_normalized: torch.Tensor,
    conservative_distance_px: torch.Tensor,
    certified: torch.Tensor,
    *,
    policy: SourceHullLatticePolicyV43 = SourceHullLatticePolicyV43(),
) -> dict[str, torch.Tensor]:
    """Exterior-gated uncapped target; equal mean contribution from four distance bands."""
    policy.validate()
    s=torch.as_tensor(sdf).float().reshape(-1)
    m=torch.as_tensor(margin_normalized,device=s.device).float().reshape(-1)
    d=torch.as_tensor(conservative_distance_px,device=s.device).float().reshape(-1)
    c=torch.as_tensor(certified,device=s.device).bool().reshape(-1)
    if not (s.shape==m.shape==d.shape==c.shape) or s.numel()==0:
        raise ValueError("hull lower-bound tensors must match")
    if not torch.isfinite(s).all() or not torch.isfinite(m).all() or not torch.isfinite(d).all():
        raise ValueError("non-finite hull inputs")
    deficit=m-s
    violation=torch.relu(deficit)
    e=tuple(float(x) for x in policy.band_edges_px)
    masks=(c&(d<=e[0]),c&(d>e[0])&(d<=e[1]),c&(d>e[1])&(d<=e[2]),c&(d>e[2]))
    names=("0_2","2_8","8_32","gt_32")
    losses=[]; rows=[]
    for name,mask in zip(names,masks):
        count=int(mask.sum().detach().cpu())
        if count:
            v=violation[mask]
            loss=F.smooth_l1_loss(v,torch.zeros_like(v),beta=float(policy.huber_beta_normalized),reduction="mean")
            losses.append(loss)
            frac=(deficit[mask]>0).float().mean()
            mx=torch.amax(deficit[mask])
        else:
            loss=s.sum()*0.0; frac=loss; mx=loss
        rows.append({"band":name,"count":count,"loss":loss,"violating_fraction":frac,"maximum_metric_deficit":mx})
    total=torch.stack(losses).mean() if losses else s.sum()*0.0
    cc=c.sum()
    masked_deficit=deficit[c]
    return {
        "total":total,
        "certified_count":cc,
        "nonpositive_exterior_fraction":((s<=0)&c).float().sum()/cc.clamp_min(1),
        "metric_deficit_violating_fraction":((deficit>0)&c).float().sum()/cc.clamp_min(1),
        "maximum_metric_deficit":torch.amax(masked_deficit) if masked_deficit.numel() else s.sum()*0.0,
        "bands":tuple(rows),
    }


def select_worst_hull_deficit_indices_v43(
    linear_indices: np.ndarray,
    sdf: np.ndarray,
    margin_normalized: np.ndarray,
    certified: np.ndarray,
    *,
    bank_size: int,
) -> np.ndarray:
    idx=np.asarray(linear_indices,dtype=np.int64).reshape(-1)
    s=np.asarray(sdf,dtype=np.float64).reshape(-1)
    m=np.asarray(margin_normalized,dtype=np.float64).reshape(-1)
    c=np.asarray(certified,dtype=bool).reshape(-1)
    if not (idx.shape==s.shape==m.shape==c.shape) or idx.size==0:
        raise ValueError("replay arrays must match")
    deficit=m-s
    good=c&(deficit>0)
    if not np.any(good):
        return np.empty(0,dtype=np.int64)
    ii=idx[good]; dd=deficit[good]
    order=np.lexsort((ii,-dd))
    return np.asarray(ii[order[:min(int(bank_size),len(order))]],dtype=np.int64)
