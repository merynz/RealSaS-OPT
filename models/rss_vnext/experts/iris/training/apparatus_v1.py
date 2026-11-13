"""Subject-free V5 query pools and F*-certified source-silhouette apparatus."""
from __future__ import annotations
import time
import numpy as np
import torch
from models.iris.v5.source_signed_field_v5 import source_signed_field_numpy_v5, source_signed_field_torch_v5, SourceSignedFieldPolicyV5
from models.iris.v5.objective_tail_repair_v5 import orthographic_source_rays_normalized_v5, stratified_ray_samples_v5, soft_ray_minimum_v5

STRATA = ("FG_BOUNDARY", "BG_BOUNDARY", "FG_GLOBAL", "BG_SUPPORT")
KNIGHT_POOL_MAX_DRAW = 32_000_000
FIT8_POOL_MAX_DRAW_DEFAULT = 536_870_912
POOL_PROGRESS_DRAW_DEFAULT = 8_388_608

def targets(subject, points, device="cpu", torch_authority=False):
    kwargs = {k: subject[k] for k in ("signed_2d", "center_xyz", "normalization_half_extent", "camera_origins", "camera_right", "camera_screen_up", "camera_half_extent")}
    policy = SourceSignedFieldPolicyV5(required_resolution=subject["resolution"])
    if str(device).startswith("cuda") or torch_authority:
        kwargs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
        with torch.no_grad():
            out = [source_signed_field_torch_v5(torch.from_numpy(q).to(device), policy=policy, **kwargs).cpu().numpy()
                   for q in np.array_split(points,max(1,(len(points)+65535)//65536))]
    else:
        kwargs = {k:v.cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
        out = [source_signed_field_numpy_v5(q,policy=policy,**kwargs)
               for q in np.array_split(points,max(1,(len(points)+65535)//65536))]
    return np.concatenate(out).astype(np.float32)

def _pool_progress(kind, label, accepted, target, drawn, started):
    rate = float(accepted) / max(int(drawn), 1)
    projected = None if accepted <= 0 else int(np.ceil(float(drawn) * float(target) / float(accepted)))
    print(
        f"[POOL] kind={kind} label={label} accepted={accepted}/{target} drawn={drawn} "
        f"acceptance={rate:.8f} projected_draws={projected} seconds={time.monotonic()-started:.1f}",
        flush=True,
    )

def near_pool(subject, count, rng, band, device="cpu", *, max_draw=KNIGHT_POOL_MAX_DRAW,
              progress_every=0, label="near_zero"):
    parts, accepted, drawn = [], 0, 0
    started=time.monotonic(); next_progress=int(progress_every) if progress_every else None
    while accepted < count:
        q = rng.uniform(-1.,1.,(131072,3)).astype(np.float32)
        f = targets(subject,q,device)
        chosen = q[np.abs(f)<=band][:count-accepted]
        if len(chosen):
            parts.append(chosen);accepted += len(chosen)
        drawn += len(q)
        if next_progress is not None and drawn >= next_progress:
            _pool_progress("near_zero",label,accepted,count,drawn,started)
            while next_progress <= drawn:
                next_progress += int(progress_every)
        if drawn>int(max_draw):
            _pool_progress("near_zero_exhausted",label,accepted,count,drawn,started)
            raise RuntimeError(
                f"APPARATUS_NEAR_ZERO_POOL_EXHAUSTED:{label}:accepted={accepted}:target={count}:drawn={drawn}:max_draw={int(max_draw)}"
            )
    return np.concatenate(parts)[:count].astype(np.float32,copy=False)

def straddle_pool(subject, count, rng, delta, base_band, device="cpu", *,
                  max_draw=KNIGHT_POOL_MAX_DRAW, progress_every=0, label="straddle"):
    if count % 2:
        raise ValueError("STRADDLE_ENDPOINT_COUNT_MUST_BE_EVEN")
    parts, accepted, drawn = [], 0, 0
    started=time.monotonic(); next_progress=int(progress_every) if progress_every else None
    while accepted < count:
        margin = delta*1.05
        q = rng.uniform(-1.+margin,1.-margin,(131072,3)).astype(np.float32)
        f = targets(subject,q,device);q = q[np.abs(f)<=base_band]
        d = rng.normal(size=q.shape).astype(np.float32)
        d /= np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
        qm,qp = q-delta*d,q+delta*d
        if len(q):
            fm,fp = targets(subject,qm,device),targets(subject,qp,device)
            good = np.flatnonzero((fm>=0)!=(fp>=0))[:(count-accepted)//2]
            pair = np.stack([qm[good],qp[good]],axis=1).reshape(-1,3)
            if len(pair):
                parts.append(pair);accepted += len(pair)
        drawn += 131072
        if next_progress is not None and drawn >= next_progress:
            _pool_progress("straddle",label,accepted,count,drawn,started)
            while next_progress <= drawn:
                next_progress += int(progress_every)
        if drawn>int(max_draw):
            _pool_progress("straddle_exhausted",label,accepted,count,drawn,started)
            raise RuntimeError(
                f"APPARATUS_STRADDLE_POOL_EXHAUSTED:{label}:accepted={accepted}:target={count}:drawn={drawn}:max_draw={int(max_draw)}"
            )
    return np.concatenate(parts)[:count].astype(np.float32,copy=False)

def row_keys(points):
    p=np.ascontiguousarray(points,dtype=np.float32)
    return p.view(np.dtype((np.void,12))).reshape(-1)

def query_pools(subject, policy, seed, device="cpu"):
    """Independent train/replay/heldout streams; heldout never enters replay."""
    rng=lambda offset:np.random.default_rng(seed+offset)
    sid=str(subject.get("subject_id","subject"))
    max_draw=int(policy.get("pool_rejection_max_draw",KNIGHT_POOL_MAX_DRAW))
    progress=int(policy.get("pool_progress_draw_interval",0))
    near=near_pool(subject,policy["near_pool_count"],rng(101),policy["near_zero_band"],device,
                   max_draw=max_draw,progress_every=progress,label=f"{sid}:train_near")
    straddle=straddle_pool(subject,policy["straddle_pool_count"],rng(102),policy["straddle_delta"],policy["straddle_base_band"],device,
                           max_draw=max_draw,progress_every=progress,label=f"{sid}:train_straddle")
    replay_rng=rng(103)
    replay=np.concatenate([replay_rng.uniform(-1,1,(policy["replay_global_count"],3)).astype(np.float32),
                          near_pool(subject,policy["replay_near_count"],replay_rng,policy["near_zero_band"],device,
                                    max_draw=max_draw,progress_every=progress,label=f"{sid}:replay_near")])
    heldout=np.concatenate([rng(201).uniform(-1,1,(policy["heldout_count"],3)).astype(np.float32),
                           near_pool(subject,policy["heldout_near_count"],rng(202),policy["near_zero_band"],device,
                                     max_draw=max_draw,progress_every=progress,label=f"{sid}:heldout_near"),
                           straddle_pool(subject,policy["heldout_straddle_count"],rng(203),policy["straddle_delta"],policy["straddle_base_band"],device,
                                         max_draw=max_draw,progress_every=progress,label=f"{sid}:heldout_straddle")])
    if len(np.intersect1d(row_keys(np.concatenate([near,straddle,replay])),row_keys(heldout))):
        raise ValueError("APPARATUS_HELDOUT_TRAIN_OR_REPLAY_LEAKAGE")
    result={"near_points":torch.from_numpy(near),"straddle_points":torch.from_numpy(straddle),
            "replay_points":torch.from_numpy(replay),"replay_targets":torch.from_numpy(targets(subject,replay,device)),
            "heldout_points":torch.from_numpy(heldout),"heldout_targets":torch.from_numpy(targets(subject,heldout,device))}
    return result

def ray_queries(subject, xy, views, phase, depth_samples, device="cpu"):
    xy,views,phase=xy.to(device),views.to(device),phase.to(device)
    kwargs={k:subject[k] for k in ("center_xyz","normalization_half_extent","camera_origins","camera_right","camera_screen_up","camera_forward","camera_half_extent")}
    kwargs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
    base,direction,enter,exit=orthographic_source_rays_normalized_v5(xy,views,resolution=subject["resolution"],**kwargs)
    return stratified_ray_samples_v5(base,direction,enter,exit,phase,depth_samples=depth_samples)

def certified_ray_pools(subject, policy, seed, device="cpu"):
    """Source masks label rays; F* only certifies the frozen finite-depth renderer."""
    start=time.monotonic();signed=subject["signed_2d"].cpu().numpy();count=policy["silhouette_pool_per_stratum"]
    xy_parts,view_parts,phase_parts,fg_parts,strata_parts,receipt=[],[],[],[],[],[]
    for view in range(8):
        inside=-signed[view];outside=signed[view];mask=signed[view]<0
        bands=[mask & (inside<=policy["source_boundary_band_px"]),
               (~mask)&(outside<=policy["source_boundary_band_px"]),mask,
               (~mask)&(outside<=policy["source_background_support_px"])]
        for category,pixels_mask in enumerate(bands):
            pixels=np.argwhere(pixels_mask)[:,[1,0]]
            if not len(pixels):
                raise RuntimeError(f"APPARATUS_EMPTY_SOURCE_RAY_STRATUM:{view}:{STRATA[category]}")
            rng=np.random.default_rng(seed+view*100+category);accepted=0;drawn=0;local_xy=[];local_phase=[]
            while accepted<count:
                # Historical Knight C apparatus always draws 512 candidate tuples per
                # certification iteration; preserve that draw granularity exactly.
                n=512
                xy=pixels[rng.integers(0,len(pixels),size=n)].astype(np.float32)
                phase=rng.random(n).astype(np.float32);views=np.full(n,view,np.int64)
                q=ray_queries(subject,torch.from_numpy(xy),torch.from_numpy(views),torch.from_numpy(phase),policy["silhouette_depth_samples"],device)
                f=targets(subject,q.reshape(-1,3).cpu().numpy(),device,torch_authority=True)
                values=torch.from_numpy(f).reshape(n,policy["silhouette_depth_samples"])
                m=soft_ray_minimum_v5(values,tau=policy["silhouette_softmin_tau"])
                foreground=category in (0,2)
                correct=(m<=0) if foreground else (m>=0)
                good=np.flatnonzero(correct.numpy())[:count-accepted]
                local_xy.append(xy[good]);local_phase.append(phase[good]);accepted+=len(good);drawn+=n
                if drawn>131072:
                    raise RuntimeError(f"APPARATUS_FSTAR_SOURCE_RAY_CERTIFICATION_EXHAUSTED:{view}:{STRATA[category]}:{accepted}/{count}")
            xy_parts.append(np.concatenate(local_xy));phase_parts.append(np.concatenate(local_phase))
            view_parts.append(np.full(count,view,np.int64));fg_parts.append(np.full(count,foreground,bool));strata_parts.append(np.full(count,category,np.int64))
            receipt.append({"view":view,"stratum":STRATA[category],"admitted":accepted,"drawn":drawn,"source_label_sign_errors":0})
    result={"ray_xy":torch.from_numpy(np.concatenate(xy_parts)),"ray_phase":torch.from_numpy(np.concatenate(phase_parts)),
            "ray_views":torch.from_numpy(np.concatenate(view_parts)),"ray_foreground":torch.from_numpy(np.concatenate(fg_parts)),
            "ray_strata":torch.from_numpy(np.concatenate(strata_parts))}
    result["ray_court"]={"schema":"RealSaS.FStarSourceSilhouetteApparatus.v1","status":"PASS",
                         "strata":receipt,"depth_samples":policy["silhouette_depth_samples"],"tau":policy["silhouette_softmin_tau"],
                         "source_masks_are_forward_inputs":False,"fstar_is_training_label":False,"elapsed_seconds":time.monotonic()-start}
    return result

def training_queries_from_rng(subject, policy, rng, replay_indices, device):
    """Knight V5 16k/8k/4k/4k query mixture from an explicit RNG authority."""
    if policy["queries_per_step"]%8:
        raise ValueError("QUERY_MIX_NOT_DIVISIBLE_BY_EIGHT")
    n=policy["queries_per_step"]
    uniform=rng.uniform(-1,1,(n//2,3)).astype(np.float32)
    pieces=[torch.from_numpy(uniform).to(device)]
    for key,count in (("near_points",n//4),("straddle_points",n//8)):
        ids=torch.from_numpy(rng.integers(0,len(subject[key]),size=count)).to(device)
        pieces.append(subject[key][ids])
    ridx=torch.from_numpy(rng.integers(0,len(replay_indices),size=n//8)).to(device)
    pieces.append(subject["replay_points"][replay_indices.to(device)[ridx]])
    return torch.cat(pieces)

def training_queries(subject, policy, seed, cycle, replay_indices, device):
    rng=np.random.default_rng(seed+cycle)
    return training_queries_from_rng(subject,policy,rng,replay_indices,device)

def training_rays(subject, policy, seed, cycle, device):
    n=policy["silhouette_rays_per_step"]
    if n%32:
        raise ValueError("SOURCE_RAYS_NOT_BALANCED_OVER_ALL_VIEWS_AND_STRATA")
    rng=np.random.default_rng(seed+cycle);per=n//32;pool=policy["silhouette_pool_per_stratum"]
    ids=np.concatenate([group*pool+rng.integers(0,pool,size=per) for group in range(32)])
    ids=torch.from_numpy(ids).to(device)
    q=ray_queries(subject,subject["ray_xy"][ids],subject["ray_views"][ids],subject["ray_phase"][ids],policy["silhouette_depth_samples"],device)
    return q,subject["ray_foreground"][ids]
