from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch


def _sha(path: Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class DemoFitBundle:
    rgba: torch.Tensor                 # [1,8,4,H,W], float32 0..1
    yaw_deg: torch.Tensor              # [8]
    teacher_depth: torch.Tensor        # [1,8,1,R,R]
    teacher_support: torch.Tensor      # [1,8,1,R,R]
    cameras: list[dict]
    joint_positions: torch.Tensor      # [J,3]
    parent_index: torch.Tensor         # [J], root=-1
    root_index: int
    source_vertices: torch.Tensor      # [M,3]
    source_vertex_weights: torch.Tensor# [M,J]
    observation_sha256: str
    truth_sha256: str


def load_fit_bundle(directory: str | Path) -> DemoFitBundle:
    root=Path(directory)
    obs_path=root/"observations.npz";truth_path=root/"mechanical_truth.npz";camera_path=root/"cameras.json"
    for p in (obs_path,truth_path,camera_path):
        if not p.is_file():raise FileNotFoundError(f"DEMO_FIT_BUNDLE_MISSING:{p.name}")
    obs=np.load(obs_path,allow_pickle=False);truth=np.load(truth_path,allow_pickle=False)
    rgba=np.asarray(obs["rgba"])
    if rgba.ndim!=4 or rgba.shape[0]!=8 or rgba.shape[-1]!=4:raise ValueError("rgba must be [8,H,W,4]")
    if rgba.dtype==np.uint8:rgba=rgba.astype(np.float32)/255.0
    else:rgba=rgba.astype(np.float32)
    rgba=np.transpose(rgba,(0,3,1,2))[None]
    yaw=np.asarray(obs["yaw_deg"],np.float32)
    depth=np.asarray(obs["teacher_depth"],np.float32)
    support=np.asarray(obs["teacher_support"],np.float32)
    if yaw.shape!=(8,):raise ValueError("yaw_deg must be [8]")
    if depth.ndim==3:depth=depth[:,None]
    if support.ndim==3:support=support[:,None]
    if depth.shape[0]!=8 or depth.shape[1]!=1 or support.shape!=depth.shape:raise ValueError("teacher depth/support must be [8,1,R,R]")
    cameras=json.loads(camera_path.read_text(encoding="utf-8"))
    if not isinstance(cameras,list) or len(cameras)!=8:raise ValueError("cameras.json must contain eight cameras")
    jp=np.asarray(truth["joint_positions"],np.float32);parent=np.asarray(truth["parent_index"],np.int64)
    root_index=int(np.asarray(truth["root_index"]).reshape(-1)[0])
    sv=np.asarray(truth["source_vertices"],np.float32);sw=np.asarray(truth["source_vertex_weights"],np.float32)
    if jp.ndim!=2 or jp.shape[1]!=3:raise ValueError("joint_positions must be [J,3]")
    if parent.shape!=(len(jp),) or not (0<=root_index<len(jp)):raise ValueError("bad skeleton truth")
    if int(parent[root_index])!=-1:raise ValueError("root parent must be -1")
    if sv.ndim!=2 or sv.shape[1]!=3 or sw.shape!=(len(sv),len(jp)):raise ValueError("bad source vertex/weight truth")
    if not np.isfinite(jp).all() or not np.isfinite(sv).all() or not np.isfinite(sw).all():raise ValueError("nonfinite mechanical truth")
    if (sw<0).any() or (sw.sum(axis=1)<=1e-12).any():raise ValueError("invalid source skin truth")
    sw=sw/sw.sum(axis=1,keepdims=True)
    return DemoFitBundle(
        torch.from_numpy(rgba),torch.from_numpy(yaw),torch.from_numpy(depth[None]),torch.from_numpy(support[None]),cameras,
        torch.from_numpy(jp),torch.from_numpy(parent),root_index,torch.from_numpy(sv),torch.from_numpy(sw),
        sha256((_sha(obs_path)+_sha(camera_path)).encode()).hexdigest(),_sha(truth_path),
    )
