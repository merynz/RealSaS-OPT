from __future__ import annotations

"""Rich, product-legal Arachne V7-native A1 conditioning.

This adapter is intentionally *not* a wider version of the historical 20D/8D
conditioning. It preserves the typed/factorized information already present in
RiggingSurfaceIR + Compiler-qualified skeleton state:

- exact GSA local-relation graph and edge metadata;
- exact per-view support and raster coordinates/validity;
- deterministic camera-yaw code bound to each canonical view slot;
- observed/completed/validity state;
- exact accepted skeleton tree/root/deform-root state;
- exact sparse joint->surface mechanical support-anchor identity;
- deterministic point<->joint/parent-segment geometry.

Provenance, source IDs, filenames, source object labels, teacher skin and raw
Geppetto alternatives are *not* neural features. They may participate only in
lineage/certificate hashes.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np

from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    RiggingSurfaceTensorV1,
    tensorize_rigging_surface_v1,
)
from models.arachne.v2.arachne_geometry_v2 import (
    PAIR_GEOMETRY_CONTRACT_V2,
    arachne_pair_geometry_v2,
)


SCHEMA = "RealSaS.ArachneRichConditioningBatch.v3"
CONDITION_QUERY_COUNT = 384


def _json_norm(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _json_norm(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (tuple, list)):
        return [_json_norm(x) for x in v]
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("non-finite conditioning payload")
    return v


def _hash(v) -> str:
    raw = json.dumps(_json_norm(v), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def deterministic_fps(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point sampling matching the current A0 policy."""
    p = np.asarray(points, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("FPS points must be finite [N,3]")
    count = int(count)
    if count <= 0 or len(p) < count:
        raise ValueError(f"condition query contract requires N >= {count}; got {len(p)}")
    out = np.empty(count, np.int64)
    dist = np.full(len(p), np.inf, np.float64)
    farthest = 0
    for i in range(count):
        out[i] = farthest
        d = np.square(p - p[farthest]).sum(axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    if len(np.unique(out)) != count:
        raise RuntimeError("deterministic FPS produced duplicate query indices")
    return out


def canonical_view_yaw_code() -> np.ndarray:
    """Canonical view identity as geometry, not a learned V0..V7 slot.

    The product camera contract binds view index i to yaw i*45 degrees. Keeping
    this code beside the corresponding raster/support evidence allows a legal
    view permutation to permute evidence and camera code together.
    """
    yaw = np.arange(8, dtype=np.float64) * (math.pi / 4.0)
    return np.stack(
        [np.sin(yaw), np.cos(yaw), np.sin(2.0*yaw), np.cos(2.0*yaw)], axis=-1
    ).astype(np.float32)


def _depths(parent: np.ndarray, valid: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, np.int64)
    valid = np.asarray(valid, bool)
    out = np.zeros(len(parent), np.int64)
    state = np.zeros(len(parent), np.int8)

    def visit(i: int) -> int:
        if not valid[i]:
            return 0
        if state[i] == 2:
            return int(out[i])
        if state[i] == 1:
            raise ValueError("qualified skeleton cycle")
        state[i] = 1
        p = int(parent[i])
        if p < 0:
            d = 0
        else:
            if p >= len(parent) or not valid[p]:
                raise ValueError("qualified skeleton parent references invalid joint")
            d = 1 + visit(p)
        out[i] = d
        state[i] = 2
        return d

    for i in range(len(parent)):
        if valid[i]:
            visit(i)
    return out


@dataclass(frozen=True)
class ArachneRichConditioningBatchV3:
    surface_ids: tuple[tuple[str, ...], ...]
    joint_ids: tuple[tuple[str, ...], ...]
    surface_positions_world: np.ndarray
    surface_positions_normalized: np.ndarray
    surface_normals: np.ndarray
    surface_normal_valid: np.ndarray
    surface_support_views: np.ndarray
    surface_raster_xy: np.ndarray
    surface_raster_valid: np.ndarray
    view_yaw_code: np.ndarray
    surface_observed: np.ndarray
    surface_completed: np.ndarray
    surface_validity_bits: np.ndarray
    validity_vocab: tuple[str, ...]
    surface_mask: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_relation_kind: np.ndarray
    relation_kind_vocab: tuple[str, ...]
    edge_mask: np.ndarray
    joint_positions_normalized: np.ndarray
    joint_mask: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    deform_root_mask: np.ndarray
    joint_depth_normalized: np.ndarray
    support_anchor_matrix: np.ndarray
    pair_geometry: np.ndarray
    pair_mask: np.ndarray
    pair_geometry_contract: tuple[str, ...]
    geometry7: np.ndarray
    condition_query_indices: np.ndarray
    source_surface_hashes: tuple[str, ...]
    source_skeleton_hashes: tuple[str, ...]
    surface_tensorization_hashes: tuple[str, ...]
    conditioning_hashes: tuple[str, ...]
    assembly_root_bindings: tuple[dict, ...]
    schema_version: str = SCHEMA


class ArachneRichConditioningAdapterV3:
    condition_query_count = CONDITION_QUERY_COUNT

    def __init__(self, *, require_scene_first: bool = True):
        self.require_scene_first = bool(require_scene_first)

    @staticmethod
    def _skeleton_rows(skeleton, tensor: RiggingSurfaceTensorV1):
        joints = tuple(sorted(tuple(skeleton.joints), key=lambda j: str(j.canonical_joint_id)))
        if not joints:
            raise ValueError("qualified skeleton contains no joints")
        jids = tuple(str(j.canonical_joint_id) for j in joints)
        if any(not x for x in jids) or len(jids) != len(set(jids)):
            raise ValueError("qualified joint IDs must be non-empty and unique")
        jmap = {x: i for i, x in enumerate(jids)}
        parent = np.full(len(joints), -1, np.int64)
        pos = np.zeros((len(joints), 3), np.float32)
        support = np.zeros((len(joints), tensor.node_count), bool)
        surface_row = {sid: i for i, sid in enumerate(tensor.surface_ids)}
        for ji, joint in enumerate(joints):
            raw = np.asarray(joint.position, np.float64)
            if raw.shape != (3,) or not np.isfinite(raw).all():
                raise ValueError(f"invalid qualified joint position:{jids[ji]}")
            pos[ji] = ((raw - tensor.normalization_center) / float(tensor.normalization_scale)).astype(np.float32)
            pid = joint.parent_canonical_id
            if pid is not None:
                if str(pid) not in jmap:
                    raise ValueError(f"qualified parent missing:{pid}")
                parent[ji] = jmap[str(pid)]
            anchors = tuple(map(str, joint.support_surface_ids))
            if len(anchors) != len(set(anchors)):
                raise ValueError(f"duplicate support anchor:{jids[ji]}")
            for sid in anchors:
                if sid not in surface_row:
                    raise ValueError(f"support anchor outside surface:{jids[ji]}::{sid}")
                support[ji, surface_row[sid]] = True
        valid = np.ones(len(joints), bool)
        depth = _depths(parent, valid)
        md = max(int(depth.max(initial=0)), 1)
        depth_norm = depth.astype(np.float32) / float(md)
        roots = np.asarray(parent < 0, bool)
        declared_root = str(getattr(skeleton, "root_id", ""))
        if declared_root:
            declared = np.zeros(len(joints), bool)
            if declared_root not in jmap:
                raise ValueError("QualifiedSkeletonIR root_id not found in joints")
            declared[jmap[declared_root]] = True
            if not np.array_equal(roots, declared):
                raise ValueError("qualified root/tree mismatch")
        deform_ids = tuple(map(str, getattr(skeleton, "deform_root_ids", ())))
        if not deform_ids:
            deform_ids = tuple(jids[i] for i in np.where(roots)[0])
        deform = np.zeros(len(joints), bool)
        for jid in deform_ids:
            if jid not in jmap:
                raise ValueError(f"deform root missing:{jid}")
            deform[jmap[jid]] = True
        return joints, jids, pos, parent, roots, deform, depth_norm, support

    def __call__(self, surfaces: Iterable, skeletons: Iterable) -> ArachneRichConditioningBatchV3:
        surfaces = tuple(surfaces)
        skeletons = tuple(skeletons)
        if not surfaces or len(surfaces) != len(skeletons):
            raise ValueError("surfaces/skeletons require equal nonzero batch")
        tensors = tuple(tensorize_rigging_surface_v1(s, require_scene_first=self.require_scene_first) for s in surfaces)
        sk_rows = tuple(self._skeleton_rows(g, t) for g, t in zip(skeletons, tensors))
        B = len(tensors)
        N = max(t.node_count for t in tensors)
        E = max(t.edge_count for t in tensors)
        J = max(len(x[1]) for x in sk_rows)
        vv = tuple(sorted({flag for t in tensors for flag in t.validity_vocab}))
        vmap = {x: i for i, x in enumerate(vv)}
        rvocab = tuple(sorted({kind for t in tensors for kind in t.relation_kind_vocab}))
        rkmap = {x: i for i, x in enumerate(rvocab)}
        pw=np.zeros((B,N,3),np.float32); pn=np.zeros((B,N,3),np.float32); sn=np.zeros((B,N,3),np.float32)
        snv=np.zeros((B,N),bool); sup=np.zeros((B,N,8),bool); rxy=np.zeros((B,N,8,2),np.float32); rv=np.zeros((B,N,8),bool)
        view_yaw=np.broadcast_to(canonical_view_yaw_code()[None],(B,8,4)).copy()
        obs=np.zeros((B,N),bool); comp=np.zeros((B,N),bool); vbits=np.zeros((B,N,len(vv)),bool); sm=np.zeros((B,N),bool)
        ei=np.zeros((B,E,2),np.int64); ef=np.zeros((B,E,4),np.float32); erk=np.zeros((B,E),np.int64); em=np.zeros((B,E),bool)
        jp=np.zeros((B,J,3),np.float32); jm=np.zeros((B,J),bool); pi=np.full((B,J),-1,np.int64); root=np.zeros((B,J),bool)
        deform=np.zeros((B,J),bool); depth=np.zeros((B,J),np.float32); anchors=np.zeros((B,J,N),bool); qidx=np.zeros((B,CONDITION_QUERY_COUNT),np.int64)
        sids_all=[]; jids_all=[]; sh=[]; gh=[]; th=[]; ch=[]; assembly_bindings=[]
        for b,(surface,skeleton,t,sr) in enumerate(zip(surfaces,skeletons,tensors,sk_rows)):
            _,jids,jpos,parent,roots,droots,dnorm,anchor=sr
            n,e,j=t.node_count,t.edge_count,len(jids)
            sids_all.append(tuple(t.surface_ids)); jids_all.append(tuple(jids)); pw[b,:n]=t.positions_world; pn[b,:n]=t.positions_normalized
            sn[b,:n]=t.normals; snv[b,:n]=t.normal_valid; sup[b,:n]=t.support; rxy[b,:n]=t.raster_xy_normalized; rv[b,:n]=t.raster_valid
            obs[b,:n]=t.observed; comp[b,:n]=t.completed; sm[b,:n]=True
            for local_idx,flag in enumerate(t.validity_vocab): vbits[b,:n,vmap[flag]]=t.validity_bits[:,local_idx]
            if e:
                ei[b,:e]=t.edge_index; ef[b,:e,0]=t.edge_score; ef[b,:e,1]=t.edge_distance_normalized
                ef[b,:e,2]=t.edge_crosses_unknown.astype(np.float32); ef[b,:e,3]=t.edge_unknown_bridge.astype(np.float32)
                local_kinds=[t.relation_kind_vocab[int(k)] for k in t.relation_kind_index]
                erk[b,:e]=np.asarray([rkmap[k] for k in local_kinds],np.int64); em[b,:e]=True
            jp[b,:j]=jpos; jm[b,:j]=True; pi[b,:j]=parent; root[b,:j]=roots; deform[b,:j]=droots; depth[b,:j]=dnorm; anchors[b,:j,:n]=anchor
            qidx[b]=deterministic_fps(t.positions_normalized,CONDITION_QUERY_COUNT)
            surface_hash=str(surface.geometry_lineage_hash); skeleton_hash=str(skeleton.skeleton_lineage_hash)
            if not surface_hash or not skeleton_hash: raise ValueError("surface/skeleton lineage hash required")
            sh.append(surface_hash); gh.append(skeleton_hash); th.append(t.tensorization_hash)
            assembly=getattr(skeleton,"assembly_root_binding",{}); assembly_bindings.append(dict(assembly) if isinstance(assembly,dict) else {"value":assembly})
        pair,pair_mask=arachne_pair_geometry_v2(pn,sn,snv,jp,pi,sm,jm)
        geometry7=np.concatenate([2.0*pn,sn,snv[...,None].astype(np.float32)],axis=-1).astype(np.float32)
        for b in range(B):
            n=int(sm[b].sum()); j=int(jm[b].sum())
            ch.append(_hash({"schema":SCHEMA,"surface_hash":sh[b],"skeleton_hash":gh[b],"surface_tensorization_hash":th[b],"surface_ids":sids_all[b],"joint_ids":jids_all[b],"positions_normalized":pn[b,:n],"support":sup[b,:n],"raster_xy":rxy[b,:n],"raster_valid":rv[b,:n],"view_yaw_code":view_yaw[b],"observed":obs[b,:n],"completed":comp[b,:n],"validity_vocab":vv,"validity_bits":vbits[b,:n],"edge_index":ei[b,em[b]],"edge_features":ef[b,em[b]],"edge_relation_kind":erk[b,em[b]],"relation_kind_vocab":rvocab,"joint_positions":jp[b,:j],"parent_indices":pi[b,:j],"root_mask":root[b,:j],"deform_root_mask":deform[b,:j],"support_anchor_matrix":anchors[b,:j,:n],"pair_geometry_contract":PAIR_GEOMETRY_CONTRACT_V2,"condition_query_indices":qidx[b]}))
        return ArachneRichConditioningBatchV3(tuple(sids_all),tuple(jids_all),pw,pn,sn,snv,sup,rxy,rv,view_yaw,obs,comp,vbits,vv,sm,ei,ef,erk,rvocab,em,jp,jm,pi,root,deform,depth,anchors,pair,pair_mask,PAIR_GEOMETRY_CONTRACT_V2,geometry7,qidx,tuple(sh),tuple(gh),tuple(th),tuple(ch),tuple(assembly_bindings))


__all__ = ["SCHEMA","CONDITION_QUERY_COUNT","canonical_view_yaw_code","deterministic_fps","ArachneRichConditioningBatchV3","ArachneRichConditioningAdapterV3"]
