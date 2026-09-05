from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TopologyPreservingCompactionResultV1:
    points: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    inverse: np.ndarray
    divisions: int
    compact_node_count: int
    split_voxel_count: int
    max_components_per_voxel: int


class _UnionFind:
    def __init__(self, n: int):
        self.parent=np.arange(n,dtype=np.int64)
        self.rank=np.zeros(n,dtype=np.int8)
    def find(self,x:int)->int:
        p=self.parent
        while p[x]!=x:
            p[x]=p[p[x]]; x=int(p[x])
        return x
    def union(self,a:int,b:int)->None:
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return
        if self.rank[ra] < self.rank[rb]: ra,rb=rb,ra
        self.parent[rb]=ra
        if self.rank[ra]==self.rank[rb]: self.rank[ra]+=1


def _normalize_rows(x:np.ndarray)->np.ndarray:
    n=np.linalg.norm(x,axis=1,keepdims=True)
    if np.any(~np.isfinite(x)) or np.any(n<=1e-12):
        raise ValueError("degenerate compact normal")
    return x/n


def _voxel_keys(points:np.ndarray,divisions:int)->np.ndarray:
    lo=points.min(0); span=np.maximum(points.max(0)-lo,1e-12)
    keys=np.floor((points-lo)/span*divisions).astype(np.int64)
    return np.clip(keys,0,divisions-1)


def _group_same_voxel_connected_components(points:np.ndarray,faces:np.ndarray,divisions:int):
    keys=_voxel_keys(points,divisions)
    uf=_UnionFind(len(points))
    for tri in faces:
        for a,b in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):
            if np.array_equal(keys[a],keys[b]):
                uf.union(a,b)
    roots=np.asarray([uf.find(i) for i in range(len(points))],dtype=np.int64)
    # Group identity is (voxel key, connectivity root). Lexicographic unique gives
    # deterministic compact ordering independent of hash iteration order.
    compound=np.concatenate([keys,roots[:,None]],axis=1)
    unique,inverse=np.unique(compound,axis=0,return_inverse=True)
    # Telemetry: number of connectivity components represented inside each voxel.
    voxel_unique, voxel_inverse=np.unique(unique[:,:3],axis=0,return_inverse=True)
    per_voxel=np.bincount(voxel_inverse,minlength=len(voxel_unique))
    return inverse.astype(np.int64), int(np.sum(per_voxel>1)), int(per_voxel.max(initial=0))


def topology_preserving_component_voxel_compact_v1(
    points,
    faces,
    dense_normals,
    *,
    target_nodes:int=1024,
    max_divisions:int=512,
)->TopologyPreservingCompactionResultV1:
    """Voxel compaction that never fuses disconnected same-voxel mesh patches.

    Unlike a pure spatial voxel average, dense vertices share a compact node only if
    they occupy the same voxel *and* are connected through mesh edges that remain
    inside that voxel.  This is a deterministic challenger for scene-first GSA; it
    carries no teacher rig/skin information.
    """
    p=np.asarray(points,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    n=np.asarray(dense_normals,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<4 or not np.isfinite(p).all():
        raise ValueError("points must be finite [N,3], N>=4")
    if n.shape!=p.shape or not np.isfinite(n).all():
        raise ValueError("dense_normals must match points")
    if f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(p)):
        raise ValueError("faces must be valid [F,3]")
    if target_nodes<64 or max_divisions<1:
        raise ValueError("invalid compactor budget")
    if len(p)<=target_nodes:
        inverse=np.arange(len(p),dtype=np.int64); divisions=0; split_voxels=0; max_components=1
    else:
        low,high=1,int(max_divisions); best=None
        while low<=high:
            mid=(low+high)//2
            inv,split,maxcomp=_group_same_voxel_connected_components(p,f,mid)
            count=int(inv.max())+1
            if count<=target_nodes:
                best=(mid,inv,split,maxcomp,count); low=mid+1
            else:
                high=mid-1
        if best is None:
            raise ValueError("topology-preserving compaction cannot satisfy target_nodes")
        divisions,inverse,split_voxels,max_components,_=best
    count=int(inverse.max())+1
    counts=np.bincount(inverse,minlength=count).astype(np.float64)
    cp=np.zeros((count,3),np.float64); cn=np.zeros((count,3),np.float64)
    np.add.at(cp,inverse,p); np.add.at(cn,inverse,n)
    cp/=counts[:,None]; cn=_normalize_rows(cn)
    mapped=inverse[f]
    edges=np.concatenate([mapped[:,[0,1]],mapped[:,[1,2]],mapped[:,[2,0]]],axis=0)
    edges=np.sort(edges,axis=1); edges=np.unique(edges[edges[:,0]!=edges[:,1]],axis=0)
    if len(edges)==0:
        raise ValueError("compaction removed all topology")
    return TopologyPreservingCompactionResultV1(
        points=cp.astype(np.float32),
        normals=cn.astype(np.float32),
        edges=edges.astype(np.int64),
        inverse=inverse,
        divisions=int(divisions),
        compact_node_count=int(count),
        split_voxel_count=int(split_voxels),
        max_components_per_voxel=int(max_components),
    )
