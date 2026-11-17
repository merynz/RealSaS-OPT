from __future__ import annotations

"""Source-free UDF -> open indexed triangle surface, from ONE shared IRIS H(x).

A clean-room directional-gradient marching-tetrahedra candidate, informed by
the MeshUDF/GeoUDF open-surface literature, NOT a port of GPL MeshUDF or an
implementation/quality claim for DCUDF.  The original DCUDF repository is
MIT licensed, but its CUDA optimization/min-cut pipeline is not vendored.

Unlike a signed iso surface, UDF has NO global inside/outside. We extract
locally supported edges with opposing *unsigned gradient directions*, preserve
the exact global grid edge address during deduplication, and project vertices
toward the UDF zero surface. Open boundaries are preserved, never capped.

The output is an UNQUALIFIED geometric candidate. Directional tests can miss
grazing or nonmanifold surfaces and can make holes in closed shapes. An
independently learned, source-free open/closed applicability decision, real
GPU quality and Compiler G2/G3/G3B are still required for product admission.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

SCHEMA = "RealSaS.RSSvNext.OpenUDFDirectionalIndexedSurface.v1"
AUTHORITY = "UNQUALIFIED_ONE_IRIS_H_OPEN_UDF_SURFACE_CANDIDATE"
# The cube is triangulated about its 000->111 main diagonal. All neighboring
# cubes share identical global grid vertex IDs; zero T-junction welding by XYZ.
TETS = ((0,1,3,7),(0,3,2,7),(0,2,6,7),
        (0,6,4,7),(0,4,5,7),(0,5,1,7))


def tensor_bytes_sha256_v1(t: Tensor) -> str:
    a=t.detach().contiguous().cpu().numpy()
    h=sha256()
    h.update(str((str(a.dtype),a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _check_topology(faces: Tensor, vertex_count: int):
    if (faces.dtype != torch.long or faces.ndim != 2
            or faces.shape[-1] != 3 or not len(faces)
            or bool((faces<0).any()) or int(faces.max())>=vertex_count):
        raise ValueError("RSS_VNEXT_UDF_INDEXED_FACE_DOMAIN")
    if bool(((faces[:,0]==faces[:,1]) |
             (faces[:,1]==faces[:,2]) |
             (faces[:,2]==faces[:,0])).any()):
        raise ValueError("RSS_VNEXT_UDF_DUPLICATE_CORNER")
    if torch.unique(faces.sort(dim=-1).values,dim=0).shape[0]!=len(faces):
        raise ValueError("RSS_VNEXT_UDF_DUPLICATE_TRIANGLE")
    edges=torch.cat((
        faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]],
    ),dim=0).sort(dim=-1).values
    edge_keys,incidence=torch.unique(edges,dim=0,return_counts=True)
    if bool((incidence>2).any()):
        raise ValueError("RSS_VNEXT_UDF_NONMANIFOLD_EDGE")
    boundary=edge_keys[incidence==1]
    if len(boundary)<3:
        raise ValueError("RSS_VNEXT_UDF_NOT_AN_OPEN_SHEET")
    # Every open border belongs to a closed boundary loop, not a dangling
    # crack or branched T-junction at a lone vertex.
    degree=torch.bincount(boundary.reshape(-1),minlength=vertex_count)
    if not bool((degree[degree>0]==2).all()):
        raise ValueError("RSS_VNEXT_UDF_BOUNDARY_BRANCH_OR_CRACK")
    return boundary


def _orient_consistently(faces: list[list[int]]) -> list[list[int]]:
    """BFS orient faces across every shared edge, fail on nonorientability."""
    incident=defaultdict(list)
    for fi,(a,b,c) in enumerate(faces):
        for u,v in ((a,b),(b,c),(c,a)):
            incident[tuple(sorted((u,v)))].append((fi, (u,v)))
    flipped=[None]*len(faces)
    for start in range(len(faces)):
        if flipped[start] is not None:
            continue
        flipped[start]=False
        queue=deque([start])
        while queue:
            current=queue.popleft()
            a,b,c=faces[current]
            for u,v in ((a,b),(b,c),(c,a)):
                pair=tuple(sorted((u,v)))
                for other,ordered in incident[pair]:
                    if other==current:
                        continue
                    same=(u,v)==ordered
                    expected=bool(flipped[current]) ^ same
                    if flipped[other] is None:
                        flipped[other]=expected
                        queue.append(other)
                    elif flipped[other]!=expected:
                        raise ValueError("RSS_VNEXT_UDF_NONORIENTABLE_SHEET")
    return [
        [a,c,b] if flip else [a,b,c]
        for (a,b,c),flip in zip(faces,flipped)
    ]


def _components(vertex_count:int, faces:Tensor) -> Tensor:
    parents=list(range(vertex_count))
    def root(x:int)->int:
        while parents[x]!=x:
            parents[x]=parents[parents[x]]
            x=parents[x]
        return x
    for a,b,c in faces.cpu().tolist():
        ra,rb,rc=root(a),root(b),root(c)
        parents[rb]=ra
        parents[rc]=ra
    names={}
    return torch.tensor([
        names.setdefault(root(i),len(names)) for i in range(vertex_count)
    ],dtype=torch.long,device=faces.device)


@dataclass(frozen=True)
class OpenUDFSurfaceCandidateV1:
    xyz: Tensor                 # [1,V,3], differentiable from SAME IRIS H
    faces: Tensor               # [F,3], exact global edge-derived indices
    component_ids: Tensor       # [V], connected indexed face graph
    boundary_edges: Tensor      # [E,2], sorted, no artificial capping
    scene_sha256: str
    sampled_udf_sha256: str
    extraction_resolution: tuple[int,int,int]
    field_max_residual: float
    source_authority: str = AUTHORITY
    canonical_mesh_authority: bool = False

    def validate(self) -> None:
        if (
            self.source_authority!=AUTHORITY
            or self.canonical_mesh_authority
            or self.xyz.ndim!=3 or self.xyz.shape[0]!=1
            or self.xyz.shape[-1]!=3 or self.xyz.shape[1]<3
            or not bool(torch.isfinite(self.xyz).all())
            or self.component_ids.shape!=(self.xyz.shape[1],)
            or self.component_ids.dtype!=torch.long
            or self.component_ids.device!=self.xyz.device
            or bool((self.component_ids<0).any())
            or self.faces.device!=self.xyz.device
            or self.boundary_edges.device!=self.xyz.device
            or any(type(s) is not str or len(s)!=64
                   or any(c not in "0123456789abcdef" for c in s)
                   for s in (self.scene_sha256,self.sampled_udf_sha256))
            or not math.isfinite(self.field_max_residual)
            or self.field_max_residual<0
        ):
            raise ValueError("RSS_VNEXT_UDF_SURFACE_DOMAIN")
        boundary=_check_topology(self.faces,self.xyz.shape[1])
        if not torch.equal(boundary,self.boundary_edges):
            raise ValueError("RSS_VNEXT_UDF_BOUNDARY_ADDRESS_DRIFT")
        if not bool((
            self.component_ids[self.faces]
            ==self.component_ids[self.faces[:,0:1]]
        ).all()):
            raise ValueError("RSS_VNEXT_UDF_FACE_COMPONENT_DRIFT")


def extract_open_udf_surface_v1(
    *, scene_triplanes: Tensor,
    query_unsigned_distance: Callable[[Tensor],Tensor],
    resolution: tuple[int,int,int] = (32,32,32),
    query_chunk: int = 8192,
    gradient_alignment: float = .55,
    distance_gate_cells: float = 1.6,
    max_active_cubes: int = 100_000,
    max_output_vertices: int = 200_000,
) -> OpenUDFSurfaceCandidateV1:
    """Source-free directional UDF mesher; not qualified as production backend.

    A full-grid CPU scalar preflight bounds peak neural activation memory.
    Topology is non-differentiable, like FlexiCubes sign/cube decisions;
    active edge interpolation makes xyz differentiable to the UDF head.
    Neither source mesh truth nor semantic role labels enter this function.
    """
    if (
        not isinstance(scene_triplanes,Tensor) or scene_triplanes.ndim!=5
        or scene_triplanes.shape[0]!=1
        or not callable(query_unsigned_distance)
        or type(resolution) is not tuple or len(resolution)!=3
        or any(type(n) is not int or not 8<=n<=96 for n in resolution)
        or math.prod(resolution)>700_000
        or type(query_chunk) is not int or not 128<=query_chunk<=65536
        or type(max_active_cubes) is not int or not 1<=max_active_cubes<=100_000
        or type(max_output_vertices) is not int
        or not 3<=max_output_vertices<=200_000
        or type(gradient_alignment) not in (int,float)
        or not .1<=gradient_alignment<=.95
        or type(distance_gate_cells) not in (int,float)
        or not .75<=distance_gate_cells<=2.0
        or not bool(torch.isfinite(scene_triplanes).all())
    ):
        raise ValueError("RSS_VNEXT_UDF_EXTRACTION_POLICY_INVALID")
    nx,ny,nz=resolution
    # Global vertex indexing: i*(ny*nz)+j*nz+k. Same across all cubes.
    cpu_axes=[torch.linspace(-1.,1.,n) for n in resolution]
    grid=torch.stack(
        torch.meshgrid(*cpu_axes,indexing="ij"),dim=-1,
    ).reshape(-1,3)
    values=[]
    with torch.no_grad():
        for start in range(0,len(grid),query_chunk):
            pts=grid[start:start+query_chunk].to(scene_triplanes)
            udf=query_unsigned_distance(pts)
            if (
                not isinstance(udf,Tensor)
                or udf.shape!=(len(pts),)
                or not bool(torch.isfinite(udf).all())
                or bool((udf<0).any())
            ):
                raise ValueError("RSS_VNEXT_UDF_SCAN_UNSIGNED_FIELD_INVALID")
            values.append(udf.detach().float().cpu())
    volume=torch.cat(values).reshape(nx,ny,nz)
    steps=tuple(2./(n-1) for n in resolution)
    derivatives=torch.gradient(volume,spacing=steps)
    grad=F.normalize(torch.stack(derivatives,dim=-1).reshape(-1,3),
                     dim=-1,eps=1.e-8)
    sdf=volume.reshape(-1)
    diag=math.sqrt(sum(s*s for s in steps))
    sample_limit=float(distance_gate_cells)*max(steps)
    edges=[];edge_index={};face_ids=[]
    active=0
    yz=ny*nz
    for ix in range(nx-1):
        for iy in range(ny-1):
            for iz in range(nz-1):
                v=ix*yz+iy*nz+iz
                cube=(v,v+yz,v+nz,v+yz+nz,
                      v+1,v+yz+1,v+nz+1,v+yz+nz+1)
                if float(sdf[list(cube)].min())>sample_limit:
                    continue
                active+=1
                if active>max_active_cubes:
                    raise ValueError("RSS_VNEXT_UDF_ACTIVE_CUBE_BUDGET")
                for tet in TETS:
                    ids=[cube[k] for k in tet]
                    gt=grad[ids]
                    distances=sdf[ids]
                    relative=(gt*gt[[0]]).sum(-1)
                    if not bool((relative.abs()>=gradient_alignment).all()):
                        continue
                    pos=[k for k in range(4) if relative[k]>0]
                    neg=[k for k in range(4) if relative[k]<0]
                    if not pos or not neg:
                        continue
                    # Underlying UDF must exhibit actual *opposing*
                    # gradient directions near the local zero set.
                    crossing_valid=True
                    for a in pos:
                        for b in neg:
                            if (float((gt[a]*gt[b]).sum())>-
                                    gradient_alignment or
                                    float(distances[a]+distances[b])>
                                    sample_limit):
                                crossing_valid=False
                                break
                        if not crossing_valid:break
                    if not crossing_valid:
                        continue
                    if len(pos)==1 or len(neg)==1:
                        one,many=(pos,neg) if len(pos)==1 else (neg,pos)
                        tris=[[(ids[one[0]],ids[k]) for k in many]]
                    else:
                        a,b=pos;c,d=neg
                        ac=(ids[a],ids[c]);ad=(ids[a],ids[d])
                        bc=(ids[b],ids[c]);bd=(ids[b],ids[d])
                        tris=[[ac,ad,bd],[ac,bd,bc]]
                    for tri in tris:
                        f=[]
                        for a,b in tri:
                            key=tuple(sorted((a,b)))
                            if key not in edge_index:
                                edge_index[key]=len(edges)
                                edges.append(key)
                                if len(edges)>max_output_vertices:
                                    raise ValueError("RSS_VNEXT_UDF_VERTEX_BUDGET")
                            f.append(edge_index[key])
                        if len(set(f))==3:
                            face_ids.append(f)
    if not face_ids:
        raise ValueError("RSS_VNEXT_UDF_NO_OPEN_SURFACE_CANDIDATE")
    global_edges=torch.tensor(edges,device=scene_triplanes.device,
                              dtype=torch.long)
    endpoints=grid[global_edges.cpu()].to(scene_triplanes)
    # Re-query only the surface-active UDF edge endpoints WITH autograd.
    # The full CPU preflight scan is detached on purpose.
    a=endpoints[:,0];b=endpoints[:,1]
    evaluated=[]
    for x in (a,b):
        parts=[]
        for start in range(0,len(x),query_chunk):
            y=query_unsigned_distance(x[start:start+query_chunk])
            if (not isinstance(y,Tensor) or y.shape!=
                    (len(x[start:start+query_chunk]),)
                    or not bool(torch.isfinite(y).all())
                    or bool((y<0).any())):
                raise ValueError("RSS_VNEXT_UDF_ACTIVE_FIELD_INVALID")
            parts.append(y)
        evaluated.append(torch.cat(parts))
    ua,ub=evaluated
    mix=ua/(ua+ub).clamp_min(1.e-9)
    positions=a*(1-mix[:,None])+b*mix[:,None]
    face_tensor=torch.tensor(face_ids,dtype=torch.long,
                             device=scene_triplanes.device)
    # Remove numerical degeneracies before manifold court. No face creation
    # by proximity, no source or body/cloth coincidence weld.
    tri=positions[face_tensor]
    dbl=torch.linalg.vector_norm(torch.cross(
        tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1,
    ),dim=-1)
    face_tensor=face_tensor[dbl>1.e-10]
    if not len(face_tensor):
        raise ValueError("RSS_VNEXT_UDF_ONLY_DEGENERATE_FACES")
    used=torch.unique(face_tensor.reshape(-1),sorted=True)
    remap=torch.full((len(positions),),-1,device=positions.device,
                     dtype=torch.long)
    remap[used]=torch.arange(len(used),device=positions.device)
    faces=remap[face_tensor]
    positions=positions[used]
    faces=torch.tensor(_orient_consistently(faces.detach().cpu().tolist()),
                       dtype=torch.long,device=positions.device)
    boundary=_check_topology(faces,len(positions))
    # A model-supplied UDF has NOT been independently proved as source
    # topology. Reject grossly misplaced extraction relative to field.
    with torch.no_grad():
        residual=[]
        for start in range(0,len(positions),query_chunk):
            x=positions[start:start+query_chunk]
            v=query_unsigned_distance(x)
            if (not isinstance(v,Tensor) or v.shape!=(len(x),)
                    or not bool(torch.isfinite(v).all())
                    or bool((v<0).any())):
                raise ValueError("RSS_VNEXT_UDF_OUTPUT_FIELD_INVALID")
            residual.append(v.detach().float())
        max_residual=float(torch.cat(residual).max().item())
    if max_residual>diag*.5:
        raise ValueError("RSS_VNEXT_UDF_OUTPUT_RESIDUAL_TOO_HIGH")
    components=_components(len(positions),faces)
    candidate=OpenUDFSurfaceCandidateV1(
        xyz=positions.unsqueeze(0),
        faces=faces,component_ids=components,
        boundary_edges=boundary,
        scene_sha256=tensor_bytes_sha256_v1(scene_triplanes),
        sampled_udf_sha256=tensor_bytes_sha256_v1(volume),
        extraction_resolution=resolution,
        field_max_residual=max_residual,
    )
    candidate.validate()
    return candidate


__all__=[
    "SCHEMA","AUTHORITY","OpenUDFSurfaceCandidateV1",
    "tensor_bytes_sha256_v1","extract_open_udf_surface_v1",
]
