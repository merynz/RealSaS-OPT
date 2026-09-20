from __future__ import annotations

"""Exact G5 multiview component coverage using Runtime-v3/v4 raster semantics."""

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import math
import numpy as np
from scipy.spatial import cKDTree
from typing import Any, Iterable, Mapping

from ..hashing import content_sha256
from ..camera_authority_v1 import camera_projection_binding_hash
from ..observation_authority_v1 import QualifiedObservationSetIR, validate_qualified_observation_set
from ..raster_contract_v2 import ReferenceRasterContractV1
from ..camera_geometry_v2 import CameraProjectionV3, project_points_xyz_v3
from ..product_authority_v1 import (
    ComponentCarrierPolicyIR,
    MeshQualificationPolicyIR,
    QualifiedMeshIR,
    validate_component_carrier_policy,
    validate_mesh_qualification_policy,
)
from ..types import QualificationError, RiggingSurfaceIR

Json = dict[str, Any]


@dataclass(frozen=True)
class ComponentObservationRasterIR:
    view_index: int
    component_id: str
    carrier_class: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    component_surface_set_hash: str
    width: int
    height: int
    mask_bytes: bytes
    mask_sha256: str
    source_observation_hash: str
    camera_binding_hash: str
    raster_contract_hash: str
    schema_version: str = "RealSaS.ComponentObservationRasterIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        # bytes are represented by exact digest + size in lineage-bearing dicts.
        return {
            "view_index": self.view_index,
            "component_id": self.component_id,
            "carrier_class": self.carrier_class,
            "partition_binding_hash": self.partition_binding_hash,
            "carrier_policy_binding_hash": self.carrier_policy_binding_hash,
            "component_surface_set_hash": self.component_surface_set_hash,
            "width": self.width,
            "height": self.height,
            "mask_sha256": self.mask_sha256,
            "mask_bytes_count": len(self.mask_bytes),
            "source_observation_hash": self.source_observation_hash,
            "camera_binding_hash": self.camera_binding_hash,
            "raster_contract_hash": self.raster_contract_hash,
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }


def mask_sha256(mask_bytes: bytes) -> str:
    return hashlib.sha256(bytes(mask_bytes)).hexdigest()


def component_surface_set_hash(component) -> str:
    return content_sha256({
        "schema": "RealSaS.ComponentSurfaceSetBinding.v1",
        "component_id": component.component_id,
        "surface_ids": tuple(sorted(component.surface_ids)),
    })


def validate_component_observation(value: ComponentObservationRasterIR, *, partition=None, carrier_policy=None) -> None:
    if not (0 <= int(value.view_index) < 8):
        raise QualificationError("G5_OBSERVATION_VIEW_INVALID")
    if not value.component_id or not value.carrier_class:
        raise QualificationError("G5_OBSERVATION_COMPONENT_INVALID")
    if int(value.width) <= 0 or int(value.height) <= 0:
        raise QualificationError("G5_OBSERVATION_DIMENSION_INVALID")
    if len(value.mask_bytes) != int(value.width) * int(value.height):
        raise QualificationError("G5_OBSERVATION_MASK_SIZE_INVALID")
    if any(x not in (0, 1) for x in value.mask_bytes):
        raise QualificationError("G5_OBSERVATION_MASK_BINARY_REQUIRED")
    if value.mask_sha256 != mask_sha256(value.mask_bytes):
        raise QualificationError("G5_OBSERVATION_MASK_HASH_MISMATCH")
    if not value.source_observation_hash or not value.camera_binding_hash or not value.raster_contract_hash:
        raise QualificationError("G5_OBSERVATION_AUTHORITY_BINDING_MISSING")
    if not value.partition_binding_hash or not value.carrier_policy_binding_hash or not value.component_surface_set_hash:
        raise QualificationError("G5_OBSERVATION_PARTITION_BINDING_MISSING")
    if partition is not None:
        if value.partition_binding_hash != partition.partition_lineage_hash:
            raise QualificationError("G5_OBSERVATION_PARTITION_LINEAGE_MISMATCH")
        component = next((row for row in partition.components if row.component_id == value.component_id), None)
        if component is None:
            raise QualificationError("G5_OBSERVATION_COMPONENT_NOT_IN_PARTITION")
        if value.component_surface_set_hash != component_surface_set_hash(component):
            raise QualificationError("G5_OBSERVATION_COMPONENT_SURFACE_SET_MISMATCH")
    if carrier_policy is not None:
        if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
            raise QualificationError("G5_OBSERVATION_CARRIER_POLICY_LINEAGE_MISMATCH")
        decision = next((row for row in carrier_policy.decisions if row.component_id == value.component_id), None)
        if decision is None or decision.carrier_class != value.carrier_class:
            raise QualificationError("G5_OBSERVATION_CARRIER_POLICY_DRIFT")


def _orient2d(a, b, px: float, py: float) -> float:
    return (float(b[0]) - float(a[0])) * (py - float(a[1])) - (float(b[1]) - float(a[1])) * (px - float(a[0]))


def _is_top_left(a, b) -> bool:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    return dy < 0.0 or (dy == 0.0 and dx > 0.0)


def _edge_accept(e: float, top_left: bool) -> bool:
    eps = 1e-12
    if e > eps:
        return True
    if e < -eps:
        return False
    return bool(top_left)


def _covers_pixel_center(a, b, c, pixel_x: int, pixel_y: int) -> bool:
    signed_area = _orient2d(a, b, float(c[0]), float(c[1]))
    if abs(signed_area) <= 1e-12:
        return False
    px = float(pixel_x) + 0.5
    py = float(pixel_y) + 0.5
    positive = signed_area > 0.0
    sign = 1.0 if positive else -1.0
    e0 = sign * _orient2d(b, c, px, py)
    e1 = sign * _orient2d(c, a, px, py)
    e2 = sign * _orient2d(a, b, px, py)
    tl0 = _is_top_left(b, c) if positive else _is_top_left(c, b)
    tl1 = _is_top_left(c, a) if positive else _is_top_left(a, c)
    tl2 = _is_top_left(a, b) if positive else _is_top_left(b, a)
    return _edge_accept(e0, tl0) and _edge_accept(e1, tl1) and _edge_accept(e2, tl2)


def rasterize_triangles_half_integer_top_left(
    triangles: Iterable[tuple[tuple[float,float], tuple[float,float], tuple[float,float]]],
    *,
    width: int,
    height: int,
) -> bytes:
    predicted = bytearray(int(width) * int(height))
    for a, b, c in triangles:
        xs = (float(a[0]), float(b[0]), float(c[0]))
        ys = (float(a[1]), float(b[1]), float(c[1]))
        minx = max(0, int(math.floor(min(xs) - 0.5)))
        maxx = min(int(width) - 1, int(math.ceil(max(xs) - 0.5)))
        miny = max(0, int(math.floor(min(ys) - 0.5)))
        maxy = min(int(height) - 1, int(math.ceil(max(ys) - 0.5)))
        if minx > maxx or miny > maxy:
            continue
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                if _covers_pixel_center(a, b, c, x, y):
                    predicted[y * int(width) + x] = 1
    return bytes(predicted)


def _connected_component_sizes_4(
    mask: bytes,
    *,
    width: int,
    height: int,
) -> tuple[int, ...]:
    n = int(width) * int(height)
    if len(mask) != n:
        raise QualificationError("G5_CONNECTED_MASK_SIZE_MISMATCH")
    visited = bytearray(n)
    sizes = []
    for seed in range(n):
        if not mask[seed] or visited[seed]:
            continue
        visited[seed] = 1
        q = deque([seed])
        size = 0
        while q:
            idx = q.popleft()
            size += 1
            x, y = idx % width, idx // width
            for nxt in (
                idx - 1 if x > 0 else -1,
                idx + 1 if x + 1 < width else -1,
                idx - width if y > 0 else -1,
                idx + width if y + 1 < height else -1,
            ):
                if nxt >= 0 and mask[nxt] and not visited[nxt]:
                    visited[nxt] = 1
                    q.append(nxt)
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def _largest_4_connected(mask: bytes, *, width: int, height: int) -> int:
    sizes = _connected_component_sizes_4(mask, width=width, height=height)
    return int(sizes[0]) if sizes else 0


def source_connected_component_recall_metrics(
    authority: bytes,
    predicted: bytes,
    *,
    width: int,
    height: int,
    minimum_foreground_fraction: float,
) -> dict:
    """Measure per-source-component coverage without categorical recognition.

    Components are 4-connected regions in the admitted source foreground mask.
    The minimum fraction is relative to total source foreground pixels and is a
    preregistered noise/speckle floor, not a semantic-object threshold.
    """
    n=int(width)*int(height)
    if len(authority)!=n or len(predicted)!=n:
        raise QualificationError("SOURCE_COMPONENT_RECALL_MASK_SIZE_MISMATCH")
    if any(x not in (0,1) for x in authority) or any(x not in (0,1) for x in predicted):
        raise QualificationError("SOURCE_COMPONENT_RECALL_BINARY_MASK_REQUIRED")
    floor=float(minimum_foreground_fraction)
    if not math.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise QualificationError("SOURCE_COMPONENT_RECALL_FRACTION_INVALID")
    foreground=int(sum(authority))
    visited=bytearray(n)
    rows=[]
    component_index=0
    for seed in range(n):
        if not authority[seed] or visited[seed]:
            continue
        visited[seed]=1
        q=deque([seed])
        pixels=[]
        while q:
            idx=q.popleft()
            pixels.append(idx)
            x,y=idx%int(width),idx//int(width)
            for nxt in (
                idx-1 if x>0 else -1,
                idx+1 if x+1<int(width) else -1,
                idx-int(width) if y>0 else -1,
                idx+int(width) if y+1<int(height) else -1,
            ):
                if nxt>=0 and authority[nxt] and not visited[nxt]:
                    visited[nxt]=1
                    q.append(nxt)
        size=len(pixels)
        covered=sum(1 for idx in pixels if predicted[idx])
        fraction=0.0 if foreground==0 else float(size)/float(foreground)
        recall=1.0 if size==0 else float(covered)/float(size)
        rows.append({
            "component_index":int(component_index),
            "pixel_count":int(size),
            "foreground_fraction":float(fraction),
            "covered_pixel_count":int(covered),
            "recall":float(recall),
            "eligible":bool(fraction+1e-15 >= floor),
        })
        component_index+=1
    eligible=[row for row in rows if row["eligible"]]
    return {
        "source_component_count":int(len(rows)),
        "eligible_component_count":int(len(eligible)),
        "minimum_foreground_fraction":float(floor),
        "minimum_eligible_component_recall":float(min((row["recall"] for row in eligible),default=1.0)),
        "components":rows,
    }


def _interior_mask_8_neighbor(authority: bytes, *, width: int, height: int) -> bytes:
    """One-reference-pixel boundary band removal via Chebyshev-radius-1 erosion."""
    out = bytearray(int(width) * int(height))
    for y in range(1, int(height) - 1):
        for x in range(1, int(width) - 1):
            idx = y * int(width) + x
            if not authority[idx]:
                continue
            if all(
                authority[(y + dy) * int(width) + (x + dx)]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ):
                out[idx] = 1
    return bytes(out)


def coverage_metrics(authority: bytes, predicted: bytes, *, width: int, height: int) -> dict:
    if len(authority) != int(width) * int(height) or len(predicted) != len(authority):
        raise QualificationError("G5_COVERAGE_MASK_SIZE_MISMATCH")
    foreground = sum(authority)
    predicted_count = sum(predicted)
    inside = sum(1 for a, p in zip(authority, predicted) if a and p)
    recall = 1.0 if foreground == 0 else float(inside) / float(foreground)
    precision = 1.0 if predicted_count == 0 else float(inside) / float(predicted_count)

    uncovered = bytes(1 if a and not p else 0 for a, p in zip(authority, predicted))
    uncovered_sizes = _connected_component_sizes_4(
        uncovered, width=width, height=height
    )
    largest = int(uncovered_sizes[0]) if uncovered_sizes else 0
    largest_fraction = 0.0 if foreground == 0 else float(largest) / float(foreground)
    singleton_uncovered = int(sum(size for size in uncovered_sizes if size == 1))
    small_uncovered = int(sum(size for size in uncovered_sizes if size <= 4))
    singleton_uncovered_fraction = (
        0.0 if foreground == 0 else float(singleton_uncovered) / float(foreground)
    )
    small_uncovered_fraction = (
        0.0 if foreground == 0 else float(small_uncovered) / float(foreground)
    )

    excess = bytes(1 if p and not a else 0 for a, p in zip(authority, predicted))
    excess_sizes = _connected_component_sizes_4(
        excess, width=width, height=height
    )
    singleton_excess = int(sum(size for size in excess_sizes if size == 1))
    small_excess = int(sum(size for size in excess_sizes if size <= 4))
    singleton_excess_fraction = (
        0.0 if predicted_count == 0 else float(singleton_excess) / float(predicted_count)
    )
    small_excess_fraction = (
        0.0 if predicted_count == 0 else float(small_excess) / float(predicted_count)
    )

    interior = _interior_mask_8_neighbor(authority, width=width, height=height)
    interior_count = sum(interior)
    interior_uncovered = sum(1 for i, p in zip(interior, predicted) if i and not p)
    interior_fraction = 0.0 if interior_count == 0 else float(interior_uncovered) / float(interior_count)

    return {
        "foreground_pixel_count": int(foreground),
        "predicted_pixel_count": int(predicted_count),
        "inside_pixel_count": int(inside),
        "recall": float(recall),
        "precision": float(precision),
        "largest_coherent_hole_pixels": int(largest),
        "largest_coherent_hole_fraction": float(largest_fraction),
        "uncovered_component_count": int(len(uncovered_sizes)),
        "singleton_uncovered_pixel_count": singleton_uncovered,
        "singleton_uncovered_fraction": float(singleton_uncovered_fraction),
        "small_uncovered_le4_pixel_count": small_uncovered,
        "small_uncovered_le4_fraction": float(small_uncovered_fraction),
        "excess_component_count": int(len(excess_sizes)),
        "singleton_excess_pixel_count": singleton_excess,
        "singleton_excess_fraction": float(singleton_excess_fraction),
        "small_excess_le4_pixel_count": small_excess,
        "small_excess_le4_fraction": float(small_excess_fraction),
        "interior_foreground_pixel_count": int(interior_count),
        "interior_uncovered_pixel_count": int(interior_uncovered),
        "interior_uncovered_fraction": float(interior_fraction),
    }


def _product_vertex_id(vertex) -> str:
    value = getattr(vertex, "canonical_mesh_vertex_id", None)
    if value is None:
        value = getattr(vertex, "candidate_vertex_id", None)
    if not value:
        raise QualificationError("G5_PRODUCT_VERTEX_ID_MISSING")
    return str(value)


def rasterize_visible_face_pixel_counts(
    mesh,
    camera: CameraProjectionV3,
    *,
    positions=None,
    width: int,
    height: int,
    pixel_mask: bytes | None = None,
) -> tuple[int, ...]:
    """Exact z-visible face pixel counts under Runtime-v3/v4 fill semantics.

    When pixel_mask is supplied, only visible pixels whose mask byte is 1 count
    as source-observed evidence. Geometry still participates in the z-buffer
    regardless of the mask, so occlusion semantics remain exact.
    """
    n=int(width)*int(height)
    if pixel_mask is not None:
        if len(pixel_mask)!=n or any(x not in (0,1) for x in pixel_mask):
            raise QualificationError("VISIBLE_FACE_PIXEL_MASK_INVALID")
    vertex_ids=[_product_vertex_id(v) for v in mesh.vertices]
    if len(vertex_ids)!=len(set(vertex_ids)):
        raise QualificationError("VISIBLE_FACE_DUPLICATE_VERTEX_ID")
    xyz=np.asarray(
        [v.P for v in mesh.vertices] if positions is None else positions,
        dtype=np.float64,
    )
    if xyz.shape!=(len(vertex_ids),3) or not np.isfinite(xyz).all():
        raise QualificationError("VISIBLE_FACE_POSITION_MATRIX_INVALID")
    projected=project_points_xyz_v3(xyz,camera)
    by_id={
        vertex_ids[i]:(float(projected[i,0]),float(projected[i,1]),float(projected[i,2]))
        for i in range(len(vertex_ids))
    }
    depth=[float("inf")]*n
    owner=[-1]*n
    tie=[None]*n
    for face_index,face in enumerate(mesh.faces):
        if len(face)!=3 or any(vid not in by_id for vid in face):
            raise QualificationError("VISIBLE_FACE_TOPOLOGY_INVALID")
        a,b,c=(by_id[vid] for vid in face)
        area=_orient2d(a,b,float(c[0]),float(c[1]))
        if abs(area)<=1e-12:
            continue
        xs=(a[0],b[0],c[0]); ys=(a[1],b[1],c[1])
        minx=max(0,int(math.floor(min(xs)-0.5))); maxx=min(int(width)-1,int(math.ceil(max(xs)-0.5)))
        miny=max(0,int(math.floor(min(ys)-0.5))); maxy=min(int(height)-1,int(math.ceil(max(ys)-0.5)))
        face_key=tuple(sorted(map(str,face)))
        for y in range(miny,maxy+1):
            for x in range(minx,maxx+1):
                if not _covers_pixel_center(a,b,c,x,y):
                    continue
                px=float(x)+0.5; py=float(y)+0.5
                w0=_orient2d(b,c,px,py)/area
                w1=_orient2d(c,a,px,py)/area
                w2=_orient2d(a,b,px,py)/area
                z=w0*a[2]+w1*b[2]+w2*c[2]
                if not math.isfinite(z):
                    raise QualificationError("VISIBLE_FACE_DEPTH_NONFINITE")
                idx=y*int(width)+x
                if z < depth[idx]-1e-12 or (
                    abs(z-depth[idx])<=1e-12 and (tie[idx] is None or face_key<tie[idx])
                ):
                    depth[idx]=z; owner[idx]=int(face_index); tie[idx]=face_key
    counts=[0]*len(mesh.faces)
    for idx,face_index in enumerate(owner):
        if face_index<0:
            continue
        if pixel_mask is not None and not pixel_mask[idx]:
            continue
        counts[int(face_index)]+=1
    return tuple(map(int,counts))


def rasterize_visible_component_masks(
    mesh,
    camera: CameraProjectionV3,
    *,
    width: int,
    height: int,
) -> dict[str, bytes]:
    """Rasterize the whole canonical mesh once and return z-visible owner masks.

    The fill rule is the same half-integer/TOP_LEFT rule as Runtime-v3/v4. Depth is
    linearly interpolated from the same projected 3D triangle; smaller camera-forward
    Z wins. Exact-depth ties use a stable lexical face key.
    """
    vertex_ids=[_product_vertex_id(vertex) for vertex in mesh.vertices]
    if len(vertex_ids)!=len(set(vertex_ids)):
        raise QualificationError("G5_DUPLICATE_PRODUCT_VERTEX_ID")
    projected=project_points_xyz_v3([tuple(map(float,v.P)) for v in mesh.vertices],camera)
    by_id={
        vertex_ids[i]:(float(projected[i,0]),float(projected[i,1]),float(projected[i,2]))
        for i in range(len(vertex_ids))
    }
    component_by_id={_product_vertex_id(v):str(v.component_id) for v in mesh.vertices}
    component_ids=set(component_by_id.values())
    depth=[float("inf")]*(int(width)*int(height))
    owner=[None]*(int(width)*int(height))
    tie=[None]*(int(width)*int(height))

    for face in mesh.faces:
        if len(face)!=3 or any(vid not in by_id for vid in face):
            raise QualificationError("G5_FACE_INVALID")
        components={component_by_id[vid] for vid in face}
        if len(components)!=1:
            raise QualificationError("G5_FACE_CROSSES_COMPONENT_BOUNDARY")
        component_id=next(iter(components))
        a,b,c=(by_id[vid] for vid in face)
        area=_orient2d(a,b,float(c[0]),float(c[1]))
        if abs(area)<=1e-12:
            continue
        xs=(a[0],b[0],c[0]); ys=(a[1],b[1],c[1])
        minx=max(0,int(math.floor(min(xs)-0.5))); maxx=min(int(width)-1,int(math.ceil(max(xs)-0.5)))
        miny=max(0,int(math.floor(min(ys)-0.5))); maxy=min(int(height)-1,int(math.ceil(max(ys)-0.5)))
        face_key=(component_id,tuple(map(str,face)))
        for y in range(miny,maxy+1):
            for x in range(minx,maxx+1):
                if not _covers_pixel_center(a,b,c,x,y):
                    continue
                px=float(x)+0.5; py=float(y)+0.5
                w0=_orient2d(b,c,px,py)/area
                w1=_orient2d(c,a,px,py)/area
                w2=_orient2d(a,b,px,py)/area
                z=w0*a[2]+w1*b[2]+w2*c[2]
                if not math.isfinite(z):
                    raise QualificationError("G5_DEPTH_NONFINITE")
                idx=y*int(width)+x
                if z < depth[idx]-1e-12 or (abs(z-depth[idx])<=1e-12 and (tie[idx] is None or face_key<tie[idx])):
                    depth[idx]=z; owner[idx]=component_id; tie[idx]=face_key

    masks={cid:bytearray(int(width)*int(height)) for cid in component_ids}
    for idx,cid in enumerate(owner):
        if cid is not None:
            masks[cid][idx]=1
    return {cid:bytes(mask) for cid,mask in masks.items()}


def derive_component_observation_rasters_v1(
    *,
    surface:RiggingSurfaceIR,
    partition,
    carrier_policy:ComponentCarrierPolicyIR,
    observation_set:QualifiedObservationSetIR,
    source_foreground_masks:Mapping[int,bytes],
    cameras:Iterable[CameraProjectionV3],
    raster_contract:ReferenceRasterContractV1=ReferenceRasterContractV1(),
)->tuple[ComponentObservationRasterIR,...]:
    validate_qualified_observation_set(observation_set)
    validate_component_carrier_policy(carrier_policy,partition)
    camera_by_view={int(c.view_index):c for c in cameras}
    if set(camera_by_view)!=set(range(8)):
        raise QualificationError("G5_DERIVED_COMPONENT_REQUIRES_EXACT_8_CAMERAS")
    obs={int(v.view_index):v for v in observation_set.views}
    component_by_surface={}
    component_by_id={c.component_id:c for c in partition.components}
    for component in partition.components:
        for sid in component.surface_ids:
            if sid in component_by_surface:
                raise QualificationError("G5_DERIVED_COMPONENT_SURFACE_OVERLAP")
            component_by_surface[sid]=component.component_id
    nodes={n.surface_id:n for n in surface.surface_nodes}
    if set(component_by_surface)!=set(nodes):
        raise QualificationError("G5_DERIVED_COMPONENT_SURFACE_ACCOUNTING_DRIFT")
    carrier={d.component_id:d.carrier_class for d in carrier_policy.decisions}
    rows=[]
    for view in range(8):
        authority=obs[view]
        camera=camera_by_view[view]
        if authority.camera_binding_hash!=camera_projection_binding_hash(camera):
            raise QualificationError("G5_DERIVED_COMPONENT_CAMERA_DRIFT")
        fg=bytes(source_foreground_masks[view])
        total=int(authority.width)*int(authority.height)
        if len(fg)!=total or any(x not in (0,1) for x in fg):
            raise QualificationError("G5_DERIVED_COMPONENT_FOREGROUND_INVALID")
        if mask_sha256(fg)!=authority.foreground_mask_sha256:
            raise QualificationError("G5_DERIVED_COMPONENT_FOREGROUND_HASH_DRIFT")
        seeds=[]
        occupied={}
        for node in surface.surface_nodes:
            matches=[tuple(map(float,xy)) for vi,xy in node.raster_bindings if int(vi)==view]
            if len(matches)>1:
                raise QualificationError("G5_DERIVED_COMPONENT_DUPLICATE_RASTER_BINDING")
            if not matches:
                continue
            cid=component_by_surface[node.surface_id]
            xy=matches[0]
            key=(round(xy[0],9),round(xy[1],9))
            prior=occupied.get(key)
            if prior is not None and prior!=cid:
                raise QualificationError("G5_DERIVED_COMPONENT_SEED_COLLISION")
            occupied[key]=cid
            seeds.append((xy,cid,node.surface_id))
        if sum(fg)>0 and not seeds:
            raise QualificationError("G5_DERIVED_COMPONENT_FOREGROUND_WITHOUT_SEEDS")
        masks={cid:bytearray(total) for cid in component_by_id}
        if seeds:
            seeds=sorted(seeds,key=lambda x:(x[1],x[2],x[0][0],x[0][1]))
            seed_xy=np.asarray([s[0] for s in seeds],dtype=np.float64)
            fg_indices=np.flatnonzero(np.frombuffer(fg,dtype=np.uint8))
            pts=np.column_stack((fg_indices%int(authority.width),fg_indices//int(authority.width))).astype(np.float64)
            d,idx=cKDTree(seed_xy).query(pts,k=1,workers=-1)
            for pix,seed_i in zip(fg_indices,np.asarray(idx).reshape(-1)):
                cid=seeds[int(seed_i)][1]
                masks[cid][int(pix)]=1
        seed_hash=content_sha256({
            "schema":"RealSaS.ComponentObservationSeedSet.v1",
            "view_index":view,
            "surface":surface.geometry_lineage_hash,
            "partition":partition.partition_lineage_hash,
            "seeds":[(cid,sid,xy) for xy,cid,sid in seeds],
        })
        for cid in sorted(component_by_id):
            raw=bytes(masks[cid])
            rows.append(ComponentObservationRasterIR(
                view_index=view,component_id=cid,carrier_class=carrier[cid],
                partition_binding_hash=partition.partition_lineage_hash,
                carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
                component_surface_set_hash=component_surface_set_hash(component_by_id[cid]),
                width=int(authority.width),height=int(authority.height),
                mask_bytes=raw,mask_sha256=mask_sha256(raw),
                source_observation_hash=authority.source_observation_hash,
                camera_binding_hash=authority.camera_binding_hash,
                raster_contract_hash=raster_contract.contract_hash,
                metadata={
                    "derivation":"SOURCE_FOREGROUND_NEAREST_QUALIFIED_S_SEED_V1",
                    "seed_set_hash":seed_hash,
                    "external_component_mask_used":False,
                    "categorical_recognition_used":False,
                },
            ))
    by={(r.view_index,r.component_id):r for r in rows}
    validate_source_component_partition(
        observations_by_key=by,
        component_ids=set(component_by_id),
        observation_set=observation_set,
        source_foreground_masks=source_foreground_masks,
        cameras=camera_by_view,
    )
    return tuple(rows)


def validate_source_component_partition(
    *,
    observations_by_key: Mapping[tuple[int,str], ComponentObservationRasterIR],
    component_ids: set[str],
    observation_set: QualifiedObservationSetIR,
    source_foreground_masks: Mapping[int, bytes],
    cameras: Mapping[int, CameraProjectionV3],
) -> None:
    validate_qualified_observation_set(observation_set)
    obs_view={int(row.view_index):row for row in observation_set.views}
    if set(obs_view)!=set(range(8)) or set(map(int,source_foreground_masks.keys()))!=set(range(8)):
        raise QualificationError("G5_SOURCE_FOREGROUND_VIEW_SET_INCOMPLETE")
    for view in range(8):
        authority=obs_view[view]
        camera=cameras[view]
        camera_hash=camera_projection_binding_hash(camera)
        if authority.camera_binding_hash!=camera_hash:
            raise QualificationError("G5_SOURCE_CAMERA_AUTHORITY_MISMATCH")
        if int(authority.width)!=int(camera.resolution) or int(authority.height)!=int(camera.resolution):
            raise QualificationError("G5_SOURCE_FOREGROUND_DIMENSION_MISMATCH")
        foreground=bytes(source_foreground_masks[view])
        n=int(authority.width)*int(authority.height)
        if len(foreground)!=n or any(x not in (0,1) for x in foreground):
            raise QualificationError("G5_SOURCE_FOREGROUND_MASK_INVALID")
        if mask_sha256(foreground)!=authority.foreground_mask_sha256:
            raise QualificationError("G5_SOURCE_FOREGROUND_HASH_MISMATCH")

        counts=[0]*n
        for component_id in component_ids:
            row=observations_by_key[(view,component_id)]
            if row.source_observation_hash!=authority.source_observation_hash:
                raise QualificationError("G5_SOURCE_OBSERVATION_HASH_MISMATCH")
            if row.camera_binding_hash!=authority.camera_binding_hash:
                raise QualificationError("G5_COMPONENT_CAMERA_AUTHORITY_MISMATCH")
            for i,value in enumerate(row.mask_bytes):
                counts[i]+=int(value)
        for fg,count in zip(foreground,counts):
            if fg:
                if count!=1:
                    raise QualificationError("G5_COMPONENT_MASKS_DO_NOT_PARTITION_SOURCE_FOREGROUND")
            elif count!=0:
                raise QualificationError("G5_COMPONENT_MASK_OUTSIDE_SOURCE_FOREGROUND")


def _mesh_component_triangles(mesh, camera: CameraProjectionV3, *, component_id: str):
    vertex_ids = [_product_vertex_id(vertex) for vertex in mesh.vertices]
    xyz = [tuple(map(float, vertex.P)) for vertex in mesh.vertices]
    projected = project_points_xyz_v3(xyz, camera)
    by_id = {
        vertex_ids[i]: (float(projected[i, 0]), float(projected[i, 1]))
        for i in range(len(vertex_ids))
    }
    component_by_id = {
        _product_vertex_id(vertex): vertex.component_id
        for vertex in mesh.vertices
    }

    triangles = []
    for face in mesh.faces:
        components = {component_by_id[vid] for vid in face}
        if len(components) != 1:
            raise QualificationError("G5_FACE_CROSSES_COMPONENT_BOUNDARY")
        if next(iter(components)) != component_id:
            continue
        triangles.append(tuple(by_id[vid] for vid in face))
    if not triangles:
        raise QualificationError("G5_COMPONENT_HAS_NO_RASTERIZABLE_FACE")
    return tuple(triangles)


def build_g5_coverage_matrix(
    mesh,
    *,
    surface: RiggingSurfaceIR,
    partition,
    carrier_policy: ComponentCarrierPolicyIR,
    mesh_policy: MeshQualificationPolicyIR,
    observations: Iterable[ComponentObservationRasterIR],
    cameras: Iterable[CameraProjectionV3],
    observation_set: QualifiedObservationSetIR,
    source_foreground_masks: Mapping[int, bytes],
    raster_contract: ReferenceRasterContractV1 = ReferenceRasterContractV1(),
) -> tuple[dict, ...]:
    validate_component_carrier_policy(carrier_policy, partition)
    validate_mesh_qualification_policy(mesh_policy)
    if raster_contract.pixel_center != "HALF_INTEGER_CENTER" or raster_contract.triangle_fill_rule != "TOP_LEFT":
        raise QualificationError("G5_REFERENCE_RASTER_CONTRACT_UNSUPPORTED")
    raster_hash = raster_contract.contract_hash

    component_ids = {component.component_id for component in partition.components}
    carrier_by_component = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    thresholds = {row.carrier_class: row for row in mesh_policy.coverage_thresholds}
    camera_by_view = {int(camera.view_index): camera for camera in cameras}
    if set(camera_by_view) != set(range(8)):
        raise QualificationError("G5_REQUIRES_EXACT_8_CAMERAS")
    if len(camera_by_view) != 8:
        raise QualificationError("G5_DUPLICATE_CAMERA_VIEW")

    by_key = {}
    for observation in observations:
        validate_component_observation(observation, partition=partition, carrier_policy=carrier_policy)
        key = (int(observation.view_index), observation.component_id)
        if key in by_key:
            raise QualificationError("G5_DUPLICATE_COMPONENT_OBSERVATION")
        if observation.component_id not in component_ids:
            raise QualificationError("G5_UNKNOWN_COMPONENT_OBSERVATION")
        if observation.carrier_class != carrier_by_component[observation.component_id]:
            raise QualificationError("G5_OBSERVATION_CARRIER_POLICY_DRIFT")
        if observation.raster_contract_hash != raster_hash:
            raise QualificationError("G5_RASTER_CONTRACT_HASH_MISMATCH")
        camera = camera_by_view[int(observation.view_index)]
        if int(camera.resolution) != int(observation.width) or int(camera.resolution) != int(observation.height):
            raise QualificationError("G5_CAMERA_OBSERVATION_DIMENSION_MISMATCH")
        if observation.camera_binding_hash != camera_projection_binding_hash(camera):
            raise QualificationError("G5_CAMERA_BINDING_HASH_MISMATCH")
        by_key[key] = observation

    expected = {(view, cid) for view in range(8) for cid in component_ids}
    if set(by_key) != expected:
        raise QualificationError("G5_COMPONENT_OBSERVATION_MATRIX_INCOMPLETE")

    validate_source_component_partition(
        observations_by_key=by_key,
        component_ids=component_ids,
        observation_set=observation_set,
        source_foreground_masks=source_foreground_masks,
        cameras=camera_by_view,
    )

    predicted_by_view={}
    for view in range(8):
        camera=camera_by_view[view]
        predicted_by_view[view]=rasterize_visible_component_masks(
            mesh,camera,width=int(camera.resolution),height=int(camera.resolution)
        )

    rows = []
    for view, component_id in sorted(expected):
        observation = by_key[(view, component_id)]
        predicted = predicted_by_view[view].get(
            component_id,
            bytes(int(observation.width)*int(observation.height)),
        )
        metrics = coverage_metrics(
            observation.mask_bytes,
            predicted,
            width=observation.width,
            height=observation.height,
        )
        threshold = thresholds[observation.carrier_class]
        passed = (
            metrics["recall"] >= threshold.min_recall
            and metrics["precision"] >= threshold.min_precision
            and metrics["largest_coherent_hole_fraction"] <= threshold.max_largest_coherent_hole_fraction
            and metrics["interior_uncovered_fraction"] <= threshold.max_interior_uncovered_fraction
        )
        rows.append({
            "view_index": int(view),
            "component_id": component_id,
            "carrier_class": observation.carrier_class,
            "partition_binding_hash": observation.partition_binding_hash,
            "carrier_policy_binding_hash": observation.carrier_policy_binding_hash,
            "component_surface_set_hash": observation.component_surface_set_hash,
            **metrics,
            "status": "PASS" if passed else "FAIL",
            "source_mask_sha256": observation.mask_sha256,
            "source_observation_hash": observation.source_observation_hash,
            "camera_binding_hash": observation.camera_binding_hash,
            "raster_contract_hash": raster_hash,
            "observation_set_hash": observation_set.observation_set_hash,
            "source_foreground_mask_sha256": mask_sha256(bytes(source_foreground_masks[view])),
            "visibility_rule": "CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1",
        })
    return tuple(rows)


def g5_coverage_evidence_hash(rows: Iterable[dict]) -> str:
    return content_sha256({
        "schema": "RealSaS.G5CoverageEvidence.v1",
        "rows": tuple(rows),
    })
