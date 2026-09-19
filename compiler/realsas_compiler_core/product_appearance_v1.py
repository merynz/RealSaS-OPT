from __future__ import annotations

"""Source-authoritative appearance over the single canonical QualifiedMeshIR.

The product mesh never gains per-view topology. Each target view receives one
AppearanceBindingIR over the same canonical faces. Every face uses one observed donor
view uniformly across all three corners; there is no cross-view color blending.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

from .hashing import content_sha256
from .observation_authority_v1 import QualifiedObservationSetIR, validate_qualified_observation_set
from .types import QualificationError
from .v4 import build_appearance_binding, validate_appearance_binding
from .v4_types import AppearanceBindingIR, AppearanceCornerBinding

Json=dict[str,Any]
_PIXEL_CENTER_COORDINATE_SYSTEM="PIXEL_CENTER_XY"


@dataclass(frozen=True)
class QualifiedAppearanceSetIR:
    bindings:tuple[AppearanceBindingIR,...]
    surface_binding_hash:str
    mesh_binding_hash:str
    observation_set_binding_hash:str
    appearance_set_hash:str
    schema_version:str="RealSaS.QualifiedAppearanceSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def qualified_appearance_set_hash(value:QualifiedAppearanceSetIR)->str:
    payload=value.to_dict()
    payload.pop("appearance_set_hash",None)
    return content_sha256(payload)


def _node_raster(node,view_index:int):
    if int(view_index) not in set(map(int,node.support_views)):
        return None
    rows=[tuple(map(float,xy)) for vi,xy in node.raster_bindings if int(vi)==int(view_index)]
    if len(rows)>1:
        raise QualificationError("PRODUCT_APPEARANCE_DUPLICATE_NODE_RASTER_BINDING")
    return rows[0] if rows else None


def _support_for_vertex(vertex,nodes):
    support=[]
    total=0.0
    for sid,coeff in vertex.support_binding.coefficients:
        if sid not in nodes:
            raise QualificationError("PRODUCT_APPEARANCE_UNKNOWN_SURFACE_SUPPORT")
        c=float(coeff)
        if not math.isfinite(c) or c<0.0:
            raise QualificationError("PRODUCT_APPEARANCE_SUPPORT_COEFFICIENT_INVALID")
        if c<=1e-15:
            continue
        total+=c
        support.append((nodes[sid],c))
    if not support or abs(total-1.0)>1e-9:
        raise QualificationError("PRODUCT_APPEARANCE_SUPPORT_SIMPLEX_INVALID")
    return tuple(support)


def _face_donor_candidates(face,vertices,nodes)->tuple[int,...]:
    required=[]
    for vid in face:
        vertex=vertices.get(str(vid))
        if vertex is None:
            raise QualificationError("PRODUCT_APPEARANCE_FACE_UNKNOWN_VERTEX")
        required.extend(node for node,_ in _support_for_vertex(vertex,nodes))
    return tuple(
        view
        for view in range(8)
        if all(_node_raster(node,view) is not None for node in required)
    )


def _donor_xy(vertex,nodes,donor:int):
    x=y=0.0
    for node,coeff in _support_for_vertex(vertex,nodes):
        xy=_node_raster(node,donor)
        if xy is None:
            raise QualificationError("PRODUCT_APPEARANCE_DONOR_SUPPORT_MISSING")
        x+=coeff*float(xy[0]); y+=coeff*float(xy[1])
    return (float(x),float(y))


def _pixel_center_to_uv(xy,*,width:int,height:int):
    x,y=map(float,xy)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise QualificationError("PRODUCT_APPEARANCE_DONOR_RASTER_NONFINITE")
    if x < -0.5-1e-9 or x > float(width)-0.5+1e-9 or y < -0.5-1e-9 or y > float(height)-0.5+1e-9:
        raise QualificationError("PRODUCT_APPEARANCE_DONOR_RASTER_OUT_OF_FRAME")
    return ((x+0.5)/float(width),1.0-(y+0.5)/float(height))


def _binding_for_view(*,surface,mesh,observation_set,target_view_index:int)->AppearanceBindingIR:
    nodes={str(node.surface_id):node for node in surface.surface_nodes}
    vertices={str(vertex.canonical_mesh_vertex_id):vertex for vertex in mesh.vertices}
    obs={int(row.view_index):row for row in observation_set.views}
    target=obs[int(target_view_index)]
    corners=[]
    face_donors=[]
    for face_index,face in enumerate(mesh.faces):
        candidates=_face_donor_candidates(face,vertices,nodes)
        if not candidates:
            raise QualificationError(f"PRODUCT_APPEARANCE_FACE_HAS_NO_OBSERVED_DONOR:{face_index}")
        donor=int(target_view_index) if int(target_view_index) in candidates else int(min(candidates))
        donor_obs=obs[donor]
        face_donors.append(donor)
        for corner_index,vid in enumerate(face):
            vertex=vertices[str(vid)]
            donor_xy=_donor_xy(vertex,nodes,donor)
            material_uv=_pixel_center_to_uv(
                donor_xy,width=int(donor_obs.width),height=int(donor_obs.height)
            )
            corners.append(AppearanceCornerBinding(
                face_index=int(face_index),
                corner_index=int(corner_index),
                material_uv=material_uv,
                donor_view_index=donor,
                donor_raster_xy=donor_xy,
                source_observation_hash=donor_obs.source_observation_hash,
                authority_class="OBSERVED_LOCAL" if donor==int(target_view_index) else "OBSERVED_CROSS_VIEW",
                completion_id="",
                confidence=1.0,
            ))
    source_rasters=tuple(
        (int(row.view_index),str(row.source_raster_sha256))
        for row in sorted(observation_set.views,key=lambda x:x.view_index)
    )
    return build_appearance_binding(
        target_view_index=int(target_view_index),
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash=target.camera_binding_hash,
        corner_bindings=tuple(corners),
        atlas_payload_hash=content_sha256({
            "schema":"RealSaS.SourceRasterSet.v1",
            "qualified_observation_set_hash":observation_set.observation_set_hash,
            "source_raster_sha256_by_view":source_rasters,
        }),
        metadata={
            "authority":"QUALIFIED_SOURCE_OBSERVATIONS_ONLY",
            "surface_binding_hash":surface.geometry_lineage_hash,
            "observation_set_binding_hash":observation_set.observation_set_hash,
            "source_raster_sha256_by_view":source_rasters,
            "face_donor_view_indices":tuple(face_donors),
            "face_uniform_donor_required":True,
            "cross_view_color_blending":False,
            "camera_refit":False,
            "source_mesh_uv_used":False,
            "unknown_completion_used":False,
            "raster_coordinate_system":_PIXEL_CENTER_COORDINATE_SYSTEM,
            "material_uv_convention":"NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1",
        },
    )


def validate_product_appearance_set(
    value:QualifiedAppearanceSetIR,
    *,
    surface,
    mesh,
    observation_set:QualifiedObservationSetIR,
)->None:
    validate_qualified_observation_set(observation_set)
    if value.surface_binding_hash!=surface.geometry_lineage_hash:
        raise QualificationError("PRODUCT_APPEARANCE_SET_SURFACE_LINEAGE_MISMATCH")
    if value.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("PRODUCT_APPEARANCE_SET_MESH_LINEAGE_MISMATCH")
    if value.observation_set_binding_hash!=observation_set.observation_set_hash:
        raise QualificationError("PRODUCT_APPEARANCE_SET_OBSERVATION_LINEAGE_MISMATCH")
    if str((surface.metadata or {}).get("raster_coordinate_system",""))!=_PIXEL_CENTER_COORDINATE_SYSTEM:
        raise QualificationError("PRODUCT_APPEARANCE_REQUIRES_PIXEL_CENTER_XY")
    rows=tuple(sorted(value.bindings,key=lambda x:x.target_view_index))
    if len(rows)!=8 or tuple(row.target_view_index for row in rows)!=tuple(range(8)):
        raise QualificationError("PRODUCT_APPEARANCE_SET_REQUIRES_EXACT_8_VIEWS")
    obs={int(row.view_index):row for row in observation_set.views}
    nodes={str(node.surface_id):node for node in surface.surface_nodes}
    vertices={str(vertex.canonical_mesh_vertex_id):vertex for vertex in mesh.vertices}
    expected_corner_keys={(fi,ci) for fi,face in enumerate(mesh.faces) for ci in range(len(face))}
    for binding in rows:
        validate_appearance_binding(binding)
        target=obs[binding.target_view_index]
        if binding.mesh_binding_hash!=mesh.mesh_lineage_hash:
            raise QualificationError("PRODUCT_APPEARANCE_BINDING_MESH_DRIFT")
        if binding.camera_binding_hash!=target.camera_binding_hash:
            raise QualificationError("PRODUCT_APPEARANCE_BINDING_CAMERA_DRIFT")
        by_key={(int(c.face_index),int(c.corner_index)):c for c in binding.corner_bindings}
        if set(by_key)!=expected_corner_keys:
            raise QualificationError("PRODUCT_APPEARANCE_CORNER_ACCOUNTING_INCOMPLETE")
        for fi,face in enumerate(mesh.faces):
            donors={int(by_key[(fi,ci)].donor_view_index) for ci in range(len(face))}
            if len(donors)!=1:
                raise QualificationError("PRODUCT_APPEARANCE_FACE_DONOR_NOT_UNIFORM")
            donor=next(iter(donors))
            candidates=_face_donor_candidates(face,vertices,nodes)
            expected_donor=binding.target_view_index if binding.target_view_index in candidates else (min(candidates) if candidates else None)
            if expected_donor is None or donor!=expected_donor:
                raise QualificationError("PRODUCT_APPEARANCE_FACE_DONOR_AUTHORITY_DRIFT")
            donor_obs=obs[donor]
            for ci,vid in enumerate(face):
                corner=by_key[(fi,ci)]
                if corner.source_observation_hash!=donor_obs.source_observation_hash:
                    raise QualificationError("PRODUCT_APPEARANCE_SOURCE_OBSERVATION_DRIFT")
                expected_xy=_donor_xy(vertices[str(vid)],nodes,donor)
                if any(abs(float(a)-float(b))>1e-12 for a,b in zip(corner.donor_raster_xy,expected_xy)):
                    raise QualificationError("PRODUCT_APPEARANCE_DONOR_RASTER_DRIFT")
                expected_uv=_pixel_center_to_uv(expected_xy,width=donor_obs.width,height=donor_obs.height)
                if any(abs(float(a)-float(b))>1e-12 for a,b in zip(corner.material_uv,expected_uv)):
                    raise QualificationError("PRODUCT_APPEARANCE_MATERIAL_UV_DRIFT")
    if value.appearance_set_hash!=qualified_appearance_set_hash(value):
        raise QualificationError("PRODUCT_APPEARANCE_SET_HASH_MISMATCH")


def build_product_appearance_set(*,surface,mesh,observation_set:QualifiedObservationSetIR)->QualifiedAppearanceSetIR:
    validate_qualified_observation_set(observation_set)
    if mesh.surface_binding_hash!=surface.geometry_lineage_hash:
        raise QualificationError("PRODUCT_APPEARANCE_MESH_SURFACE_LINEAGE_MISMATCH")
    if str((surface.metadata or {}).get("raster_coordinate_system",""))!=_PIXEL_CENTER_COORDINATE_SYSTEM:
        raise QualificationError("PRODUCT_APPEARANCE_REQUIRES_PIXEL_CENTER_XY")
    bindings=tuple(
        _binding_for_view(
            surface=surface,mesh=mesh,observation_set=observation_set,target_view_index=view
        )
        for view in range(8)
    )
    value=QualifiedAppearanceSetIR(
        bindings=bindings,
        surface_binding_hash=surface.geometry_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        observation_set_binding_hash=observation_set.observation_set_hash,
        appearance_set_hash="",
        metadata={
            "single_canonical_mesh":True,
            "face_uniform_donor":True,
            "completion_used":False,
            "source_raster_bytes_are_appearance_authority":True,
        },
    )
    value=replace(value,appearance_set_hash=qualified_appearance_set_hash(value))
    validate_product_appearance_set(value,surface=surface,mesh=mesh,observation_set=observation_set)
    return value
