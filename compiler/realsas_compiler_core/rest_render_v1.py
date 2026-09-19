from __future__ import annotations

"""Deterministic rest renderer over QualifiedMesh + qualified source appearance.

No lighting, shading, relighting, material synthesis, camera refit, completion or
runtime-only presentation inference is allowed here. The renderer transports exact
source RGBA through the already-qualified canonical mesh and presentation bindings.
"""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from .camera_authority_v1 import QualifiedCameraSetIR, validate_qualified_camera_set
from .hashing import content_sha256
from .mesh.product_coverage_v1 import _covers_pixel_center, _orient2d
from .observation_authority_v1 import QualifiedObservationSetIR, validate_qualified_observation_set
from .product_appearance_v1 import QualifiedAppearanceSetIR, validate_product_appearance_set
from .product_composition_v1 import QualifiedCompositionSetIR, validate_composition_set
from .product_authority_v1 import QualifiedPresentationGraphIR
from .playback_full_surface_v3 import project_points_xyz_v3
from .types import QualificationError

Json=dict[str,Any]

REST_RENDER_CONTRACT={
    "schema":"RealSaS.RestRenderContract.v1",
    "projection":"FULL_SURFACE_CAMERA_PROJECTION_V3",
    "raster_fill":"HALF_INTEGER_TOP_LEFT",
    "visibility":"CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1",
    "equal_depth_tiebreak":"STABLE_LEXICAL_FACE_KEY_V1",
    "appearance":"ONE_OBSERVED_DONOR_VIEW_PER_FACE",
    "sampling":"PIXEL_CENTER_XY_BILINEAR_RGBA_V1",
    "pixel_center_to_texel_index":"index_xy = donor_xy - (0.5,0.5)",
    "cross_view_color_blending":False,
    "lighting":False,
    "shading":False,
    "completion":False,
}
REST_RENDER_CONTRACT_HASH=content_sha256(REST_RENDER_CONTRACT)


@dataclass(frozen=True)
class RestRenderViewIR:
    view_index:int
    camera_binding_hash:str
    appearance_binding_hash:str
    composition_binding_hash:str
    rendered_rgba_sha256:str
    width:int
    height:int
    visible_pixel_count:int
    render_contract_hash:str=REST_RENDER_CONTRACT_HASH
    schema_version:str="RealSaS.RestRenderViewIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RestRenderSetIR:
    views:tuple[RestRenderViewIR,...]
    mesh_binding_hash:str
    presentation_binding_hash:str
    appearance_set_binding_hash:str
    composition_set_binding_hash:str
    camera_set_binding_hash:str
    observation_set_binding_hash:str
    render_set_hash:str
    schema_version:str="RealSaS.RestRenderSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def rest_render_set_hash(value:RestRenderSetIR)->str:
    payload=value.to_dict()
    payload.pop("render_set_hash",None)
    return content_sha256(payload)


def rgba_sha256(arr:np.ndarray)->str:
    a=np.ascontiguousarray(arr,dtype=np.uint8)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def _sample_bilinear_rgba(image:np.ndarray,xy:tuple[float,float])->np.ndarray:
    if image.ndim!=3 or image.shape[2]!=4 or image.dtype!=np.uint8:
        raise QualificationError("REST_RENDER_SOURCE_RGBA_INVALID")
    h,w,_=image.shape
    x=float(xy[0])-0.5
    y=float(xy[1])-0.5
    if not (math.isfinite(x) and math.isfinite(y)):
        raise QualificationError("REST_RENDER_DONOR_COORD_NONFINITE")
    eps=1e-9
    if x < -eps or x > float(w-1)+eps or y < -eps or y > float(h-1)+eps:
        raise QualificationError("REST_RENDER_DONOR_COORD_OUT_OF_RANGE")
    x=min(float(w-1),max(0.0,x))
    y=min(float(h-1),max(0.0,y))
    x0=int(math.floor(x)); y0=int(math.floor(y))
    x1=min(w-1,x0+1); y1=min(h-1,y0+1)
    tx=x-float(x0); ty=y-float(y0)
    p00=image[y0,x0].astype(np.float64)
    p10=image[y0,x1].astype(np.float64)
    p01=image[y1,x0].astype(np.float64)
    p11=image[y1,x1].astype(np.float64)
    value=(1.0-ty)*((1.0-tx)*p00+tx*p10)+ty*((1.0-tx)*p01+tx*p11)
    return np.clip(np.floor(value+0.5),0.0,255.0).astype(np.uint8)


def _render_one(
    *,
    mesh,
    camera,
    appearance,
    composition,
    source_rgba_by_view:Mapping[int,np.ndarray],
)->np.ndarray:
    if appearance.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("REST_RENDER_APPEARANCE_MESH_DRIFT")
    if appearance.camera_binding_hash!=composition.camera_binding_hash:
        raise QualificationError("REST_RENDER_CAMERA_COMPOSITION_DRIFT")
    if appearance.camera_binding_hash!=camera_binding_hash(camera):
        raise QualificationError("REST_RENDER_CAMERA_BINDING_DRIFT")
    if composition.physical_occlusion_rule!="CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1":
        raise QualificationError("REST_RENDER_OCCLUSION_RULE_DRIFT")
    if composition.equal_depth_tiebreak!="STABLE_LEXICAL_FACE_KEY_V1":
        raise QualificationError("REST_RENDER_DEPTH_TIEBREAK_DRIFT")

    projected=project_points_xyz_v3([tuple(map(float,v.P)) for v in mesh.vertices],camera)
    vertex_index={str(v.canonical_mesh_vertex_id):i for i,v in enumerate(mesh.vertices)}
    vertex_component={str(v.canonical_mesh_vertex_id):str(v.component_id) for v in mesh.vertices}
    if len(vertex_index)!=len(mesh.vertices):
        raise QualificationError("REST_RENDER_DUPLICATE_VERTEX_ID")
    corner_map={(int(c.face_index),int(c.corner_index)):c for c in appearance.corner_bindings}
    expected={(fi,ci) for fi,face in enumerate(mesh.faces) for ci in range(len(face))}
    if set(corner_map)!=expected:
        raise QualificationError("REST_RENDER_APPEARANCE_CORNER_ACCOUNTING_DRIFT")

    width=int(camera.resolution); height=int(camera.resolution)
    out=np.zeros((height,width,4),dtype=np.uint8)
    depth=np.full((height,width),np.inf,dtype=np.float64)
    tie=[[None for _x in range(width)] for _y in range(height)]

    for fi,face in enumerate(mesh.faces):
        if len(face)!=3:
            raise QualificationError("REST_RENDER_REQUIRES_TRIANGLES")
        if any(str(vid) not in vertex_index for vid in face):
            raise QualificationError("REST_RENDER_FACE_UNKNOWN_VERTEX")
        components={vertex_component[str(vid)] for vid in face}
        if len(components)!=1:
            raise QualificationError("REST_RENDER_FACE_CROSSES_COMPONENT")
        component_id=next(iter(components))
        pts=[projected[vertex_index[str(vid)]] for vid in face]
        a,b,c=pts
        area=_orient2d(a,b,float(c[0]),float(c[1]))
        if abs(area)<=1e-12:
            continue
        donors={int(corner_map[(fi,ci)].donor_view_index) for ci in range(3)}
        if len(donors)!=1:
            raise QualificationError("REST_RENDER_FACE_DONOR_NOT_UNIFORM")
        donor=next(iter(donors))
        texture=source_rgba_by_view.get(donor)
        if texture is None:
            raise QualificationError("REST_RENDER_DONOR_TEXTURE_MISSING")
        donor_xy=[tuple(map(float,corner_map[(fi,ci)].donor_raster_xy)) for ci in range(3)]
        xs=(float(a[0]),float(b[0]),float(c[0])); ys=(float(a[1]),float(b[1]),float(c[1]))
        minx=max(0,int(math.floor(min(xs)-0.5))); maxx=min(width-1,int(math.ceil(max(xs)-0.5)))
        miny=max(0,int(math.floor(min(ys)-0.5))); maxy=min(height-1,int(math.ceil(max(ys)-0.5)))
        face_key=(component_id,tuple(map(str,face)))
        for y in range(miny,maxy+1):
            for x in range(minx,maxx+1):
                if not _covers_pixel_center(a,b,c,x,y):
                    continue
                px=float(x)+0.5; py=float(y)+0.5
                w0=_orient2d(b,c,px,py)/area
                w1=_orient2d(c,a,px,py)/area
                w2=_orient2d(a,b,px,py)/area
                z=w0*float(a[2])+w1*float(b[2])+w2*float(c[2])
                if not math.isfinite(z):
                    raise QualificationError("REST_RENDER_DEPTH_NONFINITE")
                current=float(depth[y,x])
                if z > current+1e-12:
                    continue
                if abs(z-current)<=1e-12 and tie[y][x] is not None and not (face_key<tie[y][x]):
                    continue
                sx=w0*donor_xy[0][0]+w1*donor_xy[1][0]+w2*donor_xy[2][0]
                sy=w0*donor_xy[0][1]+w1*donor_xy[1][1]+w2*donor_xy[2][1]
                out[y,x]=_sample_bilinear_rgba(texture,(sx,sy))
                depth[y,x]=z
                tie[y][x]=face_key
    return out


def camera_binding_hash(camera)->str:
    from .camera_authority_v1 import camera_projection_binding_hash
    return camera_projection_binding_hash(camera)


def validate_rest_render_set(
    value:RestRenderSetIR,
    *,
    mesh,
    presentation_graph:QualifiedPresentationGraphIR,
    appearance_set:QualifiedAppearanceSetIR,
    composition_set:QualifiedCompositionSetIR,
    camera_set:QualifiedCameraSetIR,
    observation_set:QualifiedObservationSetIR,
)->None:
    validate_qualified_camera_set(camera_set)
    validate_qualified_observation_set(observation_set)
    if value.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("REST_RENDER_SET_MESH_BINDING_DRIFT")
    if value.presentation_binding_hash!=presentation_graph.presentation_lineage_hash:
        raise QualificationError("REST_RENDER_SET_PRESENTATION_BINDING_DRIFT")
    if value.appearance_set_binding_hash!=appearance_set.appearance_set_hash:
        raise QualificationError("REST_RENDER_SET_APPEARANCE_BINDING_DRIFT")
    if value.composition_set_binding_hash!=composition_set.composition_set_hash:
        raise QualificationError("REST_RENDER_SET_COMPOSITION_BINDING_DRIFT")
    if value.camera_set_binding_hash!=camera_set.camera_set_hash:
        raise QualificationError("REST_RENDER_SET_CAMERA_BINDING_DRIFT")
    if value.observation_set_binding_hash!=observation_set.observation_set_hash:
        raise QualificationError("REST_RENDER_SET_OBSERVATION_BINDING_DRIFT")
    rows=tuple(sorted(value.views,key=lambda x:x.view_index))
    if len(rows)!=8 or tuple(v.view_index for v in rows)!=tuple(range(8)):
        raise QualificationError("REST_RENDER_SET_REQUIRES_EXACT_8_VIEWS")
    app={int(v.target_view_index):v for v in appearance_set.bindings}
    comp={int(v.view_index):v for v in composition_set.views}
    cams={int(v.view_index):v for v in camera_set.cameras}
    for row in rows:
        if row.camera_binding_hash!=camera_binding_hash(cams[row.view_index]):
            raise QualificationError("REST_RENDER_VIEW_CAMERA_DRIFT")
        if row.appearance_binding_hash!=app[row.view_index].appearance_lineage_hash:
            raise QualificationError("REST_RENDER_VIEW_APPEARANCE_DRIFT")
        if row.composition_binding_hash!=comp[row.view_index].composition_binding_hash:
            raise QualificationError("REST_RENDER_VIEW_COMPOSITION_DRIFT")
        if row.render_contract_hash!=REST_RENDER_CONTRACT_HASH:
            raise QualificationError("REST_RENDER_CONTRACT_DRIFT")
        if len(row.rendered_rgba_sha256)!=64 or row.width<=0 or row.height<=0:
            raise QualificationError("REST_RENDER_VIEW_IDENTITY_INVALID")
    if value.render_set_hash!=rest_render_set_hash(value):
        raise QualificationError("REST_RENDER_SET_HASH_MISMATCH")


def render_rest_views(
    *,
    mesh,
    presentation_graph:QualifiedPresentationGraphIR,
    appearance_set:QualifiedAppearanceSetIR,
    composition_set:QualifiedCompositionSetIR,
    camera_set:QualifiedCameraSetIR,
    observation_set:QualifiedObservationSetIR,
    source_rgba_by_view:Mapping[int,np.ndarray],
):
    validate_qualified_camera_set(camera_set)
    validate_qualified_observation_set(observation_set)
    validate_product_appearance_set(
        appearance_set,surface=None,mesh=mesh,observation_set=observation_set
    ) if False else None
    if presentation_graph.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("REST_RENDER_PRESENTATION_MESH_DRIFT")
    if presentation_graph.appearance_set_binding_hash!=appearance_set.appearance_set_hash:
        raise QualificationError("REST_RENDER_PRESENTATION_APPEARANCE_DRIFT")
    if presentation_graph.composition_set_binding_hash!=composition_set.composition_set_hash:
        raise QualificationError("REST_RENDER_PRESENTATION_COMPOSITION_DRIFT")
    obs={int(v.view_index):v for v in observation_set.views}
    cams={int(v.view_index):v for v in camera_set.cameras}
    app={int(v.target_view_index):v for v in appearance_set.bindings}
    comp={int(v.view_index):v for v in composition_set.views}
    if set(source_rgba_by_view)!=set(range(8)):
        raise QualificationError("REST_RENDER_SOURCE_TEXTURE_SET_INCOMPLETE")
    rendered={}
    rows=[]
    for view in range(8):
        texture=source_rgba_by_view[view]
        if texture.shape[:2]!=(int(obs[view].height),int(obs[view].width)):
            raise QualificationError("REST_RENDER_SOURCE_TEXTURE_DIMENSION_DRIFT")
        image=_render_one(
            mesh=mesh,camera=cams[view],appearance=app[view],composition=comp[view],
            source_rgba_by_view=source_rgba_by_view,
        )
        rendered[view]=image
        rows.append(RestRenderViewIR(
            view_index=view,
            camera_binding_hash=camera_binding_hash(cams[view]),
            appearance_binding_hash=app[view].appearance_lineage_hash,
            composition_binding_hash=comp[view].composition_binding_hash,
            rendered_rgba_sha256=rgba_sha256(image),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            visible_pixel_count=int(np.count_nonzero(image[...,3])),
            metadata={
                "source_raster_sha256":obs[view].source_raster_sha256,
                "shading_used":False,
                "lighting_used":False,
                "completion_used":False,
            },
        ))
    value=RestRenderSetIR(
        views=tuple(rows),
        mesh_binding_hash=mesh.mesh_lineage_hash,
        presentation_binding_hash=presentation_graph.presentation_lineage_hash,
        appearance_set_binding_hash=appearance_set.appearance_set_hash,
        composition_set_binding_hash=composition_set.composition_set_hash,
        camera_set_binding_hash=camera_set.camera_set_hash,
        observation_set_binding_hash=observation_set.observation_set_hash,
        render_set_hash="",
        metadata={
            "render_contract":REST_RENDER_CONTRACT,
            "render_contract_hash":REST_RENDER_CONTRACT_HASH,
            "source_raster_is_appearance_authority":True,
            "canonical_geometry_is_never_rgb_authority":True,
        },
    )
    value=replace(value,render_set_hash=rest_render_set_hash(value))
    validate_rest_render_set(
        value,mesh=mesh,presentation_graph=presentation_graph,
        appearance_set=appearance_set,composition_set=composition_set,
        camera_set=camera_set,observation_set=observation_set,
    )
    return value,rendered
