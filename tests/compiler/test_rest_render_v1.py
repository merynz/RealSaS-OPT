from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from compiler.realsas_compiler_core.camera_authority_v1 import build_qualified_camera_set
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,build_qualified_observation_set,
)
from compiler.realsas_compiler_core.product_authority_v1 import PresentationViewOverlayIR
from compiler.realsas_compiler_core.rest_render_v1 import render_rest_views, _sample_bilinear_rgba
from compiler.realsas_compiler_core.types import SurfaceSupportBinding
from compiler.realsas_compiler_core.v4 import build_appearance_binding
from compiler.realsas_compiler_core.v4_types import AppearanceCornerBinding
from compiler.realsas_compiler_core.product_appearance_v1 import QualifiedAppearanceSetIR,qualified_appearance_set_hash
from compiler.realsas_compiler_core.product_composition_v1 import (
    QualifiedCompositionSetIR,QualifiedCompositionViewIR,composition_set_hash,composition_view_hash,
)
from dataclasses import replace


def _fixture():
    cameras=build_qualified_camera_set(tuple(
        {
            "schema_version":"RealSaS.FullSurfaceCameraProjection.v3",
            "view_id":f"V{i}","view_index":i,
            "origin":[0.0,0.0,-2.0-float(i)],
            "right":[1.0,0.0,0.0],"screen_up":[0.0,1.0,0.0],"forward":[0.0,0.0,1.0],
            "half_extent":2.0,"resolution":8,
        }
        for i in range(8)
    ),source_bundle_sha256="a"*64)
    observations=build_qualified_observation_set(tuple(
        QualifiedObservationViewIR(
            i,8,8,f"obs-{i}",f"{i+16:064x}",f"{i+32:064x}",
            cameras.camera_binding_hashes[i],"PASS",(f"e-{i}",)
        ) for i in range(8)
    ))
    verts=tuple(
        SimpleNamespace(
            canonical_mesh_vertex_id=f"v{i}",
            component_id="c0",
            P=p,
            support_binding=SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{i}",1.0),)),
        )
        for i,p in enumerate(((-1.5,-1.5,0.0),(1.5,-1.5,0.0),(-1.5,1.5,0.0)))
    )
    mesh=SimpleNamespace(mesh_lineage_hash="mesh-hash",vertices=verts,faces=(("v0","v1","v2"),))
    apps=[]
    for view in range(8):
        corners=(
            AppearanceCornerBinding(0,0,(0,0),view,(1.5,1.5),f"obs-{view}","OBSERVED_LOCAL"),
            AppearanceCornerBinding(0,1,(0,0),view,(6.5,1.5),f"obs-{view}","OBSERVED_LOCAL"),
            AppearanceCornerBinding(0,2,(0,0),view,(1.5,6.5),f"obs-{view}","OBSERVED_LOCAL"),
        )
        apps.append(build_appearance_binding(
            target_view_index=view,mesh_binding_hash="mesh-hash",
            camera_binding_hash=cameras.camera_binding_hashes[view],
            corner_bindings=corners,atlas_payload_hash=f"atlas-{view}",
        ))
    appset=QualifiedAppearanceSetIR(tuple(apps),"surface","mesh-hash",observations.observation_set_hash,"")
    appset=replace(appset,appearance_set_hash=qualified_appearance_set_hash(appset))
    comp_rows=[]
    for view in range(8):
        row=QualifiedCompositionViewIR(
            view,cameras.camera_binding_hashes[view],("slot0",),
            "CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1","STABLE_LEXICAL_FACE_KEY_V1",
            "UI_SETUP_ONLY__NOT_PHYSICAL_OCCLUSION","g5","",
        )
        comp_rows.append(replace(row,composition_binding_hash=composition_view_hash(row)))
    compset=QualifiedCompositionSetIR(tuple(comp_rows),"state","mesh-hash",observations.observation_set_hash,"structure","")
    compset=replace(compset,composition_set_hash=composition_set_hash(compset))
    graph=SimpleNamespace(
        mesh_binding_hash="mesh-hash",
        presentation_lineage_hash="presentation-hash",
        appearance_set_binding_hash=appset.appearance_set_hash,
        composition_set_binding_hash=compset.composition_set_hash,
    )
    textures={}
    for view in range(8):
        img=np.zeros((8,8,4),dtype=np.uint8)
        for y in range(8):
            for x in range(8):
                img[y,x]=[x*20,y*20,view*20,255]
        textures[view]=img
    return cameras,observations,mesh,appset,compset,graph,textures


def test_rest_renderer_transports_source_rgba_without_shading():
    cameras,obs,mesh,app,comp,graph,textures=_fixture()
    result,images=render_rest_views(
        mesh=mesh,presentation_graph=graph,appearance_set=app,composition_set=comp,
        camera_set=cameras,observation_set=obs,source_rgba_by_view=textures,
    )
    assert len(result.views)==8
    assert result.metadata["canonical_geometry_is_never_rgb_authority"] is True
    assert all(v.metadata["shading_used"] is False for v in result.views)
    assert all(images[i].dtype==np.uint8 and images[i].shape==(8,8,4) for i in range(8))
    assert all(v.visible_pixel_count>0 for v in result.views)
    assert all(v.geometry_visible_pixel_count>0 for v in result.views)
    assert all(v.direct_source_geometry_pixel_count==v.geometry_visible_pixel_count for v in result.views)
    assert all(v.cross_view_source_geometry_pixel_count==0 for v in result.views)


def test_rest_renderer_uses_target_local_source_texture_per_view():
    cameras,obs,mesh,app,comp,graph,textures=_fixture()
    _result,images=render_rest_views(
        mesh=mesh,presentation_graph=graph,appearance_set=app,composition_set=comp,
        camera_set=cameras,observation_set=obs,source_rgba_by_view=textures,
    )
    # Blue channel is view-coded in each source texture; any covered pixel must carry
    # the target-local donor's code because each face donor is target-local.
    for view,image in images.items():
        covered=image[...,3]>0
        assert np.all(image[...,2][covered]==view*20)


def test_observation_pixel_center_sampler_uses_integer_texel_centers_without_half_pixel_shift():
    image=np.zeros((4,4,4),dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            image[y,x]=[10*x,20*y,30+x+y,255]
    assert np.array_equal(_sample_bilinear_rgba(image,(2.0,1.0)),image[1,2])
    expected=np.floor(
        0.25*image[1,1].astype(np.float64)
        +0.25*image[1,2].astype(np.float64)
        +0.25*image[2,1].astype(np.float64)
        +0.25*image[2,2].astype(np.float64)
        +0.5
    ).astype(np.uint8)
    assert np.array_equal(_sample_bilinear_rgba(image,(1.5,1.5)),expected)
