from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from compiler.realsas_compiler_core.camera_authority_v1 import build_qualified_camera_set
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import mask_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,build_qualified_observation_set,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import measure_rest_source_preservation
from compiler.realsas_compiler_core.rest_render_v1 import RestRenderSetIR,RestRenderViewIR,rest_render_set_hash,rgba_sha256


def _fixture():
    source={}
    rendered={}
    foreground={}
    views=[]
    render_rows=[]
    for view in range(8):
        img=np.zeros((8,8,4),dtype=np.uint8)
        img[2:6,2:6]=[40+view,80,120,255]
        source[view]=img.copy()
        rendered[view]=img.copy()
        mask=(img[...,3]>0).astype(np.uint8)
        fg=bytes(mask.reshape(-1))
        foreground[view]=fg
        views.append(QualifiedObservationViewIR(
            view,8,8,f"obs-{view}",f"{view+16:064x}",mask_sha256(fg),f"cam-{view}",
            "PASS",(f"e-{view}",)
        ))
        render_rows.append(RestRenderViewIR(
            view,f"cam-{view}",f"app-{view}",f"comp-{view}",rgba_sha256(img),
            8,8,16,16,16,0,
        ))
    obs=build_qualified_observation_set(tuple(views))
    rr=RestRenderSetIR(
        tuple(render_rows),"mesh","presentation","appearance","composition","camera-set",
        obs.observation_set_hash,""
    )
    rr=replace(rr,render_set_hash=rest_render_set_hash(rr))
    return obs,rr,source,rendered,foreground


def test_identity_rest_replay_has_zero_visual_and_silhouette_error():
    obs,rr,source,rendered,foreground=_fixture()
    result=measure_rest_source_preservation(
        rest_render_set=rr,observation_set=obs,source_rgba_by_view=source,
        rendered_rgba_by_view=rendered,source_foreground_by_view=foreground,
    )
    assert len(result.views)==8
    for row in result.views:
        assert row.alpha_recall==1.0
        assert row.alpha_precision==1.0
        assert row.alpha_iou==1.0
        assert row.silhouette_edge_mean_px==0.0
        assert row.silhouette_edge_p95_px==0.0
        assert row.premultiplied_rgb_mae==0.0
        assert row.alpha_mae==0.0
        assert row.overlap_rgba_mismatch_pixel_count==0
        assert row.overlap_rgba_mismatch_fraction==0.0
        assert row.overlap_rgba_max_abs_channel_error_u8==0
        assert row.direct_source_geometry_fraction==1.0
        assert row.cross_view_source_geometry_fraction==0.0
    assert result.metadata["admission_status"]=="MEASURED_NOT_ADMITTED"


def test_shifted_render_is_detected_by_silhouette_and_color_metrics():
    obs,rr,source,rendered,foreground=_fixture()
    shifted={}
    rows=[]
    for view in range(8):
        img=np.zeros_like(rendered[view])
        img[2:6,3:7]=rendered[view][2:6,2:6]
        shifted[view]=img
        old=rr.views[view]
        rows.append(replace(old,rendered_rgba_sha256=rgba_sha256(img)))
    rr2=replace(rr,views=tuple(rows),render_set_hash="")
    rr2=replace(rr2,render_set_hash=rest_render_set_hash(rr2))
    result=measure_rest_source_preservation(
        rest_render_set=rr2,observation_set=obs,source_rgba_by_view=source,
        rendered_rgba_by_view=shifted,source_foreground_by_view=foreground,
    )
    assert all(row.alpha_recall<1.0 for row in result.views)
    assert all(row.silhouette_edge_mean_px>0.0 for row in result.views)
    assert all(row.premultiplied_rgb_mae>0.0 for row in result.views)


def test_one_lsb_color_mutation_is_exactly_detected_on_overlap():
    obs,rr,source,rendered,foreground=_fixture()
    changed={k:v.copy() for k,v in rendered.items()}
    rows=[]
    for view in range(8):
        changed[view][3,3,0]=np.uint8(int(changed[view][3,3,0])+1)
        rows.append(replace(rr.views[view],rendered_rgba_sha256=rgba_sha256(changed[view])))
    rr2=replace(rr,views=tuple(rows),render_set_hash="")
    rr2=replace(rr2,render_set_hash=rest_render_set_hash(rr2))
    result=measure_rest_source_preservation(
        rest_render_set=rr2,observation_set=obs,source_rgba_by_view=source,
        rendered_rgba_by_view=changed,source_foreground_by_view=foreground,
    )
    assert all(row.overlap_rgba_mismatch_pixel_count==1 for row in result.views)
    assert all(row.overlap_rgba_max_abs_channel_error_u8==1 for row in result.views)
