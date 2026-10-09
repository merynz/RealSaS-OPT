"""Relational failures remain failures with healthy triangles and raster parity."""
from collections import OrderedDict
import io
import os
import struct
import subprocess

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_core.camera_geometry_v2 import CameraProjectionV3, project_points_xyz_v3
from compiler.realsas_compiler_core.runtime_package_v2 import write_rss_v2
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_attachment_motion_v1 import ATTACHMENT_OPERATOR_ID, slot_rigid_transform_2d
from compiler.realsas_compiler_core.visual_depth_v2 import render_visual_depth
from compiler.realsas_compiler_core.visual_domain_v2 import OPERATOR_ID, DEPTH_CONTRACT, presentation_condition_metrics
from compiler.realsas_compiler_core.visual_motion_blend_v1 import evaluate_motion_blend
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import (
    ConnectedPresentationPalette, compile_connected_palette, apply_connected_palette, prove_connected_palette,
    apply_connected_attachment_slots,
)


def fixture(degrees=60, axis_name="Y", slope=1):
    camera = CameraProjectionV3("synthetic", 0, (0,0,0), (1,0,0), (0,1,0), (0,0,1), 2., 256)
    a=np.deg2rad(degrees); c,s=np.cos(a),np.sin(a)
    matrix=np.eye(4)
    matrix[:3,:3]=([[c,0,s],[0,1,0],[-s,0,c]] if axis_name=="Y" else [[c,-s,0],[s,c,0],[0,0,1]])
    xy=np.array([[0,-.15],[1,-.15],[1,.15],[0,.15],[1,-.15],[2,-.15],[2,.15],[1,.15],[1,0],[1,0]])
    xyz=np.c_[xy,xy[:,0]*slope]
    rest=project_points_xyz_v3(xyz,camera)
    blend=np.array([[1.,0.]]*4+[[0.,1.]]*4+[[1.,0.],[0.,1.]])
    geometry=dict(axis_positions_source=np.array([[0.,0.,0.],[1.,0.,float(slope)]]),
                  axis_parents=np.array([-1,0]),skin_matrices_source=np.stack((matrix,matrix)),camera=camera)
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
    return geometry, rest, blend, faces


@pytest.mark.parametrize("degrees,axis,slope", [(0,"Y",1),(60,"Z",1),(60,"Y",0),(60,"Y",1)])
def test_connected_palette_preserves_common_joint_and_healthy_rigid_pieces(degrees,axis,slope):
    g,rest,blend,faces=fixture(degrees,axis,slope)
    snapshots={k:v.copy() for k,v in g.items() if isinstance(v,np.ndarray)}
    weights=blend.copy()
    palette=compile_connected_palette(**g)
    actual=apply_connected_palette(rest,rest_source_xy=rest[:,:2],coefficients=blend,palette=palette)
    assert np.linalg.norm(actual[8,:2]-actual[9,:2]) < 1e-10
    metric=presentation_condition_metrics(rest[:,:2],actual[:,:2],faces)
    assert metric["area_collapse_count"] == metric["condition_failure_count"] == 0
    assert metric["minimum_signed_area_ratio"] == pytest.approx(1)
    assert metric["maximum_jacobian_condition"] == pytest.approx(1)
    assert np.array_equal(actual[:,2],rest[:,2])
    proof=prove_connected_palette(palette,**g)
    assert proof["connected_palette_relations_passed"]
    assert proof["parent_child_relation_count"] == 1
    assert proof["visual_chart_contact_coverage_qualified"] is False
    assert np.array_equal(weights,blend)
    assert all(np.array_equal(g[k],v) for k,v in snapshots.items())


def test_legacy_palette_replay_and_triangle_health_do_not_prove_joint_relations():
    g,rest,blend,faces=fixture()
    transforms=[slot_rigid_transform_2d(slot_rest_xyz=p,skin_matrix_source=m,camera=g["camera"])
                for p,m in zip(g["axis_positions_source"],g["skin_matrices_source"])]
    broken=ConnectedPresentationPalette(np.array([x[0] for x in transforms]),np.array([x[1] for x in transforms]))
    actual=apply_connected_palette(rest,rest_source_xy=rest[:,:2],coefficients=blend,palette=broken)
    replay=evaluate_motion_blend(rest,rest_source_xy=rest[:,:2],coefficients=blend,
        **{k:g[k] for k in ("axis_positions_source","skin_matrices_source","camera")})
    assert np.array_equal(actual,replay)
    metric=presentation_condition_metrics(rest[:,:2],actual[:,:2],faces)
    assert metric["area_collapse_count"] == metric["condition_failure_count"] == 0
    proof=prove_connected_palette(broken,**g)
    assert proof["connected_palette_relations_passed"] is False
    assert proof["maximum_parent_child_relation_residual_px"] == pytest.approx(23.425625842204084)


def test_setup_identity_and_explicit_root_and_local_translation_are_preserved():
    g,rest,blend,_=fixture(degrees=0)
    setup=apply_connected_palette(rest,rest_source_xy=rest[:,:2],coefficients=blend,palette=compile_connected_palette(**g))
    assert np.array_equal(setup,rest)
    g["skin_matrices_source"][:, :3,3]=[.4,-.2,.3]
    g["skin_matrices_source"][1,:3,3]+=[.1,.05,0]
    palette=compile_connected_palette(**g)
    points=project_points_xyz_v3(g["axis_positions_source"],g["camera"])[:,:2]
    moved=np.einsum("jab,jb->ja",palette.rotations,points)+palette.translations
    expected=project_points_xyz_v3(g["axis_positions_source"]+g["skin_matrices_source"][:,:3,3],g["camera"])[:,:2]
    assert np.allclose(moved,expected,atol=1e-12,rtol=0)
    assert prove_connected_palette(palette,**g)["connected_palette_relations_passed"]


def test_non_topological_joint_numbering_does_not_change_the_pose():
    g,_,_,_=fixture()
    normal=compile_connected_palette(**g)
    g["axis_positions_source"]=g["axis_positions_source"][[1,0]]
    g["skin_matrices_source"]=g["skin_matrices_source"][[1,0]]
    g["axis_parents"]=np.array([1,-1])
    reordered=compile_connected_palette(**g)
    assert np.array_equal(normal.rotations,reordered.rotations[[1,0]])
    assert np.array_equal(normal.translations,reordered.translations[[1,0]])


@pytest.mark.parametrize("parents", [np.array([-1,-1]),np.array([-1,1]),np.array([-1.,0.]),np.array([-1,2]),np.array([2**64-1,0],dtype=np.uint64)])
def test_invalid_or_inferred_hierarchy_is_rejected(parents):
    g,_,_,_=fixture()
    g["axis_parents"]=parents
    with pytest.raises(QualificationError): compile_connected_palette(**g)


def test_unobservable_out_of_plane_twist_has_no_silent_history_fallback():
    g,_,_,_=fixture(degrees=180)
    with pytest.raises(QualificationError,match="TWIST_UNOBSERVABLE"):
        compile_connected_palette(**g)


def test_equipped_art_uses_the_connected_body_slot_and_keeps_its_depth():
    g,rest,blend,_=fixture()
    palette=compile_connected_palette(**g)
    body=apply_connected_palette(rest,rest_source_xy=rest[:,:2],coefficients=blend,palette=palette)
    owners=np.array([0]*9+[1])
    attachments={"attachments":[dict(attachment_index=1,target_slot_raw_index_fit_only=1,
                                     presentation_motion_model=ATTACHMENT_OPERATOR_ID)]}
    actual=apply_connected_attachment_slots(body,rest_source_xy=rest[:,:2],
        vertex_attachment_owner=owners,attachments=attachments,palette=palette)
    assert np.array_equal(actual[:9],body[:9])
    assert np.array_equal(actual[:,2],body[:,2])
    assert np.linalg.norm(actual[8,:2]-actual[9,:2]) < 1e-10


def test_equipment_cannot_silently_infer_a_slot_or_accept_an_unknown_motion_model():
    g,rest,_,_=fixture()
    palette=compile_connected_palette(**g)
    for slot,model in [(99,ATTACHMENT_OPERATOR_ID),(1.,ATTACHMENT_OPERATOR_ID),(True,ATTACHMENT_OPERATOR_ID),(1,"UNKNOWN")]:
        with pytest.raises(QualificationError):
            apply_connected_attachment_slots(rest,rest_source_xy=rest[:,:2],
                vertex_attachment_owner=np.ones(len(rest),dtype=int),
                attachments={"attachments":[dict(attachment_index=1,target_slot_raw_index_fit_only=slot,
                                                presentation_motion_model=model)]},palette=palette)


def _native_frame(tmp_path, positions, faces):
    player=os.environ.get("REALSAS_RUNTIME_V2_PLAYER")
    if not player: pytest.skip("native binary required for byte parity fixture")
    resolution=256
    uv=np.zeros_like(positions)
    texture=np.zeros((resolution,resolution,4),dtype=np.uint8); texture[:]=(90,150,210,255)
    png=io.BytesIO();Image.fromarray(texture).save(png,format="PNG")
    depths=np.r_[np.ones(4),np.full(4,2.)]
    manifest={
        "schema":"RealSaS.RuntimePackage.v2","presentation_geometry_mode":"SOURCE_OWNED_VISUAL_PRESENTATION_V1",
        "mechanical_mesh_render_authority":"0","runtime_visual_mesh_rebuild":"0","runtime_binding_solve":"0",
        "runtime_generation":"0","donor_search_at_runtime":"0","mip_generation_authorized":"0",
        "texture_sampling_contract":"SOURCE_RGBA8_BILINEAR_STRAIGHT_TO_PM_V1",
        "view_count":"8","clip_count":"1","clip.0.id":"fixture","clip.0.frame_count":"1",
        "playback_sampling_contract":"SEALED_FRAME_INDEX_ONLY","view_selection_contract":"SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        "cross_direction_blending_authorized":"0","host_interpolation_authorized":"0",
        "presentation_state_execution_authorized":"0","clipping_authorized":"0","tint_order_visibility_authorized":"0",
        "view.0.id":"V0","view.0.mesh_entry":"mesh","view.0.texture_entry":"texture","view.0.resolution":str(resolution),
        "clip.0.view.0.positions_entry":"positions","clip.0.view.0.depths_entry":"depths",
        "visual_deformation_operator_id":OPERATOR_ID,"depth_ownership_contract":DEPTH_CONTRACT,
        "depth_tie_epsilon":"1e-9","maximum_fragment_layers":"32",
    }
    entries=OrderedDict([
        ("mesh",b"RSVM1\0\0\0"+struct.pack("<IIII",resolution,resolution,len(positions),len(faces))+uv.astype("<f8").tobytes()+faces.astype("<u4").tobytes()),
        ("positions",b"RSVP1\0\0\0"+struct.pack("<II",1,len(positions))+positions.astype("<f8").tobytes()),
        ("depths",b"RSVD1\0\0\0"+struct.pack("<II",1,len(positions))+depths.astype("<f8").tobytes()),
        ("texture",png.getvalue()),("manifest.txt",("\n".join(k+"="+v for k,v in manifest.items())+"\n").encode()),
    ])
    archive=tmp_path/"fixture.rss";write_rss_v2(archive,entries)
    paths=[tmp_path/name for name in ("rgba","provenance","source_view","owner")]
    args=[player,str(archive),"--clip","fixture","--view","V0","--frame","0"]
    for flag,path in zip(("--out-rgba","--out-provenance","--out-source-view","--out-owner"),paths): args.extend((flag,str(path)))
    proc=subprocess.run(args,capture_output=True,text=True)
    assert proc.returncode == 0,proc.stderr
    reference=render_visual_depth(positions=positions,depths=depths,faces=faces,uv=uv,texture=texture,resolution=resolution)
    assert np.any(reference.straight_rgba_u8[:,:,3])
    assert paths[0].read_bytes()==reference.straight_rgba_u8.tobytes()
    assert paths[1].read_bytes()==reference.provenance_code.tobytes()
    assert paths[3].read_bytes()==reference.owner_face_index.astype("<i4").tobytes()


def test_broken_relations_fail_with_real_native_reference_byte_identical_images(tmp_path):
    g,rest,blend,faces=fixture()
    transforms=[slot_rigid_transform_2d(slot_rest_xyz=p,skin_matrix_source=m,camera=g["camera"])
                for p,m in zip(g["axis_positions_source"],g["skin_matrices_source"])]
    broken=ConnectedPresentationPalette(np.array([x[0] for x in transforms]),np.array([x[1] for x in transforms]))
    actual=apply_connected_palette(rest,rest_source_xy=rest[:,:2],coefficients=blend,palette=broken)
    metrics=presentation_condition_metrics(rest[:,:2],actual[:,:2],faces)
    assert metrics["area_collapse_count"] == metrics["condition_failure_count"] == 0
    _native_frame(tmp_path,actual[:8,:2],faces)
    proof = prove_connected_palette(broken,**g)
    assert proof["connected_palette_relations_passed"] is False
    from compiler.realsas_compiler_core.visual_presentation_contract_v1 import relational_verdict
    # The same gate function is consumed by the scoped Stage45 adapter. Give
    # every other independent qualification a PASS so this failure is isolated.
    proof.update(qualified_contact_relations_passed=True, qualified_dynamic_coverage_passed=True,
        semantic_occlusion_passed=True, setup_identity_passed=True,
        frame0_relations_passed=True, temporal_relations_passed=True,
        native_reference_byte_parity_passed=True, area_condition_passed=True)
    assert relational_verdict(proof)["relational_presentation_blockers"] == ["connected_palette_relations_passed"]
