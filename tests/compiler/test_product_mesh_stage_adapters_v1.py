from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import QualifiedObservationViewIR, build_qualified_observation_set
from compiler.realsas_compiler_core.camera_authority_v1 import build_qualified_camera_set
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    _mesh_component_triangles,
    camera_projection_binding_hash,
    rasterize_triangles_half_integer_top_left,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import qualify_camera_v3
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_mesh_candidate_from_dict,
    canonical_puppet_state_from_dict,
    qualified_observation_set_from_dict,
    qualified_camera_set_from_dict,
    component_carrier_policy_from_dict,
    mechanical_partition_from_dict,
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_presentation_graph_from_dict,
    qualified_appearance_set_from_dict,
    qualified_composition_set_from_dict,
    qualified_presentation_structure_from_dict,
    rest_render_set_from_dict,
    rest_preservation_measurement_set_from_dict,
    rest_source_preservation_policy_from_dict,
    qualified_rest_source_preservation_from_dict,
    read_json,
    write_ir_json,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from compiler.realsas_compiler_services.orchestrator.adapters.presentation_v1 import qualify_presentation_graph_stage
from compiler.realsas_compiler_services.orchestrator.adapters.rest_render_v1 import qualify_rest_render_stage
from compiler.realsas_compiler_services.orchestrator.adapters.rest_preservation_v1 import qualify_rest_source_preservation_stage
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    build_canonical_mesh_candidate_stage,
    bind_qualified_mesh_skin_stage,
    seal_canonical_puppet_state_stage,
    qualify_canonical_mesh_stage,
    qualify_mechanical_partition_and_carriers,
    seal_deformation_capability_envelope,
)

ROOT=Path(__file__).resolve().parents[2]


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _out(path:Path,schema:str):
    return {"path":str(path.resolve()),"sha256":_sha(path),"schema":schema}


def _write(path:Path,value):
    write_ir_json(path,value)
    return path


def _fixture(tmp_path):
    run_root=tmp_path/"run"
    surface=RiggingSurfaceIR(
        (
            # Observation PIXEL_CENTER_XY uses source texel index centers.
            # For this camera these are exactly Runtime projection minus 0.5.
            SurfaceNode("s0",(0.0,0.0,0.0),tuple(range(8)),("p0",),("o0",),tuple((v,(3.5,3.5)) for v in range(8))),
            SurfaceNode("s1",(1.0,0.0,0.0),tuple(range(8)),("p1",),("o1",),tuple((v,(5.5,3.5)) for v in range(8))),
            SurfaceNode("s2",(0.0,1.0,0.0),tuple(range(8)),("p2",),("o2",),tuple((v,(3.5,1.5)) for v in range(8))),
        ),
        (
            SurfaceRelation("r01","s0","s1","LOCAL",1.0),
            SurfaceRelation("r12","s1","s2","LOCAL",1.0),
            SurfaceRelation("r02","s0","s2","LOCAL",1.0),
        ),
        "surface-hash",
        metadata={"raster_coordinate_system":"PIXEL_CENTER_XY","resolution":8},
    )
    skeleton=QualifiedSkeletonIR(
        (QualifiedJoint("j0",(0.0,0.0,0.0),None,("s0","s1","s2"),"proposal-root"),),
        "j0",{"status":"PASS"},"skeleton-hash",
    )
    skin=QualifiedSkinIR(
        tuple(QualifiedSkinRow(sid,(("j0",1.0),),0.0,0.0) for sid in ("s0","s1","s2")),
        surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,{"status":"PASS"},"skin-hash",
    )
    s_path=_write(tmp_path/"surface.json",surface)
    g_path=_write(tmp_path/"skeleton.json",skeleton)
    w_path=_write(tmp_path/"skin.json",skin)

    axis={
        "schema":"RealSaS.TestAxisContract.v1",
        "status":"FROZEN",
        "joint_axes":[
            {"canonical_joint_id":"j0","axis_xyz":[0,0,1],"legacy_scalar_to_semantic_sign":1.0,"role":"root"}
        ],
    }
    axis_path=tmp_path/"axis.json"; axis_path.write_text(json.dumps(axis,sort_keys=True)+"\n",encoding="utf-8")

    cameras=[]
    for vi in range(8):
        cameras.append({
            "schema_version":"RealSaS.FullSurfaceCameraProjection.v3",
            "view_id":f"V{vi}",
            "view_index":vi,
            "origin":[0.0,0.0,-2.0],
            "right":[1.0,0.0,0.0],
            "screen_up":[0.0,1.0,0.0],
            "forward":[0.0,0.0,1.0],
            "half_extent":2.0,
            "resolution":8,
        })
    bundle={"schema":"RealSaS.CameraProjectionBundle.v1","cameras":cameras}
    cam_path=tmp_path/"cameras.json"; cam_path.write_text(json.dumps(bundle,sort_keys=True)+"\n",encoding="utf-8")

    policy_path=ROOT/"canonical"/"QUALIFIED_MESH_PRODUCT_POLICY_V1_20260918.json"
    manifest={
        "components":{},
        "carrier_policy":{},
        "deformation_envelope":{
            "axis_contract":{"path":str(axis_path),"sha256":_sha(axis_path)},
            "joint_ranges":[
                {"canonical_joint_id":"j0","min_rotation_deg":0.0,"max_rotation_deg":0.0}
            ],
            "allowed_attachment_state_hashes":[],
        },
        "mesh_policy":{
            "document":{"path":str(policy_path.resolve()),"sha256":_sha(policy_path)}
        },
        "mesh":{"backend":"CANONICAL_RELATION_BASELINE_V1"},
        "observation":{"component_masks":[],"source_foreground_masks":[],"source_rasters":[]},
        "presentation":{"mode":"AUTO_ROLE_FREE_V1"},
        "appearance":{
            "rest_preservation_policy":{
                "path":str((ROOT/"canonical"/"REST_SOURCE_PRESERVATION_POLICY_V1_20260919.json").resolve()),
                "sha256":_sha(ROOT/"canonical"/"REST_SOURCE_PRESERVATION_POLICY_V1_20260919.json"),
            },
            "rest_preservation_calibration":{
                "path":str((ROOT/"canonical"/"REST_SOURCE_PRESERVATION_CALIBRATION_RESULT_20260919.json").resolve()),
                "sha256":_sha(ROOT/"canonical"/"REST_SOURCE_PRESERVATION_CALIBRATION_RESULT_20260919.json"),
            },
        },
    }

    observation_views=[]
    source_foreground_rows=[]
    source_raster_rows=[]
    for row in cameras:
        camera=qualify_camera_v3(row,view_id=row["view_id"],view_index=int(row["view_index"]))
        # The baseline candidate is the same single triangle as S, so source foreground
        # is materialized directly from exact S positions and the same camera.
        from compiler.realsas_compiler_core.playback_full_surface_v3 import project_points_xyz_v3
        projected=project_points_xyz_v3([node.P for node in surface.surface_nodes],camera)
        triangle=tuple((float(p[0]),float(p[1])) for p in projected)
        fg=rasterize_triangles_half_integer_top_left((triangle,),width=8,height=8)
        fg_path=tmp_path/f"source_fg_v{camera.view_index}.bin"; fg_path.write_bytes(fg)
        rgba=np.zeros((8,8,4),dtype=np.uint8)
        for yy in range(8):
            for xx in range(8):
                rgba[yy,xx]=[xx*20,yy*20,camera.view_index*20,255 if fg[yy*8+xx] else 0]
        source_path=tmp_path/f"source_V{camera.view_index}.png"
        Image.fromarray(rgba,"RGBA").save(source_path,format="PNG",compress_level=1,optimize=False)
        obs_hash=content_sha256({"view":camera.view_index,"fixture":"triangle","source_raster_sha256":_sha(source_path)})
        observation_views.append(QualifiedObservationViewIR(
            camera.view_index,8,8,obs_hash,
            _sha(source_path),
            _sha(fg_path),camera_projection_binding_hash(camera),
            "PASS",(f"fixture-observation-{camera.view_index}",),
        ))
        source_foreground_rows.append({
            "view_index":camera.view_index,
            "mask":{"path":str(fg_path),"sha256":_sha(fg_path)},
        })
        source_raster_rows.append({
            "view_index":camera.view_index,
            "image":{"path":str(source_path),"sha256":_sha(source_path)},
        })
    observation_set=build_qualified_observation_set(tuple(observation_views))
    obs_path=_write(tmp_path/"observation_set.json",observation_set)
    manifest["observation"]["source_foreground_masks"]=source_foreground_rows
    manifest["observation"]["source_rasters"]=source_raster_rows
    camera_set=build_qualified_camera_set(
        cameras,source_bundle_sha256=_sha(cam_path),
        metadata={"fixture":True},
    )
    camera_set_path=_write(tmp_path/"camera_set.json",camera_set)

    ledger={"stages":[
        {"id":"05_CAMERA_CONTRACT_SOLVED","status":"PASS","outputs":[_out(camera_set_path,"RealSaS.QualifiedCameraSetIR.v1")]},
        {"id":"07_OBSERVATION_CONTRACT_QUALIFIED","status":"PASS","outputs":[_out(obs_path,"RealSaS.QualifiedObservationSetIR.v1")]},
        {"id":"15_RIGGING_SURFACE_QUALIFIED","status":"PASS","outputs":[_out(s_path,"RealSaS.RiggingSurfaceIR.v1")]},
        {"id":"18_SKELETON_QUALIFIED","status":"PASS","outputs":[_out(g_path,"RealSaS.QualifiedSkeletonIR.v1")]},
        {"id":"22_SKIN_QUALIFIED","status":"PASS","outputs":[_out(w_path,"RealSaS.QualifiedSkinIR.v1")]},
    ]}
    return {
        "run_id":"SUBJECT_FREE_TRIANGLE_V1",
        "run_root":run_root,
        "run_manifest":manifest,
        "ledger":ledger,
    }


def _install_stage_outputs(ctx,stage_id,result):
    ctx["ledger"]["stages"].append({
        "id":stage_id,
        "status":"PASS",
        "outputs":result["outputs"],
    })


def test_stage24_to_27_typed_wiring_closes_on_subject_free_triangle(tmp_path):
    ctx=_fixture(tmp_path)

    r24=qualify_mechanical_partition_and_carriers(ctx)
    assert r24["status"]=="PASS"
    _install_stage_outputs(ctx,"24_MECHANICAL_PARTITION_QUALIFIED",r24)
    partition=mechanical_partition_from_dict(read_json(r24["outputs"][0]["path"]))
    carrier=component_carrier_policy_from_dict(read_json(r24["outputs"][1]["path"]))
    assert len(partition.components)==1
    assert carrier.decisions[0].carrier_class=="MESH"
    assert carrier.decisions[0].metadata["conservative_default"] is True

    r25=seal_deformation_capability_envelope(ctx)
    assert r25["status"]=="PASS"
    _install_stage_outputs(ctx,"25_DEFORMATION_CAPABILITY_ENVELOPE",r25)

    r26=build_canonical_mesh_candidate_stage(ctx)
    assert r26["status"]=="PASS"
    _install_stage_outputs(ctx,"26_MESH_CANDIDATE_BUILD",r26)
    candidate=canonical_mesh_candidate_from_dict(read_json(r26["outputs"][0]["path"]))
    assert len(candidate.faces)==1

    component_id=partition.components[0].component_id
    camera_set=qualified_camera_set_from_dict(
        read_json(next(out for row in ctx["ledger"]["stages"] if row["id"]=="05_CAMERA_CONTRACT_SOLVED" for out in row["outputs"])["path"])
    )
    mask_rows=[]
    for camera in camera_set.cameras:
        triangles=_mesh_component_triangles(candidate,camera,component_id=component_id)
        mask=rasterize_triangles_half_integer_top_left(triangles,width=8,height=8)
        mask_path=tmp_path/f"mask_v{camera.view_index}.bin"; mask_path.write_bytes(mask)
        mask_rows.append({
            "view_index":camera.view_index,
            "component_id":component_id,
            "mask":{"path":str(mask_path),"sha256":_sha(mask_path)},
        })
    ctx["run_manifest"]["observation"]["component_masks"]=mask_rows

    r27=qualify_canonical_mesh_stage(ctx)
    assert r27["status"]=="PASS",r27
    qualified=next(out for out in r27["outputs"] if out["schema"]=="RealSaS.QualifiedMeshIR.v1")
    mesh=qualified_mesh_from_dict(read_json(qualified["path"]))
    assert mesh.qualification_report["g3_stress_probe_status"]=="PASS"
    assert mesh.qualification_report["view_component_coverage_matrix_complete"] is True
    assert len(mesh.qualification_report["view_component_coverage"])==8
    assert mesh.qualification_report["consequential_unknown_boundary_count"]==0
    assert len(mesh.mesh_lineage_hash)==64
    _install_stage_outputs(ctx,"27_QUALIFIED_MESH_GATE",r27)

    r28=bind_qualified_mesh_skin_stage(ctx)
    assert r28["status"]=="PASS",r28
    bound=qualified_mesh_skin_from_dict(read_json(r28["outputs"][0]["path"]))
    assert bound.mesh_binding_hash==mesh.mesh_lineage_hash
    assert bound.skin_binding_hash=="skin-hash"
    assert len(bound.rows)==len(mesh.vertices)
    assert len(bound.mesh_skin_lineage_hash)==64
    _install_stage_outputs(ctx,"28_QUALIFIED_MESH_SKIN_TRANSFER",r28)

    r29=seal_canonical_puppet_state_stage(ctx)
    assert r29["status"]=="PASS",r29
    state=canonical_puppet_state_from_dict(read_json(r29["outputs"][0]["path"]))
    assert state.mesh_lineage_hash==mesh.mesh_lineage_hash
    assert state.mesh_skin_lineage_hash==bound.mesh_skin_lineage_hash
    assert state.metadata["presentation_bound"] is False
    assert state.metadata["motion_bound"] is False
    assert len(state.product_state_hash)==64
    _install_stage_outputs(ctx,"29_CANONICAL_PUPPET_STATE_SEALED",r29)

    r30=qualify_presentation_graph_stage(ctx)
    assert r30["status"]=="PASS",r30
    by_schema={out["schema"]:out for out in r30["outputs"]}
    structure=qualified_presentation_structure_from_dict(read_json(by_schema["RealSaS.QualifiedPresentationStructureIR.v1"]["path"]))
    appearance=qualified_appearance_set_from_dict(read_json(by_schema["RealSaS.QualifiedAppearanceSetIR.v1"]["path"]))
    composition=qualified_composition_set_from_dict(read_json(by_schema["RealSaS.QualifiedCompositionSetIR.v1"]["path"]))
    graph=qualified_presentation_graph_from_dict(read_json(by_schema["RealSaS.QualifiedPresentationGraphIR.v1"]["path"]))
    assert graph.presentation_structure_binding_hash==structure.structure_lineage_hash
    assert graph.appearance_set_binding_hash==appearance.appearance_set_hash
    assert graph.composition_set_binding_hash==composition.composition_set_hash
    assert graph.product_state_binding_hash==state.product_state_hash
    assert len(graph.view_overlays)==8
    assert graph.qualification_report["single_canonical_mesh"] is True
    assert graph.qualification_report["slot_order_solves_physical_occlusion"] is False
    assert graph.qualification_report["categorical_recognition_used"] is False
    for binding in appearance.bindings:
        assert {tuple(c.donor_raster_xy) for c in binding.corner_bindings}=={
            (3.5,3.5),(5.5,3.5),(3.5,1.5)
        }
        assert {c.donor_view_index for c in binding.corner_bindings}=={binding.target_view_index}
    _install_stage_outputs(ctx,"30_QUALIFIED_PRESENTATION_GRAPH",r30)

    r31=qualify_rest_render_stage(ctx)
    assert r31["status"]=="PASS",r31
    rest_out=next(out for out in r31["outputs"] if out["schema"]=="RealSaS.RestRenderSetIR.v1")
    rest=rest_render_set_from_dict(read_json(rest_out["path"]))
    assert len(rest.views)==8
    assert rest.presentation_binding_hash==graph.presentation_lineage_hash
    assert rest.appearance_set_binding_hash==appearance.appearance_set_hash
    assert rest.composition_set_binding_hash==composition.composition_set_hash
    assert rest.metadata["canonical_geometry_is_never_rgb_authority"] is True
    assert all(len(row.rendered_rgba_sha256)==64 for row in rest.views)
    assert all(len(row.metadata["png_sha256"])==64 for row in rest.views)
    assert all(
        sum(Path(row["mask"]["path"]).read_bytes())>0
        for row in ctx["run_manifest"]["observation"]["source_foreground_masks"]
    ), [sum(Path(row["mask"]["path"]).read_bytes()) for row in ctx["run_manifest"]["observation"]["source_foreground_masks"]]
    assert all(row.geometry_visible_pixel_count>0 for row in rest.views), [row.geometry_visible_pixel_count for row in rest.views]
    assert all(row.visible_pixel_count>0 for row in rest.views), [row.visible_pixel_count for row in rest.views]
    _install_stage_outputs(ctx,"31_REST_RENDER_8VIEW",r31)

    r32=qualify_rest_source_preservation_stage(ctx)
    assert r32["status"]=="PASS",r32
    by_schema={out["schema"]:out for out in r32["outputs"]}
    measurements=rest_preservation_measurement_set_from_dict(
        read_json(by_schema["RealSaS.RestPreservationMeasurementSetIR.v1"]["path"])
    )
    policy=rest_source_preservation_policy_from_dict(
        read_json(by_schema["RealSaS.RestSourcePreservationPolicyIR.v1"]["path"])
    )
    qualified_rest=qualified_rest_source_preservation_from_dict(
        read_json(by_schema["RealSaS.QualifiedRestSourcePreservationIR.v1"]["path"])
    )
    assert len(measurements.views)==8
    assert all(row.alpha_recall>=policy.min_alpha_recall for row in measurements.views)
    assert all(row.alpha_precision>=policy.min_alpha_precision for row in measurements.views)
    assert all(row.overlap_rgba_mismatch_pixel_count==0 for row in measurements.views)
    assert all(row.direct_source_geometry_fraction==1.0 for row in measurements.views)
    assert all(row.cross_view_source_geometry_fraction==0.0 for row in measurements.views)
    assert qualified_rest.qualification_report["every_view_passed_every_rule"] is True
    assert qualified_rest.qualification_report["motion_authorization_precondition_satisfied"] is True
