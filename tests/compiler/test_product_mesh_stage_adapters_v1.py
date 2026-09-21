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
    motion_source_set_from_dict,
    qualified_motion_source_seal_from_dict,
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
from compiler.realsas_compiler_services.orchestrator.adapters.motion_source_v1 import seal_motion_source_or_preset_stage
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


def _fixture(tmp_path, *, resolution: int = 8):
    resolution = int(resolution)
    if resolution < 8 or resolution % 4:
        raise ValueError("fixture resolution must be >=8 and divisible by 4")
    run_root=tmp_path/"run"
    center = float(resolution) / 2.0 - 0.5
    positive_quarter = 3.0 * float(resolution) / 4.0 - 0.5
    negative_quarter = float(resolution) / 4.0 - 0.5
    surface=RiggingSurfaceIR(
        (
            # Observation PIXEL_CENTER_XY uses source texel index centers.
            # For this camera these are exactly Runtime projection minus 0.5.
            SurfaceNode("s0",(0.0,0.0,0.0),tuple(range(8)),("p0",),("o0",),tuple((v,(center,center)) for v in range(8))),
            SurfaceNode("s1",(1.0,0.0,0.0),tuple(range(8)),("p1",),("o1",),tuple((v,(positive_quarter,center)) for v in range(8))),
            SurfaceNode("s2",(0.0,1.0,0.0),tuple(range(8)),("p2",),("o2",),tuple((v,(center,negative_quarter)) for v in range(8))),
        ),
        (
            SurfaceRelation("r01","s0","s1","LOCAL",1.0),
            SurfaceRelation("r12","s1","s2","LOCAL",1.0),
            SurfaceRelation("r02","s0","s2","LOCAL",1.0),
        ),
        "surface-hash",
        metadata={"raster_coordinate_system":"PIXEL_CENTER_XY","resolution":resolution},
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
            "resolution":resolution,
        })
    bundle={"schema":"RealSaS.CameraProjectionBundle.v1","cameras":cameras}
    cam_path=tmp_path/"cameras.json"; cam_path.write_text(json.dumps(bundle,sort_keys=True)+"\n",encoding="utf-8")

    policy_src=ROOT/"canonical"/"QUALIFIED_MESH_PRODUCT_POLICY_V2_20260919.json"
    policy_path=tmp_path/policy_src.name
    policy_path.write_bytes(policy_src.read_bytes())
    rest_policy_src=ROOT/"canonical"/"REST_SOURCE_PRESERVATION_POLICY_V2_20260919.json"
    rest_policy_path=tmp_path/rest_policy_src.name
    rest_policy_path.write_bytes(rest_policy_src.read_bytes())
    rest_calibration_src=ROOT/"canonical"/"REST_SOURCE_PRESERVATION_CALIBRATION_RESULT_20260919.json"
    rest_calibration_path=tmp_path/rest_calibration_src.name
    rest_calibration_path.write_bytes(rest_calibration_src.read_bytes())
    half=np.sin(np.deg2rad(5.0))
    qw=np.cos(np.deg2rad(5.0))
    motion_payload={
        "schema":"RealSaS.MotionSourceClip.v2",
        "clip_id":"idle_artist",
        "clip_kind":"IDLE",
        "source_space":"SOURCE_RIG_TRACKS_V2",
        "coordinate_frame":"REALSAS_OBJECT_FRAME_V1",
        "duration_seconds":1.0,
        "loop":True,
        "channel_contract":["LOCAL_ROTATION_QUAT_XYZW"],
        "source_skeleton":[
            {
                "source_joint_id":"source_root",
                "parent_source_joint_id":None,
                "rest_position":[0.0,0.0,0.0],
            }
        ],
        "tracks":[
            {
                "source_joint_id":"source_root",
                "keyframes":[
                    {"time_seconds":0.0,"local_rotation_quat_xyzw":[0.0,0.0,0.0,1.0]},
                    {"time_seconds":0.5,"local_rotation_quat_xyzw":[float(half),0.0,0.0,float(qw)]},
                    {"time_seconds":1.0,"local_rotation_quat_xyzw":[0.0,0.0,0.0,1.0]},
                ],
            }
        ],
    }
    motion_path=tmp_path/"idle_artist.motion.json"
    motion_path.write_text(json.dumps(motion_payload,sort_keys=True)+"\n",encoding="utf-8")
    manifest={
        "components":{},
        "carrier_policy":{},
        "deformation_envelope":{},
        "mesh_policy":{
            "document":{"path":str(policy_path.resolve()),"sha256":_sha(policy_path)}
        },
        "mesh":{"backend":"CANONICAL_RELATION_BASELINE_V1"},
        "observation":{"component_masks":[],"source_foreground_masks":[],"source_rasters":[]},
        "presentation":{"mode":"AUTO_ROLE_FREE_V1"},
        "motion":{
            "sources":[
                {
                    "clip_id":"idle_artist",
                    "clip_kind":"IDLE",
                    "source_kind":"EXTERNAL_ARTIST_CLIP_V1",
                    "file":{"path":str(motion_path),"sha256":_sha(motion_path)}
                }
            ],
            "compiler":{
                "root_trajectory_modes":{"idle_artist":"IN_PLACE"},
                "contacts":{}
            }
        },
        "appearance":{
            "rest_preservation_policy":{
                "path":str(rest_policy_path.resolve()),
                "sha256":_sha(rest_policy_path),
            },
            "rest_preservation_calibration":{
                "path":str(rest_calibration_path.resolve()),
                "sha256":_sha(rest_calibration_path),
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
        fg=rasterize_triangles_half_integer_top_left(
            (triangle,),
            width=resolution,
            height=resolution,
        )
        fg_path=tmp_path/f"source_fg_v{camera.view_index}.bin"; fg_path.write_bytes(fg)
        rgba=np.zeros((resolution,resolution,4),dtype=np.uint8)
        for yy in range(resolution):
            for xx in range(resolution):
                if resolution == 8:
                    red = xx * 20
                    green = yy * 20
                else:
                    red = int(round(180.0 * float(xx) / float(resolution - 1)))
                    green = int(round(180.0 * float(yy) / float(resolution - 1)))
                rgba[yy,xx]=[
                    red,
                    green,
                    camera.view_index*12,
                    255 if fg[yy*resolution+xx] else 0,
                ]
        source_path=tmp_path/f"source_V{camera.view_index}.png"
        Image.fromarray(rgba,"RGBA").save(source_path,format="PNG",compress_level=1,optimize=False)
        obs_hash=content_sha256({"view":camera.view_index,"fixture":"triangle","source_raster_sha256":_sha(source_path)})
        observation_views.append(QualifiedObservationViewIR(
            camera.view_index,resolution,resolution,obs_hash,
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
    assert carrier.decisions[0].metadata["automatic"] is True

    r25=seal_deformation_capability_envelope(ctx)
    assert r25["status"]=="PASS"
    _install_stage_outputs(ctx,"25_DEFORMATION_CAPABILITY_ENVELOPE",r25)

    r26=build_canonical_mesh_candidate_stage(ctx)
    assert r26["status"]=="PASS"
    _install_stage_outputs(ctx,"26_MESH_CANDIDATE_BUILD",r26)
    candidate=canonical_mesh_candidate_from_dict(read_json(r26["outputs"][0]["path"]))
    assert len(candidate.faces)==1

    component_id=partition.components[0].component_id
    assert ctx["run_manifest"]["observation"].get("component_masks")==[]

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
    _install_stage_outputs(ctx,"32_REST_SOURCE_PRESERVATION_GATE",r32)

    r33=seal_motion_source_or_preset_stage(ctx)
    assert r33["status"]=="PASS",r33
    by_schema={out["schema"]:out for out in r33["outputs"]}
    source_set=motion_source_set_from_dict(read_json(by_schema["RealSaS.MotionSourceSetIR.v1"]["path"]))
    source_seal=qualified_motion_source_seal_from_dict(read_json(by_schema["RealSaS.QualifiedMotionSourceSealIR.v1"]["path"]))
    assert len(source_set.assets)==1
    assert source_set.assets[0].clip_id=="idle_artist"
    assert source_set.assets[0].source_space=="SOURCE_RIG_TRACKS_V2"
    assert source_seal.source_set_binding_hash==source_set.source_set_hash
    assert source_seal.product_state_binding_hash==state.product_state_hash
    assert source_seal.rest_preservation_binding_hash==qualified_rest.preservation_lineage_hash
    assert source_seal.qualification_report["retargeting_performed"] is False
    assert source_seal.qualification_report["motion_compilation_performed"] is False
    assert source_seal.qualification_report["motion_quality_claimed"] is False
    _install_stage_outputs(ctx,"33_MOTION_SOURCE_OR_PRESET_SEAL",r33)

    # Stage34+ of the old V1 product path is preserved historical provenance,
    # not current continuation authority. The motion compiler core is now
    # intentionally V2-presentation-only; current Stage39-46 coverage lives in
    # test_v2_stage37_to46_tail_closes_on_subject_free_triangle_with_native_caa.


def test_v2_stage37_to46_tail_closes_on_subject_free_triangle_with_native_caa(tmp_path):
    import os
    from dataclasses import replace
    import pytest

    from compiler.realsas_compiler_core.appearance_authority_v2 import (
        AppearanceTextureIR,
        CAARestRenderProofIR,
        CAARestViewProofIR,
        CompleteAppearanceAssetIR,
        CompleteAppearanceQualificationIR,
        caa_rest_render_proof_hash,
        complete_appearance_asset_hash,
        complete_appearance_qualification_hash,
    )
    from compiler.realsas_compiler_core.output_presentation_v1 import (
        build_output_direction_set,
    )
    from compiler.realsas_compiler_services.orchestrator.adapters import mesh_v2
    from compiler.realsas_compiler_services.orchestrator.adapters.product_state_v2 import (
        qualify_presentation_structure_stage,
        seal_complete_puppet_stage,
    )
    from compiler.realsas_compiler_services.orchestrator.adapters.motion_v2 import (
        seal_motion_source_stage as seal_motion_source_stage_v2,
        compile_motion_stage as compile_motion_stage_v2,
        prove_dynamic_motion_stage,
    )
    from compiler.realsas_compiler_services.orchestrator.adapters.runtime_v2 import (
        build_runtime_projection_stage,
        materialize_runtime_package_stage,
        prove_native_package_playback_stage,
        prove_dynamic_visual_integrity_stage,
    )
    from compiler.realsas_compiler_services.orchestrator.adapters.closure_v2 import (
        seal_product_closure_stage,
    )

    ctx = _fixture(tmp_path)
    ctx["repo_root"] = ROOT

    def alias(old_id: str, new_id: str):
        old = next(row for row in ctx["ledger"]["stages"] if row["id"] == old_id)
        ctx["ledger"]["stages"].append(
            {"id": new_id, "status": "PASS", "outputs": list(old["outputs"])}
        )

    def run(stage_id: str, fn):
        ctx["stage"] = {"id": stage_id}
        result = fn(ctx)
        assert result["status"] == "PASS", result
        _install_stage_outputs(ctx, stage_id, result)
        return result

    alias("18_SKELETON_QUALIFIED", "28_SKELETON_QUALIFIED")
    alias("22_SKIN_QUALIFIED", "32_SKIN_QUALIFIED")

    r17 = run(
        "17_MECHANICAL_PARTITION_QUALIFIED",
        mesh_v2.qualify_mechanical_partition_and_carriers,
    )
    r34 = run(
        "34_DEFORMATION_CAPABILITY_ENVELOPE",
        mesh_v2.seal_deformation_capability_envelope,
    )
    r18 = run(
        "18_CANONICAL_MESH_ADDRESSING_BUILD",
        mesh_v2.build_canonical_mesh_candidate_stage,
    )
    r35 = run(
        "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        mesh_v2.qualify_canonical_mesh_stage,
    )
    r36 = run(
        "36_QUALIFIED_MESH_SKIN_TRANSFER",
        mesh_v2.bind_qualified_mesh_skin_stage,
    )

    camera_set = qualified_camera_set_from_dict(
        read_json(
            next(
                out
                for row in ctx["ledger"]["stages"]
                if row["id"] == "05_CAMERA_CONTRACT_SOLVED"
                for out in row["outputs"]
                if out["schema"] == "RealSaS.QualifiedCameraSetIR.v1"
            )["path"]
        )
    )
    directions = build_output_direction_set(camera_set)
    direction_path = _write(tmp_path / "directions.json", directions)
    ctx["ledger"]["stages"].append(
        {
            "id": "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "status": "PASS",
            "outputs": [
                _out(direction_path, "RealSaS.OutputPresentationDirectionSetIR.v1")
            ],
        }
    )

    candidate = canonical_mesh_candidate_from_dict(
        read_json(
            next(
                out
                for out in r18["outputs"]
                if out["schema"] == "RealSaS.CanonicalMeshCandidateIR.v1"
            )["path"]
        )
    )
    mesh = qualified_mesh_from_dict(
        read_json(
            next(
                out
                for out in r35["outputs"]
                if out["schema"] == "RealSaS.QualifiedMeshIR.v1"
            )["path"]
        )
    )
    assert len(mesh.faces) == 1

    appearance_root = tmp_path / "appearance_v2"
    appearance_root.mkdir(parents=True, exist_ok=True)
    texture_rows = []
    for vi in range(8):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, :] = (70 + vi * 5, 110, 150, 255)
        texture_path = appearance_root / f"V{vi}.png"
        Image.fromarray(rgba, "RGBA").save(
            texture_path, format="PNG", optimize=False, compress_level=1
        )
        texture_rows.append(
            AppearanceTextureIR(
                direction_index=vi,
                direction_id=f"V{vi}",
                transport_png_path=str(texture_path.resolve()),
                transport_png_sha256=_sha(texture_path),
                width=4,
                height=4,
                metadata={"fixture": True},
            )
        )
    face_uv = np.asarray(
        [[[0.20, 0.20], [0.80, 0.20], [0.20, 0.80]]],
        dtype=np.float32,
    )
    uv_path = appearance_root / "surface_uv.npz"
    np.savez_compressed(uv_path, face_uv=face_uv)
    provenance_path = appearance_root / "provenance.npz"
    np.savez_compressed(
        provenance_path,
        provenance=np.zeros((8, 4, 4), dtype=np.uint8),
    )
    asset = CompleteAppearanceAssetIR(
        compile_seal_binding_hash="c" * 64,
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash="a" * 64,
        appearance_domain_binding_hash="d" * 64,
        output_direction_set_binding_hash=directions.direction_set_hash,
        textures=tuple(texture_rows),
        uv_npz_path=str(uv_path.resolve()),
        uv_npz_sha256=_sha(uv_path),
        provenance_npz_path=str(provenance_path.resolve()),
        provenance_npz_sha256=_sha(provenance_path),
        atlas_layout={"fixture": True, "width": 4, "height": 4},
        asset_hash="",
        metadata={"total_appearance_asset": True, "fixture": True},
    )
    asset = replace(asset, asset_hash=complete_appearance_asset_hash(asset))
    asset_path = _write(appearance_root / "complete_asset.json", asset)
    ctx["ledger"]["stages"].append(
        {
            "id": "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "status": "PASS",
            "outputs": [_out(asset_path, "RealSaS.CompleteAppearanceAssetIR.v2")],
        }
    )

    qualification = CompleteAppearanceQualificationIR(
        asset_binding_hash=asset.asset_hash,
        preregistration_binding_hash="p" * 64,
        source_lock_exact_fraction=1.0,
        total_defined_fraction=1.0,
        structured_holdout_sample_count=128,
        structured_holdout_mean_rgba_l1=0.0,
        structured_holdout_p95_rgba_l1=0.0,
        provenance_boundary_pair_count=0,
        provenance_boundary_mean_rgba_l1=0.0,
        provenance_boundary_p95_rgba_l1=0.0,
        provenance_boundary_gradient_pair_count=0,
        provenance_boundary_mean_gradient_jump=0.0,
        provenance_boundary_p95_gradient_jump=0.0,
        qualification_report={
            "status": "PASS_COMPLETE_APPEARANCE",
            "source_lock_passed": True,
            "totality_passed": True,
            "holdout_passed": True,
            "seam_passed": True,
        },
        qualification_hash="",
        metadata={
            "policy": {
                "dynamic_max_compiled_unobserved_visible_fraction": 0.20,
                "dynamic_max_frame_compiled_unobserved_visible_fraction": 1.0,
                "dynamic_max_connected_compiled_unobserved_visible_fraction": 1.0,
                "dynamic_max_micro_visible_pixel_fraction_per_frame": 1.0,
                "dynamic_max_unmeasurable_consequential_visible_face_count": 1000000,
                "dynamic_max_exact_depth_ambiguous_fraction": 1.0,
                "dynamic_max_visible_orientation_flip_face_count": 1000000,
                "dynamic_max_visibility_layer_overflow_pixel_count": 0,
                "dynamic_min_visible_pixels_per_face": 1,
                "dynamic_min_projected_double_area_px2": 0.01,
                "dynamic_max_uv_to_surface_condition_number": 64.0,
                "dynamic_max_relative_surface_condition_number": 16.0,
                "dynamic_max_relative_surface_principal_stretch": 8.0,
                "dynamic_max_adjacent_frame_surface_principal_stretch": 8.0,
                "dynamic_max_texture_texels_per_output_pixel": 1.0
            },
            "fixture": True,
        },
    )
    qualification = replace(
        qualification,
        qualification_hash=complete_appearance_qualification_hash(qualification),
    )
    qualification_path = _write(
        appearance_root / "complete_qualification.json", qualification
    )
    ctx["ledger"]["stages"].append(
        {
            "id": "24_COMPLETE_APPEARANCE_QUALIFIED",
            "status": "PASS",
            "outputs": [
                _out(
                    qualification_path,
                    "RealSaS.CompleteAppearanceQualificationIR.v2",
                )
            ],
        }
    )

    proof_views = tuple(
        CAARestViewProofIR(
            direction_index=vi,
            rendered_rgba_sha256=texture_rows[vi].transport_png_sha256,
            rendered_alpha_pixel_count=1,
            source_locked_pixel_count=1,
            source_locked_fraction_of_source_foreground=1.0,
            source_locked_exact_pixel_count=1,
            source_locked_exact_fraction=1.0,
            source_locked_mean_rgba_l1=0.0,
            source_locked_p95_rgba_l1=0.0,
            source_foreground_mean_rgba_l1=0.0,
            source_foreground_p95_rgba_l1=0.0,
            source_feature_p999_rgba_l1=0.0,
            source_feature_high_error_fraction=0.0,
            largest_connected_feature_high_error_fraction=0.0,
            source_feature_edge_recall_1px=1.0,
            source_feature_edge_precision_1px=1.0,
            geometry_visible_pixel_count=1,
            final_alpha_pixel_count=1,
            geometry_visible_final_alpha_hole_count=0,
            geometry_visible_final_alpha_hole_fraction=0.0,
            source_alpha_recall=1.0,
            source_alpha_precision=1.0,
            largest_coherent_alpha_hole_fraction=0.0,
            alpha_interior_uncovered_fraction=0.0,
            metadata={"status": "PASS", "fixture": True},
        )
        for vi in range(8)
    )
    rest_proof = CAARestRenderProofIR(
        asset_binding_hash=asset.asset_hash,
        static_mesh_qualification_binding_hash="s" * 64,
        camera_set_binding_hash=camera_set.camera_set_hash,
        views=proof_views,
        qualification_report={
            "status": "PASS_CAA_REFERENCE_REST",
            "every_direction_passed": True,
        },
        proof_hash="",
        metadata={"fixture": True},
    )
    rest_proof = replace(
        rest_proof, proof_hash=caa_rest_render_proof_hash(rest_proof)
    )
    rest_proof_path = _write(appearance_root / "rest_proof.json", rest_proof)
    ctx["ledger"]["stages"].append(
        {
            "id": "25_CAA_REFERENCE_REST_RENDER_PROOF",
            "status": "PASS",
            "outputs": [_out(rest_proof_path, "RealSaS.CAARestRenderProofIR.v2")],
        }
    )

    presentation_policy = (
        ROOT / "canonical" / "PRESENTATION_PARTITION_POLICY_V1_20260921.json"
    )
    ctx["run_manifest"]["presentation"] = {
        "mode": "AUTO_ROLE_FREE_V2",
        "policy_document": {
            "path": str(presentation_policy.resolve()),
            "sha256": _sha(presentation_policy),
        },
    }

    r37 = run(
        "37_QUALIFIED_PRESENTATION_STRUCTURE",
        qualify_presentation_structure_stage,
    )
    r38 = run("38_CANONICAL_PUPPET_SEALED", seal_complete_puppet_stage)
    by38 = {out["schema"]: out for out in r38["outputs"]}
    assert "RealSaS.CompletePuppetStateIR.v2" in by38
    assert "RealSaS.QualifiedPresentationGraphIR.v2" in by38

    r39 = run("39_MOTION_SOURCE_OR_PRESET_SEAL", seal_motion_source_stage_v2)
    r40 = run("40_MOTION_COMPILE_RUN", compile_motion_stage_v2)
    r41 = run("41_MOTION_DYNAMIC_PROOF", prove_dynamic_motion_stage)
    dynamic_payload = read_json(r41["outputs"][0]["path"])
    assert dynamic_payload["qualification_report"]["dynamic_proof_passed"] is True
    assert dynamic_payload["qualification_report"]["any_clip_nonzero"] is True
    assert (
        dynamic_payload["qualification_report"]["rest_unseen_dynamic_exposure_gate_removed"]
        is True
    )
    assert (
        dynamic_payload["qualification_report"]["all_declared_plant_2d_contacts_satisfied"]
        is True
    )
    assert dynamic_payload["qualification_report"]["supported_contact_modes"] == [
        "PLANT_2D"
    ]
    assert (
        dynamic_payload["qualification_report"]["full_3d_ground_contact_quality_claimed"]
        is False
    )
    assert "all_contacts_satisfied" not in dynamic_payload["qualification_report"]

    player_raw = os.environ.get("REALSAS_RUNTIME_V2_PLAYER", "")
    if not player_raw:
        pytest.skip("native V2 player is required on self-hosted product integration gate")
    player = Path(player_raw).expanduser().resolve()
    assert player.is_file(), player
    ctx["run_manifest"]["runtime"] = {
        "native_player": {"path": str(player), "sha256": _sha(player)}
    }

    r42 = run(
        "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
        build_runtime_projection_stage,
    )
    r43 = run("43_RSS_MATERIALIZE_COMPACT", materialize_runtime_package_stage)
    r44 = run(
        "44_NATIVE_PACKAGE_OPEN_PLAYBACK",
        prove_native_package_playback_stage,
    )
    assert r44["diagnostics"]["native_reference_mismatch_pixels"] == 0
    r45 = run(
        "45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
        prove_dynamic_visual_integrity_stage,
    )
    assert r45["diagnostics"]["native_reference_mismatch_pixel_count"] == 0
    assert r45["diagnostics"]["undefined_visible_pixel_count"] == 0
    assert r45["diagnostics"]["compiled_unobserved_visible_fraction"] == 0.0
    assert r45["diagnostics"]["dynamic_conditioning_sample_count"] > 0
    assert r45["diagnostics"]["maximum_relative_surface_condition_number"] >= 1.0
    assert r45["diagnostics"]["maximum_relative_surface_principal_stretch"] >= 1.0

    r46 = run("46_PRODUCT_CLOSURE_SEAL", seal_product_closure_stage)
    closure = read_json(
        next(
            out
            for out in r46["outputs"]
            if out["schema"] == "RealSaS.ProductClosureIR.v2"
        )["path"]
    )
    assert closure["qualification_report"]["product_pass"] is True
    assert closure["qualification_report"]["appearance_authority_passed"] is True
    assert closure["qualification_report"]["presentation_partition_authority_passed"] is True
    assert closure["qualification_report"]["dynamic_appearance_conditioning_passed"] is True
    assert closure["qualification_report"]["interior_shared_edge_continuity_passed"] is True
    assert closure["qualification_report"]["cross_component_crack_authority_claimed"] is False
    assert closure["qualification_report"]["native_visual_integrity_passed"] is True
    editable = next(
        out
        for out in r46["outputs"]
        if out["schema"] == "application/x-realsas-editable-v2"
    )
    assert Path(editable["path"]).is_file()
    assert _sha(Path(editable["path"])) == editable["sha256"]


def test_vf23_stage20_to25_uses_unmodified_production_caa_policy(tmp_path):
    from dataclasses import replace

    from compiler.realsas_compiler_core.geometry_substrate_v2 import (
        GeometrySubstrateViewIR,
        build_geometry_substrate_qualification,
    )
    from compiler.realsas_compiler_core.output_presentation_v1 import (
        build_output_direction_set,
    )
    from compiler.realsas_compiler_core.appearance_authority_v2 import (
        caa_compile_artifact_from_dict,
        caa_compile_seal_from_dict,
        complete_appearance_asset_from_dict,
        complete_appearance_qualification_from_dict,
        caa_rest_render_proof_from_dict,
    )
    from compiler.realsas_compiler_services.orchestrator.adapters import mesh_v2
    from compiler.realsas_compiler_services.orchestrator.adapters.v2_architecture import (
        build_canonical_mesh_addressing_stage,
        qualify_static_canonical_mesh_stage,
    )
    from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
        preregister_caa_backend_stage,
        compile_caa_stage,
        seal_caa_compile_stage,
        bake_complete_appearance_stage,
        qualify_complete_appearance_stage,
        prove_caa_reference_rest_stage,
    )

    ctx = _fixture(tmp_path, resolution=128)
    ctx["repo_root"] = ROOT

    def run(stage_id: str, fn):
        ctx["stage"] = {"id": stage_id}
        result = fn(ctx)
        assert result["status"] == "PASS", result
        _install_stage_outputs(ctx, stage_id, result)
        return result

    # Stage16 output directions are a distinct authority even though this tiny
    # first-witness fixture uses the same eight cameras for observations.
    cameras = qualified_camera_set_from_dict(
        read_json(
            next(
                out
                for row in ctx["ledger"]["stages"]
                if row["id"] == "05_CAMERA_CONTRACT_SOLVED"
                for out in row["outputs"]
                if out["schema"] == "RealSaS.QualifiedCameraSetIR.v1"
            )["path"]
        )
    )
    directions = build_output_direction_set(cameras)
    directions_path = _write(tmp_path / "directions_v2.json", directions)
    ctx["ledger"]["stages"].append(
        {
            "id": "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "status": "PASS",
            "outputs": [
                _out(
                    directions_path,
                    "RealSaS.OutputPresentationDirectionSetIR.v1",
                )
            ],
        }
    )

    run(
        "17_MECHANICAL_PARTITION_QUALIFIED",
        mesh_v2.qualify_mechanical_partition_and_carriers,
    )
    r18 = run(
        "18_CANONICAL_MESH_ADDRESSING_BUILD",
        build_canonical_mesh_addressing_stage,
    )
    candidate = canonical_mesh_candidate_from_dict(
        read_json(
            next(
                out
                for out in r18["outputs"]
                if out["schema"] == "RealSaS.CanonicalMeshCandidateIR.v1"
            )["path"]
        )
    )
    assert len(candidate.faces) == 1

    # Stage19 owns only static carrier/addressability and inherits Stage13's
    # already-qualified geometry claim. For this CAA adapter integration test,
    # provide a subject-free exact synthetic Stage13 PASS rather than invoking
    # IRIS or using a witness.
    observation = qualified_observation_set_from_dict(
        read_json(
            next(
                out
                for row in ctx["ledger"]["stages"]
                if row["id"] == "07_OBSERVATION_CONTRACT_QUALIFIED"
                for out in row["outputs"]
                if out["schema"] == "RealSaS.QualifiedObservationSetIR.v1"
            )["path"]
        )
    )
    geometry_views = tuple(
        GeometrySubstrateViewIR(
            view_index=view,
            silhouette_recall=1.0,
            silhouette_precision=1.0,
            largest_coherent_hole_fraction=0.0,
            interior_uncovered_fraction=0.0,
            source_foreground_pixel_count=1,
            predicted_foreground_pixel_count=1,
            component_recall=1.0,
            silhouette_edge_p95_px=0.0,
            passed=True,
            metadata={"subject_free_fixture": True},
        )
        for view in range(8)
    )
    geometry = build_geometry_substrate_qualification(
        zero_surface_binding_hash="z" * 64,
        observation_set_binding_hash=observation.observation_set_hash,
        camera_set_binding_hash=cameras.camera_set_hash,
        normalization_binding_hash="n" * 64,
        policy={"fixture": True},
        views=geometry_views,
        metadata={"subject_free_fixture": True},
    )
    geometry_path = _write(tmp_path / "geometry_substrate_v2.json", geometry)
    ctx["ledger"]["stages"].append(
        {
            "id": "13_GEOMETRY_SUBSTRATE_QUALIFIED",
            "status": "PASS",
            "outputs": [
                _out(
                    geometry_path,
                    "RealSaS.GeometrySubstrateQualificationIR.v2",
                )
            ],
        }
    )
    run(
        "19_STATIC_CANONICAL_MESH_QUALIFIED",
        qualify_static_canonical_mesh_stage,
    )

    # VF-23: consume the canonical production policy bytes directly. No clone,
    # status rewrite, scale accommodation, or threshold override is permitted.
    policy_path = (
        ROOT
        / "canonical"
        / "CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"
    )
    production_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert production_policy["status"] == "FROZEN_SUBJECT_FREE_VISUAL_FIDELITY_V3"
    ctx["run_manifest"]["appearance"] = {
        "backend": "DETERMINISTIC_V1",
        "policy_document": {
            "path": str(policy_path.resolve()),
            "sha256": _sha(policy_path),
        },
    }

    r20 = run("20_CAA_BACKEND_PREREGISTERED", preregister_caa_backend_stage)
    assert r20["diagnostics"]["shipping_eligible"] is True
    r21 = run("21_CAA_COMPILE", compile_caa_stage)
    compile_artifact = caa_compile_artifact_from_dict(
        read_json(
            next(
                out
                for out in r21["outputs"]
                if out["schema"] == "RealSaS.CAACompileArtifactIR.v2"
            )["path"]
        )
    )
    assert compile_artifact.total_sample_count > 0
    assert (
        compile_artifact.direct_source_sample_count
        + compile_artifact.other_view_source_sample_count
        + compile_artifact.compiled_local_harmonic_sample_count
        == compile_artifact.total_sample_count
    )

    r22 = run("22_CAA_COMPILE_SEALED", seal_caa_compile_stage)
    seal = caa_compile_seal_from_dict(
        read_json(
            next(
                out
                for out in r22["outputs"]
                if out["schema"] == "RealSaS.CAACompileSealIR.v2"
            )["path"]
        )
    )
    assert seal.qualification_report["total_appearance_defined"] is True
    assert seal.qualification_report["direct_source_immutable"] is True

    r23 = run("23_COMPLETE_APPEARANCE_ASSET_BAKED", bake_complete_appearance_stage)
    asset = complete_appearance_asset_from_dict(
        read_json(
            next(
                out
                for out in r23["outputs"]
                if out["schema"] == "RealSaS.CompleteAppearanceAssetIR.v2"
            )["path"]
        )
    )
    assert len(asset.textures) == 8
    assert asset.metadata["runtime_generation_forbidden"] is True
    assert asset.atlas_layout["width"] <= production_policy["compile_policy"]["max_atlas_resolution"]
    assert asset.atlas_layout["height"] <= production_policy["compile_policy"]["max_atlas_resolution"]

    r24 = run("24_COMPLETE_APPEARANCE_QUALIFIED", qualify_complete_appearance_stage)
    qualification = complete_appearance_qualification_from_dict(
        read_json(
            next(
                out
                for out in r24["outputs"]
                if out["schema"] == "RealSaS.CompleteAppearanceQualificationIR.v2"
            )["path"]
        )
    )
    assert qualification.qualification_report["status"] == "PASS_COMPLETE_APPEARANCE"
    assert qualification.source_lock_exact_fraction == 1.0
    assert qualification.total_defined_fraction == 1.0

    r25 = run("25_CAA_REFERENCE_REST_RENDER_PROOF", prove_caa_reference_rest_stage)
    rest = caa_rest_render_proof_from_dict(
        read_json(
            next(
                out
                for out in r25["outputs"]
                if out["schema"] == "RealSaS.CAARestRenderProofIR.v2"
            )["path"]
        )
    )
    assert rest.qualification_report["status"] == "PASS_CAA_REFERENCE_REST"
    assert rest.qualification_report["every_direction_passed"] is True
    assert len(rest.views) == 8
    assert all(row.geometry_visible_pixel_count > 0 for row in rest.views)
