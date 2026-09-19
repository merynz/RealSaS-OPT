from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import QualifiedObservationViewIR, build_qualified_observation_set
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
    component_carrier_policy_from_dict,
    mechanical_partition_from_dict,
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
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
            SurfaceNode("s0",(0.0,0.0,0.0),tuple(range(8)),("p0",),("o0",)),
            SurfaceNode("s1",(1.0,0.0,0.0),tuple(range(8)),("p1",),("o1",)),
            SurfaceNode("s2",(0.0,1.0,0.0),tuple(range(8)),("p2",),("o2",)),
        ),
        (
            SurfaceRelation("r01","s0","s1","LOCAL",1.0),
            SurfaceRelation("r12","s1","s2","LOCAL",1.0),
            SurfaceRelation("r02","s0","s2","LOCAL",1.0),
        ),
        "surface-hash",
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
            "camera_bundle":{"path":str(cam_path),"sha256":_sha(cam_path)},
            "joint_ranges":[
                {"canonical_joint_id":"j0","min_rotation_deg":0.0,"max_rotation_deg":0.0}
            ],
            "allowed_attachment_state_hashes":[],
        },
        "mesh_policy":{
            "document":{"path":str(policy_path.resolve()),"sha256":_sha(policy_path)}
        },
        "mesh":{"backend":"CANONICAL_RELATION_BASELINE_V1"},
        "observation":{"component_masks":[],"source_foreground_masks":[]},
    }

    observation_views=[]
    source_foreground_rows=[]
    for row in cameras:
        camera=qualify_camera_v3(row,view_id=row["view_id"],view_index=int(row["view_index"]))
        # The baseline candidate is the same single triangle as S, so source foreground
        # is materialized directly from exact S positions and the same camera.
        from compiler.realsas_compiler_core.playback_full_surface_v3 import project_points_xyz_v3
        projected=project_points_xyz_v3([node.P for node in surface.surface_nodes],camera)
        triangle=tuple((float(p[0]),float(p[1])) for p in projected)
        fg=rasterize_triangles_half_integer_top_left((triangle,),width=8,height=8)
        fg_path=tmp_path/f"source_fg_v{camera.view_index}.bin"; fg_path.write_bytes(fg)
        obs_hash=content_sha256({"view":camera.view_index,"fixture":"triangle"})
        observation_views.append(QualifiedObservationViewIR(
            camera.view_index,8,8,obs_hash,_sha(fg_path),camera_projection_binding_hash(camera),
            "PASS",(f"fixture-observation-{camera.view_index}",),
        ))
        source_foreground_rows.append({
            "view_index":camera.view_index,
            "mask":{"path":str(fg_path),"sha256":_sha(fg_path)},
        })
    observation_set=build_qualified_observation_set(tuple(observation_views))
    obs_path=_write(tmp_path/"observation_set.json",observation_set)
    manifest["observation"]["source_foreground_masks"]=source_foreground_rows

    ledger={"stages":[
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
    camera_bundle=json.loads(Path(ctx["run_manifest"]["deformation_envelope"]["camera_bundle"]["path"]).read_text())
    mask_rows=[]
    for row in camera_bundle["cameras"]:
        camera=qualify_camera_v3(row,view_id=row["view_id"],view_index=int(row["view_index"]))
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
