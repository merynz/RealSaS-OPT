from __future__ import annotations

"""Localize exact intrinsic failures without changing mesh, rig, weights or gates."""

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, rigging_surface_from_dict,
    qualified_skin_from_dict, qualified_skeleton_from_dict,
    qualified_camera_set_from_dict, mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from compiler.realsas_compiler_core.mesh.dynamic_frame_court_v1 import measure_dynamic_frame_geometry_v1
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import CLIPS,FULL_MOTION_SAMPLES
from tools.audit_knight_v9_source_fidelity_v1 import sha
from tools.demo.render_knight_motion_preview_v1 import _ctx,_skin,_tracks_for_clip


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--phase1-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    ctx=_ctx(args.authority_root,args.run_id);rr=ctx["run_root"]
    parent=read(args.phase1_dir/"REPORT.json")
    g3=read(args.phase1_dir/"g3_report.json")
    candidate=canonical_mesh_candidate_from_dict(read(args.candidate_json))
    surface=rigging_surface_from_dict(read(args.phase1_dir/"refined_surface.json"))
    skin=qualified_skin_from_dict(read(args.phase1_dir/"transferred_skin.json"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda c:c.view_index))
    policy=mesh_policy_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))
    expected={"candidate_lineage_hash":candidate.candidate_lineage_hash,
              "surface_lineage_hash":surface.geometry_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash}
    for name,value in expected.items():
        if parent[name]!=value or g3[name]!=value:
            raise RuntimeError("DYNAMIC_OWNER_PARENT_BINDING_DRIFT:"+name)
    if g3["skeleton_lineage_hash"]!=skeleton.skeleton_lineage_hash or g3["qualification_policy_hash"]!=policy.qualification_policy_lineage_hash:
        raise RuntimeError("DYNAMIC_OWNER_RIG_OR_POLICY_DRIFT")
    rest,W,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)

    def details(result):
        for row in result["worst_faces"]:
            indices=faces[row["face_index"]]
            row["vertex_ids"]=[candidate.vertices[int(i)].candidate_vertex_id for i in indices]
            row["support_bindings"]=[candidate.vertices[int(i)].support_binding.to_dict() for i in indices]
            row["dominant_joints"]=[joint_ids[int(np.argmax(W[int(i)]))] for i in indices]
            row["max_pairwise_weight_l1"]=max(float(np.sum(np.abs(W[int(indices[i])]-W[int(indices[j])])) )
                                              for i,j in ((0,1),(1,2),(2,0)))
        return result

    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    # Reconstruct every failed micro probe, retaining exact offending faces.
    stress=[]
    for row in g3["per_probe"]:
        if (row["minimum_area_ratio"]>=policy.g3_min_dynamic_area_ratio
                and row["maximum_area_ratio"]<=policy.g3_max_dynamic_area_ratio
                and row["maximum_condition_number"]<=policy.g3_max_dynamic_condition_number):
            continue
        mats=_pose_skin_matrices(skeleton,frames,joint_id=row["joint_id"],
                                local_axis_index=row["local_axis_index"],degrees=row["rotation_degrees"])
        posed=_skin(rest,W,joint_ids,mats)
        stress.append({"probe_id":row["probe_id"],"geometry":details(measure_dynamic_frame_geometry_v1(
            rest=rest,posed=posed,faces=faces,policy=policy))})
    (out/"micro_stress_failures.json").write_text(json.dumps(stress,sort_keys=True)+"\n")
    source_report=read("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json")
    results=[];poses=[];locators=[];motion_hashes={}
    for clip in CLIPS:
        path=rr/"inputs/motion/quaternius_knight_v1"/(clip+".motion.json")
        payload=read(path);motion_hashes[clip]=sha(path)
        tracks,_mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.,float(payload["duration_seconds"]),FULL_MOTION_SAMPLES,endpoint=not bool(payload.get("loop")))
        for index,t in enumerate(times):
            mats,_positions,frame_hash=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            # Fail if the follow-up silently uses a different motion apparatus.
            parent_clip=next(c for c in parent["actual_motion"]["clips"] if c["clip_id"]==clip)
            if parent_clip["frames"][index]["motion_frame_hash"]!=frame_hash:
                raise RuntimeError("DYNAMIC_OWNER_MOTION_FRAME_DRIFT")
            posed=_skin(rest,W,joint_ids,mats)
            locator={"clip_id":clip,"frame_index":index,"time_seconds":float(t),"motion_frame_hash":frame_hash}
            results.append({**locator,"geometry":details(measure_dynamic_frame_geometry_v1(rest=rest,posed=posed,faces=faces,policy=policy))})
            poses.append(posed);locators.append(locator)
        print("DYNAMIC_OWNER_CLIP="+json.dumps({"clip":clip,"failed_frames":sum(not x["geometry"]["passed"] for x in results if x["clip_id"]==clip)}),flush=True)
    bundle=out/"sampled_geometry.npz"
    np.savez_compressed(bundle,rest=rest,faces=faces,weights=W,poses=np.asarray(poses))
    report={"schema":"RealSaS.DynamicFrameOwnerAudit.v1",**expected,
            "parent_report_sha256":sha(args.phase1_dir/"REPORT.json"),
            "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
            "camera_set_hash":camera_set.camera_set_hash,"motion_source_sha256":motion_hashes,
            "policy_hash":policy.qualification_policy_lineage_hash,"frame_count":len(results),
            "failed_micro_probe_count":len(stress),"failed_motion_frame_count":sum(not r["geometry"]["passed"] for r in results),
            "motion":results,"sampled_geometry_sha256":sha(bundle),"frame_locators":locators,
            "phase1_intersections":parent["self_intersection_worst_frame"],
            "product_authority_minted":False,"exhaustive_intersection_proof":False}
    (out/"REPORT.json").write_text(json.dumps(report,sort_keys=True)+"\n")
    print("DYNAMIC_FRAME_OWNER_AUDIT="+json.dumps({k:report[k] for k in ("frame_count","failed_micro_probe_count","failed_motion_frame_count")}),flush=True)
    if stress or report["failed_motion_frame_count"]:
        raise SystemExit(2)


if __name__=="__main__":main()
