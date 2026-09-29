from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, deformation_envelope_from_dict,
    mesh_policy_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    G3_LOCAL_MICRO_STRESS_ANGLE_DEG, _pose_skin_matrices,
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import _triangle_metrics_batch
from compiler.realsas_compiler_core.motion_3d_v1 import apply_lbs_matrix_v1
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import load_json
from tools.demo.render_knight_motion_preview_v1 import _ctx


def vectorized_g3(candidate, *, surface, skeleton, skin, envelope, cameras, policy):
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    rest,weights,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    probes=[("REST",None,None,0.0)]
    for jid in sorted(joint_ids):
        for axis_index,axis_name in enumerate(("X","Y","Z")):
            for sign in (-1.0,1.0):
                d=sign*G3_LOCAL_MICRO_STRESS_ANGLE_DEG
                probes.append((f"{jid}:LOCAL_{axis_name}:{d:+g}",jid,axis_index,d))

    rows=[]; failures=set()
    for probe_id,jid,axis_index,degrees in probes:
        skin_by_id=_pose_skin_matrices(
            skeleton,frames,joint_id=jid,local_axis_index=axis_index,degrees=degrees)
        matrices=np.stack([skin_by_id[x] for x in joint_ids],axis=0)
        posed=apply_lbs_matrix_v1(rest,weights,matrices)
        area,cond,emin,emax=_triangle_metrics_batch(rest,posed,faces)
        finite=np.isfinite(area)&np.isfinite(cond)&np.isfinite(emin)&np.isfinite(emax)
        if not np.all(finite):
            failures.add("NONFINITE_DEFORMATION_METRIC")
        a=area[finite]; c=cond[finite]; mi=emin[finite]; ma=emax[finite]
        if len(a):
            if np.any(a<policy.g3_min_dynamic_area_ratio): failures.add("DYNAMIC_AREA_RATIO_BELOW_MIN")
            if np.any(a>policy.g3_max_dynamic_area_ratio): failures.add("DYNAMIC_AREA_RATIO_ABOVE_MAX")
            if np.any(c>policy.g3_max_dynamic_condition_number): failures.add("DYNAMIC_CONDITION_NUMBER_ABOVE_MAX")
            rows.append({
                "probe_id":probe_id,
                "minimum_area_ratio":float(np.min(a)),
                "maximum_area_ratio":float(np.max(a)),
                "maximum_condition_number":float(np.max(c)),
                "minimum_edge_ratio":float(np.min(mi)),
                "maximum_edge_ratio":float(np.max(ma)),
            })
        else:
            rows.append({
                "probe_id":probe_id,"minimum_area_ratio":float("inf"),
                "maximum_area_ratio":0.0,"maximum_condition_number":0.0,
                "minimum_edge_ratio":float("inf"),"maximum_edge_ratio":0.0})
    return {
        "probe_count":len(rows),
        "face_count":len(faces),
        "failure_invariants":sorted(failures),
        "passed":not failures,
        "minimum_area_ratio":min(r["minimum_area_ratio"] for r in rows),
        "maximum_area_ratio":max(r["maximum_area_ratio"] for r in rows),
        "maximum_condition_number":max(r["maximum_condition_number"] for r in rows),
        "minimum_edge_ratio":min(r["minimum_edge_ratio"] for r in rows),
        "maximum_edge_ratio":max(r["maximum_edge_ratio"] for r in rows),
        "per_probe":rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    candidate=canonical_mesh_candidate_from_dict(load_json(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json"))
    surface=rigging_surface_from_dict(load_json(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    skeleton=qualified_skeleton_from_dict(load_json(rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
    skin=qualified_skin_from_dict(load_json(rr/"artifacts/32_SKIN_QUALIFIED/qualified_skin.json"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(load_json(rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json")).cameras,key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(load_json(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"))
    policy=mesh_policy_from_dict(load_json(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))

    t0=time.perf_counter()
    legacy=run_g3_local_frame_micro_stress_v2(candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy)
    legacy_s=time.perf_counter()-t0
    t1=time.perf_counter()
    fast=vectorized_g3(candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy)
    fast_s=time.perf_counter()-t1

    keys=("minimum_area_ratio","maximum_area_ratio","maximum_condition_number","minimum_edge_ratio","maximum_edge_ratio")
    global_delta={k:abs(float(getattr(legacy,k))-float(fast[k])) for k in keys}
    legacy_rows={r["probe_id"]:r for r in legacy.per_probe}
    fast_rows={r["probe_id"]:r for r in fast["per_probe"]}
    per_probe_max_delta=0.0
    for pid,row in legacy_rows.items():
        f=fast_rows[pid]
        for k in keys:
            per_probe_max_delta=max(per_probe_max_delta,abs(float(row[k])-float(f[k])))
    equivalent=(
        legacy.probe_count==fast["probe_count"] and
        legacy.face_count==fast["face_count"] and
        list(legacy.failure_invariants)==fast["failure_invariants"] and
        bool(legacy.passed)==bool(fast["passed"]) and
        per_probe_max_delta<=1e-10
    )
    report={
      "schema":"RealSaS.G3VectorizationEquivalenceBenchmark.v1",
      "status":"PASS" if equivalent else "FAIL",
      "candidate_face_count":legacy.face_count,
      "probe_count":legacy.probe_count,
      "legacy_seconds":legacy_s,
      "vectorized_seconds":fast_s,
      "speedup_x":legacy_s/max(fast_s,1e-12),
      "equivalent":equivalent,
      "global_metric_abs_delta":global_delta,
      "per_probe_max_abs_delta":per_probe_max_delta,
      "legacy_failures":list(legacy.failure_invariants),
      "vectorized_failures":fast["failure_invariants"],
      "claim_boundary":"Audit-only benchmark. No production operator mutation."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("G3_VECTOR_EQ="+json.dumps(report,sort_keys=True))
    if not equivalent: raise SystemExit(2)

if __name__=="__main__": main()
