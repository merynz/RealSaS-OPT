"""Measure mesh-edge skin-field discontinuity on archived vs V9 carriers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.demo.render_knight_motion_preview_v1 import _ctx

def read(path): return json.loads(Path(path).read_text())

def unique_edges(faces):
    edges=set()
    for a,b,c in faces:
        edges.add(tuple(sorted((int(a),int(b)))))
        edges.add(tuple(sorted((int(b),int(c)))))
        edges.add(tuple(sorted((int(c),int(a)))))
    return np.asarray(sorted(edges),dtype=np.int64)

def q(a,p): return float(np.quantile(np.asarray(a,dtype=np.float64),p))

def analyze(label,candidate,surface,skeleton,skin):
    rest,weights,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    faces=np.asarray(faces,dtype=np.int64); edges=unique_edges(faces)
    length=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    span=float(np.max(rest.max(axis=0)-rest.min(axis=0)))
    if not np.isfinite(span) or span<=1e-12 or np.any(length<=1e-12):
        raise RuntimeError(f"{label}:INVALID_EDGE_OR_SPAN")
    lnorm=length/span
    wdelta=np.abs(weights[edges[:,0]]-weights[edges[:,1]]).sum(axis=1)
    grad=wdelta/lnorm
    dom=np.argmax(weights,axis=1); domdiff=dom[edges[:,0]]!=dom[edges[:,1]]
    short_cut=q(lnorm,0.01); short=lnorm<=short_cut; tiny=lnorm<1e-3; hard=wdelta>1.0
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    top=[]
    for idx in np.argsort(grad)[::-1][:32]:
        a,b=map(int,edges[idx])
        top.append({
            "edge_index":int(idx),"vertex_a":a,"vertex_b":b,
            "normalized_length":float(lnorm[idx]),"weight_l1":float(wdelta[idx]),
            "weight_gradient":float(grad[idx]),
            "dominant_joint_a":joint_ids[int(dom[a])],"dominant_joint_b":joint_ids[int(dom[b])],
            "dominant_changed":bool(domdiff[idx]),
            "dominant_weight_a":float(weights[a,int(dom[a])]),
            "dominant_weight_b":float(weights[b,int(dom[b])]),
        })
    return {
        "label":label,"vertex_count":int(len(rest)),"face_count":int(len(faces)),
        "edge_count":int(len(edges)),"body_span":span,
        "normalized_edge_length":{"p001":q(lnorm,.001),"p01":q(lnorm,.01),"p05":q(lnorm,.05),
            "p50":q(lnorm,.5),"p95":q(lnorm,.95),"max":float(np.max(lnorm))},
        "weight_l1":{"p50":q(wdelta,.5),"p90":q(wdelta,.9),"p95":q(wdelta,.95),
            "p99":q(wdelta,.99),"p999":q(wdelta,.999),"max":float(np.max(wdelta))},
        "weight_gradient_l1_per_normalized_length":{"p50":q(grad,.5),"p90":q(grad,.9),
            "p95":q(grad,.95),"p99":q(grad,.99),"p999":q(grad,.999),"max":float(np.max(grad))},
        "dominant_joint_change_edge_fraction":float(np.mean(domdiff)),
        "shortest_one_percent":{"normalized_length_cutoff":short_cut,
            "edge_count":int(np.count_nonzero(short)),"weight_l1_p95":q(wdelta[short],.95),
            "weight_l1_max":float(np.max(wdelta[short])),"weight_gradient_p95":q(grad[short],.95),
            "weight_gradient_max":float(np.max(grad[short])),
            "dominant_joint_change_fraction":float(np.mean(domdiff[short]))},
        "tiny_edge_lt_1e_3_span":{"edge_count":int(np.count_nonzero(tiny)),
            "hard_weight_l1_gt_1_count":int(np.count_nonzero(tiny & hard)),
            "weight_l1_max":float(np.max(wdelta[tiny])) if np.any(tiny) else None,
            "weight_gradient_max":float(np.max(grad[tiny])) if np.any(tiny) else None},
        "weight_l1_gt":{"0_5":int(np.count_nonzero(wdelta>.5)),
            "1_0":int(np.count_nonzero(wdelta>1.0)),"1_5":int(np.count_nonzero(wdelta>1.5))},
        "top_gradient_edges":top,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True); ap.add_argument("--run-id",required=True)
    ap.add_argument("--v9-surface-json",type=Path,required=True); ap.add_argument("--v9-candidate-json",type=Path,required=True)
    ap.add_argument("--v9-inference-dir",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); ctx=_ctx(a.authority_root,a.run_id)
    old_surface=rigging_surface_from_dict(stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    old_candidate=canonical_mesh_candidate_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    old_skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    old_skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    v9_surface=rigging_surface_from_dict(read(a.v9_surface_json)); v9_candidate=canonical_mesh_candidate_from_dict(read(a.v9_candidate_json))
    v9_skeleton=qualified_skeleton_from_dict(read(a.v9_inference_dir/"fresh_qualified_skeleton.json"))
    v9_skin=qualified_skin_from_dict(read(a.v9_inference_dir/"fresh_qualified_skin.json"))
    old=analyze("ARCHIVED_STAGE18_STAGE32",old_candidate,old_surface,old_skeleton,old_skin)
    v9=analyze("V9_FRESH_CUDA_BF16",v9_candidate,v9_surface,v9_skeleton,v9_skin)
    og=old["weight_gradient_l1_per_normalized_length"]; ng=v9["weight_gradient_l1_per_normalized_length"]
    ratio={k:(float(ng[k]/og[k]) if float(og[k])>1e-12 else None) for k in ("p50","p90","p95","p99","p999","max")}
    report={"schema":"RealSaS.SkinFieldEdgeDiscontinuityAudit.v1","status":"MEASURED_DIAGNOSTIC_ONLY",
        "product_authority_minted":False,"training_used":False,
        "metric":"L1_SKIN_WEIGHT_DELTA_PER_BODYSPAN_NORMALIZED_MESH_EDGE_LENGTH",
        "archived":old,"v9_fresh":v9,"v9_over_archived_gradient_ratio":ratio}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SKIN_FIELD_EDGE_AUDIT="+json.dumps({
        "archived_gradient_p99":og["p99"],"archived_gradient_p999":og["p999"],"archived_gradient_max":og["max"],
        "v9_gradient_p99":ng["p99"],"v9_gradient_p999":ng["p999"],"v9_gradient_max":ng["max"],
        "ratio_p99":ratio["p99"],"ratio_p999":ratio["p999"],"ratio_max":ratio["max"],
        "archived_tiny_hard":old["tiny_edge_lt_1e_3_span"]["hard_weight_l1_gt_1_count"],
        "v9_tiny_hard":v9["tiny_edge_lt_1e_3_span"]["hard_weight_l1_gt_1_count"],
        "archived_weight_l1_gt_1":old["weight_l1_gt"]["1_0"],"v9_weight_l1_gt_1":v9["weight_l1_gt"]["1_0"],
    },sort_keys=True),flush=True)

if __name__=="__main__": main()
