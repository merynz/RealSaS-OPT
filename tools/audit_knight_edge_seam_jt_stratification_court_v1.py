from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from tools.audit_knight_edge_seam_two_core_court_v1 import _two_core
from tools.audit_knight_james_twigg_region_inference_v1 import (
    EPSILON,
    PROBE_DEGREES,
    _cluster_signatures,
    _rotation_signatures,
    _vertex_labels_from_faces,
)
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import (
    _edge_stretch,
    _edge_table,
    _truth_region,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    stress_arbitrary_weights,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


PREREG = Path("canonical/KNIGHT_EDGE_SEAM_JT_STRATIFICATION_PREREG_V1_20260929.json")


def _edge_eval(mask, truth_seam):
    tp=int(np.count_nonzero(mask & truth_seam))
    fp=int(np.count_nonzero(mask & (~truth_seam)))
    return {
        "edge_count":int(np.count_nonzero(mask)),
        "true_positive_edge_count":tp,
        "false_positive_edge_count":fp,
        "precision":float(tp/max(1,tp+fp)),
    }


def _face_eval(mask, face_edge_index, truth, unsafe, F):
    pred=np.any(mask[face_edge_index],axis=1)
    truth_mixed=np.asarray(
        [len(set(int(truth[v]) for v in row))>1 for row in F],dtype=bool
    )
    tp=int(np.count_nonzero(pred&truth_mixed))
    fp=int(np.count_nonzero(pred&(~truth_mixed)))
    unsafe_truth=unsafe&truth_mixed
    hit=int(np.count_nonzero(pred&unsafe_truth))
    return {
        "predicted_mixed_face_count":int(np.count_nonzero(pred)),
        "true_positive_mixed_face_count":tp,
        "false_positive_mixed_face_count":fp,
        "mixed_face_precision":float(tp/max(1,tp+fp)),
        "unsafe_truth_cross_region_recovered":hit,
        "unsafe_truth_cross_region_recall":float(hit/max(1,np.count_nonzero(unsafe_truth))),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--weights-npz",type=Path,required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()

    prereg=json.loads(PREREG.read_text())
    if prereg.get("status")!="FROZEN_BEFORE_JT_EDGE_STRATIFICATION_RESULT":
        raise RuntimeError("JT_EDGE_STRATIFICATION_PREREG_DRIFT")

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)

    P=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    F=face_indices(cand)
    dense=dense_supported_face_mask(rr,cand,surface)
    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        W=np.asarray(z["arachne"],dtype=np.float64)

    stress=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),dtype=bool)
    unsafe[np.asarray(stress["unsafe_face_indices"],dtype=np.int64)]=True
    safe=(~unsafe)&dense

    Z,valid,_=_rotation_signatures(P,F,W,jids,sk,cams,PROBE_DEGREES)
    face_label,core,_=_cluster_signatures(Z,safe&valid,EPSILON)
    vlabel,_,prop_meta=_vertex_labels_from_faces(P,F,face_label,core,dense,safe)

    edges,edge_faces,face_edge_index=_edge_table(P,F)
    stretch=_edge_stretch(P,W,edges,jids,sk,cams,env)
    l1=np.sum(np.abs(W[edges[:,0]]-W[edges[:,1]]),axis=1)

    in_scope=np.zeros(len(edges),dtype=bool)
    boundary_vertex=np.zeros(len(P),dtype=bool)
    for ei,edge in enumerate(map(tuple,edges.tolist())):
        fs=edge_faces[edge]
        in_scope[ei]=any(bool(unsafe[fi] and dense[fi]) for fi in fs)
        if len(fs)==1:
            boundary_vertex[int(edge[0])]=True
            boundary_vertex[int(edge[1])]=True

    raw=in_scope&(stretch>4.0)&(l1>1.0)
    seam=_two_core(edges,raw,boundary_vertex)

    u=edges[:,0];v=edges[:,1]
    ku=vlabel[u]>=0;kv=vlabel[v]>=0
    strata={
        "JT_DIFFERENT":seam&ku&kv&(vlabel[u]!=vlabel[v]),
        "JT_SAME":seam&ku&kv&(vlabel[u]==vlabel[v]),
        "JT_ONE_UNKNOWN":seam&(ku^kv),
        "JT_BOTH_UNKNOWN":seam&(~ku)&(~kv),
    }

    # Teacher begins only here.
    truth=_truth_region(a.teacher_bank,a.teacher_source,sk,cand)
    truth_seam=truth[u]!=truth[v]

    result={}
    for name,mask in strata.items():
        result[name]={
            "edge":_edge_eval(mask,truth_seam),
            "face":_face_eval(mask,face_edge_index,truth,unsafe,F),
            "stretch_p50":float(np.quantile(stretch[mask],.5)) if np.any(mask) else None,
            "stretch_p95":float(np.quantile(stretch[mask],.95)) if np.any(mask) else None,
            "weight_l1_p50":float(np.quantile(l1[mask],.5)) if np.any(mask) else None,
            "weight_l1_p95":float(np.quantile(l1[mask],.95)) if np.any(mask) else None,
        }

    report={
        "schema":"RealSaS.KnightEdgeSeamJamesTwiggStratificationCourt.v1",
        "status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "preregistration":str(PREREG),
        "teacher_used_by_stratification":False,
        "james_twigg_propagation":prop_meta,
        "base_two_core_edge_count":int(np.count_nonzero(seam)),
        "strata_teacher_eval_only":result,
        "partition_edge_count":int(sum(np.count_nonzero(x) for x in strata.values())),
        "claim_boundary":"Base mechanical seam and James/Twigg endpoint strata are fixed before teacher source topology is loaded."
    }
    if report["partition_edge_count"]!=report["base_two_core_edge_count"]:
        raise RuntimeError("JT_EDGE_STRATA_NOT_A_PARTITION")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_EDGE_SEAM_JT_STRATIFICATION_PASS",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
