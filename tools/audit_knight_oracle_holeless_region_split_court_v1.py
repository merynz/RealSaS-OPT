from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    motion_metrics,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import (
    exact, load, faces_index, teacher_weights,
)
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import (
    source_components,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def tri_double_area(P, face):
    a,b,c=P[np.asarray(face,dtype=np.int64)]
    return float(np.linalg.norm(np.cross(b-a,c-a)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    faces=faces_index(cand)
    P0=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    W0,jids,tri,sf=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    teacher_simplex_before={
        "min_weight":float(np.min(W0)),
        "max_abs_sum_minus_one":float(np.max(np.abs(W0.sum(axis=1)-1.0))),
        "zero_or_negative_sum_rows":int(np.count_nonzero(W0.sum(axis=1)<=1e-12)),
    }
    if (
        teacher_simplex_before["min_weight"] < -1e-6
        or teacher_simplex_before["max_abs_sum_minus_one"] > 1e-4
        or teacher_simplex_before["zero_or_negative_sum_rows"] > 0
    ):
        raise RuntimeError("TEACHER_WEIGHT_SIMPLEX_NOT_NUMERICAL_RESIDUE:"+json.dumps(teacher_simplex_before,sort_keys=True))
    W0_raw=W0.copy()
    W0=np.maximum(W0,0.0)
    W0/=W0.sum(axis=1,keepdims=True)
    teacher_simplex_projection_l1=np.sum(np.abs(W0-W0_raw),axis=1)
    src_comp=source_components(int(sf.max())+1,sf)
    tri_comp=np.asarray([src_comp[int(row[0])] for row in sf],dtype=np.int64)
    label=np.asarray([tri_comp[int(t)] for t in tri],dtype=np.int64)

    P=[x.copy() for x in P0]
    W=[x.copy() for x in W0]
    region=[int(x) for x in label]
    seam_vertex_cache={}

    def add_seam_edge(a0,b0,r):
        key=(min(int(a0),int(b0)),max(int(a0),int(b0)),int(r))
        if key in seam_vertex_cache:
            return seam_vertex_cache[key]
        pa,pb=P0[int(a0)],P0[int(b0)]
        pos=0.5*(pa+pb)
        candidates=[v for v in (int(a0),int(b0)) if int(label[v])==int(r)]
        if not candidates:
            raise RuntimeError("SEAM_EDGE_REGION_HAS_NO_ENDPOINT")
        ww=np.mean(W0[np.asarray(candidates,dtype=np.int64)],axis=0)
        ww=np.maximum(ww,0.0);ww/=max(float(ww.sum()),1e-15)
        idx=len(P);P.append(pos);W.append(ww);region.append(int(r));seam_vertex_cache[key]=idx
        return idx

    centroid_cache={}
    def add_centroid(fi,r,face):
        key=(int(fi),int(r))
        if key in centroid_cache:return centroid_cache[key]
        pos=P0[np.asarray(face,dtype=np.int64)].mean(axis=0)
        candidates=[int(v) for v in face if int(label[int(v)])==int(r)]
        if not candidates:raise RuntimeError("CENTROID_REGION_EMPTY")
        ww=np.mean(W0[np.asarray(candidates,dtype=np.int64)],axis=0)
        ww=np.maximum(ww,0.0);ww/=max(float(ww.sum()),1e-15)
        idx=len(P);P.append(pos);W.append(ww);region.append(int(r));centroid_cache[key]=idx
        return idx

    new_faces=[]
    parent_face=[]
    mixed_count=0
    two_label_count=0
    three_label_count=0
    area_err=[]
    for fi,face0 in enumerate(faces.tolist()):
        face=list(map(int,face0))
        labs=[int(label[v]) for v in face]
        unique=sorted(set(labs))
        before=tri_double_area(P0,face)
        produced=[]
        if len(unique)==1:
            produced=[tuple(face)]
        elif len(unique)==2:
            mixed_count+=1;two_label_count+=1
            for r in unique:
                own=[i for i,v in enumerate(face) if labs[i]==r]
                other=[i for i,v in enumerate(face) if labs[i]!=r]
                if len(own)==1:
                    i=own[0];j,k=other
                    vi=face[i]
                    mij=add_seam_edge(vi,face[j],r)
                    mik=add_seam_edge(vi,face[k],r)
                    produced.append((vi,mij,mik))
                elif len(own)==2:
                    i,j=own;k=other[0]
                    vi,vj,vk=face[i],face[j],face[k]
                    mik=add_seam_edge(vi,vk,r)
                    mjk=add_seam_edge(vj,vk,r)
                    produced.extend(((vi,vj,mjk),(vi,mjk,mik)))
                else:
                    raise RuntimeError("TWO_LABEL_ARITY_INVALID")
        elif len(unique)==3:
            mixed_count+=1;three_label_count+=1
            for i in range(3):
                r=labs[i];vi=face[i];vj=face[(i+1)%3];vk=face[(i-1)%3]
                mij=add_seam_edge(vi,vj,r)
                mki=add_seam_edge(vk,vi,r)
                cen=add_centroid(fi,r,face)
                produced.extend(((vi,mij,cen),(vi,cen,mki)))
        else:
            raise RuntimeError("FACE_LABEL_CARDINALITY_INVALID")

        P_arr=np.asarray(P,dtype=np.float64)
        after=sum(tri_double_area(P_arr,f) for f in produced)
        area_err.append(abs(after-before)/max(before,1e-15))
        new_faces.extend(produced)
        parent_face.extend([fi]*len(produced))

    P=np.asarray(P,dtype=np.float64)
    W=np.asarray(W,dtype=np.float64)
    F=np.asarray(new_faces,dtype=np.int64)
    if not np.isfinite(P).all() or not np.isfinite(W).all():
        raise RuntimeError("SPLIT_NONFINITE")
    split_simplex_before={
        "min_weight":float(np.min(W)),
        "max_abs_sum_minus_one":float(np.max(np.abs(W.sum(axis=1)-1.0))),
        "zero_or_negative_sum_rows":int(np.count_nonzero(W.sum(axis=1)<=1e-12)),
    }
    if (
        split_simplex_before["min_weight"] < -1e-6
        or split_simplex_before["max_abs_sum_minus_one"] > 1e-4
        or split_simplex_before["zero_or_negative_sum_rows"] > 0
    ):
        raise RuntimeError("SPLIT_WEIGHT_SIMPLEX_NOT_NUMERICAL_RESIDUE:"+json.dumps(split_simplex_before,sort_keys=True))
    W_raw=W.copy()
    W=np.maximum(W,0.0)
    W/=W.sum(axis=1,keepdims=True)
    split_simplex_projection_l1=np.sum(np.abs(W-W_raw),axis=1)

    # Verify each produced face is region-pure.
    R=np.asarray(region,dtype=np.int64)
    pure=(R[F[:,0]]==R[F[:,1]])&(R[F[:,1]]==R[F[:,2]])
    if not np.all(pure):
        raise RuntimeError("SPLIT_FACE_NOT_REGION_PURE")

    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    base_stress=stress_arbitrary_weights(P0,W0,faces,jids,sk,cams,env,policy)
    split_stress=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy)
    base_motion=motion_metrics(P0,W0,faces,jids,sk,cams,rr,source_report)
    split_motion=motion_metrics(P,W,F,jids,sk,cams,rr,source_report)

    report={
      "schema":"RealSaS.KnightOracleHolelessRegionSplitCourt.v1",
      "status":"TEACHER_TOPOLOGY_ORACLE_ONLY__NO_PRODUCT_REPAIR",
      "operator":{
        "label_authority":"TEACHER_SOURCE_CONNECTED_COMPONENT__ORACLE_ONLY",
        "mixed_face_rule":"BARYCENTRIC_MIDEDGE_REGION_SUBDIVISION",
        "seam_rule":"DUPLICATE_SEAM_VERTICES_PER_REGION_AT_IDENTICAL_REST_POSITION",
        "face_deletion":False,
        "rest_geometry_interpolation":"MIDPOINT_AND_FACE_CENTROID_ONLY",
        "new_weight_rule":"REGION_ENDPOINT_OR_REGION_FACE_VERTEX_AVERAGE__ORACLE_TEST",
      },
      "teacher_simplex_numerics":{
        "before":teacher_simplex_before,
        "projection_l1_max":float(np.max(teacher_simplex_projection_l1)),
        "projection_l1_p99":float(np.quantile(teacher_simplex_projection_l1,.99)),
        "split_before":split_simplex_before,
        "split_projection_l1_max":float(np.max(split_simplex_projection_l1)),
        "split_projection_l1_p99":float(np.quantile(split_simplex_projection_l1,.99)),
      },
      "topology":{
        "source_component_count":int(len(set(src_comp.tolist()))),
        "vertex_count_before":int(len(P0)),
        "vertex_count_after":int(len(P)),
        "new_vertex_count":int(len(P)-len(P0)),
        "face_count_before":int(len(faces)),
        "face_count_after":int(len(F)),
        "mixed_face_count":mixed_count,
        "two_label_face_count":two_label_count,
        "three_label_face_count":three_label_count,
        "seam_edge_copy_count":len(seam_vertex_cache),
        "centroid_copy_count":len(centroid_cache),
        "region_pure_output_face_count":int(np.count_nonzero(pure)),
        "max_parent_face_rest_area_relative_error":float(max(area_err,default=0.0)),
        "p99_parent_face_rest_area_relative_error":float(np.quantile(area_err,.99)),
        "face_deletion_count":0,
      },
      "synthetic_stress":{"before":base_stress,"after":split_stress},
      "actual_motion":{
        "before":{k:v for k,v in base_motion.items() if k!="frames"},
        "after":{k:v for k,v in split_motion.items() if k!="frames"},
      },
      "finding":{
        "rest_area_exactly_preserved":bool(max(area_err,default=0.0)<1e-10),
        "no_faces_deleted":True,
        "all_output_faces_region_pure":bool(np.all(pure)),
        "split_reduces_actual_gt10":bool(split_motion["max_edge_gt_10"]<base_motion["max_edge_gt_10"]),
        "split_eliminates_actual_gt10":bool(split_motion["max_edge_gt_10"]==0),
        "split_reduces_synthetic_unsafe":bool(split_stress["unsafe_face_count"]<base_stress["unsafe_face_count"]),
      },
      "claim_boundary":"Teacher source connected-component labels are used only to test whether a holeless topological split with vertex duplication can replace face deletion. Product inference must derive region/seam evidence without teacher topology."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_ORACLE_HOLELESS_REGION_SPLIT_COURT_PASS",json.dumps({
      "topology":report["topology"],
      "before_stress":{k:v for k,v in base_stress.items() if k!="unsafe_face_indices"},
      "after_stress":{k:v for k,v in split_stress.items() if k!="unsafe_face_indices"},
      "before_motion":report["actual_motion"]["before"],
      "after_motion":report["actual_motion"]["after"],
      "finding":report["finding"],
    },sort_keys=True))

if __name__=="__main__":main()
