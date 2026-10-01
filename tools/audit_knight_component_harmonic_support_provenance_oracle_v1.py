from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    deformation_envelope_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_holeless_partitioned_dense_candidate
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import DEFAULT_MAX_EDGE_RATIO
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR, build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json, replay_compacted_dense_face_provenance,
)
from tools.audit_knight_global_skin_region_sweep_v1 import skin_matrix
from tools.audit_knight_probe_conditioned_region_court_v1 import (
    probe_edge_risk, build_probe_partition,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    build_safe_graph, harmonic_complete, motion_metrics,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def q(a):
    x=np.asarray(a,dtype=np.float64)
    if not len(x): return {"count":0}
    return {
        "count":int(len(x)),"min":float(np.min(x)),
        "p50":float(np.quantile(x,.50)),"p90":float(np.quantile(x,.90)),
        "p95":float(np.quantile(x,.95)),"p99":float(np.quantile(x,.99)),
        "max":float(np.max(x)),
    }


def harmonic_coordinate_reconstruct(candidate, rest, Wowner, faces, unknown, edges):
    n=len(candidate.vertices)
    comp_by_vi=np.asarray([str(v.component_id) for v in candidate.vertices],dtype=object)
    anchor_sid={}
    for i,v in enumerate(candidate.vertices):
        if unknown[i]: continue
        coeff=tuple(v.support_binding.coefficients)
        if len(coeff)!=1 or abs(float(coeff[0][1])-1.0)>1e-12:
            raise RuntimeError("IDENTITY_ANCHOR_SUPPORT_INVALID")
        anchor_sid[i]=str(coeff[0][0])

    Wrec=Wowner.copy()
    support_coefficients={}
    coeff_counts=[]
    coeff_min=[]
    row_sum_error=[]
    negative_min=0.0
    dense_upper=0
    component_rows=[]

    for cid in sorted(set(comp_by_vi.tolist())):
        verts=np.nonzero(comp_by_vi==cid)[0].astype(int)
        U=np.asarray([i for i in verts if unknown[i]],dtype=np.int64)
        A=np.asarray([i for i in verts if not unknown[i]],dtype=np.int64)
        if not len(U):
            component_rows.append({"component_id":cid,"unknown":0,"anchors":len(A),"dense_upper":0})
            continue
        if not len(A):
            raise RuntimeError(f"HARMONIC_COORD_COMPONENT_NO_ANCHOR:{cid}")
        dense_upper += len(U)*len(A)

        uset={int(v) for v in U}; aset={int(v) for v in A}
        uix={int(v):i for i,v in enumerate(U)}
        aix={int(v):i for i,v in enumerate(A)}
        nbr={int(v):[] for v in verts}
        for (x,y),w in edges.items():
            if comp_by_vi[x]!=cid or comp_by_vi[y]!=cid:
                continue
            nbr[int(x)].append((int(y),float(w)))
            nbr[int(y)].append((int(x),float(w)))

        rows=[]; cols=[]; vals=[]
        B=np.zeros((len(U),len(A)),dtype=np.float64)
        for ri,u in enumerate(U):
            total=0.0
            for v,w in nbr[int(u)]:
                total += w
                if v in uset:
                    rows.append(ri); cols.append(uix[v]); vals.append(-w)
                elif v in aset:
                    B[ri,aix[v]] += w
                else:
                    raise RuntimeError("HARMONIC_COORD_NEIGHBOR_OUTSIDE_COMPONENT")
            if total<=0.0:
                raise RuntimeError("HARMONIC_COORD_ZERO_DEGREE")
            rows.append(ri); cols.append(ri); vals.append(total)

        L=csr_matrix((vals,(rows,cols)),shape=(len(U),len(U)))
        H=np.asarray(spsolve(L,B),dtype=np.float64)
        if H.ndim==1: H=H[:,None]
        negative_min=min(negative_min,float(np.min(H)))
        if np.min(H)<-1e-10:
            raise RuntimeError(f"HARMONIC_COORD_NEGATIVE::{cid}::{np.min(H)}")
        H=np.maximum(H,0.0)
        sums=H.sum(axis=1,keepdims=True)
        if np.any(sums<=1e-12):
            raise RuntimeError("HARMONIC_COORD_SIMPLEX_COLLAPSE")
        H=H/sums
        row_sum_error.extend(np.abs(H.sum(axis=1)-1.0).tolist())

        Wrec[U]=H@Wowner[A]
        for local,u in enumerate(U):
            raw=H[local]
            nz=np.nonzero(raw>1e-14)[0]
            coeff=tuple((anchor_sid[int(A[j])],float(raw[j])) for j in nz)
            support_coefficients[int(u)]=coeff
            coeff_counts.append(len(coeff))
            if len(nz): coeff_min.append(float(np.min(raw[nz])))
        component_rows.append({
            "component_id":cid,"unknown":int(len(U)),"anchors":int(len(A)),
            "dense_upper":int(len(U)*len(A)),
            "support_coeff_count":int(np.count_nonzero(H>1e-14)),
        })

    return Wrec,support_coefficients,{
        "component_count":len(component_rows),
        "dense_coefficient_upper_bound":int(dense_upper),
        "stored_coefficient_count_gt_1e14":int(sum(coeff_counts)),
        "support_count_per_generated_vertex":q(coeff_counts),
        "minimum_stored_coefficient":min(coeff_min) if coeff_min else None,
        "max_simplex_row_sum_error":max(row_sum_error,default=0.0),
        "minimum_raw_coordinate":float(negative_min),
        "components":component_rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    surface=rigging_surface_from_dict(load_json(
        rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    skeleton=qualified_skeleton_from_dict(load_json(
        rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(load_json(
        rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json"
    )).cameras,key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(load_json(
        rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"))
    skin=qualified_skin_from_dict(load_json(a.skin_json))

    explicit,prov=replay_compacted_dense_face_provenance(rr,surface)
    sids,jids,Wsrc=skin_matrix(surface,skeleton,skin)
    risk=probe_edge_risk(surface,skeleton,cameras,envelope,sids,Wsrc)
    part,unsafe_edges,crossing_edges,closure=build_probe_partition(
        surface,risk,max_edge_ratio=DEFAULT_MAX_EDGE_RATIO)
    carrier=build_component_carrier_policy(
        partition=part,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("COMPONENT_HARMONIC_PROVENANCE_ORACLE",),
            metadata={"automatic":False,"audit_only":True}) for c in part.components),
        metadata={"audit_only":True},
    )
    candidate=build_holeless_partitioned_dense_candidate(
        surface,part,carrier,
        producer_policy_hash=content_sha256({
            "audit":"COMPONENT_HARMONIC_PROVENANCE_ORACLE",
            "partition_components":len(part.components),
        }),
        explicit_face_provenance=explicit,
    )
    validate_canonical_mesh_candidate(
        candidate,surface=surface,partition=part,carrier_policy=carrier)

    rest,Wowner,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin)
    rest=np.asarray(rest,dtype=np.float64)
    Wowner=np.asarray(Wowner,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)
    unknown=np.asarray([
        str(v.support_binding.mode)=="SEAM_GEOMETRY_INTERPOLATION"
        for v in candidate.vertices
    ],dtype=bool)
    edges=build_safe_graph(rest,faces,np.ones(len(faces),dtype=bool))

    Wharm,solve=harmonic_complete(Wowner,unknown,edges)
    Wrec,support,coord=harmonic_coordinate_reconstruct(
        candidate,rest,Wowner,faces,unknown,edges)

    delta=np.abs(Wrec-Wharm)
    max_abs=float(np.max(delta))
    row_l1=np.sum(delta,axis=1)
    if max_abs>1e-10 or float(np.max(row_l1))>1e-9:
        raise RuntimeError(
            f"HARMONIC_PROVENANCE_RECONSTRUCTION_DRIFT::{max_abs}::{np.max(row_l1)}")

    # Reconstruct independently from QualifiedSkin source rows and solved support
    # coefficients. This proves the output can remain a convex transfer, not a new
    # learned/semantic skin authority.
    source={str(r.surface_id):dict((str(j),float(w)) for j,w in r.influences) for r in skin.rows}
    jix={str(j):i for i,j in enumerate(jids)}
    Wsource=np.zeros_like(Wrec)
    for vi,v in enumerate(candidate.vertices):
        if not unknown[vi]:
            coeff=tuple(v.support_binding.coefficients)
        else:
            coeff=support[vi]
        for sid,c in coeff:
            for jid,w in source[str(sid)].items():
                Wsource[vi,jix[jid]] += float(c)*float(w)
        mass=float(Wsource[vi].sum())
        if mass<=1e-12: raise RuntimeError("HARMONIC_PROVENANCE_ZERO_MASS")
        Wsource[vi]/=mass

    source_delta=np.abs(Wsource-Wharm)
    source_max=float(np.max(source_delta))
    source_row_l1=float(np.max(np.sum(source_delta,axis=1)))
    if source_max>1e-10 or source_row_l1>1e-9:
        raise RuntimeError(
            f"HARMONIC_SOURCE_RECONSTRUCTION_DRIFT::{source_max}::{source_row_l1}")

    source_report=load_json(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    motion=motion_metrics(
        rest,Wsource,faces,jids,skeleton,cameras,rr,source_report)

    report={
        "schema":"RealSaS.KnightComponentHarmonicSupportProvenanceOracle.v1",
        "status":"PASS__CONVEX_SOURCE_SUPPORT_RECONSTRUCTION",
        "partition":{
            "component_count":len(part.components),
            "unsafe_source_edge_seed_count":len(unsafe_edges),
            "cut_closed_crossing_edge_count":len(crossing_edges),
            "closure":closure,
        },
        "candidate":{
            "vertex_count":len(candidate.vertices),
            "face_count":len(candidate.faces),
            "generated_vertex_count":int(np.count_nonzero(unknown)),
            "identity_anchor_count":int(np.count_nonzero(~unknown)),
        },
        "harmonic_solve":solve,
        "coordinate_operator":coord,
        "reconstruction":{
            "coordinate_to_direct_harmonic_max_abs_delta":max_abs,
            "coordinate_to_direct_harmonic_max_row_l1":float(np.max(row_l1)),
            "qualified_skin_source_reconstruction_max_abs_delta":source_max,
            "qualified_skin_source_reconstruction_max_row_l1":source_row_l1,
        },
        "actual_motion_from_source_convex_support":motion,
        "face_provenance_replay":prov,
        "claim_boundary":[
            "Harmonic support coordinates depend only on candidate geometry/connectivity and mechanical component membership.",
            "Every generated vertex is reconstructed solely as a nonnegative convex combination of immutable QualifiedSkinIR source rows.",
            "No generated raw skin row is introduced as independent authority.",
            "No coefficient crosses a mechanical component because the solve graph is component-pure.",
            "This audit proves representability; production implementation may require a compact sparse-support storage policy."
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("HARMONIC_SUPPORT_PROVENANCE="+json.dumps({
        "status":report["status"],
        "generated":report["candidate"]["generated_vertex_count"],
        "stored_coefficients":coord["stored_coefficient_count_gt_1e14"],
        "support_count":coord["support_count_per_generated_vertex"],
        "max_coord_delta":max_abs,
        "max_source_delta":source_max,
        "motion_gt10":motion["max_edge_gt_10"],
        "motion_gt4":motion["max_edge_gt_4"],
        "motion_worst":motion["worst_edge_max"],
        "motion_p99":motion["max_edge_p99"],
    },sort_keys=True))


if __name__=="__main__":
    main()
