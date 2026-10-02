from __future__ import annotations
import argparse,json,math,time
from dataclasses import replace
from pathlib import Path
import numpy as np

from tools.audit_knight_repaired_quality_collapse_v1 import loadj,build_repaired_surface,report_quality
from tools.audit_knight_static_quality_alternating_v3 import classify
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mechanical_partition_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshVertexCandidateIR,ComponentCarrierDecisionIR,
    build_component_carrier_policy,canonical_mesh_candidate_lineage_hash,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_metric,_violates,_combine_support_bindings,
    mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    repair_candidate_fixed_vertex_flips_topology_safe_v2,_manifold_report,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_collapse_v2 import (
    repair_candidate_endpoint_collapses_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_relaxation_v2 import (
    repair_candidate_projected_relaxation_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_cavity_remesh_v1 import (
    repair_candidate_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_edge_cavity_remesh_v1 import (
    repair_candidate_edge_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_patch_purge_v1 import (
    repair_candidate_patch_interior_purge_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import SurfaceSupportBinding

def synchronized_long_edge_split(candidate,policy,max_splits=64):
    vertices={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:np.asarray(v.P,dtype=np.float64) for vid,v in vertices.items()}
    faces=[tuple(map(str,f)) for f in candidate.faces]
    inc=_edge_incidence(faces)
    metrics=[_metric(f,{k:tuple(map(float,p)) for k,p in positions.items()}) for f in faces]
    bad=[i for i,m in enumerate(metrics) if _violates(m,policy)]
    ratio_floor=2.0*math.cos(math.radians(float(policy.g3_min_angle_deg)))

    proposals=[]
    for fi in bad:
        f=faces[fi]
        rows=[]
        for u,v in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            e=_edge(u,v)
            L=float(np.linalg.norm(positions[u]-positions[v]))
            rows.append((L,e))
        rows.sort()
        minL=rows[0][0];maxL,e=rows[-1]
        if minL<=1e-12 or maxL/minL<=ratio_floor+1e-9:
            continue
        if len(inc.get(e,()))!=2:
            continue
        u,v=e
        if vertices[u].component_id!=vertices[v].component_id:
            continue
        proposals.append((-(maxL/minL),-maxL,e,fi,maxL/minL))

    proposals.sort()
    selected=[]
    occupied=set()
    for _,_,e,fi,ratio in proposals:
        u,v=e
        # closed endpoint conflict guard: no two selected split edges share an
        # endpoint. Synchronized face rewrite then has at most one split/face.
        if u in occupied or v in occupied:
            continue
        selected.append((e,fi,ratio))
        occupied.update((u,v))
        if len(selected)>=int(max_splits):
            break
    if not selected:
        return candidate,{"accepted_split_count":0,"selected":[]}

    split_map={}
    new_vertices=dict(vertices)
    for e,fi,ratio in selected:
        u,v=e
        support=_combine_support_bindings([vertices[u],vertices[v]],(0.5,0.5))
        P=tuple(map(float,0.5*(positions[u]+positions[v])))
        nid="SPLITV:"+content_sha256({
            "input_candidate":candidate.candidate_lineage_hash,
            "edge":e,"fraction":0.5,
        })[:24]
        split_map[e]=nid
        new_vertices[nid]=CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=nid,
            support_binding=support,
            component_id=vertices[u].component_id,
            P=P,
            metadata={
                "operator":"SYNCHRONIZED_INTERIOR_EDGE_MIDPOINT_SPLIT_AUDIT_V1",
                "parent_edge":list(e),"fraction":0.5,
                "product_authority_minted":False,
            },
        )

    new_faces=[]
    for face in faces:
        a,b,c=face
        hit=None
        for u,v,w in ((a,b,c),(b,c,a),(c,a,b)):
            e=_edge(u,v)
            if e in split_map:
                hit=(u,v,w,split_map[e]);break
        if hit is None:
            new_faces.append(face);continue
        u,v,w,m=hit
        new_faces.append((u,m,w))
        new_faces.append((m,v,w))

    if len(set(new_faces))!=len(new_faces):
        raise RuntimeError("SPLIT_CREATED_DUPLICATE_FACE")
    topo=_manifold_report(new_faces)
    if not topo["passed"]:
        raise RuntimeError(f"SPLIT_NONMANIFOLD:{topo}")

    used={x for f in new_faces for x in f}
    vtuple=tuple(new_vertices[x] for x in sorted(used))
    edges=tuple(sorted({
        _edge(f[i],f[j]) for f in new_faces for i,j in ((0,1),(1,2),(2,0))
    }))
    policy_hash=content_sha256({
        "schema":"RealSaS.SynchronizedInteriorEdgeSplitAuditPolicy.v1",
        "input_candidate":candidate.candidate_lineage_hash,
        "g3_min_angle":float(policy.g3_min_angle_deg),
        "ratio_trigger":"LONGEST_OVER_SHORTEST_GT_2COS_MIN_ANGLE",
        "placement":"MIDPOINT_CONVEX_SUPPORT",
        "product_authority_minted":False,
    })
    provisional=replace(
        candidate,
        vertices=vtuple,faces=tuple(sorted(new_faces)),edges=edges,
        producer_id="RealSaS.CanonicalMesh.SynchronizedInteriorEdgeSplit.audit.v1",
        producer_policy_hash=policy_hash,candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "synchronized_edge_split_audit":{
                "accepted_split_count":len(selected),
                "selected":[
                    {"edge":list(e),"seed_face":int(fi),"edge_ratio":float(r)}
                    for e,fi,r in selected
                ],
                "product_authority_minted":False,
                "stage14_stitch_proof_required_for_product_promotion":True,
            },
        },
    )
    result=replace(provisional,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional))
    return result,dict(result.metadata["synchronized_edge_split_audit"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--static-v7-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--max-cycles",type=int,default=4)
    a=ap.parse_args();rr=a.run_root.resolve();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)

    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    current=canonical_mesh_candidate_from_dict(loadj(a.static_v7_dir/"final_candidate.json"))
    partition=mechanical_partition_from_dict(loadj(a.static_v7_dir/"partition.json"))
    protected=mechanical_quality_protected_surface_ids_v1(partition)

    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    refpart=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=refpart,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_ALTERNATING_V3",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in refpart.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    reference=build_canonical_relation_candidate(
      surface,refpart,carrier,
      producer_policy_hash=content_sha256({
        "court":"REPAIRED_STATIC_QUALITY_ALTERNATING_V3",
        "policy":policy.qualification_policy_lineage_hash,
      }),
      explicit_face_provenance=explicit,
    )

    initial=report_quality(current,policy)
    previous=int(initial["policy_violating_face_count"])
    cycles=[];t0=time.perf_counter()

    for ci in range(int(a.max_cycles)):
        row={"cycle":ci,"before_violations":previous}

        split,sr=synchronized_long_edge_split(current,policy,max_splits=64)
        qs=report_quality(split,policy)
        row.update({
            "split_count":int(sr["accepted_split_count"]),
            "after_split_violations":int(qs["policy_violating_face_count"]),
        })

        fl,fr=repair_candidate_fixed_vertex_flips_topology_safe_v2(split,policy,max_passes=12)
        qf=report_quality(fl,policy)
        row.update({"flip_count":int(fr["accepted_flip_count"]),"after_flip":int(qf["policy_violating_face_count"])})

        co,cr=repair_candidate_endpoint_collapses_batched_v2(fl,policy,max_batches=64,max_collapses=2048)
        qc=report_quality(co,policy)
        row.update({"collapse_count":int(cr["accepted_collapse_count"]),"after_collapse":int(qc["policy_violating_face_count"])})

        rel,rrpt=repair_candidate_projected_relaxation_batched_v2(
            co,reference,policy,protected_surface_ids=protected,max_batches=64,max_moves=2048
        )
        qr=report_quality(rel,policy)
        row.update({"relax_count":int(rrpt["accepted_move_count"]),"after_relax":int(qr["policy_violating_face_count"])})

        vc,vr=repair_candidate_cavity_retriangulation_v1(
            rel,policy,protected_surface_ids=protected,max_batches=64,max_removed_vertices=2048
        )
        qv=report_quality(vc,policy)
        row.update({"vertex_cavity_count":int(vr["accepted_cavity_count"]),"after_vertex_cavity":int(qv["policy_violating_face_count"])})

        ec,er=repair_candidate_edge_cavity_retriangulation_v1(
            vc,policy,protected_surface_ids=protected,max_batches=64,max_removed_edges=2048
        )
        qe=report_quality(ec,policy)
        row.update({"edge_cavity_count":int(er["accepted_edge_cavity_count"]),"after_edge_cavity":int(qe["policy_violating_face_count"])})

        pp,pr=repair_candidate_patch_interior_purge_v1(
            ec,policy,protected_surface_ids=protected,max_hops=5,max_patch_faces=128,max_batches=64,max_patches=2048
        )
        qp=report_quality(pp,policy)
        topo=_manifold_report(pp.faces)
        row.update({
            "patch_purge_count":int(pr["accepted_patch_count"]),
            "after_patch_purge":int(qp["policy_violating_face_count"]),
            "min_angle_deg":float(qp["min_angle_deg"]),
            "max_aspect":float(qp["max_aspect_longest_over_min_altitude"]),
            "vertices":len(pp.vertices),"faces":len(pp.faces),"topology":topo,
        })
        cycles.append(row)
        print("STATIC_SPLIT_V8_CYCLE="+json.dumps(row,sort_keys=True),flush=True)

        now=int(qp["policy_violating_face_count"])
        # Split is an enabling move and may transiently increase violations, but
        # the transaction is admitted only if the full repair cycle improves.
        if now>=previous and int(sr["accepted_split_count"])>0:
            print("STATIC_SPLIT_V8_TRANSACTION_REJECT="+json.dumps({
                "cycle":ci,"before":previous,"after":now
            },sort_keys=True),flush=True)
            break
        current=pp
        if now==0 or now>=previous or int(sr["accepted_split_count"])==0:
            break
        previous=now

    final_quality=report_quality(current,policy)
    final_topology=_manifold_report(current.faces)
    final_class=classify(current,policy,partition)
    report={
      "schema":"RealSaS.KnightStaticQualitySplitCycleCourt.v8",
      "status":"PASS" if final_topology["passed"] and int(final_quality["policy_violating_face_count"])==0 else "FAIL",
      "initial_quality":initial,"cycles":cycles,
      "final_quality":final_quality,"final_topology":final_topology,
      "classification":final_class,
      "final_candidate_lineage_hash":current.candidate_lineage_hash,
      "final_vertex_count":len(current.vertices),"final_face_count":len(current.faces),
      "total_seconds":time.perf_counter()-t0,
      "audit_only_synchronized_split_used":any(r["split_count"]>0 for r in cycles),
      "product_stage14_stitch_proof_required":True,
      "success":bool(final_topology["passed"] and int(final_quality["policy_violating_face_count"])==0),
    }
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"final_candidate.json").write_text(json.dumps(current.to_dict(),indent=2,sort_keys=True)+"\n")
    (out/"partition.json").write_text(json.dumps(partition.to_dict(),indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STATIC_QUALITY_SPLIT_V8="+json.dumps({
      "cycles":[
        [r["cycle"],r["before_violations"],r["after_split_violations"],r["after_flip"],
         r["after_collapse"],r["after_relax"],r["after_vertex_cavity"],
         r["after_edge_cavity"],r["after_patch_purge"],r["split_count"]]
        for r in cycles
      ],
      "final_violations":final_quality["policy_violating_face_count"],
      "final_min_angle":final_quality["min_angle_deg"],
      "final_max_aspect":final_quality["max_aspect_longest_over_min_altitude"],
      "topology_pass":final_topology["passed"],"success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
