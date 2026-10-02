from __future__ import annotations
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path

from tools.audit_knight_repaired_quality_collapse_v1 import (
    loadj,build_repaired_surface,report_quality,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    repair_candidate_fixed_vertex_flips_topology_safe_v2,_manifold_report,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_collapse_v2 import (
    repair_candidate_endpoint_collapses_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _metric,_violates,_edge_incidence,mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import mesh_policy_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256

def face_components(faces, indices):
    selected=set(map(int,indices))
    vertex_to_faces=defaultdict(list)
    for fi in selected:
        for v in faces[fi]:
            vertex_to_faces[str(v)].append(fi)
    adj=defaultdict(set)
    for rows in vertex_to_faces.values():
        for a in rows:
            adj[a].update(x for x in rows if x!=a)
    seen=set(); comps=[]
    for root in sorted(selected):
        if root in seen: continue
        stack=[root];seen.add(root);cc=[]
        while stack:
            x=stack.pop();cc.append(x)
            for y in adj[x]:
                if y in selected and y not in seen:
                    seen.add(y);stack.append(y)
        comps.append(sorted(cc))
    return comps

def classify(candidate,policy,partition):
    positions={str(v.candidate_vertex_id):tuple(map(float,v.P)) for v in candidate.vertices}
    metrics=[_metric(face,positions) for face in candidate.faces]
    bad=[i for i,m in enumerate(metrics) if _violates(m,policy)]
    inc=_edge_incidence(candidate.faces)
    boundary_vertices={vid for edge,rows in inc.items() if len(rows)==1 for vid in edge}
    protected=mechanical_quality_protected_surface_ids_v1(partition)
    vmap={str(v.candidate_vertex_id):v for v in candidate.vertices}
    rows=[]
    for fi in bad:
        face=tuple(map(str,candidate.faces[fi]));m=metrics[fi]
        support_ids={
            str(sid)
            for vid in face
            for sid,_ in vmap[vid].support_binding.coefficients
        }
        rows.append({
            "face_index":fi,
            "min_angle_deg":float(m["min_angle_deg"]),
            "aspect":float(m["aspect_longest_over_min_altitude"]),
            "has_boundary_vertex":bool(any(v in boundary_vertices for v in face)),
            "has_protected_support":bool(support_ids.intersection(protected)),
        })
    comps=face_components(candidate.faces,bad)
    def count_angle(limit): return sum(r["min_angle_deg"]<limit for r in rows)
    def count_aspect(limit): return sum(r["aspect"]>limit for r in rows)
    return {
      "violating_face_count":len(rows),
      "angle_bins":{
        "lt_0_1":count_angle(0.1),"lt_0_5":count_angle(0.5),
        "lt_1":count_angle(1.0),"lt_2":count_angle(2.0),
        "lt_5":count_angle(5.0),"lt_7_5":count_angle(7.5),
      },
      "aspect_bins":{
        "gt_16":count_aspect(16.0),"gt_32":count_aspect(32.0),
        "gt_64":count_aspect(64.0),"gt_128":count_aspect(128.0),
        "gt_256":count_aspect(256.0),"gt_512":count_aspect(512.0),
      },
      "boundary_face_count":sum(r["has_boundary_vertex"] for r in rows),
      "protected_support_face_count":sum(r["has_protected_support"] for r in rows),
      "boundary_or_protected_face_count":sum(
          r["has_boundary_vertex"] or r["has_protected_support"] for r in rows
      ),
      "interior_unprotected_face_count":sum(
          (not r["has_boundary_vertex"]) and (not r["has_protected_support"]) for r in rows
      ),
      "residual_component_count":len(comps),
      "residual_component_sizes_top20":sorted((len(c) for c in comps),reverse=True)[:20],
      "worst_faces":sorted(rows,key=lambda r:(r["min_angle_deg"],-r["aspect"]))[:40],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--max-cycles",type=int,default=4)
    a=ap.parse_args(); rr=a.run_root.resolve(); out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)

    t0=time.perf_counter()
    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=partition,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_ALTERNATING_V3",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    current=build_canonical_relation_candidate(
      surface,partition,carrier,
      producer_policy_hash=content_sha256({
        "court":"REPAIRED_STATIC_QUALITY_ALTERNATING_V3",
        "policy":policy.qualification_policy_lineage_hash
      }),
      explicit_face_provenance=explicit,
    )
    initial=report_quality(current,policy)
    topo=_manifold_report(current.faces)
    if not topo["passed"]:
        raise RuntimeError(f"ALTERNATING_V3_INPUT_NOT_MANIFOLD:{topo}")

    cycles=[]
    previous=int(initial["policy_violating_face_count"])
    for ci in range(int(a.max_cycles)):
        c0=time.perf_counter()
        flipped,fr=repair_candidate_fixed_vertex_flips_topology_safe_v2(
            current,policy,max_passes=12
        )
        qf=report_quality(flipped,policy)
        c1=time.perf_counter()
        collapsed,cr=repair_candidate_endpoint_collapses_batched_v2(
            flipped,policy,max_batches=64,max_collapses=2048
        )
        qc=report_quality(collapsed,policy)
        c2=time.perf_counter()
        topo2=_manifold_report(collapsed.faces)
        row={
          "cycle":ci,
          "before_violations":previous,
          "flip_count":int(fr["accepted_flip_count"]),
          "after_flip_violations":int(qf["policy_violating_face_count"]),
          "flip_rejected_shape":int(fr["rejected_shape_deviation_count"]),
          "flip_rejected_topology":int(fr["rejected_topology_count"]),
          "flip_rejected_batch_conflict":int(fr["rejected_batch_conflict_count"]),
          "collapse_count":int(cr["accepted_collapse_count"]),
          "collapse_batches":int(cr["batch_count"]),
          "collapse_rejected_boundary":int(cr["rejected_boundary_vertex_count"]),
          "collapse_rejected_link":int(cr["rejected_link_condition_count"]),
          "collapse_rejected_shape":int(cr["rejected_shape_deviation_count"]),
          "collapse_rejected_quality":int(cr["rejected_quality_count"]),
          "collapse_rejected_duplicate":int(cr["rejected_duplicate_face_count"]),
          "after_collapse_violations":int(qc["policy_violating_face_count"]),
          "min_angle_deg":float(qc["min_angle_deg"]),
          "max_aspect":float(qc["max_aspect_longest_over_min_altitude"]),
          "vertices":len(collapsed.vertices),"faces":len(collapsed.faces),
          "topology":topo2,
          "flip_seconds":c1-c0,"collapse_seconds":c2-c1,
        }
        cycles.append(row)
        print("STATIC_V3_CYCLE="+json.dumps(row,sort_keys=True),flush=True)
        current=collapsed
        now=int(qc["policy_violating_face_count"])
        if now==0 or now>=previous:
            break
        previous=now

    final_quality=report_quality(current,policy)
    final_topology=_manifold_report(current.faces)
    classification=classify(current,policy,partition)
    report={
      "schema":"RealSaS.KnightStaticQualityAlternatingCourt.v3",
      "status":"PASS" if (
        final_topology["passed"] and final_quality["policy_violating_face_count"]==0
      ) else "FAIL",
      "input_nodes":len(surface.surface_nodes),
      "initial_quality":initial,
      "cycles":cycles,
      "final_quality":final_quality,
      "final_topology":final_topology,
      "classification":classification,
      "final_candidate_lineage_hash":current.candidate_lineage_hash,
      "final_vertex_count":len(current.vertices),
      "final_face_count":len(current.faces),
      "total_wall_seconds":time.perf_counter()-t0,
      "success":bool(final_topology["passed"] and final_quality["policy_violating_face_count"]==0),
    }
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"final_candidate.json").write_text(json.dumps(current.to_dict(),indent=2,sort_keys=True)+"\n")
    (out/"partition.json").write_text(json.dumps(partition.to_dict(),indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STATIC_QUALITY_ALTERNATING_V3="+json.dumps({
      "cycles":[
        [r["cycle"],r["before_violations"],r["after_flip_violations"],r["after_collapse_violations"],
         r["flip_count"],r["collapse_count"]]
        for r in cycles
      ],
      "final_violations":final_quality["policy_violating_face_count"],
      "final_min_angle":final_quality["min_angle_deg"],
      "final_max_aspect":final_quality["max_aspect_longest_over_min_altitude"],
      "topology_pass":final_topology["passed"],
      "boundary_faces":classification["boundary_face_count"],
      "protected_faces":classification["protected_support_face_count"],
      "interior_unprotected":classification["interior_unprotected_face_count"],
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
