from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path

from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import _metric,_violates
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from compiler.realsas_compiler_core.artifact_codec_v2 import mesh_policy_from_dict

def loadj(p:Path): return json.loads(p.read_text())

def surface_quality(surface_payload,prov_payload,policy):
    nodes={str(n["surface_id"]):tuple(map(float,n["P"])) for n in surface_payload["surface_nodes"]}
    faces=[tuple(map(str,f)) for f in prov_payload["compact_faces"]]
    missing=sorted({v for f in faces for v in f if v not in nodes})
    if missing: raise RuntimeError(f"SURFACE_FACE_ID_MISSING:{len(missing)}")
    metrics=[_metric(f,nodes) for f in faces]
    bad=[m for m in metrics if _violates(m,policy)]
    min_angle=min((float(m["min_angle_deg"]) for m in metrics),default=float("inf"))
    max_aspect=max((float(m["aspect_longest_over_min_altitude"]) for m in metrics),default=0.0)
    return {
      "node_count":len(nodes),"face_count":len(faces),
      "quality_violating_face_count":len(bad),
      "below_min_angle_face_count":sum(float(m["min_angle_deg"])<float(policy.g3_min_angle_deg) for m in metrics),
      "above_max_aspect_face_count":sum(float(m["aspect_longest_over_min_altitude"])>float(policy.g3_max_aspect_longest_over_min_altitude) for m in metrics),
      "min_angle_deg":min_angle,"max_aspect":max_aspect,
      "topology":_manifold_report(faces),
      "triangle_authority":prov_payload.get("triangle_authority"),
      "three_clique_face_minting_allowed":prov_payload.get("three_clique_face_minting_allowed"),
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    rr=a.run_root.resolve()
    s13=loadj(rr/"artifacts/13_GEOMETRY_SUBSTRATE_QUALIFIED/geometry_substrate_qualification.json")
    s14_report=loadj(rr/"artifacts/14_GSA_BUILD/substrate_adequacy_report.json")
    s14_surface=loadj(rr/"artifacts/14_GSA_BUILD/rigging_surface_candidate.json")
    s14_prov=loadj(rr/"artifacts/14_GSA_BUILD/compacted_dense_face_provenance.json")
    s15_surface=loadj(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json")
    s15_qual=loadj(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/rigging_surface_qualification.json")
    s15_prov=loadj(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/compacted_dense_face_provenance.json")
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))

    stage13_report=dict(s13.get("qualification_report") or {})
    views=list(s13.get("views") or [])
    s14q=surface_quality(s14_surface,s14_prov,policy)
    s15q=surface_quality(s15_surface,s15_prov,policy)
    same_lineage=str(s14_surface.get("geometry_lineage_hash"))==str(s15_surface.get("geometry_lineage_hash"))
    same_faces=tuple(map(tuple,s14_prov.get("compact_faces") or ()))==tuple(map(tuple,s15_prov.get("compact_faces") or ()))

    evaluated=list(s14_report.get("evaluated_candidates") or ())
    passing=sum(bool(x.get("passed")) for x in evaluated)
    closest=dict(s14_report.get("diagnostic_closest_nonpassing_candidate") or {})
    report={
      "schema":"RealSaS.KnightUpstreamSurfaceQualityLineageAudit.v1",
      "run_id":"SUBJECT2_KNIGHT_SOLVED_LINEAGE_V1_20260929",
      "stage13":{
        "qualification_every_view_passed":stage13_report.get("every_view_passed"),
        "view_pass_count":sum(bool(v.get("passed")) for v in views),
        "view_count":len(views),
        "min_recall":min((float(v.get("silhouette_recall",1.0)) for v in views),default=None),
        "min_precision":min((float(v.get("silhouette_precision",1.0)) for v in views),default=None),
        "max_edge_p95_px":max((float(v.get("silhouette_edge_p95_px",0.0)) for v in views),default=None),
        "metadata":s13.get("metadata") or {},
      },
      "stage14":{
        "adequacy_status":s14_report.get("status"),
        "selected_target_node_cap":s14_report.get("selected_target_node_cap"),
        "selected_actual_node_count":s14_report.get("selected_actual_node_count"),
        "evaluated_candidate_count":len(evaluated),
        "passing_candidate_count":passing,
        "closest_nonpassing":closest,
        "surface_quality":s14q,
      },
      "stage15":{
        "qualification_status":s15_qual.get("status"),
        "qualification":s15_qual,
        "surface_quality":s15q,
      },
      "lineage":{
        "stage14_stage15_geometry_lineage_equal":same_lineage,
        "stage14_stage15_compact_faces_equal":same_faces,
        "stage14_geometry_lineage_hash":s14_surface.get("geometry_lineage_hash"),
        "stage15_geometry_lineage_hash":s15_surface.get("geometry_lineage_hash"),
        "stage14_provenance_hash":s14_prov.get("provenance_hash"),
        "stage15_provenance_hash":s15_prov.get("provenance_hash"),
      },
      "policy":{
        "g3_min_angle_deg":float(policy.g3_min_angle_deg),
        "g3_max_aspect":float(policy.g3_max_aspect_longest_over_min_altitude),
      },
      "finding":{
        "stage13_scientifically_passed":bool(stage13_report.get("every_view_passed")),
        "stage14_product_adequacy_passed":bool(str(s14_report.get("status"))=="PASS"),
        "stage14_has_static_quality_gate":False,
        "stage14_has_full_vertex_link_manifold_gate":False,
        "stage15_inherits_stage14_geometry":same_lineage and same_faces,
      },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_UPSTREAM_SURFACE_QUALITY="+json.dumps({
      "stage13_every_view_passed":report["stage13"]["qualification_every_view_passed"],
      "stage14_adequacy_status":report["stage14"]["adequacy_status"],
      "stage14_passing_candidates":passing,
      "stage14_nodes":s14q["node_count"],
      "stage14_bad_faces":s14q["quality_violating_face_count"],
      "stage14_min_angle":s14q["min_angle_deg"],
      "stage14_max_aspect":s14q["max_aspect"],
      "stage14_nonmanifold_edges":s14q["topology"]["nonmanifold_edge_count"],
      "stage14_illegal_vertex_links":s14q["topology"]["illegal_vertex_link_count"],
      "stage15_same_geometry":same_lineage and same_faces,
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
