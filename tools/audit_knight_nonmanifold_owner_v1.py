from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.demo.render_knight_motion_preview_v1 import _ctx


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root.expanduser().resolve(),a.run_id)
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"
    ))
    vi={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    vertices=list(candidate.vertices)
    faces=[[vi[str(x)] for x in f] for f in candidate.faces]

    def vclass(i:int)->str:
        v=vertices[i]
        mode=str(v.support_binding.mode)
        coeffs=tuple(v.support_binding.coefficients)
        if mode=="IDENTITY_SURFACE_NODE" and len(coeffs)==1 and abs(float(coeffs[0][1])-1.0)<=1e-10:
            return "IDENTITY"
        if mode=="SEAM_GEOMETRY_INTERPOLATION":
            return "SEAM_GENERATED"
        return "OTHER_GENERATED"

    def fclass(face)->str:
        cs=[vclass(i) for i in face]
        if all(x=="IDENTITY" for x in cs): return "IDENTITY_FACE"
        if all(x!="IDENTITY" for x in cs): return "ALL_GENERATED_FACE"
        return "MIXED_GENERATED_FACE"

    incidence=defaultdict(list)
    for fi,face in enumerate(faces):
        for ia,ib in ((0,1),(1,2),(2,0)):
            x,y=face[ia],face[ib]
            e=(x,y) if x<y else (y,x)
            incidence[e].append(fi)

    bad={e:rows for e,rows in incidence.items() if len(rows)>2}
    incidence_hist=Counter(len(rows) for rows in bad.values())
    endpoint_class=Counter()
    incident_face_class=Counter()
    edge_face_class_signature=Counter()
    component_signature=Counter()
    examples=[]

    graph=defaultdict(set)
    for (a0,b0),rows in bad.items():
        graph[a0].add(b0); graph[b0].add(a0)
        endpoint_class[tuple(sorted((vclass(a0),vclass(b0))))]+=1
        classes=[fclass(faces[fi]) for fi in rows]
        incident_face_class.update(classes)
        edge_face_class_signature[tuple(sorted(Counter(classes).items()))]+=1
        comps=[]
        for fi in rows:
            cs={str(vertices[v].component_id) for v in faces[fi]}
            comps.append("+".join(sorted(cs)))
        component_signature[tuple(sorted(Counter(comps).items()))]+=1
        if len(examples)<64:
            examples.append({
                "edge_vertex_indices":[a0,b0],
                "endpoint_vertex_classes":[vclass(a0),vclass(b0)],
                "endpoint_modes":[str(vertices[a0].support_binding.mode),str(vertices[b0].support_binding.mode)],
                "incidence_count":len(rows),
                "incident_face_indices":rows,
                "incident_face_classes":classes,
                "incident_face_components":comps,
            })

    seen=set(); cluster_sizes=[]
    for start in graph:
        if start in seen: continue
        q=[start]; seen.add(start); count_edges=0; verts=0
        while q:
            u=q.pop(); verts+=1; count_edges+=len(graph[u])
            for v in graph[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        cluster_sizes.append({"vertex_count":verts,"edge_count":count_edges//2})
    cluster_sizes.sort(key=lambda x:(x["edge_count"],x["vertex_count"]),reverse=True)

    bad_vertices=set(v for e in bad for v in e)
    report={
        "schema":"RealSaS.KnightNonmanifoldOwnerAudit.v1",
        "status":"MEASURED",
        "run_id":a.run_id,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "vertex_count":len(vertices),
        "face_count":len(faces),
        "nonmanifold_edge_count":len(bad),
        "nonmanifold_vertex_count":len(bad_vertices),
        "incidence_histogram":{str(k):v for k,v in sorted(incidence_hist.items())},
        "endpoint_class_counts":{str(k):v for k,v in endpoint_class.most_common()},
        "incident_face_class_counts":dict(incident_face_class),
        "edge_face_class_signatures":[
            {"signature":[[k,n] for k,n in sig],"edge_count":count}
            for sig,count in edge_face_class_signature.most_common()
        ],
        "component_signatures":[
            {"signature":[[k,n] for k,n in sig],"edge_count":count}
            for sig,count in component_signature.most_common(32)
        ],
        "nonmanifold_cluster_count":len(cluster_sizes),
        "largest_clusters":cluster_sizes[:32],
        "examples":examples,
        "candidate_metadata":{
            "producer_id":candidate.producer_id,
            "face_provenance_mode":dict(candidate.metadata or {}).get("face_provenance_mode"),
            "mechanical_skin_transfer":dict(candidate.metadata or {}).get("mechanical_skin_transfer"),
            "face_deletion_count":dict(candidate.metadata or {}).get("face_deletion_count"),
            "three_clique_face_minting_allowed":dict(candidate.metadata or {}).get("three_clique_face_minting_allowed"),
        },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_NONMANIFOLD_OWNER="+json.dumps({
        "candidate":candidate.candidate_lineage_hash,
        "nonmanifold_edge_count":len(bad),
        "nonmanifold_vertex_count":len(bad_vertices),
        "incidence_histogram":report["incidence_histogram"],
        "endpoint_class_counts":report["endpoint_class_counts"],
        "incident_face_class_counts":report["incident_face_class_counts"],
        "cluster_count":len(cluster_sizes),
        "largest_clusters":cluster_sizes[:10],
        "top_face_signatures":report["edge_face_class_signatures"][:10],
    },sort_keys=True))


if __name__=="__main__":
    main()
