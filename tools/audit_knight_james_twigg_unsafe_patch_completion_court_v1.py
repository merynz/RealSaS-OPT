from __future__ import annotations

import argparse, heapq, json
from collections import defaultdict, deque
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
from tools.audit_knight_james_twigg_region_inference_v1 import (
    EPSILON, PROBE_DEGREES,
    _cluster_signatures, _rotation_signatures, _split_holeless,
    _teacher_region_eval, _vertex_labels_from_faces,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask, exact, face_indices, load, motion_metrics,
    stress_arbitrary_weights,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx

def build_face_adjacency(F, mask):
    edge_faces=defaultdict(list)
    for fi in np.where(mask)[0]:
        a,b,c=map(int,F[fi])
        for x,y in ((a,b),(b,c),(c,a)):
            if x>y:x,y=y,x
            edge_faces[(x,y)].append(int(fi))
    adj=defaultdict(set)
    for fs in edge_faces.values():
        if len(fs)<2: continue
        for i in range(len(fs)):
            for j in range(i+1,len(fs)):
                adj[fs[i]].add(fs[j]);adj[fs[j]].add(fs[i])
    return adj

def unsafe_components(F, unsafe, dense):
    mask=np.asarray(unsafe)&np.asarray(dense)
    adj=build_face_adjacency(F,mask)
    unseen=set(map(int,np.where(mask)[0]))
    comps=[]
    while unseen:
        s=min(unseen);unseen.remove(s);q=[s];comp=[]
        while q:
            u=q.pop();comp.append(u)
            for v in adj.get(u,()):
                if v in unseen:
                    unseen.remove(v);q.append(v)
        comps.append(sorted(comp))
    comps.sort(key=lambda x:(-len(x),x[0]))
    return comps

def edge_graph_from_faces(P,F,face_indices_subset):
    edges={}
    for fi in face_indices_subset:
        a,b,c=map(int,F[int(fi)])
        for x,y in ((a,b),(b,c),(c,a)):
            if x>y:x,y=y,x
            w=float(np.linalg.norm(P[x]-P[y]))
            if not np.isfinite(w) or w<=1e-15: continue
            edges[(x,y)]=min(edges.get((x,y),np.inf),w)
    nbr=defaultdict(list)
    for (a,b),w in edges.items():
        nbr[a].append((b,w));nbr[b].append((a,w))
    return nbr,edges

def complete_unsafe_patches(P,F,dense,unsafe,initial_label):
    labels=np.asarray(initial_label,dtype=np.int64).copy()

    # Dense-only graph: invented clique faces cannot create repair support.
    dense_faces=np.where(dense)[0]
    dense_nbr,dense_edges=edge_graph_from_faces(P,F,dense_faces)

    # Which faces touch each vertex, used to recognize safe boundary neighbors.
    vfaces=defaultdict(list)
    for fi,face in enumerate(F.tolist()):
        for v in face:vfaces[int(v)].append(int(fi))

    comps=unsafe_components(F,unsafe,dense)
    reports=[]
    total_new=0
    for ci,comp in enumerate(comps):
        patch_vertices=sorted(set(int(v) for fi in comp for v in F[fi]))
        patch_set=set(patch_vertices)

        # Boundary seeds are already-inferred safe vertices directly adjacent through
        # a dense edge to the unsafe patch. No teacher labels are consulted.
        seeds=[]
        for u in patch_vertices:
            for v,w in dense_nbr.get(u,()):
                safe_neighbor=any((not unsafe[fi]) and dense[fi] for fi in vfaces[v])
                if v not in patch_set and safe_neighbor and labels[v]>=0:
                    seeds.append((int(v),int(labels[v]),float(w)))
                elif v in patch_set and labels[v]>=0:
                    # A high-confidence label already reaching a patch vertex is also
                    # valid as a boundary anchor.
                    seeds.append((int(v),int(labels[v]),0.0))
        # unique seed vertex -> deterministic existing label
        seed_map={}
        for v,lab,w in seeds:
            seed_map.setdefault(v,lab)
        unique_labels=sorted(set(seed_map.values()))

        # Fail-closed: one region on the whole boundary is not evidence of a seam.
        if len(unique_labels)<2:
            reports.append({
              "component_index":ci,"unsafe_face_count":len(comp),
              "vertex_count":len(patch_vertices),"boundary_seed_count":len(seed_map),
              "boundary_label_count":len(unique_labels),"completed_vertex_count":0,
              "status":"ABSTAIN_LT2_BOUNDARY_REGIONS",
            })
            continue

        allowed=set(patch_vertices)|set(seed_map.keys())
        best={v:np.inf for v in allowed};best_lab={v:-1 for v in allowed}
        heap=[]
        for v,lab in sorted(seed_map.items()):
            best[v]=0.0;best_lab[v]=lab;heapq.heappush(heap,(0.0,lab,v))
        while heap:
            d,lab,u=heapq.heappop(heap)
            if d!=best[u] or lab!=best_lab[u]:continue
            for v,w in dense_nbr.get(u,()):
                if v not in allowed:continue
                nd=d+w
                if nd<best[v]-1e-15 or (abs(nd-best[v])<=1e-15 and (best_lab[v]<0 or lab<best_lab[v])):
                    best[v]=nd;best_lab[v]=lab;heapq.heappush(heap,(nd,lab,v))

        changed=0;reached=0
        for v in patch_vertices:
            if np.isfinite(best.get(v,np.inf)):
                reached+=1
                # Only fill unresolved patch vertices. Existing James/Twigg labels
                # remain immutable anchors.
                if labels[v]<0:
                    labels[v]=best_lab[v];changed+=1
        total_new+=changed
        reports.append({
          "component_index":ci,"unsafe_face_count":len(comp),
          "vertex_count":len(patch_vertices),"boundary_seed_count":len(seed_map),
          "boundary_label_count":len(unique_labels),
          "reachable_patch_vertex_count":reached,
          "completed_vertex_count":changed,
          "status":"COMPLETED" if reached==len(patch_vertices) else "PARTIAL",
        })
    return labels,reports,{"unsafe_patch_count":len(comps),"newly_labeled_vertex_count":total_new}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--weights-npz",type=Path,required=True)
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
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)

    P=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    F=face_indices(cand)
    dense=dense_supported_face_mask(rr,cand,surface)
    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        W=np.asarray(z["arachne"],dtype=np.float64)
    if W.shape!=(len(P),len(jids)):raise RuntimeError("PATCH_COMPLETION_WEIGHT_SHAPE_DRIFT")

    base_stress=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),dtype=bool)
    unsafe[np.asarray(base_stress["unsafe_face_indices"],dtype=np.int64)]=True
    safe=(~unsafe)&dense

    Z,valid,probes=_rotation_signatures(P,F,W,jids,sk,cams,PROBE_DEGREES)
    face_label,core,cluster_meta=_cluster_signatures(Z,safe&valid,EPSILON)
    initial_label,conf,prop_meta=_vertex_labels_from_faces(P,F,face_label,core,dense,safe)

    completed_label,patch_reports,patch_meta=complete_unsafe_patches(
        P,F,dense,unsafe,initial_label
    )

    before_eval=_teacher_region_eval(a.teacher_bank,a.teacher_source,sk,cand,initial_label,F,unsafe)
    after_eval=_teacher_region_eval(a.teacher_bank,a.teacher_source,sk,cand,completed_label,F,unsafe)

    P2,W2,F2,split_meta=_split_holeless(P,W,F,completed_label)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    before_motion=motion_metrics(P,W,F,jids,sk,cams,rr,source_report)
    after_motion=motion_metrics(P2,W2,F2,jids,sk,cams,rr,source_report)
    after_stress=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy)

    report={
      "schema":"RealSaS.KnightJamesTwiggUnsafePatchCompletionCourt.v1",
      "status":"TEACHER_FREE_REGION_COMPLETION__NO_PRODUCT_MUTATION",
      "teacher_used_by_region_inference":False,
      "operator":{
        "seed_source":"JAMES_TWIGG_ARACHNE_ROTATION_SIGNATURE_CORE_LABELS",
        "patch_source":"STAGE35_ALL_FACE_UNSAFE_CONNECTED_COMPONENTS",
        "support_graph":"DENSE_FACE_SUPPORTED_EDGES_ONLY",
        "completion":"PATCH_LOCAL_MULTI_SOURCE_GEODESIC_VORONOI",
        "abstain_rule":"LT2_DISTINCT_BOUNDARY_REGION_LABELS",
        "holeless_repair":"VERTEX_DUPLICATION_PLUS_REGION_PURE_SUBDIVISION",
      },
      "probe":{"degrees":PROBE_DEGREES,"count":len(probes)},
      "clustering":cluster_meta,
      "initial_propagation":prop_meta,
      "patch_completion":{**patch_meta,
        "completed_patch_count":sum(r["status"]=="COMPLETED" for r in patch_reports),
        "partial_patch_count":sum(r["status"]=="PARTIAL" for r in patch_reports),
        "abstained_patch_count":sum(r["status"].startswith("ABSTAIN") for r in patch_reports),
      },
      "patches":patch_reports,
      "teacher_evaluation_only":{
        "before_patch_completion":before_eval,
        "after_patch_completion":after_eval,
      },
      "holeless_split":split_meta,
      "mechanics":{
        "before_motion":{k:v for k,v in before_motion.items() if k!="frames"},
        "after_motion":{k:v for k,v in after_motion.items() if k!="frames"},
        "before_stress":{k:v for k,v in base_stress.items() if k!="unsafe_face_indices"},
        "after_stress":{k:v for k,v in after_stress.items() if k!="unsafe_face_indices"},
      },
      "finding":{
        "patch_completion_improves_unsafe_cross_region_recall":bool(after_eval["unsafe_truth_cross_region_recall"]>before_eval["unsafe_truth_cross_region_recall"]),
        "holeless_split_reduces_gt10":bool(after_motion["max_edge_gt_10"]<before_motion["max_edge_gt_10"]),
        "holeless_split_reduces_gt4":bool(after_motion["max_edge_gt_4"]<before_motion["max_edge_gt_4"]),
        "holeless_split_reduces_synthetic_unsafe":bool(after_stress["unsafe_face_count"]<base_stress["unsafe_face_count"]),
        "holeless_split_eliminates_gt10":bool(after_motion["max_edge_gt_10"]==0),
        "holeless_split_eliminates_gt4":bool(after_motion["max_edge_gt_4"]==0),
      },
      "claim_boundary":"Teacher source topology is evaluation-only. No teacher label participates in clustering, patch discovery, geodesic completion, abstention, or holeless split."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_JT_UNSAFE_PATCH_COMPLETION_PASS",json.dumps({
      "patch_completion":report["patch_completion"],
      "teacher_eval":report["teacher_evaluation_only"],
      "before_motion":report["mechanics"]["before_motion"],
      "after_motion":report["mechanics"]["after_motion"],
      "before_stress":report["mechanics"]["before_stress"],
      "after_stress":report["mechanics"]["after_stress"],
      "finding":report["finding"],
    },sort_keys=True))

if __name__=="__main__":main()
