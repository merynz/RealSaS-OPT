from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    bind_region_visual_vertices_to_mechanical_affine_v1,
    build_visual_mesh_from_region_labels_v1,
    evaluate_region_visual_binding_v1,
)
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)

CLIPS=("demo_idle_v1","demo_run_v1","demo_slash_v1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"SLOT_COURT_INPUT_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_motion_helper(path: Path):
    spec=importlib.util.spec_from_file_location("realsas_slot_court_motion_helper",path)
    if spec is None or spec.loader is None:
        raise RuntimeError("SLOT_COURT_MOTION_HELPER_IMPORT_SPEC_INVALID")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_tracks_for_clip","_candidate_skin_weights","_skin"):
        if not hasattr(module,name):
            raise RuntimeError("SLOT_COURT_MOTION_HELPER_SYMBOL_MISSING:"+name)
    return module


def _sealed_parent_inputs(report_path: Path) -> dict[str, dict]:
    report=_load_json(report_path)
    rows=tuple(report.get("input_artifacts") or ())
    if len(rows)<6:
        raise RuntimeError("SLOT_COURT_PARENT_INPUT_SET_INCOMPLETE")
    by_schema={}
    for row in rows:
        path=Path(os.path.expandvars(str(row["path"]))).expanduser().resolve()
        expected=str(row.get("sha256") or "")
        if not path.is_file() or len(expected)!=64 or _sha256(path)!=expected:
            raise RuntimeError("SLOT_COURT_PARENT_INPUT_SHA_DRIFT:"+str(path))
        payload=_load_json(path)
        schema=str(payload.get("schema_version") or payload.get("schema") or "")
        if not schema:
            raise RuntimeError("SLOT_COURT_PARENT_INPUT_SCHEMA_MISSING:"+str(path))
        if schema in by_schema:
            raise RuntimeError("SLOT_COURT_PARENT_INPUT_SCHEMA_DUPLICATE:"+schema)
        by_schema[schema]={"payload":payload,"path":str(path),"sha256":expected}
    return by_schema


def _require(parent: dict[str,dict],schema: str) -> dict:
    if schema not in parent:
        raise RuntimeError("SLOT_COURT_PARENT_SCHEMA_MISSING:"+schema)
    return parent[schema]["payload"]


def _candidate_face_indices(candidate) -> np.ndarray:
    index={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    faces=np.asarray(
        [[index[str(vertex_id)] for vertex_id in face] for face in candidate.faces],
        dtype=np.int64,
    )
    if faces.shape!=(len(candidate.faces),3):
        raise RuntimeError("SLOT_COURT_FACE_INDEX_DRIFT")
    return faces


def _logical_slot_face_labels(candidate) -> tuple[np.ndarray,tuple[dict,...]]:
    """Current Stage37 component + connected-face-island semantics, on candidate IDs."""
    component_by_vertex={
        str(v.candidate_vertex_id):str(v.component_id)
        for v in candidate.vertices
    }
    face_component=[]
    by_component=defaultdict(list)
    for fi,face in enumerate(candidate.faces):
        comps={component_by_vertex[str(vertex_id)] for vertex_id in face}
        if len(comps)!=1:
            raise RuntimeError("SLOT_COURT_FACE_CROSSES_COMPONENT:"+str(fi))
        comp=next(iter(comps))
        face_component.append(comp)
        by_component[comp].append(int(fi))

    labels=np.full((len(candidate.faces),),-1,dtype=np.int32)
    rows=[]
    next_slot=0
    for component_id in sorted(by_component):
        selected=set(by_component[component_id])
        edge_faces=defaultdict(list)
        for fi in sorted(selected):
            face=tuple(map(str,candidate.faces[fi]))
            for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
                edge_faces[tuple(sorted((a,b)))].append(fi)
        neighbors={fi:set() for fi in selected}
        for incident in edge_faces.values():
            for a in incident:
                for b in incident:
                    if a!=b:
                        neighbors[int(a)].add(int(b))
        unseen=set(selected)
        while unseen:
            seed=min(unseen);unseen.remove(seed)
            stack=[seed];group=[]
            while stack:
                current=stack.pop();group.append(current)
                for nxt in sorted(neighbors[current]):
                    if nxt in unseen:
                        unseen.remove(nxt);stack.append(nxt)
            group=tuple(sorted(group))
            for fi in group:
                labels[fi]=next_slot
            rows.append({
                "slot_index":next_slot,
                "component_id":component_id,
                "face_count":len(group),
                "first_face_index":group[0],
            })
            next_slot+=1
    if np.any(labels<0):
        raise RuntimeError("SLOT_COURT_FACE_SLOT_ACCOUNTING_INCOMPLETE")
    return labels,tuple(rows)


def _partition_source_by_slot_adjacency(
    mask:np.ndarray,
    owner:np.ndarray,
    faces:np.ndarray,
    face_slot:np.ndarray,
):
    mask=np.asarray(mask,dtype=bool)
    owner=np.asarray(owner,dtype=np.int64)
    faces=np.asarray(faces,dtype=np.int64)
    if owner.shape!=mask.shape:
        raise RuntimeError("SLOT_COURT_OWNER_SHAPE_DRIFT")
    valid=mask & (owner>=0)
    if np.any(owner[valid]>=len(faces)):
        raise RuntimeError("SLOT_COURT_OWNER_FACE_DRIFT")

    edge_faces=defaultdict(list)
    for fi,(a,b,c) in enumerate(faces.tolist()):
        for u,v in ((a,b),(b,c),(c,a)):
            edge_faces[(min(u,v),max(u,v))].append(fi)
    allowed=set()
    for incident in edge_faces.values():
        if len(incident)==2:
            a,b=map(int,incident)
            if int(face_slot[a])==int(face_slot[b]):
                allowed.add((min(a,b),max(a,b)))

    h,w=mask.shape
    linear=np.arange(h*w,dtype=np.int64).reshape(h,w)
    parent=np.arange(h*w,dtype=np.int64)
    rank=np.zeros(h*w,dtype=np.uint8)

    def find(x:int)->int:
        x=int(x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=int(parent[x])
        return x

    def union(a:int,b:int):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if rank[ra]<rank[rb]:ra,rb=rb,ra
        parent[rb]=ra
        if rank[ra]==rank[rb]:rank[ra]+=1

    def connect(a:int,b:int)->bool:
        if int(face_slot[a])!=int(face_slot[b]):return False
        return a==b or (min(a,b),max(a,b)) in allowed

    for y in range(h):
        for x in np.flatnonzero(valid[y]).tolist():
            fa=int(owner[y,x])
            if x+1<w and valid[y,x+1] and connect(fa,int(owner[y,x+1])):
                union(int(linear[y,x]),int(linear[y,x+1]))
            if y+1<h and valid[y+1,x] and connect(fa,int(owner[y+1,x])):
                union(int(linear[y,x]),int(linear[y+1,x]))

    seed=np.full(mask.shape,-1,dtype=np.int32)
    root_to_chart={}
    chart_slot={}
    for y,x in zip(*np.nonzero(valid)):
        root=find(int(linear[y,x]))
        if root not in root_to_chart:
            cid=len(root_to_chart)
            root_to_chart[root]=cid
            chart_slot[cid]=int(face_slot[int(owner[y,x])])
        cid=root_to_chart[root]
        if chart_slot[cid]!=int(face_slot[int(owner[y,x])]):
            raise RuntimeError("SLOT_COURT_CHART_SLOT_DRIFT")
        seed[y,x]=cid
    if not root_to_chart:
        raise RuntimeError("SLOT_COURT_NO_SAFE_DRIVER_SEEDS")

    # Fill foreground pixels that have no rest-space mechanical first-hit only
    # inside their own source connected component.
    from scipy import ndimage
    from scipy.spatial import cKDTree
    source_cc,ncc=ndimage.label(
        mask,
        structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8),
    )
    final=np.full(mask.shape,-1,dtype=np.int32)
    max_fill=0.0
    p95_parts=[]
    for cc in range(1,int(ncc)+1):
        cm=source_cc==cc
        ys,xs=np.nonzero(cm & (seed>=0))
        if not len(xs):
            raise RuntimeError(
                "SLOT_COURT_SOURCE_COMPONENT_WITHOUT_MECHANICAL_DRIVER:"+str(cc)
            )
        qy,qx=np.nonzero(cm)
        tree=cKDTree(np.column_stack((xs.astype(float),ys.astype(float))))
        dist,nn=tree.query(np.column_stack((qx.astype(float),qy.astype(float))),k=1)
        nn=np.asarray(nn,dtype=np.int64)
        assigned=seed[ys[nn],xs[nn]]
        if np.any(assigned<0):
            raise RuntimeError("SLOT_COURT_FILL_INVALID")
        final[qy,qx]=assigned
        if len(dist):
            max_fill=max(max_fill,float(np.max(dist)))
            p95_parts.extend(map(float,np.asarray(dist).tolist()))
    if np.any(final[mask]<0):
        raise RuntimeError("SLOT_COURT_FOREGROUND_UNASSIGNED")
    return final,seed,chart_slot,max_fill,float(np.quantile(p95_parts,.95) if p95_parts else 0.0)


def _visual_qa(rest,posed,faces):
    rest=np.asarray(rest,float);posed=np.asarray(posed,float);faces=np.asarray(faces,np.int64)
    rr=rest[faces];pp=posed[faces]
    def area(t):
        a=t[:,1]-t[:,0];b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ra=area(rr);pa=area(pp)
    flips=int(np.count_nonzero((ra*pa)<0))
    re=np.stack((
        np.linalg.norm(rr[:,1]-rr[:,0],axis=1),
        np.linalg.norm(rr[:,2]-rr[:,1],axis=1),
        np.linalg.norm(rr[:,0]-rr[:,2],axis=1),
    ),axis=1)
    pe=np.stack((
        np.linalg.norm(pp[:,1]-pp[:,0],axis=1),
        np.linalg.norm(pp[:,2]-pp[:,1],axis=1),
        np.linalg.norm(pp[:,0]-pp[:,2],axis=1),
    ),axis=1)
    ratio=pe/np.maximum(re,1e-9)
    return {
        "flipped_triangles":flips,
        "flip_fraction":float(flips/max(1,len(faces))),
        "p95_edge_stretch":float(np.quantile(ratio,.95)),
        "p99_edge_stretch":float(np.quantile(ratio,.99)),
        "max_edge_stretch":float(np.max(ratio)),
    }


def run(*,authority_root:Path,run_id:str,parent_report:Path,motion_helper_path:Path,source_report_path:Path,out_path:Path):
    root=authority_root/"runs"/run_id
    manifest=_load_json(root/"run_manifest.json")
    ctx={"run_manifest":manifest}
    parent=_sealed_parent_inputs(parent_report)

    candidate=canonical_mesh_candidate_from_dict(_require(parent,"RealSaS.CanonicalMeshCandidateIR.v1"))
    skeleton=qualified_skeleton_from_dict(_require(parent,"RealSaS.QualifiedSkeletonIR.v1"))
    skin=qualified_skin_from_dict(_require(parent,"RealSaS.QualifiedSkinIR.v1"))
    camera_set=qualified_camera_set_from_dict(_require(parent,"RealSaS.QualifiedCameraSetIR.v1"))
    observation=qualified_observation_set_from_dict(_require(parent,"RealSaS.QualifiedObservationSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda c:int(c.view_index)))
    _rgba,masks=_load_source_inputs(ctx,observation)

    motion_helper=_load_motion_helper(motion_helper_path)
    source_report=_load_json(source_report_path)
    joint_ids,W=motion_helper._candidate_skin_weights(candidate,skin,skeleton)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    faces=_candidate_face_indices(candidate)
    face_slot,slot_rows=_logical_slot_face_labels(candidate)

    clip_rows={}
    for clip_id in CLIPS:
        payload=_load_json(
            root/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json"
        )
        tracks,mapping=motion_helper._tracks_for_clip(
            payload,skeleton,cameras,source_report
        )
        clip_rows[clip_id]=(payload,tracks,mapping)

    report={
        "schema":"RealSaS.KnightSlotPresentationHypothesisCourt.v2",
        "status":"MEASURED_HYPOTHESIS_COURT",
        "product_authority_claimed":False,
        "historical_parent_authority_claimed":False,
        "mechanical_truth_mutated":False,
        "parent_input_sha_verified":True,
        "hypothesis":"LOGICAL_SLOT_X_VIEW_LOCAL_CHART_X_3D_MECHANICAL_DRIVER",
        "logical_slot_count":len(slot_rows),
        "logical_slots":slot_rows,
        "views":[],
        "hypothesis_verdict":"UNSET",
        "blockers":[],
    }

    any_measured=False
    for vi in (0,2):
        mask=np.asarray(masks[vi],dtype=bool)
        camera=cameras[vi]
        vis=rasterize_visible_owner(
            candidate,camera,positions=rest,
            width=mask.shape[1],height=mask.shape[0],max_layers=4,
        )
        labels,seeds,chart_slot,max_fill,p95_fill=_partition_source_by_slot_adjacency(
            mask,vis.owner_face_index,faces,face_slot
        )
        region=build_visual_mesh_from_region_labels_v1(mask,labels,target_edge_px=16)
        visual_mesh=region.mesh
        row={
            "view_index":vi,
            "visible_logical_slot_count":len(set(chart_slot.values())),
            "chart_count":len(chart_slot),
            "visual_vertex_count":int(len(visual_mesh.positions)),
            "visual_face_count":int(len(visual_mesh.faces)),
            "source_foreground_pixel_count":int(mask.sum()),
            "max_unowned_fill_distance_px":max_fill,
            "p95_unowned_fill_distance_px":p95_fill,
            "binding_status":"UNSET",
            "qa_by_clip":{},
        }
        try:
            binding=bind_region_visual_vertices_to_mechanical_affine_v1(
                points_source_xy=visual_mesh.positions,
                vertex_region_id=region.vertex_region_id,
                seed_region_labels=seeds,
                owner_face_index=vis.owner_face_index,
                mechanical_positions_xyz=rest,
                mechanical_faces=faces,
                camera=camera,
                candidate_seed_count=16,
                max_seed_distance_px=8.0,
            )
        except QualificationError as exc:
            row["binding_status"]="FAIL_CLOSED"
            row["binding_blocker"]=str(exc)
            report["blockers"].append(f"V{vi}:{exc}")
            report["views"].append(row)
            continue

        row["binding_status"]="PASS"
        row["max_binding_seed_distance_px"]=float(np.max(binding["nearest_safe_seed_distance_px"]))
        row["p95_binding_seed_distance_px"]=float(np.quantile(binding["nearest_safe_seed_distance_px"],.95))
        row["max_affine_extrapolation_penalty"]=float(np.max(binding["extrapolation_penalty"]))
        rest_eval=evaluate_region_visual_binding_v1(
            binding,posed_mechanical_positions_xyz=rest,camera=camera
        )
        row["max_rest_error_px"]=float(np.max(
            np.linalg.norm(rest_eval-np.asarray(visual_mesh.positions,float),axis=1)
        ))

        for clip_id in CLIPS:
            payload,tracks,_mapping=clip_rows[clip_id]
            times=np.linspace(
                0.0,float(payload["duration_seconds"]),4,
                endpoint=not bool(payload.get("loop")),dtype=np.float64
            )
            qa=[]
            for t in times:
                skin_mats,_,_=_joint_pose_v2(
                    skeleton=skeleton,tracks=tracks,
                    time_seconds=float(t),cameras=cameras,
                )
                posed=motion_helper._skin(rest,W,joint_ids,skin_mats)
                visual=evaluate_region_visual_binding_v1(
                    binding,posed_mechanical_positions_xyz=posed,camera=camera
                )
                qa.append({
                    "time_seconds":float(t),
                    **_visual_qa(visual_mesh.positions,visual,visual_mesh.faces),
                })
            row["qa_by_clip"][clip_id]=qa
        report["views"].append(row)
        any_measured=True

    if report["blockers"]:
        report["hypothesis_verdict"]="BINDING_COVERAGE_OR_DISTANCE_FAIL_CLOSED"
    elif any_measured:
        report["hypothesis_verdict"]="MEASURED_DYNAMIC_SLOT_CHART_BEHAVIOR"
    else:
        report["hypothesis_verdict"]="NO_MEASURABLE_VIEW"

    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--parent-report",required=True)
    p.add_argument("--motion-helper",required=True)
    p.add_argument("--source-report",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(
        authority_root=Path(a.authority_root).resolve(),
        run_id=a.run_id,
        parent_report=Path(a.parent_report).resolve(),
        motion_helper_path=Path(a.motion_helper).resolve(),
        source_report_path=Path(a.source_report).resolve(),
        out_path=Path(a.out).resolve(),
    )

if __name__=="__main__":
    main()
