from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict,
    qualified_mesh_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    qualified_dynamic_motion_v2_from_dict,
)
from compiler.realsas_compiler_core.product_state_v2 import (
    presentation_structure_v2_from_dict,
)
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    bind_region_visual_vertices_to_mechanical_affine_v1,
    build_visual_mesh_from_region_labels_v1,
    evaluate_region_visual_binding_v1,
)
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)
CLIP_IDS=("demo_idle_v1","demo_run_v1","demo_slash_v1")


def _read_json(path: Path):
    if not path.is_file():
        raise RuntimeError(f"SLOT_COURT_ARTIFACT_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_payload(root: Path, ledger: dict, stage_id: str, schema: str):
    row=next((x for x in ledger.get("stages") or () if str(x.get("id"))==stage_id),None)
    if row is None:
        raise RuntimeError(f"SLOT_COURT_STAGE_MISSING:{stage_id}")
    matches=[x for x in row.get("outputs") or () if str(x.get("schema"))==schema]
    if len(matches)!=1:
        raise RuntimeError(
            f"SLOT_COURT_SCHEMA_CARDINALITY:{stage_id}:{schema}:{len(matches)}"
        )
    ref=matches[0]
    raw=os.path.expandvars(str(ref.get("path") or ""))
    path=Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"SLOT_COURT_LEDGER_OUTPUT_MISSING:{stage_id}:{schema}:{path}")
    expected=str(ref.get("sha256") or "")
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if len(expected)!=64 or actual!=expected:
        raise RuntimeError(
            f"SLOT_COURT_LEDGER_OUTPUT_SHA_DRIFT:{stage_id}:{schema}"
        )
    payload=_read_json(path)
    embedded=str(payload.get("schema_version") or payload.get("schema") or "")
    if embedded!=schema:
        raise RuntimeError(
            f"SLOT_COURT_EMBEDDED_SCHEMA_DRIFT:{stage_id}:{schema}:{embedded}"
        )
    return payload


def _slot_face_labels(structure, face_count: int) -> tuple[np.ndarray, dict[int,str]]:
    labels=np.full((face_count,),-1,dtype=np.int32)
    slot_by_index={}
    for slot_index,slot in enumerate(structure.slots):
        slot_by_index[int(slot_index)]=str(slot.slot_id)
        faces=tuple(int(x) for x in dict(slot.metadata or {}).get("mesh_face_indices") or ())
        if not faces:
            raise RuntimeError(f"SLOT_COURT_SLOT_WITHOUT_FACE_SET:{slot.slot_id}")
        for fi in faces:
            if fi<0 or fi>=face_count:
                raise RuntimeError(f"SLOT_COURT_FACE_INDEX_INVALID:{slot.slot_id}:{fi}")
            if labels[fi]>=0:
                raise RuntimeError(f"SLOT_COURT_FACE_MULTI_SLOT:{fi}")
            labels[fi]=int(slot_index)
    if np.any(labels<0):
        raise RuntimeError(
            "SLOT_COURT_FACE_NOT_OWNED:"
            + str(int(np.count_nonzero(labels<0)))
        )
    return labels,slot_by_index


def _partition_source_by_slot_adjacency(
    mask: np.ndarray,
    owner: np.ndarray,
    faces: np.ndarray,
    face_slot: np.ndarray,
):
    mask=np.asarray(mask,dtype=bool)
    owner=np.asarray(owner,dtype=np.int64)
    faces=np.asarray(faces,dtype=np.int64)
    if owner.shape!=mask.shape:
        raise RuntimeError("SLOT_COURT_OWNER_SHAPE_DRIFT")

    edge_faces=defaultdict(list)
    for fi,(a,b,c) in enumerate(faces.tolist()):
        for u,v in ((a,b),(b,c),(c,a)):
            edge_faces[(min(u,v),max(u,v))].append(fi)
    adjacent=set()
    for rows in edge_faces.values():
        if len(rows)==2:
            a,b=map(int,rows)
            if int(face_slot[a])==int(face_slot[b]):
                adjacent.add((min(a,b),max(a,b)))

    h,w=mask.shape
    valid=mask & (owner>=0)
    if np.any(owner[valid]>=len(faces)):
        raise RuntimeError("SLOT_COURT_OWNER_FACE_DRIFT")
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

    def connect(fa:int,fb:int)->bool:
        if fa==fb:return True
        if int(face_slot[fa])!=int(face_slot[fb]):return False
        return (min(fa,fb),max(fa,fb)) in adjacent

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
        raise RuntimeError("SLOT_COURT_NO_SEEDS")

    source_cc,ncc=ndimage.label(
        mask,
        structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8),
    )
    final=np.full(mask.shape,-1,dtype=np.int32)
    max_fill_distance=0.0
    for cc in range(1,int(ncc)+1):
        cm=source_cc==cc
        ys,xs=np.nonzero(cm & (seed>=0))
        if not len(xs):
            raise RuntimeError(f"SLOT_COURT_SOURCE_COMPONENT_WITHOUT_DRIVER:{cc}")
        qy,qx=np.nonzero(cm)
        tree=cKDTree(np.column_stack((xs.astype(float),ys.astype(float))))
        dist,nn=tree.query(np.column_stack((qx.astype(float),qy.astype(float))),k=1)
        nn=np.asarray(nn,dtype=np.int64)
        assigned=seed[ys[nn],xs[nn]]
        if np.any(assigned<0):
            raise RuntimeError("SLOT_COURT_FILL_INVALID")
        final[qy,qx]=assigned
        max_fill_distance=max(max_fill_distance,float(np.max(dist,initial=0.0)))

    if np.any(final[mask]<0):
        raise RuntimeError("SLOT_COURT_FOREGROUND_UNASSIGNED")
    return final,seed,chart_slot,max_fill_distance


def _visual_qa(rest,posed,faces):
    rest=np.asarray(rest,float)
    posed=np.asarray(posed,float)
    faces=np.asarray(faces,np.int64)
    rr=rest[faces]
    pp=posed[faces]
    def area(t):
        a=t[:,1]-t[:,0]
        b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ra=area(rr)
    pa=area(pp)
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


def run(*,authority_root:Path,run_id:str,out_path:Path):
    root=authority_root/"runs"/run_id
    manifest=_read_json(root/"run_manifest.json")
    ledger=_read_json(root/"ACTIVE_RUN_V2.json")
    ctx={"run_manifest":manifest}
    mesh=qualified_mesh_from_dict(_stage_payload(
        root,ledger,"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "RealSaS.QualifiedMeshIR.v1",
    ))
    structure=presentation_structure_v2_from_dict(_stage_payload(
        root,ledger,"37_QUALIFIED_PRESENTATION_STRUCTURE",
        "RealSaS.QualifiedPresentationStructureIR.v2",
    ))
    cameras=qualified_camera_set_from_dict(_stage_payload(
        root,ledger,"05_CAMERA_CONTRACT_SOLVED",
        "RealSaS.QualifiedCameraSetIR.v1",
    ))
    observation=qualified_observation_set_from_dict(_stage_payload(
        root,ledger,"07_OBSERVATION_CONTRACT_QUALIFIED",
        "RealSaS.QualifiedObservationSetIR.v1",
    ))
    dynamic=qualified_dynamic_motion_v2_from_dict(_stage_payload(
        root,ledger,"41_MOTION_DYNAMIC_PROOF",
        "RealSaS.QualifiedDynamicMotionIR.v2",
    ))

    ordered=tuple(sorted(cameras.cameras,key=lambda c:int(c.view_index)))
    _rgba_by_view,mask_by_view=_load_source_inputs(ctx,observation)
    vertex_ids=tuple(str(v.canonical_mesh_vertex_id) for v in mesh.vertices)
    vertex_index={vid:i for i,vid in enumerate(vertex_ids)}
    if len(vertex_index)!=len(vertex_ids):
        raise RuntimeError("SLOT_COURT_DUPLICATE_MECHANICAL_VERTEX")
    rest=np.asarray([v.P for v in mesh.vertices],dtype=np.float64)
    faces=np.asarray(
        [[vertex_index[str(vid)] for vid in face] for face in mesh.faces],
        dtype=np.int64,
    )
    face_slot,slot_by_index=_slot_face_labels(structure,len(faces))
    clips={clip.clip_id:clip for clip in dynamic.clips if clip.clip_id in CLIP_IDS}
    if set(clips)!=set(CLIP_IDS):
        raise RuntimeError("SLOT_COURT_DYNAMIC_CLIP_SET_INCOMPLETE")

    report={
        "schema":"RealSaS.KnightSlotPresentationHypothesisCourt.v1",
        "status":"MEASURED_HYPOTHESIS_COURT",
        "run_id":run_id,
        "product_authority_claimed":False,
        "mechanical_truth_mutated":False,
        "hypothesis":"LOGICAL_PRESENTATION_SLOT_X_VIEW_LOCAL_CHART_X_3D_MECHANICAL_DRIVER",
        "slot_count":len(structure.slots),
        "views":[],
    }
    for vi in (0,2):
        mask=np.asarray(mask_by_view[vi],dtype=bool)
        camera=ordered[vi]
        vis=rasterize_visible_owner(
            mesh,camera,positions=rest,
            width=mask.shape[1],height=mask.shape[0],max_layers=4,
        )
        labels,seeds,chart_slot,max_fill=_partition_source_by_slot_adjacency(
            mask,vis.owner_face_index,faces,face_slot
        )
        region=build_visual_mesh_from_region_labels_v1(
            mask,labels,target_edge_px=16
        )
        visual_mesh=region.mesh
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
        rest_eval=evaluate_region_visual_binding_v1(
            binding,posed_mechanical_positions_xyz=rest,camera=camera
        )
        row={
            "view_index":vi,
            "visible_logical_slot_count":len(set(chart_slot.values())),
            "chart_count":len(chart_slot),
            "visual_vertex_count":int(len(visual_mesh.positions)),
            "visual_face_count":int(len(visual_mesh.faces)),
            "source_foreground_pixel_count":int(mask.sum()),
            "max_unowned_fill_distance_px":float(max_fill),
            "max_binding_seed_distance_px":float(
                np.max(binding["nearest_safe_seed_distance_px"])
            ),
            "p95_binding_seed_distance_px":float(
                np.quantile(binding["nearest_safe_seed_distance_px"],.95)
            ),
            "max_affine_extrapolation_penalty":float(
                np.max(binding["extrapolation_penalty"])
            ),
            "max_rest_error_px":float(
                np.max(np.linalg.norm(rest_eval-np.asarray(visual_mesh.positions,float),axis=1))
            ),
            "qa_by_clip":{},
        }
        for clip_id in CLIP_IDS:
            clip=clips[clip_id]
            qa=[]
            for frame in clip.frames:
                by_id={
                    str(vid):tuple(map(float,xyz))
                    for vid,xyz in frame.posed_vertex_xyz
                }
                if set(by_id)!=set(vertex_ids):
                    raise RuntimeError(
                        "SLOT_COURT_STAGE41_VERTEX_ID_SET_DRIFT:"+clip_id
                    )
                posed=np.asarray([by_id[vid] for vid in vertex_ids],dtype=np.float64)
                visual=evaluate_region_visual_binding_v1(
                    binding,posed_mechanical_positions_xyz=posed,camera=camera
                )
                qa.append({"time_seconds":float(frame.time_seconds),**_visual_qa(
                    region.mesh.positions,visual,region.mesh.faces
                )})
            row["qa_by_clip"][clip_id]=qa
        report["views"].append(row)

    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(
        authority_root=Path(a.authority_root).resolve(),
        run_id=a.run_id,
        out_path=Path(a.out),
    )

if __name__=="__main__":
    main()
