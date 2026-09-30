from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
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
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)
from tools.demo.render_knight_visual_mesh_arap_witness_v1 import (
    _candidate_face_indices,
)

CLIPS=(
    ("demo_idle_v1","IDLE"),
    ("demo_run_v1","RUN"),
    ("demo_slash_v1","SLASH"),
)


def _read_json(path: Path):
    if not path.is_file():
        raise RuntimeError(f"SLOT_COURT_ARTIFACT_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    ctx=_ctx(authority_root,run_id)
    candidate=canonical_mesh_candidate_from_dict(_read_json(
        root/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json"
    ))
    structure=presentation_structure_v2_from_dict(_read_json(
        root/"artifacts/37_QUALIFIED_PRESENTATION_STRUCTURE/qualified_presentation_structure_v2.json"
    ))
    cameras=qualified_camera_set_from_dict(_read_json(
        root/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json"
    ))
    observation=qualified_observation_set_from_dict(_read_json(
        root/"artifacts/07_OBSERVATION_CONTRACT_QUALIFIED/qualified_observation_set.json"
    ))
    skeleton=qualified_skeleton_from_dict(_read_json(
        root/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"
    ))
    skin=qualified_skin_from_dict(_read_json(
        root/"artifacts/32_SKIN_QUALIFIED/qualified_skin.json"
    ))

    ordered=tuple(sorted(cameras.cameras,key=lambda c:int(c.view_index)))
    rgba_by_view,mask_by_view=_load_source_inputs(ctx,observation)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    faces=_candidate_face_indices(candidate)
    face_slot,slot_by_index=_slot_face_labels(structure,len(faces))
    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)

    source_report=_read_json(Path(
        "canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"
    ))
    clips={}
    for clip_id,short in CLIPS:
        payload=_read_json(
            root/"inputs/motion/quaternius_knight_v1"/f"{clip_id}.motion.json"
        )
        tracks,mapping=_tracks_for_clip(payload,skeleton,ordered,source_report)
        clips[clip_id]=(payload,tracks,mapping)

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
            candidate,camera,positions=rest,
            width=mask.shape[1],height=mask.shape[0],max_layers=4,
        )
        labels,seeds,chart_slot,max_fill=_partition_source_by_slot_adjacency(
            mask,vis.owner_face_index,faces,face_slot
        )
        region=build_visual_mesh_from_region_labels_v1(
            mask,labels,target_edge_px=16
        )
        mesh=region.mesh
        binding=bind_region_visual_vertices_to_mechanical_affine_v1(
            points_source_xy=mesh.positions,
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
            "visual_vertex_count":int(len(mesh.positions)),
            "visual_face_count":int(len(mesh.faces)),
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
                np.max(np.linalg.norm(rest_eval-np.asarray(mesh.positions,float),axis=1))
            ),
            "qa_by_clip":{},
        }
        for clip_id,short in CLIPS:
            payload,tracks,_mapping=clips[clip_id]
            times=np.linspace(
                0.0,float(payload["duration_seconds"]),4,
                endpoint=not bool(payload.get("loop")),dtype=np.float64
            )
            qa=[]
            for t in times:
                mats,_,_=_joint_pose_v2(
                    skeleton=skeleton,tracks=tracks,
                    time_seconds=float(t),cameras=ordered
                )
                posed=_skin(rest,W,joint_ids,mats)
                visual=evaluate_region_visual_binding_v1(
                    binding,posed_mechanical_positions_xyz=posed,camera=camera
                )
                qa.append({"time_seconds":float(t),**_visual_qa(
                    mesh.positions,visual,mesh.faces
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
