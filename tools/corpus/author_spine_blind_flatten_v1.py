#!/usr/bin/env python3
"""Blind original human-authored Spine export -> anonymous MERGED surface.

This is an independent ARTIST SOURCE ADAPTER, not artist-label reconstruction.
The court reference is sealed separately from model input, all author slot,
attachment, bone, object names and counts excluded from anonymous bytes.

Only source skeleton setup pose geometry is extracted. Atlas pixels, Spine IK,
transform constraints, clipping, alternate skins, animation and native alpha
compositing are not reproduced: composite parity/artist court remain ABSTAIN.

Spine authoring/export assets carry their own license; do not vendor or train
on third-party source without explicit appropriate rights. This adapter uses
source data as evaluation-only temporary bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA="RealSaS.BlindArtistOriginalSpineSetupFlatten.v1"
MERGED="RealSaS.AnonymousMergedSurfaceArtistBlindInput.v1"
HIDDEN="RealSaS.OriginalArtistDrawablesCourtOnly.v1"

def require(ok:bool,reason:str)->None:
    if not ok:
        raise ValueError(reason)

def digest(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()

def canonical_bytes(x:Any)->bytes:
    return (json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)+
            "\n").encode("utf-8")

def _setup_bone_matrices(bones:list[dict])->list[np.ndarray|None]:
    names={b.get("name"):i for i,b in enumerate(bones)}
    require(len(names)==len(bones) and None not in names,
            "ARTIST_SOURCE_DUPLICATE_OR_MISSING_BONES")
    stored={}
    resolving=set()
    def world(i):
        if i in stored:return stored[i]
        require(i not in resolving,"ARTIST_SOURCE_BONE_CYCLE")
        resolving.add(i)
        b=bones[i]
        parent=b.get("parent")
        require(parent is None or parent in names,
                "ARTIST_SOURCE_MISSING_PARENT_BONE")
        inherit=b.get("inherit","normal")
        if inherit!="normal":
            # Cannot silently treat noScale/noRotation as normal.
            matrix=None
        else:
            theta=math.radians(float(b.get("rotation",0.)))
            shx=math.radians(float(b.get("shearX",0.)))
            shy=math.radians(float(b.get("shearY",0.)))
            sx=float(b.get("scaleX",1.))
            sy=float(b.get("scaleY",1.))
            local=np.array([
                [math.cos(theta+shx)*sx, math.cos(theta+math.pi/2+shy)*sy,float(b.get("x",0.))],
                [math.sin(theta+shx)*sx, math.sin(theta+math.pi/2+shy)*sy,float(b.get("y",0.))],
                [0.,0.,1.]
            ],dtype=np.float64)
            matrix=local if parent is None else (
                None if (pm:=world(names[parent])) is None else pm@local
            )
        if matrix is not None:
            require(np.isfinite(matrix).all(),
                    "ARTIST_SOURCE_NONFINITE_BONE_TRANSFORM")
        stored[i]=matrix
        resolving.remove(i)
        return matrix
    return [world(i) for i in range(len(bones))]

def _as_xy(points:list[float],matrix:np.ndarray)->np.ndarray:
    x=np.asarray(points,dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==2 and np.isfinite(x).all(),
            "ARTIST_SOURCE_INVALID_LOCAL_COORDINATES")
    return (np.column_stack((x,np.ones(len(x))))@matrix.T)[:,:2]

def _unweighted_mesh(part,world_bone):
    n=len(part["uvs"])//2
    require(len(part["uvs"])==2*n and n>=3 and
            len(part["vertices"])==2*n,
            "ARTIST_SOURCE_MESH_UNWEIGHTED_COORDINATE_CONTRACT")
    require(world_bone is not None,"ARTIST_SOURCE_UNSUPPORTED_BONE_INHERITANCE")
    local=np.asarray(part["vertices"],dtype=np.float64).reshape(-1,2)
    return _as_xy(local,world_bone),"MESH_UNWEIGHTED"

def _weighted_mesh(part,world_bones):
    uvs=part["uvs"]
    n=len(uvs)//2
    require(n>=3 and len(uvs)==2*n,"ARTIST_SOURCE_MESH_UV_DOMAIN")
    a=part.get("vertices")
    require(isinstance(a,list) and len(a)>2*n,
            "ARTIST_SOURCE_MESH_WEIGHTED_ENCODING_EXPECTED")
    p=0
    result=[]
    for _ in range(n):
        require(p<len(a),"ARTIST_SOURCE_WEIGHTED_VERTEX_EOF")
        count=int(a[p])
        require(float(a[p])==count and 1<=count<=256,
                "ARTIST_SOURCE_MESH_BONE_COUNT_ENCODING")
        p+=1
        pos=np.zeros(2,dtype=np.float64)
        weights=0.
        for _ in range(count):
            require(p+4<=len(a),"ARTIST_SOURCE_WEIGHTED_INFLUENCE_EOF")
            k=int(a[p]);x=float(a[p+1]);y=float(a[p+2]);w=float(a[p+3])
            require(float(a[p])==k and 0<=k<len(world_bones)
                    and np.isfinite([x,y,w]).all() and w>=0,
                    "ARTIST_SOURCE_WEIGHTED_BONE_ID_OR_WEIGHT")
            require(world_bones[k] is not None,
                    "ARTIST_SOURCE_UNSUPPORTED_WEIGHTED_BONE_INHERITANCE")
            pos+=_as_xy([[x,y]],world_bones[k])[0]*w
            weights+=w
            p+=4
        require(abs(weights-1.)<=1e-3,
                "ARTIST_SOURCE_WEIGHTED_INFLUENCE_NOT_NORMALIZED")
        result.append(pos)
    require(p==len(a),"ARTIST_SOURCE_TRAILING_MESH_VERTEX_BYTES")
    return np.asarray(result,dtype=np.float64),"MESH_WEIGHTED"

def _geometry(part,owner_bone,world_bones):
    kind=part.get("type","region")
    if kind=="region":
        mat=world_bones[owner_bone]
        require(mat is not None,"ARTIST_SOURCE_UNSUPPORTED_REGION_BONE_INHERITANCE")
        width=float(part["width"]);height=float(part["height"])
        sx=float(part.get("scaleX",1.));sy=float(part.get("scaleY",1.))
        theta=math.radians(float(part.get("rotation",0.)))
        require(np.isfinite([width,height,sx,sy,theta]).all()
                and width>0 and height>0,"ARTIST_SOURCE_BAD_REGION_DIMENSIONS")
        base=np.array([[-width/2,-height/2],[-width/2,height/2],
                       [width/2,height/2],[width/2,-height/2]],dtype=np.float64)
        base[:,0]*=sx;base[:,1]*=sy
        rotate=np.array([[math.cos(theta),-math.sin(theta)],
                         [math.sin(theta),math.cos(theta)]])
        base=base@rotate.T+np.array([float(part.get("x",0.)),
                                    float(part.get("y",0.))])
        vertices=_as_xy(base,mat)
        triangles=np.array([[0,1,2],[2,3,0]],np.int64)
        # Local UV, not the actual Spine atlas page transform.
        uv=np.array([[0,0],[0,1],[1,1],[1,0]],dtype=np.float64)
        method="REGION_QUAD_NO_ATLAS_CAA"
    elif kind=="mesh":
        uvs=part.get("uvs");tri=part.get("triangles")
        require(isinstance(uvs,list) and len(uvs)>=6 and len(uvs)%2==0
                and isinstance(tri,list) and len(tri)>=3 and len(tri)%3==0,
                "ARTIST_SOURCE_MESH_UV_TRIANGLE_ENCODING")
        if len(part.get("vertices",[]))==len(uvs):
            vertices,method=_unweighted_mesh(part,world_bones[owner_bone])
        else:
            vertices,method=_weighted_mesh(part,world_bones)
        uv=np.asarray(uvs,np.float64).reshape(-1,2)
        triangles=np.asarray(tri,np.int64).reshape(-1,3)
    else:
        raise ValueError("ARTIST_SOURCE_UNSUPPORTED_VISUAL_ATTACHMENT_"+str(kind))
    require(np.isfinite(vertices).all() and np.isfinite(uv).all()
            and np.isfinite(triangles).all() and
            (len(triangles)>0) and triangles.min()>=0
            and triangles.max()<len(vertices),
            "ARTIST_SOURCE_INVALID_TRIANGULATED_ATTACHMENT")
    return vertices,triangles,uv,method

def flatten(source:dict,*,source_bytes:bytes,seed:int=370)->tuple[dict,dict,dict]:
    """Returns (anonymous input, hidden artist reference, transparent audit).

    Caller MUST place these in separately permissioned directories/runtimes.
    Returning multiple values here is NOT cryptographic isolation by itself.
    """
    require(isinstance(source,dict) and isinstance(source.get("slots"),list)
            and isinstance(source.get("bones"),list)
            and isinstance(source.get("skins"),list)
            and len(source["skins"])>=1
            and type(seed) is int,
            "ARTIST_SOURCE_ORIGINAL_SPINE_JSON_FORMAT")
    skin=next((s for s in source["skins"] if s.get("name")=="default"),None)
    require(skin is not None and isinstance(skin.get("attachments"),dict),
            "ARTIST_SOURCE_DEFAULT_SKIN_REQUIRED")
    slots=source["slots"];bones=source["bones"]
    names={b["name"]:i for i,b in enumerate(bones)}
    require(len(names)==len(bones),"ARTIST_SOURCE_BONE_NAME_UNIQUENESS")
    matrices=_setup_bone_matrices(bones)
    vertex_arrays=[];face_arrays=[];uv_arrays=[];artist_group=[];private_group=[]
    unsupported=[]
    active_count=0
    vbase=0
    for draw_position,slot in enumerate(slots):
        if not slot.get("attachment"):continue
        active_count+=1
        selected=slot["attachment"]
        group=skin["attachments"].get(slot["name"],{})
        require(selected in group,
                "ARTIST_SOURCE_SELECTED_ATTACHMENT_NOT_IN_DEFAULT_SKIN")
        part=group[selected]
        kind=part.get("type","region")
        if kind not in ("mesh","region"):
            unsupported.append({
                "slot_draw_order_teacher_only":draw_position,
                "reason":"UNSUPPORTED_VISIBLE_ARTIST_ATTACHMENT_TYPE",
                "attachment_kind_teacher_only":kind
            })
            continue
        try:
            require(slot["bone"] in names,
                    "ARTIST_SOURCE_SLOT_PARENT_BONE_NOT_FOUND")
            vertices,tri,uv,method=_geometry(part,names[slot["bone"]],matrices)
        except (ValueError,KeyError,TypeError) as e:
            unsupported.append({
                "slot_draw_order_teacher_only":draw_position,
                "reason":str(e),
                "attachment_kind_teacher_only":kind
            })
            continue
        vertex_arrays.append(vertices)
        face_arrays.append(tri+vbase)
        uv_arrays.append(uv)
        artist_group.extend([draw_position]*len(tri))
        private_group.append({
            "slot_name_teacher_only":slot["name"],
            "attachment_name_teacher_only":selected,
            "slot_draw_order_teacher_only":draw_position,
            "rig_bone_name_teacher_only":slot["bone"],
            "attachment_kind_teacher_only":kind,
            "method":method,
            "triangle_count":len(tri)
        })
        vbase+=len(vertices)
    require(vertex_arrays and face_arrays,"ARTIST_SOURCE_NO_SUPPORTED_VISUAL_SETUP_ATTACHMENTS")
    vertices=np.vstack(vertex_arrays)
    triangles=np.vstack(face_arrays)
    uv=np.vstack(uv_arrays)
    labels=np.asarray(artist_group,np.int64)
    require(len(labels)==len(triangles),"ARTIST_SOURCE_FACE_COURT_DOMAIN")
    # Shuffle source faces and vertices, not author group counts/boundaries.
    # This erases sequential slot ordering; disconnected geometric components
    # STILL disclose a likely author's split and must be declared as leakage.
    rng=np.random.default_rng(seed)
    vp=rng.permutation(len(vertices))
    ip=np.empty(len(vertices),dtype=np.int64)
    ip[vp]=np.arange(len(vertices),dtype=np.int64)
    fp=rng.permutation(len(triangles))
    anonymous_vertices=vertices[vp].tolist()
    anonymous_uv=uv[vp].tolist()
    anonymous_triangles=ip[triangles[fp]].tolist()
    hidden_labels=labels[fp].tolist()
    source_sha=digest(source_bytes)
    merged={
        "schema":MERGED,
        "input_kind":"ORIGINAL_HUMAN_AUTHORED_2D_SPINE_SETUP_ANONYMOUS_MERGED_TRIANGLE_PREVIEW",
        "vertices_setup_xy":anonymous_vertices,
        "faces":anonymous_triangles,
        "source_local_uv_without_atlas":anonymous_uv,
        "image_texture_and_alpha_unavailable":True,
        "original_art_layer_count_visible":False,
        "original_slot_attachment_bone_names_present":False,
        "original_layer_order_present":False,
        "original_source_animations_present":False,
        "source_author_face_permutation_randomized":True,
        "disconnected_topology_can_leak_original_author_partition":True,
        "native_spine_ik_constraints_solved":False,
        "source_composite_parity_qualified":False,
        "training_authorized":False
    }
    merged_bytes=canonical_bytes(merged)
    common_address_sha=digest(canonical_bytes(list(range(len(anonymous_triangles)))))
    hidden={
        "schema":HIDDEN,
        "human_authored_source_export_sha256":source_sha,
        "source_original_git_blob_sha_expected_only_from_external_verification":None,
        "native_artist_editable_project_file_verified":False,
        "original_atlas_and_alpha_artwork_reconstructed":False,
        "anonymous_merged_surface_sha256":digest(merged_bytes),
        "common_face_address_sha256":common_address_sha,
        "artist_slot_id_teacher_only_per_face":hidden_labels,
        "artist_slot_draw_order_teacher_only":private_group,
        "original_slot_count_teacher_only":len(slots),
        "original_setup_active_attachment_count_teacher_only":active_count,
        "unsupported_original_attachment_cases_teacher_only":unsupported,
        "all_original_artist_active_attachments_supported":len(unsupported)==0,
        "artist_true_2d_drawable_source_metadata":True,
        "original_spine_setup_pose_native_runtime_parity_qualified":False,
        "source_appearance_rgba_alpha_parity_qualified":False,
        "artist_same_drawable_reconstruction_qualified":False,
        "training_authorized":False
    }
    receipt={
        "schema":SCHEMA,
        "status":"REAL_SPINE_SOURCE_SETUP_GEOMETRY_FLATTENED__ARTIST_CAA_AND_RUNTIME_UNQUALIFIED",
        "original_source_SHA256":source_sha,
        "anonymous_merged_surface_sha256":digest(merged_bytes),
        "hidden_reference_SHA256":digest(canonical_bytes(hidden)),
        "common_face_address_sha256":common_address_sha,
        "source_vertex_count":len(vertices),
        "source_triangle_count":len(triangles),
        "original_source_bone_count_teacher_only":len(bones),
        "original_source_slot_count_teacher_only":len(slots),
        "setup_active_attachment_count_teacher_only":active_count,
        "setup_supported_attachment_count_teacher_only":len(private_group),
        "unsupported_original_attachment_count":len(unsupported),
        "original_human_authored_spine_export_slot_structure_measured":True,
        "editable_native_spine_artist_project_bytes_verified":False,
        "original_atlas_image_and_author_alpha_CAA_verified":False,
        "original_artist_layers_authority_for_2d_source_only":False,
        "anonymous_input_author_names_hidden":True,
        "native_composite_appearance_CAA_qualified":False,
        "native_spine_animation_runtime_qualified":False,
        "model_prediction_evaluated_against_artist_labels":False,
        "true_artist_layer_benchmark_PASS":False,
        "training_authorized":False
    }
    return merged,hidden,receipt

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-source",type=Path,required=True)
    parser.add_argument("--anonymous-input",type=Path,required=True)
    parser.add_argument("--hidden-court-reference",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=370)
    args=parser.parse_args()
    require(args.anonymous_input.resolve()!=args.hidden_court_reference.resolve(),
            "ARTIST_SOURCE_PRIVATE_REFERENCE_PATH_EQUALS_ALGORITHM_INPUT")
    raw=args.original_source.read_bytes()
    original=json.loads(raw)
    merged,hidden,receipt=flatten(original,source_bytes=raw,seed=args.seed)
    for target,data in ((args.anonymous_input,merged),
                        (args.hidden_court_reference,hidden),
                        (args.receipt,receipt)):
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(canonical_bytes(data))
    print(json.dumps(receipt,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
