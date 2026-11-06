#!/usr/bin/env python3
"""Original GLB linked-UV source atlas edit addresses, NOT drawable segmentation.

A deterministic source-UV tile workload. Uses original GLB binary mesh
accessors, external source PNG bytes and source-geometry face census. Neither
2D render outcomes nor artist object/group names are selection inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

SCHEMA="RealSaS.Corpus370OriginalSourceUVAtlasEditPlan.v1"
TILE_GRID=8
MIN_FACE_CENTROIDS_PER_TILE=16
OPERATION_COUNT=3
RULE="ATLAS_TILE_U8_RGB_INVERT__ALPHA_UNCHANGED"
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")
COMPONENT={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32}
DIMENSION={"SCALAR":1,"VEC2":2,"VEC3":3}

def require(ok,code):
    if not ok:raise ValueError("CAT_SOURCE_UV_EDIT_PLAN_"+code)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def glb_document(source):
    b=Path(source).read_bytes()
    require(len(b)>=20,"GLB_TOO_SMALL")
    magic,version,length=struct.unpack_from("<III",b,0)
    require(magic==0x46546c67 and version==2 and length==len(b),"GLB_CONTAINER_INVALID")
    json_len,kind=struct.unpack_from("<II",b,12)
    require(kind==0x4e4f534a and json_len>0 and 20+json_len+8<=len(b),
            "GLB_DOCUMENT_MISSING")
    doc=json.loads(b[20:20+json_len].decode())
    bin_len,bin_kind=struct.unpack_from("<II",b,20+json_len)
    require(bin_kind==0x004e4942 and
            len(b)==20+json_len+8+bin_len,
            "GLB_EXPECTED_EXACT_SINGLE_BIN")
    return doc,memoryview(b)[20+json_len+8:]

def accessor(doc,buffer,index):
    require(type(index) is int and 0<=index<len(doc["accessors"]),
            "SOURCE_ACCESSOR_INDEX")
    a=doc["accessors"][index]
    require("sparse" not in a and "bufferView" in a and
            a["componentType"] in COMPONENT and
            a["type"] in DIMENSION and
            not a.get("normalized",False),
            "SOURCE_ACCESSOR_REPRESENTATION")
    dt=np.dtype(COMPONENT[a["componentType"]]).newbyteorder("<")
    width=DIMENSION[a["type"]]
    count=int(a["count"])
    ref=doc["bufferViews"][a["bufferView"]]
    require(ref.get("buffer")==0 and count>0,
            "SOURCE_BUFFER_BACKING")
    stride=int(ref.get("byteStride",width*dt.itemsize))
    offset=int(ref.get("byteOffset",0))+int(a.get("byteOffset",0))
    require(stride>=width*dt.itemsize and offset>=0 and
            offset+(count-1)*stride+width*dt.itemsize<=len(buffer) and
            int(a.get("byteOffset",0))+(count-1)*stride+width*dt.itemsize<=ref["byteLength"],
            "SOURCE_ACCESSOR_BUFFER_BOUNDS")
    return np.ndarray((count,width),dtype=dt,buffer=buffer,offset=offset,
                      strides=(stride,dt.itemsize)).copy()

def uv_centroid_tile_census(doc,binary,grid=TILE_GRID):
    require(type(grid) is int and grid>=2,"UV_TILE_GRID")
    require(doc.get("asset",{}).get("version")=="2.0" and
            len(doc.get("meshes",[]))>0,"SOURCE_GLTF_MESH")
    counters=Counter()
    total=0
    for mesh in doc["meshes"]:
        require(len(mesh.get("primitives",[]))>0,"SOURCE_GLTF_EMPTY_PRIMITIVES")
        for prim in mesh["primitives"]:
            att=prim.get("attributes",{})
            require(prim.get("mode",4)==4 and "indices" in prim and
                    "POSITION" in att and "TEXCOORD_0" in att,
                    "SOURCE_LINKED_UV_TRIANGLE_REQUIRED")
            pos=accessor(doc,binary,att["POSITION"])
            uv=accessor(doc,binary,att["TEXCOORD_0"])
            ix=accessor(doc,binary,prim["indices"]).ravel()
            require(len(pos)==len(uv) and pos.shape[1]==3 and uv.shape[1]==2
                    and ix.size%3==0 and
                    np.issubdtype(ix.dtype,np.integer) and
                    int(ix.max())<len(pos) and
                    np.isfinite(pos).all() and np.isfinite(uv).all() and
                    np.min(uv)>=0 and np.max(uv)<=1,
                    "SOURCE_UV_TRIANGLE_DOMAIN")
            faces=ix.reshape(-1,3)
            uv_centroid=uv[faces].mean(axis=1)
            tiles=np.clip(np.floor(uv_centroid*grid).astype(np.int64),0,grid-1)
            for u,v in tiles:
                counters[(int(u),int(v))]+=1
            total+=len(faces)
    require(sum(counters.values())==total and total>0,
            "SOURCE_GLTF_FACE_CENSUS")
    return counters,total

def choose_uv_tiles(census, *,count=OPERATION_COUNT,minimum=MIN_FACE_CENTROIDS_PER_TILE):
    require(type(count) is int and count>0 and type(minimum) is int and minimum>0,
            "SOURCE_TILE_SELECTION_POLICY")
    options=sorted(((int(n),int(u),int(v)) for (u,v),n in census.items()
                    if type(n) is int and n>=minimum),key=lambda x:(-x[0],x[2],x[1]))
    require(len(options)>=count,"SOURCE_UV_TILES_INSUFFICIENT")
    return [{"uv_tile_u":u,"uv_tile_v":v,"source_triangle_uv_centroid_count":n}
            for n,u,v in options[:count]]

def external_image_path(source,doc):
    require(len(doc.get("materials",[]))==1 and
            len(doc.get("textures",[]))==1 and len(doc.get("images",[]))==1 and
            len(doc.get("samplers",[]))==1 and
            all(len(mesh["primitives"])==1 and mesh["primitives"][0].get("material")==0
                for mesh in doc["meshes"]) and
            doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]==0
            and doc["textures"][0]["source"]==0,
            "SINGLE_LINKED_SOURCE_ATLAS_MATERIAL_ONLY")
    uri=doc["images"][0].get("uri")
    require(isinstance(uri,str) and uri and not uri.startswith("data:"),
            "SOURCE_ATLAS_NOT_EXTERNAL")
    u=urlparse(uri)
    candidate=unquote(u.path)
    relative=PurePosixPath(candidate)
    require(not u.scheme and not u.netloc and not u.query and not u.fragment and
            not relative.is_absolute() and "\\" not in candidate and
            all(x not in ("","..",".") for x in candidate.split("/")),
            "SOURCE_TEXTURE_PATH_UNSAFE")
    src=Path(source).resolve()
    path=(src.parent/candidate).resolve()
    require(path.is_relative_to(src.parent) and path.is_file(),
            "SOURCE_TEXTURE_DEPENDENCY_NOT_PRESENT")
    return path,uri

def compile_plan(source,atlas,rest_geometry,camera,policy,*,subject_id,source_sha,atlas_sha):
    require(digest(source)==source_sha and digest(atlas)==atlas_sha,
            "ORIGINAL_SOURCE_IMAGE_SHA_DRIFT")
    doc,binary=glb_document(source)
    bound,relative=external_image_path(source,doc)
    require(bound.samefile(atlas),"SOURCE_IMAGE_NOT_IMPORTED_DEPENDENCY")
    census,triangle_count=uv_centroid_tile_census(doc,binary)
    with np.load(rest_geometry,allow_pickle=False) as npz:
        rest_faces=np.asarray(npz["faces"])
    require(len(rest_faces)==triangle_count and
            (rest_faces.ndim==2 and rest_faces.shape[1]==3),
            "SOURCE_REST_FACE_CENSUS_DRIFT")
    with Image.open(atlas) as image:
        require(image.mode=="RGBA" and image.size==(512,512),
                "SOURCE_RGBA_ATLAS_OR_RESOLUTION")
        rgba=np.asarray(image)
    require(rgba.shape==(512,512,4) and np.all(rgba[...,3]==255),
            "SOURCE_ALPHA_POLICY_UNSUPPORTED")
    source_policy=json.loads(Path(policy).read_text(encoding="utf-8"))
    cameras=json.loads(Path(camera).read_text(encoding="utf-8"))
    require(source_policy["source_sha256"]==source_sha and
            source_policy["material_mode"]=="unlit_preserve_alpha" and
            source_policy["camera_bundle"]["sha256"]==digest(camera) and
            len(cameras.get("cameras",[]))==8 and
            len(source_policy.get("material_transform",[]))==1 and
            source_policy["material_transform"][0]["source_color_linked"] is True and
            source_policy["material_transform"][0]["alpha_policy"]==
                "IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE",
            "ORIGINAL_SOURCE_SHADER_AND_CAMERA")
    chosen=choose_uv_tiles(census)
    for i,item in enumerate(chosen):
        u,v=item["uv_tile_u"],item["uv_tile_v"]
        item.update({
            "source_uv_edit_operation_index":i,
            "source_rgba_png_pixel_rectangle_xyxy_exclusive":[
                u*64,(7-v)*64,(u+1)*64,(8-v)*64],
            "atlas_decoded_rgba_pixel_addresses":64*64,
            "original_source_face_group_identity":"UNKNOWN",
            "source_artist_drawable_identity":"UNKNOWN",
        })
    return {
        "schema":SCHEMA,"status":"ORIGINAL_SOURCE_UV_OPERATION_PLAN_FROZEN__NOT_RENDER_COURTED",
        "subject_id":subject_id,"source_original_glb_sha256":source_sha,
        "source_original_texture_sha256":atlas_sha,
        "source_original_texture_relative_uri":relative,
        "source_original_rest_geometry_sha256":digest(rest_geometry),
        "source_original_camera_sha256":digest(camera),
        "source_original_policy_sha256":digest(policy),
        "source_glb_triangle_count":triangle_count,
        "source_glb_mesh_primitive_count":len(doc["meshes"]),
        "source_texture_size_px":[512,512],
        "source_texture_source_image_alpha_all_opaque":True,
        "uv_tile_grid_size":TILE_GRID,"uv_tile_texel_width_px":64,
        "tile_selection_rule":"ORIGINAL_GLTF_TRIANGLE_UV_CENTROID_COUNT_DESC__Y_X_ASC__MIN_16",
        "source_uv_tile_census":[{"u":u,"v":v,"triangle_centroid_count":n}
            for (u,v),n in sorted(census.items(),key=lambda x:(x[0][1],x[0][0]))],
        "selected_original_source_uv_edits":chosen,"original_source_uv_operation_count":len(chosen),
        "original_source_texture_edit_rule":RULE,
        "selection_used_original_rendered_2d_image":False,
        "operations_are_source_uv_texel_addresses_not_art_drawables":True,
        "original_artist_drawable_owner":"UNKNOWN",
        "model_training_target_count":0,
        **{k:False for k in FLAGS},
    }

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for k in ("source","atlas","rest-geometry","camera-bundle","source-policy","output"):
        a.add_argument("--"+k,type=Path,required=True)
    a.add_argument("--subject-id",required=True)
    a.add_argument("--expected-source-sha256",required=True)
    a.add_argument("--expected-atlas-sha256",required=True)
    args=a.parse_args()
    result=compile_plan(args.source,args.atlas,args.rest_geometry,args.camera_bundle,
                        args.source_policy,subject_id=args.subject_id,
                        source_sha=args.expected_source_sha256,
                        atlas_sha=args.expected_atlas_sha256)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],
        "original_uv_tile_operations":[(x["uv_tile_u"],x["uv_tile_v"]) for x in
            result["selected_original_source_uv_edits"]],
        "source_triangle_count":result["source_glb_triangle_count"],
        "training_authorized":False},sort_keys=True))

if __name__=="__main__":main()
