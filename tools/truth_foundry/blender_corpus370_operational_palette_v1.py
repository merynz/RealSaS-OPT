#!/usr/bin/env python3
"""Source-private constant shader palette facts from ORIGINAL Blender geometry.

No semantic material name, object ID, image segmentation, fixed K or actor
control ID is ever used as a palette grouping key. Linked/unsupported shader
faces have UNKNOWN palette membership, not a negative label. Producer only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.observation.blender_controlled_8view_v2 import import_rest

SCHEMA="RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1"
POLICY="EXACT_FLOAT_CONSTANT_SOURCE_BASE_COLOR__OPAQUE_REPLAY__UNKNOWN_LINKED"
DENIED=("graphics_owner_authorized","iris_training_authorized",
        "tessa_training_authorized","materialization_authorized","training_authorized")

def require(c,code):
    if not c:raise ValueError(code)

def digest_file(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def hash_faces(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype="<i8").tobytes()).hexdigest()

def canonical(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),
            ensure_ascii=False,allow_nan=False)+"\n").encode()

def classify(material,policy):
    if material is None:return None,"NO_MATERIAL"
    row=policy.get(material.name)
    if row is None:return None,"SOURCE_POLICY_NOT_BOUND"
    if row.get("operation")!="BSDF_LIGHT_RESPONSE_TO_EMISSION__ALPHA_GRAPH_RETAINED":
        return None,"SOURCE_POLICY_NOT_PRINCIPLED_REPLAY"
    if row.get("alpha_policy")!="IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE":
        return None,"NONOPAQUE_POLICY_UNSUPPORTED"
    if not material.use_nodes or material.node_tree is None:
        return None,"NODE_GRAPH_MISSING"
    tree=material.node_tree
    bsdfs=[n for n in tree.nodes if n.type=="BSDF_PRINCIPLED"]
    outputs=[n for n in tree.nodes if n.type=="OUTPUT_MATERIAL" and n.is_active_output]
    if len(bsdfs)!=1 or len(outputs)!=1:
        return None,"NONUNIQUE_PRINCIPLED_OR_OUTPUT"
    base=bsdfs[0].inputs.get("Base Color")
    alpha=bsdfs[0].inputs.get("Alpha")
    surface=outputs[0].inputs.get("Surface")
    if base is None or alpha is None or surface is None:
        return None,"SOURCE_SOCKET_MISSING"
    if bool(base.is_linked)!=row.get("source_color_linked") or bool(alpha.is_linked)!=row.get("source_alpha_linked"):
        raise ValueError("ORIGINAL_SHADER_POLICY_LINK_DRIFT")
    if bool(base.is_linked):return None,"LINKED_BASE_COLOR_UNSUPPORTED"
    if bool(alpha.is_linked):return None,"LINKED_ALPHA_UNSUPPORTED"
    if len(surface.links)!=1 or surface.links[0].from_node!=bsdfs[0]:
        return None,"COMPLEX_SURFACE_SHADER_UNSUPPORTED"
    rgba=tuple(float(v).hex() for v in base.default_value)
    require(len(rgba)==4 and all(math.isfinite(float.fromhex(v)) for v in rgba),
            "SOURCE_CONSTANT_RGBA_NONFINITE")
    return rgba,None

def measure(source,expected_sha,v18a_npz,rest_npz,source_policy,output_dir):
    require(digest_file(source)==expected_sha,"ORIGINAL_SHADER_SOURCE_SHA_DRIFT")
    expected=json.loads(Path(source_policy).read_text(encoding="utf-8"))
    require(expected.get("source_sha256")==expected_sha,"ORIGINAL_SHADER_SOURCE_POLICY_SHA")
    transforms=expected.get("material_transform")
    require(isinstance(transforms,list) and transforms,"ORIGINAL_SHADER_SOURCE_POLICY_MISSING")
    policy={}
    for row in transforms:
        name=row.get("material")
        require(isinstance(name,str) and name not in policy,
                "ORIGINAL_SHADER_SOURCE_POLICY_DUPLICATE")
        policy[name]=row
    with np.load(v18a_npz,allow_pickle=False) as f:
        faces_expected=np.asarray(f["faces"],dtype=np.int64)
    with np.load(rest_npz,allow_pickle=False) as f:
        faces_rest=np.asarray(f["faces"],dtype=np.int64)
        vertices_rest=np.asarray(f["rest_vertices_source_world"],dtype=np.float64)
    require(np.array_equal(faces_expected,faces_rest),
            "ORIGINAL_SHADER_V18A_REST_FACE_ORDER_MISMATCH")
    import_rest(Path(source),Path(output_dir)/"source_import")
    deps=bpy.context.evaluated_depsgraph_get()
    world_vertices=[]
    triangle_indices=[]
    keys=[]
    unsupported={}
    material_count=0
    for source_obj in sorted(bpy.context.scene.objects,key=lambda o:o.name):
        if source_obj.type!="MESH" or source_obj.hide_render:continue
        evaluated=source_obj.evaluated_get(deps)
        mesh=evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset=len(world_vertices)
            world_vertices.extend(tuple(float(v) for v in (evaluated.matrix_world @ vert.co))
                                  for vert in mesh.vertices)
            for tri in mesh.loop_triangles:
                triangle_indices.append([offset+int(v) for v in tri.vertices])
                slot_index=int(tri.material_index)
                material=source_obj.material_slots[slot_index].material if (
                    0<=slot_index<len(source_obj.material_slots)) else None
                color,error=classify(material,policy)
                if error is not None:
                    unsupported[error]=unsupported.get(error,0)+1
                else:
                    material_count+=1
                keys.append(color)
        finally:
            evaluated.to_mesh_clear()
    faces_observed=np.asarray(triangle_indices,dtype=np.int64)
    vertices_observed=np.asarray(world_vertices,dtype=np.float64)
    require(np.array_equal(faces_observed,faces_expected),
            "ORIGINAL_SHADER_IMPORT_FACE_INDEX_DRIFT")
    require(vertices_observed.shape==vertices_rest.shape
        and np.allclose(vertices_observed,vertices_rest,atol=2e-6,rtol=0),
        "ORIGINAL_SHADER_IMPORT_WORLD_VERTEX_DRIFT")
    unique=sorted(set(k for k in keys if k is not None))
    known=sum(k is not None for k in keys)
    result={"schema":SCHEMA,"policy":POLICY,
        "original_source_sha256":expected_sha,
        "original_v18a_npz_sha256":digest_file(v18a_npz),
        "original_rest_npz_sha256":digest_file(rest_npz),
        "original_material_policy_sha256":digest_file(source_policy),
        "source_face_order_sha256":hash_faces(faces_expected),
        "source_face_count":len(keys),
        "qualified_constant_face_count":known,
        "unsupported_face_count":len(keys)-known,
        "unsupported_reason_counts":unsupported,
        "constant_palette_color_keys_hex_rgba":unique,
        "face_palette_color_key_index_or_null":[unique.index(k) if k is not None else None for k in keys],
        "face_palette_validity":[k is not None for k in keys],
        "palette_operation_count":len(unique),
        "observed_shader_source_policy":"UNLIT_SOURCE_REPLAY__NOT_RAW_LIT_RENDER_PARITY",
        "source_rendered_tint_consequence":"NOT_EVALUATED",
        "source_texture_uv_graphics_relation":"UNKNOWN_IF_LINKED",
        "original_art_drawable_owner":"UNKNOWN",
        **{k:False for k in DENIED}}
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    dest=Path(output_dir)/"PRIVATE_ORIGINAL_SHADER_CONSTANT_PALETTE_FACTS_V1.json"
    dest.write_bytes(canonical(result))
    print(json.dumps({"status":"SOURCE_CONSTANT_PALETTE_FACTS__OWNER_UNKNOWN",
        "source_face_count":len(keys),
        "qualified_faces":known,"unsupported_faces":len(keys)-known,
        "palette_colors":len(unique),"reason_counts":unsupported,
        "training_authorized":False},sort_keys=True),flush=True)
    return result

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-sha256",required=True)
    p.add_argument("--v18a-npz",type=Path,required=True)
    p.add_argument("--rest-npz",type=Path,required=True)
    p.add_argument("--source-policy",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(raw)
    measure(a.source,a.source_sha256,a.v18a_npz,a.rest_npz,a.source_policy,a.output_dir)

if __name__=="__main__":main()
