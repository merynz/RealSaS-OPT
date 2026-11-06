#!/usr/bin/env python3
"""Independent original-source constant palette court; NO producer import.

Re-imports ORIGINAL GLB/FBX, independently enumerates triangles and traces
the active material output's Surface connection. Verifies every source face,
shader Base Color and UNKNOWN. No native render/edit consequence is claimed.
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

SCHEMA="RealSaS.Corpus370IndependentOriginalShaderPaletteCourt.v1"
POLICY="EXACT_FLOAT_CONSTANT_SOURCE_BASE_COLOR__OPAQUE_REPLAY__UNKNOWN_LINKED"
NONPROMOTE=("graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","materialization_authorized","training_authorized")

def check(cond,label):
    if not cond:raise ValueError("ORIGINAL_PALETTE_COURT_"+label)

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def rgba_of_material(mat,policy):
    if mat is None:return None
    link_row=policy.get(mat.name)
    if link_row is None:return None
    if link_row.get("operation")!="BSDF_LIGHT_RESPONSE_TO_EMISSION__ALPHA_GRAPH_RETAINED" or \
       link_row.get("alpha_policy")!="IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE":
        return None
    if not mat.use_nodes or mat.node_tree is None:return None
    nodes=mat.node_tree.nodes
    outputs=[x for x in nodes if x.type=="OUTPUT_MATERIAL" and x.is_active_output]
    shaders=[x for x in nodes if x.type=="BSDF_PRINCIPLED"]
    if len(outputs)!=1 or len(shaders)!=1:return None
    surface=outputs[0].inputs.get("Surface")
    base=shaders[0].inputs.get("Base Color")
    alpha=shaders[0].inputs.get("Alpha")
    if surface is None or base is None or alpha is None:return None
    if bool(base.is_linked)!=link_row.get("source_color_linked") or \
       bool(alpha.is_linked)!=link_row.get("source_alpha_linked"):
        raise ValueError("ORIGINAL_PALETTE_COURT_SOURCE_POLICY_DRIFT")
    if bool(base.is_linked) or bool(alpha.is_linked):return None
    if len(surface.links)!=1 or surface.links[0].from_node!=shaders[0]:
        return None
    color=[float(k) for k in base.default_value]
    if len(color)!=4 or not all(math.isfinite(z) for z in color):return None
    return tuple(x.hex() for x in color)

def independent_court(source,source_sha,reference_rest_npz,v18a_npz,
                      source_policy,facts_path,output_dir):
    facts=json.loads(Path(facts_path).read_text(encoding="utf-8"))
    policy_record=json.loads(Path(source_policy).read_text(encoding="utf-8"))
    check(sha(source)==source_sha=="{}".format(facts.get("original_source_sha256")), "SOURCE_IDENTITY")
    check(policy_record.get("source_sha256")==source_sha, "SOURCE_RENDER_POLICY_IDENTITY")
    check(facts.get("schema")=="RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1"
          and facts.get("policy")==POLICY,"FACT_SCHEMA")
    check(sha(v18a_npz)==facts.get("original_v18a_npz_sha256")
          and sha(reference_rest_npz)==facts.get("original_rest_npz_sha256")
          and sha(source_policy)==facts.get("original_material_policy_sha256"),
          "PARENT_SHA")
    with np.load(v18a_npz,allow_pickle=False) as v:
        face_array=np.asarray(v["faces"],np.int64)
    with np.load(reference_rest_npz,allow_pickle=False) as v:
        check(np.array_equal(face_array,v["faces"]),"REST_V18A_FACE")
        rest=np.asarray(v["rest_vertices_source_world"],np.float64)
    check(hashlib.sha256(np.ascontiguousarray(face_array,dtype="<i8").tobytes()).hexdigest()
          ==facts.get("source_face_order_sha256"),"FACE_ORDER_SHA")
    frozen={}
    for row in policy_record["material_transform"]:
        key=row["material"]
        check(key not in frozen,"DUPLICATE_SOURCE_MATERIAL_RECEIPT")
        frozen[key]=row

    import_rest(Path(source),Path(output_dir)/"court_source_import")
    observed_faces=[]
    observed_xyz=[]
    observed_colors=[]
    deps=bpy.context.evaluated_depsgraph_get()
    for obj in sorted(bpy.context.scene.objects,key=lambda r:r.name):
        if obj.type!="MESH" or obj.hide_render:continue
        eval_obj=obj.evaluated_get(deps)
        mesh=eval_obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            off=len(observed_xyz)
            for v in mesh.vertices:
                observed_xyz.append(tuple(float(z) for z in (eval_obj.matrix_world @ v.co)))
            for tri in mesh.loop_triangles:
                observed_faces.append([off+int(j) for j in tri.vertices])
                idx=int(tri.material_index)
                mat=obj.material_slots[idx].material if 0<=idx<len(obj.material_slots) else None
                observed_colors.append(rgba_of_material(mat,frozen))
        finally:
            eval_obj.to_mesh_clear()
    check(np.array_equal(np.asarray(observed_faces,dtype=np.int64),face_array),
          "INDEPENDENT_TRIANGLE_ORDER")
    check(np.asarray(observed_xyz).shape==rest.shape and
          np.allclose(observed_xyz,rest,atol=2e-6,rtol=0),
          "INDEPENDENT_REST_WORLD_GEOMETRY")
    expect_colors=sorted(set(x for x in observed_colors if x is not None))
    check(facts.get("constant_palette_color_keys_hex_rgba")==[list(c) for c in expect_colors],
          "INDEPENDENT_EXACT_RGBA_PALETTE")
    known=[x is not None for x in observed_colors]
    indices=[expect_colors.index(c) if c is not None else None for c in observed_colors]
    check(facts.get("face_palette_validity")==known and
          facts.get("face_palette_color_key_index_or_null")==indices,
          "INDEPENDENT_PER_FACE_SHADER_ADDRESSING")
    check(facts.get("source_face_count")==len(observed_faces)
          and facts.get("qualified_constant_face_count")==sum(known)
          and facts.get("unsupported_face_count")==len(known)-sum(known)
          and facts.get("palette_operation_count")==len(expect_colors),
          "SOURCE_SHADER_CENSUS")
    check(all(facts.get(flag) is False for flag in NONPROMOTE),
          "AUTHORITY_NONPROMOTION")
    check(facts.get("original_art_drawable_owner")=="UNKNOWN"
          and facts.get("source_rendered_tint_consequence")=="NOT_EVALUATED",
          "RENDER_NOT_MISCLAIMED")

    result={"schema":SCHEMA,
            "status":"INDEPENDENT_SOURCE_SHADER_ADDRESS_VERIFIED__NO_OWNER_AUTHORITY",
            "original_source_sha256":source_sha,
            "palette_producer_sha256":sha(facts_path),
            "qualified_constant_face_count":sum(known),
            "unsupported_face_count":len(known)-sum(known),
            "palette_color_count":len(expect_colors),
            "original_face_count":len(observed_faces),
            "renderer_physical_edit_verified":False,
            "graphics_owner_authorized":False,
            "iris_training_authorized":False,
            "materialization_authorized":False,
            "training_authorized":False}
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    (Path(output_dir)/"PRIVATE_INDEPENDENT_ORIGINAL_SHADER_PALETTE_COURT_V1.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"faces":len(known),
        "constant_palette_faces":sum(known),
        "unsupported_faces":len(known)-sum(known),
        "palette_colors":len(expect_colors),"training_authorized":False}),flush=True)
    return result

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    for k in ("source","reference-rest-npz","v18a-npz","source-policy","facts","output-dir"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--source-sha256",required=True)
    a=p.parse_args(raw)
    independent_court(a.source,a.source_sha256,a.reference_rest_npz,a.v18a_npz,
                      a.source_policy,a.facts,a.output_dir)

if __name__=="__main__":main()
