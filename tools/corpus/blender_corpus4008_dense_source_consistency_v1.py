#!/usr/bin/env python3
"""Full imported-vertex/rig-rest and sparse-influence normalization census.

All source-native geometry coordinates, source bone rest matrices and sparse
group weights are inspected in one isolated Blender import. The reversible
transforms are an in-memory mathematical study; they are NOT materialization
or a source-original rig/skin/semantic authority. No pose is manufactured.

The imported scene may be transformed by FBX/glTF importers. Original
authored front/up/meters, IK scripts, weights that were stripped by export,
texture appearance, deformation and attachments remain independent courts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_source_bounds_v1 import measure
from tools.corpus.build_corpus4008_normalization_plan_v1 import (
    matrix,point,four_by_four,require_identity
)
from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    sha,require,finite_matrix,
)

SCHEMA="RealSaS.Corpus4008ImportedDenseNormalizationConsistency.v1"


def similarity(bounds):
    lo=bounds["source_world_min"]
    hi=bounds["source_world_max"]
    center=[(a+b)*.5 for a,b in zip(lo,hi)]
    extent=max(b-a for a,b in zip(lo,hi))
    require(math.isfinite(extent) and extent>1e-12,"DENSE_DEGENERATE_EXTENT")
    half=extent/2.
    forward=matrix(1./half,[-x/half for x in center])
    back=matrix(half,center)
    require_identity(four_by_four(forward,back))
    require_identity(four_by_four(back,forward))
    return forward,back,extent


def matrix_parity(forward,back,original):
    normalized=four_by_four(forward,original)
    recovered=four_by_four(back,normalized)
    return max(abs(float(a)-float(b)) for a,b in zip(recovered,original))


def weight_record(groups):
    """Stable source-vertex sparse influence digest; no reweight or guessing."""
    ordered=sorted((int(g),float(w)) for g,w in groups)
    require(all(g>=0 and math.isfinite(w) and w>=0 for g,w in ordered),
            "INVALID_IMPORTED_SOURCE_INFLUENCE")
    return ordered


def digest_weight_row(h,mesh_id,vertex_id,values):
    h.update(struct.pack("<III",mesh_id,vertex_id,len(values)))
    for group,weight in values:
        h.update(struct.pack("<Id",group,weight))


def imported_census(bpy,bounds):
    from mathutils import Matrix
    forward,back,extent=similarity(bounds)
    objects=sorted(bpy.context.scene.objects,key=lambda o:o.name)
    original_vertices=0
    original_bones=0
    original_influences=0
    meshes_with_armature=0
    mesh_skin_candidate_vertices=0
    mesh_unweighted_vertices=0
    bad_weight_sum_vertices=0
    nondeform_group_assignments=0
    parent_bone_attachments=0
    max_vertex_error=0.
    max_bone_world_rest_error=0.
    max_object_matrix_error=0.
    max_weight_sum_error=0.
    negative_handed_world_objects=0
    mesh_count=0
    armature_count=0
    failed_domain=0
    max_iris_abs=0.
    h=hashlib.sha256()
    per_mesh=[]
    for oi,obj in enumerate(objects):
        original_matrix=finite_matrix(obj.matrix_world)
        max_object_matrix_error=max(
            max_object_matrix_error,
            matrix_parity(forward,back,original_matrix))
        parent_bone_attachments += bool(obj.parent and obj.parent_type=="BONE")
        if obj.type=="ARMATURE":
            armature_count+=1
            for bone in obj.data.bones:
                rest_world=obj.matrix_world@bone.matrix_local
                max_bone_world_rest_error=max(
                    max_bone_world_rest_error,
                    matrix_parity(forward,back,finite_matrix(rest_world)))
                original_bones+=1
        if obj.type!="MESH":
            continue
        mesh_count+=1
        det=obj.matrix_world.to_3x3().determinant()
        require(math.isfinite(det),"SOURCE_MESH_MATRIX_DET_NONFINITE")
        if det<0:negative_handed_world_objects+=1
        groups={g.index:g.name for g in obj.vertex_groups}
        for group_id,name in sorted(groups.items()):
            h.update(struct.pack("<II",oi,group_id))
            h.update(name.encode("utf-8","surrogatepass")+b"\0")
        armature_objects=[
            m.object for m in obj.modifiers
            if m.type=="ARMATURE" and m.object is not None
        ]
        armature_bound=bool(armature_objects or
                           (obj.parent and obj.parent.type=="ARMATURE"))
        meshes_with_armature+=int(armature_bound)
        local_influences=0
        local_unweighted=0
        for v in obj.data.vertices:
            original_vertices+=1
            srcworld=obj.matrix_world@v.co
            srcworld=[float(x) for x in srcworld]
            require(all(math.isfinite(x) for x in srcworld),
                    "SOURCE_VERTEX_COORDINATE_NONFINITE")
            n=point(forward,srcworld)
            max_iris_abs=max(max_iris_abs,max(abs(x) for x in n))
            if max(abs(x) for x in n)>1.+1e-6:failed_domain+=1
            recovered=point(back,n)
            diff=max(abs(a-b) for a,b in zip(recovered,srcworld))
            max_vertex_error=max(max_vertex_error,diff)
            all_groups=weight_record(
                (item.group,item.weight) for item in v.groups)
            require(all(g in groups for g,_ in all_groups),
                    "SOURCE_GROUP_ADDRESS_OUT_OF_BOUNDS")
            local_influences+=len(all_groups)
            original_influences+=len(all_groups)
            digest_weight_row(h,oi,int(v.index),all_groups)
            if armature_bound:
                mesh_skin_candidate_vertices+=1
                if not all_groups:mesh_unweighted_vertices+=1
                if all_groups:
                    err=abs(sum(w for _,w in all_groups)-1.)
                    max_weight_sum_error=max(max_weight_sum_error,err)
                    bad_weight_sum_vertices+=int(err>1e-3)
            elif all_groups:
                nondeform_group_assignments+=len(all_groups)
        per_mesh.append({
            "imported_mesh_object_name_teacher_only":obj.name,
            "source_import_vertex_count":len(obj.data.vertices),
            "source_import_polygon_count":len(obj.data.polygons),
            "source_import_has_armature_binding_candidate":armature_bound,
            "source_import_vertex_group_count":len(groups),
            "source_import_sparse_influence_entries":local_influences,
            "source_import_vertex_without_weight_groups":local_unweighted,
            "source_import_local_negative_determinant":det<0,
        })
    require(mesh_count>0 and original_vertices>0,
            "SOURCE_HAS_NO_MESH_VERTEX")
    require(original_vertices<=3_000_000 and original_bones<=50_000,
            "SOURCE_VERTEX_OR_BONE_BUDGET")
    require(failed_domain==0,"SOURCE_VERTEX_NORMALIZED_DOMAIN")
    require(max_vertex_error/extent<=1e-6,
            "SOURCE_VERTEX_NUMERICAL_ROUNDTRIP")
    require(max_bone_world_rest_error/extent<=1e-6,
            "SOURCE_BONE_REST_NUMERICAL_ROUNDTRIP")
    require(max_object_matrix_error/max(1.,extent)<=1e-6,
            "SOURCE_OBJECT_MATRIX_NUMERICAL_ROUNDTRIP")
    require(original_vertices==bounds["mesh_vertex_count"]
            and mesh_count==bounds["mesh_object_count"],
            "BOUNDS_IMPORT_AND_PARITY_MISMATCH")
    return {
        "schema":SCHEMA,
        "status":"IMPORTED_SCENE_FULL_VERTEX_AND_BONE_MATH_CHECKED__SOURCE_TRUTH_UNKNOWN",
        "canonical_asset_id":bounds["canonical_asset_id"],
        "candidate_id":bounds["candidate_id"],
        "original_source_sha256":bounds["source_sha256"],
        "imported_world_source_space":"IMPORTED_BLENDER_SOURCE_WORLD",
        "geometry_imported_vertex_count":original_vertices,
        "imported_mesh_object_count":mesh_count,
        "imported_bone_rest_matrix_count":original_bones,
        "imported_armature_object_count":armature_count,
        "imported_parent_bone_attachment_candidate_count":parent_bone_attachments,
        "imported_meshes_with_armature_binding_candidate":meshes_with_armature,
        "imported_skin_candidate_vertex_count":mesh_skin_candidate_vertices,
        "imported_skin_candidate_vertices_without_groups":mesh_unweighted_vertices,
        "imported_skin_candidate_vertices_bad_sum_1e_minus3":bad_weight_sum_vertices,
        "max_imported_skin_candidate_weight_sum_error":max_weight_sum_error,
        "imported_nonarmature_vertex_group_assignments":nondeform_group_assignments,
        "imported_sparse_influence_total_entries":original_influences,
        "source_sparse_weights_sha256":h.hexdigest(),
        "imported_negative_handed_mesh_objects":negative_handed_world_objects,
        "source_world_extent":extent,
        "normalized_iris_max_coordinate_abs":max_iris_abs,
        "source_vertex_roundtrip_max_relative_error":max_vertex_error/extent,
        "source_bone_rest_world_roundtrip_max_relative_error":
            max_bone_world_rest_error/extent,
        "source_object_world_matrix_roundtrip_max_relative_error":
            max_object_matrix_error/max(1.,extent),
        "mesh_source_index_not_reordered":True,
        "sparse_weights_not_mutated":True,
        "source_semantics_from_group_names":"NOT_AUTHORIZED",
        "original_source_authoring_poses_courted":False,
        "original_source_units_courted":False,
        "source_native_IK_controller_truth_courted":False,
        "skin_bind_to_deformation_consequence_courted":False,
        "source_original_graphics_rgba_courted":False,
        "graphics_stylized_2d_art_courted":False,
        "source_original_material_courted":False,
        "original_source_anatomical_orientation":"UNKNOWN",
        "independent_source_truth_courted":False,
        "model_head_training_admitted":False,
        "training_authorized":False,
        "materialization_authorized":False,
        "mesh_observations":per_mesh,
    }


def main():
    import bpy
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-sha256",required=True)
    p.add_argument("--asset-id",required=True)
    p.add_argument("--candidate-id",required=True)
    p.add_argument("--bounds-output",type=Path,required=True)
    p.add_argument("--dense-output",type=Path,required=True)
    a=p.parse_args()
    bounds=measure(bpy,a.source.resolve(),a.source_sha256,
                   a.asset_id,a.candidate_id)
    report=imported_census(bpy,bounds)
    for dest,value in ((a.bounds_output,bounds),(a.dense_output,report)):
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(value,sort_keys=True,indent=2,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "source_id":a.asset_id,
        "imported_vertices":report["geometry_imported_vertex_count"],
        "imported_bones":report["imported_bone_rest_matrix_count"],
        "imported_influence_entries":report["imported_sparse_influence_total_entries"],
        "numerical_consistency":"PASS_WITHIN_IMPORTED_WORLD_ONLY",
        "source_training_admitted":False,
    },sort_keys=True))


if __name__=="__main__":
    argv=sys.argv
    try:
        sys.argv=[argv[0]]+argv[argv.index("--")+1:]
        main()
    finally:
        sys.argv=argv
