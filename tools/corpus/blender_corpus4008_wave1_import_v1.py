#!/usr/bin/env python3
"""Isolated original-source imported scene + full rig/skin/controller census.

Each source is opened under PINNED Blender --background --disable-autoexec.
Original import observations may be incomplete: FBX/GLB can bake/strip
controllers, textures and authoring semantics. This script does not fabricate
independent character labels, semantic regions, live IK responses or Graphics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    import_source, scene_import_report, material_inventory,
    sha, require,
)
from tools.corpus.blender_corpus4008_control_profile_v1 import source_control_profile
from tools.corpus.validate_corpus4008_deep_diagnostic_v1 import verify_control

SCHEMA="RealSaS.Corpus4008Wave1SourceRigSceneObservation.v1"


def census(bpy,source,expected_sha256,asset_id,candidate_id):
    require(source.is_file(),"SOURCE_ABSENT")
    require(sha(source)==expected_sha256,"ORIGINAL_SHA_DRIFT")
    require(candidate_id.endswith(expected_sha256) or
            candidate_id.endswith(expected_sha256[:24]),
            "ORIGINAL_SOURCE_IDENTITY")
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "UNTRUSTED_SCRIPTS_MUST_BE_DISABLED")
    # Geometry in REST mode: do not silently choose animated frame 0 as rest.
    import_source(bpy,source,load_anim=False)
    scene=scene_import_report(bpy)
    material=material_inventory(bpy)
    require(scene["training_authorized"] is False
            and scene["character_vs_prop"]=="UNKNOWN"
            and scene["source_anatomical_up"]=="UNKNOWN",
            "SCENE_AUTHORITY")
    # Independent Blender import with actions enabled.
    import_source(bpy,source,load_anim=True)
    profile=source_control_profile(bpy)
    control={
        "schema":"RealSaS.Corpus4008ImportedControlProfile.v1",
        "status":"PINNED_IMPORTED_CONTROL_OBSERVATION_ONLY",
        "asset_id":asset_id,
        "candidate_id":candidate_id,
        "original_sha256":expected_sha256,
        "imported_control":profile,
        "training_authorized":False,
        "model_family_authorized":False,
        "semantic_truth_authorized":False,
    }
    verified=verify_control(control)
    def sum_f(key):
        return sum(x.get(key,0) for x in scene["meshes"])
    report={
        "schema":SCHEMA,
        "status":"FULL_IMPORTED_SOURCE_SCENE_AND_CONTROL_CENSUS__NO_SOURCE_TRUTH_COURTS",
        "asset_id":asset_id,"candidate_id":candidate_id,
        "source_sha256":expected_sha256,
        "source_format":source.suffix.lower(),
        "blender_version":bpy.app.version_string,
        "source_scripts_autoexec_disabled":True,
        "original_source_scene":{
            "imported_scene":scene,
            "material_and_texture_inventory":material,
        },
        "imported_rig_and_animation_control":profile,
        "numeric_census":{
            "mesh_object_count":len(scene["meshes"]),
            "source_scene_node_count":len(scene["imported_nodes"]),
            "armature_count":len(scene["armatures"]),
            "vertex_count":sum_f("vertex_count"),
            "polygon_count":sum_f("polygon_count"),
            "mesh_boundary_edges":sum(x["index_topology"]["boundary_edge_count"]
                                       for x in scene["meshes"]),
            "mesh_nonmanifold_edges":sum(x["index_topology"]["nonmanifold_edge_count"]
                                         for x in scene["meshes"]),
            "mesh_disconnected_components":sum(x["index_topology"]["source_index_component_count"]
                                               for x in scene["meshes"]),
            "skin_rows_sampled":sum_f("skin_rows_sampled"),
            "skin_rows_zero_weights":sum_f("skin_rows_without_any_groups"),
            "skin_rows_bad_sum":sum_f("skin_rows_nonunity_above_1e-3"),
            "bone_parented_mesh_candidates":scene["bone_parented_mesh_candidate_count"],
            "image_dependency_missing_count":material["images_missing_external_count"],
            "material_count":material["material_count"],
            "image_count":material["image_count"],
            "observed_bone_count":verified["bones"],
            "observed_ik_constraint_count":verified["imported_ik_constraints"],
            "observed_driver_channel_count":verified["imported_driver_channels"],
            "observed_action_curve_count":verified["imported_action_curves"],
            "observed_action_count":profile["imported_action_count"],
            "observed_nla_strip_count":profile["imported_nla_strip_count"],
            "observed_nondeform_bone_count":profile["imported_nondeform_bone_count"],
        },
        "target_family_observation_routes":{
            "IRIS_GEOMETRY":"IMPORTED_MESH_AND_TOPOLOGY_OBSERVATION",
            "IRIS_SEMANTIC":"SOURCE_GROUP_HINTS_ONLY_NOT_LABEL",
            "IRIS_GRAPHICS":"MATERIAL_AND_TEXTURE_CENSUS_NOT_2D_ART",
            "TESSA":"SOURCE_SURFACE_PLUS_RIG_CANDIDATE",
            "AXIS":"IMPORTED_RIG_MOTION_CONTROL_HINTS_ONLY",
            "MIRA":"VERTEX_GROUP_WEIGHT_CANDIDATE_NOT_VALID_BIND",
        },
        "source_character_vs_prop":"UNKNOWN",
        "source_anatomical_orientation":"UNKNOWN",
        "functional_native_ik_fk":"UNKNOWN",
        "geometry_3d_truth_courted":False,
        "rig_controller_functionality_courted":False,
        "skin_deformation_response_courted":False,
        "2d_art_visual_quality_courted":False,
        "graphics_2d_art_target":"REQUIRED_NOT_AVAILABLE_FROM_THIS_OBSERVATION",
        "semantic_region_truth_courted":False,
        "commercial_ml_rights_courted":False,
        "trainable_model_family_count":0,
        "training_authorized":False,
    }
    require(report["numeric_census"]["mesh_object_count"]>0,"EMPTY_IMPORT")
    return report


def main():
    import bpy
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-sha256",required=True)
    p.add_argument("--asset-id",required=True)
    p.add_argument("--candidate-id",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    args=p.parse_args()
    report=census(bpy,args.source.resolve(),args.source_sha256,
                  args.asset_id,args.candidate_id)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(report,sort_keys=True,allow_nan=False).encode()
    require(len(b)<70*1024*1024,"UNBOUNDED_SOURCE_RECEIPT")
    args.output.write_bytes(gzip.compress(b,compresslevel=6,mtime=0))
    summary={k:report[k] for k in (
        "schema","status","asset_id","candidate_id","source_sha256",
        "source_format","blender_version","numeric_census",
        "source_character_vs_prop","functional_native_ik_fk",
        "commercial_ml_rights_courted","training_authorized")}
    summary["receipt_gz_sha256"]=sha(args.output)
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "asset_id":args.asset_id,
        "source_format":report["source_format"],
        "observations":report["numeric_census"],
        "model_truth_admitted":0,
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    old=sys.argv
    try:
        sys.argv=[old[0]]+old[old.index("--")+1:]
        main()
    finally:
        sys.argv=old
