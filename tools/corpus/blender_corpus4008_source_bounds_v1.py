#!/usr/bin/env python3
"""Measure an imported source-space full-scene AABB, SHA locked and unqualified.

An AABB is NOT proof of original authoring-frame axes, anatomical front,
native pose, rig controls, mesh parity or artistic style. It merely enables
a reversible dimensionless similarity transform for all future models.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    import_source,sha,require,
)

SCHEMA="RealSaS.Corpus4008MeasuredImportedSourceBounds.v1"


def measure(bpy,source,expected_sha256,aid,candidate):
    require(source.is_file(),"BOUNDS_SOURCE_MISSING")
    require(sha(source)==expected_sha256,"BOUNDS_SOURCE_SHA_MISMATCH")
    require(candidate.endswith(expected_sha256) or
            candidate.endswith(expected_sha256[:24]),
            "BOUNDS_CANDIDATE_SHA_MISMATCH")
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "SOURCE_AUTOEXEC_FORBIDDEN")
    import_source(bpy,source,load_anim=False)
    from mathutils import Vector
    vertices=[]
    mesh_objects=0
    mesh_vertices=0
    for ob in bpy.context.scene.objects:
        if ob.type!="MESH":continue
        mesh_objects+=1
        mesh_vertices+=len(ob.data.vertices)
        # Bound box corners are a conservative native-import scene AABB
        # after local/world transforms. They do not encode topology truth.
        for corner in ob.bound_box:
            p=ob.matrix_world @ Vector(corner)
            vertices.append(tuple(float(x) for x in p))
    require(vertices,"SOURCE_NO_MESH_BOUNDS")
    mn=[min(p[i] for p in vertices) for i in range(3)]
    mx=[max(p[i] for p in vertices) for i in range(3)]
    require(all(math.isfinite(x) for x in mn+mx),"NONFINITE_BOUNDS")
    require(max(b-a for a,b in zip(mn,mx))>1e-12,
            "DEGENERATE_BOUNDS")
    return {
        "schema":SCHEMA,
        "status":"IMPORTED_SOURCE_BOUNDS_ONLY",
        "canonical_asset_id":aid,
        "candidate_id":candidate,
        "source_sha256":expected_sha256,
        "source_format":source.suffix.lower(),
        "source_space":"IMPORTED_BLENDER_SOURCE_WORLD",
        "source_world_min":mn,"source_world_max":mx,
        "mesh_object_count":mesh_objects,
        "mesh_vertex_count":mesh_vertices,
        "camera_count_imported":sum(o.type=="CAMERA" for o in bpy.context.scene.objects),
        "blender_version":bpy.app.version_string,
        "source_rest_pose":"IMPORTER_NO_ANIMATION__AUTHORING_REST_NOT_PROVEN",
        "source_rest_parity_courted":False,
        "source_unit_conversion_preexisting":"IMPORTED_SCENE_DEPENDENT_UNKNOWN",
        "source_meter_unit":"UNKNOWN",
        "anatomical_front":"UNKNOWN",
        "anatomical_up":"UNKNOWN",
        "parent_attachment_world_parity_courted":False,
        "independent_source_truth_courted":False,
        "native_2d_art_courted":False,
        "training_authorized":False,
    }


def main():
    import bpy
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-sha256",required=True)
    p.add_argument("--asset-id",required=True)
    p.add_argument("--candidate-id",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    v=measure(bpy,a.source.resolve(),a.source_sha256,
              a.asset_id,a.candidate_id)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(v,sort_keys=True,indent=2,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "mesh_objects":v["mesh_object_count"],
        "mesh_vertices":v["mesh_vertex_count"],
        "source_bounds_observed":True,
        "anatomical_frame":"UNKNOWN",
        "source_truth":False,
    },sort_keys=True))


if __name__=="__main__":
    argv=sys.argv
    try:
        sys.argv=[argv[0]]+argv[argv.index("--")+1:]
        main()
    finally:
        sys.argv=argv
