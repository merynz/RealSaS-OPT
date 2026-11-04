#!/usr/bin/env python3
"""Blender teacher-only source-locus FULL/REMOVE/STANDALONE/recolour probe.

Exact V5 observation source + rest geometry + pixel ray-stack provenance are
required. The Blender object/material node names are privileged TEACHER ONLY.
No part/rig/semantics label is ever minted for a student. Alpha-preserving
palette perturbations are performed in a disposable render scene; original
source files are read-only. This is a small pilot, not family authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.observation import blender_controlled_8view_v2 as observations
from tools.iris_vnext import blender_authored_teacher_export_v1 as source_transport
from tools.iris_vnext import blender_authored_teacher_export_v3 as geometry_transport
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as replay
from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as appearance
from tools.truth_foundry.blender_teacher_scene_v1 import _appearance_and_uv
from tools.corpus.select_generic_source_locus_rays_v1 import select_rays as select_generic_rays


SCHEMA = "RealSaS.Corpus370SourceLocusPaletteIntervention.v1"
TOL = 0.025
COLOR_A = (0.93, 0.17, 0.21, 1.0)
COLOR_B = (0.14, 0.82, 0.48, 1.0)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(ref, code):
    path=Path(str(ref.get("path") or "")).resolve()
    if not path.is_file() or sha(path)!=ref.get("sha256"):
        raise RuntimeError(code)
    return path


def require(ok,code):
    if not ok:
        raise RuntimeError(code)


def _palette_material(original, color, *, suffix):
    """Clone original alpha/BSDF graph; change only ONE proven emission RGB."""
    require(original is not None and original.use_nodes and original.node_tree,
            "PALETTE_ORIGINAL_SHADER_GRAPH_MISSING")
    clone=original.copy()
    clone.name="RealSaS_Palette_"+suffix
    emissive=[node for node in clone.node_tree.nodes if node.type=="EMISSION"
              and any(link.from_node == node for link in clone.node_tree.links)]
    if len(emissive)!=1:
        bpy.data.materials.remove(clone,do_unlink=True)
        raise RuntimeError("PALETTE_AMBIGUOUS_SOURCE_EMISSION_GRAPH")
    emission=emissive[0]
    for link in list(emission.inputs["Color"].links):
        clone.node_tree.links.remove(link)
    emission.inputs["Color"].default_value=color
    emission.inputs["Strength"].default_value=1.
    return clone


def _material_domain(global_map,face_index):
    mesh, local, material, _ = global_map[face_index]
    return str(mesh.name)+":"+str(material)


def _remove_two_faces(global_map,face_a,face_b):
    """Render removal of both source-addressed faces, not a semantic layer."""
    touched=set()
    for face in (face_a,face_b):
        mesh,local,original,transparent=global_map[face]
        mesh.polygons[local].material_index=transparent
        touched.add(mesh)
    for mesh in touched:mesh.update()
    bpy.context.view_layer.update()


def _recolor_one_face(global_map,face,color):
    """Change a single addressed face only; other domains are untouched."""
    mesh,local,original,transparent=global_map[face]
    original_material=mesh.materials[original]
    recolor=_palette_material(original_material,color,suffix="A_"+str(face))
    target=len(mesh.materials)
    mesh.materials.append(recolor)
    mesh.polygons[local].material_index=target
    mesh.update()
    bpy.context.view_layer.update()


def _modify_two_faces(global_map,face_a,face_b,*,color_a,color_b):
    touched=set()
    for face,color in ((face_a,color_a),(face_b,color_b)):
        mesh,local,original,transparent=global_map[face]
        native=mesh.materials[original]
        material=_palette_material(native,color,suffix=str(face))
        index=len(mesh.materials)
        mesh.materials.append(material)
        mesh.polygons[local].material_index=index
        touched.add(mesh)
    for mesh in touched:mesh.update()
    bpy.context.view_layer.update()


def _premul(value,mode):
    alpha,rgb=appearance._premultiplied(np.asarray(value,dtype=np.float64),mode)
    return float(alpha),np.asarray(rgb,np.float64)


def _measure(camera,cam,ray,tag,scratch):
    appearance._set_micro_camera(camera,cam,int(ray["pixel_x"]),int(ray["pixel_y"]),1/32.)
    return appearance._render_centre_rgba(scratch/(tag+".exr"))


def _rgba(value):
    return [float(x) for x in np.asarray(value,dtype=np.float64)]


def run(job_path,output_path):
    job=json.loads(Path(job_path).read_text())
    require(job.get("schema")=="RealSaS.Corpus370SourceLocusPaletteJob.v1"
            and job.get("training_authorized") is False,"PALETTE_JOB_SCOPE")
    source=read(job["source"],"PALETTE_SOURCE_BYTES")
    admission_path=read(job["admission"],"PALETTE_ADMISSION_BYTES")
    candidate_path=read(job["candidate"],"PALETTE_CANDIDATE_BYTES")
    camera_path=read(job["camera"],"PALETTE_CAMERA_BYTES")
    geometry_path=read(job["geometry"],"PALETTE_GEOMETRY_BYTES")
    ray_path=read(job["geometric_rays"],"PALETTE_RAY_BYTES")
    admission=json.loads(admission_path.read_text())
    candidate=json.loads(candidate_path.read_text())
    rays=json.loads(ray_path.read_text())
    bundle=json.loads(camera_path.read_text())
    subject=job["subject_id"]
    require(all(v.get("subject_id")==subject and v.get("source_sha256")==job["source"]["sha256"]
                for v in (admission,candidate,rays)),
            "PALETTE_SUBJECT_SOURCE_IDENTITY")
    require(candidate.get("camera_bundle",{}).get("sha256")==sha(camera_path),
            "PALETTE_SOURCE_CAMERA_BINDING")
    require(candidate.get("material_mode")=="unlit_preserve_alpha",
            "PALETTE_SOURCE_MATERIAL_POLICY")
    require(rays.get("source_rest_geometry",{}).get("sha256")==sha(geometry_path),
            "PALETTE_SOURCE_RAY_REST_BINDING")
    with np.load(geometry_path,allow_pickle=False) as geom:
        expected_v=np.asarray(geom["rest_vertices_source_world"],np.float64)
        expected_f=np.asarray(geom["faces"],np.int64)
        expected_owner=np.asarray(geom["face_owner_privileged"],np.int32)
    result={"schema":SCHEMA,
        "status":"TEACHER_INTERVENTION_MEASUREMENT_INCOMPLETE",
        "subject_id":subject,"source_sha256":sha(source),
        "parents":job,"palette_seed":int(job["palette_seed"]),
        "renderer":"BLENDER_CYCLES_EXR_PREMULTIPLIED_CENTRE_RAY",
        "rows":[],
        "authority":{"material_and_owner_names_teacher_only":True,
                     "source_material_alpha_unchanged":True,
                     "student_semantic_owner_labels_minted":False,
                     "graphics_family_minted":False,
                     "training_authorized":False},
    }
    with tempfile.TemporaryDirectory(prefix="corpus370_palette_") as temp:
        tmp=Path(temp)
        observations.import_rest(source,tmp)
        meshes=source_transport._mesh_objects()
        snapshot=geometry_transport._extract_snapshot(meshes,freeze_triangulation=True)
        actual_v=np.asarray(snapshot["vertices"],np.float64)
        actual_f=np.asarray(snapshot["faces"],np.int64)
        actual_owner=np.asarray(snapshot["face_owner"],np.int32)
        require(np.array_equal(actual_f,expected_f) and
                np.array_equal(actual_owner,expected_owner) and
                actual_v.shape==expected_v.shape,
                "PALETTE_REIMPORTED_TOPOLOGY_OR_OWNER_DRIFT")
        scale=max(1.,float(np.linalg.norm(np.ptp(expected_v,axis=0))))
        replay_err=float(np.max(np.abs(actual_v-expected_v)))
        require(replay_err<=1e-6*scale,
                "PALETTE_REIMPORTED_VERTEX_DRIFT")
        materials=replay._materials_from_render_meshes(meshes)
        receipt=replay._replay_transform(materials,list(candidate.get("material_transform") or ()))
        appearance._configure_measurement(int(job.get("cycles_samples",64)))
        cameras={int(c["view_index"]):c for c in bundle["cameras"]}
        cam=appearance._make_camera()
        target_rays=[r for r in rays["rays"] if
                     r.get("status")=="VALID_GEOMETRIC_FULL_STACK_CANDIDATE"
                     and len(r.get("hits") or [])>=2]
        require(bool(target_rays),"PALETTE_NO_MULTIHIT_VALID_SOURCE_RAY")
        selected=select_generic_rays(
            target_rays,maximum=int(job.get("maximum_rays",2)))
        result["sampled_view_indices"]=[int(r["view_index"]) for r in selected]
        result["sample_policy"]="GEOMETRY_ONLY_OPPOSITE_VIEW_DIVERSE_V1"
        scratch=tmp/"scratch";scratch.mkdir(parents=True,exist_ok=True)
        first=selected[0]
        appearance._set_micro_camera(cam,cameras[first["view_index"]],
                                      first["pixel_x"],first["pixel_y"],1/32.)
        carrier=appearance._calibrate_rgb_carrier(cam,cameras[first["view_index"]],scratch)
        mode=str(carrier["mode"])
        result["carrier_mode"]=mode
        result["carrier_calibration"]=carrier
        original={}
        for index,ray in enumerate(selected):
            original[index]=_measure(cam,cameras[ray["view_index"]],ray,
                                     "original_"+str(index),scratch)
        uv,uv_valid,materials_truth,images=_appearance_and_uv(meshes,snapshot)
        arrays={
            "rest_vertices_source_world":expected_v,
            "faces":expected_f,
            "face_owner_privileged":expected_owner,
            "face_material_privileged":np.asarray(snapshot["face_material"],np.int32),
            "triangle_uv":uv,
            "triangle_uv_valid":uv_valid,
        }
        transparent=appearance._transparent_material()
        soup,global_map=appearance._build_triangle_soup(arrays,meshes,transparent)
        for mesh in meshes:mesh.hide_render=True
        for obj in soup:obj.hide_render=False
        bpy.context.view_layer.update()
        try:
            for index,ray in enumerate(selected):
                key=cameras[ray["view_index"]]
                face_a=int(ray["hits"][0]["face_index_teacher_only"])
                face_b=int(ray["hits"][1]["face_index_teacher_only"])
                row={
                    "view_index":ray["view_index"],
                    "pixel_x":ray["pixel_x"],"pixel_y":ray["pixel_y"],
                    "source_faces_teacher_only":[face_a,face_b],
                    "source_barycentric_teacher_only":[
                        ray["hits"][0]["source_barycentric_teacher_only"],
                        ray["hits"][1]["source_barycentric_teacher_only"]],
                    "source_material_domains_teacher_only":[
                        _material_domain(global_map,face_a),
                        _material_domain(global_map,face_b)],
                    "material_semantic_name_used_as_label":False,
                    "same_drawable_label_minted":False,
                }
                appearance._set_soup_mode(global_map,mode="FULL")
                baseline=_measure(cam,key,ray,"full_"+str(index),scratch)
                alpha,bgr=_premul(baseline,mode)
                original_a,original_rgb=_premul(original[index],mode)
                baseline_error=float(max(abs(alpha-original_a),
                                         np.max(np.abs(bgr-original_rgb))))
                row["source_reimport_vs_full_soup_max_error"]=baseline_error
                row["original_rgba"]=_rgba(original[index])
                row["full_soup_rgba"]=_rgba(baseline)
                if baseline_error>TOL:
                    row.update(status="ABSTAIN_RENDER_SOURCE_PARITY",
                               reason="ORIGINAL_SOURCE_RGBA_VS_SOUP_DIFFERENCE")
                    result["rows"].append(row)
                    continue
                appearance._set_soup_mode(global_map,mode="STANDALONE",face_index=face_a)
                alone=_measure(cam,key,ray,"standalone_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="REMOVAL",face_index=face_a)
                removal=_measure(cam,key,ray,"removal_"+str(index),scratch)
                standalone_a,standalone_rgb=_premul(alone,mode)
                removal_a,removal_rgb=_premul(removal,mode)
                # Frontmost contributor reference. No assumption that semantic
                # object and drawable groups are identical.
                reconstruct_rgb=standalone_rgb+(1-standalone_a)*removal_rgb
                reconstruct_alpha=standalone_a+(1-standalone_a)*removal_a
                consequence=float(max(abs(alpha-reconstruct_alpha),
                                       np.max(np.abs(bgr-reconstruct_rgb))))
                row.update(
                    standalone_rgba=_rgba(alone),
                    removal_rgba=_rgba(removal),
                    front_face_reconstruction_error=consequence)
                if consequence>TOL:
                    row.update(status="ABSTAIN_ALPHA_COMPOSITING_RECONSTRUCTION")
                    result["rows"].append(row)
                    continue
                # Required morphology-agnostic reciprocal source-locus
                # controls: B-only removal and (A,B) removal. A hidden B
                # can become observable after A is removed. This is NOT
                # a drawable or mechanical-owner label.
                appearance._set_soup_mode(global_map,mode="REMOVAL",face_index=face_b)
                remove_b=_measure(cam,key,ray,"remove_b_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="FULL")
                _remove_two_faces(global_map,face_a,face_b)
                remove_ab=_measure(cam,key,ray,"remove_ab_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="FULL")
                no_op=_measure(cam,key,ray,"noop_"+str(index),scratch)
                _recolor_one_face(global_map,face_a,COLOR_B)
                recolor_a=_measure(cam,key,ray,"recolor_a_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="FULL")
                recolor_a_noop=_measure(cam,key,ray,"recolor_a_noop_"+str(index),scratch)
                # V2: isolate B's causal contribution under both normal
                # visibility and A-removal. An unchanged removal image does
                # NOT imply zero effect: same-colour/opaque layers can mask it.
                appearance._set_soup_mode(global_map,mode="FULL")
                _recolor_one_face(global_map,face_b,COLOR_B)
                recolor_b=_measure(cam,key,ray,"recolor_b_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="FULL")
                recolor_b_noop=_measure(cam,key,ray,"recolor_b_noop_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="REMOVAL",face_index=face_b)
                _recolor_one_face(global_map,face_a,COLOR_B)
                recolor_a_given_remove_b=_measure(
                    cam,key,ray,"recolor_a_given_remove_b_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="REMOVAL",face_index=face_a)
                _recolor_one_face(global_map,face_b,COLOR_B)
                recolor_b_given_remove_a=_measure(
                    cam,key,ray,"recolor_b_given_remove_a_"+str(index),scratch)
                appearance._set_soup_mode(global_map,mode="FULL")
                row["generic_counterfactual"]={
                    "schema":"RealSaS.Corpus370GenericLocusCounterfactual.v1",
                    "source_sha256":sha(source),
                    "camera_sha256":sha(camera_path),
                    "rest_geometry_sha256":sha(geometry_path),
                    "view_index":int(ray["view_index"]),
                    "pixel_x":int(ray["pixel_x"]),
                    "pixel_y":int(ray["pixel_y"]),
                    "locus_a":{
                        "face_index_teacher_only":face_a,
                        "barycentric_teacher_only":ray["hits"][0]["source_barycentric_teacher_only"],
                    },
                    "locus_b":{
                        "face_index_teacher_only":face_b,
                        "barycentric_teacher_only":ray["hits"][1]["source_barycentric_teacher_only"],
                    },
                    "carrier_mode":mode,
                    "camera_ray_complete":bool(ray["complete"]),
                    "depth_tie":bool(ray["has_depth_tie"]),
                    "interventions_preserve_geometry_camera_pose_alpha":True,
                    "teacher_labels_to_student":False,
                    "owner_identity_minted":False,
                    "training_authorized":False,
                    "semantic_or_material_name_as_label":False,
                    "rgba":{
                        "full":_rgba(baseline),
                        "noop":_rgba(no_op),
                        "remove_a":_rgba(removal),
                        "remove_b":_rgba(remove_b),
                        "remove_both":_rgba(remove_ab),
                        "recolor_a":_rgba(recolor_a),
                        "recolor_a_noop":_rgba(recolor_a_noop),
                    },
                }
                # Preserve the old V1 raw measurements byte-for-byte in
                # their own declared schema; V2 gets newly measured arms and
                # is independently courted into a DIFFERENT receipt.
                v1=row["generic_counterfactual"]
                row["generic_counterfactual_v2"]={
                    **v1,
                    "schema":"RealSaS.Corpus370GenericLocusReciprocalPalette.v2",
                    "rgba":{
                        **v1["rgba"],
                        "recolor_b":_rgba(recolor_b),
                        "recolor_a_given_remove_b":_rgba(recolor_a_given_remove_b),
                        "recolor_b_given_remove_a":_rgba(recolor_b_given_remove_a),
                        "recolor_b_noop":_rgba(recolor_b_noop),
                    },
                }
                appearance._set_soup_mode(global_map,mode="FULL")
                _modify_two_faces(global_map,face_a,face_b,
                                  color_a=COLOR_B,color_b=COLOR_A)
                split=_measure(cam,key,ray,"palette_split_"+str(index),scratch)
                split_a,split_rgb=_premul(split,mode)
                appearance._set_soup_mode(global_map,mode="FULL")
                _modify_two_faces(global_map,face_a,face_b,
                                  color_a=COLOR_A,color_b=COLOR_A)
                collision=_measure(cam,key,ray,"palette_collision_"+str(index),scratch)
                collision_a,collision_rgb=_premul(collision,mode)
                alpha_delta=max(abs(split_a-alpha),abs(collision_a-alpha))
                row.update(
                    palette_distinct_rgba=_rgba(split),
                    palette_collision_rgba=_rgba(collision),
                    palette_alpha_max_delta=float(alpha_delta),
                    palette_rgb_difference_norm=float(np.linalg.norm(split_rgb-collision_rgb)),
                    different_source_domains=bool(row["source_material_domains_teacher_only"][0]!=
                                                  row["source_material_domains_teacher_only"][1]),
                    split_and_collision_are_color_only=True,
                    status=("ABSTAIN_ALPHA_CHANGED_BY_RECOLOR"
                            if alpha_delta>TOL else
                            "ABSTAIN_PALETTE_NO_OBSERVABLE_RGB_DELTA"
                            if float(np.linalg.norm(split_rgb-collision_rgb))<=0.005
                            else "PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE"))
                result["rows"].append(row)
        finally:
            appearance._set_soup_mode(global_map,mode="FULL")
    pass_rows=sum(r["status"]=="PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE" for r in result["rows"])
    result["status"]="MEASURED_SOURCE_PALETTE_PILOT__NO_GRAPHICS_FAMILY_AUTHORITY"
    result["qualified_intervention_rows"]=pass_rows
    result["abstain_intervention_rows"]=len(result["rows"])-pass_rows
    result["replayed_material_count"]=len(receipt)
    result["maximum_rest_vertex_replay_error_source_units"]=replay_err
    output_path=Path(output_path).resolve()
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"subject_id":subject,"status":result["status"],
                      "qualified_rows":pass_rows,
                      "abstain_rows":result["abstain_intervention_rows"]},sort_keys=True),flush=True)


def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(raw)
    run(a.job,a.output)


if __name__=="__main__":main()
