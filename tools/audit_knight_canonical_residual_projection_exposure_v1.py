from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index,
    load_face_uv,
    load_provenance_atlas,
    render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from tools.demo.render_knight_motion_preview_v1 import (
    _ctx,
    _joint_pose_v2,
    _load_texture_pages,
    _skin,
    _tracks_for_clip,
)

CLIPS = ("demo_idle_v1", "demo_run_v1", "demo_slash_v1")


def _face_pixel_counts(render, pixel_mask: np.ndarray, face_count: int) -> np.ndarray:
    mask = np.asarray(pixel_mask, dtype=bool)
    contribution = np.asarray(render.contributing_layer_mask, dtype=bool)
    owners = np.asarray(render.layer_owner_face_index, dtype=np.int64)
    selected = mask[:, :, None] & contribution & (owners >= 0)
    if not np.any(selected):
        return np.zeros((face_count,), dtype=np.int64)
    flat_pixel = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    pixel_values = np.broadcast_to(flat_pixel[:, :, None], owners.shape)[selected]
    selected_faces = owners[selected]
    base = np.int64(face_count + 1)
    unique_pairs = np.unique(pixel_values * base + selected_faces)
    face_values = unique_pairs % base
    return np.bincount(face_values.astype(np.int64), minlength=face_count).astype(np.int64)


def _edge_ratio(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray) -> np.ndarray:
    r = rest[faces]
    p = posed[faces]
    rl = np.stack((
        np.linalg.norm(r[:, 1] - r[:, 0], axis=1),
        np.linalg.norm(r[:, 2] - r[:, 1], axis=1),
        np.linalg.norm(r[:, 0] - r[:, 2], axis=1),
    ), axis=1)
    pl = np.stack((
        np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
        np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
        np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
    ), axis=1)
    return (pl / np.maximum(rl, 1e-15)).max(axis=1)


def _pixel_union_for_face_class(render, pixel_mask, face_class):
    mask = np.asarray(pixel_mask, dtype=bool)
    contribution = np.asarray(render.contributing_layer_mask, dtype=bool)
    owners = np.asarray(render.layer_owner_face_index, dtype=np.int64)
    valid = contribution & (owners >= 0)
    hit = np.zeros_like(valid)
    if np.any(valid):
        hit[valid] = face_class[owners[valid]]
    return mask & np.any(hit, axis=2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()

    ctx=_ctx(a.authority_root.expanduser().resolve(),a.run_id)
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    surface=rigging_surface_from_dict(stage_output_payload(
        ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(
        ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    skin=qualified_skin_from_dict(stage_output_payload(
        ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")).cameras,key=lambda x:int(x.view_index)))
    observation=qualified_observation_set_from_dict(stage_output_payload(
        ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1"))
    _source_rgba,source_masks=_load_source_inputs(ctx,observation)
    appearance=complete_appearance_asset_from_dict(stage_output_payload(
        ctx,"23_COMPLETE_APPEARANCE_ASSET_BAKED","RealSaS.CompleteAppearanceAssetIR.v2"))
    face_uv=load_face_uv(appearance)
    face_page=load_face_page_index(appearance)
    provenance=load_provenance_atlas(appearance)
    texture_by_view={int(row.direction_index):row for row in appearance.textures}

    rest,W,faces_tuple=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin)
    rest=np.asarray(rest,dtype=np.float64)
    W=np.asarray(W,dtype=np.float64)
    faces=np.asarray(faces_tuple,dtype=np.int64)
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    rest_renders={}
    rest_face_pixels={}
    for view in (0,2):
        tex=_load_texture_pages(texture_by_view[view])
        rr=render_caa_reference(
            mesh=candidate,camera=cameras[view],face_uv=face_uv,
            texture_rgba_u8=tex,provenance_atlas=provenance[view],
            face_page_index=face_page,positions=rest)
        rest_renders[view]=rr
        rest_face_pixels[view]=_face_pixel_counts(rr,np.asarray(rr.final_alpha,dtype=bool),len(candidate.faces))

    report={
        "schema":"RealSaS.KnightCanonicalResidualProjectionExposureAudit.v1",
        "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_AUTHORITY",
        "run_id":a.run_id,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "appearance_asset_hash":appearance.asset_hash,
        "clips":[],
        "claim_boundary":[
            "Uses canonical mechanical seam support, not the frozen renderer geometry-support bug.",
            "Measures whether residual screen-space extra alpha is produced by mechanically safe faces and/or faces newly exposed relative to rest.",
            "This is a presentation/projection attribution audit, not a product-authority promotion."
        ],
    }

    for clip_id in CLIPS:
        payload=json.loads((ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json").read_text())
        tracks,_=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        frame_rows=[]
        for frame_index,t in enumerate(times):
            mats,_,frame_hash=_joint_pose_v2(
                skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,mats)
            er=_edge_ratio(rest,posed,faces)
            views=[]
            for view in (0,2):
                tex=_load_texture_pages(texture_by_view[view])
                rr=render_caa_reference(
                    mesh=candidate,camera=cameras[view],face_uv=face_uv,
                    texture_rgba_u8=tex,provenance_atlas=provenance[view],
                    face_page_index=face_page,positions=posed)
                source=np.asarray(source_masks[view],dtype=bool)
                alpha=np.asarray(rr.final_alpha,dtype=bool)
                extra=alpha & ~source
                extra_count=int(np.count_nonzero(extra))
                posed_face_pixels=_face_pixel_counts(rr,alpha,len(candidate.faces))
                newly_exposed_face=(rest_face_pixels[view]==0) & (posed_face_pixels>0)

                classes={
                    "edge_le_1_10": er<=1.10,
                    "edge_le_1_25": er<=1.25,
                    "edge_le_1_50": er<=1.50,
                    "edge_le_2_00": er<=2.00,
                    "edge_gt_2_00": er>2.00,
                    "newly_exposed_face": newly_exposed_face,
                    "newly_exposed_and_edge_le_1_50": newly_exposed_face & (er<=1.50),
                    "newly_exposed_and_edge_le_2_00": newly_exposed_face & (er<=2.00),
                }
                attribution={}
                for name,face_class in classes.items():
                    pix=_pixel_union_for_face_class(rr,extra,face_class)
                    count=int(np.count_nonzero(pix))
                    attribution[name]={
                        "extra_pixel_count":count,
                        "fraction_of_extra_alpha":None if extra_count==0 else float(count/extra_count),
                        "face_count":int(np.count_nonzero(face_class)),
                    }

                views.append({
                    "view_index":view,
                    "rendered_alpha_pixel_count":int(np.count_nonzero(alpha)),
                    "source_foreground_pixel_count":int(np.count_nonzero(source)),
                    "extra_outside_source_pixel_count":extra_count,
                    "extra_fraction_of_rendered_alpha":float(extra_count/max(1,np.count_nonzero(alpha))),
                    "edge_ratio_max":float(np.max(er)),
                    "edge_ratio_p99":float(np.quantile(er,.99)),
                    "attribution":attribution,
                })
            frame_rows.append({
                "frame_index":frame_index,
                "time_seconds":float(t),
                "motion_frame_hash":str(frame_hash),
                "views":views,
            })
        report["clips"].append({"clip_id":clip_id,"frames":frame_rows})

    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    summary=[]
    for clip in report["clips"]:
        for frame in clip["frames"]:
            for view in frame["views"]:
                if view["view_index"]==2:
                    summary.append({
                        "clip":clip["clip_id"],"frame":frame["frame_index"],
                        "extra":view["extra_outside_source_pixel_count"],
                        "edge_p99":view["edge_ratio_p99"],
                        "edge_max":view["edge_ratio_max"],
                        "rigidish_1_5":view["attribution"]["edge_le_1_50"]["fraction_of_extra_alpha"],
                        "safe_2_0":view["attribution"]["edge_le_2_00"]["fraction_of_extra_alpha"],
                        "newly_exposed":view["attribution"]["newly_exposed_face"]["fraction_of_extra_alpha"],
                        "newly_exposed_safe_2_0":view["attribution"]["newly_exposed_and_edge_le_2_00"]["fraction_of_extra_alpha"],
                    })
    print("KNIGHT_CANONICAL_RESIDUAL_PROJECTION_EXPOSURE="+json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    main()
