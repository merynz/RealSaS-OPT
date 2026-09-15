from __future__ import annotations

"""Dense zero-surface BODY materializer qualified against the FINAL motion engine.

V1 established the topology/coverage treatment. V2 intentionally reuses its exact
geometry, raster, compaction and dynamic metric helpers, but replaces the provisional
rotation-only preset builder with the same historical phase-motion + quality compiler
consumed by the final product. Therefore face admission and final PRODUCT motion proof
exercise the same motion_state_hash. No training or threshold relaxation occurs.
"""

import argparse
from pathlib import Path
import json

import numpy as np

from compiler.realsas_compiler_core.appearance_atlas import build_sprite_panel_appearance
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.dense_zero_surface_bridge import (
    build_dense_zero_surface_candidate,
    project_dense_vertices,
    qualify_dense_zero_surface_mesh,
    replay_dense_zero_surface_compaction,
)
from compiler.realsas_compiler_core.mesh.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.motion_locomotion import build_mage_historical_phase_motion
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_core.v4 import build_mechanical_state, validate_appearance_binding

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_fit2_dense_zero_surface_body_v1 as v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1

SCHEMA = "RealSaS.MageFIT2.DenseZeroSurfaceBody.v2.final_historical_motion"
STATUS = "PASS__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY_V0_V7"
MOTION_AUTHORITY = "HISTORICAL_PHASE_QUALITY_MOTION_STATE"


def _final_motion(mechanical) -> dict:
    phase = build_mage_historical_phase_motion(mechanical)
    quality = compile_motion_quality(phase, mechanical)
    value = quality.to_dict()
    md = dict(value.get("metadata") or {})
    if md.get("historical_motion_engine_recovered") is not True:
        raise RuntimeError("DENSE_BODY_V2_HISTORICAL_MOTION_ENGINE_NOT_RECOVERED")
    if md.get("phase_authored_locomotion") is not True:
        raise RuntimeError("DENSE_BODY_V2_PHASE_MOTION_NOT_BOUND")
    if md.get("historical_mesh_authority_used") is True or md.get("historical_weight_authority_used") is True:
        raise RuntimeError("DENSE_BODY_V2_HISTORICAL_MECHANICAL_AUTHORITY_FORBIDDEN")
    if value.get("order_tracks") or value.get("visibility_tracks"):
        raise RuntimeError("DENSE_BODY_V2_NON_ROTATION_DIRECTIONAL_SCOPE")
    for tr in value.get("joint_tracks") or ():
        for key in tr.get("keys") or ():
            if max(abs(float(x)) for x in key.get("translation_xy", (0.0, 0.0))) > 1e-12:
                raise RuntimeError("DENSE_BODY_V2_TRANSLATION_SCOPE_NOT_QUALIFIED")
            if max(abs(float(x) - 1.0) for x in key.get("scale_xy", (1.0, 1.0))) > 1e-12:
                raise RuntimeError("DENSE_BODY_V2_SCALE_SCOPE_NOT_QUALIFIED")
            if abs(float(key.get("depth_offset", 0.0))) > 1e-12:
                raise RuntimeError("DENSE_BODY_V2_DEPTH_SCOPE_NOT_QUALIFIED")
    return value


def run(args) -> dict:
    zero_path, surface_path = Path(args.zero_surface), Path(args.fit2_surface)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if v1._sha(surface_path) != v1.EXPECTED_SURFACE_FILE_SHA256:
        raise RuntimeError("DENSE_BODY_V2_SURFACE_FILE_SHA_DRIFT")
    surface = fit2io.load_surface(surface_path)
    skeleton = fit2io.load_skeleton(Path(args.skeleton))
    skin = fit2io.load_skin(Path(args.fit2_skin))
    mechanical = build_mechanical_state(surface, skeleton, skin)
    dense_vertices, dense_faces = v1._load_zero_surface(zero_path)
    replay = replay_dense_zero_surface_compaction(
        surface, dense_vertices, source_zero_surface_sha256=v1.EXPECTED_ZERO_SURFACE_SHA256
    )
    if replay.compact_node_count != fit2io.EXPECTED_SURFACE_NODES:
        raise RuntimeError("DENSE_BODY_V2_REPLAY_GSA_CARDINALITY_DRIFT")
    cameras = v1._load_cameras(tuple(map(Path, args.cameras)))
    observations = v1._load_observations(tuple(map(Path, args.observations)))
    _owner_manifest, rigid_masks = v1._load_owner_masks(Path(args.owner_manifest), tuple(map(Path, args.owner_rasters)))
    _foreground_manifest, foreground_rows = v1._load_foreground_manifest(Path(args.foreground_manifest))

    motion = _final_motion(mechanical)
    tracks, sample_rows = v1._motion_samples(motion)
    order, by_id = v1._joint_order(skeleton)
    joint_ids = tuple(sorted(by_id))
    surface_W = v1._surface_skin_matrix(surface, skin, joint_ids)
    dense_W = surface_W[replay.dense_vertex_to_compact_index]
    if not np.allclose(dense_W.sum(1), 1.0, atol=1e-7, rtol=0.0):
        raise RuntimeError("DENSE_BODY_V2_DENSE_WEIGHT_SIMPLEX_DRIFT")
    print("DENSE_BODY_V2_REPLAY_PASS " + json.dumps(replay.summary(), sort_keys=True), flush=True)
    print("DENSE_BODY_V2_FINAL_MOTION " + json.dumps({
        "motion_state_hash": motion["motion_state_hash"],
        "authority": MOTION_AUTHORITY,
        "edge_limit": v1.EDGE_STRETCH_LIMIT,
        "area_limit": v1.AREA_CHANGE_LIMIT,
        "flip_limit": 0,
        "sample_count": len(sample_rows),
    }, sort_keys=True), flush=True)

    artifacts, view_rows = [], []
    for view in range(8):
        print(f"\n{'='*96}\nDENSE_BODY_V2_V{view}_START\n{'='*96}", flush=True)
        camera = cameras[view]
        dense_xy = project_dense_vertices(dense_vertices, camera, view_index=view)
        body_need = observations[view] & (~rigid_masks[view])
        candidate_ids = v1._candidate_faces(dense_xy, dense_faces, body_need)
        if not len(candidate_ids):
            raise RuntimeError(f"DENSE_BODY_V2_NO_CANDIDATE_FACE_V{view}")
        print(f"V{view} candidate_faces={len(candidate_ids)}", flush=True)

        coeff, affine_report = v1._fit_view_affine(surface, skeleton, view)
        pivots = {jid: np.asarray([*map(float, by_id[jid].position), 1.0]) @ coeff for jid in joint_ids}
        frame_matrices, frame_keys = [], []
        for cid, t in sample_rows:
            frame_matrices.append(v1._skinning_matrices(order, by_id, joint_ids, pivots, tracks.get(cid, {}), float(t)))
            frame_keys.append((cid, float(t)))
        safe_mask, max_edge, max_area, flipped = v1._dynamic_safe_faces(
            dense_xy, dense_faces, candidate_ids, dense_W, tuple(frame_matrices)
        )
        safe_ids = candidate_ids[safe_mask]
        print(f"V{view} dynamically_safe_faces={len(safe_ids)}/{len(candidate_ids)}", flush=True)
        selected, coverage = v1._select_coverage_faces(
            dense_xy, dense_faces, safe_ids, observations[view], rigid_masks[view]
        )
        selected_set = set(selected)
        local = np.asarray([i for i, fi in enumerate(candidate_ids) if int(fi) in selected_set], dtype=np.int64)
        if not len(local):
            raise RuntimeError(f"DENSE_BODY_V2_EMPTY_SELECTED_SET_V{view}")
        sel_edge = float(max_edge[local].max())
        sel_area = float(max_area[local].max())
        sel_flips = int(flipped[local].sum())
        if sel_edge > v1.EDGE_STRETCH_LIMIT + 1e-12 or sel_area > v1.AREA_CHANGE_LIMIT + 1e-12 or sel_flips:
            raise RuntimeError(f"DENSE_BODY_V2_SELECTED_DYNAMIC_GATE_FAIL_V{view}:{sel_edge}:{sel_area}:{sel_flips}")
        selected_sha = v1._array_sha(np.asarray(sorted(selected_set), dtype=np.int64))

        dynamic = {
            "schema": SCHEMA + ".DynamicFaceWitness.v1",
            "view": view,
            "source_zero_surface_sha256": v1.EXPECTED_ZERO_SURFACE_SHA256,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "motion_state_hash": str(motion["motion_state_hash"]),
            "motion_authority": MOTION_AUTHORITY,
            "historical_motion_engine_recovered": True,
            "sample_rows": frame_keys,
            "candidate_face_count": int(len(candidate_ids)),
            "dynamically_safe_face_count": int(len(safe_ids)),
            "selected_face_count": len(selected_set),
            "selected_face_set_sha256": selected_sha,
            "edge_stretch_limit": v1.EDGE_STRETCH_LIMIT,
            "area_change_limit": v1.AREA_CHANGE_LIMIT,
            "triangle_flip_limit": 0,
            "selected_max_edge_stretch_ratio": sel_edge,
            "selected_max_area_change_ratio": sel_area,
            "selected_triangle_flip_count": sel_flips,
            "affine_projection_report": affine_report,
        }
        dynamic_hash = content_sha256(dynamic)
        dynamic_path = output_dir / f"V{view}_DENSE_BODY_DYNAMIC_FACE_WITNESS.json"
        v1._write_json(dynamic_path, {**dynamic, "witness_hash": dynamic_hash})
        coverage_witness = {
            "schema": SCHEMA + ".CoverageWitness.v1",
            "view": view,
            "observation_sha256": ceiling_v1.OBSERVATION_SHA256[view],
            "owner_raster_sha256": v1._sha(Path(args.owner_rasters[view])),
            "owner_manifest_sha256": v1.EXPECTED_OWNER_MANIFEST_SHA256,
            "selected_face_set_sha256": selected_sha,
            "required_recall": v1.REQUIRED_RECALL,
            "required_precision": v1.REQUIRED_PRECISION,
            **coverage,
        }
        coverage_hash = content_sha256(coverage_witness)
        coverage_path = output_dir / f"V{view}_DENSE_BODY_COVERAGE_WITNESS.json"
        v1._write_json(coverage_path, {**coverage_witness, "witness_hash": coverage_hash})

        candidate = build_dense_zero_surface_candidate(
            surface, dense_vertices, dense_faces, replay,
            selected_face_indices=selected_set,
            camera=camera,
            view_index=view,
            camera_binding_hash=ceiling_v1.CAMERA_SHA256[view],
            source_zero_surface_sha256=v1.EXPECTED_ZERO_SURFACE_SHA256,
            dynamic_witness_hash=dynamic_hash,
            coverage_witness_hash=coverage_hash,
        )
        mesh, topology_report = qualify_dense_zero_surface_mesh(
            surface, candidate, dense_vertices, dense_faces, replay,
            camera=camera, source_zero_surface_sha256=v1.EXPECTED_ZERO_SURFACE_SHA256,
        )
        mesh_skin = bind_mwb2_mesh_skin(surface, skeleton, skin, mesh, require_anchor_unified_topology=False)
        if mesh_skin.skin_binding_hash != fit2io.EXPECTED_SKIN_LINEAGE or mesh_skin.metadata.get("historical_weight_transfer_used") is not False:
            raise RuntimeError(f"DENSE_BODY_V2_MESH_SKIN_AUTHORITY_DRIFT_V{view}")
        fg = foreground_rows[view]
        appearance = build_sprite_panel_appearance(
            mesh,
            target_view_index=view,
            source_observation_hash=ceiling_v1.OBSERVATION_SHA256[view],
            source_width=int(fg["width"]),
            source_height=int(fg["height"]),
            atlas_width=int(fg["atlas_width"]),
            atlas_height=int(fg["atlas_height"]),
            panel_x_offset=0,
            atlas_payload_hash=str(fg["atlas_payload_hash"]),
        )
        validate_appearance_binding(appearance)
        names = {
            "mesh": f"V{view}_DENSE_ZERO_SURFACE_BODY_MESH.json",
            "skin": f"V{view}_DENSE_ZERO_SURFACE_BODY_SKIN.json",
            "appearance": f"V{view}_DENSE_ZERO_SURFACE_BODY_APPEARANCE.json",
            "dynamic": dynamic_path.name,
            "coverage": coverage_path.name,
        }
        for key, obj in (("mesh", mesh), ("skin", mesh_skin), ("appearance", appearance)):
            v1._write_json(output_dir / names[key], obj.to_dict())
        local_artifacts = []
        for name in names.values():
            p = output_dir / name
            a = {"path": name, "sha256": v1._sha(p), "bytes": p.stat().st_size}
            local_artifacts.append(a); artifacts.append(a)
        view_rows.append({
            "view": view,
            "camera_sha256": ceiling_v1.CAMERA_SHA256[view],
            "observation_sha256": ceiling_v1.OBSERVATION_SHA256[view],
            "owner_raster_sha256": v1._sha(Path(args.owner_rasters[view])),
            "candidate_face_count": int(len(candidate_ids)),
            "dynamically_safe_face_count": int(len(safe_ids)),
            "selected_face_count": len(selected_set),
            "selected_face_set_sha256": selected_sha,
            "selected_max_edge_stretch_ratio": sel_edge,
            "selected_max_area_change_ratio": sel_area,
            "selected_triangle_flip_count": sel_flips,
            "coverage": coverage,
            "topology_qualification": topology_report,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "mesh_skin_lineage_hash": mesh_skin.mesh_skin_lineage_hash,
            "appearance_lineage_hash": appearance.appearance_lineage_hash,
            "files": names,
            "artifacts": local_artifacts,
        })
        print("DENSE_BODY_V2_V" + str(view) + "_PASS " + json.dumps({
            "faces": len(selected_set),
            "recall": coverage["source_alpha_recall"],
            "precision": coverage["precision_inside_alpha"],
            "edge": sel_edge,
            "area": sel_area,
            "flips": sel_flips,
        }, sort_keys=True), flush=True)

    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "topology_method": v1.DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
        "zero_surface_sha256": v1.EXPECTED_ZERO_SURFACE_SHA256,
        "surface_file_sha256": v1.EXPECTED_SURFACE_FILE_SHA256,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "motion_state_hash": str(motion["motion_state_hash"]),
        "dynamic_motion_authority": MOTION_AUTHORITY,
        "historical_motion_engine_recovered": True,
        "owner_manifest_sha256": v1.EXPECTED_OWNER_MANIFEST_SHA256,
        "foreground_manifest_sha256": v1.EXPECTED_FOREGROUND_MANIFEST_SHA256,
        "compaction_replay": replay.summary(),
        "policy": {
            "edge_stretch_limit": v1.EDGE_STRETCH_LIMIT,
            "area_change_limit": v1.AREA_CHANGE_LIMIT,
            "triangle_flip_limit": 0,
            "required_recall": v1.REQUIRED_RECALL,
            "required_precision": v1.REQUIRED_PRECISION,
            "uniform_sample_count": v1.UNIFORM_SAMPLE_COUNT,
        },
        "views": view_rows,
        "artifacts": artifacts,
        "P1_or_P1Q_topology_used": False,
        "source_or_teacher_mesh_topology_used": False,
        "historical_weight_transfer_used": False,
        "model_query_used": False,
        "training_executed": False,
        "threshold_relaxation_used": False,
        "product_pass_claimed": False,
    }
    manifest_path = output_dir / "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"
    v1._write_json(manifest_path, manifest)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY_FINAL_MOTION",
        "manifest": manifest_path.name,
        "manifest_sha256": v1._sha(manifest_path),
        "artifact_count": len(artifacts),
        "motion_state_hash": str(motion["motion_state_hash"]),
        "dynamic_motion_authority": MOTION_AUTHORITY,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "product_pass_claimed": False,
    }
    v1._write_json(output_dir / "FIT2_DENSE_ZERO_SURFACE_BODY_SEAL.json", seal)
    print("DENSE_BODY_V2_MATERIALIZATION=" + json.dumps(seal, sort_keys=True), flush=True)
    return {"manifest": manifest, "seal": seal}


def parse_args():
    return v1.parse_args()


if __name__ == "__main__":
    run(parse_args())
