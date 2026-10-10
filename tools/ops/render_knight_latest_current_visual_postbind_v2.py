from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.deformation_envelope_derivation_v2 import derive_deformation_envelope_v2
from compiler.realsas_compiler_core.joint_frames_v2 import derive_joint_frames_post_bind_v2
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import mechanical_carrier_evidence_from_dict
from compiler.realsas_compiler_core import motion_dynamic_proof_v2 as motion_proof
from compiler.realsas_compiler_core.skin_topology_compatibility_carrier_v1 import run_skin_topology_compatibility_carrier_v1
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_core.visual_domain_v2 import build_domain_binding, evaluate_domain_binding
from compiler.realsas_compiler_core.visual_material_render_v1 import render_visual_material
from compiler.realsas_compiler_core.visual_material_v1 import load_visual_material
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    build_visual_mesh_from_region_labels_v1,
    partition_source_mask_by_safe_face_adjacency_v1,
)
from compiler.realsas_compiler_core.visual_motion_blend_v1 import (
    build_motion_blend_coefficients,
    evaluate_motion_blend,
)
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import (
    apply_connected_palette,
    compile_connected_palette,
    prove_connected_palette,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from compiler.realsas_compiler_core.types import QualificationError
from tools.demo import render_knight_motion_preview_v1 as motion_preview
from tools.demo.render_knight_motion_preview_v1 import _skin, _tracks_for_clip
from tools.ops import render_knight_latest_current_visual_v1 as base

CAA_RUN = "KNIGHT_LATEST_CURRENT_CAA_VISUAL_20261010"
MAT_RUN = "KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010"
SKELETON_NAME = "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_QUALIFIED_SKELETON_V541.json"
RESEARCH_V6_SHA = "59dc7f4e10eeb56abc4f247ec9ba045a46a6a496"
VIEW_INDEX = 1
CLIP_ID = "demo_idle_v1"
FRAME_TIME = 0.0


def _stage_payload(ledger: dict, sid: str, schema: str) -> dict:
    row = next(r for r in ledger["stages"] if r["id"] == sid)
    out = next(o for o in row.get("outputs") or () if o.get("schema") == schema)
    return json.loads(Path(out["path"]).read_text())


def _depth_result(kwargs: dict) -> tuple[dict, np.ndarray | None]:
    try:
        rr = render_visual_material(**kwargs)
        return ({
            "sealed_pass": True,
            "unresolved_depth_ties": int(rr.unresolved_depth_tie_count),
            "fragment_overflow": int(rr.fragment_overflow_count),
            "overlap_pixel_count": int(rr.overlap_pixel_count),
        }, np.asarray(rr.straight_rgba_u8, dtype=np.uint8))
    except QualificationError as exc:
        return ({"sealed_pass": False, "error": str(exc)}, None)


def main() -> None:
    auth = Path(os.environ["AUTHORITY_ROOT"]).resolve()
    input_root = Path(os.environ["INPUT_ROOT"]).resolve()
    out = Path(os.environ["OUT_DIR"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    caa = auth / "runs" / CAA_RUN
    mat = auth / "runs" / MAT_RUN
    ledger = json.loads((caa / "ACTIVE_RUN_V2.json").read_text())
    manifest = json.loads((caa / "run_manifest.json").read_text())

    candidate = canonical_mesh_candidate_from_dict(json.loads((mat / "rebound_candidate.json").read_text()))
    carrier = mechanical_carrier_evidence_from_dict(json.loads((mat / "mechanical_carrier_evidence.json").read_text()))
    carrier_skin = qualified_carrier_skin_from_dict(json.loads((mat / "qualified_carrier_skin.json").read_text()))
    skeleton = qualified_skeleton_from_dict(json.loads((input_root / SKELETON_NAME).read_text()))
    cameras = tuple(sorted(qualified_camera_set_from_dict(_stage_payload(
        ledger, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1")).cameras,
        key=lambda c: int(c.view_index)))
    observation = qualified_observation_set_from_dict(_stage_payload(
        ledger, "07_OBSERVATION_CONTRACT_QUALIFIED", "RealSaS.QualifiedObservationSetIR.v1"))
    policy = mesh_policy_from_dict(_stage_payload(
        ledger, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1"))
    asset = complete_appearance_asset_from_dict(_stage_payload(
        ledger, "23_COMPLETE_APPEARANCE_ASSET_BAKED", "RealSaS.CompleteAppearanceAssetIR.v2"))

    rest_m = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    mech_faces = base.mechanical_faces(candidate)
    _axis_payload, envelope = derive_deformation_envelope_v2(
        skeleton=skeleton,
        camera_set=qualified_camera_set_from_dict(_stage_payload(
            ledger, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1")),
    )
    compatibility = run_skin_topology_compatibility_carrier_v1(
        carrier, skeleton=skeleton, skin=carrier_skin, envelope=envelope,
        cameras=cameras, policy=policy)
    unsafe = set(map(int, compatibility.get("unsafe_face_indices") or ()))

    ctx = {"repo_root": Path(".").resolve(), "authority_root": auth,
           "run_root": caa, "run_id": CAA_RUN, "run_manifest_path": caa / "run_manifest.json",
           "run_manifest": manifest, "ledger": ledger, "stage": {"id": "V6_CURRENT_MICRO"}}
    _rgba_by_view, mask_by_view = _load_source_inputs(ctx, observation)
    vi = VIEW_INDEX
    camera = cameras[vi]
    mask = np.asarray(mask_by_view[vi], dtype=bool)
    h, w = mask.shape
    visibility = rasterize_visible_owner(candidate, camera, positions=rest_m, width=w, height=h, max_layers=4)
    labels, seeds, charts = partition_source_mask_by_safe_face_adjacency_v1(
        mask, visibility.owner_face_index, mech_faces, unsafe, minimum_seed_pixels=64)
    region = build_visual_mesh_from_region_labels_v1(mask, labels, target_edge_px=16)
    vm = region.mesh

    binding = build_domain_binding(
        points_source_xy=np.asarray(vm.positions, dtype=np.float64),
        visual_faces=np.asarray(vm.faces, dtype=np.int64),
        vertex_region_id=np.asarray(region.vertex_region_id, dtype=np.int32),
        seed_region_labels=np.asarray(seeds, dtype=np.int32),
        owner_face_index=np.asarray(visibility.owner_face_index, dtype=np.int64),
        mechanical_positions_xyz=rest_m,
        mechanical_faces=mech_faces,
        camera=camera,
    )

    with np.load(input_root / "MIRA_AXIS41_TESSA_M_SKIN_FIELD_V55.npz", allow_pickle=False) as z:
        W = np.asarray(z["weights_axis41_canonical"], dtype=np.float64)
        joint_ids = tuple(map(str, np.asarray(z["canonical_joint_ids"]).tolist()))
        basis = str(np.asarray(z["carrier_basis_sha256"]).item())
    if W.shape != (len(rest_m), len(joint_ids)):
        raise RuntimeError("V6_CURRENT_W_SHAPE_DRIFT")
    if basis != "dbac301b9c47fa7f31a024ef0612008d8315b590d7a2dd1552a75f75c7ac9ff2":
        raise RuntimeError("V6_CURRENT_W_BASIS_DRIFT")
    blend = build_motion_blend_coefficients(
        binding, visual_faces=np.asarray(vm.faces, dtype=np.int64), mechanical_weights=W)

    joints = {str(j.canonical_joint_id): j for j in skeleton.joints}
    if set(joint_ids) != set(joints):
        raise RuntimeError("V6_CURRENT_JOINT_SET_DRIFT")
    jindex = {jid: i for i, jid in enumerate(joint_ids)}
    axis_positions = np.asarray([joints[jid].position for jid in joint_ids], dtype=np.float64)
    axis_parents = np.asarray([
        -1 if joints[jid].parent_canonical_id is None else jindex[str(joints[jid].parent_canonical_id)]
        for jid in joint_ids
    ], dtype=np.int64)

    frames, frame_qualification = derive_joint_frames_post_bind_v2(
        skeleton, carrier_skin=carrier_skin, cameras=cameras)
    if frame_qualification["status"] != "PASS":
        raise RuntimeError("V6_CURRENT_POST_BIND_FRAME_FAIL")
    def _canonical_frames(_skeleton, *, cameras):
        return frames
    motion_preview.derive_joint_frames_from_skeleton = _canonical_frames
    motion_proof.derive_joint_frames_from_skeleton = _canonical_frames

    payload, motion_path = base.motion_payloads(input_root)[CLIP_ID]
    source_report = json.loads(base.SOURCE_REPORT.read_text())
    tracks, mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
    skin_mats, _, frame_hash = motion_proof._joint_pose_v2(
        skeleton=skeleton, tracks=tracks, time_seconds=FRAME_TIME, cameras=cameras)
    matrices = np.asarray([skin_mats[jid] for jid in joint_ids], dtype=np.float64)
    posed_m = _skin(rest_m, W, joint_ids, skin_mats)

    harmonic = evaluate_domain_binding(
        binding, visual_faces=np.asarray(vm.faces, dtype=np.int64),
        posed_mechanical_positions_xyz=posed_m, camera=camera)
    legacy_v4 = evaluate_motion_blend(
        harmonic, rest_source_xy=np.asarray(vm.positions, dtype=np.float64),
        coefficients=blend, axis_positions_source=axis_positions,
        skin_matrices_source=matrices, camera=camera)
    palette = compile_connected_palette(
        axis_positions_source=axis_positions,
        axis_parents=axis_parents,
        skin_matrices_source=matrices,
        camera=camera,
    )
    connected = apply_connected_palette(
        harmonic, rest_source_xy=np.asarray(vm.positions, dtype=np.float64),
        coefficients=blend, palette=palette)
    palette_proof = prove_connected_palette(
        palette, axis_positions_source=axis_positions, axis_parents=axis_parents,
        skin_matrices_source=matrices, camera=camera)

    texrow = {int(t.direction_index): t for t in asset.textures}[vi]
    rgba = np.asarray(Image.open(Path(texrow.transport_png_path)).convert("RGBA"), dtype=np.uint8)
    provenance, donor = load_visual_material(asset, view_index=vi, rgba=rgba)
    common = dict(
        faces=np.asarray(vm.faces, dtype=np.int64), uv=np.asarray(vm.uv, dtype=np.float64),
        texture=rgba, provenance=provenance, source_view=donor,
        view_index=vi, resolution=320,
    )
    legacy_depth, legacy_image = _depth_result({**common, "positions": legacy_v4[:, :2], "depths": legacy_v4[:, 2]})
    connected_depth, connected_image = _depth_result({**common, "positions": connected[:, :2], "depths": connected[:, 2]})
    legacy_metrics = base.visual_metrics(vm.positions, legacy_v4[:, :2], vm.faces)
    connected_metrics = base.visual_metrics(vm.positions, connected[:, :2], vm.faces)

    if legacy_image is not None:
        Image.fromarray(legacy_image, "RGBA").save(out / "V1_IDLE_T0_LEGACY_V4.png")
    if connected_image is not None:
        Image.fromarray(connected_image, "RGBA").save(out / "V1_IDLE_T0_CONNECTED_V6.png")

    passed = bool(
        palette_proof["connected_palette_relations_passed"]
        and connected_depth["sealed_pass"]
        and connected_metrics["flipped_triangles"] == 0
        and connected_metrics["max_edge_ratio"] <= 4.0
    )
    report = {
        "schema": "RealSaS.CurrentKnightConnectedPaletteV6MicroCourt.v1",
        "status": "PASS_V6_CURRENT_MICRO" if passed else "FAIL_V6_CURRENT_MICRO",
        "product_authority_claimed": False,
        "research_v6_source_sha": RESEARCH_V6_SHA,
        "clip_id": CLIP_ID,
        "view_index": vi,
        "time_seconds": FRAME_TIME,
        "motion_path": str(motion_path),
        "frame_hash": frame_hash,
        "visual_vertex_count": int(len(vm.positions)),
        "visual_face_count": int(len(vm.faces)),
        "chart_count": int(len(charts)),
        "harmonic_anchor_count": int(len(binding["anchor_vertex"])),
        "palette_proof": palette_proof,
        "legacy_v4": {"depth": legacy_depth, "metrics": legacy_metrics},
        "connected_v6": {"depth": connected_depth, "metrics": connected_metrics},
        "mapping": mapping,
        "mechanics_mutated": False,
        "appearance_mutated": False,
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("V6_CURRENT_MICRO", json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("V6_CURRENT_MICRO_FAIL")


if __name__ == "__main__":
    main()
