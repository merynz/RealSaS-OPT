"""Resumable current-material V6 diagnostic; never mints product authority."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import FunctionType

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, mesh_policy_from_dict,
    qualified_camera_set_from_dict, qualified_observation_set_from_dict, qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.deformation_envelope_derivation_v2 import derive_deformation_envelope_v2
from compiler.realsas_compiler_core.joint_frames_v2 import derive_joint_frames_post_bind_v2
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import mechanical_carrier_evidence_from_dict
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.skin_topology_compatibility_carrier_v1 import run_skin_topology_compatibility_carrier_v1
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_core.visual_domain_v2 import build_domain_binding, evaluate_domain_binding
from compiler.realsas_compiler_core.visual_material_render_v1 import render_visual_material
from compiler.realsas_compiler_core.visual_material_v1 import load_visual_material
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    build_visual_mesh_from_region_labels_v1, partition_source_mask_by_safe_face_adjacency_v1,
)
from compiler.realsas_compiler_core.visual_motion_blend_v1 import build_motion_blend_coefficients
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import (
    apply_connected_palette, compile_connected_palette, prove_connected_palette,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from compiler.realsas_compiler_services.orchestrator.mainline import _local_import_closure
from tools.demo.render_knight_motion_preview_v1 import _skin, _tracks_for_clip
from tools.ops import knight_render_inputs_v1 as base
from tools.platform_render_checkpoint import RenderCheckpoint, atomic_json, checkout_identity, digest

CAA_RUN = "KNIGHT_LATEST_CURRENT_CAA_VISUAL_20261010"
MAT_RUN = "KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010"
SKELETON_NAME = "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_QUALIFIED_SKELETON_V541.json"


def _bind_frames(function, frames):
    # Invocation-local frame provider; no module monkey-patch or artifact edits.
    namespace = {**function.__globals__, "derive_joint_frames_from_skeleton":
                 lambda skeleton, *, cameras: frames}
    return FunctionType(function.__code__, namespace, function.__name__,
                        function.__defaults__, function.__closure__)


def _source_files(repo):
    result = dict(_local_import_closure("tools.ops.render_knight_sealed_v6"))
    for module in list(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if raw is not None:
            path = Path(raw).resolve()
            if getattr(module, "__name__", "").startswith(("compiler.", "tools.")) and not path.is_relative_to(repo):
                raise RuntimeError("RENDER_IMPORTED_SOURCE_OUTSIDE_CHECKOUT")
            if path.suffix == ".py" and path.is_relative_to(repo):
                result[str(path.relative_to(repo))] = digest(path)
    return dict(sorted(result.items()))


def _gif(images, path, duration_ms):
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms,
                   loop=0, disposal=2)


def main():
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    code_sha = checkout_identity(repo, os.environ["EXPECTED_CODE_SHA"])
    auth, inputs, out = [Path(os.environ[key]).resolve()
                         for key in ("AUTHORITY_ROOT", "INPUT_ROOT", "OUT_DIR")]
    resolution = int(os.environ.get("RENDER_RESOLUTION", "256"))
    frame_count = int(os.environ.get("RENDER_FRAMES", "12"))
    views = tuple(int(v) for v in os.environ.get("RENDER_VIEWS", "0,1,2,3,4,5,6,7").split(","))
    if resolution < 64 or frame_count < 2 or len(set(views)) != len(views) or any(v not in range(8) for v in views):
        raise RuntimeError("RENDER_SETTINGS_INVALID")
    caa, mat = auth / "runs" / CAA_RUN, auth / "runs" / MAT_RUN
    ledger_path, manifest_path = caa / "ACTIVE_RUN_V2.json", caa / "run_manifest.json"
    ledger, manifest = [json.loads(p.read_text()) for p in (ledger_path, manifest_path)]
    paths = [ledger_path, manifest_path, mat / "RECEIPT.json", caa / "CAA_RECEIPT.json",
             mat / "rebound_candidate.json", mat / "mechanical_carrier_evidence.json",
             mat / "qualified_carrier_skin.json", inputs / SKELETON_NAME,
             inputs / "MIRA_AXIS41_TESSA_M_SKIN_FIELD_V55.npz", repo / base.SOURCE_REPORT]
    def stage(sid, schema):
        payload, path = base.stage_payload(ledger, sid, schema)
        paths.append(path)
        return payload
    candidate = canonical_mesh_candidate_from_dict(json.loads(paths[4].read_text()))
    carrier = mechanical_carrier_evidence_from_dict(json.loads(paths[5].read_text()))
    skin = qualified_carrier_skin_from_dict(json.loads(paths[6].read_text()))
    skeleton = qualified_skeleton_from_dict(json.loads(paths[7].read_text()))
    camera_set = qualified_camera_set_from_dict(stage("05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"))
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    observation = qualified_observation_set_from_dict(stage("07_OBSERVATION_CONTRACT_QUALIFIED", "RealSaS.QualifiedObservationSetIR.v1"))
    policy = mesh_policy_from_dict(stage("18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1"))
    asset = complete_appearance_asset_from_dict(stage("23_COMPLETE_APPEARANCE_ASSET_BAKED", "RealSaS.CompleteAppearanceAssetIR.v2"))
    if len(cameras) != 8 or tuple(int(c.view_index) for c in cameras) != tuple(range(8)):
        raise RuntimeError("RENDER_CAMERA_SET_DRIFT")
    if asset.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise RuntimeError("CAA_CANDIDATE_BINDING_DRIFT")
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    mech_faces = base.mechanical_faces(candidate)
    if tuple(map(str, carrier.ordered_vertex_ids)) != tuple(str(v.candidate_vertex_id) for v in candidate.vertices):
        raise RuntimeError("CARRIER_ORDER_DRIFT")
    with np.load(inputs / "MIRA_AXIS41_TESSA_M_SKIN_FIELD_V55.npz", allow_pickle=False) as z:
        W = np.asarray(z["weights_axis41_canonical"], dtype=np.float64)
        joint_ids = tuple(map(str, np.asarray(z["canonical_joint_ids"]).tolist()))
        basis = str(np.asarray(z["carrier_basis_sha256"]).item())
    inventory = json.loads((repo / "canonical/PLATFORM_KNIGHT_INPUT_INVENTORY_V1.json").read_text())
    expected = {row["name"]: row["sha256"] for row in inventory["required_files"]}
    for name in (SKELETON_NAME, "MIRA_AXIS41_TESSA_M_SKIN_FIELD_V55.npz"):
        if digest(inputs / name) != expected[name]:
            raise RuntimeError("FROZEN_MODEL_INPUT_BYTES_DRIFT:" + name)
    if W.shape != (len(rest), len(joint_ids)) or basis != "dbac301b9c47fa7f31a024ef0612008d8315b590d7a2dd1552a75f75c7ac9ff2":
        raise RuntimeError("W_M_BASIS_DRIFT")
    joints = {str(j.canonical_joint_id): j for j in skeleton.joints}
    if set(joint_ids) != set(joints):
        raise RuntimeError("W_M_JOINT_SET_DRIFT")
    jindex = {jid: i for i, jid in enumerate(joint_ids)}
    axis = np.asarray([joints[jid].position for jid in joint_ids], dtype=np.float64)
    parents = np.asarray([-1 if joints[jid].parent_canonical_id is None else
                          jindex[str(joints[jid].parent_canonical_id)] for jid in joint_ids], dtype=np.int64)
    frames, frame_qualification = derive_joint_frames_post_bind_v2(skeleton, carrier_skin=skin, cameras=cameras)
    if frame_qualification["status"] != "PASS":
        raise RuntimeError("POST_BIND_FRAME_FAIL")
    tracks_for_clip, joint_pose = [_bind_frames(fn, frames) for fn in (_tracks_for_clip, _joint_pose_v2)]
    source_report = json.loads((repo / base.SOURCE_REPORT).read_text())
    motions = base.motion_payloads(inputs)
    paths.extend(p for _, p in motions.values())
    for tex in asset.textures:
        p = Path(tex.transport_png_path)
        if digest(p) != tex.transport_png_sha256:
            raise RuntimeError("TEXTURE_BYTES_DRIFT")
        paths.append(p)
    textures = {int(t.direction_index): t for t in asset.textures}
    materials = {}
    for vi in views:
        rgba = np.asarray(Image.open(textures[vi].transport_png_path).convert("RGBA"), dtype=np.uint8)
        materials[vi] = (rgba, *load_visual_material(asset, view_index=vi, rgba=rgba))
    ctx = {"repo_root": repo, "authority_root": auth, "run_root": caa, "run_id": CAA_RUN,
           "run_manifest_path": manifest_path, "run_manifest": manifest, "ledger": ledger,
           "stage": {"id": "SEALED_RENDER_WITNESS"}}
    _, masks = _load_source_inputs(ctx, observation)
    arrays = {str(vi): {"mask_shape": list(np.asarray(masks[vi]).shape),
                       "mask": hashlib.sha256(np.asarray(masks[vi], dtype=bool).tobytes()).hexdigest(),
                       "material": [hashlib.sha256(a.tobytes()).hexdigest() for a in materials[vi]]} for vi in views}
    contract = {"schema": "RealSaS.SealedRenderWitnessContract.v1", "product_authority_claimed": False,
                "source_files": _source_files(repo), "files": {str(p): digest(p) for p in sorted(set(paths))},
                "arrays": arrays, "settings": {"resolution": resolution, "frames": frame_count, "views": list(views)},
                "frame_contract": "POST_BIND_CANONICAL__RETARGET_AND_POSE_SAME_FRAME_SET",
                "mechanical_carrier_evidence_hash": carrier.carrier_evidence_hash,
                "carrier_skin_lineage_hash": skin.skin_lineage_hash, "caa_asset_hash": asset.asset_hash}
    checkpoint = RenderCheckpoint(out, contract)
    report = {"schema": "RealSaS.SealedKnightRenderWitness.v1", "status": "RUNNING",
              "code_sha": code_sha, "contract_sha256": checkpoint.sha, "product_authority_claimed": False,
              "mechanics_mutated": False, "appearance_mutated": False, "model_inference_executed": False,
              "renderer": "PYTHON_SEALED_VISUAL_MATERIAL_REFERENCE",
              "caa_completion_field_consumed_by_visual_material_transport": False,
              "open_acceptance": ["AMODAL_SUPPORT_AND_MATERIAL", "INTER_CHART_CONTACT_AND_SEMANTIC_ORDER", "NATIVE_END_TO_END_PARITY"],
              "clips": {}, "outputs": [], "failures": [], "views": []}
    atomic_json(out / "REPORT.json", report)
    print("RENDER_CONTRACT", checkpoint.sha, "SOURCE", code_sha, flush=True)
    _, envelope = derive_deformation_envelope_v2(skeleton=skeleton, camera_set=camera_set)
    compatibility = run_skin_topology_compatibility_carrier_v1(carrier, skeleton=skeleton, skin=skin,
                              envelope=envelope, cameras=cameras, policy=policy)
    unsafe = set(map(int, compatibility.get("unsafe_face_indices") or ()))
    visual = {}
    for vi in views:
        mask, camera = np.asarray(masks[vi], dtype=bool), cameras[vi]
        h, w = mask.shape
        visibility = rasterize_visible_owner(candidate, camera, positions=rest, width=w, height=h, max_layers=4)
        labels, seeds, charts = partition_source_mask_by_safe_face_adjacency_v1(
            mask, visibility.owner_face_index, mech_faces, unsafe, minimum_seed_pixels=64)
        region = build_visual_mesh_from_region_labels_v1(mask, labels, target_edge_px=16)
        vm = region.mesh
        binding = build_domain_binding(points_source_xy=np.asarray(vm.positions), visual_faces=np.asarray(vm.faces),
            vertex_region_id=np.asarray(region.vertex_region_id), seed_region_labels=np.asarray(seeds),
            owner_face_index=np.asarray(visibility.owner_face_index), mechanical_positions_xyz=rest,
            mechanical_faces=mech_faces, camera=camera)
        blend = build_motion_blend_coefficients(binding, visual_faces=np.asarray(vm.faces), mechanical_weights=W)
        visual[vi] = vm, binding, blend
        report["views"].append({"view_index": vi, "vertices": len(vm.positions), "faces": len(vm.faces), "charts": len(charts)})
        print("VIEW_BOUND", vi, "seconds", round(time.monotonic() - started, 2), flush=True)
    for clip_id, label in (("demo_run_v1", "RUN"), ("demo_idle_v1", "IDLE"), ("demo_slash_v1", "SLASH")):
        payload, path = motions[clip_id]
        tracks, mapping = tracks_for_clip(payload, skeleton, cameras, source_report)
        times = np.linspace(0, float(payload["duration_seconds"]), frame_count, endpoint=not bool(payload.get("loop")))
        images = {vi: [] for vi in views}
        measurements = {vi: [] for vi in views}
        for fi, t in enumerate(times):
            mats, _, frame_hash = joint_pose(skeleton=skeleton, tracks=tracks, time_seconds=float(t), cameras=cameras)
            matrices = np.asarray([mats[jid] for jid in joint_ids])
            posed = _skin(rest, W, joint_ids, mats)
            for vi in views:
                key = f"{label}_V{vi}_{fi:04d}"
                cached = checkpoint.read(key)
                if cached:
                    p, measurement = cached
                    im = Image.open(p).convert("RGBA")
                    print("FRAME_REUSE", key, flush=True)
                else:
                    vm, binding, blend = visual[vi]
                    camera = cameras[vi]
                    try:
                        harmonic = evaluate_domain_binding(binding, visual_faces=np.asarray(vm.faces),
                                      posed_mechanical_positions_xyz=posed, camera=camera)
                        palette = compile_connected_palette(axis_positions_source=axis,
                                      axis_parents=parents, skin_matrices_source=matrices, camera=camera)
                        proof = prove_connected_palette(palette, axis_positions_source=axis,
                                      axis_parents=parents, skin_matrices_source=matrices, camera=camera)
                        if not proof["connected_palette_relations_passed"]:
                            raise QualificationError("CONNECTED_PALETTE_RELATION_FAIL")
                        pos = apply_connected_palette(harmonic, rest_source_xy=np.asarray(vm.positions),
                                      coefficients=blend, palette=palette)
                        rgba, provenance, donor = materials[vi]
                        rr = render_visual_material(positions=pos[:, :2], depths=pos[:, 2],
                            faces=np.asarray(vm.faces), uv=np.asarray(vm.uv), texture=rgba,
                            provenance=provenance, source_view=donor, view_index=vi, resolution=resolution)
                        im = Image.fromarray(np.asarray(rr.straight_rgba_u8, dtype=np.uint8))
                        measurement = {"frame_hash": frame_hash, "time_seconds": float(t), "palette_proof": proof,
                            "geometry": base.visual_metrics(vm.positions, pos[:, :2], vm.faces),
                            "overlap_pixels": int(rr.overlap_pixel_count), "unresolved_depth_ties": int(rr.unresolved_depth_tie_count),
                            "fragment_overflow": int(rr.fragment_overflow_count)}
                        p = out / f"{key}.png"
                        im.save(p)
                        checkpoint.commit(key, p, measurement)
                        print("FRAME_RENDERED", key, "seconds", round(time.monotonic() - started, 2), flush=True)
                    except QualificationError as error:
                        report["failures"].append({"key": key, "error": str(error)})
                        atomic_json(out / "REPORT.json", report)
                        print("FRAME_FAILED", key, str(error), flush=True)
                        continue
                images[vi].append(im)
                measurements[vi].append(measurement)
        duration = max(40, round(float(payload["duration_seconds"]) * 1000 /
                       (frame_count if payload.get("loop") else frame_count - 1)))
        for vi in views:
            if len(images[vi]) != frame_count:
                continue
            p = out / f"KNIGHT_{label}_CURRENT_CAA_V6_V{vi}.gif"
            _gif(images[vi], p, duration)
            report["outputs"].append({"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size})
            print("GIF_COMPLETE", p.name, digest(p), flush=True)
        if views == tuple(range(8)) and all(len(images[vi]) == frame_count for vi in views):
            grid = [base.compose_grid([images[vi][f] for vi in views], [f"V{vi}" for vi in views]) for f in range(frame_count)]
            p = out / f"KNIGHT_{label}_CURRENT_CAA_V6_ALL_VIEWS.gif"
            _gif(grid, p, duration)
            report["outputs"].append({"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size})
        report["clips"][clip_id] = {"motion_sha256": digest(path), "mapping": mapping,
                                    "measurements": measurements, "frame_count": frame_count}
        atomic_json(out / "REPORT.json", report)
    report["status"] = "RENDER_FAILURES_MEASURED" if report["failures"] else "COMPLETE_DIAGNOSTIC_RENDER"
    report["duration_seconds"] = time.monotonic() - started
    if any(digest(Path(p)) != sha for p, sha in contract["files"].items()):
        raise RuntimeError("RENDER_INPUTS_MUTATED_DURING_EXECUTION")
    atomic_json(out / "REPORT.json", report)
    if report["failures"]:
        raise RuntimeError("RENDER_FRAME_CONTRACT_FAILED__SEE_REPORT")


if __name__ == "__main__":
    main()
