#!/usr/bin/env python3
"""Teacher-only original Cat V3 -> reference puppet predictions (NO source targets).

Uses a verified source-face/corner UV witness and RealSaS's existing CPU CAA
renderer. Input excludes physical source sample arrays by design. This is NOT
one-image student inference, original artist drawable ownership or product
activation. Run independent court separately against immutable Cycles samples.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.operational_representation_synthesis_v0 import (
    SourceGraphicsEvidenceV0, PuppetCapabilityContractV0,
    RepresentationPolicyV0, synthesize_operational_representations_v0,
)
from compiler.realsas_compiler_core.operational_puppet_runtime_v0 import (
    OperationalPuppetRuntimeV0,
)

SOURCE = "1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
TEXTURE = "2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d"
TEACHER = "c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603"
PLAN = "8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b"
CAMERA = "300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0"
POSES = (0, 14)
VIEWS = (0, 2, 4, 6)
ARMS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
RES = 192


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError("CAT_REFERENCE_PUPPET_" + reason)


@dataclass(frozen=True)
class Vertex:
    canonical_mesh_vertex_id: str
    P: tuple[float, float, float]


@dataclass(frozen=True)
class Mesh:
    vertices: tuple[Vertex, ...]
    faces: tuple[tuple[str, str, str], ...]


def _mesh(rest, faces):
    return Mesh(
        vertices=tuple(Vertex(f"teacher_vertex_{i}", tuple(map(float, point)))
                       for i, point in enumerate(rest)),
        faces=tuple(tuple(f"teacher_vertex_{int(vid)}" for vid in row)
                    for row in faces),
    )


def load_source_context(*, root: Path):
    root = Path(root)
    entries = (
        ("texture.png", TEXTURE),
        ("teacher.npz", TEACHER),
        ("plan.json", PLAN),
        ("camera.json", CAMERA),
    )
    for name, sha in entries:
        require(digest(root / name) == sha, "PRIVATE_SOURCE_PARENT_SHA_DRIFT:" + name)
    alignment_path = root / "alignment.json"
    alignment = json.loads(alignment_path.read_text())
    require(alignment.get("original_glb_sha256") == SOURCE
            and alignment.get("original_teacher_sha256") == TEACHER
            and alignment.get("source_to_teacher_face_uv_binding") ==
            "VERIFIED_NATIVE_BLENDER_FACE_ORDER_AND_RAW_GLTF_UV_MULTISET"
            and alignment.get("source_uv_comparison", {}).get("matched") is True
            and alignment.get("source_artist_owner") == "UNKNOWN"
            and alignment.get("training_authorized") is False,
            "UV_ALIGNMENT_UNQUALIFIED")
    with np.load(root / "face-uv.npz", allow_pickle=False) as f:
        uv = np.asarray(f["face_uv"], dtype=np.float64)
    uv_hash = hashlib.sha256(np.ascontiguousarray(uv.astype("<f8")).tobytes()).hexdigest()
    require(alignment.get("face_uv_array_sha256") == uv_hash,
            "UV_ARRAY_WITNESS_HASH_DRIFT")
    with np.load(root / "teacher.npz", allow_pickle=False) as f:
        rest = np.asarray(f["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(f["faces"], dtype=np.int64)
        poses = np.asarray(f["posed_vertices_source_world"], dtype=np.float64)
    require(rest.ndim == 2 and rest.shape[1] == 3 and
            faces.ndim == 2 and faces.shape[1] == 3 and
            uv.shape == (len(faces), 3, 2) and
            poses.ndim == 3 and poses.shape[1:] == rest.shape and
            len(poses) > max(POSES) and
            len(rest) > 0 and
            np.isfinite(rest).all() and np.isfinite(poses).all() and
            np.isfinite(uv).all() and
            faces.min() >= 0 and faces.max() < len(rest),
            "TEACHER_SOURCE_GEOMETRY_OR_FACE_UV_INVALID")
    source_atlas = np.asarray(Image.open(root / "texture.png").convert("RGBA"), dtype=np.uint8)
    require(source_atlas.shape == (512, 512, 4) and np.all(source_atlas[..., 3] == 255),
            "SOURCE_ATLAS_ALPHA_SCOPE_DRIFT")
    plan = json.loads((root / "plan.json").read_text())
    require(plan.get("source_original_glb_sha256") == SOURCE
            and plan.get("source_original_texture_sha256") == TEXTURE
            and plan.get("source_uv_coordinate_origin") ==
                "GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN"
            and plan.get("source_uv_to_png_y_flip_applied") is False
            and plan.get("original_source_uv_operation_count") == 3
            and len(plan["selected_original_source_uv_edits"]) == 3,
            "SOURCE_UV_OPERATION_PLAN_DRIFT")
    rectangles = tuple(
        tuple(map(int, row["source_rgba_png_pixel_rectangle_xyxy_exclusive"]))
        for row in plan["selected_original_source_uv_edits"]
    )
    camera = json.loads((root / "camera.json").read_text())
    require(len(camera.get("cameras", ())) == 8
            and all(camera["cameras"][v].get("view_index") == v
                    for v in VIEWS), "ORIGINAL_CAMERA_VIEW_INDEX_DRIFT")
    return (rest, faces, poses, uv, source_atlas, rectangles,
            camera["cameras"], uv_hash)


def compile_and_render(root: Path, output: Path):
    rest, faces, poses, uv, texture, rectangles, cameras, uv_hash = load_source_context(
        root=root)
    mesh = _mesh(rest, faces)
    evidence = SourceGraphicsEvidenceV0(
        mesh_binding_hash=content_sha256({
            "teacher_sha256": TEACHER, "source_face_count": len(faces),
            "source_uv_sha256": uv_hash, "scope": "TEACHER_SOURCE_ONLY",
        }),
        skin_binding_hash=TEACHER,  # teacher mechanics identity, NOT qualified SkinIR
        source_graphics_binding_hash=SOURCE,
        face_count=len(faces),
        input_mode="TEACHER_SOURCE_KNOWN",
    )
    contract = PuppetCapabilityContractV0(
        independent_edit_address_count=3,
        required_operations=(
            "INDEPENDENT_ATLAS_EDITS", "POSE_AWARE_SOURCE_UV",
            "LOCAL_DEPTH_SOURCE_OVER",
        ),
    )
    synthesis = synthesize_operational_representations_v0(
        evidence, contract, RepresentationPolicyV0(max_faces_per_packet=32))
    candidate = next(x for x in synthesis.candidates
                     if x.candidate_id == "addressed_aggregate")
    runtime = OperationalPuppetRuntimeV0(
        evidence=evidence, candidate=candidate, mesh=mesh, face_uv=uv,
        texture_rgba_u8=texture, edit_rectangles=rectangles,
    )
    predictions = np.empty((len(POSES), len(VIEWS), len(ARMS),
                            RES, RES, 4), dtype=np.float32)
    diagnostics = []
    for pi, pose_index in enumerate(POSES):
        xyz = poses[pose_index]
        for vi, view in enumerate(VIEWS):
            cam = dict(cameras[view])
            cam["resolution"] = RES  # Cycles original source V3 proof resolution.
            camera = qualify_camera_v3(cam, view_id=f"V{view}", view_index=view)
            for ai, control_bits in enumerate(ARMS):
                frame = runtime.render(
                    camera=camera, posed_positions=xyz,
                    control_bits=control_bits,
                )
                predictions[pi, vi, ai] = frame.premultiplied_rgba.astype(np.float32)
                diagnostics.append({
                    "pose_ordinal": pose_index, "view_index": view,
                    "source_arm_index": ai, "source_edit_control_bits": list(control_bits),
                    "reference_runtime_qualification": frame.qualification,
                    "layer_overflow_pixels": int(np.count_nonzero(frame.layer_overflow)),
                    "exact_depth_ambiguity_pixels": int(np.count_nonzero(
                        frame.exact_depth_ambiguity)),
                    "reference_geometry_visible_pixels": int(np.count_nonzero(
                        frame.owner_face_index >= 0)),
                })
            print(json.dumps({"pose": pose_index, "view": view,
                "rendered_arms": len(ARMS),
                "meaning": "REFERENCE_PUPPET_ONLY__SOURCE_PARITY_UNMEASURED"}),
                flush=True)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "PRIVATE_CAT_V3_PUPPET_REFERENCE_PREDICTIONS_V0.npz"
    np.savez_compressed(target, puppet_reference_premultiplied_linear_rgba=predictions)
    report = {
        "schema": "RealSaS.CatV3SourceTeacherPuppetReferencePredictions.v0",
        "source_original_glb_sha256": SOURCE,
        "source_original_texture_sha256": TEXTURE,
        "source_original_native_teacher_sha256": TEACHER,
        "source_original_uv_plan_sha256": PLAN,
        "source_original_camera_sha256": CAMERA,
        "source_native_face_uv_float64_sha256": uv_hash,
        "source_face_count": int(len(faces)),
        "source_vertex_count": int(len(rest)),
        "source_pose_ordinals": list(POSES),
        "source_view_indices": list(VIEWS),
        "source_edit_control_bit_arms": [list(a) for a in ARMS],
        "source_target_pixels_loaded_in_prediction": False,
        "source_original_artist_ownership": "UNKNOWN",
        "candidate_id": candidate.candidate_id,
        "candidate_hash": candidate.candidate_hash,
        "reference_renderer": "REAL_SAS_CAA_V2_DEPTH_K4__2X2_COVERAGE",
        "reference_prediction_shape": list(predictions.shape),
        "reference_prediction_sha256": digest(target),
        "frame_diagnostics": diagnostics,
        "source_physical_parity": "UNKNOWN",
        "training_authorized": False,
        "materialization_authorized": False,
        "iris_training_authorized": False,
        "tessa_training_authorized": False,
        "graphics_owner_authorized": False,
    }
    out = output / "PRIVATE_CAT_V3_PUPPET_REFERENCE_PREDICTIONS_V0.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True,
                              allow_nan=False) + "\n")
    print(json.dumps({"status": "REFERENCE_CAA_RENDER_EXECUTED__PHYSICAL_COURT_REQUIRED",
                      "predictions": len(diagnostics),
                      "prediction_sha256": report["reference_prediction_sha256"],
                      "unknown_layer_overflow_frame_count": sum(
                          x["layer_overflow_pixels"] > 0 for x in diagnostics),
                      "training_authorized": False}, sort_keys=True), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    compile_and_render(args.root, args.output)


if __name__ == "__main__":
    main()
