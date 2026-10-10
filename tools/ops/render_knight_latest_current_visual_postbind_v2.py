from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.joint_frames_v2 import (
    derive_joint_frames_post_bind_v2,
    frame_set_hash_v2,
)
from compiler.realsas_compiler_core import motion_dynamic_proof_v2 as motion_proof
from tools.demo import render_knight_motion_preview_v1 as motion_preview
from tools.ops import render_knight_latest_current_visual_v1 as renderer

CAA_RUN = "KNIGHT_LATEST_CURRENT_CAA_VISUAL_20261010"
MAT_RUN = "KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010"
SKELETON_NAME = "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_QUALIFIED_SKELETON_V541.json"
F6_OPERATOR = "REGION_COHERENT_MECHANICAL_SAMPLE_SIMILARITY_V1"


def _load_stage_payload(ledger: dict, stage_id: str, schema: str) -> dict:
    row = next(x for x in ledger["stages"] if x["id"] == stage_id)
    out = next(x for x in row.get("outputs") or () if x.get("schema") == schema)
    return json.loads(Path(out["path"]).read_text())


def _region_similarity(rest: np.ndarray, target: np.ndarray, region: np.ndarray) -> np.ndarray:
    out = np.empty_like(target)
    for rid in sorted(set(map(int, region.tolist()))):
        ids = np.flatnonzero(region == rid)
        x = np.asarray(rest[ids], dtype=np.float64)
        y = np.asarray(target[ids], dtype=np.float64)
        if len(x) == 0:
            raise RuntimeError("F6_REGION_EMPTY")
        xm = np.mean(x, axis=0)
        ym = np.mean(y, axis=0)
        xc = x - xm
        yc = y - ym
        denom = float(np.sum(xc * xc))
        if len(x) < 2 or denom <= 1.0e-12:
            pred = x + (ym - xm)
        else:
            H = xc.T @ yc
            U, s, Vt = np.linalg.svd(H)
            R = U @ Vt
            if float(np.linalg.det(R)) < 0.0:
                U[:, -1] *= -1.0
                R = U @ Vt
                s[-1] *= -1.0
            scale = max(1.0e-9, float(np.sum(s) / denom))
            linear = scale * R
            if float(np.linalg.det(linear)) <= 0.0:
                raise RuntimeError("F6_REGION_SIMILARITY_ORIENTATION_INVALID")
            pred = xc @ linear + ym
        out[ids] = pred
    return out


def main() -> None:
    authority_root = Path(os.environ["AUTHORITY_ROOT"]).resolve()
    input_root = Path(os.environ["INPUT_ROOT"]).resolve()

    ledger = json.loads(
        (authority_root / "runs" / CAA_RUN / "ACTIVE_RUN_V2.json").read_text()
    )
    camera_set = qualified_camera_set_from_dict(
        _load_stage_payload(
            ledger,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda row: int(row.view_index)))
    skeleton = qualified_skeleton_from_dict(
        json.loads((input_root / SKELETON_NAME).read_text())
    )
    carrier_skin = qualified_carrier_skin_from_dict(
        json.loads(
            (
                authority_root
                / "runs"
                / MAT_RUN
                / "qualified_carrier_skin.json"
            ).read_text()
        )
    )

    frames, qualification = derive_joint_frames_post_bind_v2(
        skeleton,
        carrier_skin=carrier_skin,
        cameras=cameras,
    )
    qualification_hash = content_sha256(qualification)
    frame_hash = frame_set_hash_v2(
        frames,
        carrier_skin_lineage_hash=carrier_skin.skin_lineage_hash,
    )
    print(
        "POST_BIND_JOINT_FRAME_QUALIFICATION",
        json.dumps(
            {
                "status": qualification["status"],
                "qualification_content_hash": qualification_hash,
                "frame_set_hash": frame_hash,
                "coincident_edge_count": qualification["coincident_edge_count"],
                "invented_epsilon": qualification["invented_epsilon"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def _canonical_target_frames(_skeleton, *, cameras):
        if str(_skeleton.skeleton_lineage_hash) != str(skeleton.skeleton_lineage_hash):
            raise RuntimeError("POST_BIND_FRAME_SKELETON_DRIFT")
        return frames

    motion_preview.derive_joint_frames_from_skeleton = _canonical_target_frames
    motion_proof.derive_joint_frames_from_skeleton = _canonical_target_frames

    original_bind = renderer.bind_region_visual_vertices_to_mechanical_affine_v1
    bind_call_index = 0
    presentation_by_view: dict[int, dict[str, np.ndarray]] = {}

    def _capture_binding(**kwargs):
        nonlocal bind_call_index
        binding = original_bind(**kwargs)
        vi = bind_call_index
        bind_call_index += 1
        presentation_by_view[vi] = {
            "rest": np.asarray(kwargs["points_source_xy"], dtype=np.float64).copy(),
            "region": np.asarray(kwargs["vertex_region_id"], dtype=np.int32).copy(),
        }
        return binding

    original_eval = renderer.evaluate_region_visual_binding_v1
    eval_call_index = 0

    def _coherent_eval(*args, **kwargs):
        nonlocal eval_call_index
        sampled = np.asarray(original_eval(*args, **kwargs), dtype=np.float64)
        current = eval_call_index
        eval_call_index += 1
        if current < 8:
            # Exact rest qualification remains owned by the original mechanical
            # sample binding. No presentation motion operator is applied at rest.
            return sampled
        vi = (current - 8) % 8
        presentation = presentation_by_view.get(vi)
        if presentation is None:
            raise RuntimeError("F6_PRESENTATION_VIEW_NOT_CAPTURED")
        return _region_similarity(
            presentation["rest"], sampled, presentation["region"]
        )

    renderer.bind_region_visual_vertices_to_mechanical_affine_v1 = _capture_binding
    renderer.evaluate_region_visual_binding_v1 = _coherent_eval
    renderer.main()

    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    report_path = out_dir / "REPORT.json"
    report = json.loads(report_path.read_text())
    worst_flip = 0
    worst_edge = 0.0
    for clip in report.get("clips", {}).values():
        for row in clip.get("qa", {}).values():
            worst_flip = max(worst_flip, int(row["maximum_flipped_triangles"]))
            worst_edge = max(worst_edge, float(row["maximum_edge_ratio"]))
    report["visual_deformation_operator"] = F6_OPERATOR
    report["f6_micro_court_run_id"] = 38041435324
    report["f6_dynamic_integrity"] = {
        "maximum_flipped_triangles": worst_flip,
        "maximum_edge_ratio": worst_edge,
        "required_maximum_flipped_triangles": 0,
        "required_maximum_edge_ratio": 4.0,
        "pass": bool(worst_flip == 0 and worst_edge <= 4.0),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("F6_FULL_RENDER_GATE", json.dumps(report["f6_dynamic_integrity"], sort_keys=True), flush=True)
    if worst_flip != 0:
        raise RuntimeError(f"F6_FULL_RENDER_FLIP_FAIL::{worst_flip}")
    if worst_edge > 4.0:
        raise RuntimeError(f"F6_FULL_RENDER_EDGE_FAIL::{worst_edge}")


if __name__ == "__main__":
    main()
