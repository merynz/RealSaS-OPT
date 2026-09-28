from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    RUNTIME_COVERAGE_SCALE,
    load_face_page_index,
    load_face_uv,
    load_provenance_atlas,
    render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_texture_pages,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _composite_sheet,
    _ctx,
    _skin,
    _tracks_for_clip,
)

BASELINE_COMMIT = "62aebc388605"
BASELINE_PRODUCER_BLOB = "7917be02f325ba00b5dcd2b31a786898e45f640f"
BASELINE_REPORT_BLOB = "9120723f6bc00142fb6bb992d3ee7a85c601bf4f"
BASELINE_SHEET_BLOBS = {
    "demo_idle_v1": "cae86d8a04bde6768ac9c3dafed13a2ab318cf29",
    "demo_run_v1": "8e9ed58a3bfbbace73f54e3e5593691f90eef4f1",
    "demo_slash_v1": "7a87f310473d602c99eb712be964dcd430b47cfb",
}
BASELINE_GIF_BLOBS = {
    "demo_idle_v1": "31b64edff1da106bc8feb00faede61ffe68dc89c",
    "demo_run_v1": "b8f0e97e74396a0987444bb72ea93db4bb519640",
    "demo_slash_v1": "73a567f3ff17786d3316e3fb02a723ee3c026eca",
}


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)], text=True
    ).strip()


def _json_hash(value) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hash_fields(value) -> dict:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, item in value.items():
        low = str(key).lower()
        if low.endswith("_hash") or low.endswith("_sha256") or "lineage_hash" in low:
            if isinstance(item, (str, int, float, bool)) or item is None:
                out[str(key)] = item
    return out


def _find_stage_row(value, stage_id: str):
    if isinstance(value, dict):
        if str(value.get("stage_id") or value.get("id") or "") == stage_id:
            return value
        for item in value.values():
            found = _find_stage_row(item, stage_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_stage_row(item, stage_id)
            if found is not None:
                return found
    return None


def _owner_color(owner: np.ndarray) -> np.ndarray:
    face = np.asarray(owner, dtype=np.int64)
    h, w = face.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = face >= 0
    x = face[valid].astype(np.uint64) + np.uint64(1)
    # Deterministic integer hash. Adjacent faces become visually distinct
    # without importing any appearance data.
    x ^= x >> np.uint64(16)
    x *= np.uint64(0x7FEB352D)
    x ^= x >> np.uint64(15)
    x *= np.uint64(0x846CA68B)
    x ^= x >> np.uint64(16)
    rgba[..., 0][valid] = (x & np.uint64(255)).astype(np.uint8)
    rgba[..., 1][valid] = ((x >> np.uint64(8)) & np.uint64(255)).astype(np.uint8)
    rgba[..., 2][valid] = ((x >> np.uint64(16)) & np.uint64(255)).astype(np.uint8)
    rgba[..., 3][valid] = 255
    return rgba


def _provenance_color(code: np.ndarray, visible: np.ndarray) -> np.ndarray:
    value = np.asarray(code, dtype=np.uint8)
    vis = np.asarray(visible, dtype=bool)
    rgba = np.zeros((*value.shape, 4), dtype=np.uint8)
    lut = {
        0: (40, 220, 90),    # direct source
        1: (40, 160, 240),   # other source support
        2: (245, 205, 60),
        3: (235, 80, 80),
        4: (210, 80, 225),
        255: (128, 128, 128),
    }
    for key, rgb in lut.items():
        take = vis & (value == int(key))
        rgba[..., 0][take] = rgb[0]
        rgba[..., 1][take] = rgb[1]
        rgba[..., 2][take] = rgb[2]
        rgba[..., 3][take] = 255
    return rgba


def _representative_uv(
    *,
    candidate,
    camera,
    positions: np.ndarray,
    face_uv: np.ndarray,
    face_page: np.ndarray,
):
    visibility = rasterize_visible_owner(
        candidate,
        camera,
        positions=positions,
        coverage_scale=RUNTIME_COVERAGE_SCALE,
    )
    scale = int(RUNTIME_COVERAGE_SCALE)
    hh, ww = visibility.owner_face_index.shape
    if hh % scale or ww % scale:
        raise RuntimeError("BASELINE_FORENSIC_COVERAGE_SHAPE_DRIFT")
    h, w = hh // scale, ww // scale

    def grid(value: np.ndarray) -> np.ndarray:
        a = np.asarray(value)
        tail = tuple(a.shape[2:])
        r = a.reshape(h, scale, w, scale, *tail)
        axes = (0, 2, 1, 3) + tuple(range(4, r.ndim))
        return r.transpose(axes)

    owner_samples = grid(visibility.owner_face_index).reshape(
        h, w, scale * scale
    )
    depth_samples = grid(visibility.depth).reshape(
        h, w, scale * scale
    )
    bary_samples = grid(visibility.barycentric).reshape(
        h, w, scale * scale, 3
    )
    representative = np.argmin(depth_samples, axis=2)
    owner = np.take_along_axis(
        owner_samples, representative[:, :, None], axis=2
    )[:, :, 0]
    bary = np.take_along_axis(
        bary_samples,
        representative[:, :, None, None],
        axis=2,
    )[:, :, 0, :]

    valid = owner >= 0
    uv = np.zeros((h, w, 2), dtype=np.float64)
    page = np.zeros((h, w), dtype=np.int32)
    if np.any(valid):
        oi = owner[valid].astype(np.int64)
        weights = bary[valid].astype(np.float64)
        uv[valid] = np.sum(face_uv[oi] * weights[:, :, None], axis=1)
        page[valid] = face_page[oi]

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0][valid] = np.rint(
        np.clip(uv[..., 0][valid], 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    rgba[..., 1][valid] = np.rint(
        np.clip(uv[..., 1][valid], 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    if np.any(valid):
        max_page = max(1, int(np.max(face_page)))
        rgba[..., 2][valid] = np.rint(
            np.clip(page[valid] / float(max_page), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    rgba[..., 3][valid] = 255
    return rgba, visibility


def _face_deformation_metrics(candidate, rest_projected, posed_projected):
    index_by_id = {
        str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)
    }
    face_index = np.asarray(
        [
            [index_by_id[str(vertex_id)] for vertex_id in face]
            for face in candidate.faces
        ],
        dtype=np.int64,
    )
    rest = np.asarray(rest_projected, dtype=np.float64)[face_index, :2]
    posed = np.asarray(posed_projected, dtype=np.float64)[face_index, :2]

    def signed_area(tri):
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def edge_lengths(tri):
        return np.stack(
            [
                np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
            ],
            axis=1,
        )

    rest_area = signed_area(rest)
    posed_area = signed_area(posed)
    valid = np.abs(rest_area) > 1.0e-9
    ratio = np.full((len(rest_area),), np.nan, dtype=np.float64)
    ratio[valid] = np.abs(posed_area[valid]) / np.abs(rest_area[valid])
    flips = valid & (rest_area * posed_area < 0.0)

    rest_edge = edge_lengths(rest)
    posed_edge = edge_lengths(posed)
    edge_ratio = posed_edge / np.maximum(rest_edge, 1.0e-9)
    finite_ratio = ratio[np.isfinite(ratio)]
    return {
        "valid_rest_face_count": int(np.count_nonzero(valid)),
        "flipped_face_count": int(np.count_nonzero(flips)),
        "area_ratio_p95": float(np.percentile(finite_ratio, 95.0))
        if len(finite_ratio) else None,
        "area_ratio_p99": float(np.percentile(finite_ratio, 99.0))
        if len(finite_ratio) else None,
        "area_ratio_max": float(np.max(finite_ratio))
        if len(finite_ratio) else None,
        "edge_ratio_p95": float(np.percentile(edge_ratio, 95.0)),
        "edge_ratio_p99": float(np.percentile(edge_ratio, 99.0)),
        "edge_ratio_max": float(np.max(edge_ratio)),
    }


def run(*, authority_root: Path, run_id: str, out_dir: Path, clip_ids: tuple[str, ...] | None = None):
    producer_path = Path("tools/demo/render_knight_motion_preview_v1.py")
    producer_blob = _git_blob(producer_path)
    if producer_blob != BASELINE_PRODUCER_BLOB:
        raise RuntimeError(
            "BASELINE_FORENSIC_PRODUCER_DRIFT:"
            + producer_blob
        )

    ctx = _ctx(authority_root, run_id)

    candidate_payload = stage_output_payload(
        ctx,
        "18_CANONICAL_MESH_ADDRESSING_BUILD",
        "RealSaS.CanonicalMeshCandidateIR.v1",
    )
    skeleton_payload = stage_output_payload(
        ctx,
        "28_SKELETON_QUALIFIED",
        "RealSaS.QualifiedSkeletonIR.v1",
    )
    skin_payload = stage_output_payload(
        ctx,
        "32_SKIN_QUALIFIED",
        "RealSaS.QualifiedSkinIR.v1",
    )
    camera_payload = stage_output_payload(
        ctx,
        "05_CAMERA_CONTRACT_SOLVED",
        "RealSaS.QualifiedCameraSetIR.v1",
    )
    appearance_payload = stage_output_payload(
        ctx,
        "23_COMPLETE_APPEARANCE_ASSET_BAKED",
        "RealSaS.CompleteAppearanceAssetIR.v2",
    )

    candidate = canonical_mesh_candidate_from_dict(candidate_payload)
    skeleton = qualified_skeleton_from_dict(skeleton_payload)
    skin = qualified_skin_from_dict(skin_payload)
    camera_set = qualified_camera_set_from_dict(camera_payload)
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    appearance = complete_appearance_asset_from_dict(appearance_payload)

    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {
        int(row.direction_index): row for row in appearance.textures
    }

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)

    stage35_row = _find_stage_row(
        ctx["ledger"], "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "RealSaS.KnightBaselineForensicRenderCapsule.v1",
        "status": "MEASURED_BASELINE_FORENSIC",
        "baseline": {
            "commit": BASELINE_COMMIT,
            "producer_blob": BASELINE_PRODUCER_BLOB,
            "report_blob": BASELINE_REPORT_BLOB,
            "sheet_blobs": BASELINE_SHEET_BLOBS,
            "gif_blobs": BASELINE_GIF_BLOBS,
        },
        "execution": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "diagnostic_script_blob": _git_blob(Path(__file__)),
            "producer_blob": producer_blob,
            "run_id": run_id,
        },
        "authority": {
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "skeleton_lineage_hash": getattr(
                skeleton, "skeleton_lineage_hash", None
            ),
            "skin_lineage_hash": getattr(skin, "skin_lineage_hash", None),
            "camera_set_hash": getattr(camera_set, "camera_set_hash", None),
            "stage18_payload_sha256": _json_hash(candidate_payload),
            "stage28_payload_sha256": _json_hash(skeleton_payload),
            "stage32_payload_sha256": _json_hash(skin_payload),
            "stage05_payload_sha256": _json_hash(camera_payload),
            "stage23_payload_sha256": _json_hash(appearance_payload),
            "stage18_hash_fields": _hash_fields(candidate_payload),
            "stage28_hash_fields": _hash_fields(skeleton_payload),
            "stage32_hash_fields": _hash_fields(skin_payload),
            "stage05_hash_fields": _hash_fields(camera_payload),
            "stage23_hash_fields": _hash_fields(appearance_payload),
            "stage35_ledger_row": stage35_row,
        },
        "contract": {
            "mechanical_carrier": "STAGE18_CANONICAL_MESH_CANDIDATE",
            "skeleton_authority": "STAGE28_QUALIFIED_SKELETON",
            "skin_authority": "STAGE32_QUALIFIED_SKIN",
            "appearance_authority": "STAGE23_COMPLETE_APPEARANCE",
            "retarget": "NONINJECTIVE_REST_GEOMETRY_OBJECT_FRAME_REBASE_V1",
            "views": [0, 2],
            "samples_per_view": 4,
        },
        "clips": [],
    }

    clips = [
        ("demo_idle_v1", "idle"),
        ("demo_run_v1", "run"),
        ("demo_slash_v1", "slash"),
    ]
    if clip_ids:
        requested = set(map(str, clip_ids))
        clips = [row for row in clips if row[0] in requested]
        missing = requested - {row[0] for row in clips}
        if missing:
            raise RuntimeError("BASELINE_FORENSIC_UNKNOWN_CLIP:" + ",".join(sorted(missing)))
    if not clips:
        raise RuntimeError("BASELINE_FORENSIC_NO_CLIPS")

    for clip_id, short in clips:
        path = (
            ctx["run_root"]
            / "inputs"
            / "motion"
            / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(path.read_text())
        tracks, mapping_details = _tracks_for_clip(
            payload, skeleton, cameras, source_report
        )
        duration = float(payload["duration_seconds"])
        sample_times = np.linspace(
            0.0,
            duration,
            4,
            endpoint=not bool(payload.get("loop")),
        )

        actual_frames = []
        owner_frames = []
        uv_frames = []
        provenance_frames = []
        metrics = []

        rest_visibility_by_view = {
            view: rasterize_visible_owner(
                candidate,
                cameras[view],
                positions=rest,
                coverage_scale=1,
            )
            for view in (0, 2)
        }

        for view in (0, 2):
            texture = _load_texture_pages(texture_by_view[view])
            for frame_index, time_seconds in enumerate(sample_times):
                skin_mats, joint_pos, frame_hash = _joint_pose_v2(
                    skeleton=skeleton,
                    tracks=tracks,
                    time_seconds=float(time_seconds),
                    cameras=cameras,
                )
                posed = _skin(rest, W, joint_ids, skin_mats)
                actual = render_caa_reference(
                    mesh=candidate,
                    camera=cameras[view],
                    face_uv=face_uv,
                    texture_rgba_u8=texture,
                    provenance_atlas=provenance[view],
                    face_page_index=face_page,
                    positions=posed,
                )
                uv_rgba, visibility_hi = _representative_uv(
                    candidate=candidate,
                    camera=cameras[view],
                    positions=posed,
                    face_uv=face_uv,
                    face_page=face_page,
                )
                owner_rgba = _owner_color(actual.owner_face_index)
                prov_rgba = _provenance_color(
                    actual.provenance_code,
                    actual.geometry_visible,
                )

                title = f"{short} V{view} t={time_seconds:.2f}"
                actual_frames.append(
                    (
                        title,
                        Image.fromarray(
                            np.asarray(actual.straight_rgba_u8, dtype=np.uint8),
                            mode="RGBA",
                        ),
                    )
                )
                owner_frames.append(
                    (title, Image.fromarray(owner_rgba, mode="RGBA"))
                )
                uv_frames.append(
                    (title, Image.fromarray(uv_rgba, mode="RGBA"))
                )
                provenance_frames.append(
                    (title, Image.fromarray(prov_rgba, mode="RGBA"))
                )

                deformation = _face_deformation_metrics(
                    candidate,
                    rest_visibility_by_view[view].projected_vertices,
                    visibility_hi.projected_vertices,
                )
                metrics.append(
                    {
                        "view_index": int(view),
                        "frame_index": int(frame_index),
                        "time_seconds": float(time_seconds),
                        "motion_frame_hash": frame_hash,
                        "joint_position_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                np.asarray(
                                    [
                                        joint_pos[key]
                                        for key in sorted(joint_pos)
                                    ],
                                    dtype="<f8",
                                )
                            ).tobytes()
                        ).hexdigest(),
                        "posed_vertex_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                np.asarray(posed, dtype="<f8")
                            ).tobytes()
                        ).hexdigest(),
                        "geometry_visible_pixel_count": int(
                            np.count_nonzero(actual.geometry_visible)
                        ),
                        "final_alpha_pixel_count": int(
                            np.count_nonzero(actual.final_alpha)
                        ),
                        "undefined_visible_pixel_count": int(
                            np.count_nonzero(
                                np.asarray(actual.geometry_visible)
                                & (np.asarray(actual.provenance_code) == 255)
                            )
                        ),
                        "deformation": deformation,
                    }
                )

        actual_sheet = _composite_sheet(
            actual_frames, columns=4, label=short
        )
        owner_sheet = _composite_sheet(
            owner_frames, columns=4, label=short + "_owner"
        )
        uv_sheet = _composite_sheet(
            uv_frames, columns=4, label=short + "_uv"
        )
        provenance_sheet = _composite_sheet(
            provenance_frames, columns=4, label=short + "_provenance"
        )

        actual_path = out_dir / f"KNIGHT_{short.upper()}_BASELINE_ACTUAL_RGB_V1.png"
        owner_path = out_dir / f"KNIGHT_{short.upper()}_BASELINE_GEOMETRY_OWNER_V1.png"
        uv_path = out_dir / f"KNIGHT_{short.upper()}_BASELINE_UV_V1.png"
        prov_path = out_dir / f"KNIGHT_{short.upper()}_BASELINE_PROVENANCE_V1.png"
        actual_sheet.save(actual_path)
        owner_sheet.save(owner_path)
        uv_sheet.save(uv_path)
        provenance_sheet.save(prov_path)

        actual_blob = _git_blob(actual_path)
        expected_blob = BASELINE_SHEET_BLOBS[clip_id]
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"BASELINE_FORENSIC_ACTUAL_RGB_DRIFT:{clip_id}:"
                f"{actual_blob}:{expected_blob}"
            )

        report["clips"].append(
            {
                "clip_id": clip_id,
                "motion_json_sha256": _sha256_bytes(path),
                "mapping_sha256": _json_hash(mapping_details),
                "mapping": mapping_details,
                "actual_rgb_path": str(actual_path),
                "actual_rgb_blob": actual_blob,
                "expected_baseline_blob": expected_blob,
                "geometry_owner_path": str(owner_path),
                "geometry_owner_blob": _git_blob(owner_path),
                "uv_path": str(uv_path),
                "uv_blob": _git_blob(uv_path),
                "provenance_path": str(prov_path),
                "provenance_blob": _git_blob(prov_path),
                "frame_metrics": metrics,
            }
        )

    report_path = out_dir / "BASELINE_FORENSIC_CAPSULE_V1.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "KNIGHT_BASELINE_FORENSIC_CAPSULE_PASS",
        json.dumps(
            {
                "candidate_lineage_hash": candidate.candidate_lineage_hash,
                "producer_blob": producer_blob,
                "report_path": str(report_path),
                "stage35_status": (
                    stage35_row.get("status")
                    if isinstance(stage35_row, dict)
                    else None
                ),
            },
            sort_keys=True,
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--clip-id", action="append", default=[])
    a = p.parse_args()
    run(
        authority_root=Path(a.authority_root).expanduser().resolve(),
        run_id=a.run_id,
        out_dir=Path(a.out_dir),
        clip_ids=tuple(a.clip_id) if a.clip_id else None,
    )


if __name__ == "__main__":
    main()
