from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
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
from compiler.realsas_compiler_core.product_authority_v1 import (
    canonical_mesh_candidate_lineage_hash,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
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

CLIPS = (
    ("demo_idle_v1", "idle", 833),
    ("demo_run_v1", "run", 208),
    ("demo_slash_v1", "slash", 278),
)
VIEWS = (0, 2)
EDGE_LIMIT = 4.0
AREA_MAX = 20.0
AREA_MIN = 0.05
COND_MAX = 16.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(a: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def _faces(candidate) -> np.ndarray:
    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    if len(vi) != len(candidate.vertices):
        raise RuntimeError("ORACLE_VERTEX_ID_DUPLICATE")
    out = []
    for face in candidate.faces:
        if len(face) != 3 or any(str(v) not in vi for v in face):
            raise RuntimeError("ORACLE_FACE_INVALID")
        out.append(tuple(vi[str(v)] for v in face))
    return np.asarray(out, dtype=np.int64)


def _identity_surface_ids(candidate) -> tuple[str, ...]:
    out = []
    for vertex in candidate.vertices:
        coeffs = tuple(vertex.support_binding.coefficients)
        if len(coeffs) != 1:
            raise RuntimeError("ORACLE_REQUIRES_IDENTITY_SURFACE_BINDING")
        sid, coeff = coeffs[0]
        if abs(float(coeff) - 1.0) > 1e-9:
            raise RuntimeError("ORACLE_REQUIRES_UNIT_SURFACE_BINDING")
        out.append(str(sid))
    return tuple(out)


def _source_category(ta: int, tb: int, source_face_sets) -> str:
    if ta == tb:
        return "SAME_FACE"
    common = len(source_face_sets[ta] & source_face_sets[tb])
    if common == 2:
        return "SHARE_EDGE"
    if common == 1:
        return "SHARE_VERTEX_ONLY"
    return "NONINCIDENT"


def _oracle_face_mask(
    candidate,
    *,
    bank_surface_ids: tuple[str, ...],
    source_triangle_index: np.ndarray,
    source_faces: np.ndarray,
):
    candidate_surface_ids = _identity_surface_ids(candidate)
    bank_row = {sid: i for i, sid in enumerate(bank_surface_ids)}
    if len(bank_row) != len(bank_surface_ids):
        raise RuntimeError("ORACLE_BANK_SURFACE_ID_DUPLICATE")
    source_face_sets = [set(map(int, row)) for row in source_faces.tolist()]
    tri_by_vertex = []
    for sid in candidate_surface_ids:
        if sid not in bank_row:
            raise RuntimeError(f"ORACLE_SURFACE_ID_NOT_IN_TEACHER_BANK:{sid}")
        ti = int(source_triangle_index[bank_row[sid]])
        if ti < 0 or ti >= len(source_face_sets):
            raise RuntimeError("ORACLE_SOURCE_TRIANGLE_INDEX_INVALID")
        tri_by_vertex.append(ti)

    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    keep = np.ones((len(candidate.faces),), dtype=bool)
    categories = {
        "SAME_FACE": 0,
        "SHARE_EDGE": 0,
        "SHARE_VERTEX_ONLY": 0,
        "NONINCIDENT": 0,
    }
    face_rows = []
    for fi, face in enumerate(candidate.faces):
        idx = tuple(vi[str(x)] for x in face)
        tris = tuple(tri_by_vertex[i] for i in idx)
        cats = (
            _source_category(tris[0], tris[1], source_face_sets),
            _source_category(tris[1], tris[2], source_face_sets),
            _source_category(tris[2], tris[0], source_face_sets),
        )
        for c in cats:
            categories[c] += 1
        bad = any(c == "NONINCIDENT" for c in cats)
        keep[fi] = not bad
        if bad and len(face_rows) < 256:
            face_rows.append(
                {
                    "face_index": int(fi),
                    "vertex_ids": list(map(str, face)),
                    "source_triangle_indices": list(map(int, tris)),
                    "pair_categories": list(cats),
                }
            )
    return keep, categories, face_rows


def _subset_candidate(candidate, keep: np.ndarray):
    keep_indices = np.nonzero(np.asarray(keep, dtype=bool))[0]
    faces = tuple(candidate.faces[int(i)] for i in keep_indices.tolist())
    edges = sorted(
        {
            tuple(sorted((str(face[a]), str(face[b]))))
            for face in faces
            for a, b in ((0, 1), (1, 2), (2, 0))
        }
    )
    provisional = replace(
        candidate,
        faces=faces,
        edges=tuple(edges),
        producer_id="RealSaS.TeacherTopologyOracleNonincidentCut.v1",
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "diagnostic_only": True,
            "teacher_topology_inference_authority": False,
            "oracle_rule": "DROP_FACE_IF_ANY_VERTEX_PAIR_SOURCE_TRIANGLES_NONINCIDENT",
            "source_candidate_lineage_hash": candidate.candidate_lineage_hash,
            "removed_face_count": int(len(candidate.faces) - len(faces)),
            "vertex_position_mutation": False,
            "skin_weight_mutation": False,
        },
    )
    repaired = replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return repaired, keep_indices


def _triangle_metrics(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray) -> dict:
    r = rest[faces]
    p = posed[faces]

    re = np.stack(
        (
            np.linalg.norm(r[:, 1] - r[:, 0], axis=1),
            np.linalg.norm(r[:, 2] - r[:, 1], axis=1),
            np.linalg.norm(r[:, 0] - r[:, 2], axis=1),
        ),
        axis=1,
    )
    pe = np.stack(
        (
            np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
        ),
        axis=1,
    )
    edge = pe / np.maximum(re, 1e-12)
    edge_max = edge.max(axis=1)

    r1 = r[:, 1] - r[:, 0]
    r2 = r[:, 2] - r[:, 0]
    p1 = p[:, 1] - p[:, 0]
    p2 = p[:, 2] - p[:, 0]
    rn = np.cross(r1, r2)
    pn = np.cross(p1, p2)
    rn_norm = np.linalg.norm(rn, axis=1)
    pn_norm = np.linalg.norm(pn, axis=1)
    orientation_dot = np.sum(rn * pn, axis=1)
    degenerate = (rn_norm <= 1e-12) | (pn_norm <= 1e-12)
    flipped = (~degenerate) & (orientation_dot < 0.0)

    l1 = np.linalg.norm(r1, axis=1)
    u = r1 / np.maximum(l1[:, None], 1e-12)
    x2 = np.sum(r2 * u, axis=1)
    perp = r2 - x2[:, None] * u
    y2 = np.linalg.norm(perp, axis=1)
    inv = np.zeros((len(faces), 2, 2), dtype=np.float64)
    inv[:, 0, 0] = 1.0 / np.maximum(l1, 1e-12)
    inv[:, 0, 1] = -x2 / np.maximum(l1 * y2, 1e-12)
    inv[:, 1, 1] = 1.0 / np.maximum(y2, 1e-12)
    pedges = np.stack((p1, p2), axis=2)
    F = np.einsum("nij,njk->nik", pedges, inv)
    singular = np.linalg.svd(F, compute_uv=False)
    smax = singular[:, 0]
    smin = singular[:, 1]
    area = smax * smin
    condition = smax / np.maximum(smin, 1e-15)

    catastrophic = (
        flipped
        | degenerate
        | (edge_max > EDGE_LIMIT)
        | (area > AREA_MAX)
        | (area < AREA_MIN)
        | (condition > COND_MAX)
    )
    return {
        "edge_max": edge_max,
        "area": area,
        "condition": condition,
        "flipped": flipped,
        "degenerate": degenerate,
        "catastrophic": catastrophic,
    }


def _q(a: np.ndarray, q: float) -> float:
    x = np.asarray(a, dtype=np.float64)
    return float(np.quantile(x, q)) if len(x) else 0.0


def _summary(metrics: dict) -> dict:
    edge = metrics["edge_max"]
    area = metrics["area"]
    cond = metrics["condition"]
    return {
        "face_count": int(len(edge)),
        "flipped_face_count": int(np.count_nonzero(metrics["flipped"])),
        "degenerate_face_count": int(np.count_nonzero(metrics["degenerate"])),
        "catastrophic_face_count": int(np.count_nonzero(metrics["catastrophic"])),
        "edge_ratio_p95": _q(edge, 0.95),
        "edge_ratio_p99": _q(edge, 0.99),
        "edge_ratio_max": float(np.max(edge)) if len(edge) else 0.0,
        "area_ratio_p95": _q(area, 0.95),
        "area_ratio_p99": _q(area, 0.99),
        "area_ratio_max": float(np.max(area)) if len(area) else 0.0,
        "area_ratio_min": float(np.min(area)) if len(area) else 0.0,
        "condition_p95": _q(cond, 0.95),
        "condition_p99": _q(cond, 0.99),
        "condition_max": float(np.max(cond)) if len(cond) else 0.0,
    }


def _catastrophic_pixels(render, catastrophic: np.ndarray) -> int:
    owner = np.asarray(render.owner_face_index, dtype=np.int64)
    alpha = np.asarray(render.final_alpha, dtype=bool)
    valid = alpha & (owner >= 0) & (owner < len(catastrophic))
    if not np.any(valid):
        return 0
    return int(np.count_nonzero(catastrophic[owner[valid]]))


def _pair_image(base: Image.Image, oracle: Image.Image, title: str) -> Image.Image:
    if base.size != oracle.size:
        raise RuntimeError("ORACLE_COMPARE_IMAGE_SIZE_DRIFT")
    w, h = base.size
    header = 26
    out = Image.new("RGBA", (2 * w, h + header), (0, 0, 0, 0))
    out.alpha_composite(base, (0, header))
    out.alpha_composite(oracle, (w, header))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), f"BASELINE | ORACLE NONINCIDENT CUT | {title}", fill=(255, 255, 255, 255))
    return out


def _save_gif(frames, path: Path, duration_ms: int):
    pal = []
    for frame in frames:
        img = frame
        if img.height > 640:
            nh = 640
            nw = max(1, round(img.width * nh / img.height))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        pal.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=255))
    pal[0].save(
        path,
        save_all=True,
        append_images=pal[1:],
        duration=int(duration_ms),
        loop=0,
        disposal=2,
        optimize=False,
        transparency=0,
    )


def _reduction(before: float, after: float) -> float:
    if before <= 0.0:
        return 0.0
    return float(1.0 - after / before)


def run(
    authority_root: Path,
    run_id: str,
    teacher_bank_path: Path,
    teacher_source_path: Path,
    out_dir: Path,
):
    ctx = _ctx(authority_root, run_id)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {int(row.direction_index): row for row in appearance.textures}

    with np.load(teacher_bank_path, allow_pickle=False) as z:
        bank_surface_ids = tuple(map(str, z["surface_ids"].tolist()))
        source_triangle_index = np.asarray(z["source_triangle_index"], dtype=np.int64)
    with np.load(teacher_source_path, allow_pickle=False) as z:
        source_faces = np.asarray(z["faces"], dtype=np.int64)

    keep, pair_category_counts, removed_examples = _oracle_face_mask(
        candidate,
        bank_surface_ids=bank_surface_ids,
        source_triangle_index=source_triangle_index,
        source_faces=source_faces,
    )
    oracle_candidate, keep_indices = _subset_candidate(candidate, keep)
    baseline_faces = _faces(candidate)
    oracle_faces = _faces(oracle_candidate)

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    baseline_alpha_total = 0
    oracle_alpha_total = 0
    baseline_cat_pixel_total = 0
    oracle_cat_pixel_total = 0
    baseline_flip_counts = []
    oracle_flip_counts = []
    baseline_edge_p95 = []
    oracle_edge_p95 = []
    motion_hashes = {}
    preview_rows = []

    for clip_id, short, duration_ms in CLIPS:
        motion_path = (
            ctx["run_root"]
            / "inputs"
            / "motion"
            / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        motion_hashes[clip_id] = _sha256(motion_path)
        payload = json.loads(motion_path.read_text())
        tracks, mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
        times = np.linspace(
            0.0,
            float(payload["duration_seconds"]),
            4,
            endpoint=not bool(payload.get("loop")),
        )
        clip_images = []
        gif_frames = []

        for frame_index, t in enumerate(times):
            mats, _, _ = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(t),
                cameras=cameras,
            )
            posed = _skin(rest, W, joint_ids, mats)
            baseline_metrics = _triangle_metrics(rest, posed, baseline_faces)
            oracle_metrics = _triangle_metrics(rest, posed, oracle_faces)
            bs = _summary(baseline_metrics)
            os = _summary(oracle_metrics)
            baseline_flip_counts.append(bs["flipped_face_count"])
            oracle_flip_counts.append(os["flipped_face_count"])
            baseline_edge_p95.append(bs["edge_ratio_p95"])
            oracle_edge_p95.append(os["edge_ratio_p95"])

            render_rows = []
            for view in VIEWS:
                texture = _load_texture_pages(texture_by_view[view])
                base_render = render_caa_reference(
                    mesh=candidate,
                    camera=cameras[view],
                    face_uv=face_uv,
                    texture_rgba_u8=texture,
                    provenance_atlas=provenance[view],
                    face_page_index=face_page,
                    positions=posed,
                )
                oracle_render = render_caa_reference(
                    mesh=oracle_candidate,
                    camera=cameras[view],
                    face_uv=face_uv[keep_indices],
                    texture_rgba_u8=texture,
                    provenance_atlas=provenance[view],
                    face_page_index=face_page[keep_indices],
                    positions=posed,
                )
                base_alpha = int(np.count_nonzero(base_render.final_alpha))
                oracle_alpha = int(np.count_nonzero(oracle_render.final_alpha))
                base_cat = _catastrophic_pixels(base_render, baseline_metrics["catastrophic"])
                oracle_cat = _catastrophic_pixels(oracle_render, oracle_metrics["catastrophic"])
                baseline_alpha_total += base_alpha
                oracle_alpha_total += oracle_alpha
                baseline_cat_pixel_total += base_cat
                oracle_cat_pixel_total += oracle_cat
                render_rows.append(
                    {
                        "view_index": int(view),
                        "baseline_final_alpha_pixels": base_alpha,
                        "oracle_final_alpha_pixels": oracle_alpha,
                        "baseline_catastrophic_owned_pixels": base_cat,
                        "oracle_catastrophic_owned_pixels": oracle_cat,
                    }
                )
                base_img = Image.fromarray(
                    np.asarray(base_render.straight_rgba_u8, dtype=np.uint8),
                    "RGBA",
                )
                oracle_img = Image.fromarray(
                    np.asarray(oracle_render.straight_rgba_u8, dtype=np.uint8),
                    "RGBA",
                )
                pair = _pair_image(
                    base_img,
                    oracle_img,
                    f"{short} V{view} t={float(t):.3f}",
                )
                clip_images.append(
                    (
                        f"{short} V{view} t={float(t):.3f}",
                        pair,
                    )
                )
                if view == 2:
                    gif_frames.append(pair)

            samples.append(
                {
                    "clip_id": clip_id,
                    "frame_index": int(frame_index),
                    "time_seconds": float(t),
                    "baseline": bs,
                    "oracle": os,
                    "renders": render_rows,
                }
            )

        sheet = _composite_sheet(clip_images, columns=2, label=short)
        sheet_path = out_dir / f"KNIGHT_{short.upper()}_ORACLE_TOPOLOGY_COMPARE_V1.png"
        gif_path = out_dir / f"KNIGHT_{short.upper()}_ORACLE_TOPOLOGY_COMPARE_V1.gif"
        sheet.save(sheet_path)
        _save_gif(gif_frames, gif_path, duration_ms)
        preview_rows.append(
            {
                "clip_id": clip_id,
                "sheet_path": str(sheet_path),
                "gif_path": str(gif_path),
                "mapping": mapping,
            }
        )

    alpha_retention = float(oracle_alpha_total / max(baseline_alpha_total, 1))
    catastrophic_pixel_reduction = _reduction(
        float(baseline_cat_pixel_total),
        float(oracle_cat_pixel_total),
    )
    baseline_median_flip = float(np.median(np.asarray(baseline_flip_counts, dtype=np.float64)))
    oracle_median_flip = float(np.median(np.asarray(oracle_flip_counts, dtype=np.float64)))
    median_flip_reduction = _reduction(baseline_median_flip, oracle_median_flip)
    baseline_max_edge_p95 = float(max(baseline_edge_p95))
    oracle_max_edge_p95 = float(max(oracle_edge_p95))
    max_edge_p95_reduction = _reduction(
        baseline_max_edge_p95,
        oracle_max_edge_p95,
    )

    dominant = bool(
        alpha_retention >= 0.70
        and catastrophic_pixel_reduction >= 0.75
        and median_flip_reduction >= 0.50
        and max_edge_p95_reduction >= 0.50
    )
    material = bool(
        alpha_retention >= 0.70
        and catastrophic_pixel_reduction >= 0.50
        and (
            median_flip_reduction >= 0.25
            or max_edge_p95_reduction >= 0.25
        )
    )
    verdict = (
        "DOMINANT_TOPOLOGY_OWNER_CONFIRMED"
        if dominant
        else "MATERIAL_TOPOLOGY_OWNER_CONFIRMED"
        if material
        else "INCONCLUSIVE_OR_NOT_OWNER"
    )

    report = {
        "schema": "RealSaS.KnightOracleTopologyRepairCausality.v1",
        "status": "DIAGNOSTIC_ORACLE_ONLY__NO_PRODUCT_REPAIR",
        "run_id": run_id,
        "preregistration": "canonical/KNIGHT_ORACLE_TOPOLOGY_REPAIR_CAUSAL_PREREG_V1_20260928.json",
        "authorities": {
            "source_candidate_lineage_hash": candidate.candidate_lineage_hash,
            "oracle_candidate_lineage_hash": oracle_candidate.candidate_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "rest_vertex_positions_sha256": _array_sha256(rest),
            "skin_weight_matrix_sha256": _array_sha256(W),
            "teacher_bank_sha256": _sha256(teacher_bank_path),
            "teacher_source_sha256": _sha256(teacher_source_path),
            "motion_input_sha256": motion_hashes,
        },
        "intervention": {
            "teacher_topology_shipping_authority": False,
            "rule": "DROP_FACE_IF_ANY_VERTEX_PAIR_SOURCE_TRIANGLES_NONINCIDENT",
            "face_count_before": int(len(candidate.faces)),
            "face_count_after": int(len(oracle_candidate.faces)),
            "removed_face_count": int(len(candidate.faces) - len(oracle_candidate.faces)),
            "removed_face_fraction": float(
                (len(candidate.faces) - len(oracle_candidate.faces))
                / max(len(candidate.faces), 1)
            ),
            "vertex_count_unchanged": bool(
                len(candidate.vertices) == len(oracle_candidate.vertices)
            ),
            "vertex_position_mutation": False,
            "skin_weight_mutation": False,
            "skeleton_mutation": False,
            "motion_mutation": False,
            "appearance_pixel_mutation": False,
            "source_pair_category_counts": pair_category_counts,
            "removed_face_examples": removed_examples,
            "known_smear_face_253_removed": bool(
                len(keep) > 253 and not bool(keep[253])
            ),
        },
        "aggregate": {
            "baseline_final_alpha_pixels": int(baseline_alpha_total),
            "oracle_final_alpha_pixels": int(oracle_alpha_total),
            "oracle_final_alpha_retention": alpha_retention,
            "baseline_catastrophic_owned_pixels": int(baseline_cat_pixel_total),
            "oracle_catastrophic_owned_pixels": int(oracle_cat_pixel_total),
            "catastrophic_owned_pixel_reduction": catastrophic_pixel_reduction,
            "baseline_median_flipped_faces": baseline_median_flip,
            "oracle_median_flipped_faces": oracle_median_flip,
            "median_flipped_face_reduction": median_flip_reduction,
            "baseline_max_edge_ratio_p95": baseline_max_edge_p95,
            "oracle_max_edge_ratio_p95": oracle_max_edge_p95,
            "max_edge_ratio_p95_reduction": max_edge_p95_reduction,
        },
        "preregistered_verdict": {
            "dominant_owner_confirmed": dominant,
            "material_owner_confirmed": material,
            "verdict": verdict,
        },
        "samples": samples,
        "previews": preview_rows,
        "claim_boundary": (
            "Teacher topology is used only as a causal oracle. Confirmation means "
            "the current canonical topology is a causal owner of the measured smear "
            "class with skin/skeleton/motion/vertices/appearance held fixed; it does "
            "not authorize teacher topology at product inference and does not define "
            "the generic repair operator."
        ),
    }
    (out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "KNIGHT_ORACLE_TOPOLOGY_CAUSAL_EXPERIMENT_PASS",
        json.dumps(
            {
                "verdict": verdict,
                "removed_face_count": report["intervention"]["removed_face_count"],
                "alpha_retention": alpha_retention,
                "catastrophic_pixel_reduction": catastrophic_pixel_reduction,
                "median_flip_reduction": median_flip_reduction,
                "max_edge_p95_reduction": max_edge_p95_reduction,
            },
            sort_keys=True,
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--teacher-bank", type=Path, required=True)
    p.add_argument("--teacher-source", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(
        a.authority_root,
        a.run_id,
        a.teacher_bank,
        a.teacher_source,
        a.out_dir,
    )


if __name__ == "__main__":
    main()
