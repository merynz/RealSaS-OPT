from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.mesh.conditioning_v1 import triangle_rest_metric
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_texture_pages
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)

MIN_ANGLE_DEG = 7.5
MAX_ASPECT = 16.0
CASES = (
    ("idle_v2_f0", "demo_idle_v1", 2, 0),
    ("run_v2_f0", "demo_run_v1", 2, 0),
    ("slash_v0_f0", "demo_slash_v1", 0, 0),
)


def _rgba(render):
    return np.asarray(render.straight_rgba_u8, dtype=np.uint8)


def _counts(render, face_count: int) -> np.ndarray:
    owners = np.asarray(render.layer_owner_face_index, dtype=np.int64)
    contrib = np.asarray(render.contributing_layer_mask, dtype=bool)
    vals = owners[contrib]
    vals = vals[vals >= 0]
    if not len(vals):
        return np.zeros((face_count,), dtype=np.int64)
    return np.bincount(vals, minlength=face_count)[:face_count]


def _metric_rows(candidate, rest, posed, W):
    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    rows = []
    for fi, face in enumerate(candidate.faces):
        idx = np.asarray([vi[str(v)] for v in face], dtype=np.int64)
        rp = rest[idx]
        pp = posed[idx]
        q = triangle_rest_metric(tuple(tuple(map(float, x)) for x in rp))
        rest_edges = np.asarray([
            np.linalg.norm(rp[1] - rp[0]),
            np.linalg.norm(rp[2] - rp[1]),
            np.linalg.norm(rp[0] - rp[2]),
        ])
        posed_edges = np.asarray([
            np.linalg.norm(pp[1] - pp[0]),
            np.linalg.norm(pp[2] - pp[1]),
            np.linalg.norm(pp[0] - pp[2]),
        ])
        edge_ratio = float(np.max(posed_edges / np.maximum(rest_edges, 1e-12)))
        rest_area = float(0.5 * np.linalg.norm(np.cross(rp[1] - rp[0], rp[2] - rp[0])))
        posed_area = float(0.5 * np.linalg.norm(np.cross(pp[1] - pp[0], pp[2] - pp[0])))
        skin_div = max(
            float(np.sum(np.abs(W[idx[a]] - W[idx[b]])))
            for a, b in ((0,1),(1,2),(2,0))
        )
        max_disp = float(np.max(np.linalg.norm(pp - rp, axis=1)))
        rows.append({
            "face_index": fi,
            "vertex_ids": list(map(str, face)),
            "min_angle_deg": float(q["min_angle_deg"]),
            "aspect": float(q["aspect_longest_over_min_altitude"]),
            "degenerate": bool(q["degenerate"]),
            "policy_violating": bool(
                q["degenerate"]
                or float(q["min_angle_deg"]) + 1e-9 < MIN_ANGLE_DEG
                or float(q["aspect_longest_over_min_altitude"]) - 1e-9 > MAX_ASPECT
            ),
            "max_edge_stretch": edge_ratio,
            "area_ratio_3d": float(posed_area / max(rest_area, 1e-18)),
            "max_vertex_displacement": max_disp,
            "skin_weight_pair_l1_max": skin_div,
        })
    return rows


def _submesh(candidate, face_indices):
    return replace(candidate, faces=tuple(candidate.faces[i] for i in face_indices))


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _panel(items, title):
    w, h = items[0][1].size
    header = 30
    out = Image.new("RGBA", (w * len(items), h + header), (0,0,0,0))
    draw = ImageDraw.Draw(out)
    for i, (label, img) in enumerate(items):
        out.alpha_composite(img, (i*w, header))
        draw.text((i*w+6, 7), label, fill=(255,255,255,255))
    draw.text((6, h + 8), title, fill=(255,255,255,255))
    return out


def run(authority_root: Path, run_id: str, out_dir: Path):
    ctx = _ctx(authority_root, run_id)
    candidate = canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.CanonicalMeshCandidateIR.v1"
    ))
    skeleton = qualified_skeleton_from_dict(stage_output_payload(
        ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
    ))
    skin = qualified_skin_from_dict(stage_output_payload(
        ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedSkinIR.v1"
    ))
    camera_set = qualified_camera_set_from_dict(stage_output_payload(
        ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
    ))
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    appearance = complete_appearance_asset_from_dict(stage_output_payload(
        ctx, "23_COMPLETE_APPEARANCE_ASSET_BAKED", "RealSaS.CompleteAppearanceAssetIR.v2"
    ))
    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {int(row.direction_index): row for row in appearance.textures}
    source_report = json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "RealSaS.KnightSmearFaceCausalityDiagnostic.v1",
        "status": "DIAGNOSTIC_ONLY",
        "run_id": run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "face_count": len(candidate.faces),
        "thresholds": {"min_angle_deg": MIN_ANGLE_DEG, "max_aspect": MAX_ASPECT},
        "cases": [],
        "product_authority_claimed": False,
        "repair_attempted": False,
    }

    for label, clip_id, view, frame_index in CASES:
        payload = json.loads((
            ctx["run_root"] / "inputs" / "motion" / "quaternius_knight_v1" / f"{clip_id}.motion.json"
        ).read_text())
        tracks, _ = _tracks_for_clip(payload, skeleton, cameras, source_report)
        sample_times = np.linspace(
            0.0, float(payload["duration_seconds"]), 4, endpoint=not bool(payload.get("loop"))
        )
        t = float(sample_times[frame_index])
        skin_mats, _, _ = _joint_pose_v2(
            skeleton=skeleton, tracks=tracks, time_seconds=t, cameras=cameras
        )
        posed = _skin(rest, W, joint_ids, skin_mats)
        texture = _load_texture_pages(texture_by_view[view])

        full = render_caa_reference(
            mesh=candidate, camera=cameras[view], face_uv=face_uv,
            texture_rgba_u8=texture, provenance_atlas=provenance[view],
            face_page_index=face_page, positions=posed,
        )
        rest_render = render_caa_reference(
            mesh=candidate, camera=cameras[view], face_uv=face_uv,
            texture_rgba_u8=texture, provenance_atlas=provenance[view],
            face_page_index=face_page, positions=rest,
        )

        rows = _metric_rows(candidate, rest, posed, W)
        dyn_counts = _counts(full, len(candidate.faces))
        rest_counts = _counts(rest_render, len(candidate.faces))
        for row in rows:
            fi = row["face_index"]
            row["dynamic_contribution_samples"] = int(dyn_counts[fi])
            row["rest_contribution_samples"] = int(rest_counts[fi])
            row["contribution_growth_samples"] = int(dyn_counts[fi] - rest_counts[fi])
            row["contribution_ratio"] = float(
                dyn_counts[fi] / max(int(rest_counts[fi]), 1)
            )

        violating = [r["face_index"] for r in rows if r["policy_violating"]]
        keep = [i for i in range(len(candidate.faces)) if i not in set(violating)]

        bad_mesh = _submesh(candidate, violating)
        good_mesh = _submesh(candidate, keep)
        bad_render = render_caa_reference(
            mesh=bad_mesh, camera=cameras[view], face_uv=face_uv[violating],
            texture_rgba_u8=texture, provenance_atlas=provenance[view],
            face_page_index=face_page[violating], positions=posed,
        )
        good_render = render_caa_reference(
            mesh=good_mesh, camera=cameras[view], face_uv=face_uv[keep],
            texture_rgba_u8=texture, provenance_atlas=provenance[view],
            face_page_index=face_page[keep], positions=posed,
        )

        full_rgba = _rgba(full)
        bad_rgba = _rgba(bad_render)
        good_rgba = _rgba(good_render)
        diff = np.any(full_rgba != good_rgba, axis=2)

        img_full = Image.fromarray(full_rgba, "RGBA")
        img_bad = Image.fromarray(bad_rgba, "RGBA")
        img_good = Image.fromarray(good_rgba, "RGBA")
        panel = Image.new("RGBA", (img_full.width*3, img_full.height+30), (0,0,0,0))
        draw = ImageDraw.Draw(panel)
        for i, (name, img) in enumerate((("FULL",img_full),("VIOLATING_ONLY",img_bad),("WITHOUT_VIOLATING",img_good))):
            panel.alpha_composite(img, (i*img_full.width,30))
            draw.text((i*img_full.width+6,7), name, fill=(255,255,255,255))
        panel_path = out_dir / f"{label}_causal_panel.png"
        panel.save(panel_path)

        ranked = sorted(
            rows,
            key=lambda r: (
                r["contribution_growth_samples"],
                r["dynamic_contribution_samples"],
                r["max_edge_stretch"],
            ),
            reverse=True,
        )
        top = ranked[:40]
        top_bad = [r for r in ranked if r["policy_violating"]][:40]
        full_alpha = np.asarray(full.final_alpha, dtype=bool)
        bad_alpha = np.asarray(bad_render.final_alpha, dtype=bool)
        good_alpha = np.asarray(good_render.final_alpha, dtype=bool)
        result["cases"].append({
            "label": label,
            "clip_id": clip_id,
            "view_index": view,
            "frame_index": frame_index,
            "time_seconds": t,
            "violating_face_count": len(violating),
            "full_alpha_pixels": int(np.count_nonzero(full_alpha)),
            "violating_only_alpha_pixels": int(np.count_nonzero(bad_alpha)),
            "without_violating_alpha_pixels": int(np.count_nonzero(good_alpha)),
            "full_vs_without_changed_pixels": int(np.count_nonzero(diff)),
            "full_vs_without_changed_bbox": _bbox(diff),
            "violating_only_bbox": _bbox(bad_alpha),
            "full_bbox": _bbox(full_alpha),
            "without_violating_bbox": _bbox(good_alpha),
            "top_dynamic_growth_faces": top,
            "top_policy_violating_faces": top_bad,
            "panel_path": str(panel_path),
        })

    report = out_dir / "REPORT.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_SMEAR_FACE_CAUSALITY_DIAGNOSTIC_PASS", json.dumps({
        "cases": len(result["cases"]),
        "report": str(report),
        "repair_attempted": False,
    }, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.authority_root, a.run_id, a.out_dir)


if __name__ == "__main__":
    main()
