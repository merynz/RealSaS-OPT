from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, label as connected_label

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
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_texture_pages,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import (
    _candidate_skin_weights,
    _ctx,
    _mapping,
    _skin,
    _tracks_for_clip,
)


CLIPS = (
    ("demo_idle_v1", "IDLE", 833),
    ("demo_run_v1", "RUN", 208),
    ("demo_slash_v1", "SLASH", 278),
)
VIEWS = (0, 2)
MAX_DRAWABLES = 30


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _face_vertex_indices(candidate) -> np.ndarray:
    by_id = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    if len(by_id) != len(candidate.vertices):
        raise RuntimeError("DRAWABLE_ORACLE_DUPLICATE_VERTEX_ID")
    rows = []
    for face in candidate.faces:
        if len(face) != 3 or any(str(x) not in by_id for x in face):
            raise RuntimeError("DRAWABLE_ORACLE_FACE_INVALID")
        rows.append(tuple(by_id[str(x)] for x in face))
    return np.asarray(rows, dtype=np.int64)


def _source_parent_map(source_report: dict) -> dict[str, str | None]:
    return {
        str(row["source_joint_id"]): (
            None
            if row.get("parent_source_joint_id") is None
            else str(row["parent_source_joint_id"])
        )
        for row in source_report.get("bone_roles") or ()
    }


def _partition_faces(
    *,
    candidate,
    skeleton,
    skin,
    source_report: dict,
    reference_motion_payload: dict,
) -> tuple[np.ndarray, list[str], dict]:
    mapping, mapping_details, _, _, _ = _mapping(
        reference_motion_payload, skeleton, source_report
    )
    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    source_for_target = {
        str(jid): str(mapping.get(str(jid), f"UNMAPPED:{jid}"))
        for jid in joint_ids
    }
    source_ids = sorted(set(source_for_target.values()))
    si = {sid: i for i, sid in enumerate(source_ids)}
    grouped = np.zeros((W.shape[0], len(source_ids)), dtype=np.float64)
    for j, jid in enumerate(joint_ids):
        grouped[:, si[source_for_target[str(jid)]]] += W[:, j]
    if not np.allclose(grouped.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise RuntimeError("DRAWABLE_ORACLE_GROUPED_WEIGHT_NOT_SIMPLEX")

    fvi = _face_vertex_indices(candidate)
    face_score = grouped[fvi].mean(axis=1)
    labels = np.argmax(face_score, axis=1).astype(np.int64)
    face_source = np.asarray([source_ids[int(i)] for i in labels], dtype=object)

    parent = _source_parent_map(source_report)
    merge_log = []

    def counts() -> dict[str, int]:
        vals, c = np.unique(face_source, return_counts=True)
        return {str(v): int(n) for v, n in zip(vals.tolist(), c.tolist())}

    current = counts()
    while len(current) > MAX_DRAWABLES:
        child = min(current, key=lambda k: (current[k], k))
        ancestor = parent.get(child)
        while ancestor is not None and ancestor not in current:
            ancestor = parent.get(ancestor)
        fallback = False
        if ancestor is None:
            candidates = [k for k in current if k != child]
            if not candidates:
                break
            ancestor = max(candidates, key=lambda k: (current[k], k))
            fallback = True
        face_source[face_source == child] = ancestor
        merge_log.append(
            {
                "child": child,
                "into": ancestor,
                "child_face_count": int(current[child]),
                "fallback_nonancestor": bool(fallback),
            }
        )
        current = counts()

    drawables = sorted(current)
    gi = {name: i for i, name in enumerate(drawables)}
    face_group = np.asarray([gi[str(x)] for x in face_source], dtype=np.int32)
    details = {
        "target_to_source_mapping": mapping_details,
        "initial_source_group_count": len(source_ids),
        "final_drawable_count": len(drawables),
        "merge_log": merge_log,
        "drawable_face_counts": {
            name: int(np.count_nonzero(face_group == gi[name]))
            for name in drawables
        },
    }
    return face_group, drawables, details


def _subset_candidate(candidate, face_indices: np.ndarray):
    indices = np.asarray(face_indices, dtype=np.int64)
    faces = tuple(candidate.faces[int(i)] for i in indices.tolist())
    edges = sorted(
        {
            tuple(sorted((str(face[a]), str(face[b]))))
            for face in faces
            for a, b in ((0, 1), (1, 2), (2, 0))
        }
    )
    return replace(candidate, faces=faces, edges=tuple(edges))


def _order_from_depth_evidence(
    render,
    *,
    face_group: np.ndarray,
    drawables: list[str],
) -> tuple[list[int], dict]:
    owner = np.asarray(render.owner_face_index, dtype=np.int64)
    second = np.asarray(render.second_owner_face_index, dtype=np.int64)
    valid = (
        (owner >= 0)
        & (owner < len(face_group))
        & (second >= 0)
        & (second < len(face_group))
    )
    front = face_group[owner[valid]] if np.any(valid) else np.empty((0,), dtype=np.int32)
    back = face_group[second[valid]] if np.any(valid) else np.empty((0,), dtype=np.int32)
    cross = front != back
    front = front[cross]
    back = back[cross]
    n = len(drawables)
    wins = np.zeros((n,), dtype=np.int64)
    losses = np.zeros((n,), dtype=np.int64)
    pair = np.zeros((n, n), dtype=np.int64)
    if len(front):
        np.add.at(wins, front, 1)
        np.add.at(losses, back, 1)
        np.add.at(pair, (front, back), 1)
    score = wins - losses
    # Higher score means more often in front. Composite back -> front.
    order = sorted(range(n), key=lambda i: (int(score[i]), drawables[i]))
    return order, {
        "cross_drawable_depth_pair_pixels": int(len(front)),
        "score": {drawables[i]: int(score[i]) for i in range(n)},
        "wins": {drawables[i]: int(wins[i]) for i in range(n)},
        "losses": {drawables[i]: int(losses[i]) for i in range(n)},
        "order_back_to_front": [drawables[i] for i in order],
    }


def _alpha_composite_layers(layer_rgba: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    if not order:
        raise RuntimeError("DRAWABLE_ORACLE_EMPTY_ORDER")
    first = np.asarray(layer_rgba[order[0]], dtype=np.uint8)
    canvas = Image.new("RGBA", (first.shape[1], first.shape[0]), (0, 0, 0, 0))
    for group in order:
        image = Image.fromarray(np.asarray(layer_rgba[group], dtype=np.uint8), mode="RGBA")
        canvas = Image.alpha_composite(canvas, image)
    return np.asarray(canvas, dtype=np.uint8)


def _mask_metrics(rgba: np.ndarray) -> dict:
    value = np.asarray(rgba, dtype=np.uint8)
    alpha = value[:, :, 3] > 0
    filled = binary_fill_holes(alpha)
    holes = filled & ~alpha
    _, fg_components = connected_label(alpha)
    _, hole_components = connected_label(holes)
    return {
        "foreground_alpha_pixels": int(np.count_nonzero(alpha)),
        "foreground_connected_component_count": int(fg_components),
        "enclosed_transparent_hole_pixels": int(np.count_nonzero(holes)),
        "enclosed_transparent_hole_component_count": int(hole_components),
    }


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    x = np.asarray(a, dtype=np.int16)
    y = np.asarray(b, dtype=np.int16)
    if x.shape != y.shape:
        raise RuntimeError("DRAWABLE_ORACLE_DIFF_SHAPE")
    changed = np.any(x != y, axis=2)
    return {
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "mean_abs_rgba_delta": float(np.mean(np.abs(x - y))),
        "max_abs_rgba_delta": int(np.max(np.abs(x - y))),
    }


def _comparison_frame(
    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    clip: str,
    frame_index: int,
    time_seconds: float,
) -> Image.Image:
    if not rows:
        raise RuntimeError("DRAWABLE_ORACLE_COMPARISON_EMPTY")
    h, w = rows[0][1].shape[:2]
    header = 34
    row_header = 26
    canvas = Image.new(
        "RGBA",
        (3 * w, len(rows) * (h + row_header) + header),
        (0, 0, 0, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        f"{clip} frame={frame_index} t={time_seconds:.4f}s | BASELINE | STATIC DRAWABLE | DYNAMIC ORACLE DRAWABLE",
        fill=(255, 255, 255, 255),
    )
    for row_index, (view, base, static, dynamic) in enumerate(rows):
        y0 = header + row_index * (h + row_header)
        draw.text((8, y0 + 5), f"view {view}", fill=(255, 255, 255, 255))
        y = y0 + row_header
        for col, rgba in enumerate((base, static, dynamic)):
            canvas.alpha_composite(Image.fromarray(rgba, "RGBA"), (col * w, y))
    max_width = 1800
    if canvas.width > max_width:
        target_w = max_width
        target_h = max(1, round(canvas.height * target_w / canvas.width))
        canvas = canvas.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return canvas


def _save_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    pal = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=255) for f in frames]
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


def run(*, authority_root: Path, run_id: str, out_dir: Path) -> dict:
    ctx = _ctx(authority_root.resolve(), run_id)
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
    if dict(appearance.metadata or {}).get("source_owned_visual_mesh_mode") is True:
        raise RuntimeError("DRAWABLE_ORACLE_REQUIRES_MECHANICAL_CAA_BASELINE")
    if appearance.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise RuntimeError("DRAWABLE_ORACLE_STAGE23_CANDIDATE_DRIFT")

    seal_path = ctx["run_root"] / "IMMUTABLE_STAGE23_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("DRAWABLE_ORACLE_IMMUTABLE_STAGE23_SEAL_MISSING")
    seal = json.loads(seal_path.read_text())
    if (
        seal.get("status") != "SEALED_IMMUTABLE_STAGE23"
        or seal.get("run_id") != run_id
        or seal.get("overwrite_forbidden") is not True
        or seal.get("source_owned_visual_mesh_mode") is not False
    ):
        raise RuntimeError("DRAWABLE_ORACLE_IMMUTABLE_STAGE23_SEAL_INVALID")
    if seal.get("stage23_asset_hash") != appearance.asset_hash:
        raise RuntimeError("DRAWABLE_ORACLE_IMMUTABLE_STAGE23_ASSET_DRIFT")

    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {int(row.direction_index): row for row in appearance.textures}
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )

    reference_motion_path = (
        ctx["run_root"]
        / "inputs"
        / "motion"
        / "quaternius_knight_v1"
        / "demo_idle_v1.motion.json"
    )
    reference_motion_payload = json.loads(reference_motion_path.read_text())
    face_group, drawables, partition_details = _partition_faces(
        candidate=candidate,
        skeleton=skeleton,
        skin=skin,
        source_report=source_report,
        reference_motion_payload=reference_motion_payload,
    )
    if not (1 <= len(drawables) <= MAX_DRAWABLES):
        raise RuntimeError(
            f"DRAWABLE_ORACLE_PARTITION_CARDINALITY:{len(drawables)}"
        )

    subsets = {}
    for group, name in enumerate(drawables):
        indices = np.nonzero(face_group == group)[0].astype(np.int64)
        if len(indices) <= 0:
            raise RuntimeError(f"DRAWABLE_ORACLE_EMPTY_GROUP:{name}")
        subsets[group] = {
            "name": name,
            "indices": indices,
            "mesh": _subset_candidate(candidate, indices),
            "face_uv": face_uv[indices],
            "face_page": face_page[indices],
        }

    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)

    static_orders = {}
    static_order_evidence = {}
    for view in VIEWS:
        texture = _load_texture_pages(texture_by_view[view])
        rest_render = render_caa_reference(
            mesh=candidate,
            camera=cameras[view],
            face_uv=face_uv,
            texture_rgba_u8=texture,
            provenance_atlas=provenance[view],
            face_page_index=face_page,
            positions=rest,
        )
        order, evidence = _order_from_depth_evidence(
            rest_render,
            face_group=face_group,
            drawables=drawables,
        )
        static_orders[view] = order
        static_order_evidence[str(view)] = evidence

    out_dir.mkdir(parents=True, exist_ok=True)
    clips_report = []
    aggregate_rows = []

    for clip_id, clip_name, duration_ms in CLIPS:
        motion_path = (
            ctx["run_root"]
            / "inputs"
            / "motion"
            / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(motion_path.read_text())
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
        comparison_frames = []
        frame_rows = []

        for frame_index, time_seconds in enumerate(sample_times):
            skin_mats, _, frame_hash = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(time_seconds),
                cameras=cameras,
            )
            posed = _skin(rest, W, joint_ids, skin_mats)
            view_images = []

            for view in VIEWS:
                texture = _load_texture_pages(texture_by_view[view])
                baseline = render_caa_reference(
                    mesh=candidate,
                    camera=cameras[view],
                    face_uv=face_uv,
                    texture_rgba_u8=texture,
                    provenance_atlas=provenance[view],
                    face_page_index=face_page,
                    positions=posed,
                )
                dynamic_order, dynamic_evidence = _order_from_depth_evidence(
                    baseline,
                    face_group=face_group,
                    drawables=drawables,
                )

                layer_rgba = {}
                for group in range(len(drawables)):
                    subset = subsets[group]
                    layer_render = render_caa_reference(
                        mesh=subset["mesh"],
                        camera=cameras[view],
                        face_uv=subset["face_uv"],
                        texture_rgba_u8=texture,
                        provenance_atlas=provenance[view],
                        face_page_index=subset["face_page"],
                        positions=posed,
                    )
                    layer_rgba[group] = np.asarray(
                        layer_render.straight_rgba_u8,
                        dtype=np.uint8,
                    )

                baseline_rgba = np.asarray(baseline.straight_rgba_u8, dtype=np.uint8)
                static_rgba = _alpha_composite_layers(
                    layer_rgba, static_orders[view]
                )
                dynamic_rgba = _alpha_composite_layers(
                    layer_rgba, dynamic_order
                )

                row = {
                    "clip_id": clip_id,
                    "clip_name": clip_name,
                    "frame_index": int(frame_index),
                    "time_seconds": float(time_seconds),
                    "frame_hash": str(frame_hash),
                    "view_index": int(view),
                    "baseline": _mask_metrics(baseline_rgba),
                    "static_drawable": _mask_metrics(static_rgba),
                    "dynamic_drawable": _mask_metrics(dynamic_rgba),
                    "static_vs_baseline": _diff_metrics(static_rgba, baseline_rgba),
                    "dynamic_vs_baseline": _diff_metrics(dynamic_rgba, baseline_rgba),
                    "static_order_back_to_front": [
                        drawables[i] for i in static_orders[view]
                    ],
                    "dynamic_order": dynamic_evidence,
                }
                frame_rows.append(row)
                aggregate_rows.append(row)
                view_images.append(
                    (int(view), baseline_rgba, static_rgba, dynamic_rgba)
                )

            comparison_frames.append(
                _comparison_frame(
                    view_images,
                    clip=clip_name,
                    frame_index=int(frame_index),
                    time_seconds=float(time_seconds),
                )
            )

        gif_path = out_dir / f"KNIGHT_{clip_name}_ORACLE_DRAWABLE_COURT_V1.gif"
        _save_gif(comparison_frames, gif_path, duration_ms)
        first_png = out_dir / f"KNIGHT_{clip_name}_ORACLE_DRAWABLE_COURT_FRAME0_V1.png"
        comparison_frames[0].save(first_png)
        clips_report.append(
            {
                "clip_id": clip_id,
                "clip_name": clip_name,
                "duration_seconds": duration,
                "gif_path": str(gif_path),
                "gif_sha256": _sha256(gif_path),
                "frame0_path": str(first_png),
                "frame0_sha256": _sha256(first_png),
                "frames": frame_rows,
                "mapping": mapping_details,
            }
        )

    def mean_metric(variant: str, key: str) -> float:
        return float(np.mean([row[variant][key] for row in aggregate_rows]))

    baseline_holes = mean_metric("baseline", "enclosed_transparent_hole_pixels")
    static_holes = mean_metric("static_drawable", "enclosed_transparent_hole_pixels")
    dynamic_holes = mean_metric("dynamic_drawable", "enclosed_transparent_hole_pixels")
    summary = {
        "frame_view_sample_count": len(aggregate_rows),
        "mean_enclosed_hole_pixels": {
            "baseline": baseline_holes,
            "static_drawable": static_holes,
            "dynamic_drawable": dynamic_holes,
        },
        "mean_foreground_components": {
            variant: mean_metric(variant, "foreground_connected_component_count")
            for variant in ("baseline", "static_drawable", "dynamic_drawable")
        },
        "mean_foreground_alpha_pixels": {
            variant: mean_metric(variant, "foreground_alpha_pixels")
            for variant in ("baseline", "static_drawable", "dynamic_drawable")
        },
        "hole_pixel_change_fraction_vs_baseline": {
            "static": None if baseline_holes <= 0 else float((static_holes - baseline_holes) / baseline_holes),
            "dynamic": None if baseline_holes <= 0 else float((dynamic_holes - baseline_holes) / baseline_holes),
        },
        "automatic_scientific_verdict": "VISUAL_REVIEW_REQUIRED__METRICS_DESCRIPTIVE_ONLY",
    }

    partition_path = out_dir / "DRAWABLE_PARTITION.json"
    partition_path.write_text(
        json.dumps(
            {
                "schema": "RealSaS.KnightTeacherMechanicalDrawablePartition.v1",
                "authority": "RESEARCH_ORACLE_ONLY",
                "inference_admissible": False,
                "drawables": drawables,
                **partition_details,
                "static_order_evidence": static_order_evidence,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    report = {
        "schema": "RealSaS.KnightOracleDrawableCourt.v1",
        "status": "MEASURED_RESEARCH_ORACLE_ONLY",
        "run_id": run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "appearance_asset_hash": appearance.asset_hash,
        "immutable_stage23_seal_sha256": _sha256(seal_path),
        "mechanics_mutated": False,
        "skin_mutated": False,
        "motion_mutated": False,
        "appearance_asset_mutated": False,
        "partition_inference_admissible": False,
        "artist_authored_partition_claimed": False,
        "complete_layer_kind": "INDEPENDENT_PER_DRAWABLE_RENDER_FROM_SAME_POSED_MECHANICAL_CARRIER_AND_IMMUTABLE_CAA",
        "drawables": drawables,
        "partition": partition_details,
        "summary": summary,
        "clips": clips_report,
        "limitations": [
            "Drawable partition is privileged mechanics-derived, not artist-authored graphics truth.",
            "Complete layers are an optimistic oracle generated by independently rendering each mechanical drawable; they are not learned IRIS/TESSA outputs.",
            "Static order is one global order per view; dynamic order is privileged per-frame mechanical-depth evidence.",
            "A negative result cannot falsify a Bunraku/Live2D-style artist-authored partition because the partition basis differs.",
            "This court does not qualify grip/contact correctness or product presentation authority."
        ],
    }
    report_path = out_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_ORACLE_DRAWABLE_COURT_V1="
        + json.dumps(
            {
                "status": report["status"],
                "drawables": len(drawables),
                "candidate": candidate.candidate_lineage_hash,
                "skin": skin.skin_lineage_hash,
                "appearance": appearance.asset_hash,
                "summary": summary,
                "report_sha256": _sha256(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(
        authority_root=a.authority_root,
        run_id=a.run_id,
        out_dir=a.out_dir,
    )


if __name__ == "__main__":
    main()
