#!/usr/bin/env python3
from __future__ import annotations

"""Downstream-only Knight presentation-order research court.

Reuses sealed semantic masks, reconstructs semantic M fusion v2, compiles typed
source+motion order evidence, and compares canonical geometric-z composition to
research-only local semantic fragment precedence. No model inference, geometry,
rig, skin, retarget, CAA generation, or canonical runtime behavior is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.presentation_order_evidence_v1 import (
    compile_presentation_order_evidence_v1,
)
from compiler.realsas_compiler_core.semantic_relation_evidence_v2 import (
    lift_mask_evidence_to_m_v2,
)
from compiler.realsas_compiler_core.single_m_caa_v1 import (
    array_hash,
    load_material_bundle,
    material_sampler,
    render_single_m_caa,
    write_material_bundle,
)
from models.iris.semantic_evidence_v1 import load_mask_evidence_v1
from tools.demo.render_knight_m_caa_v1 import load_inputs
from tools.research.presentation_order_renderer_v1 import (
    render_m_depth_with_presentation_order,
)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _face_owner_ids(mesh, face_count):
    out = np.full(face_count, "", dtype=object)
    for oi, owner_id in enumerate(mesh.owner_ids):
        for fi in np.asarray(mesh.owner_face_indices[oi], dtype=int):
            out[fi] = owner_id
    return out


def _pair_residual(result, face_owner_ids, front, back):
    layers = np.asarray(result.layer_owner_face_index)
    flat = layers.reshape(-1, layers.shape[2])
    overlap = reversed_pixels = 0
    for row in flat:
        ids = row[row >= 0]
        if len(ids) < 2:
            continue
        owners = [str(face_owner_ids[int(fi)]) for fi in ids]
        if front not in owners or back not in owners:
            continue
        overlap += 1
        if owners.index(back) < owners.index(front):
            reversed_pixels += 1
    return {"overlap_pixels": overlap, "reversed_pixels": reversed_pixels}


def _render_court(material, posed, view, resolution, face_owner_ids, active_edges):
    projected = project_points_xyz_v3(posed, material["cameras"][view])
    sampler, counts = material_sampler(
        material["rest"],
        material["faces"],
        material["source"],
        material["maps"],
        material["cameras"],
        material["caa"],
        view,
        True,
    )
    result = render_m_depth_with_presentation_order(
        positions=projected[:, :2] - 0.5,
        depths=projected[:, 2],
        faces=material["faces"],
        uv=np.zeros((len(material["rest"]), 2)),
        texture=material["source"][view],
        resolution=resolution,
        sample_material=sampler,
        face_owner_ids=face_owner_ids,
        precedence_edges=[(x.front_owner_id, x.back_owner_id) for x in active_edges],
    )
    result.material_fragments = counts
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json",
    )
    p.add_argument("--semantic-evidence", type=Path, required=True)
    p.add_argument("--semantic-sha256", required=True)
    p.add_argument("--r9-report", type=Path, required=True)
    p.add_argument("--r9-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--focus-label", default="cape")
    p.add_argument("--views", default="3,4,7")
    p.add_argument("--resolution", type=int, default=256)
    args = p.parse_args()

    if digest(args.semantic_evidence) != args.semantic_sha256:
        raise RuntimeError("R8_SEMANTIC_EVIDENCE_BYTES_DRIFT")
    if digest(args.r9_report) != args.r9_report_sha256:
        raise RuntimeError("R9_RELATION_REPORT_BYTES_DRIFT")

    manifest, arrays, source, cameras, caa_arrays = load_inputs(args.baseline)
    rest_before = arrays["rest"].copy()
    poses_before = arrays["poses"].copy()
    evidence = load_mask_evidence_v1(args.semantic_evidence, expected_sha256=args.semantic_sha256)
    mesh = lift_mask_evidence_to_m_v2(
        rest=arrays["rest"],
        faces=arrays["faces"],
        cameras=cameras,
        evidence=evidence,
        minimum_label_support_views=2,
    )
    face_owner_ids = _face_owner_ids(mesh, len(arrays["faces"]))

    r9 = json.loads(args.r9_report.read_text(encoding="utf-8"))
    if (
        r9.get("schema") != "RealSaS.KnightFrozenSemanticRelationDiagnostic.v2"
        or r9.get("canonical_geometry") != "M_UNCHANGED"
        or r9.get("semantic_evidence_hash") != evidence.evidence_hash()
        or not r9.get("rest_geometry_byte_equal_after")
        or not r9.get("recorded_pose_byte_equal_after")
    ):
        raise RuntimeError("R9_RELATION_CONTRACT_INVALID")

    edges, order_audit = compile_presentation_order_evidence_v1(
        source_relations=r9["source_overlap_relations"],
        pose_conflicts=r9["pose_order_conflicts"],
    )
    selected = [
        x
        for x in edges
        if args.focus_label in {
            x.front_owner_id.split(":")[1],
            x.back_owner_id.split(":")[1],
        }
    ]
    if not selected:
        raise RuntimeError("NO_FOCUSED_PRESENTATION_ORDER_EDGE")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = write_material_bundle(
        out / "sealed_single_m_caa_material.npz",
        rest=arrays["rest"],
        faces=arrays["faces"],
        source=source,
        cameras=cameras,
        caa_arrays=caa_arrays,
        input_identities=manifest["inputs"],
    )
    material = load_material_bundle(
        ref["path"],
        expected_sha256=ref["sha256"],
        rest=arrays["rest"],
        faces=arrays["faces"],
    )

    views = sorted({int(x) for x in args.views.split(",") if x.strip()})
    if any(v < 0 or v >= 8 for v in views):
        raise RuntimeError("INVALID_VIEW_SELECTION")

    rows = []
    baseline_reversed = court_reversed = changed_total = override_total = 0
    inactive_mismatches = 0
    invariant_failures = []
    frame_grids = []

    for frame_index, posed in enumerate(arrays["poses"]):
        tiles = []
        for view in views:
            active = [
                e for e in selected
                if e.view_index == view and frame_index in e.valid_frame_indices
            ]
            baseline = render_single_m_caa(
                **{k: material[k] for k in ("rest", "faces", "source", "maps", "cameras", "caa")},
                posed=posed,
                view=view,
                resolution=args.resolution,
            )
            court = _render_court(
                material, posed, view, args.resolution, face_owner_ids, active
            )

            alpha_equal = bool(np.array_equal(baseline.final_alpha, court.final_alpha))
            geometry_equal = bool(np.array_equal(baseline.geometry_visible, court.geometry_visible))
            tie_equal = baseline.unresolved_depth_tie_count == court.unresolved_depth_tie_count
            overflow_equal = baseline.fragment_overflow_count == court.fragment_overflow_count
            geometric_owner_equal = bool(
                np.array_equal(baseline.owner_face_index, court.geometric_owner_face_index)
            )
            changed = np.any(
                baseline.straight_rgba_u8 != court.straight_rgba_u8, axis=2
            )
            changed_count = int(np.count_nonzero(changed))
            changed_subset_override = bool(np.all(changed <= court.presentation_override_mask))
            if not active and changed_count:
                inactive_mismatches += changed_count
            if not all((alpha_equal, geometry_equal, tie_equal, overflow_equal, geometric_owner_equal, changed_subset_override)):
                invariant_failures.append({"frame_index": frame_index, "view_index": view})

            pair_rows = []
            for edge in active:
                br = _pair_residual(
                    baseline, face_owner_ids, edge.front_owner_id, edge.back_owner_id
                )
                cr = _pair_residual(
                    court, face_owner_ids, edge.front_owner_id, edge.back_owner_id
                )
                baseline_reversed += br["reversed_pixels"]
                court_reversed += cr["reversed_pixels"]
                pair_rows.append(
                    {
                        "front_owner_id": edge.front_owner_id,
                        "back_owner_id": edge.back_owner_id,
                        "baseline": br,
                        "court": cr,
                    }
                )
            changed_total += changed_count
            override_total += court.presentation_override_pixel_count

            Image.fromarray(baseline.straight_rgba_u8).save(
                out / f"baseline_V{view}_F{frame_index:03d}.png"
            )
            Image.fromarray(court.straight_rgba_u8).save(
                out / f"court_V{view}_F{frame_index:03d}.png"
            )
            diff = np.zeros_like(court.straight_rgba_u8)
            diff[changed] = (255, 255, 255, 255)
            Image.fromarray(diff).save(out / f"diff_V{view}_F{frame_index:03d}.png")
            tiles.append((baseline.straight_rgba_u8, court.straight_rgba_u8, diff))

            rows.append(
                {
                    "frame_index": frame_index,
                    "view_index": view,
                    "active_edges": [x.to_dict() for x in active],
                    "pair_residuals": pair_rows,
                    "changed_rgba_pixels": changed_count,
                    "presentation_override_pixels": court.presentation_override_pixel_count,
                    "presentation_cycle_pixels": court.presentation_cycle_pixel_count,
                    "final_alpha_equal": alpha_equal,
                    "geometry_visible_equal": geometry_equal,
                    "geometric_owner_equal": geometric_owner_equal,
                    "depth_ties_equal": tie_equal,
                    "fragment_overflow_equal": overflow_equal,
                    "changed_subset_override": changed_subset_override,
                }
            )

        # Baseline/court/diff rows for selected views.
        w = args.resolution * len(views)
        h = args.resolution * 3
        grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for vi, triplet in enumerate(tiles):
            for ri, rgba in enumerate(triplet):
                grid.paste(Image.fromarray(rgba), (vi * args.resolution, ri * args.resolution))
        grid.save(out / f"court_grid_F{frame_index:03d}.png")
        frame_grids.append(grid)

    gif = out / "Knight_RUN_PRESENTATION_ORDER_COURT_V3_V4_V7.gif"
    frame_grids[0].save(
        gif,
        save_all=True,
        append_images=frame_grids[1:],
        duration=80,
        loop=0,
        disposal=2,
    )

    rest_equal = bool(np.array_equal(arrays["rest"], rest_before))
    poses_equal = bool(np.array_equal(arrays["poses"], poses_before))
    infrastructure_pass = (
        rest_equal
        and poses_equal
        and not invariant_failures
        and inactive_mismatches == 0
        and court_reversed == 0
    )
    hypothesis = (
        "SUPPORTED"
        if infrastructure_pass and baseline_reversed > 0 and changed_total > 0
        else "NO_VISIBLE_EFFECT"
        if infrastructure_pass and baseline_reversed > 0 and changed_total == 0
        else "NOT_SUPPORTED"
    )

    report = {
        "schema": "RealSaS.KnightPresentationOrderCourt.v1",
        "status": "PASS" if infrastructure_pass else "FAIL",
        "hypothesis_status": hypothesis,
        "focus_label": args.focus_label,
        "views": views,
        "qualified_order_edges": [x.to_dict() for x in edges],
        "selected_order_edges": [x.to_dict() for x in selected],
        "order_compiler_audit": order_audit,
        "baseline_reversed_pixels": baseline_reversed,
        "court_reversed_pixels": court_reversed,
        "changed_rgba_pixels": changed_total,
        "presentation_override_pixels": override_total,
        "inactive_edge_rgba_mismatches": inactive_mismatches,
        "invariant_failures": invariant_failures,
        "rest_geometry_byte_equal_after": rest_equal,
        "recorded_pose_byte_equal_after": poses_equal,
        "semantic_evidence_sha256": args.semantic_sha256,
        "r9_report_sha256": args.r9_report_sha256,
        "material": ref,
        "gif_sha256": digest(gif),
        "rows": rows,
        "claims": {
            "runtime_behavior_changed": False,
            "canonical_renderer_modified": False,
            "product_authority_claimed": False,
            "geometry_authority": "M_UNCHANGED",
            "research_court_only": True,
        },
    }
    (out / "presentation_order_court_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "hypothesis_status": hypothesis,
                "selected_edges": len(selected),
                "baseline_reversed_pixels": baseline_reversed,
                "court_reversed_pixels": court_reversed,
                "changed_rgba_pixels": changed_total,
                "inactive_edge_rgba_mismatches": inactive_mismatches,
                "rest_geometry_equal": rest_equal,
                "recorded_poses_equal": poses_equal,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if infrastructure_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
