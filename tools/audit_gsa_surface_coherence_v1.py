from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    _adaptive_voxel_compact,
    _self_zbuffer_support,
)

ROOT = Path(__file__).resolve().parents[1]


def _grid_faces(n: int, offset: int) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = offset + y * n + x
            b = a + 1
            c = offset + (y + 1) * n + x
            d = c + 1
            rows.append((a, b, d))
            rows.append((a, d, c))
    return rows


def compact_same_component_fold():
    # Two locally parallel sheets belong to ONE connected component (their
    # connection may live outside this local patch). They are distinct surface
    # loci but nearly coincident in Euclidean XYZ.
    #
    # Add two distant Z anchors so the production axis-normalized voxelizer has a
    # realistic global span. The anchors do not create an artificial component:
    # precomputed labels explicitly state all vertices belong to one component.
    n = 7
    xy = np.linspace(-0.30, 0.30, n, dtype=np.float64)
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    lower = np.column_stack(
        (xx.reshape(-1), yy.reshape(-1), np.zeros(n * n, dtype=np.float64))
    )
    upper = np.column_stack(
        (
            xx.reshape(-1),
            yy.reshape(-1),
            np.full(n * n, 0.001, dtype=np.float64),
        )
    )
    anchors = np.asarray(
        [
            [0.95, 0.95, -1.0],
            [-0.95, -0.95, 1.0],
        ],
        dtype=np.float64,
    )
    points = np.concatenate((lower, upper, anchors), axis=0)
    normals = np.tile(
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        (len(points), 1),
    )
    faces = np.asarray(
        _grid_faces(n, 0) + _grid_faces(n, n * n),
        dtype=np.int64,
    )
    labels = np.zeros((len(points),), dtype=np.int64)

    compact, compact_normals, edges, divisions, inverse = _adaptive_voxel_compact(
        points,
        faces,
        normals,
        target_nodes=64,
        preserve_connected_components=True,
        precomputed_component_labels=labels,
    )

    pair_rows = []
    for index in range(n * n):
        li = int(inverse[index])
        ui = int(inverse[n * n + index])
        merged = li == ui
        cp = compact[li] if merged else None
        off_surface_distance = (
            min(abs(float(cp[2]) - 0.0), abs(float(cp[2]) - 0.001))
            if cp is not None
            else 0.0
        )
        pair_rows.append(
            {
                "pair_index": index,
                "lower_compact_id": li,
                "upper_compact_id": ui,
                "merged": bool(merged),
                "compact_z": None if cp is None else float(cp[2]),
                "off_surface_distance": float(off_surface_distance),
            }
        )

    merged_rows = [row for row in pair_rows if row["merged"]]
    off_surface_rows = [
        row for row in merged_rows if row["off_surface_distance"] > 1.0e-6
    ]
    return {
        "input_vertex_count": int(len(points)),
        "sheet_vertex_count_each": int(n * n),
        "target_nodes": 64,
        "compact_node_count": int(len(compact)),
        "compact_edge_count": int(len(edges)),
        "voxel_divisions": int(divisions),
        "same_component_label_for_both_sheets": True,
        "corresponding_sheet_pair_count": int(n * n),
        "merged_corresponding_sheet_pair_count": int(len(merged_rows)),
        "off_surface_merged_pair_count": int(len(off_surface_rows)),
        "maximum_off_surface_distance": float(
            max((row["off_surface_distance"] for row in merged_rows), default=0.0)
        ),
        "example_merged_pairs": merged_rows[:8],
        "distinct_sheets_merged": bool(merged_rows),
        "merged_centroid_leaves_both_input_sheets": bool(off_surface_rows),
        "compact_normal_finite": bool(np.isfinite(compact_normals).all()),
    }


def vertex_splat_visibility():
    camera = {
        "origin": (0.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
        "screen_up": (0.0, 1.0, 0.0),
        "forward": (0.0, 0.0, 1.0),
        "half_extent": 1.0,
        "resolution": 128,
    }
    # The front triangle covers the center as a continuous surface, but the
    # current self-zbuffer receives only dense VERTICES. Its three corners do not
    # write the center pixel; the back point therefore writes itself and appears
    # visible.
    front = np.asarray(
        [
            [-0.65, -0.65, 1.0],
            [0.65, -0.65, 1.0],
            [0.00, 0.65, 1.0],
        ],
        dtype=np.float64,
    )
    back = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64)
    dense = np.concatenate((front, back), axis=0)
    compact = back.copy()
    support, raster, counts = _self_zbuffer_support(
        dense,
        compact,
        (camera,) * 8,
        depth_tolerance=1.0e-6,
    )
    return {
        "point_splat_marks_back_visible_all_views": bool(np.all(support[0])),
        "support": support[0].tolist(),
        "raster": raster[0].tolist(),
        "visible_counts": list(map(int, counts)),
        "continuous_front_triangle_contains_center": True,
        "continuous_surface_expected_back_visibility": False,
    }


fold = compact_same_component_fold()
vis = vertex_splat_visibility()
payload = {
    "schema": "RealSaS.GSASurfaceCoherenceAdversaries.v1",
    "status": "AUDIT_ONLY__NO_REPAIR_APPLIED",
    "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "same_component_fold_compaction": fold,
    "vertex_splat_visibility": vis,
    "findings": [
        {
            "id": "GSA_COMPONENT_AWARE_VOXEL_COMPACTION_CAN_MERGE_GEODESICALLY_DISTINCT_SHEETS",
            "severity": "P0",
            "confirmed": bool(
                fold["distinct_sheets_merged"]
                and fold["merged_centroid_leaves_both_input_sheets"]
            ),
            "class": "SURFACE_TOPOLOGY_COLLAPSE",
            "design_before_code": (
                "Compare surface-sampling, cell-aware clustering, geodesic/topological "
                "clustering, and projection-back-to-surface variants under the fixed "
                "node budget. Connected-component labels alone are insufficient."
            ),
        },
        {
            "id": "GSA_VERTEX_SPLAT_ZBUFFER_CAN_MISCLASSIFY_OCCLUDED_SURFACE_AS_VISIBLE",
            "severity": "P1",
            "confirmed": bool(vis["point_splat_marks_back_visible_all_views"]),
            "class": "VISIBILITY_EVIDENCE_APPROXIMATION",
            "design_before_code": (
                "Use a topology-aware surface visibility operator (for example exact "
                "admitted-face raster/depth) or calibrate a conservative equivalent; "
                "do not infer source support from vertex splats."
            ),
        },
    ],
    "claim_boundary": (
        "Synthetic adversaries isolate current GSA operators. They do not choose "
        "the final point budget, compact surface representation, or visibility implementation."
    ),
}
out = ROOT / "canonical" / "GSA_SURFACE_COHERENCE_ADVERSARIES_V1_20260928.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
if not all(row["confirmed"] for row in payload["findings"]):
    raise SystemExit(2)
