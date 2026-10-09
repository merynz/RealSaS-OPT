#!/usr/bin/env python3
"""Reproduce a presentation-palette counterexample; never mint qualification.

Two pieces share one rest joint and exactly the same rigid 3D motion. Evaluate
the real body palette with different pivots, then measure joint coincidence.
The optional reanchoring comparison illustrates a relational constraint only;
it is neither a production repair nor a Knight-derived experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compiler.realsas_compiler_core.camera_geometry_v2 import (  # noqa: E402
    CameraProjectionV3, project_points_xyz_v3,
)
from compiler.realsas_compiler_core.visual_domain_v2 import (  # noqa: E402
    presentation_condition_metrics,
)
from compiler.realsas_compiler_core.visual_motion_blend_v1 import (  # noqa: E402
    canonical_pose_palette_residual, evaluate_motion_blend,
)


def run_probe():
    camera = CameraProjectionV3(
        "SYNTHETIC_DIAGNOSTIC", 0, (0, 0, 0), (1, 0, 0),
        (0, 1, 0), (0, 0, 1), 2.0, 256,
    )
    rows, scenes = [], []
    cases = [
        ("identity", "Y", 0.0, 1.0),
        ("in_plane_positive_control", "Z", 60.0, 1.0),
        ("out_of_plane_flat_rest", "Y", 60.0, 0.0),
        ("out_of_plane_depth_varying_rest", "Y", 60.0, 1.0),
    ]
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    for name, axis, degrees, rest_slope in cases:
        angle = np.deg2rad(degrees)
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.eye(4)
        matrix[:3, :3] = (
            [[c, 0, s], [0, 1, 0], [-s, 0, c]] if axis == "Y"
            else [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        )
        joint = np.array([1.0, 0.0, rest_slope])
        # Rectangles meet along an entire edge, including the marked joint.
        xy = np.array([[0, -.15], [1, -.15], [1, .15], [0, .15],
                       [1, -.15], [2, -.15], [2, .15], [1, .15]])
        xyz = np.c_[xy, xy[:, 0] * rest_slope]
        xyz = np.vstack((xyz, joint, joint))
        blend = np.array([[1., 0.]] * 4 + [[0., 1.]] * 4 +
                         [[1., 0.], [0., 1.]])
        palette = np.stack((matrix, matrix))
        axis_xyz = np.stack((np.zeros(3), joint))
        projected_rest = project_points_xyz_v3(xyz, camera)
        rest_xy = projected_rest[:, :2]
        posed_xyz = (np.c_[xyz, np.ones(len(xyz))] @ matrix.T)[:, :3]
        projected_posed = project_points_xyz_v3(posed_xyz, camera)
        actual = evaluate_motion_blend(
            projected_rest, rest_source_xy=rest_xy, coefficients=blend,
            axis_positions_source=axis_xyz, skin_matrices_source=palette,
            camera=camera,
        )
        # Controlled diagnostic: only reanchor the child at the parent's image
        # of the common joint. This is deliberately not a general FK algorithm.
        coupled = actual[:, :2].copy()
        coupled[[4, 5, 6, 7, 9]] += actual[8, :2] - actual[9, :2]
        metrics = presentation_condition_metrics(rest_xy, actual[:, :2], faces)
        direct_shared_residual = float(np.linalg.norm(
            projected_posed[8, :2] - projected_posed[9, :2]))
        mechanical_shared_residual = float(np.linalg.norm(posed_xyz[8] - posed_xyz[9]))
        row = {
            "case": name, "axis": axis, "rotation_degrees": degrees,
            "rest_joint_xyz": joint.tolist(),
            "three_dimensional_shared_joint_residual": mechanical_shared_residual,
            "canonical_pose_palette_residual": canonical_pose_palette_residual(
                rest_xyz=xyz, posed_xyz=posed_xyz, mechanical_weights=blend,
                skin_matrices_source=palette,
            ),
            "direct_projected_shared_joint_separation_px": direct_shared_residual,
            "parent_visual_joint_xy": actual[8, :2].tolist(),
            "child_visual_joint_xy": actual[9, :2].tolist(),
            "visual_shared_joint_separation_px": float(np.linalg.norm(actual[8, :2] - actual[9, :2])),
            "parent_to_child_gap_x_px": float(actual[9, 0] - actual[8, 0]),
            "reanchored_diagnostic_shared_joint_separation_px": float(np.linalg.norm(coupled[8] - coupled[9])),
            "triangle_metrics": metrics,
        }
        assert mechanical_shared_residual < 1e-10
        assert direct_shared_residual < 1e-10
        assert row["canonical_pose_palette_residual"] < 1e-10
        assert metrics["area_collapse_count"] == metrics["condition_failure_count"] == 0
        assert abs(metrics["minimum_signed_area_ratio"] - 1) < 1e-10
        assert abs(metrics["maximum_jacobian_condition"] - 1) < 1e-10
        assert row["reanchored_diagnostic_shared_joint_separation_px"] < 1e-10
        rows.append(row)
        scenes.append((rest_xy, actual[:, :2], coupled))
    assert rows[0]["visual_shared_joint_separation_px"] < 1e-10
    assert rows[1]["visual_shared_joint_separation_px"] < 1e-10
    assert abs(rows[2]["visual_shared_joint_separation_px"] - 32) < 1e-10
    assert abs(rows[3]["parent_to_child_gap_x_px"] - 23.425625842204084) < 1e-10
    files = ["compiler/realsas_compiler_core/visual_motion_blend_v1.py",
             "compiler/realsas_compiler_core/visual_attachment_motion_v1.py",
             "compiler/realsas_compiler_core/visual_domain_v2.py",
             "compiler/realsas_compiler_core/camera_geometry_v2.py"]
    result = {
        "schema": "RealSaS.VisualPaletteJointDiagnostic.v1",
        "qualification": "SYNTHETIC_DIAGNOSTIC_ONLY",
        "synthetic_input_not_derived_from_knight": True,
        "sealed_mechanics_modified": False,
        "production_repair_implemented": False,
        "stage45_executed_on_this_fixture": False,
        "runtime_render_executed": False,
        "resolution": 256,
        "numpy_version": np.__version__,
        "implementation_file_sha256": {
            p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in files
        },
        "rows": rows,
        "conclusion": "INDEPENDENT_PIVOT_CAMERA_TWIST_DOES_NOT_PRESERVE_JOINT_COINCIDENCE",
        "comparison_limit": "CHILD_TRANSLATION_REANCHOR_ONLY__NOT_A_GENERAL_PRESENTATION_MAPPING",
    }
    return result, scenes[-1]


def plot_probe(path, scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), constrained_layout=True)
    titles = ["Shared rest edge", "Current body palette: 23.43 px gap",
              "Reanchoring comparison: 0 px gap"]
    for ax, points, title in zip(axes, scene, titles):
        for indices, color in (([0, 1, 2, 3], "#4678bf"), ([4, 5, 6, 7], "#e39739")):
            ax.add_patch(Polygon(points[indices], facecolor=color, alpha=.75, edgecolor=color))
        ax.scatter(points[[8, 9], 0], points[[8, 9], 1], s=45, c=["#244a7e", "#9d5500"], zorder=4)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(115, 290)
        ax.set_ylim(156, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("Camera pixel X (256 px projection)")
        ax.grid(alpha=.2)
    axes[1].annotate("", xy=(scene[1][8, 0], 148), xytext=(scene[1][9, 0], 148),
                     arrowprops={"arrowstyle": "<->", "color": "#333"})
    fig.suptitle("Synthetic counterexample using the actual RealSaS operator", fontsize=14)
    fig.text(.5, .015, "Same proper 3D motion; healthy triangles. Diagnostic only: no Knight qualification or production repair.",
             ha="center", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional diagnostic JSON output.")
    parser.add_argument("--plot", type=Path, help="Optional PNG/SVG diagnostic figure (requires matplotlib).")
    args = parser.parse_args()
    result, scene = run_probe()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    if args.plot:
        plot_probe(args.plot, scene)
    print(encoded)


if __name__ == "__main__":
    main()
