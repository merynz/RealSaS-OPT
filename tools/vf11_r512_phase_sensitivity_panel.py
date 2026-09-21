from __future__ import annotations

"""VF-11 X1: subject-free R512 lattice-phase sensitivity panel.

This does not tune thresholds and does not use a subject. It holds physical
geometry, cameras, rasterizer, decoder resolution and the frozen Stage13 profile
fixed. Only the signed-field sampling lattice origin changes by a deterministic
2x2x2 half-voxel phase bank.

Question:
    Are the remaining R512 failures primarily a lattice-phase artifact that a
    bounded multiphase decoder candidate bank could solve, or do they persist
    across every phase and therefore require a different extraction
    representation/adaptive sharp-feature treatment?
"""

import gc
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from skimage import measure

# The imported apparatus reads this environment variable at import time.
os.environ.setdefault("REALSAS_VF11_DECODER_RESOLUTION", "512")

from tools import vf11_geometry_feature_survival_panel as base


OUT = Path(os.environ.get("REALSAS_VF11_X1_OUT", "vf11_x1_out"))
OUT.mkdir(parents=True, exist_ok=True)

if base.R != 512:
    raise RuntimeError("VF11_X1_REQUIRES_R512")

SELECTED = (
    ("THIN_STRAP", 2.0),
    ("SMALL_COMPONENT", 2.0),
    ("FORK_GAP", 2.0),
    ("CONCAVE_NOTCH", 2.0),
)
PHASE_FRACTIONS = tuple(itertools.product((0.0, 0.5), repeat=3))


def _field(
    boxes: tuple[dict, ...],
    *,
    phase_xyz: tuple[float, float, float],
) -> np.ndarray:
    lo, hi = base.BOUNDS
    dx, dy, dz = (
        float(value) * base.VOXEL
        for value in phase_xyz
    )
    axis = np.linspace(lo, hi, base.R, dtype=np.float32)
    axis_x = axis + np.float32(dx)
    axis_y = axis + np.float32(dy)
    axis_z = axis + np.float32(dz)
    yy, xx = np.meshgrid(axis_y, axis_x, indexing="ij")
    field = np.empty((base.R, base.R, base.R), dtype=np.float32)
    slab = max(1, int(os.environ.get("REALSAS_VF11_Z_SLAB", "4")))
    for z0 in range(0, base.R, slab):
        z1 = min(base.R, z0 + slab)
        zz = axis_z[z0:z1, None, None]
        x = xx[None, :, :]
        y = yy[None, :, :]
        local = np.full(
            (z1 - z0, base.R, base.R),
            np.inf,
            dtype=np.float32,
        )
        for box in boxes:
            local = np.minimum(local, base._box_sdf(x, y, zz, box))
        field[z0:z1] = local
        del local
    return field


def _extract(
    field: np.ndarray,
    *,
    phase_xyz: tuple[float, float, float],
):
    fmin = float(field.min())
    fmax = float(field.max())
    if not fmin < 0.0 < fmax:
        return None, {
            "status": "NO_ZERO_BRACKET",
            "field_min": fmin,
            "field_max": fmax,
        }

    spacing = (base.VOXEL, base.VOXEL, base.VOXEL)
    v_zyx, faces, _n_zyx, _ = measure.marching_cubes(
        np.asarray(field, dtype=np.float32),
        level=0.0,
        spacing=spacing,
        allow_degenerate=False,
    )
    lo = float(base.BOUNDS[0])
    dx, dy, dz = (
        float(value) * base.VOXEL
        for value in phase_xyz
    )
    vertices = np.stack(
        [
            lo + dx + v_zyx[:, 2],
            lo + dy + v_zyx[:, 1],
            lo + dz + v_zyx[:, 0],
        ],
        axis=-1,
    ).astype(np.float64)
    mesh = SimpleNamespace(
        vertices_normalized=vertices,
        faces=np.asarray(faces, dtype=np.int64),
    )
    return mesh, {
        "status": "EXTRACTED",
        "field_min": fmin,
        "field_max": fmax,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }


def _phase_id(phase_xyz: tuple[float, float, float]) -> str:
    return "P" + "".join("0" if value == 0.0 else "5" for value in phase_xyz)


def main() -> int:
    cameras = base._cameras()
    cases = []
    for family, width_reference_voxels in SELECTED:
        boxes, feature_boxes, feature_mode = base._case_boxes(
            family,
            width_reference_voxels,
        )
        source_by_view = tuple(
            base._render_boxes(boxes, camera)
            for camera in cameras
        )
        phase_rows = []

        for phase_xyz in PHASE_FRACTIONS:
            field = _field(boxes, phase_xyz=phase_xyz)
            mesh, extraction = _extract(field, phase_xyz=phase_xyz)
            view_rows = []
            if mesh is not None:
                for camera, source in zip(cameras, source_by_view):
                    predicted = base._render_decoder_mesh(mesh, camera)
                    metrics = base._coverage(source, predicted)
                    metrics["feature"] = base._feature_metric(
                        mode=feature_mode,
                        feature_boxes=feature_boxes,
                        source=source,
                        predicted=predicted,
                        camera=camera,
                    )
                    metrics["frozen_stage13_profile_passed"] = (
                        base._stage13_profile_pass(metrics)
                    )
                    view_rows.append(metrics)

            all_views_passed = bool(
                view_rows
                and all(
                    row["frozen_stage13_profile_passed"]
                    for row in view_rows
                )
            )
            feature_recall = [
                float(row["feature"]["feature_recall"])
                for row in view_rows
                if row["feature"]["feature_recall"] is not None
            ]
            gap_preservation = [
                float(row["feature"]["gap_preservation"])
                for row in view_rows
                if row["feature"]["gap_preservation"] is not None
            ]
            phase_rows.append(
                {
                    "phase_id": _phase_id(phase_xyz),
                    "phase_fraction_xyz": list(map(float, phase_xyz)),
                    "phase_world_xyz": [
                        float(value) * base.VOXEL
                        for value in phase_xyz
                    ],
                    "extraction": extraction,
                    "all_views_frozen_stage13_profile_passed": (
                        all_views_passed
                    ),
                    "passed_view_count": int(
                        sum(
                            row["frozen_stage13_profile_passed"]
                            for row in view_rows
                        )
                    ),
                    "worst": {
                        "min_recall": (
                            min(row["recall"] for row in view_rows)
                            if view_rows else 0.0
                        ),
                        "min_precision": (
                            min(row["precision"] for row in view_rows)
                            if view_rows else 0.0
                        ),
                        "min_component_recall": (
                            min(
                                row["minimum_component_recall"]
                                for row in view_rows
                            )
                            if view_rows else 0.0
                        ),
                        "max_largest_coherent_hole_fraction": (
                            max(
                                row["largest_coherent_hole_fraction"]
                                for row in view_rows
                            )
                            if view_rows else 1.0
                        ),
                        "max_silhouette_edge_p95_px": (
                            max(
                                row["silhouette_edge_p95_px"]
                                for row in view_rows
                            )
                            if view_rows else float(base.RES)
                        ),
                        "min_feature_recall": (
                            min(feature_recall)
                            if feature_recall else None
                        ),
                        "min_gap_preservation": (
                            min(gap_preservation)
                            if gap_preservation else None
                        ),
                    },
                    "views": view_rows,
                }
            )
            del field
            del mesh
            gc.collect()

        passing = [
            row["phase_id"]
            for row in phase_rows
            if row["all_views_frozen_stage13_profile_passed"]
        ]
        cases.append(
            {
                "family": family,
                "feature_width_reference_voxels": (
                    float(width_reference_voxels)
                ),
                "feature_width_world": (
                    float(width_reference_voxels)
                    * base.REFERENCE_VOXEL
                ),
                "phase_count": len(phase_rows),
                "passing_phase_ids": passing,
                "any_phase_all_views_passed": bool(passing),
                "maximum_passed_view_count": max(
                    row["passed_view_count"] for row in phase_rows
                ),
                "phase_rows": phase_rows,
            }
        )

    payload = {
        "schema": "RealSaS.VF11R512PhaseSensitivityPanel.v1",
        "status": "MEASURED_SUBJECT_FREE__NO_THRESHOLD_CHANGE",
        "decoder": "RealSaS.ZeroSurfaceDecoder.MarchingCubes.v3",
        "decoder_resolution": base.R,
        "raster_resolution": base.RES,
        "decoder_voxel_size": base.VOXEL,
        "reference_voxel_size": base.REFERENCE_VOXEL,
        "phase_bank": {
            "mode": "XYZ_BINARY_HALF_VOXEL_OFFSETS",
            "phase_fraction_values_per_axis": [0.0, 0.5],
            "phase_count": len(PHASE_FRACTIONS),
        },
        "frozen_stage13_profile": dict(base.FROZEN_STAGE13_PROFILE),
        "physical_geometry_held_fixed": True,
        "threshold_relaxation_used": False,
        "subject_inputs_used": False,
        "knight_result_used": False,
        "mage_result_used": False,
        "teacher_geometry_used": False,
        "cases": cases,
        "decision": {
            "all_selected_families_have_a_passing_phase": all(
                row["any_phase_all_views_passed"]
                for row in cases
            ),
            "families_without_passing_phase": [
                row["family"]
                for row in cases
                if not row["any_phase_all_views_passed"]
            ],
            "rule": (
                "A family with no all-view passing phase falsifies bounded "
                "global multiphase MC as a sufficient repair for that family."
            ),
        },
    }
    out = OUT / "VF11_R512_PHASE_SENSITIVITY_PANEL.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["decision"], sort_keys=True))
    for row in cases:
        print(
            "VF11_X1_CASE",
            row["family"],
            "passing=" + ",".join(row["passing_phase_ids"]),
            "max_passed_views=" + str(row["maximum_passed_view_count"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
