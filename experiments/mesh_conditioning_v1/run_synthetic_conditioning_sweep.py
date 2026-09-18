#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.mesh.conditioning_v1 import (
    triangle_deformation_metric,
    triangle_rest_metric,
)

ANGLES_DEG = (0.25,0.5,1.0,2.0,3.0,5.0,7.5,10.0,15.0,20.0,30.0,45.0,60.0)
SCALES = (1e-3,1.0,1e3)
POSE_NOISE_FRACTIONS = (0.0,1e-5,1e-4,1e-3)
MAX_ALLOWED_RELATIVE_METRIC_ERROR = 0.05

_angle = math.radians(37.0)
RY37 = np.array([
    [math.cos(_angle),0.0,math.sin(_angle)],
    [0.0,1.0,0.0],
    [-math.sin(_angle),0.0,math.cos(_angle)],
])
BASE = np.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])
TRANSFORMS = {
    "IDENTITY": BASE,
    "RIGID_Y37": RY37 @ BASE,
    "ANISO_2x_0p5": np.array([[2.0,0.0],[0.0,0.5],[0.0,0.0]]),
    "SHEAR_1": np.array([[1.0,1.0],[0.0,1.0],[0.0,0.0]]),
    "COMPRESS_0p25": np.array([[1.0,0.0],[0.0,0.25],[0.0,0.0]]),
}
NOISE_PATTERN = np.array([
    [0.7,-0.2,0.1],
    [-0.3,0.8,-0.4],
    [0.2,-0.6,0.9],
], dtype=np.float64)
NOISE_PATTERN /= np.max(np.linalg.norm(NOISE_PATTERN, axis=1))


def run_sweep() -> dict:
    per_angle = {}
    sample_count = 0
    for angle in ANGLES_DEG:
        worst = 0.0
        rest_aspect = None
        worst_case = None
        for scale in SCALES:
            height = 0.5 * math.tan(math.radians(angle)) * scale
            rest = np.array([
                [0.0,0.0,0.0],
                [scale,0.0,0.0],
                [0.5*scale,height,0.0],
            ], dtype=np.float64)
            rest_aspect = triangle_rest_metric(rest)["aspect_longest_over_min_altitude"]
            for name, transform in TRANSFORMS.items():
                true_singular = np.linalg.svd(transform, compute_uv=False)
                truth = {
                    "sigma_min": float(true_singular[1]),
                    "area_ratio": float(true_singular[0] * true_singular[1]),
                    "condition_number": float(true_singular[0] / true_singular[1]),
                }
                base_pose = (transform @ rest[:,:2].T).T
                for noise_fraction in POSE_NOISE_FRACTIONS:
                    posed = base_pose + noise_fraction * scale * NOISE_PATTERN
                    metric = triangle_deformation_metric(rest, posed, dtype=np.float32)
                    errors = {
                        key: abs(float(metric[key]) - truth[key]) / truth[key]
                        for key in truth
                    }
                    error = max(errors.values())
                    sample_count += 1
                    if error > worst:
                        worst = error
                        worst_case = {
                            "scale": scale,
                            "transform": name,
                            "pose_noise_fraction": noise_fraction,
                            "errors": errors,
                        }
        per_angle[str(angle)] = {
            "min_angle_deg": angle,
            "aspect_longest_over_min_altitude": float(rest_aspect),
            "worst_relative_metric_error": float(worst),
            "passes_5pct_conditioning_rule": bool(worst <= MAX_ALLOWED_RELATIVE_METRIC_ERROR),
            "worst_case": worst_case,
        }

    passing = [
        angle for angle in ANGLES_DEG
        if per_angle[str(angle)]["passes_5pct_conditioning_rule"]
    ]
    selected = min(passing) if passing else None
    aspect_ceiling = None
    if selected is not None:
        aspect_ceiling = int(math.ceil(
            per_angle[str(selected)]["aspect_longest_over_min_altitude"]
        ))
    return {
        "schema": "RealSaS.AnimationMeshConditioningSyntheticSweep.v1",
        "authority": "SYNTHETIC_CALIBRATION_ONLY__NO_KNIGHT_OR_MAGE_MESH_INPUT",
        "angles_deg": list(ANGLES_DEG),
        "scales": list(SCALES),
        "pose_noise_fractions_of_local_scale": list(POSE_NOISE_FRACTIONS),
        "transform_ids": list(TRANSFORMS),
        "relative_metric_error_limit": MAX_ALLOWED_RELATIVE_METRIC_ERROR,
        "sample_count": sample_count,
        "per_angle": per_angle,
        "selection_rule_result": {
            "smallest_passing_min_angle_deg": selected,
            "derived_integer_aspect_ceiling": aspect_ceiling,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    args = parser.parse_args()
    report = run_sweep()
    payload = json.dumps(report, sort_keys=True, indent=2)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    return 0 if report["selection_rule_result"]["smallest_passing_min_angle_deg"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
