from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    _triangle_metrics_batch,
)

ROOT = Path(__file__).resolve().parents[1]

rest = np.asarray(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    dtype=np.float64,
)
posed = np.asarray(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]],
    dtype=np.float64,
)
faces = np.asarray([[0, 1, 2]], dtype=np.int64)

# _triangle_metrics_batch expects vertex arrays plus face indices.
rest_vertices = rest[0]
posed_vertices = posed[0]
area, condition, edge_min, edge_max = _triangle_metrics_batch(
    rest_vertices,
    posed_vertices,
    faces,
)

rest_n = np.cross(
    rest_vertices[1] - rest_vertices[0],
    rest_vertices[2] - rest_vertices[0],
)
posed_n = np.cross(
    posed_vertices[1] - posed_vertices[0],
    posed_vertices[2] - posed_vertices[0],
)
orientation_dot = float(np.dot(rest_n, posed_n))
inverted = bool(orientation_dot < 0.0)

metric_identity = bool(
    abs(float(area[0]) - 1.0) < 1e-12
    and abs(float(condition[0]) - 1.0) < 1e-12
    and abs(float(edge_min[0]) - 1.0) < 1e-12
    and abs(float(edge_max[0]) - 1.0) < 1e-12
)

payload = {
    "schema": "RealSaS.Stage35OrientationInversionAdversary.v1",
    "status": "AUDIT_ONLY__NO_REPAIR_APPLIED",
    "repo_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "construction": (
        "unit right triangle reflected across X in its rest tangent plane; "
        "edge lengths and singular values are unchanged while orientation reverses"
    ),
    "stage35_current_metrics": {
        "area_ratio": float(area[0]),
        "condition_number": float(condition[0]),
        "min_edge_ratio": float(edge_min[0]),
        "max_edge_ratio": float(edge_max[0]),
    },
    "orientation": {
        "rest_normal": rest_n.tolist(),
        "posed_normal": posed_n.tolist(),
        "normal_dot": orientation_dot,
        "inverted": inverted,
    },
    "finding": {
        "id": "STAGE35_SKIN_TOPOLOGY_METRICS_DROP_ORIENTATION_SIGN",
        "confirmed": bool(inverted and metric_identity),
        "class": "IMPLEMENTATION_DEVIATION_FROM_DYNAMIC_NO_INVERSION_CONTRACT",
        "severity": "P0",
        "owner": "STAGE35_SKIN_TOPOLOGY_COMPATIBILITY",
        "mechanism": (
            "Stage35 compatibility uses singular-value area/condition plus unsigned "
            "edge ratios. A reflection has identical values to an isometry, so a "
            "pure inversion can pass these metrics unless another independent gate "
            "catches it."
        ),
        "claim_boundary": (
            "This adversary proves an information-loss class in the Stage35 "
            "compatibility metric. It does not claim the known Knight smear offenders "
            "were missed; those are already classified unsafe by other distortion terms."
        ),
    },
}

out = ROOT / "canonical" / "STAGE35_ORIENTATION_INVERSION_ADVERSARY_V1_20260928.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))

if not payload["finding"]["confirmed"]:
    raise SystemExit("STAGE35_ORIENTATION_INVERSION_ADVERSARY_NOT_CONFIRMED")
