from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

rest = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

# Proper rigid rotation: 180 degrees around +X. det(R)=+1.
R = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)
posed = rest @ R.T

def edges(x):
    return np.asarray(
        [
            np.linalg.norm(x[1] - x[0]),
            np.linalg.norm(x[2] - x[1]),
            np.linalg.norm(x[0] - x[2]),
        ],
        dtype=np.float64,
    )

def area3(x):
    return 0.5 * float(
        np.linalg.norm(np.cross(x[1] - x[0], x[2] - x[0]))
    )

def signed_area_xy(x):
    a = x[1, :2] - x[0, :2]
    b = x[2, :2] - x[0, :2]
    return 0.5 * float(a[0] * b[1] - a[1] * b[0])

rest_n = np.cross(rest[1] - rest[0], rest[2] - rest[0])
posed_n = np.cross(posed[1] - posed[0], posed[2] - posed[0])
normal_dot = float(np.dot(rest_n, posed_n))
rest_e = edges(rest)
posed_e = edges(posed)
edge_ratio = posed_e / rest_e
area_ratio = area3(posed) / area3(rest)
rest_proj_area = signed_area_xy(rest)
posed_proj_area = signed_area_xy(posed)
projected_winding_flip = bool(rest_proj_area * posed_proj_area < 0.0)

orthogonality_error = float(np.linalg.norm(R.T @ R - np.eye(3)))
rotation_det = float(np.linalg.det(R))

# 2D deformation singular values in a co-rotated material frame are exactly 1.
# Applying R^-1 to the posed triangle recovers rest exactly.
corotated = posed @ R
corotated_error = float(np.max(np.abs(corotated - rest)))

payload = {
    "schema": "RealSaS.DeformationWitnessValidityAudit.v1",
    "status": "AUDIT_CORRECTION__NO_REPAIR_APPLIED",
    "repo_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "construction": (
        "A unit triangle undergoes a proper 180-degree rigid rotation around X. "
        "No material stretch, compression, reflection, or non-rigid deformation occurs."
    ),
    "proper_rigid_transform": {
        "matrix": R.tolist(),
        "determinant": rotation_det,
        "orthogonality_error": orthogonality_error,
        "corotated_reconstruction_max_abs_error": corotated_error,
    },
    "intrinsic_deformation": {
        "edge_ratios": edge_ratio.tolist(),
        "area_ratio_3d_unsigned": area_ratio,
        "max_edge_ratio_error_from_one": float(np.max(np.abs(edge_ratio - 1.0))),
    },
    "orientation_witnesses": {
        "rest_normal": rest_n.tolist(),
        "posed_normal": posed_n.tolist(),
        "rest_normal_dot_posed_normal": normal_dot,
        "normal_dot_declares_flip": bool(normal_dot < 0.0),
        "rest_xy_signed_area": rest_proj_area,
        "posed_xy_signed_area": posed_proj_area,
        "fixed_xy_projection_declares_winding_flip": projected_winding_flip,
    },
    "findings": [
        {
            "id": "REST_NORMAL_DOT_IS_NOT_INTRINSIC_SURFACE_INVERSION_PROOF",
            "confirmed": bool(
                rotation_det > 0.0
                and orthogonality_error < 1e-12
                and corotated_error < 1e-12
                and normal_dot < 0.0
                and np.max(np.abs(edge_ratio - 1.0)) < 1e-12
                and abs(area_ratio - 1.0) < 1e-12
            ),
            "severity": "P0_AUDIT_WITNESS_CORRECTION",
            "consequence": (
                "A face count based only on dot(rest_normal, posed_normal)<0 "
                "cannot be used as proof of mechanical inversion in a 3D surface."
            ),
        },
        {
            "id": "FIXED_CAMERA_PROJECTED_WINDING_FLIP_IS_NOT_INTRINSIC_INVERSION_PROOF",
            "confirmed": bool(
                rotation_det > 0.0
                and projected_winding_flip
                and np.max(np.abs(edge_ratio - 1.0)) < 1e-12
                and abs(area_ratio - 1.0) < 1e-12
            ),
            "severity": "P0_AUDIT_WITNESS_CORRECTION",
            "consequence": (
                "Projected winding changes are useful visibility/render witnesses, "
                "but a rigid 3D rotation through edge-on can flip projected winding "
                "without any material deformation."
            ),
        },
        {
            "id": "PRIOR_REFLECTION_ADVERSARY_DOES_NOT_DISTINGUISH_2D_REFLECTION_FROM_3D_PROPER_ROTATION",
            "confirmed": bool(
                np.allclose(
                    posed,
                    np.asarray(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                        dtype=np.float64,
                    ),
                )
                and rotation_det > 0.0
            ),
            "severity": "P0_AUDIT_AUTHORITY_CORRECTION",
            "consequence": (
                "The earlier G3/Stage35 single-triangle 'reflection' adversary "
                "proves those scalar metrics are unsigned, but does not prove an "
                "intrinsic 3D surface inversion was missed. Any no-foldover contract "
                "needs a transported/local-neighborhood or self-intersection definition."
            ),
        },
    ],
    "authority_rule": {
        "for_remaining_knight_owner_audit": [
            "Do not use rest-normal sign as mechanical guilt.",
            "Do not use fixed-camera projected winding sign as mechanical guilt.",
            "Use rotation-invariant intrinsic edge/area/condition metrics for stretch/compression.",
            "Treat screen-space growth as visual consequence evidence, not intrinsic inversion proof.",
            "Use explicit neighborhood fold/self-intersection evidence if a foldover claim is required.",
        ],
        "affected_previous_evidence": [
            "canonical/G3_ORIENTATION_INVERSION_ADVERSARY_V1_20260928.json",
            "canonical/G3_ORIENTATION_SIGN_INFORMATION_LOSS_AUDIT_V1_20260928.json",
            "canonical/STAGE35_ORIENTATION_INVERSION_ADVERSARY_V1_20260928.json",
            "canonical/knight_baseline_forensic_capsule_v1/BASELINE_FORENSIC_CAPSULE_V1.json projected flipped_face_count fields",
            "canonical/knight_oracle_topology_repair_causality_v1/REPORT.json flipped_face and flip-derived verdict terms",
        ],
    },
}

if not all(row["confirmed"] for row in payload["findings"]):
    raise SystemExit("DEFORMATION_WITNESS_VALIDITY_AUDIT_NOT_CONFIRMED")

out = ROOT / "canonical" / "DEFORMATION_WITNESS_VALIDITY_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
