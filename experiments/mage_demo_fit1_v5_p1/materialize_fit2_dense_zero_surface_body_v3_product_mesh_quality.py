from __future__ import annotations

"""Dense BODY treatment with final-motion and frozen product mesh-quality admission.

V2 already binds face admission to the exact final historical motion state. V3 adds
one missing product invariant before coverage selection: every admitted original dense
zero-surface face must also satisfy the frozen raster triangle angle/aspect policy.
The final materialized mesh is remeasured with the official mesh-quality implementation.
No retriangulation, new adjacency, training, model query or threshold relaxation occurs.
"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    mesh_raster_quality_report,
    raster_quality_gate_failures,
)

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_fit2_dense_zero_surface_body_v1 as v1
import experiments.mage_demo_fit1_v5_p1.materialize_fit2_dense_zero_surface_body_v2_historical_motion as v2
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base

SCHEMA = "RealSaS.MageFIT2.DenseZeroSurfaceBody.v3.final_motion_product_mesh_quality"
STATUS = v2.STATUS


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _static_quality_mask(dense_xy: np.ndarray, dense_faces: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    face_ids = np.asarray(face_ids, dtype=np.int64)
    tri = np.asarray(dense_xy, dtype=np.float64)[np.asarray(dense_faces, dtype=np.int64)[face_ids]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def angle(x, y, z):
        cosine = np.clip((x*x + y*y - z*z) / np.maximum(2.0*x*y, 1e-18), -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    min_angle = np.minimum.reduce((angle(ab, ca, bc), angle(ab, bc, ca), angle(bc, ca, ab)))
    longest = np.maximum.reduce((ab, bc, ca))
    altitude = (2.0 * area) / np.maximum(longest, 1e-18)
    aspect = longest / np.maximum(altitude, 1e-18)
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    return (
        (area > 1e-12)
        & (min_angle >= float(policy.min_raster_triangle_angle_deg) - 1e-12)
        & (aspect <= float(policy.max_raster_triangle_aspect_ratio) + 1e-12)
    )


def run(args) -> dict:
    # V2 calls v1._select_coverage_faces exactly once per canonical view in order.
    # Narrowly intercept that seam to remove static raster-policy-invalid source faces
    # before the unchanged precision-constrained coverage selector runs.
    original_selector = v1._select_coverage_faces
    filter_rows = []

    def quality_filtered_selector(dense_xy, dense_faces, safe_ids, observation, rigid_mask):
        view = len(filter_rows)
        ids = np.asarray(safe_ids, dtype=np.int64)
        mask = _static_quality_mask(dense_xy, dense_faces, ids)
        filtered = ids[mask]
        filter_rows.append({
            "view": view,
            "dynamic_safe_face_count": int(len(ids)),
            "product_mesh_quality_safe_face_count": int(len(filtered)),
            "product_mesh_quality_rejected_face_count": int(len(ids) - len(filtered)),
        })
        print(
            f"V{view} product_mesh_quality_safe_faces={len(filtered)}/{len(ids)} "
            f"rejected={len(ids)-len(filtered)}",
            flush=True,
        )
        return original_selector(dense_xy, dense_faces, filtered, observation, rigid_mask)

    v1._select_coverage_faces = quality_filtered_selector
    try:
        result = v2.run(args)
    finally:
        v1._select_coverage_faces = original_selector

    if len(filter_rows) != 8:
        raise RuntimeError(f"DENSE_BODY_V3_STATIC_QUALITY_VIEW_CALL_DRIFT:{len(filter_rows)}")

    out = Path(args.output_dir)
    manifest_path = out / "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    surface = fit2io.load_surface(Path(args.fit2_surface))
    rows = {int(row["view"]): row for row in manifest.get("views") or ()}
    if set(rows) != set(range(8)):
        raise RuntimeError("DENSE_BODY_V3_REQUIRES_8_VIEWS")

    artifact_rows = list(manifest.get("artifacts") or ())
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    for view in range(8):
        row = rows[view]
        mesh_path = out / row["files"]["mesh"]
        mesh = v5base._load_mesh(mesh_path, surface, str(row["mesh_lineage_hash"]))
        report = mesh_raster_quality_report(mesh, surface=None, view_index=view)
        failures = tuple(raster_quality_gate_failures(report, policy=policy))
        if failures:
            raise RuntimeError(f"DENSE_BODY_V3_PRODUCT_MESH_QUALITY_FAIL_V{view}:{failures}:{report}")
        witness = {
            "schema": SCHEMA + ".ProductMeshQualityWitness.v1",
            "view": view,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "policy": policy.to_dict(),
            "report": report,
            "failure_invariants": [],
            "passed": True,
            **filter_rows[view],
        }
        witness_name = f"V{view}_DENSE_BODY_PRODUCT_MESH_QUALITY_WITNESS.json"
        witness_path = out / witness_name
        _write_json(witness_path, witness)
        row["files"]["mesh_quality"] = witness_name
        row["product_mesh_quality"] = witness
        artifact = {"path": witness_name, "sha256": _sha(witness_path), "bytes": witness_path.stat().st_size}
        row.setdefault("artifacts", []).append(artifact)
        artifact_rows.append(artifact)

    manifest.update({
        "schema": SCHEMA,
        "status": STATUS,
        "views": [rows[i] for i in range(8)],
        "artifacts": artifact_rows,
        "static_product_mesh_quality_gate_applied": True,
        "static_product_mesh_quality_policy": policy.to_dict(),
        "static_product_mesh_quality_all_views_passed": True,
        "threshold_relaxation_used": False,
        "product_pass_claimed": False,
    })
    _write_json(manifest_path, manifest)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY_FINAL_MOTION_PRODUCT_MESH_QUALITY",
        "manifest": manifest_path.name,
        "manifest_sha256": _sha(manifest_path),
        "artifact_count": len(artifact_rows),
        "surface_lineage_hash": manifest["surface_lineage_hash"],
        "skeleton_lineage_hash": manifest["skeleton_lineage_hash"],
        "skin_lineage_hash": manifest["skin_lineage_hash"],
        "motion_state_hash": manifest["motion_state_hash"],
        "static_product_mesh_quality_all_views_passed": True,
        "product_pass_claimed": False,
    }
    _write_json(out / "FIT2_DENSE_ZERO_SURFACE_BODY_SEAL.json", seal)
    print("DENSE_BODY_V3_MATERIALIZATION=" + json.dumps(seal, sort_keys=True), flush=True)
    return {"manifest": manifest, "seal": seal}


def parse_args():
    return v2.parse_args()


if __name__ == "__main__":
    run(parse_args())
