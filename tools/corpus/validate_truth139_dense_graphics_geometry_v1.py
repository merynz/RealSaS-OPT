#!/usr/bin/env python3
"""Validate canonical dense Graphics front-surface rows with the independent CPU ray oracle.

The underlying geometric measurement is intentionally identical to the sealed legacy dense G-buffer
court.  This adapter changes only the parent schema/status/identity vocabulary from legacy
Appearance to canonical Graphics; it does not weaken or reinterpret the ray, barycentric, depth or
surface-point checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus import validate_truth139_dense_gbuffer_geometry_v1 as legacy

SCHEMA = "RealSaS.Truth139DenseGraphicsGeometryCourt.v1"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsGBuffer.v1"
GRAPHICS_STATUS = "PASS_DENSE_GRAPHICS_FRONT_SURFACE__VISIBILITY_OCCLUSION_COURT_PENDING"
PASS_STATUS = "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate(
    manifest: dict,
    *,
    maximum_samples_per_view: int = 64,
    minimum_validated_per_view: int = 16,
    interior_barycentric_floor: float = 1.0e-4,
    barycentric_abs_tolerance: float = 5.0e-4,
    depth_relative_tolerance: float = 2.0e-5,
) -> dict:
    if manifest.get("schema") != GRAPHICS_SCHEMA:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_COURT_SCHEMA")
    if manifest.get("status") != GRAPHICS_STATUS:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_COURT_PARENT_NOT_PENDING_PASS")
    graphics_identity = str(manifest.get("graphics_identity_sha256") or "")
    if len(graphics_identity) != 64:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_COURT_IDENTITY")

    # Reuse the already independent CPU ray court after a vocabulary-only compatibility projection.
    projected = dict(manifest)
    projected["schema"] = legacy.GBUFFER_SCHEMA
    projected["status"] = "PASS_DENSE_SURFACE_ADDRESSED_APPEARANCE_GBUFFER__STAGE48_INTERVENTION_WITNESS_STILL_REQUIRED"
    projected["authority_identity_sha256"] = graphics_identity
    result = legacy.validate(
        projected,
        maximum_samples_per_view=maximum_samples_per_view,
        minimum_validated_per_view=minimum_validated_per_view,
        interior_barycentric_floor=interior_barycentric_floor,
        barycentric_abs_tolerance=barycentric_abs_tolerance,
        depth_relative_tolerance=depth_relative_tolerance,
    )
    if result.get("status") != PASS_STATUS:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_COURT_LEGACY_MEASUREMENT_NOT_PASS")

    result = dict(result)
    result["schema"] = SCHEMA
    result["dense_graphics_identity_sha256"] = graphics_identity
    result.pop("dense_gbuffer_authority_identity_sha256", None)
    result["compatibility_projection"] = {
        "kind": "VOCABULARY_ONLY__MEASUREMENT_UNCHANGED",
        "legacy_measurement_module": "tools/corpus/validate_truth139_dense_gbuffer_geometry_v1.py",
        "legacy_parent_schema_used_internally": legacy.GBUFFER_SCHEMA,
        "material_rgb_semantic_or_owner_evidence_used": False,
    }
    result.pop("court_identity_sha256", None)
    result["court_identity_sha256"] = _digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-graphics-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-samples-per-view", type=int, default=64)
    parser.add_argument("--minimum-validated-per-view", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.dense_graphics_manifest.resolve().read_text())
    result = validate(
        manifest,
        maximum_samples_per_view=args.maximum_samples_per_view,
        minimum_validated_per_view=args.minimum_validated_per_view,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "total_validated_pixel_count": result["total_validated_pixel_count"],
        "court_identity_sha256": result["court_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
