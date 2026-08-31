from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gate0_geometry_v1 import (
    hull_frontier,
    make_dense_lattice,
    orbit_camera,
    robust_multiview_evidence,
    streaming_chunk_plan,
    thickness_cell_strata,
)


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sphere_masks(resolution: int = 128, radius: float = 0.30, half_extent: float = 0.54):
    yy, xx = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    gx = 2.0 * (xx + 0.5) / resolution - 1.0
    gy = 2.0 * (yy + 0.5) / resolution - 1.0
    sx = gx * half_extent
    sz = -gy * half_extent
    m = sx * sx + sz * sz <= radius * radius
    return [m.copy() for _ in range(8)]


def main() -> None:
    rng = np.random.default_rng(20260831)
    cams = [orbit_camera(45 * v) for v in range(8)]

    u = rng.normal(size=(4000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    truth = 0.30 * u
    lattice, shape = make_dense_lattice(((-0.40, 0.40), (-0.40, 0.40), (-0.40, 0.40)), 0.04)
    frontier = hull_frontier(truth, lattice, cams, sphere_masks(), paddings_px=(0, 1, 2, 4, 8))

    descriptor_base = rng.normal(size=(32,))
    descriptors = np.stack([descriptor_base + 0.01 * rng.normal(size=(32,)) for _ in range(7)] + [-descriptor_base], axis=0)
    evidence = robust_multiview_evidence(descriptors, trim_fraction=0.25)

    thickness = {
        str(s): thickness_cell_strata(np.asarray([0.002, 0.004, 0.008, 0.012, 0.020, 0.040]), s)
        for s in (0.016, 0.008, 0.004)
    }
    chunk = streaming_chunk_plan(2_000_000, 8, 64, 64 * 1024 * 1024)

    result = {
        "schema": "RealSaS.IRIS.ReprojectionV2.Gate0SyntheticPreflight.v1",
        "status": "PASS_SYNTHETIC_GATE0" if frontier[-1]["truth_containment"] >= 0.99 and evidence["cosine_trimmed_mean"] > 0.95 else "FAIL_SYNTHETIC_GATE0",
        "seed": 20260831,
        "synthetic_lattice_shape_zyx": list(shape),
        "hull_frontier": frontier,
        "thin_structure_spacing_sweep": thickness,
        "robust_evidence": evidence,
        "streaming_plan_example": chunk,
        "scientific_optimizer_steps": 0,
        "training_authorized": False,
    }
    result["content_sha256"] = canonical_sha(result)
    out = Path(__file__).with_name("GATE0_SYNTHETIC_PREFLIGHT_RESULT_V1.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_SYNTHETIC_GATE0":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
