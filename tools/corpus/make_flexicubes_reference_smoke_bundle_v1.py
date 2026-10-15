from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import (
    build_dense_flexicubes_grid_v1,
)


SCHEMA = "RealSaS.FFlexiCubesReferenceInputBundle.v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_smoke_bundle(*, resolution: int, output_npz: Path, output_manifest: Path) -> None:
    if resolution < 4:
        raise ValueError("FLEXICUBES_SMOKE_RESOLUTION_TOO_SMALL")
    pitch = 1.0 / float(resolution)
    grid = build_dense_flexicubes_grid_v1(
        cell_min_ijk=(0, 0, 0),
        cell_max_exclusive_ijk=(resolution, resolution, resolution),
        lattice_origin_xyz=(-0.5, -0.5, -0.5),
        extraction_pitch=pitch,
    )
    x = np.asarray(grid.vertex_xyz, dtype=np.float32)
    scalar = np.linalg.norm(x, axis=1).astype(np.float32) - np.float32(0.30)
    cubes = np.asarray(grid.cube_fx8, dtype=np.int64)
    orientation = np.ones((x.shape[0],), dtype=np.bool_)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        x_nx3=x,
        s_n=scalar,
        cube_fx8=cubes,
        orientation_valid=orientation,
    )
    h = "0" * 64
    manifest = {
        "schema": SCHEMA,
        "subject_key": "SYNTHETIC_SPHERE_SMOKE",
        "arm": "F_A",
        "res": [resolution, resolution, resolution],
        "global_sign_authority": True,
        "support_clipping_required": False,
        "teacher_bundle_sha256": h,
        "extraction_truth_sha256": h,
        "measurement_protocol_sha256": h,
        "bundle_npz_sha256": _sha256(output_npz),
        "provenance": "DETERMINISTIC_ANALYTIC_SPHERE_SMOKE_ONLY_NOT_FIT8_EVIDENCE",
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    make_smoke_bundle(
        resolution=args.resolution,
        output_npz=args.output_npz,
        output_manifest=args.output_manifest,
    )


if __name__ == "__main__":
    main()
