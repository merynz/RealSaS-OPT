from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.corpus.fit8_f_reference_dense_grid_authority_v1 import (
    FReferenceDenseGridAuthorityV1,
    validate_dense_grid_bundle_arrays_v1,
)
from tools.corpus.run_fit8_f_flexicubes_reference_v2 import run_reference_v2

OUTPUT_SCHEMA = "RealSaS.FFlexiCubesReferenceRun.v3"


def _load_grid_authority(path: Path) -> FReferenceDenseGridAuthorityV1:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    value = FReferenceDenseGridAuthorityV1(
        cell_min_ijk=tuple(int(v) for v in raw["cell_min_ijk"]),
        cell_max_exclusive_ijk=tuple(int(v) for v in raw["cell_max_exclusive_ijk"]),
        lattice_origin_xyz=tuple(float(v) for v in raw["lattice_origin_xyz"]),
        extraction_pitch=float(raw["extraction_pitch"]),
        cube_ordering=str(raw["cube_ordering"]),
        scalar_sign_convention=str(raw["scalar_sign_convention"]),
        vertex_position_mode=str(raw["vertex_position_mode"]),
        bundle_npz_sha256=str(raw["bundle_npz_sha256"]),
        provenance=str(raw["provenance"]),
        metadata=dict(raw.get("metadata", {})),
        schema_version=str(raw["schema_version"]),
    )
    value.validate()
    return value


def _prevalidate_grid(
    *,
    bundle_npz: Path,
    bundle_manifest: Path,
    grid_authority_path: Path,
) -> FReferenceDenseGridAuthorityV1:
    manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    authority = _load_grid_authority(grid_authority_path)
    manifest_bundle_sha = str(manifest["bundle_npz_sha256"])
    if authority.bundle_npz_sha256 != manifest_bundle_sha:
        raise ValueError("FLEXICUBES_REFERENCE_V3_GRID_BUNDLE_SHA_DRIFT")
    archive = np.load(bundle_npz, allow_pickle=False)
    arrays = {name: archive[name] for name in archive.files}
    validate_dense_grid_bundle_arrays_v1(
        authority,
        arrays,
        manifest_res=tuple(int(v) for v in manifest["res"]),
    )
    return authority


def run_reference_v3(
    *,
    bundle_npz: Path,
    bundle_manifest: Path,
    grid_authority: Path,
    upstream_root: Path,
    output_npz: Path,
    output_manifest: Path,
    device: str,
) -> dict[str, Any]:
    authority = _prevalidate_grid(
        bundle_npz=bundle_npz,
        bundle_manifest=bundle_manifest,
        grid_authority_path=grid_authority,
    )
    result = run_reference_v2(
        bundle_npz=bundle_npz,
        bundle_manifest=bundle_manifest,
        upstream_root=upstream_root,
        output_npz=output_npz,
        output_manifest=output_manifest,
        device=device,
    )
    result = dict(result)
    result["schema"] = OUTPUT_SCHEMA
    result["dense_grid_authority_sha256"] = authority.sha256()
    result["dense_grid_authority_provenance"] = authority.provenance
    output_manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-npz", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--grid-authority", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_reference_v3(
        bundle_npz=args.bundle_npz,
        bundle_manifest=args.bundle_manifest,
        grid_authority=args.grid_authority,
        upstream_root=args.upstream_root,
        output_npz=args.output_npz,
        output_manifest=args.output_manifest,
        device=args.device,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
