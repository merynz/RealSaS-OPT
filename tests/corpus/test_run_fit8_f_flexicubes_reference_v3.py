import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import build_dense_flexicubes_grid_v1
from tools.corpus.fit8_f_reference_dense_grid_authority_v1 import (
    CUBE_ORDERING,
    REGULAR_LATTICE,
    SCALAR_SIGN_CONVENTION,
)
from tools.corpus.run_fit8_f_flexicubes_reference_v3 import _prevalidate_grid


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    grid = build_dense_flexicubes_grid_v1(
        cell_min_ijk=(0, 0, 0),
        cell_max_exclusive_ijk=(2, 2, 2),
        lattice_origin_xyz=(-0.5, -0.5, -0.5),
        extraction_pitch=0.5,
    )
    bundle = tmp_path / "input.npz"
    np.savez_compressed(
        bundle,
        x_nx3=np.asarray(grid.vertex_xyz, dtype=np.float32),
        s_n=np.ones((len(grid.vertex_xyz),), dtype=np.float32),
        cube_fx8=np.asarray(grid.cube_fx8, dtype=np.int64),
        orientation_valid=np.ones((len(grid.vertex_xyz),), dtype=bool),
    )
    digest = _sha(bundle)
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"res": [2, 2, 2], "bundle_npz_sha256": digest}))
    authority = tmp_path / "grid.json"
    authority.write_text(json.dumps({
        "schema_version": "RealSaS.FReferenceDenseGridAuthority.v1",
        "cell_min_ijk": [0, 0, 0],
        "cell_max_exclusive_ijk": [2, 2, 2],
        "lattice_origin_xyz": [-0.5, -0.5, -0.5],
        "extraction_pitch": 0.5,
        "cube_ordering": CUBE_ORDERING,
        "scalar_sign_convention": SCALAR_SIGN_CONVENTION,
        "vertex_position_mode": REGULAR_LATTICE,
        "bundle_npz_sha256": digest,
        "provenance": "synthetic_guard",
        "metadata": {},
    }))
    return bundle, manifest, authority


def test_v3_prevalidation_accepts_exact_grid(tmp_path):
    bundle, manifest, authority = _fixture(tmp_path)
    value = _prevalidate_grid(
        bundle_npz=bundle,
        bundle_manifest=manifest,
        grid_authority_path=authority,
    )
    assert value.res == (2, 2, 2)


def test_v3_rejects_grid_authority_bound_to_other_bundle(tmp_path):
    bundle, manifest, authority = _fixture(tmp_path)
    raw = json.loads(authority.read_text())
    raw["bundle_npz_sha256"] = "b" * 64
    authority.write_text(json.dumps(raw))
    try:
        _prevalidate_grid(
            bundle_npz=bundle,
            bundle_manifest=manifest,
            grid_authority_path=authority,
        )
    except ValueError as exc:
        assert "GRID_BUNDLE_SHA_DRIFT" in str(exc)
    else:
        raise AssertionError("expected grid/bundle binding rejection")


def test_v3_rejects_cube_order_drift(tmp_path):
    bundle, manifest, authority = _fixture(tmp_path)
    data = np.load(bundle, allow_pickle=False)
    arrays = {name: data[name] for name in data.files}
    arrays["cube_fx8"] = arrays["cube_fx8"].copy()
    arrays["cube_fx8"][[0, 1]] = arrays["cube_fx8"][[1, 0]]
    np.savez_compressed(bundle, **arrays)
    new_digest = _sha(bundle)
    raw_manifest = json.loads(manifest.read_text())
    raw_manifest["bundle_npz_sha256"] = new_digest
    manifest.write_text(json.dumps(raw_manifest))
    raw_authority = json.loads(authority.read_text())
    raw_authority["bundle_npz_sha256"] = new_digest
    authority.write_text(json.dumps(raw_authority))
    try:
        _prevalidate_grid(
            bundle_npz=bundle,
            bundle_manifest=manifest,
            grid_authority_path=authority,
        )
    except ValueError as exc:
        assert "CUBE_ORDER_DRIFT" in str(exc)
    else:
        raise AssertionError("expected exact cube ordering rejection")
