from __future__ import annotations

"""Materialize the sealed P1_B2_G10 temporary render mesh as exact V0..V7 artifacts.

This is intentionally NOT a new mesh experiment. It replays the already preregistered
P1_B2_G10 treatment through the exact FIT2 implementation, requires the previously
sealed per-view QualifiedEditableMeshIR lineage hashes, then serializes those meshes.

Scientific/product boundaries:
- does not claim mesh scientific PASS or PRODUCT_PASS;
- does not change GSA authority;
- does not run Arachne;
- does not transfer historical weights;
- output is the exact temporary render/deformation mesh authority for the Mage demo.
"""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.mesh.mesh_binding import qualify_supported_mesh
from compiler.realsas_compiler_core.mesh.mwb2_cdt import build_mwb2_observation_cdt_candidate
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)

import experiments.mage_full_subject_reclosure_v1.run_fit2_baseline_preserving_adaptive_patch_cdt_v1 as patch_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageDemo.P1B2G10Materialization.v1"
TREATMENT = {"name": "P1_B2_G10", "repair_hops": 1, "boundary_stride": 2, "interior_spacing": 10}
EXPECTED_MESH_LINEAGE = {
    0: "0c59f380b4817a2a05ff0489a37cf0c745b03bdf0ec219a4f41ad67abc0dd5c4",
    1: "6748278190007d6f887f5f6b2cace482d938f077bc4c4bfe25d5390ccca78c07",
    2: "eb8368c63cf73c05fb1ec017d724e3730617e595d9cce28d57273dcfa0e2be4c",
    3: "795e56aad0aa73774f5e9f7c96f655e49877b0c99f459defe8046a232cae5c49",
    4: "4f5b7a74670a1380a91048eccbcfaee1cd9beb8d4943a57d5cd1f560338bf466",
    5: "6b006a76a529ff718017ecb13bdd137aa5aba506387894e007933b98ccc0aee1",
    6: "1eacb8a0a73275f34e5c707a2c5ca7ad7add96b285b6446680c6c634d5c2ccfa",
    7: "fd4afa11aa7eda12bdf3e842d3ed90fb143d5d18f76c1087adecd3ad0c9e2c33",
}


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n", encoding="utf-8")
    tmp.replace(path)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write byte-stable ZIP_STORED npz: fixed entry order, metadata and timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[key])))
    tmp.replace(path)


def _mesh_npz_arrays(mesh) -> dict[str, np.ndarray]:
    ordered = tuple(mesh.vertices)
    ids = tuple(str(v.canonical_mesh_vertex_id) for v in ordered)
    index = {vid: i for i, vid in enumerate(ids)}
    if len(index) != len(ids):
        raise RuntimeError("P1_MATERIALIZE_DUPLICATE_CANONICAL_VERTEX_ID")
    max_id_len = max([1, *(len(x) for x in ids)])
    positions = np.asarray([v.P for v in ordered], dtype=np.float64)
    faces = np.asarray([[index[str(v)] for v in face] for face in mesh.faces], dtype=np.int64)
    edges = np.asarray([[index[str(a)], index[str(b)]] for a, b in mesh.edges], dtype=np.int64)
    raster_xy = np.asarray([v.metadata["raster_xy"] for v in ordered], dtype=np.float64)
    return {
        "canonical_mesh_vertex_id": np.asarray(ids, dtype=f"<U{max_id_len}"),
        "positions": positions,
        "faces": faces,
        "edges": edges,
        "raster_xy": raster_xy,
        "view_index": np.asarray([int(mesh.view_index)], dtype=np.int64),
    }


def _baseline_inputs(surface, obs_paths: tuple[Path, ...], camera_paths: tuple[Path, ...]):
    sealed_path = Path(patch_v1.__file__).with_name("MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    expected_by_view = {int(row["view"]): row for row in sealed["views"]}
    ceiling_v1._load_cameras(list(camera_paths))
    ceiling_v1._load_observation_masks(obs_paths)

    baselines, domains = [], []
    for view, obs_path in enumerate(obs_paths):
        with Image.open(obs_path) as im:
            source = im.convert("RGBA")
        authority = np.asarray(source, dtype=np.uint8)[..., 3] >= 8
        domain = ObservationRasterDomain.from_rows(
            authority.tolist(),
            view_index=view,
            source_alpha_sha256=ceiling_v1.OBSERVATION_SHA256[view],
        )
        baseline = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=view,
            camera_binding_hash=ceiling_v1.CAMERA_SHA256[view],
            observation_domain=domain,
        )
        checks = ceiling_v1._exact_baseline_checks(baseline, expected_by_view[view])
        if not all(bool(v) for v in checks.values()):
            raise RuntimeError(f"P1_BASELINE_REPLAY_DRIFT_V{view}:{checks}")
        baselines.append(baseline)
        domains.append(domain)
    return tuple(baselines), tuple(domains)


def run(args) -> dict:
    surface, _tensor, gsa_replay = ceiling_v2._preflight_surface(args)
    if not bool(gsa_replay.get("gsa_lineage_exact_match")):
        raise RuntimeError("P1_MATERIALIZE_GSA_LINEAGE_DRIFT")

    obs_paths = tuple(Path(p) for p in args.observations)
    camera_paths = tuple(Path(p) for p in args.cameras)
    baselines, domains = _baseline_inputs(surface, obs_paths, camera_paths)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    view_rows = []

    for view in range(8):
        candidate = patch_v1._build_hybrid_candidate(
            surface,
            baselines[view],
            view=view,
            camera_hash=ceiling_v1.CAMERA_SHA256[view],
            domain=domains[view],
            treatment=TREATMENT,
        )
        mesh = qualify_supported_mesh(surface, candidate)
        expected = EXPECTED_MESH_LINEAGE[view]
        if mesh.mesh_lineage_hash != expected:
            raise RuntimeError(f"P1_MESH_LINEAGE_DRIFT_V{view}:{mesh.mesh_lineage_hash}:{expected}")

        quality = evaluate_mesh_quality(
            coverage=dict(candidate.residual_report),
            raster_report=mesh_raster_quality_report(mesh, surface=surface, view_index=view),
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        if float(quality["precision_inside_alpha"]) < 1.0 - 1.0e-12:
            raise RuntimeError(f"P1_PRECISION_REGRESSION_V{view}:{quality['precision_inside_alpha']}")

        json_name = f"P1_B2_G10_V{view}_QUALIFIED_MESH_IR.json"
        npz_name = f"P1_B2_G10_V{view}_GEOMETRY.npz"
        json_path, npz_path = out / json_name, out / npz_name
        _write_json(json_path, mesh.to_dict())
        _write_deterministic_npz(npz_path, _mesh_npz_arrays(mesh))

        artifacts.extend(
            [
                {"path": json_name, "sha256": _sha(json_path), "bytes": json_path.stat().st_size},
                {"path": npz_name, "sha256": _sha(npz_path), "bytes": npz_path.stat().st_size},
            ]
        )
        view_rows.append(
            {
                "view": view,
                "mesh_lineage_hash": mesh.mesh_lineage_hash,
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "edge_count": len(mesh.edges),
                "source_alpha_recall": float(quality["source_alpha_recall"]),
                "precision_inside_alpha": float(quality["precision_inside_alpha"]),
                "full_frozen_policy_pass": bool(quality["passed"]),
                "failure_invariants": list(quality["failure_invariants"]),
                "qualified_mesh_json": json_name,
                "geometry_npz": npz_name,
            }
        )

    manifest = {
        "schema": SCHEMA,
        "status": "PASS__EXACT_P1_B2_G10_V0_V7_MATERIALIZED_AND_LINEAGE_VERIFIED",
        "treatment": dict(TREATMENT),
        "authority": "canonical/MAGE_DEMO_FIT1_V5_P1_EXECUTION_AUTHORITY_V1.json",
        "source_mesh_seal": "canonical/MAGE_FIT2_TEMPORARY_RENDER_MESH_P1_B2_G10_SEAL_V1.json",
        "mesh_scientific_pass_claimed": False,
        "product_pass_claimed": False,
        "gsa_lineage_relabelled": False,
        "arachne_executed": False,
        "historical_weight_transfer_used": False,
        "views": view_rows,
        "artifacts": artifacts,
    }
    manifest_path = out / "P1_B2_G10_MATERIALIZATION_MANIFEST.json"
    _write_json(manifest_path, manifest)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__EXACT_P1_B2_G10_V0_V7_MATERIALIZATION",
        "manifest": manifest_path.name,
        "manifest_sha256": _sha(manifest_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "required_mesh_lineage_hashes": {f"V{k}": v for k, v in EXPECTED_MESH_LINEAGE.items()},
        "mesh_scientific_pass_claimed": False,
        "product_pass_claimed": False,
    }
    seal_path = out / "P1_B2_G10_MATERIALIZATION_SEAL.json"
    _write_json(seal_path, seal)
    print("P1_B2_G10_MATERIALIZATION=" + json.dumps(seal, sort_keys=True), flush=True)
    return {"manifest": manifest, "seal": seal}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
