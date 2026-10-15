from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from tools.corpus.fit8_f_reference_input_bundle_contract_v1 import (
    FReferenceInputBundleManifestV1,
)

OUTPUT_SCHEMA = "RealSaS.FFlexiCubesReferenceRun.v2"
UPSTREAM_REPOSITORY = "nv-tlabs/FlexiCubes"
UPSTREAM_COMMIT_SHA = "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"
UPSTREAM_FLEXICUBES_BLOB_SHA = "e799870beb7896e88c19bdcfd3171c439dbbdb13"
UPSTREAM_TABLES_BLOB_SHA = "659bca7ba1f2f22a0955e99c17c76f740954967a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load_manifest(path: Path) -> FReferenceInputBundleManifestV1:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "RealSaS.FReferenceInputBundleContract.v1":
        raise ValueError("FLEXICUBES_REFERENCE_V2_MANIFEST_SCHEMA_INVALID")
    value = FReferenceInputBundleManifestV1(
        subject_key=str(raw["subject_key"]),
        arm=str(raw["arm"]),
        authority_state=str(raw["authority_state"]),
        res=tuple(int(v) for v in raw["res"]),
        global_sign_authority=bool(raw["global_sign_authority"]),
        support_clipping_required=bool(raw["support_clipping_required"]),
        source_sha256=str(raw["source_sha256"]),
        teacher_bundle_sha256=str(raw["teacher_bundle_sha256"]),
        extraction_truth_sha256=str(raw["extraction_truth_sha256"]),
        canonical_camera_contract_sha256=str(raw["canonical_camera_contract_sha256"]),
        extraction_scale_manifest_sha256=str(raw["extraction_scale_manifest_sha256"]),
        measurement_protocol_sha256=str(raw["measurement_protocol_sha256"]),
        bundle_npz_sha256=str(raw["bundle_npz_sha256"]),
        provenance=str(raw["provenance"]),
        metadata=dict(raw.get("metadata", {})),
        schema_version=str(raw["schema_version"]),
    )
    value.validate()
    if value.arm != "F_A":
        raise ValueError("FLEXICUBES_REFERENCE_V2_ONLY_ADMITS_F_A")
    if not value.global_sign_authority:
        raise ValueError("FLEXICUBES_REFERENCE_V2_GLOBAL_SIGN_AUTHORITY_REQUIRED")
    if value.support_clipping_required:
        raise ValueError("FLEXICUBES_REFERENCE_V2_SUPPORT_CLIPPING_UNSUPPORTED")
    return value


def _validate_upstream(root: Path) -> dict[str, str]:
    if _git(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT_SHA:
        raise ValueError("FLEXICUBES_REFERENCE_V2_UPSTREAM_COMMIT_DRIFT")
    expected = {
        "flexicubes.py": UPSTREAM_FLEXICUBES_BLOB_SHA,
        "tables.py": UPSTREAM_TABLES_BLOB_SHA,
    }
    local_sha256: dict[str, str] = {}
    for name, blob_sha in expected.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"FLEXICUBES_REFERENCE_V2_UPSTREAM_FILE_MISSING:{name}")
        if _git(root, "hash-object", name) != blob_sha:
            raise ValueError(f"FLEXICUBES_REFERENCE_V2_UPSTREAM_BLOB_DRIFT:{name}")
        local_sha256[name] = _sha256(path)
    return local_sha256


def _load_bundle(path: Path, manifest: FReferenceInputBundleManifestV1) -> dict[str, np.ndarray]:
    if _sha256(path) != manifest.bundle_npz_sha256:
        raise ValueError("FLEXICUBES_REFERENCE_V2_BUNDLE_SHA_MISMATCH")
    archive = np.load(path, allow_pickle=False)
    required = {"x_nx3", "s_n", "cube_fx8", "orientation_valid"}
    if not required <= set(archive.files):
        raise ValueError("FLEXICUBES_REFERENCE_V2_BUNDLE_ARRAY_MISSING")
    arrays = {name: archive[name] for name in archive.files}
    x = np.asarray(arrays["x_nx3"])
    s = np.asarray(arrays["s_n"])
    cubes = np.asarray(arrays["cube_fx8"])
    orientation = np.asarray(arrays["orientation_valid"])
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("FLEXICUBES_REFERENCE_V2_X_SHAPE_INVALID")
    if s.shape != (x.shape[0],):
        raise ValueError("FLEXICUBES_REFERENCE_V2_S_SHAPE_INVALID")
    if orientation.shape != (x.shape[0],) or not np.all(orientation):
        raise ValueError("FLEXICUBES_REFERENCE_V2_ORIENTATION_INVALID")
    if cubes.ndim != 2 or cubes.shape[1] != 8:
        raise ValueError("FLEXICUBES_REFERENCE_V2_CUBE_SHAPE_INVALID")
    if cubes.shape[0] != int(np.prod(np.asarray(manifest.res, dtype=np.int64))):
        raise ValueError("FLEXICUBES_REFERENCE_V2_DENSE_CUBE_COUNT_REQUIRED")
    if cubes.size and (cubes.min() < 0 or cubes.max() >= x.shape[0]):
        raise ValueError("FLEXICUBES_REFERENCE_V2_CUBE_INDEX_OOB")
    if not np.isfinite(x).all() or not np.isfinite(s).all():
        raise ValueError("FLEXICUBES_REFERENCE_V2_INPUT_NONFINITE")
    for optional, width in (("beta_fx12", 12), ("alpha_fx8", 8)):
        if optional in arrays:
            value = np.asarray(arrays[optional])
            if value.shape != (cubes.shape[0], width) or not np.isfinite(value).all():
                raise ValueError(f"FLEXICUBES_REFERENCE_V2_OPTIONAL_SHAPE_INVALID:{optional}")
    if "gamma_f" in arrays:
        gamma = np.asarray(arrays["gamma_f"])
        if gamma.shape != (cubes.shape[0],) or not np.isfinite(gamma).all():
            raise ValueError("FLEXICUBES_REFERENCE_V2_OPTIONAL_SHAPE_INVALID:gamma_f")
    return arrays


def _tensor(arrays: dict[str, np.ndarray], name: str, *, device: str, dtype: torch.dtype):
    if name not in arrays:
        return None
    return torch.as_tensor(arrays[name], device=device, dtype=dtype)


def run_reference_v2(
    *,
    bundle_npz: Path,
    bundle_manifest: Path,
    upstream_root: Path,
    output_npz: Path,
    output_manifest: Path,
    device: str,
) -> dict[str, Any]:
    manifest = _load_manifest(bundle_manifest)
    upstream_sha256 = _validate_upstream(upstream_root)
    arrays = _load_bundle(bundle_npz, manifest)

    sys.path.insert(0, str(upstream_root))
    sys.modules.pop("tables", None)
    sys.modules.pop("flexicubes", None)
    FlexiCubes = getattr(importlib.import_module("flexicubes"), "FlexiCubes")
    extractor = FlexiCubes(device=device)

    x = _tensor(arrays, "x_nx3", device=device, dtype=torch.float32)
    s = _tensor(arrays, "s_n", device=device, dtype=torch.float32)
    cubes = _tensor(arrays, "cube_fx8", device=device, dtype=torch.long)
    beta = _tensor(arrays, "beta_fx12", device=device, dtype=torch.float32)
    alpha = _tensor(arrays, "alpha_fx8", device=device, dtype=torch.float32)
    gamma = _tensor(arrays, "gamma_f", device=device, dtype=torch.float32)

    with torch.no_grad():
        vertices, faces, regularizer = extractor(
            x,
            s,
            cubes,
            list(manifest.res),
            beta_fx12=beta,
            alpha_fx8=alpha,
            gamma_f=gamma,
            training=False,
            output_tetmesh=False,
            grad_func=None,
        )

    vertices_np = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    regularizer_np = regularizer.detach().cpu().numpy().astype(np.float32, copy=False)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, vertices=vertices_np, faces=faces_np, regularizer=regularizer_np)

    result = {
        "schema": OUTPUT_SCHEMA,
        "subject_key": manifest.subject_key,
        "arm": manifest.arm,
        "authority_state": manifest.authority_state,
        "matched_court_eligible": manifest.matched_court_eligible,
        "input_manifest_sha256": manifest.manifest_sha256(),
        "input_bundle_npz_sha256": manifest.bundle_npz_sha256,
        "source_sha256": manifest.source_sha256,
        "teacher_bundle_sha256": manifest.teacher_bundle_sha256,
        "extraction_truth_sha256": manifest.extraction_truth_sha256,
        "canonical_camera_contract_sha256": manifest.canonical_camera_contract_sha256,
        "extraction_scale_manifest_sha256": manifest.extraction_scale_manifest_sha256,
        "measurement_protocol_sha256": manifest.measurement_protocol_sha256,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit_sha": UPSTREAM_COMMIT_SHA,
        "upstream_file_sha256": upstream_sha256,
        "device": device,
        "res": list(manifest.res),
        "candidate_vertex_count": int(vertices_np.shape[0]),
        "candidate_face_count": int(faces_np.shape[0]),
        "regularizer_count": int(regularizer_np.size),
        "candidate_npz_sha256": _sha256(output_npz),
        "provenance": "EXACT_BOUND_UPSTREAM_FLEXICUBES_REFERENCE_RUN_V2",
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-npz", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_reference_v2(
        bundle_npz=args.bundle_npz,
        bundle_manifest=args.bundle_manifest,
        upstream_root=args.upstream_root,
        output_npz=args.output_npz,
        output_manifest=args.output_manifest,
        device=args.device,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
