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


INPUT_SCHEMA = "RealSaS.FFlexiCubesReferenceInputBundle.v1"
OUTPUT_SCHEMA = "RealSaS.FFlexiCubesReferenceRun.v1"
UPSTREAM_REPOSITORY = "nv-tlabs/FlexiCubes"
UPSTREAM_COMMIT_SHA = "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"
UPSTREAM_FLEXICUBES_BLOB_SHA = "e799870beb7896e88c19bdcfd3171c439dbbdb13"
UPSTREAM_TABLES_BLOB_SHA = "659bca7ba1f2f22a0955e99c17c76f740954967a"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
    ).strip()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != INPUT_SCHEMA:
        raise ValueError("FLEXICUBES_REFERENCE_INPUT_SCHEMA_INVALID")
    if value.get("arm") != "F_A":
        raise ValueError("FLEXICUBES_REFERENCE_V1_ONLY_ADMITS_F_A")
    if not value.get("subject_key"):
        raise ValueError("FLEXICUBES_REFERENCE_SUBJECT_MISSING")
    if value.get("global_sign_authority") is not True:
        raise ValueError("FLEXICUBES_REFERENCE_GLOBAL_SIGN_AUTHORITY_REQUIRED")
    if value.get("support_clipping_required") is not False:
        raise ValueError("FLEXICUBES_REFERENCE_SUPPORT_CLIPPING_UNSUPPORTED")
    res = value.get("res")
    if not isinstance(res, list) or len(res) != 3 or any(int(v) <= 0 for v in res):
        raise ValueError("FLEXICUBES_REFERENCE_RES_INVALID")
    for name in (
        "teacher_bundle_sha256",
        "extraction_truth_sha256",
        "measurement_protocol_sha256",
        "bundle_npz_sha256",
    ):
        field = str(value.get(name, ""))
        if len(field) != 64:
            raise ValueError(f"FLEXICUBES_REFERENCE_SHA_INVALID:{name}")
    if not value.get("provenance"):
        raise ValueError("FLEXICUBES_REFERENCE_PROVENANCE_MISSING")
    return value


def _validate_upstream(root: Path) -> dict[str, str]:
    if _git(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT_SHA:
        raise ValueError("FLEXICUBES_REFERENCE_UPSTREAM_COMMIT_DRIFT")
    files = {
        "flexicubes.py": UPSTREAM_FLEXICUBES_BLOB_SHA,
        "tables.py": UPSTREAM_TABLES_BLOB_SHA,
    }
    sha256: dict[str, str] = {}
    for name, expected_blob in files.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"FLEXICUBES_REFERENCE_UPSTREAM_FILE_MISSING:{name}")
        if _git(root, "hash-object", name) != expected_blob:
            raise ValueError(f"FLEXICUBES_REFERENCE_UPSTREAM_BLOB_DRIFT:{name}")
        sha256[name] = _sha256(path)
    return sha256


def _load_bundle(npz_path: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    if _sha256(npz_path) != manifest["bundle_npz_sha256"]:
        raise ValueError("FLEXICUBES_REFERENCE_BUNDLE_SHA_MISMATCH")
    archive = np.load(npz_path, allow_pickle=False)
    required = {"x_nx3", "s_n", "cube_fx8", "orientation_valid"}
    if not required <= set(archive.files):
        raise ValueError("FLEXICUBES_REFERENCE_BUNDLE_ARRAY_MISSING")
    arrays = {name: archive[name] for name in archive.files}
    x = np.asarray(arrays["x_nx3"])
    s = np.asarray(arrays["s_n"])
    cubes = np.asarray(arrays["cube_fx8"])
    orientation = np.asarray(arrays["orientation_valid"])
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("FLEXICUBES_REFERENCE_X_SHAPE_INVALID")
    if s.shape != (x.shape[0],):
        raise ValueError("FLEXICUBES_REFERENCE_S_SHAPE_INVALID")
    if orientation.shape != (x.shape[0],) or not np.all(orientation):
        raise ValueError("FLEXICUBES_REFERENCE_ORIENTATION_INVALID")
    if cubes.ndim != 2 or cubes.shape[1] != 8:
        raise ValueError("FLEXICUBES_REFERENCE_CUBE_SHAPE_INVALID")
    expected_cubes = int(np.prod(np.asarray(manifest["res"], dtype=np.int64)))
    if cubes.shape[0] != expected_cubes:
        raise ValueError("FLEXICUBES_REFERENCE_DENSE_CUBE_COUNT_REQUIRED")
    if cubes.size and (cubes.min() < 0 or cubes.max() >= x.shape[0]):
        raise ValueError("FLEXICUBES_REFERENCE_CUBE_INDEX_OOB")
    if not np.isfinite(x).all() or not np.isfinite(s).all():
        raise ValueError("FLEXICUBES_REFERENCE_INPUT_NONFINITE")
    for optional, shape_tail in (
        ("beta_fx12", 12),
        ("alpha_fx8", 8),
    ):
        if optional in arrays:
            value = np.asarray(arrays[optional])
            if value.shape != (cubes.shape[0], shape_tail) or not np.isfinite(value).all():
                raise ValueError(f"FLEXICUBES_REFERENCE_OPTIONAL_SHAPE_INVALID:{optional}")
    if "gamma_f" in arrays:
        value = np.asarray(arrays["gamma_f"])
        if value.shape != (cubes.shape[0],) or not np.isfinite(value).all():
            raise ValueError("FLEXICUBES_REFERENCE_OPTIONAL_SHAPE_INVALID:gamma_f")
    return arrays


def _tensor(arrays: dict[str, np.ndarray], name: str, *, device: str, dtype: torch.dtype):
    if name not in arrays:
        return None
    return torch.as_tensor(arrays[name], device=device, dtype=dtype)


def run_reference(
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
    module = importlib.import_module("flexicubes")
    FlexiCubes = getattr(module, "FlexiCubes")
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
            [int(v) for v in manifest["res"]],
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
    np.savez_compressed(
        output_npz,
        vertices=vertices_np,
        faces=faces_np,
        regularizer=regularizer_np,
    )

    result = {
        "schema": OUTPUT_SCHEMA,
        "subject_key": manifest["subject_key"],
        "arm": manifest["arm"],
        "input_bundle_npz_sha256": manifest["bundle_npz_sha256"],
        "teacher_bundle_sha256": manifest["teacher_bundle_sha256"],
        "extraction_truth_sha256": manifest["extraction_truth_sha256"],
        "measurement_protocol_sha256": manifest["measurement_protocol_sha256"],
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit_sha": UPSTREAM_COMMIT_SHA,
        "upstream_file_sha256": upstream_sha256,
        "device": device,
        "res": [int(v) for v in manifest["res"]],
        "candidate_vertex_count": int(vertices_np.shape[0]),
        "candidate_face_count": int(faces_np.shape[0]),
        "regularizer_count": int(regularizer_np.size),
        "candidate_npz_sha256": _sha256(output_npz),
        "provenance": "EXACT_BOUND_UPSTREAM_FLEXICUBES_REFERENCE_RUN",
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    result = run_reference(
        bundle_npz=args.bundle_npz,
        bundle_manifest=args.bundle_manifest,
        upstream_root=args.upstream_root,
        output_npz=args.output_npz,
        output_manifest=args.output_manifest,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
