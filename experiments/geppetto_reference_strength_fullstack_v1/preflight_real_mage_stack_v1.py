from __future__ import annotations

"""Fail-closed real Mage input/target preflight for the reference-strength Geppetto FIT.

No training happens here. The script reconstructs the current shipping-side
IRIS->GSA->RiggingSurfaceIR seam from the promoted signed zero-surface and exact
8 cameras, tensorizes that IR losslessly, and independently derives the
training-only anonymous mechanical-core teacher target.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.mage_mechanical_core_target_v1 import (
    project_mechanical_core_from_arrays_v1,
    project_mechanical_core_from_normalized_npz_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)


SCHEMA = "RealSaS.GeppettoReferenceStrengthMagePreflight.v1"
SOURCE_RUN_ID = "20260904T220929Z"
SOURCE_CHECKPOINT_SHA256 = "766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2"
ZERO_SURFACE_SHA256 = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
NORMALIZED_TEACHER_SHA256 = "528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f"
CAMERA_SHA256 = (
    "73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541",
    "bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4",
    "3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c",
    "24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d",
    "5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39",
    "7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe",
    "daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec",
    "68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f",
)

EXPECTED_NODE_COUNT = 950
EXPECTED_EDGE_COUNT = 2813
EXPECTED_OBSERVED_COUNT = 897
EXPECTED_COMPLETED_COUNT = 53
EXPECTED_SUPPORT_COUNTS = (343, 378, 309, 367, 343, 356, 296, 390)
EXPECTED_RASTER_STATS = {
    "mean_x_std": 0.15032009780406952,
    "mean_y_std": 0.46240127086639404,
    "std_x_mean": 0.16560441255569458,
    "std_y_mean": 3.267214733869353e-18,
}
EXPECTED_MECHANICAL_CORE_COUNT = 22
EXPECTED_MECHANICAL_ROOT_COUNT = 1


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_cameras(camera_dir: Path) -> tuple[dict, ...]:
    cameras = []
    for i, expected in enumerate(CAMERA_SHA256):
        path = camera_dir / f"V{i}.camera.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        got = _file_sha(path)
        if got != expected:
            raise RuntimeError(f"camera hash drift V{i}:{got}")
        payload = json.loads(path.read_text())
        if int(payload.get("view_index", -1)) != i:
            raise RuntimeError(f"camera view_index drift V{i}")
        cameras.append(payload)
    first = cameras[0]
    center = np.asarray(first["center"], dtype=np.float64)
    half = float(first["half_extent"])
    resolution = int(first["resolution"])
    if center.shape != (3,) or not np.isfinite(center).all() or half <= 0 or resolution <= 0:
        raise RuntimeError("camera normalization contract invalid")
    for i, cam in enumerate(cameras[1:], start=1):
        if not np.allclose(np.asarray(cam["center"], float), center, atol=0.0, rtol=0.0):
            raise RuntimeError(f"camera center drift V{i}")
        if float(cam["half_extent"]) != half or int(cam["resolution"]) != resolution:
            raise RuntimeError(f"camera extent/resolution drift V{i}")
    return tuple(cameras)


def _raster_stats(tensor) -> dict[str, float]:
    means = np.zeros((tensor.node_count, 2), dtype=np.float64)
    stds = np.zeros((tensor.node_count, 2), dtype=np.float64)
    for i in range(tensor.node_count):
        vv = np.flatnonzero(tensor.raster_valid[i])
        if len(vv):
            x = np.asarray(tensor.raster_xy_normalized[i, vv], dtype=np.float64)
            means[i] = x.mean(axis=0)
            stds[i] = x.std(axis=0)
    return {
        "mean_x_std": float(means[:, 0].std()),
        "mean_y_std": float(means[:, 1].std()),
        "std_x_mean": float(stds[:, 0].mean()),
        "std_y_mean": float(stds[:, 1].mean()),
    }


def _assert_close(name: str, got: float, expected: float, *, atol: float = 2e-6) -> None:
    if abs(float(got) - float(expected)) > atol:
        raise AssertionError(f"{name} drift got={got} expected={expected} atol={atol}")


def _teacher_permutation_invariance(npz_path: Path, reference) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        parents = np.asarray(z["parents"], dtype=np.int64)
        deform = np.asarray(z["deform_mask"], dtype=bool)
        skin = np.asarray(z["skin"], dtype=np.float64)
        heads = np.asarray(z["bone_heads"], dtype=np.float64)
        inverse = np.asarray(z["inverse_canonical_transform"], dtype=np.float64)

    rng = np.random.default_rng(20260907)
    perm = rng.permutation(len(parents))
    inverse_perm = np.empty(len(parents), dtype=np.int64)
    inverse_perm[perm] = np.arange(len(parents), dtype=np.int64)
    parents_p = np.asarray(
        [-1 if parents[old] < 0 else inverse_perm[int(parents[old])] for old in perm],
        dtype=np.int64,
    )
    other = project_mechanical_core_from_arrays_v1(
        parents=parents_p,
        deform_mask=deform[perm],
        skin=skin[:, perm],
        bone_heads=heads[perm],
        inverse_canonical_transform=inverse,
    )
    max_position_delta = float(
        np.max(np.abs(
            np.asarray(reference.positions_world, dtype=np.float64)
            - np.asarray(other.positions_world, dtype=np.float64)
        ))
    )
    topology_equal = bool(np.array_equal(reference.parent_indices, other.parent_indices))
    root_equal = bool(np.array_equal(reference.root_mask, other.root_mask))
    if max_position_delta != 0.0 or not topology_equal or not root_equal:
        raise AssertionError("mechanical-core projection is not source-row permutation invariant")
    return {
        "seed": 20260907,
        "max_position_delta": max_position_delta,
        "topology_equal": topology_equal,
        "root_equal": root_equal,
    }


def run(
    *,
    zero_surface_path: Path,
    teacher_normalized_path: Path,
    camera_dir: Path,
    output_path: Path,
) -> dict:
    if _file_sha(zero_surface_path) != ZERO_SURFACE_SHA256:
        raise RuntimeError("promoted zero-surface hash drift")
    if _file_sha(teacher_normalized_path) != NORMALIZED_TEACHER_SHA256:
        raise RuntimeError("training teacher normalized.npz hash drift")
    cameras = _load_cameras(camera_dir)

    with np.load(zero_surface_path, allow_pickle=False) as z:
        required = ("vertices", "faces", "normals")
        if any(k not in z.files for k in required):
            raise RuntimeError("zero-surface artifact missing required arrays")
        world = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        implicit_normals = np.asarray(z["normals"], dtype=np.float64)

    camera_center = np.asarray(cameras[0]["center"], dtype=np.float64)
    camera_half = float(cameras[0]["half_extent"])
    vertices_normalized = (world - camera_center[None, :]) / camera_half

    surface = rigging_surface_from_scene_first_zero_mesh_v1(
        vertices_normalized,
        faces,
        implicit_normals,
        cameras,
        normalization_center=camera_center,
        normalization_half_extent=camera_half,
        authority_label="IRIS_V3_PROMOTED_MAGE_FIT_WITNESS",
        source_run_id=SOURCE_RUN_ID,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_zero_surface_sha256=ZERO_SURFACE_SHA256,
        target_nodes=1024,
        normal_k=64,
        visibility_depth_tolerance_norm=0.02,
        metadata={"preflight_only": True},
    )
    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)

    support_counts = tuple(map(int, tensor.support.sum(axis=0).tolist()))
    raster_stats = _raster_stats(tensor)
    if tensor.node_count != EXPECTED_NODE_COUNT:
        raise AssertionError(f"surface node drift:{tensor.node_count}")
    if tensor.edge_count != EXPECTED_EDGE_COUNT:
        raise AssertionError(f"surface edge drift:{tensor.edge_count}")
    if int(tensor.observed.sum()) != EXPECTED_OBSERVED_COUNT:
        raise AssertionError("observed node count drift")
    if int(tensor.completed.sum()) != EXPECTED_COMPLETED_COUNT:
        raise AssertionError("completed node count drift")
    if support_counts != EXPECTED_SUPPORT_COUNTS:
        raise AssertionError(f"support-count drift:{support_counts}")
    for key, expected in EXPECTED_RASTER_STATS.items():
        _assert_close(key, raster_stats[key], expected)

    target = project_mechanical_core_from_normalized_npz_v1(teacher_normalized_path)
    if target.count != EXPECTED_MECHANICAL_CORE_COUNT:
        raise AssertionError(f"mechanical core count drift:{target.count}")
    if int(np.asarray(target.root_mask, bool).sum()) != EXPECTED_MECHANICAL_ROOT_COUNT:
        raise AssertionError("mechanical root count drift")
    permutation = _teacher_permutation_invariance(teacher_normalized_path, target)

    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "scientific_scope": "PREFLIGHT_ONLY__NO_TRAINING__NO_PROMOTION",
        "source": {
            "run_id": SOURCE_RUN_ID,
            "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "zero_surface_sha256": ZERO_SURFACE_SHA256,
            "teacher_normalized_sha256": NORMALIZED_TEACHER_SHA256,
            "camera_sha256": CAMERA_SHA256,
            "teacher_mesh_used_to_construct_surface": False,
            "teacher_skin_used_to_construct_surface": False,
        },
        "surface": {
            "geometry_lineage_hash": surface.geometry_lineage_hash,
            "tensorization_hash": tensor.tensorization_hash,
            "certificate_hash": tensor.certificate_hash,
            "node_count": tensor.node_count,
            "edge_count": tensor.edge_count,
            "observed_count": int(tensor.observed.sum()),
            "completed_count": int(tensor.completed.sum()),
            "support_counts_by_view": support_counts,
            "raster_stats": raster_stats,
            "normalization_center": tensor.normalization_center.tolist(),
            "normalization_scale": float(tensor.normalization_scale),
            "relation_kind_vocab": tensor.relation_kind_vocab,
            "validity_vocab": tensor.validity_vocab,
        },
        "teacher_target": {
            "role": "TRAINING_EVALUATION_ONLY",
            "rule_id": target.rule_id,
            "count": target.count,
            "root_count": int(np.asarray(target.root_mask, bool).sum()),
            "content_hash": target.content_hash,
            "source_indices_provenance_only": target.source_indices,
            "source_parent_indices_provenance_only": target.source_parent_indices,
            "source_names_used_for_selection": False,
            "handwritten_source_index_list_used_for_selection": False,
            "permutation_invariance": permutation,
        },
        "gate": {
            "expected_node_count": EXPECTED_NODE_COUNT,
            "expected_edge_count": EXPECTED_EDGE_COUNT,
            "expected_support_counts": EXPECTED_SUPPORT_COUNTS,
            "expected_mechanical_core_count": EXPECTED_MECHANICAL_CORE_COUNT,
            "passed": True,
        },
        "generalization_claim": False,
        "geppetto_promotion_authorized": False,
        "experiment_activation_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("GEPPETTO_REFERENCE_STRENGTH_PREFLIGHT=" + json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-surface", type=Path, required=True)
    ap.add_argument("--teacher-normalized", type=Path, required=True)
    ap.add_argument("--camera-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run(
        zero_surface_path=args.zero_surface,
        teacher_normalized_path=args.teacher_normalized,
        camera_dir=args.camera_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
