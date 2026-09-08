from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.substrate import scene_first_signed
from compiler.realsas_compiler_core.substrate.scene_first_signed import rigging_surface_from_scene_first_zero_mesh_v1
from models.arachne.v2 import conditioning_v1, conditioning_v2, arachne_geometry_v2, geppetto_conditioning_v2
from models.arachne.v2.conditioning_v1 import ArachneConditioningAdapter
from models.arachne.v2.conditioning_v2 import ArachneConditioningAdapterV2
from models.geppetto.v2 import geppetto_conditioning_v2 as geppetto_conditioning_v2_canonical

SCHEMA = "RealSaS.ArachneMageFS1ConditioningCache.v2"
CACHE_BINDING_SCHEMA = "RealSaS.ArachneMageFS1ConditioningCacheBinding.v1"

EXPECTED_FS1_TARGET_SHA256 = "7154f5ad98b446c0019c472fd91863b5843b98846cdb38b38e4f78ba7fb5b93f"
EXPECTED_FS1_BINDING_SHA256 = "ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf"
EXPECTED_FS1_WEIGHTS_SHA256 = "7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d"
EXPECTED_ZERO_SURFACE_SHA256 = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
EXPECTED_QUALIFIED_SKELETON_FILE_SHA256 = "48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992"
EXPECTED_SURFACE_LINEAGE = "67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb"
EXPECTED_SKELETON_LINEAGE = "738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306"
EXPECTED_ND_OPERATOR_SHA256 = "5b5c0e89a18c98f6f7a231941480a61e7158d93d91a9a8edcc5362d6c5e0b1be"
EXPECTED_V1_CONDITIONING_HASH = "475d71523a38c5763756713e2047b337662ad6c01735aaa89671c6297326e42c"
EXPECTED_V2_CONDITIONING_HASH = "9e979b824e6e30fcaf1158e4bceeb080ce5fa1cff463290e1c4a7c2ac4f5f4e9"
EXPECTED_PAIR_GEOMETRY_RAW_SHA256 = "416d5e5848ec579eb67e5e402dbf6e60908f4d9a30dbe0650a1d19be7183e4e7"
EXPECTED_SUPERVISED_ROWS = 934
EXPECTED_LOW_ROWS = 16
LOW_CONFIDENCE_CODE = 3
SOURCE_RUN_ID = "20260904T220929Z"
IRIS_CHECKPOINT_SHA256 = "766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2"
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

SOURCE_GIT_BLOBS = {
    "scene_first_signed": "a55e431faf0680e0d7f2920a85fda6341d45117d",
    "arachne_conditioning_v1": "b3539ed6c4580ac69338bdb9bd4b93c6e78a262c",
    "arachne_conditioning_v2": "3383cb7801b22aeb634f05551a847a04e674c7c8",
    "arachne_geometry_v2": "0a39450b69de48da3b7748cb08972d0b69ff9700",
    "arachne_geppetto_v2_bridge": "35a63245799710ff7ede082d0f8ffcf19f6af505",
    "canonical_geppetto_conditioning_v2": "9a33ab34ce050842c77dfc64407123303c4bf456",
}

HISTORICAL_OPAQUE_HASHES_RETIRED = {
    "conditioning_v1_hash": "7b82d57b1967858c6a66de871ecd570a9bdae408b30ff84c3fcbd8e908551047",
    "conditioning_v2_hash": "89973cd1f0ac989d10cdfeafdef89545668fda4998363df3e872961646879939",
    "local_cache_sha256": "12484afc23d5c03cbad8020266ed5b39c3201d979e78f96380e748902152be6e",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, separators=(",", ": "), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_json_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _npy_bytes(a: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.lib.format.write_array(out, np.asarray(a), allow_pickle=False)
    return out.getvalue()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            zf.writestr(info, _npy_bytes(np.asarray(arrays[key])), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return out.getvalue()


def module_blob_guard() -> dict[str, str]:
    paths = {
        "scene_first_signed": Path(scene_first_signed.__file__).resolve(),
        "arachne_conditioning_v1": Path(conditioning_v1.__file__).resolve(),
        "arachne_conditioning_v2": Path(conditioning_v2.__file__).resolve(),
        "arachne_geometry_v2": Path(arachne_geometry_v2.__file__).resolve(),
        "arachne_geppetto_v2_bridge": Path(geppetto_conditioning_v2.__file__).resolve(),
        "canonical_geppetto_conditioning_v2": Path(geppetto_conditioning_v2_canonical.__file__).resolve(),
    }
    got = {k: git_blob_sha(v) for k, v in paths.items()}
    if got != SOURCE_GIT_BLOBS:
        raise RuntimeError(f"CONDITIONING_SOURCE_GIT_BLOB_DRIFT:{got}:{SOURCE_GIT_BLOBS}")
    return got


def load_surface(zero_surface: Path, camera_dir: Path):
    if sha256_file(zero_surface) != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("ZERO_SURFACE_SHA_DRIFT")
    cameras = []
    for i, expected in enumerate(CAMERA_SHA256):
        path = camera_dir / f"V{i}.camera.json"
        if sha256_file(path) != expected:
            raise RuntimeError(f"CAMERA_SHA_DRIFT_V{i}")
        cameras.append(json.loads(path.read_text(encoding="utf-8")))
    with np.load(zero_surface, allow_pickle=False) as z:
        world = np.asarray(z["vertices"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
        hints = np.asarray(z["normals"], np.float64)
    center = np.asarray(cameras[0]["center"], np.float64)
    half = float(cameras[0]["half_extent"])
    surface = rigging_surface_from_scene_first_zero_mesh_v1(
        (world - center[None, :]) / half,
        faces,
        hints,
        tuple(cameras),
        normalization_center=center,
        normalization_half_extent=half,
        authority_label="IRIS_SCENE_FIRST_SIGNED_V3_PROMOTED_MAGE_FIT",
        source_run_id=SOURCE_RUN_ID,
        source_checkpoint_sha256=IRIS_CHECKPOINT_SHA256,
        source_zero_surface_sha256=EXPECTED_ZERO_SURFACE_SHA256,
        target_nodes=1024,
        normal_k=64,
        visibility_depth_tolerance_norm=0.02,
        metadata={
            "camera_contract": "CANONICAL_8_ORTHOGRAPHIC_YAW_45_DEG",
            "teacher_truth_used": False,
            "geppetto_reference_strength_fit1": True,
        },
    )
    if surface.geometry_lineage_hash != EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("SURFACE_LINEAGE_DRIFT")
    if len(surface.surface_nodes) != 950 or len(surface.local_relations) != 2813:
        raise RuntimeError("SURFACE_SHAPE_DRIFT")
    if str(surface.metadata.get("Nd_operator_sha256", "")) != EXPECTED_ND_OPERATOR_SHA256:
        raise RuntimeError("ND_OPERATOR_SHA_DRIFT")
    return surface


def load_skeleton(path: Path):
    if sha256_file(path) != EXPECTED_QUALIFIED_SKELETON_FILE_SHA256:
        raise RuntimeError("QUALIFIED_SKELETON_FILE_SHA_DRIFT")
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["skeleton_lineage_hash"] != EXPECTED_SKELETON_LINEAGE or len(d["joints"]) != 22:
        raise RuntimeError("SKELETON_LINEAGE_OR_COUNT_DRIFT")
    joints = tuple(
        SimpleNamespace(
            canonical_joint_id=x["canonical_joint_id"],
            position=tuple(map(float, x["position"])),
            parent_canonical_id=x["parent_canonical_id"],
            support_surface_ids=tuple(x.get("support_surface_ids", ())),
        )
        for x in d["joints"]
    )
    return SimpleNamespace(
        joints=joints,
        deform_root_ids=tuple(d["deform_root_ids"]),
        skeleton_lineage_hash=d["skeleton_lineage_hash"],
    )


def array_digest(a: np.ndarray) -> str:
    a = np.asarray(a)
    if a.dtype.kind in {"U", "S", "O"}:
        return sha256_bytes(canonical_json_bytes(a.tolist()))
    return sha256_bytes(np.ascontiguousarray(a).tobytes(order="C"))


def array_inventory(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        k: {
            "dtype": str(np.asarray(v).dtype),
            "shape": list(np.asarray(v).shape),
            "content_sha256": array_digest(np.asarray(v)),
        }
        for k, v in sorted(arrays.items())
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs1-target", type=Path, required=True)
    ap.add_argument("--zero-surface", type=Path, required=True)
    ap.add_argument("--camera-dir", type=Path, required=True)
    ap.add_argument("--qualified-skeleton", type=Path, required=True)
    ap.add_argument("--output-npz", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    args = ap.parse_args()

    source_blobs = module_blob_guard()
    if sha256_file(args.fs1_target) != EXPECTED_FS1_TARGET_SHA256:
        raise RuntimeError("FS1_TARGET_SHA_DRIFT")

    surface = load_surface(args.zero_surface, args.camera_dir)
    skeleton = load_skeleton(args.qualified_skeleton)

    base = ArachneConditioningAdapter()((surface,), (skeleton,))
    v2 = ArachneConditioningAdapterV2()((surface,), (skeleton,))
    if base.conditioning_hashes[0] != EXPECTED_V1_CONDITIONING_HASH:
        raise RuntimeError("V1_CONDITIONING_HASH_DRIFT")
    if v2.conditioning_hashes[0] != EXPECTED_V2_CONDITIONING_HASH:
        raise RuntimeError("V2_CONDITIONING_HASH_DRIFT")
    pair_sha = sha256_bytes(np.asarray(v2.pair_geometry[0], dtype="<f4").tobytes(order="C"))
    if pair_sha != EXPECTED_PAIR_GEOMETRY_RAW_SHA256:
        raise RuntimeError("PAIR_GEOMETRY_CONTENT_DRIFT")

    surface_ids = np.asarray(base.surface_ids[0], dtype="<U24")
    joint_ids = np.asarray(base.joint_ids[0], dtype="<U22")
    surface_features = np.asarray(base.surface_features[0], dtype="<f4")
    joint_features = np.asarray(base.joint_features[0], dtype="<f4")
    surface_positions_normalized = np.asarray(base.surface_positions_normalized[0], dtype="<f4")
    parent_indices = np.asarray(base.parent_indices[0], dtype="<i8")
    pair_geometry = np.asarray(v2.pair_geometry[0], dtype="<f4")
    pair_mask = np.asarray(v2.pair_mask[0], dtype=np.uint8)

    nodes = tuple(sorted(surface.surface_nodes, key=lambda n: n.surface_id))
    rest_points_world = np.asarray([n.P for n in nodes], dtype="<f4")

    with np.load(args.fs1_target, allow_pickle=False) as z:
        fs1_surface_ids = np.asarray(z["surface_ids"], dtype="<U24")
        fs1_joint_ids = np.asarray(z["joint_ids"], dtype="<U22")
        teacher_weights = np.asarray(z["weights"], dtype="<f4")
        confidence_code = np.asarray(z["confidence_code"], dtype=np.uint8)

    if not np.array_equal(surface_ids, fs1_surface_ids):
        raise RuntimeError("FS1_SURFACE_ID_BINDING_DRIFT")
    if not np.array_equal(joint_ids, fs1_joint_ids):
        raise RuntimeError("FS1_JOINT_ID_BINDING_DRIFT")
    if teacher_weights.shape != (950, 22):
        raise RuntimeError("FS1_WEIGHT_SHAPE_DRIFT")
    if sha256_bytes(teacher_weights.tobytes(order="C")) != EXPECTED_FS1_WEIGHTS_SHA256:
        raise RuntimeError("FS1_WEIGHT_CONTENT_SHA_DRIFT")

    teacher_supervision_mask = np.asarray(confidence_code != LOW_CONFIDENCE_CODE, dtype=np.uint8)
    low_rows = int((teacher_supervision_mask == 0).sum())
    supervised_rows = int(teacher_supervision_mask.sum())
    if low_rows != EXPECTED_LOW_ROWS or supervised_rows != EXPECTED_SUPERVISED_ROWS:
        raise RuntimeError("FS1_SUPERVISION_MASK_COUNT_DRIFT")

    arrays = {
        "confidence_code": confidence_code,
        "joint_features": joint_features,
        "joint_ids": joint_ids,
        "pair_geometry": pair_geometry,
        "pair_mask": pair_mask,
        "parent_indices": parent_indices,
        "rest_points_world": rest_points_world,
        "surface_features": surface_features,
        "surface_ids": surface_ids,
        "surface_positions_normalized": surface_positions_normalized,
        "teacher_supervision_mask": teacher_supervision_mask,
        "teacher_weights": teacher_weights,
    }

    inventory = array_inventory(arrays)
    cache_binding_payload = {
        "schema": CACHE_BINDING_SCHEMA,
        "surface_geometry_lineage_hash": EXPECTED_SURFACE_LINEAGE,
        "skeleton_lineage_hash": EXPECTED_SKELETON_LINEAGE,
        "Nd_operator_sha256": EXPECTED_ND_OPERATOR_SHA256,
        "fs1_target_npz_sha256": EXPECTED_FS1_TARGET_SHA256,
        "fs1_target_binding_sha256": EXPECTED_FS1_BINDING_SHA256,
        "teacher_weights_content_sha256": EXPECTED_FS1_WEIGHTS_SHA256,
        "conditioning_v1_hash_semantics": "ArachneConditioningAdapter.conditioning_hashes[0]",
        "conditioning_v1_hash": base.conditioning_hashes[0],
        "conditioning_v2_hash_semantics": "ArachneConditioningAdapterV2.conditioning_hashes[0]",
        "conditioning_v2_hash": v2.conditioning_hashes[0],
        "source_git_blobs": source_blobs,
        "array_inventory": inventory,
    }
    cache_binding_sha256 = sha256_bytes(canonical_json_bytes(cache_binding_payload))

    npz_data = deterministic_npz_bytes(arrays)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.write_bytes(npz_data)
    cache_npz_sha256 = sha256_bytes(npz_data)

    manifest = {
        "schema": SCHEMA,
        "status": "FS1_CONDITIONING_CACHE_BUILT__A0_STILL_BLOCKED_UNTIL_REBIND_AND_RUNNER_REGRESSIONS",
        "scientific_scope": "DETERMINISTIC_S_G_CONDITIONING_PLUS_SEALED_FS1_TEACHER_TARGET__NO_OPTIMIZER_STEP",
        "inputs": {
            "fs1_target_npz_sha256": EXPECTED_FS1_TARGET_SHA256,
            "fs1_target_binding_sha256": EXPECTED_FS1_BINDING_SHA256,
            "teacher_weights_content_sha256": EXPECTED_FS1_WEIGHTS_SHA256,
            "zero_surface_sha256": EXPECTED_ZERO_SURFACE_SHA256,
            "camera_sha256": list(CAMERA_SHA256),
            "qualified_skeleton_file_sha256": EXPECTED_QUALIFIED_SKELETON_FILE_SHA256,
            "surface_geometry_lineage_hash": EXPECTED_SURFACE_LINEAGE,
            "skeleton_lineage_hash": EXPECTED_SKELETON_LINEAGE,
            "Nd_operator_sha256": EXPECTED_ND_OPERATOR_SHA256,
        },
        "source_contract": {
            "git_blobs": source_blobs,
            "conditioning_v1_hash_semantics": "ArachneConditioningAdapter.conditioning_hashes[0]",
            "conditioning_v1_hash": base.conditioning_hashes[0],
            "conditioning_v2_hash_semantics": "ArachneConditioningAdapterV2.conditioning_hashes[0]",
            "conditioning_v2_hash": v2.conditioning_hashes[0],
            "pair_geometry_contract": list(arachne_geometry_v2.PAIR_GEOMETRY_CONTRACT_V2),
            "pair_geometry_raw_sha256": pair_sha,
        },
        "retired_historical_opaque_fields": {
            **HISTORICAL_OPAQUE_HASHES_RETIRED,
            "reason": "Historical report did not retain the byte-generating cache builder or hash payload semantics; values remain history and are not FS1 authority.",
        },
        "cache": {
            "npz_sha256": cache_npz_sha256,
            "binding_sha256": cache_binding_sha256,
            "array_inventory": inventory,
            "surface_rows": 950,
            "joint_count": 22,
            "pair_count": 20900,
            "supervised_rows": supervised_rows,
            "low_rows": low_rows,
        },
        "authorization": {
            "a0_optimizer_authorized": False,
            "a1_optimizer_authorized": False,
            "main_scientific_a0_optimizer_steps": 0,
            "next_gate": "DOUBLE_REPLAY_BYTE_DETERMINISM__SEAL_CACHE__REBIND_A0_PREREG_AND_RUNNER",
        },
    }
    args.output_manifest.write_bytes(json_bytes(manifest))
    print(json.dumps({
        "status": manifest["status"],
        "cache_npz_sha256": cache_npz_sha256,
        "cache_binding_sha256": cache_binding_sha256,
        "conditioning_v1_hash": base.conditioning_hashes[0],
        "conditioning_v2_hash": v2.conditioning_hashes[0],
        "supervised_rows": supervised_rows,
        "low_rows": low_rows,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
