#!/usr/bin/env python3
from __future__ import annotations

"""Build source-native authored animation kinematics from immutable glTF 2.0 bytes.

Authored clips are kinematic truth, not a physical-law claim. Exact source sampler
times/values/interpolation and target joint identities are preserved. Canonical or
Blender replay equivalence remains a separate court before final family authority.
Both JSON .gltf and binary .glb containers are accepted under the same contract.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.build_gltf_rig_skin_authority_v1 import _accessor, _buffer_bytes
from tools.observation.source_scene_contract_v2 import _json_and_chunks


SCHEMA = "RealSaS.GltfAnimationKinematicsAuthority.v1"
STATUS = "PASS_GLTF_ANIMATION_KINEMATICS_SOURCE_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype == object or (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()):
            raise ValueError("GLTF_ANIMATION_INVALID_ARRAY:" + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(source_path: Path, output_dir: Path) -> dict:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    gltf, chunks = _json_and_chunks(source_path)
    if not str((gltf.get("asset") or {}).get("version") or "").startswith("2"):
        raise ValueError("GLTF_ANIMATION_GLTF2_REQUIRED")
    buffers = _buffer_bytes(gltf, source_path, chunks)
    nodes = list(gltf.get("nodes") or ())
    animations = list(gltf.get("animations") or ())
    if not animations:
        raise ValueError("GLTF_ANIMATION_NONE")
    skin_joint_nodes = {
        int(node_index)
        for skin in gltf.get("skins") or ()
        for node_index in skin.get("joints") or ()
    }
    if not skin_joint_nodes:
        raise ValueError("GLTF_ANIMATION_NO_SKIN_JOINTS")

    arrays: dict[str, np.ndarray] = {}
    animation_rows = []
    total_joint_channels = 0
    total_channels = 0
    total_keys = 0
    for animation_index, animation in enumerate(animations):
        samplers = list(animation.get("samplers") or ())
        channels = list(animation.get("channels") or ())
        if not samplers or not channels:
            raise ValueError("GLTF_ANIMATION_EMPTY_CLIP")
        channel_rows = []
        clip_min = np.inf
        clip_max = -np.inf
        joint_channel_count = 0
        for channel_index, channel in enumerate(channels):
            sampler_index = int(channel.get("sampler", -1))
            if sampler_index < 0 or sampler_index >= len(samplers):
                raise ValueError("GLTF_ANIMATION_SAMPLER_INDEX")
            sampler = samplers[sampler_index]
            interpolation = str(sampler.get("interpolation", "LINEAR"))
            if interpolation not in {"LINEAR", "STEP", "CUBICSPLINE"}:
                raise ValueError("GLTF_ANIMATION_INTERPOLATION")
            times = np.asarray(_accessor(gltf, buffers, int(sampler["input"])), dtype=np.float64).reshape(-1)
            values = np.asarray(_accessor(gltf, buffers, int(sampler["output"])))
            if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("GLTF_ANIMATION_TIME_SEQUENCE")
            if not np.isfinite(values).all():
                raise ValueError("GLTF_ANIMATION_VALUE_FINITE")
            target = dict(channel.get("target") or {})
            target_node = target.get("node")
            path = str(target.get("path") or "")
            if target_node is None or int(target_node) < 0 or int(target_node) >= len(nodes):
                raise ValueError("GLTF_ANIMATION_TARGET_NODE")
            target_node = int(target_node)
            if path not in {"translation", "rotation", "scale", "weights"}:
                raise ValueError("GLTF_ANIMATION_TARGET_PATH")
            expected_rows = len(times) * (3 if interpolation == "CUBICSPLINE" else 1)
            if len(values) != expected_rows:
                raise ValueError("GLTF_ANIMATION_OUTPUT_CARDINALITY")
            expected_width = {"translation": 3, "rotation": 4, "scale": 3}.get(path)
            if expected_width is not None and (values.ndim != 2 or values.shape[1] != expected_width):
                raise ValueError("GLTF_ANIMATION_OUTPUT_WIDTH")
            key_prefix = f"a{animation_index:04d}_c{channel_index:04d}"
            time_key = key_prefix + "_time"
            value_key = key_prefix + "_value"
            arrays[time_key] = times.astype(np.float64)
            arrays[value_key] = np.asarray(values)
            target_is_joint = target_node in skin_joint_nodes
            joint_channel_count += int(target_is_joint and path in {"translation", "rotation", "scale"})
            channel_rows.append(
                {
                    "channel_index": channel_index,
                    "sampler_index": sampler_index,
                    "target_node": target_node,
                    "target_name": nodes[target_node].get("name"),
                    "target_path": path,
                    "target_is_skin_joint": bool(target_is_joint),
                    "interpolation": interpolation,
                    "key_count": int(len(times)),
                    "time_key": time_key,
                    "value_key": value_key,
                    "value_shape": list(map(int, values.shape)),
                    "time_min": float(times[0]),
                    "time_max": float(times[-1]),
                }
            )
            clip_min = min(clip_min, float(times[0]))
            clip_max = max(clip_max, float(times[-1]))
            total_keys += int(len(times))
        if joint_channel_count == 0:
            raise ValueError("GLTF_ANIMATION_NO_JOINT_TRANSFORM_CHANNELS")
        animation_rows.append(
            {
                "animation_index": animation_index,
                "name": animation.get("name"),
                "channel_count": len(channels),
                "joint_transform_channel_count": int(joint_channel_count),
                "time_min": float(clip_min),
                "time_max": float(clip_max),
                "duration": float(clip_max - clip_min),
                "channels": channel_rows,
            }
        )
        total_joint_channels += joint_channel_count
        total_channels += len(channels)

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "GLTF_ANIMATION_KINEMATICS_V1.npz"
    _write_npz(arrays_path, arrays)
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "source": {"path": str(source_path), "sha256": sha256(source_path)},
        "source_format": "GLTF_2_0",
        "source_container": "GLB" if source_path.suffix.lower() == ".glb" else "GLTF_JSON",
        "animations": animation_rows,
        "counts": {
            "animation_count": len(animation_rows),
            "channel_count": int(total_channels),
            "joint_transform_channel_count": int(total_joint_channels),
            "sampler_key_count_sum": int(total_keys),
        },
        "outputs": {"arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path)}},
        "policy": {
            "source_values_preserved": True,
            "authored_clip_is_kinematic_truth_not_physical_law": True,
            "joint_transform_paths": ["translation", "rotation", "scale"],
            "non_joint_or_morph_channels_remain_evidence_but_do_not_define_rig_kinematics": True,
            "canonical_replay_court_required": True,
            "source_mutated": False,
        },
        "authority": {
            "raw_source_candidate_only": True,
            "independent_source_court_required": True,
            "canonical_replay_court_required": True,
            "family_authority_minted": False,
            "training_authorized": False,
        },
    }
    payload["animation_identity_sha256"] = digest(payload)
    manifest_path = output_dir / "GLTF_ANIMATION_KINEMATICS_MANIFEST_V1.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": STATUS, "animation_count": len(animation_rows), "joint_transform_channel_count": total_joint_channels, "animation_identity_sha256": payload["animation_identity_sha256"]}, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output_dir)


if __name__ == "__main__":
    main()
