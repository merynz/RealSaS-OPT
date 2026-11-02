#!/usr/bin/env python3
from __future__ import annotations

"""Independent source court for authored glTF animation kinematics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF


PARENT_SCHEMA = "RealSaS.GltfAnimationKinematicsAuthority.v1"
PARENT_STATUS = "PASS_GLTF_ANIMATION_KINEMATICS_SOURCE_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.GltfAnimationKinematicsAuthorityCourt.v1"
STATUS = "PASS_GLTF_ANIMATION_KINEMATICS_SOURCE_COURT__CANONICAL_REPLAY_REQUIRED"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bound(row, code):
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("GLTF_ANIMATION_COURT_PARENT")
    source_path = bound(manifest.get("source"), "GLTF_ANIMATION_COURT_SOURCE_DRIFT")
    arrays_path = bound((manifest.get("outputs") or {}).get("arrays"), "GLTF_ANIMATION_COURT_ARRAY_DRIFT")
    source = SourceGLTF(source_path)
    doc = source.doc
    animations = list(doc.get("animations") or ())
    joint_nodes = {int(node) for skin in doc.get("skins") or () for node in skin.get("joints") or ()}
    if not animations or not joint_nodes:
        raise ValueError("GLTF_ANIMATION_COURT_SOURCE_SCOPE")

    with np.load(arrays_path, allow_pickle=False) as z:
        stored = {name: np.asarray(z[name]) for name in z.files}
    expected_keys = set()
    expected_rows = []
    total_joint = 0
    total_channels = 0
    total_keys = 0
    for ai, animation in enumerate(animations):
        samplers = list(animation.get("samplers") or ())
        channels = list(animation.get("channels") or ())
        channel_rows = []
        clip_min = np.inf
        clip_max = -np.inf
        joint_count = 0
        for ci, channel in enumerate(channels):
            sampler_index = int(channel["sampler"])
            sampler = samplers[sampler_index]
            interpolation = str(sampler.get("interpolation", "LINEAR"))
            time = np.asarray(source.accessor(int(sampler["input"])), dtype=np.float64).reshape(-1)
            value = np.asarray(source.accessor(int(sampler["output"])))
            if not len(time) or np.any(np.diff(time) <= 0) or not np.isfinite(time).all() or not np.isfinite(value).all():
                raise ValueError("GLTF_ANIMATION_COURT_SAMPLER")
            target = dict(channel.get("target") or {})
            node = int(target["node"])
            path = str(target["path"])
            expected_row_count = len(time) * (3 if interpolation == "CUBICSPLINE" else 1)
            if len(value) != expected_row_count:
                raise ValueError("GLTF_ANIMATION_COURT_OUTPUT_CARDINALITY")
            prefix = f"a{ai:04d}_c{ci:04d}"
            time_key, value_key = prefix + "_time", prefix + "_value"
            expected_keys.update((time_key, value_key))
            if time_key not in stored or value_key not in stored:
                raise ValueError("GLTF_ANIMATION_COURT_ARRAY_MISSING")
            if not np.array_equal(stored[time_key], time):
                raise ValueError("GLTF_ANIMATION_COURT_TIME_DRIFT")
            actual_value = stored[value_key]
            if not np.array_equal(actual_value, value):
                if actual_value.shape != value.shape or np.max(np.abs(actual_value.astype(np.float64) - value.astype(np.float64))) > 2.0e-12:
                    raise ValueError("GLTF_ANIMATION_COURT_VALUE_DRIFT")
            is_joint = node in joint_nodes
            joint_count += int(is_joint and path in {"translation", "rotation", "scale"})
            channel_rows.append({
                "channel_index": ci,
                "sampler_index": sampler_index,
                "target_node": node,
                "target_name": (doc.get("nodes") or [])[node].get("name"),
                "target_path": path,
                "target_is_skin_joint": bool(is_joint),
                "interpolation": interpolation,
                "key_count": int(len(time)),
                "time_key": time_key,
                "value_key": value_key,
                "value_shape": list(map(int, value.shape)),
                "time_min": float(time[0]),
                "time_max": float(time[-1]),
            })
            clip_min = min(clip_min, float(time[0]))
            clip_max = max(clip_max, float(time[-1]))
            total_keys += int(len(time))
        if joint_count == 0:
            raise ValueError("GLTF_ANIMATION_COURT_NO_JOINT_CHANNEL")
        expected_rows.append({
            "animation_index": ai,
            "name": animation.get("name"),
            "channel_count": len(channels),
            "joint_transform_channel_count": int(joint_count),
            "time_min": float(clip_min),
            "time_max": float(clip_max),
            "duration": float(clip_max - clip_min),
            "channels": channel_rows,
        })
        total_joint += joint_count
        total_channels += len(channels)
    if set(stored) != expected_keys:
        raise ValueError("GLTF_ANIMATION_COURT_ARRAY_SET")
    if list(manifest.get("animations") or ()) != expected_rows:
        raise ValueError("GLTF_ANIMATION_COURT_MANIFEST_DRIFT")
    counts = {
        "animation_count": len(expected_rows),
        "channel_count": int(total_channels),
        "joint_transform_channel_count": int(total_joint),
        "sampler_key_count_sum": int(total_keys),
    }
    if dict(manifest.get("counts") or {}) != counts:
        raise ValueError("GLTF_ANIMATION_COURT_COUNT_DRIFT")

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "source": {"path": str(source_path), "sha256": sha256(source_path)},
        "producer_arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path)},
        "counts": counts,
        "independent_method": {
            "reader": "tools.corpus.gltf_bind_scene_v1.SourceGLTF",
            "producer_reader_reused": False,
            "samplers_redecoded_from_source": True,
            "target_joint_membership_rebuilt_from_source_skins": True,
        },
        "family_candidate_outcome": "SOURCE_CORE_CANDIDATE__CANONICAL_REPLAY_REQUIRED",
        "authority": {
            "source_animation_court_passed": True,
            "canonical_replay_required": True,
            "family_authority_minted": False,
            "training_authorized": False,
        },
    }
    payload["court_identity_sha256"] = digest(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    payload = validate(manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "family_candidate_outcome": payload["family_candidate_outcome"], "court_identity_sha256": payload["court_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
