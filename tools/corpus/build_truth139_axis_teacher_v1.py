#!/usr/bin/env python3
"""Build anonymous continuous AXIS rig teacher truth from an admitted Stage47 teacher scene.

The producer uses source skin only to identify deformation-supported controls and necessary
ancestor bridges.  Bone/control names and source row indices remain privileged provenance.
The output stores continuous source-world and canonical joint positions plus the parent graph;
AXIS V5.4.1 XYZ256 quantization is deliberately left to a later model projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.Truth139AxisTeacher.v1"
SCENE_SCHEMA = "RealSaS.TeacherSceneIR.v1"
CONTRACT = "canonical/TRUTH139_AXIS_TEACHER_V1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _matrix(value, code: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (4, 4) or not np.isfinite(out).all():
        raise ValueError(code)
    if not np.allclose(out[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(code + "_AFFINE")
    return out


def _load(scene_path: Path, geometry_path: Path):
    scene = json.loads(scene_path.read_text())
    if scene.get("schema") != SCENE_SCHEMA:
        raise ValueError("TRUTH139_AXIS_SCENE_SCHEMA")
    refs = list(scene.get("geometry_payloads") or ())
    if len(refs) != 1 or str(refs[0].get("sha256") or "") != sha256(geometry_path):
        raise ValueError("TRUTH139_AXIS_GEOMETRY_BINDING")
    with np.load(geometry_path, allow_pickle=False) as z:
        required = {
            "rest_vertices_source_world",
            "source_weight_offsets",
            "source_weight_bone_indices_privileged",
            "source_weight_values",
            "source_weight_coverage",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_AXIS_WEIGHT_FIELDS_MISSING")
        arrays = {name: np.asarray(z[name]) for name in required}
    return scene, arrays


def _parents_and_positions(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("TRUTH139_AXIS_EMPTY_FRAMES")
    key_to_index: dict[tuple[str, str], int] = {}
    for i, row in enumerate(frames):
        key = (str(row.get("armature") or ""), str(row.get("name") or ""))
        if not key[0] or not key[1] or key in key_to_index:
            raise ValueError("TRUTH139_AXIS_FRAME_IDENTITY")
        key_to_index[key] = i
    parent = np.full(len(frames), -1, dtype=np.int64)
    position = np.zeros((len(frames), 3), dtype=np.float64)
    for i, row in enumerate(frames):
        armature = str(row["armature"])
        parent_name = row.get("parent")
        if parent_name not in (None, ""):
            p = key_to_index.get((armature, str(parent_name)))
            if p is None:
                raise ValueError("TRUTH139_AXIS_PARENT_FRAME_MISSING")
            parent[i] = int(p)
        transform = _matrix(row.get("rest_matrix_world"), "TRUTH139_AXIS_REST_MATRIX")
        position[i] = transform[:3, 3]
    for start in range(len(parent)):
        seen: set[int] = set()
        node = start
        while node >= 0:
            if node in seen:
                raise ValueError("TRUTH139_AXIS_PARENT_CYCLE")
            seen.add(node)
            node = int(parent[node])
    return parent, position


def _skin_mass(arrays: dict, bone_count: int) -> np.ndarray:
    offsets = np.asarray(arrays["source_weight_offsets"], dtype=np.int64)
    indices = np.asarray(arrays["source_weight_bone_indices_privileged"], dtype=np.int64)
    weights = np.asarray(arrays["source_weight_values"], dtype=np.float64)
    coverage = np.asarray(arrays["source_weight_coverage"], dtype=bool)
    vertices = np.asarray(arrays["rest_vertices_source_world"], dtype=np.float64)
    if (
        offsets.shape != (len(vertices) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(weights)
        or np.any(np.diff(offsets) < 0)
        or indices.shape != weights.shape
        or coverage.shape != (len(vertices),)
        or not np.array_equal(coverage, np.diff(offsets) > 0)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or (len(indices) and (indices.min() < 0 or indices.max() >= bone_count))
    ):
        raise ValueError("TRUTH139_AXIS_SOURCE_WEIGHT_CSR")
    mass = np.zeros(bone_count, dtype=np.float64)
    np.add.at(mass, indices, weights)
    return mass


def _select_supported_with_bridges(parent: np.ndarray, mass: np.ndarray, epsilon: float) -> np.ndarray:
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("TRUTH139_AXIS_SUPPORT_EPSILON")
    seed = set(np.flatnonzero(mass > epsilon).tolist())
    if not seed:
        raise ValueError("TRUTH139_AXIS_NO_SKIN_SUPPORTED_CONTROL")
    selected = set(seed)
    for child in sorted(seed):
        chain: list[int] = []
        p = int(parent[child])
        while p >= 0:
            if p in seed:
                selected.update(chain)
                break
            chain.append(p)
            p = int(parent[p])
    return np.asarray(sorted(selected), dtype=np.int64)


def _project_parent(parent: np.ndarray, selected: np.ndarray) -> np.ndarray:
    local = {int(source): i for i, source in enumerate(selected.tolist())}
    out = np.full(len(selected), -1, dtype=np.int64)
    for i, source in enumerate(selected.tolist()):
        p = int(parent[source])
        while p >= 0 and p not in local:
            p = int(parent[p])
        out[i] = -1 if p < 0 else int(local[p])
    return out


def _signature(node: int, positions: np.ndarray, parents: np.ndarray, children: list[list[int]], state, cache):
    if state[node] == 1:
        raise ValueError("TRUTH139_AXIS_PROJECTED_CYCLE")
    if state[node] == 2:
        return cache[node]
    state[node] = 1
    pos = tuple(float(0.0 if x == 0.0 else x) for x in positions[node])
    child_sig = tuple(sorted(_signature(c, positions, parents, children, state, cache) for c in children[node]))
    cache[node] = (pos, child_sig)
    state[node] = 2
    return cache[node]


def _content_order(positions: np.ndarray, parents: np.ndarray) -> np.ndarray:
    n = len(positions)
    children = [[] for _ in range(n)]
    roots = []
    for i, p in enumerate(parents.tolist()):
        if p < 0:
            roots.append(i)
        else:
            children[int(p)].append(i)
    state = np.zeros(n, dtype=np.int8)
    cache = [None] * n
    for root in roots:
        _signature(root, positions, parents, children, state, cache)
    if np.any(state != 2):
        raise ValueError("TRUTH139_AXIS_UNREACHABLE_PROJECTED_NODE")
    order: list[int] = []
    def emit(node: int):
        order.append(node)
        groups: dict[tuple, list[int]] = {}
        for child in children[node]:
            groups.setdefault(cache[child], []).append(child)
        for sig in sorted(groups):
            for child in groups[sig]:
                emit(child)
    root_groups: dict[tuple, list[int]] = {}
    for root in roots:
        root_groups.setdefault(cache[root], []).append(root)
    for sig in sorted(root_groups):
        for root in root_groups[sig]:
            emit(root)
    order_array = np.asarray(order, dtype=np.int64)
    if set(order_array.tolist()) != set(range(n)):
        raise ValueError("TRUTH139_AXIS_CONTENT_ORDER_NOT_PERMUTATION")
    return order_array


def build(scene_path: Path, geometry_path: Path, output_dir: Path, *, support_epsilon: float = 1e-8) -> dict:
    scene_path = scene_path.resolve(); geometry_path = geometry_path.resolve(); output_dir = output_dir.resolve()
    scene, arrays = _load(scene_path, geometry_path)
    frames = list((scene.get("calibration") or {}).get("native_control_frames_privileged") or ())
    parent_source, position_source = _parents_and_positions(frames)
    mass_source = _skin_mass(arrays, len(frames))
    selected = _select_supported_with_bridges(parent_source, mass_source, float(support_epsilon))
    positions = position_source[selected]
    parents = _project_parent(parent_source, selected)
    order = _content_order(positions, parents)
    inverse = np.empty(len(order), dtype=np.int64); inverse[order] = np.arange(len(order), dtype=np.int64)
    serialized_parent = np.asarray([
        -1 if parents[old] < 0 else inverse[int(parents[old])]
        for old in order.tolist()
    ], dtype=np.int64)
    if any(p >= i for i, p in enumerate(serialized_parent.tolist()) if p >= 0):
        raise ValueError("TRUTH139_AXIS_PARENT_BEFORE_CHILD")

    source_world_to_canonical = _matrix(
        (scene.get("coordinate_contract") or {}).get("source_world_to_normalized"),
        "TRUTH139_AXIS_SOURCE_TO_CANONICAL",
    )
    world_h = np.concatenate((positions[order], np.ones((len(order), 1), dtype=np.float64)), axis=1)
    canonical_h = (source_world_to_canonical @ world_h.T).T
    if not np.allclose(canonical_h[:, 3], 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("TRUTH139_AXIS_CANONICAL_HOMOGENEOUS")
    canonical = canonical_h[:, :3]

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "AXIS_TEACHER.npz"
    np.savez_compressed(
        arrays_path,
        positions_source_world=positions[order].astype(np.float64),
        positions_canonical=canonical.astype(np.float64),
        parent_indices=serialized_parent,
        root_mask=(serialized_parent < 0),
        source_indices_provenance_only=selected[order],
        skin_mass=mass_source[selected][order].astype(np.float64),
        source_world_to_canonical=source_world_to_canonical,
    )
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_AXIS_CONTINUOUS_RIG_TEACHER_CANDIDATE__MODEL_SPECIFIC_TRAINING_PROJECTION_NOT_MINTED",
        "subject_id": str(scene.get("subject_key") or ""),
        "source_sha256": str(scene.get("source_sha256") or ""),
        "contract": CONTRACT,
        "teacher_scene": {"path": str(scene_path), "sha256": sha256(scene_path)},
        "geometry_payload": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
        "arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path)},
        "joint_count": int(len(order)),
        "root_count": int(np.count_nonzero(serialized_parent < 0)),
        "source_control_count": int(len(frames)),
        "skin_supported_source_control_count": int(np.count_nonzero(mass_source > support_epsilon)),
        "selection": "SKIN_SUPPORTED_PLUS_REQUIRED_SUPPORTED_ANCESTOR_BRIDGES",
        "serialization": "PARENT_BEFORE_CHILD_CONTENT_BASED__SOURCE_NAMES_NOT_TARGETS",
        "continuous_truth_preserved_before_xyz256_quantization": True,
        "source_names_exported_as_student_targets": False,
        "training_authorized": False,
        "qualification_minted": False,
    }
    payload["axis_teacher_identity_sha256"] = digest(payload)
    manifest = output_dir / "AXIS_TEACHER_MANIFEST.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "subject_id": payload["subject_id"], "joint_count": payload["joint_count"], "root_count": payload["root_count"], "axis_teacher_identity_sha256": payload["axis_teacher_identity_sha256"]}, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-scene", required=True, type=Path)
    parser.add_argument("--geometry-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--support-epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    build(args.teacher_scene, args.geometry_npz, args.output_dir, support_epsilon=args.support_epsilon)


if __name__ == "__main__":
    main()
