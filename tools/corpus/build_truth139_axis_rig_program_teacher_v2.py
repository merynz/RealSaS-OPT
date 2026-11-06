#!/usr/bin/env python3
"""Build AXIS V6 RigProgram teacher truth from Truth139 measured mechanics.

Measured truth:
- source rest joint graph / frames,
- source skin support,
- isolated native local-rotation deformation responses,
- source-world -> canonical transform.

Derived teacher truth:
- anonymous causal deform serialization,
- canonical rest frames and behaviorally active rotational DOFs,
- GLOBAL/ROOT/FK controls,
- maximal non-branching chains,
- rest-neutral generic IK effector/pole availability on eligible terminal chains.

The derived control program is NOT claimed to be source-authored. No source bone
name, pack label, humanoid taxonomy, or Compiler repair enters the student target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    validate_rig_constraint_operator_v1,
)
from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    RigProgramChainProposalIR,
    RigProgramConstraintProposalIR,
    RigProgramControlProposalIR,
    RigProgramJointProposalIR,
    RigProgramLocalFrameProposalIR,
    RigProgramProposalIR,
    qualify_rig_program_v1,
    seal_rig_program_proposal_v1,
)

SCENE_SCHEMA = "RealSaS.TeacherSceneIR.v1"
TEACHER_SCHEMA = "RealSaS.Truth139AxisRigProgramTeacher.v2"
CONTRACT = "canonical/TRUTH139_AXIS_RIG_PROGRAM_TEACHER_V2.json"
DERIVED_AUTHORITY = "DERIVED_CANONICAL_CONTROL_PROGRAM__NOT_SOURCE_AUTHORED_CONTROL_RIG"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _matrix(value: Any, code: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (4, 4) or not np.isfinite(out).all():
        raise ValueError(code)
    if not np.allclose(out[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(code + "_AFFINE")
    return out


def _load(scene_path: Path, geometry_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    scene = json.loads(scene_path.read_text())
    if scene.get("schema") != SCENE_SCHEMA:
        raise ValueError("TRUTH139_AXIS_V2_SCENE_SCHEMA")
    refs = list(scene.get("geometry_payloads") or ())
    if len(refs) != 1 or str(refs[0].get("sha256") or "") != _sha256(geometry_path):
        raise ValueError("TRUTH139_AXIS_V2_GEOMETRY_BINDING")
    required = {
        "rest_vertices_source_world",
        "source_weight_offsets",
        "source_weight_bone_indices_privileged",
        "source_weight_values",
        "source_weight_coverage",
        "pose_identity_baseline",
        "isolated_position",
    }
    with np.load(geometry_path, allow_pickle=False) as z:
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_AXIS_V2_REQUIRED_ARRAYS_MISSING")
        arrays = {name: np.asarray(z[name]) for name in required}
    return scene, arrays


def _source_graph(scene: dict) -> tuple[list[dict], np.ndarray, np.ndarray, list[np.ndarray]]:
    rows = list((scene.get("calibration") or {}).get("native_control_frames_privileged") or ())
    if not rows:
        raise ValueError("TRUTH139_AXIS_V2_EMPTY_NATIVE_FRAMES")
    by_key: dict[tuple[str, str], int] = {}
    world: list[np.ndarray] = []
    for index, row in enumerate(rows):
        key = (str(row.get("armature") or ""), str(row.get("name") or ""))
        if not key[0] or not key[1] or key in by_key:
            raise ValueError("TRUTH139_AXIS_V2_NATIVE_FRAME_IDENTITY")
        by_key[key] = index
        world.append(_matrix(row.get("rest_matrix_world"), "TRUTH139_AXIS_V2_REST_MATRIX"))
    parent = np.full(len(rows), -1, dtype=np.int64)
    position = np.zeros((len(rows), 3), dtype=np.float64)
    for index, row in enumerate(rows):
        parent_name = row.get("parent")
        if parent_name not in (None, ""):
            p = by_key.get((str(row["armature"]), str(parent_name)))
            if p is None:
                raise ValueError("TRUTH139_AXIS_V2_NATIVE_PARENT_MISSING")
            parent[index] = p
        position[index] = world[index][:3, 3]
    for start in range(len(parent)):
        seen: set[int] = set()
        node = start
        while node >= 0:
            if node in seen:
                raise ValueError("TRUTH139_AXIS_V2_NATIVE_PARENT_CYCLE")
            seen.add(node)
            node = int(parent[node])
    return rows, parent, position, world


def _skin_mass(arrays: dict[str, np.ndarray], bone_count: int) -> np.ndarray:
    vertices = np.asarray(arrays["rest_vertices_source_world"], dtype=np.float64)
    offsets = np.asarray(arrays["source_weight_offsets"], dtype=np.int64)
    indices = np.asarray(arrays["source_weight_bone_indices_privileged"], dtype=np.int64)
    weights = np.asarray(arrays["source_weight_values"], dtype=np.float64)
    coverage = np.asarray(arrays["source_weight_coverage"], dtype=bool)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or offsets.shape != (len(vertices) + 1,)
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
        raise ValueError("TRUTH139_AXIS_V2_SOURCE_SKIN_CSR_INVALID")
    mass = np.zeros(bone_count, dtype=np.float64)
    np.add.at(mass, indices, weights)
    return mass


def _select_deform(parent: np.ndarray, mass: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("TRUTH139_AXIS_V2_SUPPORT_EPSILON_INVALID")
    seeds = set(np.flatnonzero(mass > epsilon).tolist())
    if not seeds:
        raise ValueError("TRUTH139_AXIS_V2_NO_SKIN_SUPPORTED_DEFORM_JOINT")
    selected = set(seeds)
    for seed in sorted(seeds):
        bridge: list[int] = []
        node = int(parent[seed])
        while node >= 0:
            if node in seeds:
                selected.update(bridge)
                break
            bridge.append(node)
            node = int(parent[node])
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    local = {int(source): index for index, source in enumerate(selected_array.tolist())}
    projected = np.full(len(selected_array), -1, dtype=np.int64)
    for index, source in enumerate(selected_array.tolist()):
        node = int(parent[source])
        while node >= 0 and node not in local:
            node = int(parent[node])
        projected[index] = -1 if node < 0 else local[node]
    return selected_array, projected


def _content_order(positions: np.ndarray, parents: np.ndarray) -> np.ndarray:
    children = [[] for _ in range(len(parents))]
    roots: list[int] = []
    for index, parent in enumerate(parents.tolist()):
        if parent < 0:
            roots.append(index)
        else:
            children[int(parent)].append(index)
    state = np.zeros(len(parents), dtype=np.int8)
    cache: list[Any] = [None] * len(parents)

    def signature(node: int):
        if state[node] == 1:
            raise ValueError("TRUTH139_AXIS_V2_PROJECTED_PARENT_CYCLE")
        if state[node] == 2:
            return cache[node]
        state[node] = 1
        point = tuple(float(0.0 if x == 0.0 else x) for x in positions[node])
        child_sig = tuple(sorted(signature(child) for child in children[node]))
        cache[node] = (point, child_sig)
        state[node] = 2
        return cache[node]

    for root in roots:
        signature(root)
    if np.any(state != 2):
        raise ValueError("TRUTH139_AXIS_V2_PROJECTED_NODE_UNREACHABLE")
    out: list[int] = []

    def emit(node: int) -> None:
        out.append(node)
        groups: dict[tuple, list[int]] = {}
        for child in children[node]:
            groups.setdefault(cache[child], []).append(child)
        for key in sorted(groups):
            for child in groups[key]:
                emit(child)

    groups: dict[tuple, list[int]] = {}
    for root in roots:
        groups.setdefault(cache[root], []).append(root)
    for key in sorted(groups):
        for root in groups[key]:
            emit(root)
    result = np.asarray(out, dtype=np.int64)
    if set(result.tolist()) != set(range(len(parents))):
        raise ValueError("TRUTH139_AXIS_V2_CONTENT_ORDER_NOT_PERMUTATION")
    return result


def _polar_rotation(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("TRUTH139_AXIS_V2_FRAME_LINEAR_INVALID")
    u, _, vt = np.linalg.svd(value)
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError("TRUTH139_AXIS_V2_FRAME_NOT_ORTHONORMAL")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("TRUTH139_AXIS_V2_FRAME_NOT_RIGHT_HANDED")
    return rotation


def _behavioral_dof(
    scene: dict,
    arrays: dict[str, np.ndarray],
    frame_rows: list[dict],
) -> dict[int, dict[int, float]]:
    refs = list(scene.get("privileged_control_refs") or ())
    actions = list((scene.get("calibration") or {}).get("actions") or ())
    isolated = np.asarray(arrays["isolated_position"], dtype=np.float64)
    baseline = np.asarray(arrays["pose_identity_baseline"], dtype=np.float64)
    vertices = np.asarray(arrays["rest_vertices_source_world"], dtype=np.float64)
    if isolated.shape != (len(actions), len(vertices), 3) or baseline.shape != vertices.shape:
        raise ValueError("TRUTH139_AXIS_V2_ISOLATED_RESPONSE_LAYOUT")
    diagonal = float(np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0)))
    if not math.isfinite(diagonal) or diagonal <= 1e-12:
        raise ValueError("TRUTH139_AXIS_V2_SOURCE_DIAGONAL_DEGENERATE")
    frame_by_key = {
        (str(row["armature"]), str(row["name"])): index
        for index, row in enumerate(frame_rows)
    }
    control_binding: dict[int, tuple[int, int]] = {}
    for control_index, ref in enumerate(refs):
        lookup = dict(ref.get("source_lookup_privileged_only") or {})
        source_index = frame_by_key.get((str(lookup.get("armature") or ""), str(lookup.get("bone") or "")))
        axis = int(lookup.get("axis", -1))
        if source_index is None or axis not in (0, 1, 2):
            raise ValueError("TRUTH139_AXIS_V2_CONTROL_REF_BINDING")
        control_binding[control_index] = (source_index, axis)
    response = {index: {0: 0.0, 1: 0.0, 2: 0.0} for index in range(len(frame_rows))}
    for row_index, action in enumerate(actions):
        binding = control_binding.get(int(action.get("control_index_privileged", -1)))
        if binding is None:
            raise ValueError("TRUTH139_AXIS_V2_ACTION_CONTROL_BINDING")
        source_index, axis = binding
        displacement = isolated[row_index] - baseline
        magnitude = float(np.max(np.linalg.norm(displacement, axis=1))) / diagonal
        response[source_index][axis] = max(response[source_index][axis], magnitude)
    return response


def _maximal_chains(parents: np.ndarray) -> list[tuple[int, ...]]:
    """Return maximal parent-contiguous segments between roots/branches/leaves."""
    children: list[list[int]] = [[] for _ in range(len(parents))]
    roots: list[int] = []
    for index, parent in enumerate(parents.tolist()):
        if parent < 0:
            roots.append(index)
        else:
            children[int(parent)].append(index)
    chains: list[tuple[int, ...]] = []
    # A segment begins at every root or branching joint. This is intentionally
    # based on the node's own out-degree, not its parent's out-degree.
    starts = sorted(set(roots + [i for i, row in enumerate(children) if len(row) > 1]))
    for start in starts:
        for child in sorted(children[start]):
            path = [start, child]
            node = child
            while len(children[node]) == 1:
                node = children[node][0]
                path.append(node)
            chains.append(tuple(path))
    return sorted(set(chains))


def _pole(points: np.ndarray, root_basis: np.ndarray) -> np.ndarray:
    root, end = points[0], points[-1]
    axis = end - root
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-10:
        raise ValueError("TRUTH139_AXIS_V2_IK_CHAIN_ENDPOINT_DEGENERATE")
    axis = axis / axis_norm
    best: np.ndarray | None = None
    best_norm = 0.0
    for point in points[1:-1]:
        radial = (point - root) - axis * float(np.dot(point - root, axis))
        norm = float(np.linalg.norm(radial))
        if norm > best_norm:
            best, best_norm = radial, norm
    if best is None or best_norm <= 1e-8:
        for candidate in root_basis:
            radial = candidate - axis * float(np.dot(candidate, axis))
            norm = float(np.linalg.norm(radial))
            if norm > best_norm:
                best, best_norm = radial, norm
    if best is None or best_norm <= 1e-8:
        candidate = min(np.eye(3), key=lambda row: abs(float(np.dot(row, axis))))
        best = candidate - axis * float(np.dot(candidate, axis))
        best_norm = float(np.linalg.norm(best))
    if best_norm <= 1e-10:
        raise ValueError("TRUTH139_AXIS_V2_IK_POLE_DIRECTION_DEGENERATE")
    total = float(sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)))
    return (root + end) * 0.5 + best / best_norm * max(0.5 * total, 1e-4)


def build(
    scene_path: Path,
    geometry_path: Path,
    output_dir: Path,
    *,
    support_epsilon: float = 1e-8,
    dof_response_threshold_normalized: float = 1e-6,
) -> dict:
    scene_path = scene_path.resolve()
    geometry_path = geometry_path.resolve()
    output_dir = output_dir.resolve()
    if not math.isfinite(dof_response_threshold_normalized) or dof_response_threshold_normalized < 0.0:
        raise ValueError("TRUTH139_AXIS_V2_DOF_THRESHOLD_INVALID")
    scene, arrays = _load(scene_path, geometry_path)
    frame_rows, parent_source, position_source, world_frames = _source_graph(scene)
    skin_mass = _skin_mass(arrays, len(frame_rows))
    selected, projected_parent = _select_deform(parent_source, skin_mass, support_epsilon)
    order = _content_order(position_source[selected], projected_parent)
    selected = selected[order]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    parents = np.asarray([
        -1 if projected_parent[old] < 0 else inverse[int(projected_parent[old])]
        for old in order.tolist()
    ], dtype=np.int64)
    if any(parent >= index for index, parent in enumerate(parents.tolist()) if parent >= 0):
        raise ValueError("TRUTH139_AXIS_V2_PARENT_NOT_CAUSAL")

    source_to_canonical = _matrix(
        (scene.get("coordinate_contract") or {}).get("source_world_to_normalized"),
        "TRUTH139_AXIS_V2_SOURCE_TO_CANONICAL",
    )
    source_positions = position_source[selected]
    homogeneous = np.concatenate((source_positions, np.ones((len(source_positions), 1))), axis=1)
    canonical_h = (source_to_canonical @ homogeneous.T).T
    if not np.allclose(canonical_h[:, 3], 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("TRUTH139_AXIS_V2_CANONICAL_HOMOGENEOUS")
    position = canonical_h[:, :3]
    response = _behavioral_dof(scene, arrays, frame_rows)

    joint_ids = tuple(f"T:J:{index:04d}" for index in range(len(selected)))
    joints: list[RigProgramJointProposalIR] = []
    frames: list[RigProgramLocalFrameProposalIR] = []
    basis_by_joint: dict[int, np.ndarray] = {}
    dof_evidence: list[dict] = []
    for index, source_index in enumerate(selected.tolist()):
        parent = int(parents[index])
        basis = _polar_rotation(source_to_canonical[:3, :3] @ world_frames[source_index][:3, :3])
        basis_by_joint[index] = basis
        values = response[source_index]
        labels = tuple(
            label for axis, label in enumerate(("RX", "RY", "RZ"))
            if float(values[axis]) >= dof_response_threshold_normalized
        )
        joints.append(RigProgramJointProposalIR(
            proposal_joint_id=joint_ids[index],
            P=tuple(map(float, position[index])),
            parent_proposal_joint_id=None if parent < 0 else joint_ids[parent],
            sequence_index=index,
            metadata={"truth_authority": "SOURCE_MEASURED_DEFORM_GRAPH_AND_REST_LOCUS"},
        ))
        frames.append(RigProgramLocalFrameProposalIR(
            proposal_joint_id=joint_ids[index],
            basis=tuple(tuple(float(v) for v in row) for row in basis),
            dof_labels=labels,
            metadata={
                "frame_authority": "SOURCE_REST_FRAME_POLAR_ROTATION_IN_CANONICAL_SPACE",
                "dof_authority": "MEASURED_ISOLATED_DEFORMATION_RESPONSE",
                "physical_joint_limit_claimed": False,
            },
        ))
        dof_evidence.append({
            "joint_index": index,
            "response_over_source_diagonal": [float(values[axis]) for axis in range(3)],
            "active_labels": list(labels),
        })

    roots = [index for index, parent in enumerate(parents.tolist()) if parent < 0]
    global_rest = np.mean(position[roots], axis=0)
    controls: list[RigProgramControlProposalIR] = [
        RigProgramControlProposalIR(
            "T:C:GLOBAL", "GLOBAL", (), None, tuple(map(float, global_rest)),
            {"teacher_authority": DERIVED_AUTHORITY},
        )
    ]
    joint_control: dict[int, str] = {}
    for index, parent in enumerate(parents.tolist()):
        cid = f"T:C:J:{index:04d}"
        joint_control[index] = cid
        controls.append(RigProgramControlProposalIR(
            cid,
            "ROOT" if parent < 0 else "FK",
            (joint_ids[index],),
            "T:C:GLOBAL" if parent < 0 else joint_control[int(parent)],
            tuple(map(float, position[index])),
            {"teacher_authority": DERIVED_AUTHORITY},
        ))

    children: list[list[int]] = [[] for _ in range(len(parents))]
    for index, parent in enumerate(parents.tolist()):
        if parent >= 0:
            children[int(parent)].append(index)
    chains: list[RigProgramChainProposalIR] = []
    constraints: list[RigProgramConstraintProposalIR] = []
    chain_evidence: list[dict] = []
    for chain_index, members in enumerate(_maximal_chains(parents)):
        chain_id = f"T:K:{chain_index:04d}"
        member_ids = tuple(joint_ids[index] for index in members)
        chains.append(RigProgramChainProposalIR(
            chain_id, member_ids, "GENERIC_KINEMATIC_CHAIN", {"teacher_authority": DERIVED_AUTHORITY}
        ))
        points = position[np.asarray(members, dtype=np.int64)]
        segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
        terminal = len(children[members[-1]]) == 0
        ik_eligible = bool(len(members) >= 3 and terminal and np.all(segment_lengths > 1e-8))
        evidence = {
            "chain_index": chain_index,
            "joint_indices": list(map(int, members)),
            "terminal": terminal,
            "ik_eligible": ik_eligible,
        }
        if ik_eligible:
            effector = f"T:C:IK:{chain_index:04d}"
            pole = f"T:C:POLE:{chain_index:04d}"
            controls.extend((
                RigProgramControlProposalIR(
                    effector, "IK_EFFECTOR", (), "T:C:GLOBAL", tuple(map(float, points[-1])),
                    {"teacher_authority": DERIVED_AUTHORITY},
                ),
                RigProgramControlProposalIR(
                    pole, "IK_POLE", (), "T:C:GLOBAL", tuple(map(float, _pole(points, basis_by_joint[members[0]]))),
                    {"teacher_authority": DERIVED_AUTHORITY},
                ),
            ))
            parameters = {"p0": 1.0, "p1": 1.0, "p2": 1.0, "p3": 0.0}
            validate_rig_constraint_operator_v1(
                operator_id="IK",
                input_control_count=2,
                output_joint_count=len(member_ids),
                has_chain=True,
                parameters=parameters,
            )
            constraints.append(RigProgramConstraintProposalIR(
                f"T:Q:IK:{chain_index:04d}", "IK", (effector, pole), member_ids, chain_id,
                parameters,
                {"teacher_authority": DERIVED_AUTHORITY, "source_authored_constraint_claimed": False},
            ))
            evidence.update({"effector_control_id": effector, "pole_control_id": pole})
        chain_evidence.append(evidence)

    binding_hash = _digest({
        "teacher_scene_sha256": _sha256(scene_path),
        "geometry_sha256": _sha256(geometry_path),
        "subject_id": str(scene.get("subject_key") or ""),
    })
    proposal = seal_rig_program_proposal_v1(RigProgramProposalIR(
        joints=tuple(joints),
        local_frames=tuple(frames),
        controls=tuple(controls),
        chains=tuple(chains),
        constraints=tuple(constraints),
        attachment_anchors=(),
        mesh_lineage_hash=binding_hash,
        feature_evidence_binding_hash=binding_hash,
        model_provenance="TRUTH139_AXIS_RIG_PROGRAM_TEACHER_V2__NO_MODEL_INFERENCE",
        metadata={
            "authority_class": "TEACHER_PROJECTION_ONLY",
            "control_program_authority": DERIVED_AUTHORITY,
            "source_authored_control_rig_claimed": False,
            "attachment_anchor_teacher_status": "ABSTAIN__EXACT_RIGID_OWNER_BINDING_NOT_MATERIALIZED_IN_V2",
            "skin_codes_present": False,
            "product_authority_claimed": False,
        },
    ))
    qualified = qualify_rig_program_v1(proposal)

    output_dir.mkdir(parents=True, exist_ok=False)
    proposal_path = output_dir / "RIG_PROGRAM_TEACHER.json"
    proposal_path.write_text(json.dumps(proposal.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n")
    evidence = {
        "schema": "RealSaS.Truth139AxisRigProgramTeacherEvidence.v2",
        "dof_response_threshold_normalized": float(dof_response_threshold_normalized),
        "dof": dof_evidence,
        "chains": chain_evidence,
        "source_names_exported": False,
        "source_joint_indices_exported_as_student_targets": False,
    }
    evidence_path = output_dir / "RIG_PROGRAM_TEACHER_EVIDENCE.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema": TEACHER_SCHEMA,
        "date": "2026-10-30",
        "status": "PASS_TRUTH139_AXIS_V6_RIG_PROGRAM_TEACHER__SKIN_CODE_BINDING_PENDING",
        "subject_id": str(scene.get("subject_key") or ""),
        "source_sha256": str(scene.get("source_sha256") or ""),
        "contract": CONTRACT,
        "teacher_scene": {"path": str(scene_path), "sha256": _sha256(scene_path)},
        "geometry_payload": {"path": str(geometry_path), "sha256": _sha256(geometry_path)},
        "rig_program_teacher": {"path": str(proposal_path), "sha256": _sha256(proposal_path)},
        "teacher_evidence": {"path": str(evidence_path), "sha256": _sha256(evidence_path)},
        "joint_count": len(joints),
        "root_count": len(roots),
        "control_count": len(controls),
        "chain_count": len(chains),
        "constraint_count": len(constraints),
        "active_dof_count": int(sum(len(row["active_labels"]) for row in dof_evidence)),
        "attachment_anchor_count": 0,
        "control_program_authority": DERIVED_AUTHORITY,
        "source_authored_control_rig_claimed": False,
        "compiler_qualification_hash": qualified.qualification_hash,
        "compiler_qualification_status": qualified.qualification_report["status"],
        "compiler_synthesis_performed": False,
        "mira_skin_code_binding_status": "PENDING_FROZEN_MIRA_V6_CODEC",
        "training_authorized": False,
        "product_authority_claimed": False,
        "qualification_minted": False,
    }
    manifest["axis_rig_program_teacher_identity_sha256"] = _digest(manifest)
    manifest_path = output_dir / "RIG_PROGRAM_TEACHER_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "subject_id": manifest["subject_id"],
        "joint_count": manifest["joint_count"],
        "control_count": manifest["control_count"],
        "chain_count": manifest["chain_count"],
        "constraint_count": manifest["constraint_count"],
        "active_dof_count": manifest["active_dof_count"],
        "identity": manifest["axis_rig_program_teacher_identity_sha256"],
    }, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-scene", required=True, type=Path)
    parser.add_argument("--geometry-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--support-epsilon", type=float, default=1e-8)
    parser.add_argument("--dof-response-threshold-normalized", type=float, default=1e-6)
    args = parser.parse_args()
    build(
        args.teacher_scene,
        args.geometry_npz,
        args.output_dir,
        support_epsilon=args.support_epsilon,
        dof_response_threshold_normalized=args.dof_response_threshold_normalized,
    )


if __name__ == "__main__":
    main()
