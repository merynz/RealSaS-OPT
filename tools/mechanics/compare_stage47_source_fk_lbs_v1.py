from __future__ import annotations

"""Stage47 measured Blender vs source-local FK/LBS vs RealSaS LBS oracle probe.

The source rig, named bone frames and source skin rows are PRIVILEGED TEACHER
data. This court neither trains a student nor constructs a qualified CanonicalRig.
It tests whether the *existing* RealSaS deformation kernel can consume a
source-exact, ORACLE-provided pose. It does not qualify source-to-canonical
geometry/rig mapping or independent AXIS generation.

Independent source replay uses an analytical one-joint descendant mass identity:
a single local rotation about a rest-frame bone rotates its entire rest subtree
by one common world transform; source LBS then mixes by descendants' total skin
mass per vertex. This independent implementation does not call any RealSaS
runtime helper. Optional product-kernel probes separately call deform_mesh_lbs_v1
on an explicit teacher-only oracle carrier, without mint_canonical_*.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.MechanicsStage47NativeFKLBSComparison.v1"


class Stage47CorrespondenceError(ValueError):
    pass


def _sha(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            hash_value.update(part)
    return hash_value.hexdigest()


def _rotation4(axis: int, radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    result = np.eye(4, dtype=np.float64)
    if axis == 0:
        result[1:3, 1:3] = ((c, -s), (s, c))
    elif axis == 1:
        result[0, 0], result[0, 2] = c, s
        result[2, 0], result[2, 2] = -s, c
    elif axis == 2:
        result[0:2, 0:2] = ((c, -s), (s, c))
    else:
        raise Stage47CorrespondenceError("SOURCE_REPLAY_AXIS_NOT_3D")
    return result


def _native_forest(scene: dict) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    refs = scene["privileged_control_refs"]
    frames = scene["calibration"]["native_control_frames_privileged"]
    if not frames or len(refs) != 3 * len(frames):
        raise Stage47CorrespondenceError("SOURCE_REPLAY_FRAME_CONTROL_CARDINALITY")
    names = [str(row["name"]) for row in frames]
    if len(set(names)) != len(names):
        raise Stage47CorrespondenceError("SOURCE_REPLAY_DUPLICATE_BONE")
    by_name = {name: i for i, name in enumerate(names)}
    matrices = np.asarray([row["rest_matrix_world"] for row in frames], dtype=np.float64)
    if (matrices.shape != (len(frames), 4, 4) or
            not np.isfinite(matrices).all() or
            np.max(np.abs(matrices[:, 3] - (0, 0, 0, 1))) > 1e-6):
        raise Stage47CorrespondenceError("SOURCE_REPLAY_REST_MATRIX_INVALID")
    if (np.min(np.linalg.det(matrices[:, :3, :3])) <= 0.0 or
            np.max(np.abs(matrices[:, :3, :3] @
                          matrices[:, :3, :3].transpose(0, 2, 1)
                          - np.eye(3))) > 1e-3):
        raise Stage47CorrespondenceError("SOURCE_REPLAY_REST_FRAME_NOT_RIGID")
    parents = []
    for i, row in enumerate(frames):
        if row.get("parent") is None:
            parents.append(-1)
        elif row["parent"] not in by_name:
            raise Stage47CorrespondenceError("SOURCE_REPLAY_PARENT_MISSING")
        else:
            parents.append(by_name[row["parent"]])
        for axis in range(3):
            ref = refs[3*i + axis]
            lookup = ref.get("source_lookup_privileged_only", {})
            if (ref.get("kind") != "NATIVE_POSE_BONE_LOCAL_ROTATION"
                    or ref.get("units") != "RADIAN"
                    or ref.get("authored_capability") !=
                    "KINEMATIC_PARAMETER__NOT_PHYSICAL_DOF_OR_JOINT_LIMIT"
                    or int(ref.get("axis_in_native_rest_frame", -1)) != axis
                    or lookup.get("axis") != axis
                    or lookup.get("bone") != names[i]
                    or lookup.get("armature") != row.get("armature")
                    or lookup.get("source_sha256") != scene["source_sha256"]):
                raise Stage47CorrespondenceError("SOURCE_REPLAY_CONTROL_FRAME_BINDING_DRIFT")
    parents = np.asarray(parents, dtype=np.int64)
    descendants = [[] for _ in frames]
    for i in range(len(frames)):
        visiting: set[int] = set()
        node = i
        while node != -1:
            if node in visiting:
                raise Stage47CorrespondenceError("SOURCE_REPLAY_PARENT_CYCLE")
            visiting.add(node)
            descendants[node].append(i)
            node = int(parents[node])
    return matrices, parents, descendants


def _weights(z, bone_count: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(z["source_weight_offsets"], dtype=np.int64)
    values = np.asarray(z["source_weight_values"], dtype=np.float64)
    bones = np.asarray(z["source_weight_bone_indices_privileged"], dtype=np.int64)
    coverage = np.asarray(z["source_weight_coverage"], dtype=np.bool_)
    if (offsets.shape != (n+1,) or offsets[0] != 0 or
            np.any(np.diff(offsets) < 0) or
            offsets[-1] != len(values) or
            bones.shape != values.shape or coverage.shape != (n,) or
            not np.isfinite(values).all() or np.any(values < 0) or
            np.any(bones < 0) or np.any(bones >= bone_count) or
            not coverage.all()):
        raise Stage47CorrespondenceError("SOURCE_REPLAY_SKIN_CSR_INVALID_OR_MASKED")
    indices = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
    row_sums = np.bincount(indices, weights=values, minlength=n)
    if np.max(np.abs(row_sums - 1.0)) > 1e-6:
        raise Stage47CorrespondenceError("SOURCE_REPLAY_SKIN_SIMPLEX_INVALID")
    return indices, bones, values, offsets


def _teacher_oracle_product_probe(
    rest: np.ndarray,
    faces: np.ndarray,
    mats: np.ndarray,
    parents: np.ndarray,
    descendants: list[list[int]],
    indices: np.ndarray,
    bones: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
    actions: list[dict],
    observed: np.ndarray,
    *,
    selected_indices: list[int],
) -> dict:
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
        CanonicalMeshIR, CanonicalMeshVertexIR,
    )
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
        CanonicalRigIRV2, CanonicalDeformJointIRV2,
    )
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
        CanonicalSkinIRV2, CanonicalSkinRowIRV2,
    )
    from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
        MechanicsPoseIRV1, SolvedJointTransformIRV1, deform_mesh_lbs_v1,
    )
    from compiler.realsas_compiler_core.types import SurfaceSupportBinding

    # This is a typed *oracle probe*, not product minting. The SHA identifies
    # only an ephemeral research instance; none of these objects are sealed.
    vertex_ids = tuple(f"SOURCE_ORACLE:V:{i}" for i in range(len(rest)))
    mesh_sha = hashlib.sha256(rest.tobytes() + faces.tobytes()).hexdigest()
    rig_sha = hashlib.sha256(mats.tobytes() + parents.tobytes()).hexdigest()
    skin_sha = hashlib.sha256(
        indices.tobytes() + bones.tobytes() + weights.tobytes()
    ).hexdigest()
    vertices = tuple(CanonicalMeshVertexIR(
        canonical_mesh_vertex_id=vid,
        support_binding=SurfaceSupportBinding("EXACT_SURFACE_NODE", ((f"ORACLE:{i}",1.0),), {}),
        component_id="SOURCE_ORACLE_UNQUALIFIED", P=tuple(map(float, rest[i])),
        source_candidate_vertex_id=f"TEACHER_ONLY:{i}",
    ) for i, vid in enumerate(vertex_ids))
    mesh = CanonicalMeshIR(
        vertices=vertices, faces=(), edges=(),
        source_evidence_binding_hash="1"*64, partition_binding_hash="2"*64,
        carrier_policy_binding_hash="3"*64, static_qualification_hash="4"*64,
        carrier_evidence_hash="5"*64, carrier_topology_hash="6"*64,
        carrier_geometry_hash="7"*64, proposal_evidence_refs=(),
        source_candidate_lineage_hash="8"*64, mesh_lineage_hash=mesh_sha,
        metadata={"authority": "UNQUALIFIED_SOURCE_ORACLE_RESEARCH_ONLY"},
    )
    joints = tuple(CanonicalDeformJointIRV2(
        canonical_joint_id=f"SOURCE_ORACLE:J:{i}",
        P=tuple(map(float, mats[i, :3, 3])),
        parent_canonical_joint_id=None if parents[i] == -1
             else f"SOURCE_ORACLE:J:{parents[i]}",
        source_proposal_joint_id=f"TEACHER_ONLY:{i}",
    ) for i in range(len(mats)))
    rig = CanonicalRigIRV2(
        joints=joints, local_frames=(), controls=(), chains=(),
        constraints=(), attachment_anchors=(),
        deform_root_joint_ids=tuple(j.canonical_joint_id for j in joints
                                    if j.parent_canonical_joint_id is None),
        mesh_lineage_hash=mesh_sha,
        feature_evidence_binding_hash="1"*64, proposal_evidence_hash="2"*64,
        rig_program_qualification_hash="3"*64,
        deform_structure_hash=rig_sha, control_program_hash="4"*64,
        rig_lineage_hash=rig_sha,
        metadata={"authority": "UNQUALIFIED_SOURCE_ORACLE_RESEARCH_ONLY"},
    )
    grouped = [[] for _ in vertex_ids]
    for vertex_id, bone_id, w in zip(indices, bones, weights):
        grouped[int(vertex_id)].append((f"SOURCE_ORACLE:J:{int(bone_id)}",float(w)))
    skin = CanonicalSkinIRV2(
        rows=tuple(CanonicalSkinRowIRV2(
            canonical_mesh_vertex_id=vid, influences=tuple(grouped[i]),
            source_carrier_vertex_id=f"TEACHER_ONLY:{i}",
        ) for i, vid in enumerate(vertex_ids)),
        mesh_lineage_hash=mesh_sha, deform_structure_hash=rig_sha,
        mesh_carrier_evidence_hash="5"*64, mesh_carrier_topology_hash="6"*64,
        mesh_carrier_geometry_hash="7"*64,
        carrier_skin_qualification_hash="8"*64, skin_code_proposal_hash="9"*64,
        skin_codec_architecture_id="TEACHER_ORACLE_ONLY",
        skin_codec_checkpoint_hash="a"*64, skin_lineage_hash=skin_sha,
        metadata={"authority": "UNQUALIFIED_SOURCE_ORACLE_RESEARCH_ONLY"},
    )
    maximum = 0.0
    per_probe: list[float] = []
    for action_index in selected_indices:
        action = actions[action_index]
        bone_id = int(action["control_index_privileged"]) // 3
        axis = int(action["control_index_privileged"]) % 3
        transform = mats[bone_id] @ _rotation4(
            axis, float(action["sign"])*float(action["amplitude"])
        ) @ np.linalg.inv(mats[bone_id])
        active = set(descendants[bone_id])
        solved_joints = []
        for i, joint in enumerate(joints):
            rest_p = mats[i, :3, 3]
            if i in active:
                pos = transform[:3,:3] @ rest_p + transform[:3,3]
                rotation = transform[:3,:3]
            else:
                pos = rest_p
                rotation = np.eye(3)
            solved_joints.append(SolvedJointTransformIRV1(
                canonical_joint_id=joint.canonical_joint_id,
                rest_P=joint.P, solved_P=tuple(map(float, pos)),
                rotation_basis=tuple(tuple(map(float,row)) for row in rotation),
            ))
        pose = MechanicsPoseIRV1(
            joints=tuple(solved_joints), rig_lineage_hash=rig_sha,
            control_program_hash=rig.control_program_hash,
            metadata={"authority": "TEACHER_ORACLE_POSE_UNQUALIFIED"},
        )
        result = deform_mesh_lbs_v1(mesh, rig, skin, pose)
        arr = np.asarray(result.positions, dtype=np.float64)
        error = float(np.max(np.linalg.norm(arr-observed[action_index],axis=1)))
        per_probe.append(error)
        maximum = max(maximum,error)
    return {
        "probe_count": len(per_probe),
        "probe_action_indices": selected_indices,
        "max_vertex_error_world": maximum,
        "median_probe_max_error_world": float(np.median(per_probe)),
        "product_deformer_invoked": True,
        "oracle_pose_supplied": True,
        "canonical_rig_minted": False,
        "compiler_skeleton_or_controls_produced": False,
    }


def compare_stage47_source_fk_lbs_v1(
    teacher_path: Path,
    geometry_path: Path,
    *,
    error_threshold_source_world: float = 2.5e-6,
    product_probe_count: int = 0,
    expected_teacher_sha256: str | None = None,
    expected_geometry_sha256: str | None = None,
) -> dict:
    if (not math.isfinite(float(error_threshold_source_world)) or
            error_threshold_source_world <= 0 or
            product_probe_count < 0 or product_probe_count > 128):
        raise Stage47CorrespondenceError("SOURCE_REPLAY_POLICY_INVALID")
    teacher_sha, geometry_sha = _sha(teacher_path), _sha(geometry_path)
    if expected_teacher_sha256 and teacher_sha != expected_teacher_sha256:
        raise Stage47CorrespondenceError("SOURCE_REPLAY_TEACHER_SHA_DRIFT")
    if expected_geometry_sha256 and geometry_sha != expected_geometry_sha256:
        raise Stage47CorrespondenceError("SOURCE_REPLAY_GEOMETRY_SHA_DRIFT")
    scene = json.loads(teacher_path.read_text(encoding="utf-8"))
    if (scene.get("schema") != "RealSaS.TeacherSceneIR.v2"
            or scene.get("student_input_allowed") is not False
            or scene.get("qualification_minted") is not False
            or scene.get("native_temporal_rest_compatible") is not False
            or scene.get("coordinate_contract", {}).get("frame") != "SOURCE_WORLD"
            or scene.get("constraint_capabilities", {}).get(
                "source_weight_lbs_equivalence") != "NOT_QUALIFIED"
            or len(scene.get("geometry_payloads") or ()) != 1
            or scene["geometry_payloads"][0]["sha256"] != geometry_sha):
        raise Stage47CorrespondenceError("SOURCE_REPLAY_AUTHORITY_OR_GEOMETRY_BINDING_DRIFT")

    mats, parents, descendants = _native_forest(scene)
    actions = scene["calibration"]["actions"]
    if not actions or len(actions)%2:
        raise Stage47CorrespondenceError("SOURCE_REPLAY_ACTION_COUNT_INVALID")
    controls = scene["privileged_control_refs"]
    if len(actions) != len(controls)*6:
        raise Stage47CorrespondenceError("SOURCE_REPLAY_SIGN_PAIR_OR_AMPLITUDE_DRIFT")
    with np.load(geometry_path, allow_pickle=False) as z:
        rest = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        observed = np.asarray(z["isolated_position"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        if (rest.ndim != 2 or rest.shape[1] != 3
                or observed.shape != (len(actions), len(rest), 3)
                or faces.ndim != 2 or faces.shape[1] != 3
                or not np.isfinite(rest).all() or not np.isfinite(observed).all()):
            raise Stage47CorrespondenceError("SOURCE_REPLAY_GEOMETRY_SHAPE_INVALID")
        indices, bones, weights, offsets = _weights(z, len(mats), len(rest))
        # Source-native action (Blender) is independent of the analytical
        # descendant-mass FK/LBS replayer implemented below.
        mass = np.zeros((len(mats),len(rest)),dtype=np.float64)
        for bone, bone_descendants in enumerate(descendants):
            valid = np.isin(bones,bone_descendants)
            np.add.at(mass[bone],indices[valid],weights[valid])
        max_err = 0.0
        sum_sq = 0.0
        sum_l1 = 0.0
        count = 0
        per_action_max: list[float] = []
        signal: list[float] = []
        for index, action in enumerate(actions):
            cid = int(action["control_index_privileged"])
            if cid < 0 or cid >= len(controls):
                raise Stage47CorrespondenceError("SOURCE_REPLAY_ACTION_CONTROL_OUT_OF_RANGE")
            if (action["opaque_privileged_handle"] != controls[cid]["opaque_privileged_handle"]
                    or action["kind"] != "ISOLATED_KINEMATIC_PARAMETER_PERTURBATION"
                    or action["units"] != "RADIAN"
                    or action["sign"] != (-1 if index%2==0 else 1)
                    or cid != index//6):
                raise Stage47CorrespondenceError("SOURCE_REPLAY_ACTION_REFERENT_DRIFT")
            if index%2 == 0:
                next_action = actions[index+1]
                if (next_action["amplitude"] != action["amplitude"] or
                        next_action["amplitude_index"] != action["amplitude_index"] or
                        next_action["control_index_privileged"] != cid):
                    raise Stage47CorrespondenceError("SOURCE_REPLAY_ACTION_SIGN_PAIR_DRIFT")
            angle = float(action["sign"])*float(action["amplitude"])
            if not math.isfinite(angle) or not 0 < abs(angle) < math.pi:
                raise Stage47CorrespondenceError("SOURCE_REPLAY_ANGLE_INVALID")
            bone, axis = divmod(cid,3)
            T = mats[bone] @ _rotation4(axis,angle) @ np.linalg.inv(mats[bone])
            moved = rest @ T[:3,:3].T + T[:3,3]
            oracle = rest + mass[bone,:,None]*(moved-rest)
            diff = oracle - observed[index]
            err = np.linalg.norm(diff,axis=-1)
            max_err = max(max_err,float(np.max(err)))
            sum_sq += float(np.square(diff).sum())
            sum_l1 += float(np.linalg.norm(diff,axis=-1).sum())
            count += len(rest)
            per_action_max.append(float(np.max(err)))
            signal.append(float(np.linalg.norm(observed[index]-rest,axis=-1).max()))
        probes = None
        if product_probe_count:
            selected = np.linspace(0,len(actions)-1,num=product_probe_count,dtype=np.int64).tolist()
            probes = _teacher_oracle_product_probe(
                rest,faces,mats,parents,descendants,
                indices,bones,weights,offsets,actions,observed,
                selected_indices=selected,
            )
        passed = max_err <= error_threshold_source_world and (
            probes is None or probes["max_vertex_error_world"] <= error_threshold_source_world
        )
        return {
            "schema": SCHEMA,
            "status": ("PASS_SOURCE_NATIVE_FK_LBS_AND_OPTIONAL_ORACLE_KERNEL"
                       if passed else "FAIL_SOURCE_NATIVE_FK_LBS_OR_ORACLE_KERNEL"),
            "source_subject_key": scene["subject_key"],
            "teacher_scene_sha256": teacher_sha,
            "geometry_response_sha256": geometry_sha,
            "source_native_bone_count": len(mats),
            "source_native_control_parameter_count": len(controls),
            "source_vertex_count": len(rest),
            "action_count": len(actions),
            "measured_nonzero_action_count": sum(v>1e-9 for v in signal),
            "source_native_fk_lbs_max_vertex_error_world": max_err,
            "source_native_fk_lbs_mean_vertex_error_world": sum_l1/count,
            "source_native_fk_lbs_rms_coordinate_error_world": math.sqrt(sum_sq/(3*count)),
            "source_native_fk_lbs_action_max_error_p95_world": float(
                np.percentile(per_action_max,95)),
            "source_native_fk_lbs_action_max_error_median_world": float(
                np.median(per_action_max)),
            "error_threshold_source_world": float(error_threshold_source_world),
            "oracle_product_deformer_probe": probes,
            "student_input_allowed": False,
            "source_to_canonical_geometry_binding_qualified": False,
            "source_to_canonical_rig_binding_qualified": False,
            "learned_mechanics_unseen_proven": False,
            "product_mechanics_authority_minted": False,
            "research_exposed_subject": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--npz",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--threshold-source-world",type=float,default=2.5e-6)
    parser.add_argument("--product-probes",type=int,default=0)
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument("--expected-npz-sha256")
    args=parser.parse_args()
    result=compare_stage47_source_fk_lbs_v1(
        args.teacher,args.npz,
        error_threshold_source_world=args.threshold_source_world,
        product_probe_count=args.product_probes,
        expected_teacher_sha256=args.expected_teacher_sha256,
        expected_geometry_sha256=args.expected_npz_sha256,
    )
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))
    if result["status"].startswith("FAIL"):
        raise SystemExit(1)


if __name__=="__main__":
    main()
