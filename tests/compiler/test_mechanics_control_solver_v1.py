from __future__ import annotations

from dataclasses import replace

import numpy as np

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigChainIRV2,
    CanonicalRigConstraintIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import CanonicalSkinIRV2, CanonicalSkinRowIRV2
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
    deform_mesh_lbs_v1,
    solve_control_pose_v1,
)
from compiler.realsas_compiler_core.types import SurfaceSupportBinding


def _support(i: int):
    return SurfaceSupportBinding("EXACT_SURFACE_NODE", ((f"S:{i}", 1.0),), {})


def _mesh():
    value = CanonicalMeshIR(
        vertices=(
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id="MV:0",
                support_binding=_support(0),
                component_id="body",
                P=(0.0, 0.0, 0.0),
                source_candidate_vertex_id="CV:0",
            ),
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id="MV:1",
                support_binding=_support(1),
                component_id="body",
                P=(0.0, 1.0, 0.0),
                source_candidate_vertex_id="CV:1",
            ),
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id="MV:2",
                support_binding=_support(2),
                component_id="body",
                P=(0.0, 2.0, 0.0),
                source_candidate_vertex_id="CV:2",
            ),
        ),
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0", "MV:1"), ("MV:1", "MV:2"), ("MV:0", "MV:2")),
        source_evidence_binding_hash="1" * 64,
        partition_binding_hash="2" * 64,
        carrier_policy_binding_hash="3" * 64,
        static_qualification_hash="4" * 64,
        carrier_evidence_hash="5" * 64,
        carrier_topology_hash="6" * 64,
        carrier_geometry_hash="7" * 64,
        proposal_evidence_refs=("8" * 64,),
        source_candidate_lineage_hash="9" * 64,
        mesh_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_MESH"},
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def _rig(mesh):
    I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return CanonicalRigIRV2(
        joints=(
            CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "PJ0"),
            CanonicalDeformJointIRV2("J1", (0.0, 1.0, 0.0), "J0", "PJ1"),
            CanonicalDeformJointIRV2("J2", (0.0, 2.0, 0.0), "J1", "PJ2"),
        ),
        local_frames=(
            CanonicalJointFrameIRV2("J0", I, ("RX", "RY", "RZ")),
            CanonicalJointFrameIRV2("J1", I, ("RX", "RY", "RZ")),
            CanonicalJointFrameIRV2("J2", I, ("RX", "RY", "RZ")),
        ),
        controls=(
            CanonicalRigControlIRV2("C_ROOT", "ROOT", ("J0",), None, (0.0, 0.0, 0.0), "PC0"),
            CanonicalRigControlIRV2("C_IK", "IK_EFFECTOR", ("J2",), "C_ROOT", (0.0, 2.0, 0.0), "PC1"),
            CanonicalRigControlIRV2("C_POLE", "IK_POLE", (), "C_ROOT", (1.0, 1.0, 0.0), "PC2"),
        ),
        chains=(CanonicalRigChainIRV2("K0", ("J0", "J1", "J2"), "GENERIC_KINEMATIC_CHAIN", "PK0"),),
        constraints=(
            CanonicalRigConstraintIRV2(
                "Q0", "IK", ("C_IK", "C_POLE"), ("J2",), "K0",
                {"p0": 1.0, "p1": 1.0, "p2": 1.0}, "PQ0",
            ),
        ),
        attachment_anchors=(),
        deform_root_joint_ids=("J0",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash="1" * 64,
        proposal_evidence_hash="a" * 64,
        rig_program_qualification_hash="b" * 64,
        deform_structure_hash="c" * 64,
        control_program_hash="d" * 64,
        rig_lineage_hash="e" * 64,
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )


def _skin(mesh, rig):
    return CanonicalSkinIRV2(
        rows=(
            CanonicalSkinRowIRV2("MV:0", (("J0", 1.0),), "CV:0"),
            CanonicalSkinRowIRV2("MV:1", (("J1", 1.0),), "CV:1"),
            CanonicalSkinRowIRV2("MV:2", (("J2", 1.0),), "CV:2"),
        ),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        carrier_skin_qualification_hash="f" * 64,
        skin_token_proposal_hash="1" * 64,
        skin_codec_architecture_id="codec",
        skin_codec_checkpoint_hash="2" * 64,
        skin_lineage_hash="3" * 64,
        metadata={"authority": "COMPILER_CANONICAL_SKIN"},
    )


def test_ik_effector_and_pole_drive_deform_chain_without_rig_mutation():
    mesh = _mesh()
    rig = _rig(mesh)
    target = (1.0, 1.5, 0.0)
    pole = (0.0, 1.0, 1.0)
    pose = solve_control_pose_v1(
        rig,
        (
            ControlTransformIRV1("C_IK", translation=(target[0], target[1] - 2.0, target[2])),
            ControlTransformIRV1("C_POLE", translation=(pole[0] - 1.0, pole[1] - 1.0, pole[2])),
        ),
    )
    by = {j.canonical_joint_id: np.asarray(j.solved_P) for j in pose.joints}
    assert np.linalg.norm(by["J2"] - np.asarray(target)) < 1e-4
    assert abs(np.linalg.norm(by["J1"] - by["J0"]) - 1.0) < 1e-4
    assert abs(np.linalg.norm(by["J2"] - by["J1"]) - 1.0) < 1e-4
    assert pose.metadata["rig_program_mutated"] is False
    assert pose.metadata["learned_inference_performed"] is False
    assert pose.constraint_residuals[0][1] < 1e-4


def test_lbs_consumes_canonical_skin_and_mechanics_pose():
    mesh = _mesh()
    rig = _rig(mesh)
    skin = _skin(mesh, rig)
    pose = solve_control_pose_v1(
        rig,
        (ControlTransformIRV1("C_ROOT", translation=(0.5, 0.0, 0.0)),),
    )
    deformed = deform_mesh_lbs_v1(mesh, rig, skin, pose)
    assert deformed.canonical_mesh_vertex_ids == ("MV:0", "MV:1", "MV:2")
    assert np.allclose(np.asarray(deformed.positions[0]), np.asarray((0.5, 0.0, 0.0)), atol=1e-6)
    assert deformed.metadata["deformation_method"] == "LINEAR_BLEND_SKINNING"
    assert deformed.metadata["visual_rebind_performed"] is False
