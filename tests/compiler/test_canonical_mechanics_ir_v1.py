from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    canonical_mechanics_from_dict_v1,
    mint_canonical_mechanics_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
    canonical_rig_lineage_hash_v2,
    deform_structure_hash_v2,
    control_program_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2,
    CanonicalSkinRowIRV2,
    canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.types import SurfaceSupportBinding


def _support(i: int):
    return SurfaceSupportBinding("EXACT_SURFACE_NODE", ((f"S:{i}", 1.0),), {})


def _mesh():
    value = CanonicalMeshIR(
        vertices=(
            CanonicalMeshVertexIR("MV:0", _support(0), (0.0, 0.0, 0.0), "body", "CV:0"),
            CanonicalMeshVertexIR("MV:1", _support(1), (0.0, 1.0, 0.0), "body", "CV:1"),
            CanonicalMeshVertexIR("MV:2", _support(2), (1.0, 0.0, 0.0), "body", "CV:2"),
        ),
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0", "MV:1"), ("MV:0", "MV:2"), ("MV:1", "MV:2")),
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
    value = CanonicalRigIRV2(
        joints=(CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "PJ0"),),
        local_frames=(CanonicalJointFrameIRV2("J0", I, ("RX", "RY", "RZ")),),
        controls=(CanonicalRigControlIRV2("C0", "ROOT", ("J0",), None, (0.0, 0.0, 0.0), "PC0"),),
        chains=(), constraints=(), attachment_anchors=(), deform_root_joint_ids=("J0",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash="1" * 64,
        proposal_evidence_hash="a" * 64,
        rig_program_qualification_hash="b" * 64,
        deform_structure_hash="",
        control_program_hash="",
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "canonical_parent": "CANONICAL_MESH",
            "skin_authority_claimed": False,
            "motion_authority_claimed": False,
        },
    )
    value = replace(
        value,
        deform_structure_hash=deform_structure_hash_v2(value),
        control_program_hash=control_program_hash_v2(value),
    )
    return replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v2(value))


def _skin(mesh, rig):
    value = CanonicalSkinIRV2(
        rows=(
            CanonicalSkinRowIRV2("MV:0", (("J0", 1.0),), "CV:0"),
            CanonicalSkinRowIRV2("MV:1", (("J0", 1.0),), "CV:1"),
            CanonicalSkinRowIRV2("MV:2", (("J0", 1.0),), "CV:2"),
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
        skin_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_SKIN",
            "canonical_parents": ["CANONICAL_MESH", "CANONICAL_RIG_DEFORM_STRUCTURE"],
            "control_program_is_skin_parent": False,
            "learned_family_embedded_in_schema": False,
        },
    )
    return replace(value, skin_lineage_hash=canonical_skin_lineage_hash_v2(value))


def test_canonical_mechanics_seals_shared_compile_and_runtime_solver_contract():
    mesh = _mesh()
    rig = _rig(mesh)
    skin = _skin(mesh, rig)
    mechanics = mint_canonical_mechanics_v1(mesh, rig, skin)
    assert mechanics.metadata["authority"] == "COMPILER_CANONICAL_MECHANICS"
    assert mechanics.metadata["runtime_control_pose_solution_uses_same_solver"] is True
    assert mechanics.metadata["compiler_rig_synthesis_performed"] is False
    assert mechanics.constraint_schedule == ()
    assert canonical_mechanics_from_dict_v1(mechanics.to_dict()) == mechanics
