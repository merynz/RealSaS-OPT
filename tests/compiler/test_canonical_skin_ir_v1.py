from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v1 import (
    CanonicalRigIR,
    CanonicalRigJointIR,
    canonical_rig_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v1 import (
    canonical_skin_from_dict_v1,
    mint_canonical_skin_v1,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import (
    CarrierSkinInfluenceProposal,
    CarrierSkinProposalIR,
    qualify_carrier_skin_v1,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceSupportBinding


class _Carrier:
    def __init__(self, mesh: CanonicalMeshIR):
        self.ordered_vertex_ids = tuple(v.source_candidate_vertex_id for v in mesh.vertices)
        self.carrier_evidence_hash = mesh.carrier_evidence_hash
        self.topology_hash = mesh.carrier_topology_hash
        self.geometry_hash = mesh.carrier_geometry_hash


class _Skeleton:
    def __init__(self, rig: CanonicalRigIR):
        self.skeleton_lineage_hash = rig.structure_qualification_hash
        self.joints = tuple(type("J", (), {"canonical_joint_id": j.canonical_joint_id})() for j in rig.joints)


def _mesh() -> CanonicalMeshIR:
    support = SurfaceSupportBinding(mode="TEST", coefficients=(("S0", 1.0),))
    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=f"MV:{index}",
            support_binding=support,
            component_id="BODY",
            P=(float(index), 0.0, 0.0),
            source_candidate_vertex_id=f"CV:{index}",
        )
        for index in range(3)
    )
    value = CanonicalMeshIR(
        vertices=vertices,
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0", "MV:1"), ("MV:1", "MV:2"), ("MV:0", "MV:2")),
        source_evidence_binding_hash="surface-evidence",
        partition_binding_hash="partition",
        carrier_policy_binding_hash="carrier-policy",
        static_qualification_hash="static-qualification",
        carrier_evidence_hash="carrier-evidence",
        carrier_topology_hash="carrier-topology",
        carrier_geometry_hash="carrier-geometry",
        proposal_evidence_refs=(),
        source_candidate_lineage_hash="candidate-lineage",
        mesh_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_MESH"},
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def _rig(mesh: CanonicalMeshIR) -> CanonicalRigIR:
    joints = (
        CanonicalRigJointIR(
            canonical_joint_id="J:root",
            P=(0.0, 0.0, 0.0),
            parent_canonical_joint_id=None,
            source_proposal_id="PJ:root",
        ),
        CanonicalRigJointIR(
            canonical_joint_id="J:tip",
            P=(1.0, 0.0, 0.0),
            parent_canonical_joint_id="J:root",
            source_proposal_id="PJ:tip",
        ),
    )
    value = CanonicalRigIR(
        joints=joints,
        root_joint_ids=("J:root",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        feature_evidence_binding_hash="feature-evidence",
        proposal_evidence_hash="proposal-evidence",
        structure_qualification_hash="qualified-skeleton-lineage",
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "canonical_root_semantics": "NON_EMPTY_DEFORM_ROOT_SET",
        },
    )
    return replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v1(value))


def _qualified(mesh: CanonicalMeshIR, rig: CanonicalRigIR):
    carrier = _Carrier(mesh)
    skeleton = _Skeleton(rig)
    proposal = CarrierSkinProposalIR(
        influences=(
            CarrierSkinInfluenceProposal("CV:0", "J:root", 1.0),
            CarrierSkinInfluenceProposal("CV:1", "J:root", 0.5),
            CarrierSkinInfluenceProposal("CV:1", "J:tip", 0.5),
            CarrierSkinInfluenceProposal("CV:2", "J:tip", 1.0),
        ),
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        skeleton_binding_hash=rig.structure_qualification_hash,
        model_provenance="proposal-evidence-ref-0001",
        metadata={
            "output_domain": "MECHANICAL_CARRIER_M",
            "direct_simplex_readout": True,
            "semantic_skin_transfer_performed": False,
            "gsa_skin_field_authority_minted": False,
            "source_skin_runtime_authority": False,
            "mesh_mutation_invalidates_prediction": True,
        },
    )
    return qualify_carrier_skin_v1(
        carrier,
        skeleton,
        proposal,
        max_simplex_repair_l1=0.0,
        max_total_correction_l1=0.0,
        negative_tolerance=0.0,
        max_influences=16,
    )


def test_canonical_skin_maps_legacy_carrier_ids_to_canonical_mesh_ids():
    mesh = _mesh()
    rig = _rig(mesh)
    qualified = _qualified(mesh, rig)
    skin = mint_canonical_skin_v1(
        mesh,
        rig,
        qualified_skin=qualified,
        proposal_evidence_refs=("proposal-evidence-ref-0001",),
    )
    assert tuple(row.canonical_mesh_vertex_id for row in skin.rows) == ("MV:0", "MV:1", "MV:2")
    assert tuple(row.source_carrier_vertex_id for row in skin.rows) == ("CV:0", "CV:1", "CV:2")
    assert skin.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert skin.rig_lineage_hash == rig.rig_lineage_hash
    assert skin.metadata["authority"] == "COMPILER_CANONICAL_SKIN"
    assert skin.metadata["post_hoc_surface_skin_transfer_performed"] is False
    assert skin.metadata["deformation_authority_claimed"] is False


def test_canonical_skin_rejects_tampered_qualified_receipt():
    mesh = _mesh()
    rig = _rig(mesh)
    qualified = _qualified(mesh, rig)
    tampered = replace(
        qualified,
        qualification_report={**qualified.qualification_report, "carrier_vertex_count": 999},
    )
    with pytest.raises(QualificationError, match="QUALIFIED_RECEIPT_HASH_DRIFT"):
        mint_canonical_skin_v1(mesh, rig, qualified_skin=tampered)


def test_canonical_skin_rejects_rig_binding_drift():
    mesh = _mesh()
    rig = _rig(mesh)
    qualified = replace(_qualified(mesh, rig), skeleton_binding_hash="other-rig")
    # Re-seal the fabricated qualified receipt so this test reaches the parent-binding court.
    from compiler.realsas_compiler_core.hashing import content_sha256
    sealed = replace(
        qualified,
        skin_lineage_hash=content_sha256(
            {
                "carrier_evidence_hash": qualified.carrier_evidence_hash,
                "carrier_topology_hash": qualified.carrier_topology_hash,
                "carrier_geometry_hash": qualified.carrier_geometry_hash,
                "skeleton_binding_hash": qualified.skeleton_binding_hash,
                "rows": [row.to_dict() for row in qualified.rows],
                "qualification_report": qualified.qualification_report,
                "metadata": qualified.metadata,
            }
        ),
    )
    with pytest.raises(QualificationError, match="RIG_QUALIFICATION_BINDING_DRIFT"):
        mint_canonical_skin_v1(mesh, rig, qualified_skin=sealed)


def test_canonical_skin_codec_rejects_lineage_tamper():
    mesh = _mesh()
    rig = _rig(mesh)
    skin = mint_canonical_skin_v1(mesh, rig, qualified_skin=_qualified(mesh, rig))
    payload = skin.to_dict()
    restored = canonical_skin_from_dict_v1(payload)
    assert restored == skin
    payload["rows"][0]["influences"] = (("J:root", 0.25), ("J:tip", 0.75))
    with pytest.raises(QualificationError, match="LINEAGE_HASH_DRIFT"):
        canonical_skin_from_dict_v1(payload)
