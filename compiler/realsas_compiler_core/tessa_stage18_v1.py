from __future__ import annotations

"""Subject-free Stage18 construction for the TESSA learned carrier arm."""

from typing import Any, Callable

import numpy as np

from .hashing import content_sha256
from .tessa_candidate_bridge_v1 import (
    build_tessa_candidate_bridge_v1,
    tessa_mesh_proposal_from_dict_v1,
)
from .tessa_surface_field_v1 import build_tessa_material_support_field_v1
from .types import QualificationError

Json = dict[str, Any]


def build_tessa_stage18_candidate_v1(
    *,
    mesh_cfg: Json,
    surface,
    partition,
    carrier_policy,
    mesh_policy,
    artifact_root,
    load_file_ref: Callable[..., dict],
    write_ir: Callable[..., dict],
    relation_quality_report: Callable[[Any, Any], dict],
) -> dict:
    """Build exact TESSA Stage18 outputs without minting product authority.

    Input is an exact-hash typed proposal.  Decoded proposal component labels are
    treated only as connected-component identities; Compiler deterministically
    maps those components onto the qualified mechanical partition while building
    the continuous material-support field.
    """
    allowed = {"backend", "proposal", "support_binding"}
    unknown = set(mesh_cfg) - allowed
    if unknown:
        raise QualificationError(
            "TESSA_STAGE18_MESH_CONFIG_KEYS_UNSUPPORTED:"
            + ",".join(sorted(map(str, unknown)))
        )
    if str(mesh_cfg.get("backend") or "") != "TESSA_PROPOSAL_V1":
        raise QualificationError("TESSA_STAGE18_BACKEND_INVALID")

    proposal_ref = dict(mesh_cfg.get("proposal") or {})
    if not proposal_ref:
        raise QualificationError("TESSA_STAGE18_PROPOSAL_REF_MISSING")
    proposal_payload = load_file_ref(
        proposal_ref,
        expected_schema="RealSaS.TESSAMeshProposal.v1",
    )
    proposal = tessa_mesh_proposal_from_dict_v1(proposal_payload)
    if proposal.source_geometry_lineage_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_STAGE18_PROPOSAL_SURFACE_LINEAGE_MISMATCH")

    support_cfg = dict(mesh_cfg.get("support_binding") or {})
    allowed_support = {
        "max_support_nodes",
        "max_graph_hops",
        "inverse_distance_power",
    }
    unknown_support = set(support_cfg) - allowed_support
    if unknown_support:
        raise QualificationError(
            "TESSA_STAGE18_SUPPORT_CONFIG_KEYS_UNSUPPORTED:"
            + ",".join(sorted(map(str, unknown_support)))
        )
    max_support_nodes = int(support_cfg.get("max_support_nodes", 4))
    max_graph_hops = int(support_cfg.get("max_graph_hops", 2))
    inverse_distance_power = float(support_cfg.get("inverse_distance_power", 2.0))

    component_labels = sorted({str(vertex.component_id) for vertex in proposal.vertices})
    if not component_labels or any(not label for label in component_labels):
        raise QualificationError("TESSA_STAGE18_DECODED_COMPONENT_LABEL_INVALID")
    component_index = {label: index for index, label in enumerate(component_labels)}
    decoded_components = np.asarray(
        [component_index[str(vertex.component_id)] for vertex in proposal.vertices],
        dtype=np.int64,
    )
    vertices_world = np.asarray(
        [tuple(map(float, vertex.P)) for vertex in proposal.vertices],
        dtype=np.float64,
    )
    proposal_vertex_ids = tuple(str(vertex.proposal_vertex_id) for vertex in proposal.vertices)

    support_field = build_tessa_material_support_field_v1(
        vertices_world=vertices_world,
        decoded_component_indices=decoded_components,
        surface=surface,
        partition=partition,
        topology_sequence_hash=str(proposal.topology_sequence_hash),
        proposal_vertex_ids=proposal_vertex_ids,
        max_support_nodes=max_support_nodes,
        max_graph_hops=max_graph_hops,
        inverse_distance_power=inverse_distance_power,
    )

    producer_policy_hash = content_sha256({
        "schema": "RealSaS.TESSAStage18ProducerPolicy.v1",
        "backend": "TESSA_PROPOSAL_V1",
        "proposal_sha256": str(proposal_ref.get("sha256") or ""),
        "proposal_geometry_lineage_hash": str(proposal.source_geometry_lineage_hash),
        "topology_sequence_hash": str(proposal.topology_sequence_hash),
        "support_binding": {
            "max_support_nodes": max_support_nodes,
            "max_graph_hops": max_graph_hops,
            "inverse_distance_power": inverse_distance_power,
        },
        "mesh_policy_hash": mesh_policy.qualification_policy_lineage_hash,
        "material_support_is_geometry_authority": False,
        "stage19_exact_candidate_remeasurement_required": True,
    })
    candidate, bridge_evidence = build_tessa_candidate_bridge_v1(
        proposal=proposal,
        support_field=support_field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        producer_policy_hash=producer_policy_hash,
    )

    quality = dict(relation_quality_report(candidate, mesh_policy))
    quality["schema"] = "RealSaS.TESSAStage18StaticPreflight.v1"
    quality["producer_semantics"] = "TESSA_LEARNED_GEOMETRY_PROPOSAL"
    quality["diagnostic_only"] = True
    quality["stage19_remeasurement_required"] = True
    quality["product_authority_claimed"] = False

    outputs = [
        write_ir(
            artifact_root / "canonical_mesh_candidate.json",
            candidate,
            authority_class="TESSA_LEARNED_MESH_CANDIDATE",
        ),
        write_ir(
            artifact_root / "mesh_qualification_policy.json",
            mesh_policy,
            authority_class="FROZEN_MESH_QUALIFICATION_POLICY",
        ),
        write_ir(
            artifact_root / "tessa_material_support_field.json",
            support_field,
            authority_class="TESSA_MATERIAL_SUPPORT_PROPOSAL",
        ),
        write_ir(
            artifact_root / "tessa_candidate_bridge_evidence.json",
            bridge_evidence,
            authority_class="TESSA_CANDIDATE_BRIDGE_EVIDENCE",
        ),
    ]
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "backend": "TESSA_PROPOSAL_V1",
            "effective_backend": "TESSA_PROPOSAL_V1",
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "proposal_geometry_hash": bridge_evidence.proposal_geometry_hash,
            "topology_sequence_hash": proposal.topology_sequence_hash,
            "material_support_field_hash": support_field.field_lineage_hash,
            "candidate_bridge_evidence_hash": bridge_evidence.evidence_hash,
            "decoded_component_count": len(component_labels),
            "mechanical_component_count": len(partition.components),
            "vertex_count": len(candidate.vertices),
            "face_count": len(candidate.faces),
            "mesh_policy_hash": mesh_policy.qualification_policy_lineage_hash,
            "static_preflight": quality,
            "stage19_remeasurement_required": True,
            "material_support_is_geometry_authority": False,
            "learned_xyz_preserved_exactly": True,
            "teacher_vertex_index_used": False,
            "product_authority_claimed": False,
            "motion_capability_claimed": False,
        },
    }
