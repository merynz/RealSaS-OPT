from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
    write_ir_json,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import (
    CarrierSkinInfluenceProposal,
    CarrierSkinProposalIR,
    qualify_carrier_skin_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    canonical_mesh_candidate_lineage_hash,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    build_surface_addressing,
    static_mesh_qualification_hash,
)
from compiler.realsas_compiler_core.types import SurfaceSupportBinding

EXPECTED = {
    "surface": "7b71969e12316503dd1cd540b3fa2a17e5cbc96a9b04c03aed88d69077f79182",
    "candidate": "1fa2c1d67314124d4c22af55e57cc8eb9b5e65bf1ff65870acefc7ea1d104bd1",
    "m_support": "8a625d2e5373cc679dbbaa460c8045956a520b65799bca15cb6975a50e75f514",
    "skeleton": "c5a3c7f5675ba84233c8d7eb4ccab076e69c105462f86673dc259e30e2ca6ca5",
    "weights": "880dd73b2f3a21196c229156a3271fa95e7261def3f50a3dfe9f1c49ca2bb27b",
    "carrier_basis": "dbac301b9c47fa7f31a024ef0612008d8315b590d7a2dd1552a75f75c7ac9ff2",
    "surface_lineage": "452cf6564a2cde35a4d6d420021491d231f90cec267df9f41d4424dc8d74895a",
    "partition_lineage": "ba14cb67ff5bdf78a609e56ccddcfd8203075ffe304734df3be8fcf657635dec",
    "skeleton_lineage": "526d314e5cf9166702b511bf76e545b3632207138162d65992b4a0109805a93e",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"PINNED_SHA_DRIFT::{path.name}::{actual}::{expected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()
    inp = args.input_root.resolve()
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    surface_path = inp / "qualified_rigging_surface.json"
    candidate_path = inp / "TESSA_STATIC_G3_REPAIRED_CANDIDATE_V3.json"
    support_path = inp / "M_EVIDENCE_SUPPORT_V55.npz"
    skeleton_path = inp / "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_QUALIFIED_SKELETON_V541.json"
    weights_path = inp / "MIRA_AXIS41_TESSA_M_SKIN_FIELD_V55.npz"
    for path, key in (
        (surface_path, "surface"), (candidate_path, "candidate"),
        (support_path, "m_support"), (skeleton_path, "skeleton"),
        (weights_path, "weights"),
    ):
        require_sha(path, EXPECTED[key])

    surface = rigging_surface_from_dict(json.loads(surface_path.read_text(encoding="utf-8")))
    if surface.geometry_lineage_hash != EXPECTED["surface_lineage"]:
        raise RuntimeError("SURFACE_LINEAGE_DRIFT")

    partition = build_structural_partition(surface, boundary_overrides=())
    if partition.partition_lineage_hash != EXPECTED["partition_lineage"]:
        raise RuntimeError(
            f"PARTITION_LINEAGE_DRIFT::{partition.partition_lineage_hash}::{EXPECTED['partition_lineage']}"
        )
    decisions = tuple(
        ComponentCarrierDecisionIR(
            component.component_id,
            "MESH",
            ("AUTOMATIC_CONSERVATIVE_MESH_CARRIER_V1",),
            metadata={
                "automatic": True,
                "semantic_recognition_used": False,
                "planar_optimization_deferred": True,
            },
        )
        for component in partition.components
    )
    carrier_policy = build_component_carrier_policy(
        partition=partition,
        decisions=decisions,
        metadata={
            "default_carrier": "MESH",
            "automatic": True,
            "manual_carrier_authoring_used": False,
            "clip_is_presentation_only": True,
        },
    )

    source_candidate = canonical_mesh_candidate_from_dict(
        json.loads(candidate_path.read_text(encoding="utf-8"))
    )
    with np.load(support_path, allow_pickle=False) as z:
        indptr = np.asarray(z["indptr"], dtype=np.int64)
        indices = np.asarray(z["indices"], dtype=np.int64)
        coeff = np.asarray(z["coefficients"], dtype=np.float64)
        evidence_positions = np.asarray(z["candidate_vertices_world"], dtype=np.float64)
        evidence_faces = np.asarray(z["candidate_faces"], dtype=np.int64)
        surface_ids = np.asarray(z["gsa_surface_ids"]).astype(str)
    if len(source_candidate.vertices) != 3653 or len(source_candidate.faces) != 6928:
        raise RuntimeError("CARRIER_CARDINALITY_DRIFT")
    positions = np.asarray([v.P for v in source_candidate.vertices], dtype=np.float64)
    id_to_index = {str(v.candidate_vertex_id): i for i, v in enumerate(source_candidate.vertices)}
    faces = np.asarray(
        [[id_to_index[str(x)] for x in face] for face in source_candidate.faces],
        dtype=np.int64,
    )
    if not np.array_equal(positions, evidence_positions):
        raise RuntimeError("M_EVIDENCE_POSITION_DRIFT")
    if not np.array_equal(faces, evidence_faces):
        raise RuntimeError("M_EVIDENCE_TOPOLOGY_DRIFT")
    if indptr.shape != (3654,) or int(indptr[-1]) != len(indices) or len(indices) != len(coeff):
        raise RuntimeError("M_EVIDENCE_CSR_SHAPE_DRIFT")

    owner = {
        str(sid): str(component.component_id)
        for component in partition.components
        for sid in component.surface_ids
    }
    rebound_vertices = []
    max_simplex_residual = 0.0
    cross_component_rows = 0
    for i, vertex in enumerate(source_candidate.vertices):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        rows = []
        owners = set()
        for j in range(lo, hi):
            sid = str(surface_ids[int(indices[j])])
            weight = float(coeff[j])
            if sid not in owner:
                raise RuntimeError(f"M_EVIDENCE_UNKNOWN_SURFACE::{sid}")
            if weight < 0.0 or not np.isfinite(weight):
                raise RuntimeError("M_EVIDENCE_WEIGHT_INVALID")
            if weight > 0.0:
                owners.add(owner[sid])
            rows.append((sid, weight))
        residual = abs(sum(w for _, w in rows) - 1.0)
        max_simplex_residual = max(max_simplex_residual, residual)
        if residual > 1e-6:
            raise RuntimeError(f"M_EVIDENCE_SIMPLEX_DRIFT::{i}::{residual}")
        if len(owners) != 1:
            cross_component_rows += 1
            raise RuntimeError(f"M_EVIDENCE_CROSS_COMPONENT::{i}::{sorted(owners)}")
        component_id = next(iter(owners))
        rebound_vertices.append(
            replace(
                vertex,
                support_binding=SurfaceSupportBinding(
                    "LOCAL_CONVEX_INTERPOLATION",
                    tuple(rows),
                    metadata={
                        "authority_class": "MATERIAL_SUPPORT_ONLY",
                        "geometry_position_derived_from_support": False,
                        "teacher_vertex_index_used": False,
                        "source": "M_EVIDENCE_SUPPORT_V55",
                    },
                ),
                component_id=component_id,
                metadata={
                    **dict(vertex.metadata or {}),
                    "material_support_is_not_geometry_support": True,
                    "geometry_position_derived_from_material_support": False,
                    "teacher_vertex_index_used": False,
                    "support_rebound_from_m_evidence_v55": True,
                },
            )
        )

    producer_policy_hash = content_sha256({
        "schema": "RealSaS.KnightLatestCarrierSupportRebindPolicy.v1",
        "source_candidate_sha256": EXPECTED["candidate"],
        "support_sha256": EXPECTED["m_support"],
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "partition_lineage_hash": partition.partition_lineage_hash,
        "xyz_topology_mutation": False,
        "product_authority_claimed": False,
    })
    rebound = replace(
        source_candidate,
        vertices=tuple(rebound_vertices),
        surface_binding_hash=surface.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        producer_id="RealSaS.TESSACarrierSupportRebindMaterialization.v1",
        producer_policy_hash=producer_policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(source_candidate.metadata or {}),
            "research_only_materialization": True,
            "product_authority_claimed": False,
            "source_candidate_sha256": EXPECTED["candidate"],
            "m_evidence_support_sha256": EXPECTED["m_support"],
            "xyz_topology_mutation": False,
        },
    )
    rebound = replace(rebound, candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(rebound))
    validate_canonical_mesh_candidate(
        rebound,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    rebound_positions = np.asarray([v.P for v in rebound.vertices], dtype=np.float64)
    rebound_id_to_index = {str(v.candidate_vertex_id): i for i, v in enumerate(rebound.vertices)}
    rebound_faces = np.asarray(
        [[rebound_id_to_index[str(x)] for x in face] for face in rebound.faces],
        dtype=np.int64,
    )
    if not np.array_equal(rebound_positions, evidence_positions) or not np.array_equal(rebound_faces, evidence_faces):
        raise RuntimeError("REBIND_CHANGED_XYZ_OR_TOPOLOGY")

    addressing = build_surface_addressing(rebound)
    static = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=rebound.candidate_lineage_hash,
        surface_addressing_binding_hash=addressing.addressing_hash,
        appearance_domain_binding_hash=content_sha256({"research_only": True, "candidate": rebound.candidate_lineage_hash}),
        geometry_gate_binding_hash=content_sha256({"sealed_tessa_static_g3": EXPECTED["candidate"]}),
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report={
            "status": "RESEARCH_ONLY_MATERIALIZATION_FROM_SEALED_STATIC_TESSA_V3",
            "product_authority_claimed": False,
            "source_candidate_sha256": EXPECTED["candidate"],
            "xyz_topology_mutation": False,
            "current_stage19_full_source_fidelity_requalification_performed": False,
        },
        qualification_hash="",
        metadata={"research_only_materialization": True},
    )
    static = replace(static, qualification_hash=static_mesh_qualification_hash(static))
    mechanical = build_mechanical_carrier_evidence_v1(
        rebound,
        static_qualification=static,
        surface_addressing=addressing,
    )

    skeleton = qualified_skeleton_from_dict(json.loads(skeleton_path.read_text(encoding="utf-8")))
    if skeleton.skeleton_lineage_hash != EXPECTED["skeleton_lineage"]:
        raise RuntimeError("SKELETON_LINEAGE_DRIFT")
    with np.load(weights_path, allow_pickle=False) as z:
        w_positions = np.asarray(z["vertices_rest_source_frame"], dtype=np.float64)
        w_faces = np.asarray(z["faces"], dtype=np.int64)
        W = np.asarray(z["weights_axis41_canonical"], dtype=np.float64)
        joint_ids = tuple(map(str, np.asarray(z["canonical_joint_ids"]).tolist()))
        carrier_basis = str(np.asarray(z["carrier_basis_sha256"]).item())
    if carrier_basis != EXPECTED["carrier_basis"]:
        raise RuntimeError("MIRA_CARRIER_BASIS_DRIFT")
    if not np.array_equal(w_positions, evidence_positions) or not np.array_equal(w_faces, evidence_faces):
        raise RuntimeError("MIRA_W_M_CARRIER_ARRAY_DRIFT")
    if W.shape != (3653, 41) or joint_ids != tuple(map(str, joint_ids)):
        raise RuntimeError("MIRA_W_M_SHAPE_DRIFT")
    if set(joint_ids) != {str(j.canonical_joint_id) for j in skeleton.joints}:
        raise RuntimeError("MIRA_W_M_JOINT_SET_DRIFT")
    if np.min(W) < 0.0 or np.max(np.abs(W.sum(axis=1) - 1.0)) > 1e-12:
        raise RuntimeError("MIRA_W_M_SIMPLEX_DRIFT")

    carrier_vertex_ids = tuple(map(str, mechanical.ordered_vertex_ids))
    if carrier_vertex_ids != tuple(str(v.candidate_vertex_id) for v in rebound.vertices):
        raise RuntimeError("CARRIER_VERTEX_ORDER_DRIFT")
    influences = tuple(
        CarrierSkinInfluenceProposal(carrier_vertex_ids[i], joint_ids[j], float(W[i, j]))
        for i in range(W.shape[0])
        for j in range(W.shape[1])
        if float(W[i, j]) > 0.0
    )
    proposal = CarrierSkinProposalIR(
        influences=influences,
        carrier_evidence_hash=mechanical.carrier_evidence_hash,
        carrier_topology_hash=mechanical.topology_hash,
        carrier_geometry_hash=mechanical.geometry_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        model_provenance="MIRA_M_DIRECT_V55__PINNED_TERMINAL_PASS",
        metadata={
            "output_domain": "MECHANICAL_CARRIER_M",
            "semantic_skin_transfer_performed": False,
            "gsa_skin_field_authority_minted": False,
            "direct_simplex_readout": True,
            "source_skin_runtime_authority": False,
            "mesh_mutation_invalidates_prediction": True,
            "research_only_materialization": True,
            "pinned_w_m_sha256": EXPECTED["weights"],
            "pinned_carrier_basis_sha256": EXPECTED["carrier_basis"],
        },
    )
    skin = qualify_carrier_skin_v1(
        mechanical,
        skeleton,
        proposal,
        max_simplex_repair_l1=1e-12,
        max_total_correction_l1=1e-9,
        negative_tolerance=0.0,
        max_influences=41,
    )

    write_ir_json(out / "mechanical_partition.json", partition)
    write_ir_json(out / "component_carrier_policy.json", carrier_policy)
    write_ir_json(out / "rebound_candidate.json", rebound)
    write_ir_json(out / "surface_addressing.json", addressing)
    write_ir_json(out / "research_static_binding.json", static)
    write_ir_json(out / "mechanical_carrier_evidence.json", mechanical)
    write_ir_json(out / "qualified_carrier_skin.json", skin)
    receipt = {
        "schema": "RealSaS.KnightLatestCarrierMaterializationWitness.v1",
        "status": "PASS_RESEARCH_ONLY_M_G_W_M_MATERIALIZATION",
        "product_authority_claimed": False,
        "stage19_full_source_fidelity_requalification_performed": False,
        "pinned_inputs": EXPECTED,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "partition_lineage_hash": partition.partition_lineage_hash,
        "carrier_policy_lineage_hash": carrier_policy.carrier_policy_lineage_hash,
        "rebound_candidate_lineage_hash": rebound.candidate_lineage_hash,
        "mechanical_carrier_evidence_hash": mechanical.carrier_evidence_hash,
        "mechanical_carrier_topology_hash": mechanical.topology_hash,
        "mechanical_carrier_geometry_hash": mechanical.geometry_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "carrier_skin_lineage_hash": skin.skin_lineage_hash,
        "m_evidence_max_simplex_residual": max_simplex_residual,
        "m_evidence_cross_component_row_count": cross_component_rows,
        "mira_w_m_row_sum_max_abs_residual": float(np.max(np.abs(W.sum(axis=1) - 1.0))),
        "mira_w_m_min_weight": float(np.min(W)),
        "xyz_exact_preserved": True,
        "topology_exact_preserved": True,
        "carrier_basis_pinned": carrier_basis,
    }
    (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
