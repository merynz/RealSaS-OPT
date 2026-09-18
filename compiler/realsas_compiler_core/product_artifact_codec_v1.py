from __future__ import annotations

"""Typed JSON codec for current product-authority IRs.

This module is intentionally boring. It does not qualify, repair or infer anything.
It only reconstructs exact current dataclasses from explicit schema-tagged JSON so
orchestrator stages can exchange sealed artifacts without treating arbitrary dicts as
authority.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .playback_full_surface_v3 import CameraProjectionV3
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,
    ComponentBoundaryConstraintIR,
    ComponentCarrierDecisionIR,
    ComponentCarrierPolicyIR,
    ComponentRegionIR,
    DeformationCapabilityEnvelopeIR,
    GeometricRefinementIR,
    JointCapabilityRangeIR,
    MechanicalPartitionIR,
    MeshQualificationPolicyIR,
    QualifiedMeshIR,
    QualifiedMeshVertexIR,
)
from .types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
    SurfaceSupportBinding,
)

Json = dict[str, Any]


def _schema(payload: Json, expected: str) -> None:
    actual = str(payload.get("schema_version") or payload.get("schema") or "")
    if actual != expected:
        raise ValueError(f"PRODUCT_ARTIFACT_SCHEMA_MISMATCH:{actual}!={expected}")


def _tuple2(rows):
    return tuple((str(a), float(b)) for a, b in rows)


def _support(payload: Json) -> SurfaceSupportBinding:
    _schema(payload, "RealSaS.SurfaceSupportBinding.v1")
    return SurfaceSupportBinding(
        mode=str(payload["mode"]),
        coefficients=_tuple2(payload.get("coefficients") or ()),
        metadata=dict(payload.get("metadata") or {}),
        schema_version=str(payload.get("schema_version") or "RealSaS.SurfaceSupportBinding.v1"),
    )


def _refinement(payload: Json | None):
    if payload is None:
        return None
    return GeometricRefinementIR(
        dense_lineage_hash=str(payload["dense_lineage_hash"]),
        base_position=tuple(map(float, payload["base_position"])),
        refined_position=tuple(map(float, payload["refined_position"])),
        local_scale=float(payload["local_scale"]),
        normal_component=float(payload["normal_component"]),
        tangential_component=float(payload["tangential_component"]),
        method=str(payload["method"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def rigging_surface_from_dict(payload: Json) -> RiggingSurfaceIR:
    _schema(payload, "RealSaS.RiggingSurfaceIR.v1")
    nodes = tuple(
        SurfaceNode(
            surface_id=str(row["surface_id"]),
            P=tuple(map(float, row["P"])),
            support_views=tuple(map(int, row.get("support_views") or ())),
            provenance_refs=tuple(map(str, row.get("provenance_refs") or ())),
            source_observation_ids=tuple(map(str, row.get("source_observation_ids") or ())),
            raster_bindings=tuple(
                (int(vi), tuple(map(float, xy)))
                for vi, xy in (row.get("raster_bindings") or ())
            ),
            persistence_group_id=None if row.get("persistence_group_id") is None else str(row["persistence_group_id"]),
            derived_normal=None if row.get("derived_normal") is None else tuple(map(float, row["derived_normal"])),
            validity_flags=tuple(map(str, row.get("validity_flags") or ())),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in (payload.get("surface_nodes") or ())
    )
    relations = tuple(
        SurfaceRelation(
            relation_id=str(row["relation_id"]),
            a_surface_id=str(row["a_surface_id"]),
            b_surface_id=str(row["b_surface_id"]),
            relation_kind=str(row["relation_kind"]),
            score=float(row.get("score", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in (payload.get("local_relations") or ())
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash=str(payload.get("geometry_lineage_hash") or ""),
        builder_id=str(payload.get("builder_id") or "RealSaS.GeometricSubstrateAssembler.current"),
        schema_version=str(payload.get("schema_version") or "RealSaS.RiggingSurfaceIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def qualified_skeleton_from_dict(payload: Json) -> QualifiedSkeletonIR:
    _schema(payload, "RealSaS.QualifiedSkeletonIR.v1")
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=str(row["canonical_joint_id"]),
            position=tuple(map(float, row["position"])),
            parent_canonical_id=None if row.get("parent_canonical_id") is None else str(row["parent_canonical_id"]),
            support_surface_ids=tuple(map(str, row.get("support_surface_ids") or ())),
            source_proposal_id=str(row.get("source_proposal_id") or ""),
        )
        for row in (payload.get("joints") or ())
    )
    return QualifiedSkeletonIR(
        joints=joints,
        root_id=str(payload["root_id"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        skeleton_lineage_hash=str(payload["skeleton_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedSkeletonIR.v1"),
    )


def qualified_skin_from_dict(payload: Json) -> QualifiedSkinIR:
    _schema(payload, "RealSaS.QualifiedSkinIR.v1")
    return QualifiedSkinIR(
        rows=tuple(
            QualifiedSkinRow(
                surface_id=str(row["surface_id"]),
                influences=tuple((str(jid), float(weight)) for jid, weight in (row.get("influences") or ())),
                simplex_residual_before=float(row.get("simplex_residual_before", 0.0)),
                correction_l1=float(row.get("correction_l1", 0.0)),
            )
            for row in (payload.get("rows") or ())
        ),
        surface_binding_hash=str(payload["surface_binding_hash"]),
        skeleton_binding_hash=str(payload["skeleton_binding_hash"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedSkinIR.v1"),
    )


def mechanical_partition_from_dict(payload: Json) -> MechanicalPartitionIR:
    _schema(payload, "RealSaS.MechanicalPartitionIR.v1")
    return MechanicalPartitionIR(
        components=tuple(
            ComponentRegionIR(
                component_id=str(row["component_id"]),
                surface_ids=tuple(map(str, row.get("surface_ids") or ())),
                evidence_refs=tuple(map(str, row.get("evidence_refs") or ())),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in (payload.get("components") or ())
        ),
        boundary_constraints=tuple(
            ComponentBoundaryConstraintIR(
                constraint_id=str(row["constraint_id"]),
                a_surface_id=str(row["a_surface_id"]),
                b_surface_id=str(row["b_surface_id"]),
                decision=str(row["decision"]),
                evidence_refs=tuple(map(str, row.get("evidence_refs") or ())),
                confidence=float(row.get("confidence", 1.0)),
                metadata=dict(row.get("metadata") or {}),
                schema_version=str(row.get("schema_version") or "RealSaS.ComponentBoundaryConstraintIR.v1"),
            )
            for row in (payload.get("boundary_constraints") or ())
        ),
        surface_lineage_hash=str(payload["surface_lineage_hash"]),
        partition_lineage_hash=str(payload["partition_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.MechanicalPartitionIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def component_carrier_policy_from_dict(payload: Json) -> ComponentCarrierPolicyIR:
    _schema(payload, "RealSaS.ComponentCarrierPolicyIR.v1")
    return ComponentCarrierPolicyIR(
        partition_binding_hash=str(payload["partition_binding_hash"]),
        decisions=tuple(
            ComponentCarrierDecisionIR(
                component_id=str(row["component_id"]),
                carrier_class=str(row["carrier_class"]),
                evidence_refs=tuple(map(str, row.get("evidence_refs") or ())),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in (payload.get("decisions") or ())
        ),
        carrier_policy_lineage_hash=str(payload["carrier_policy_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.ComponentCarrierPolicyIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def deformation_envelope_from_dict(payload: Json) -> DeformationCapabilityEnvelopeIR:
    _schema(payload, "RealSaS.DeformationCapabilityEnvelopeIR.v1")
    return DeformationCapabilityEnvelopeIR(
        skeleton_lineage_hash=str(payload["skeleton_lineage_hash"]),
        joint_ranges=tuple(
            JointCapabilityRangeIR(
                canonical_joint_id=str(row["canonical_joint_id"]),
                min_rotation_deg=float(row["min_rotation_deg"]),
                max_rotation_deg=float(row["max_rotation_deg"]),
                translation_radius=float(row.get("translation_radius", 0.0)),
                min_scale=float(row.get("min_scale", 1.0)),
                max_scale=float(row.get("max_scale", 1.0)),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in (payload.get("joint_ranges") or ())
        ),
        camera_binding_hashes=tuple(map(str, payload.get("camera_binding_hashes") or ())),
        allowed_attachment_state_hashes=tuple(map(str, payload.get("allowed_attachment_state_hashes") or ())),
        axis_contract_hash=str(payload["axis_contract_hash"]),
        probe_plan_hash=str(payload["probe_plan_hash"]),
        envelope_lineage_hash=str(payload["envelope_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.DeformationCapabilityEnvelopeIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def mesh_policy_from_dict(payload: Json) -> MeshQualificationPolicyIR:
    _schema(payload, "RealSaS.MeshQualificationPolicyIR.v1")
    return MeshQualificationPolicyIR(
        g1_max_normal_refinement_ratio=float(payload["g1_max_normal_refinement_ratio"]),
        g1_max_tangential_to_normal_ratio=float(payload["g1_max_tangential_to_normal_ratio"]),
        g3_min_angle_deg=float(payload["g3_min_angle_deg"]),
        g3_max_aspect_longest_over_min_altitude=float(payload["g3_max_aspect_longest_over_min_altitude"]),
        coverage_thresholds=tuple(
            CarrierCoverageThresholdIR(
                carrier_class=str(row["carrier_class"]),
                min_recall=float(row["min_recall"]),
                min_precision=float(row["min_precision"]),
                max_largest_coherent_hole_fraction=float(row["max_largest_coherent_hole_fraction"]),
                max_interior_uncovered_fraction=float(row["max_interior_uncovered_fraction"]),
            )
            for row in (payload.get("coverage_thresholds") or ())
        ),
        qualification_policy_lineage_hash=str(payload["qualification_policy_lineage_hash"]),
        g3_min_dynamic_area_ratio=float(payload.get("g3_min_dynamic_area_ratio", 0.05)),
        g3_max_dynamic_area_ratio=float(payload.get("g3_max_dynamic_area_ratio", 20.0)),
        g3_max_dynamic_condition_number=float(payload.get("g3_max_dynamic_condition_number", 16.0)),
        schema_version=str(payload.get("schema_version") or "RealSaS.MeshQualificationPolicyIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def canonical_mesh_candidate_from_dict(payload: Json) -> CanonicalMeshCandidateIR:
    _schema(payload, "RealSaS.CanonicalMeshCandidateIR.v1")
    return CanonicalMeshCandidateIR(
        vertices=tuple(
            CanonicalMeshVertexCandidateIR(
                candidate_vertex_id=str(row["candidate_vertex_id"]),
                support_binding=_support(dict(row["support_binding"])),
                component_id=str(row["component_id"]),
                P=tuple(map(float, row["P"])),
                refinement=_refinement(row.get("refinement")),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in (payload.get("vertices") or ())
        ),
        faces=tuple(tuple(map(str, face)) for face in (payload.get("faces") or ())),
        edges=tuple(tuple(map(str, edge)) for edge in (payload.get("edges") or ())),
        surface_binding_hash=str(payload["surface_binding_hash"]),
        partition_binding_hash=str(payload["partition_binding_hash"]),
        carrier_policy_binding_hash=str(payload["carrier_policy_binding_hash"]),
        producer_id=str(payload["producer_id"]),
        producer_policy_hash=str(payload["producer_policy_hash"]),
        candidate_lineage_hash=str(payload["candidate_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CanonicalMeshCandidateIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def qualified_mesh_from_dict(payload: Json) -> QualifiedMeshIR:
    _schema(payload, "RealSaS.QualifiedMeshIR.v1")
    return QualifiedMeshIR(
        vertices=tuple(
            QualifiedMeshVertexIR(
                canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
                support_binding=_support(dict(row["support_binding"])),
                component_id=str(row["component_id"]),
                P=tuple(map(float, row["P"])),
                source_candidate_vertex_id=str(row["source_candidate_vertex_id"]),
                refinement=_refinement(row.get("refinement")),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in (payload.get("vertices") or ())
        ),
        faces=tuple(tuple(map(str, face)) for face in (payload.get("faces") or ())),
        edges=tuple(tuple(map(str, edge)) for edge in (payload.get("edges") or ())),
        surface_binding_hash=str(payload["surface_binding_hash"]),
        partition_binding_hash=str(payload["partition_binding_hash"]),
        carrier_policy_binding_hash=str(payload["carrier_policy_binding_hash"]),
        envelope_binding_hash=str(payload["envelope_binding_hash"]),
        qualification_policy_hash=str(payload["qualification_policy_hash"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedMeshIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def qualified_mesh_skin_from_dict(payload: Json) -> QualifiedMeshSkinIR:
    _schema(payload, "RealSaS.QualifiedMeshSkinIR.v1")
    return QualifiedMeshSkinIR(
        rows=tuple(
            QualifiedMeshSkinRow(
                canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
                influences=tuple((str(jid),float(weight)) for jid,weight in (row.get("influences") or ())),
                source_support_coefficients=tuple((str(sid),float(weight)) for sid,weight in (row.get("source_support_coefficients") or ())),
                simplex_residual_before=float(row.get("simplex_residual_before",0.0)),
                correction_l1=float(row.get("correction_l1",0.0)),
            )
            for row in (payload.get("rows") or ())
        ),
        surface_binding_hash=str(payload["surface_binding_hash"]),
        skeleton_binding_hash=str(payload["skeleton_binding_hash"]),
        skin_binding_hash=str(payload["skin_binding_hash"]),
        mesh_binding_hash=str(payload["mesh_binding_hash"]),
        transfer_method=str(payload["transfer_method"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        mesh_skin_lineage_hash=str(payload["mesh_skin_lineage_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedMeshSkinIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def camera_projection_from_dict(payload: Json) -> CameraProjectionV3:
    _schema(payload, "RealSaS.FullSurfaceCameraProjection.v3")
    return CameraProjectionV3(
        view_id=str(payload["view_id"]),
        view_index=int(payload["view_index"]),
        origin=tuple(map(float, payload["origin"])),
        right=tuple(map(float, payload["right"])),
        screen_up=tuple(map(float, payload["screen_up"])),
        forward=tuple(map(float, payload["forward"])),
        half_extent=float(payload["half_extent"]),
        resolution=int(payload["resolution"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.FullSurfaceCameraProjection.v3"),
    )


def read_json(path: str | Path) -> Json:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_ir_json(path: str | Path, value) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = value.to_dict() if hasattr(value, "to_dict") else asdict(value)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return p
