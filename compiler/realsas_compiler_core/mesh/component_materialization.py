from __future__ import annotations

"""Compiler-owned per-component source-backed mesh materialization.

This module materializes each qualified mechanical component directly from current
Surface/Skeleton/Skin authority plus exact source observations.  It deliberately does
not consume a historical full-subject render mesh, source-component truth, owner
rasters, semantic filenames, or teacher topology.

The full observation alpha is a hard containment authority for every component mesh.
Coverage is judged on the UNION of all qualified component meshes, while each
individual component is qualified against its exact mechanical surface-id domain.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ..appearance import build_observed_appearance_binding
from ..hashing import content_sha256
from ..mechanical_component_partition import validate_mechanical_component_partition
from ..types import QualificationError
from .mwb2_cdt import (
    build_mwb2_observation_cdt_candidate,
    qualify_mwb2_component_observation_cdt_mesh,
)
from .mwb2_skin import bind_mwb2_mesh_skin
from .quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    coverage_gate_failures,
    mesh_raster_quality_report,
    raster_quality_gate_failures,
)


COMPONENT_MATERIALIZATION_SCHEMA = "RealSaS.MechanicalComponentObservationMaterialization.v1"


@dataclass(frozen=True)
class MaterializedMechanicalComponentIR:
    component_id: str
    mesh: Any
    mesh_skin: Any
    appearance: Any
    qualification_report: Mapping[str, Any]
    schema_version: str = COMPONENT_MATERIALIZATION_SCHEMA + ".Component"

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": str(self.component_id),
            "mesh": self.mesh.to_dict(),
            "mesh_skin": self.mesh_skin.to_dict(),
            "appearance": self.appearance.to_dict(),
            "qualification_report": dict(self.qualification_report),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class MechanicalComponentViewMaterializationIR:
    view_index: int
    component_partition_hash: str
    components: tuple[MaterializedMechanicalComponentIR, ...]
    union_coverage_report: Mapping[str, Any]
    qualification_report: Mapping[str, Any]
    materialization_hash: str
    schema_version: str = COMPONENT_MATERIALIZATION_SCHEMA + ".View"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_index": int(self.view_index),
            "component_partition_hash": str(self.component_partition_hash),
            "components": [row.to_dict() for row in self.components],
            "union_coverage_report": dict(self.union_coverage_report),
            "qualification_report": dict(self.qualification_report),
            "materialization_hash": str(self.materialization_hash),
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }


def mechanical_component_view_materialization_hash(
    value: MechanicalComponentViewMaterializationIR,
) -> str:
    payload = value.to_dict()
    payload.pop("materialization_hash", None)
    return content_sha256(payload)


def _raster_triangles(mesh) -> tuple[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
]:
    by_id: dict[str, tuple[float, float]] = {}
    for vertex in mesh.vertices:
        metadata = dict(getattr(vertex, "metadata", {}) or {})
        xy = metadata.get("raster_xy")
        if xy is None or len(tuple(xy)) != 2:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_MATERIALIZATION_RASTER_BINDING_MISSING:"
                f"{vertex.canonical_mesh_vertex_id}"
            )
        by_id[str(vertex.canonical_mesh_vertex_id)] = tuple(map(float, xy))
    out = []
    for face in mesh.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(vertex_id not in by_id for vertex_id in ids):
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_FACE_BINDING_INVALID"
            )
        out.append((by_id[ids[0]], by_id[ids[1]], by_id[ids[2]]))
    return tuple(out)


def _verify_rigid_skin(mesh_skin, *, component_id: str, parent_joint_id: str) -> None:
    parent = str(parent_joint_id)
    if not parent:
        raise QualificationError(
            f"MECHANICAL_COMPONENT_MATERIALIZATION_RIGID_PARENT_MISSING:{component_id}"
        )
    for row in mesh_skin.rows:
        influences = {str(jid): float(weight) for jid, weight in row.influences}
        owner = float(influences.get(parent, 0.0))
        other = float(sum(weight for jid, weight in influences.items() if jid != parent))
        if owner < 0.999 - 1.0e-12 or other > 0.001 + 1.0e-12:
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_RIGID_SKIN_DRIFT:"
                f"{component_id}:{row.canonical_mesh_vertex_id}:"
                f"owner={owner}:other={other}"
            )


def materialize_mechanical_component_view(
    *,
    surface,
    skeleton,
    skin,
    mechanical,
    partition,
    view_index: int,
    camera_binding_hash: str,
    observation_domain,
    observation_hash_by_view: Mapping[int, str],
    atlas_payload_hash: str,
    require_product_mesh_quality: bool = True,
) -> MechanicalComponentViewMaterializationIR:
    """Build one view as disjoint mechanical domains directly from current S/G/W.

    Every emitted mesh vertex is an admitted current Surface node in exactly one
    compiler-qualified mechanical component.  Cross-component triangles are therefore
    impossible by construction.  No historical full-subject mesh is subset, clipped,
    or reinterpreted.
    """

    view_index = int(view_index)
    validate_mechanical_component_partition(
        partition,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    observation_domain.validate()
    if int(observation_domain.view_index) != view_index:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_OBSERVATION_VIEW_DRIFT"
        )
    if not camera_binding_hash:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_CAMERA_HASH_REQUIRED"
        )
    if view_index not in observation_hash_by_view:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_OBSERVATION_HASH_MISSING"
        )

    rows: list[MaterializedMechanicalComponentIR] = []
    union_triangles = []
    component_reports: dict[str, Any] = {}

    for component_id in sorted(partition.component_surface_ids):
        allowed_surface_ids = tuple(partition.component_surface_ids[component_id])
        if not allowed_surface_ids:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_MATERIALIZATION_EMPTY_DOMAIN:{component_id}"
            )

        candidate = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=view_index,
            camera_binding_hash=str(camera_binding_hash),
            observation_domain=observation_domain,
            allowed_surface_ids=allowed_surface_ids,
            candidate_namespace=str(component_id),
        )
        mesh = qualify_mwb2_component_observation_cdt_mesh(
            surface,
            candidate,
            allowed_surface_ids=allowed_surface_ids,
            min_precision_inside_alpha=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.min_precision_inside_alpha,
        )
        raster_report = mesh_raster_quality_report(
            mesh,
            surface=surface,
            view_index=view_index,
        )
        raster_failures = raster_quality_gate_failures(
            raster_report,
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        if require_product_mesh_quality and raster_failures:
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_RASTER_QUALITY_FAIL:"
                f"V{view_index}:{component_id}:{','.join(raster_failures)}"
            )

        mesh_skin = bind_mwb2_mesh_skin(
            surface,
            skeleton,
            skin,
            mesh,
        )
        mechanical_class = str(
            partition.component_mechanical_classes[component_id]
        )
        parent_joint_id = str(
            partition.component_parent_joint_ids[component_id]
        )
        if mechanical_class == "RIGID_SKINNED_COMPONENT":
            _verify_rigid_skin(
                mesh_skin,
                component_id=str(component_id),
                parent_joint_id=parent_joint_id,
            )

        appearance = build_observed_appearance_binding(
            surface=surface,
            mesh=mesh,
            target_view_index=view_index,
            camera_binding_hash=str(camera_binding_hash),
            observation_hash_by_view={
                int(k): str(v) for k, v in observation_hash_by_view.items()
            },
            atlas_payload_hash=str(atlas_payload_hash),
        )
        triangles = _raster_triangles(mesh)
        union_triangles.extend(triangles)
        report = {
            "passed": True,
            "component_id": str(component_id),
            "mechanical_class": mechanical_class,
            "parent_joint_id": parent_joint_id,
            "allowed_surface_count": len(allowed_surface_ids),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "precision_inside_alpha": float(
                candidate.residual_report.get("precision_inside_alpha", -1.0)
            ),
            "source_alpha_recall_diagnostic_only": float(
                candidate.residual_report.get("source_alpha_recall", -1.0)
            ),
            "raster_quality": raster_report,
            "teacher_truth_used": False,
            "source_mesh_used": False,
            "cross_component_faces_possible": False,
        }
        component_reports[str(component_id)] = report
        rows.append(
            MaterializedMechanicalComponentIR(
                component_id=str(component_id),
                mesh=mesh,
                mesh_skin=mesh_skin,
                appearance=appearance,
                qualification_report=report,
            )
        )

    expected_components = set(map(str, partition.component_surface_ids))
    actual_components = {row.component_id for row in rows}
    if actual_components != expected_components:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_COMPONENT_ACCOUNTING_DRIFT"
        )

    union_coverage = observation_domain.coverage(union_triangles)
    coverage_failures = coverage_gate_failures(
        union_coverage,
        policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    )
    if require_product_mesh_quality and coverage_failures:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_UNION_COVERAGE_FAIL:"
            f"V{view_index}:{','.join(coverage_failures)}"
        )

    provisional = MechanicalComponentViewMaterializationIR(
        view_index=view_index,
        component_partition_hash=str(partition.partition_hash),
        components=tuple(rows),
        union_coverage_report=union_coverage,
        qualification_report={
            "passed": True,
            "component_count": len(rows),
            "component_reports": component_reports,
            "coverage_failures": tuple(coverage_failures),
            "product_mesh_quality_required": bool(require_product_mesh_quality),
            "teacher_truth_used": False,
            "source_component_truth_used": False,
            "historical_full_subject_mesh_used": False,
            "source_owner_raster_used": False,
            "cross_component_faces_generated": False,
        },
        materialization_hash="",
        metadata={
            "authority": "CURRENT_QUALIFIED_S_G_W_PLUS_EXACT_SOURCE_OBSERVATION",
            "mesh_order": "MECHANICAL_PARTITION_FIRST_THEN_COMPONENT_LOCAL_CDT",
            "observation_alpha_role": "HARD_CONTAINMENT_AND_UNION_COVERAGE_AUTHORITY",
            "component_identity_authority": "COMPILER_MECHANICAL_PARTITION",
        },
    )
    value = replace(
        provisional,
        materialization_hash=mechanical_component_view_materialization_hash(provisional),
    )
    if value.materialization_hash != mechanical_component_view_materialization_hash(value):
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_HASH_MISMATCH"
        )
    return value


__all__ = [
    "COMPONENT_MATERIALIZATION_SCHEMA",
    "MaterializedMechanicalComponentIR",
    "MechanicalComponentViewMaterializationIR",
    "mechanical_component_view_materialization_hash",
    "materialize_mechanical_component_view",
]
