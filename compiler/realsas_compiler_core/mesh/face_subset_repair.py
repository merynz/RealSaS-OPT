from __future__ import annotations

"""Fail-closed exact face-subset repair for qualified external render meshes.

This operator exists for a narrow product repair class: remove a small set of raster
sliver faces that individually violate the already-frozen mesh-shape policy, while
preserving every vertex, edge record, skin row, support binding, camera binding and
artist-sourced appearance payload on retained faces. It does not move geometry,
retriangulate, synthesize weights, generate pixels, or change policy thresholds.

Coverage is deliberately not manufactured here. The caller must provide an exact
observation-domain coverage report plus its measurement witness hash; the repaired
mesh is then re-measured against the frozen product policy before qualification.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from ..hashing import content_sha256
from ..mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from ..product_external_render import validate_external_renderable_component
from ..types import QualificationError
from ..v4 import appearance_lineage_hash, build_appearance_binding
from ..v4_types import AppearanceCornerBinding, AppearanceBindingIR, RenderableComponentIR
from .quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    MeshQualityPolicyV1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
    raster_quality_gate_failures,
)

_ALLOWED_REMOVAL_FAILURES = {
    "min_raster_triangle_angle_deg",
    "max_raster_triangle_aspect_ratio",
}
_PLACEHOLDER_HASHES = {"", "latest", "current", "newest", "pending", "todo"}


def _required_hash(name: str, value: str) -> str:
    normalized = str(value or "").strip()
    if normalized.lower() in _PLACEHOLDER_HASHES:
        raise QualificationError(f"FACE_SUBSET_REPAIR_INVALID_{name}")
    return normalized


def _payload_hash(value) -> str:
    if hasattr(value, "to_dict"):
        return content_sha256(value.to_dict())
    return content_sha256(value)


def _rows_payload_hash(mesh_skin) -> str:
    return content_sha256({"rows": [row.to_dict() for row in mesh_skin.rows]})


def _vertices_payload_hash(mesh) -> str:
    return content_sha256({"vertices": [vertex.to_dict() for vertex in mesh.vertices]})


def _edges_payload_hash(mesh) -> str:
    return content_sha256({"edges": [tuple(edge) for edge in mesh.edges]})


def _retained_artist_payload_hash(appearance: AppearanceBindingIR, retained_old_face_indices: tuple[int, ...]) -> str:
    retained = set(retained_old_face_indices)
    rows = []
    for corner in appearance.corner_bindings:
        if int(corner.face_index) not in retained:
            continue
        payload = corner.to_dict()
        payload.pop("face_index", None)
        rows.append({
            "old_face_index": int(corner.face_index),
            "corner_index": int(corner.corner_index),
            "payload": payload,
        })
    return content_sha256({"retained_artist_corners": rows})


@dataclass(frozen=True)
class QualifiedFaceSubsetRepairIR:
    view_index: int
    source_component_state_hash: str
    source_external_qualification_hash: str
    source_mesh_lineage_hash: str
    repaired_mesh_lineage_hash: str
    source_mesh_skin_lineage_hash: str
    repaired_mesh_skin_lineage_hash: str
    source_appearance_lineage_hash: str
    repaired_appearance_lineage_hash: str
    removed_face_indices: tuple[int, ...]
    removed_face_keys_sha256: str
    source_face_count: int
    repaired_face_count: int
    vertices_payload_sha256: str
    edges_payload_sha256: str
    weight_rows_payload_sha256: str
    retained_artist_payload_sha256: str
    source_truth_ownership_sha256: str
    coverage_measurement_sha256: str
    frozen_policy_sha256: str
    qualification_report: dict[str, Any]
    repair_hash: str
    schema_version: str = "RealSaS.QualifiedFaceSubsetRepairIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedFaceSubsetRepairResult:
    mesh: Any
    mesh_skin: Any
    appearance: AppearanceBindingIR
    qualification: QualifiedFaceSubsetRepairIR


def face_subset_repair_hash(value: QualifiedFaceSubsetRepairIR) -> str:
    payload = value.to_dict()
    payload.pop("repair_hash", None)
    return content_sha256(payload)


def _single_face_report(mesh, face_index: int) -> dict[str, Any]:
    face = mesh.faces[int(face_index)]
    one_face = replace(mesh, faces=(face,))
    return mesh_raster_quality_report(one_face, surface=None, view_index=int(mesh.view_index))


def _validate_requested_removals(mesh, remove_face_indices: tuple[int, ...], policy: MeshQualityPolicyV1) -> tuple[dict[str, Any], ...]:
    if not remove_face_indices:
        raise QualificationError("FACE_SUBSET_REPAIR_REQUIRES_EXPLICIT_REMOVALS")
    indices = tuple(int(index) for index in remove_face_indices)
    if len(indices) != len(set(indices)):
        raise QualificationError("FACE_SUBSET_REPAIR_DUPLICATE_FACE_INDEX")
    if tuple(sorted(indices)) != indices:
        raise QualificationError("FACE_SUBSET_REPAIR_FACE_INDICES_MUST_BE_SORTED")
    if indices[0] < 0 or indices[-1] >= len(mesh.faces):
        raise QualificationError("FACE_SUBSET_REPAIR_FACE_INDEX_OUT_OF_RANGE")
    if len(indices) >= len(mesh.faces):
        raise QualificationError("FACE_SUBSET_REPAIR_CANNOT_REMOVE_ALL_FACES")

    reports = []
    for index in indices:
        report = _single_face_report(mesh, index)
        failures = tuple(raster_quality_gate_failures(report, policy=policy))
        if not failures:
            raise QualificationError(f"FACE_SUBSET_REPAIR_FACE_NOT_POLICY_VIOLATION:{index}")
        if not set(failures).issubset(_ALLOWED_REMOVAL_FAILURES):
            raise QualificationError(
                f"FACE_SUBSET_REPAIR_UNSUPPORTED_FACE_FAILURE:{index}:{sorted(failures)}"
            )
        reports.append({
            "face_index": index,
            "face": tuple(mesh.faces[index]),
            "failure_invariants": failures,
            "min_raster_triangle_angle_deg": float(report["min_raster_triangle_angle_deg"]),
            "max_raster_triangle_aspect_ratio": float(report["max_raster_triangle_aspect_ratio"]),
            "min_raster_triangle_area": float(report["min_raster_triangle_area"]),
        })
    return tuple(reports)


def _project_retained_appearance(
    source: AppearanceBindingIR,
    *,
    retained_old_face_indices: tuple[int, ...],
    repaired_mesh_lineage_hash: str,
) -> AppearanceBindingIR:
    source_by_key = {
        (int(corner.face_index), int(corner.corner_index)): corner
        for corner in source.corner_bindings
    }
    corners: list[AppearanceCornerBinding] = []
    for new_face_index, old_face_index in enumerate(retained_old_face_indices):
        old_corner_indices = sorted(
            corner_index
            for face_index, corner_index in source_by_key
            if face_index == old_face_index
        )
        if not old_corner_indices:
            raise QualificationError(
                f"FACE_SUBSET_REPAIR_RETAINED_FACE_MISSING_APPEARANCE:{old_face_index}"
            )
        for corner_index in old_corner_indices:
            source_corner = source_by_key[(old_face_index, corner_index)]
            corners.append(replace(source_corner, face_index=int(new_face_index)))

    value = build_appearance_binding(
        target_view_index=int(source.target_view_index),
        mesh_binding_hash=str(repaired_mesh_lineage_hash),
        camera_binding_hash=str(source.camera_binding_hash),
        corner_bindings=tuple(corners),
        atlas_payload_hash=str(source.atlas_payload_hash),
        metadata={
            **dict(source.metadata or {}),
            "exact_face_subset_projection": True,
            "new_pixels_generated": False,
            "artist_corner_payload_preserved": True,
        },
    )
    return value


def qualify_exact_policy_face_subset_repair(
    source_component: RenderableComponentIR,
    mechanical,
    *,
    remove_face_indices: tuple[int, ...],
    repaired_coverage_report: Mapping[str, Any],
    coverage_measurement_sha256: str,
    source_truth_ownership_sha256: str,
    policy: MeshQualityPolicyV1 = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    metadata: Mapping[str, Any] | None = None,
) -> QualifiedFaceSubsetRepairResult:
    """Qualify an exact external face subset that closes only frozen-policy slivers."""
    validate_external_renderable_component(source_component, mechanical)
    policy.validate()
    coverage_measurement_sha256 = _required_hash(
        "COVERAGE_MEASUREMENT_SHA256", coverage_measurement_sha256
    )
    source_truth_ownership_sha256 = _required_hash(
        "SOURCE_TRUTH_OWNERSHIP_SHA256", source_truth_ownership_sha256
    )
    supplied_metadata = dict(metadata or {})
    forbidden_true = (
        "vertices_mutated",
        "edges_mutated",
        "weights_mutated",
        "retriangulated",
        "new_pixels_generated",
        "policy_thresholds_changed",
        "scientific_mechanical_surface_relabelled",
    )
    for key in forbidden_true:
        if bool(supplied_metadata.get(key, False)):
            raise QualificationError(f"FACE_SUBSET_REPAIR_FORBIDDEN_MUTATION:{key}")

    mesh = source_component.mesh
    mesh_skin = source_component.mesh_skin
    appearance = source_component.appearance
    removal_reports = _validate_requested_removals(mesh, remove_face_indices, policy)
    removed = set(int(index) for index in remove_face_indices)
    retained_old_face_indices = tuple(index for index in range(len(mesh.faces)) if index not in removed)
    repaired_faces = tuple(mesh.faces[index] for index in retained_old_face_indices)

    vertices_hash = _vertices_payload_hash(mesh)
    edges_hash = _edges_payload_hash(mesh)
    rows_hash = _rows_payload_hash(mesh_skin)
    retained_artist_hash = _retained_artist_payload_hash(appearance, retained_old_face_indices)

    repaired_mesh = replace(
        mesh,
        faces=repaired_faces,
        qualification_report={
            **dict(mesh.qualification_report or {}),
            "status": "PASS_EXACT_FROZEN_POLICY_FACE_SUBSET_REPAIR",
            "source_mesh_lineage_hash": str(mesh.mesh_lineage_hash),
            "removed_face_indices": tuple(int(index) for index in remove_face_indices),
            "coverage_measurement_sha256": coverage_measurement_sha256,
            "source_truth_ownership_sha256": source_truth_ownership_sha256,
            "face_subset_only": True,
            "vertices_mutated": False,
            "edges_mutated": False,
            "retriangulated": False,
            "policy_thresholds_changed": False,
        },
        mesh_lineage_hash="",
        metadata={
            **dict(mesh.metadata or {}),
            "source_mesh_lineage_hash": str(mesh.mesh_lineage_hash),
            "exact_face_subset_repair": True,
            "source_truth_ownership_sha256": source_truth_ownership_sha256,
        },
    )
    repaired_mesh = replace(repaired_mesh, mesh_lineage_hash=mesh_lineage_hash(repaired_mesh))

    repaired_mesh_skin = replace(
        mesh_skin,
        mesh_binding_hash=str(repaired_mesh.mesh_lineage_hash),
        qualification_report={
            **dict(mesh_skin.qualification_report or {}),
            "source_mesh_skin_lineage_hash": str(mesh_skin.mesh_skin_lineage_hash),
            "exact_rows_preserved": True,
            "weights_mutated": False,
            "only_mesh_binding_hash_rebound": True,
        },
        mesh_skin_lineage_hash="",
        metadata={
            **dict(mesh_skin.metadata or {}),
            "source_mesh_skin_lineage_hash": str(mesh_skin.mesh_skin_lineage_hash),
            "exact_rows_preserved": True,
            "weights_mutated": False,
        },
    )
    repaired_mesh_skin = replace(
        repaired_mesh_skin,
        mesh_skin_lineage_hash=mesh_skin_lineage_hash(repaired_mesh_skin),
    )

    repaired_appearance = _project_retained_appearance(
        appearance,
        retained_old_face_indices=retained_old_face_indices,
        repaired_mesh_lineage_hash=repaired_mesh.mesh_lineage_hash,
    )

    if _vertices_payload_hash(repaired_mesh) != vertices_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_VERTEX_PAYLOAD_DRIFT")
    if _edges_payload_hash(repaired_mesh) != edges_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_EDGE_PAYLOAD_DRIFT")
    if _rows_payload_hash(repaired_mesh_skin) != rows_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_WEIGHT_ROW_PAYLOAD_DRIFT")
    if _retained_artist_payload_hash(repaired_appearance, tuple(range(len(repaired_faces)))) != content_sha256({
        "retained_artist_corners": [
            {
                "old_face_index": new_face_index,
                "corner_index": int(corner.corner_index),
                "payload": {key: value for key, value in corner.to_dict().items() if key != "face_index"},
            }
            for new_face_index in range(len(repaired_faces))
            for corner in repaired_appearance.corner_bindings
            if int(corner.face_index) == new_face_index
        ]
    }):
        raise QualificationError("FACE_SUBSET_REPAIR_INTERNAL_APPEARANCE_PROJECTION_ERROR")

    # Compare retained artist payload directly across the old->new face-index map,
    # ignoring the intentionally changed face index and mesh lineage binding.
    projected_source_payload = []
    projected_repaired_payload = []
    source_by_key = {
        (int(corner.face_index), int(corner.corner_index)): corner
        for corner in appearance.corner_bindings
    }
    repaired_by_key = {
        (int(corner.face_index), int(corner.corner_index)): corner
        for corner in repaired_appearance.corner_bindings
    }
    for new_face_index, old_face_index in enumerate(retained_old_face_indices):
        corner_indices = sorted(ci for fi, ci in source_by_key if fi == old_face_index)
        for corner_index in corner_indices:
            old_payload = source_by_key[(old_face_index, corner_index)].to_dict()
            new_payload = repaired_by_key[(new_face_index, corner_index)].to_dict()
            old_payload.pop("face_index", None)
            new_payload.pop("face_index", None)
            projected_source_payload.append(old_payload)
            projected_repaired_payload.append(new_payload)
    if projected_source_payload != projected_repaired_payload:
        raise QualificationError("FACE_SUBSET_REPAIR_ARTIST_APPEARANCE_PAYLOAD_DRIFT")

    repaired_raster = mesh_raster_quality_report(
        repaired_mesh,
        surface=None,
        view_index=int(repaired_mesh.view_index),
    )
    evaluated = evaluate_mesh_quality(
        coverage=dict(repaired_coverage_report),
        raster_report=repaired_raster,
        policy=policy,
    )
    if not bool(evaluated.get("passed", False)):
        raise QualificationError(
            f"FACE_SUBSET_REPAIR_FROZEN_POLICY_FAILED:{tuple(evaluated.get('failure_invariants', ())) }"
        )

    external = source_component.metadata.get("external_render_support_qualification")
    if not isinstance(external, Mapping):
        raise QualificationError("FACE_SUBSET_REPAIR_EXTERNAL_QUALIFICATION_REQUIRED")
    source_external_hash = _required_hash(
        "SOURCE_EXTERNAL_QUALIFICATION_HASH", external.get("qualification_hash", "")
    )
    removed_face_keys_sha256 = content_sha256({
        "removed_faces": [
            {"index": int(row["face_index"]), "face": tuple(row["face"])}
            for row in removal_reports
        ]
    })
    policy_sha256 = content_sha256(policy.to_dict())
    qualification = QualifiedFaceSubsetRepairIR(
        view_index=int(mesh.view_index),
        source_component_state_hash=str(source_component.component_state_hash),
        source_external_qualification_hash=source_external_hash,
        source_mesh_lineage_hash=str(mesh.mesh_lineage_hash),
        repaired_mesh_lineage_hash=str(repaired_mesh.mesh_lineage_hash),
        source_mesh_skin_lineage_hash=str(mesh_skin.mesh_skin_lineage_hash),
        repaired_mesh_skin_lineage_hash=str(repaired_mesh_skin.mesh_skin_lineage_hash),
        source_appearance_lineage_hash=str(appearance.appearance_lineage_hash),
        repaired_appearance_lineage_hash=str(repaired_appearance.appearance_lineage_hash),
        removed_face_indices=tuple(int(index) for index in remove_face_indices),
        removed_face_keys_sha256=removed_face_keys_sha256,
        source_face_count=len(mesh.faces),
        repaired_face_count=len(repaired_mesh.faces),
        vertices_payload_sha256=vertices_hash,
        edges_payload_sha256=edges_hash,
        weight_rows_payload_sha256=rows_hash,
        retained_artist_payload_sha256=retained_artist_hash,
        source_truth_ownership_sha256=source_truth_ownership_sha256,
        coverage_measurement_sha256=coverage_measurement_sha256,
        frozen_policy_sha256=policy_sha256,
        qualification_report={
            "status": "PASS_EXACT_FROZEN_POLICY_FACE_SUBSET_REPAIR",
            "passed": True,
            "removed_face_reports": removal_reports,
            "repaired_mesh_quality": evaluated,
            "vertices_preserved_exactly": True,
            "edges_preserved_exactly": True,
            "weight_rows_preserved_exactly": True,
            "retained_artist_corner_payload_preserved_exactly": True,
            "face_subset_only": True,
            "retriangulated": False,
            "new_pixels_generated": False,
            "policy_thresholds_changed": False,
            "scientific_mechanical_surface_relabelled": False,
        },
        repair_hash="",
        metadata={
            "repair_scope": "EXACT_FACE_SUBSET_ONLY",
            "allowed_removal_failure_invariants": tuple(sorted(_ALLOWED_REMOVAL_FAILURES)),
            **supplied_metadata,
        },
    )
    qualification = replace(
        qualification,
        repair_hash=face_subset_repair_hash(qualification),
    )
    validate_face_subset_repair_result(
        QualifiedFaceSubsetRepairResult(
            repaired_mesh,
            repaired_mesh_skin,
            repaired_appearance,
            qualification,
        ),
        source_component=source_component,
        policy=policy,
    )
    return QualifiedFaceSubsetRepairResult(
        repaired_mesh,
        repaired_mesh_skin,
        repaired_appearance,
        qualification,
    )


def validate_face_subset_repair_result(
    result: QualifiedFaceSubsetRepairResult,
    *,
    source_component: RenderableComponentIR,
    policy: MeshQualityPolicyV1 = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
) -> None:
    value = result.qualification
    if value.repair_hash != face_subset_repair_hash(value):
        raise QualificationError("FACE_SUBSET_REPAIR_HASH_MISMATCH")
    if value.source_component_state_hash != source_component.component_state_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_SOURCE_COMPONENT_DRIFT")
    if value.source_mesh_lineage_hash != source_component.mesh.mesh_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_SOURCE_MESH_DRIFT")
    if value.source_mesh_skin_lineage_hash != source_component.mesh_skin.mesh_skin_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_SOURCE_MESH_SKIN_DRIFT")
    if value.source_appearance_lineage_hash != source_component.appearance.appearance_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_SOURCE_APPEARANCE_DRIFT")
    if value.repaired_mesh_lineage_hash != result.mesh.mesh_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_MESH_DRIFT")
    if result.mesh.mesh_lineage_hash != mesh_lineage_hash(result.mesh):
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_MESH_HASH_INVALID")
    if value.repaired_mesh_skin_lineage_hash != result.mesh_skin.mesh_skin_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_SKIN_DRIFT")
    if result.mesh_skin.mesh_skin_lineage_hash != mesh_skin_lineage_hash(result.mesh_skin):
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_SKIN_HASH_INVALID")
    if result.mesh_skin.mesh_binding_hash != result.mesh.mesh_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_SKIN_MESH_BINDING_DRIFT")
    if value.repaired_appearance_lineage_hash != result.appearance.appearance_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_APPEARANCE_DRIFT")
    if result.appearance.appearance_lineage_hash != appearance_lineage_hash(result.appearance):
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_APPEARANCE_HASH_INVALID")
    if result.appearance.mesh_binding_hash != result.mesh.mesh_lineage_hash:
        raise QualificationError("FACE_SUBSET_REPAIR_REPAIRED_APPEARANCE_MESH_BINDING_DRIFT")
    if _vertices_payload_hash(result.mesh) != value.vertices_payload_sha256:
        raise QualificationError("FACE_SUBSET_REPAIR_VERTEX_PAYLOAD_DRIFT")
    if _edges_payload_hash(result.mesh) != value.edges_payload_sha256:
        raise QualificationError("FACE_SUBSET_REPAIR_EDGE_PAYLOAD_DRIFT")
    if _rows_payload_hash(result.mesh_skin) != value.weight_rows_payload_sha256:
        raise QualificationError("FACE_SUBSET_REPAIR_WEIGHT_ROW_PAYLOAD_DRIFT")
    if value.frozen_policy_sha256 != content_sha256(policy.to_dict()):
        raise QualificationError("FACE_SUBSET_REPAIR_POLICY_DRIFT")
    _required_hash("COVERAGE_MEASUREMENT_SHA256", value.coverage_measurement_sha256)
    _required_hash("SOURCE_TRUTH_OWNERSHIP_SHA256", value.source_truth_ownership_sha256)
    report = value.qualification_report
    if not bool(report.get("passed", False)):
        raise QualificationError("FACE_SUBSET_REPAIR_REPORT_NOT_PASS")
    for key in (
        "vertices_preserved_exactly",
        "edges_preserved_exactly",
        "weight_rows_preserved_exactly",
        "retained_artist_corner_payload_preserved_exactly",
        "face_subset_only",
    ):
        if not bool(report.get(key, False)):
            raise QualificationError(f"FACE_SUBSET_REPAIR_REQUIRED_INVARIANT_FALSE:{key}")
    for key in (
        "retriangulated",
        "new_pixels_generated",
        "policy_thresholds_changed",
        "scientific_mechanical_surface_relabelled",
    ):
        if bool(report.get(key, True)):
            raise QualificationError(f"FACE_SUBSET_REPAIR_FORBIDDEN_INVARIANT_TRUE:{key}")


__all__ = [
    "QualifiedFaceSubsetRepairIR",
    "QualifiedFaceSubsetRepairResult",
    "face_subset_repair_hash",
    "qualify_exact_policy_face_subset_repair",
    "validate_face_subset_repair_result",
]
