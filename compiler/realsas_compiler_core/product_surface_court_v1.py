from __future__ import annotations

"""Compiler court from current ShapeProposal evidence to RiggingSurfaceIR.

The learned Shape lane supplies two distinct evidence products:
- IRIS surface witnesses with orientation-bearing normals;
- TESSA decoded topology in the normalized TESSA frame.

Neither is canonical Mesh authority. This court binds them to the Compiler
normalization/camera contract, uses TESSA face incidence as the *only* normal
neighborhood authority, uses IRIS normals only to orient the sign of that local
topology normal, reuses the frozen Stage14 numerical adequacy search, and emits
one explicitly product-scoped RiggingSurface evidence profile. The legacy
SIGNED_ZERO_SURFACE labels produced by the numerical Stage14 helper are never
exported by this module.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .hashing import content_sha256
from .substrate.adequacy_v1 import select_adequate_rigging_surface_v1
from .types import QualificationError, RiggingSurfaceIR, SurfaceNode, SurfaceRelation


PRODUCT_SURFACE_PROFILE_V1 = "PRODUCT_SHAPE_TOPOLOGY_SURFACE_V1"
PRODUCT_SURFACE_RELATION_V1 = "SHAPE_TOPOLOGY_NEIGHBOR"
PRODUCT_SURFACE_OBSERVED_V1 = "OBSERVED_PRODUCT_SURFACE_EVIDENCE"
PRODUCT_SURFACE_COMPLETED_V1 = "MODEL_COMPLETED_PRODUCT_SURFACE_EVIDENCE"
PRODUCT_SURFACE_BUILDER_V1 = "RealSaS.Compiler.ShapeTopologySurfaceCourt.v1"
PRODUCT_SURFACE_POLICY_SCHEMA_V1 = "RealSaS.ProductShapeSurfaceCourtPolicy.v1"
PRODUCT_SURFACE_QUALIFICATION_SCHEMA_V1 = "RealSaS.ProductShapeSurfaceQualification.v1"


@dataclass(frozen=True)
class ProductSurfaceCourtResultV1:
    surface: RiggingSurfaceIR
    qualification: dict[str, Any]


def _as_points(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 1 or not np.isfinite(array).all():
        raise QualificationError(f"PRODUCT_SURFACE_{name}_INVALID")
    return array


def _unit_rows(value: np.ndarray, *, code: str) -> np.ndarray:
    length = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(~np.isfinite(length)) or np.any(length <= 1e-12):
        raise QualificationError(code)
    return value / length


def _shape_model_sha(shape: Any, family: str) -> str:
    models = dict(shape.metadata.get("models") or {})
    row = dict(models.get(family) or {})
    digest = str(row.get("checkpoint_content_sha256") or "")
    if len(digest) != 64:
        raise QualificationError("PRODUCT_SURFACE_MODEL_CHECKPOINT_BINDING_MISSING:" + family)
    return digest


def _validate_policy(
    policy: Mapping[str, Any],
    *,
    expected_policy_sha256: str,
    source_fidelity_document: Mapping[str, Any],
    source_fidelity_sha256: str,
    adequacy_document: Mapping[str, Any],
    adequacy_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    value = dict(policy)
    if value.get("schema") != PRODUCT_SURFACE_POLICY_SCHEMA_V1:
        raise QualificationError("PRODUCT_SURFACE_POLICY_SCHEMA_DRIFT")
    if str(value.get("status") or "") != "FROZEN_BEFORE_CURRENT_KNIGHT_PRODUCT_SURFACE_RESULT":
        raise QualificationError("PRODUCT_SURFACE_POLICY_NOT_FROZEN")
    if len(expected_policy_sha256) != 64:
        raise QualificationError("PRODUCT_SURFACE_POLICY_SHA_REQUIRED")

    source_ref = dict(value.get("source_fidelity_policy") or {})
    if (
        source_ref.get("sha256") != source_fidelity_sha256
        or source_fidelity_document.get("schema") != "RealSaS.Stage13RestReprojectionPolicy.v2"
    ):
        raise QualificationError("PRODUCT_SURFACE_SOURCE_FIDELITY_POLICY_DRIFT")
    adequacy_ref = dict(value.get("surface_adequacy_policy") or {})
    if (
        adequacy_ref.get("sha256") != adequacy_sha256
        or adequacy_document.get("schema") != "RealSaS.Stage14SubstrateAdequacyPolicy.v3"
    ):
        raise QualificationError("PRODUCT_SURFACE_ADEQUACY_POLICY_DRIFT")

    frame = dict(value.get("frame_contract") or {})
    if (
        frame.get("shape_evidence_frame") != "RealSaS.TESSA.SequenceNormalizedFrame.v1"
        or float(frame.get("tessa_to_normalization_half_extent_scale", 0.0)) != 2.0
        or frame.get("compiler_world_frame_owner") is not True
    ):
        raise QualificationError("PRODUCT_SURFACE_FRAME_POLICY_DRIFT")

    orientation = dict(value.get("normal_orientation") or {})
    if (
        orientation.get("topology_neighborhood_authority") != "DECODED_TESSA_FACE_INCIDENCE_ONLY"
        or orientation.get("orientation_hint_source") != "IRIS_SURFACE_WITNESS_NORMALS"
        or orientation.get("orientation_hint_role") != "SIGN_ONLY_NOT_NEIGHBORHOOD_AUTHORITY"
        or orientation.get("cross_sheet_euclidean_normal_borrowing_forbidden") is not True
    ):
        raise QualificationError("PRODUCT_SURFACE_NORMAL_POLICY_DRIFT")
    p95_limit = float(orientation.get("nearest_hint_p95_distance_norm_max", -1.0))
    max_limit = float(orientation.get("nearest_hint_max_distance_norm_max", -1.0))
    if p95_limit <= 0.0 or max_limit <= 0.0 or p95_limit > max_limit:
        raise QualificationError("PRODUCT_SURFACE_HINT_DISTANCE_POLICY_INVALID")

    profile = dict(value.get("surface_profile") or {})
    expected_profile = {
        "profile_id": PRODUCT_SURFACE_PROFILE_V1,
        "relation_kind": PRODUCT_SURFACE_RELATION_V1,
        "observed_flag": PRODUCT_SURFACE_OBSERVED_V1,
        "completed_flag": PRODUCT_SURFACE_COMPLETED_V1,
        "builder_id": PRODUCT_SURFACE_BUILDER_V1,
    }
    if any(profile.get(key) != expected for key, expected in expected_profile.items()):
        raise QualificationError("PRODUCT_SURFACE_PROFILE_POLICY_DRIFT")
    return orientation, dict(source_fidelity_document["thresholds"]), dict(adequacy_document["gsa"])


def _nearest_iris_orientation_hints(
    decoded_tessa: np.ndarray,
    witness_tessa: np.ndarray,
    witness_normals: np.ndarray,
    *,
    p95_limit_norm: float,
    max_limit_norm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(witness_tessa) != len(witness_normals):
        raise QualificationError("PRODUCT_SURFACE_IRIS_WITNESS_CARDINALITY_DRIFT")
    witness_normals = _unit_rows(
        witness_normals,
        code="PRODUCT_SURFACE_IRIS_WITNESS_NORMAL_DEGENERATE",
    )
    tree = cKDTree(witness_tessa)
    distance_tessa, index = tree.query(decoded_tessa, k=1, workers=-1)
    distance_norm = np.asarray(distance_tessa, dtype=np.float64) * 2.0
    if not np.isfinite(distance_norm).all():
        raise QualificationError("PRODUCT_SURFACE_IRIS_HINT_DISTANCE_NONFINITE")
    p95 = float(np.percentile(distance_norm, 95.0))
    maximum = float(np.max(distance_norm))
    if p95 > float(p95_limit_norm) + 1e-12 or maximum > float(max_limit_norm) + 1e-12:
        raise QualificationError(
            "PRODUCT_SURFACE_IRIS_HINT_DISTANCE_GATE_FAILED:"
            f"p95={p95:.9g}>{p95_limit_norm:.9g}:max={maximum:.9g}>{max_limit_norm:.9g}"
        )
    hints = witness_normals[np.asarray(index, dtype=np.int64)]
    return hints, {
        "witness_count": int(len(witness_tessa)),
        "decoded_vertex_count": int(len(decoded_tessa)),
        "nearest_hint_p95_distance_norm": p95,
        "nearest_hint_max_distance_norm": maximum,
        "nearest_hint_p95_distance_norm_limit": float(p95_limit_norm),
        "nearest_hint_max_distance_norm_limit": float(max_limit_norm),
        "hint_role": "ORIENTATION_SIGN_ONLY",
        "hint_used_for_topology_neighborhood": False,
    }


def _reprofile_surface(
    intermediate: RiggingSurfaceIR,
    *,
    shape_content_sha256: str,
    decoded_topology_sha256: str,
    iris_checkpoint_sha256: str,
    tessa_checkpoint_sha256: str,
    normalization_hash: str,
    policy_sha256: str,
    hint_diagnostics: Mapping[str, Any],
    adequacy_summary: Mapping[str, Any],
) -> RiggingSurfaceIR:
    old_to_new: dict[str, str] = {}
    nodes: list[SurfaceNode] = []
    for index, node in enumerate(intermediate.surface_nodes):
        support = tuple(map(int, node.support_views))
        sid = "PSS:" + content_sha256(
            {
                "shape": shape_content_sha256,
                "index": int(index),
                "P": tuple(map(float, node.P)),
                "support_views": support,
            }
        )[:20]
        old_to_new[str(node.surface_id)] = sid
        observed = bool(support)
        nodes.append(
            SurfaceNode(
                surface_id=sid,
                P=tuple(map(float, node.P)),
                support_views=support,
                provenance_refs=(
                    "ShapeProposal:" + shape_content_sha256,
                    "IRISCheckpoint:" + iris_checkpoint_sha256,
                    "TESSACheckpoint:" + tessa_checkpoint_sha256,
                ),
                source_observation_ids=tuple(node.source_observation_ids),
                raster_bindings=tuple(node.raster_bindings),
                persistence_group_id=f"PRODUCT_SHAPE_SURFACE:{index:05d}",
                derived_normal=None if node.derived_normal is None else tuple(map(float, node.derived_normal)),
                validity_flags=(
                    (PRODUCT_SURFACE_OBSERVED_V1,)
                    if observed
                    else (PRODUCT_SURFACE_COMPLETED_V1,)
                ),
                metadata={
                    **dict(node.metadata or {}),
                    "surface_evidence_profile": PRODUCT_SURFACE_PROFILE_V1,
                    "normal_orientation_hint_source": "IRIS_SURFACE_WITNESS_NORMALS",
                    "normal_orientation_hint_role": "SIGN_ONLY_NOT_NEIGHBORHOOD_AUTHORITY",
                    "topology_neighborhood_authority": "DECODED_TESSA_FACE_INCIDENCE_ONLY",
                    "scene_first_signed_geometry": False,
                    "signed_zero_surface_claimed": False,
                    "teacher_truth_used": False,
                },
            )
        )

    relations: list[SurfaceRelation] = []
    for relation in intermediate.local_relations:
        a = old_to_new[str(relation.a_surface_id)]
        b = old_to_new[str(relation.b_surface_id)]
        rid = "PSSREL:" + content_sha256(
            {
                "shape": shape_content_sha256,
                "a": a,
                "b": b,
                "distance": float((relation.metadata or {}).get("world_distance", 0.0)),
            }
        )[:20]
        relations.append(
            SurfaceRelation(
                relation_id=rid,
                a_surface_id=a,
                b_surface_id=b,
                relation_kind=PRODUCT_SURFACE_RELATION_V1,
                score=float(relation.score),
                metadata={
                    **dict(relation.metadata or {}),
                    "topology_neighborhood_authority": "DECODED_TESSA_FACE_INCIDENCE_ONLY",
                    "surface_evidence_profile": PRODUCT_SURFACE_PROFILE_V1,
                    "teacher_truth_used": False,
                },
            )
        )

    lineage = content_sha256(
        {
            "schema": "RealSaS.ProductShapeTopologyToRiggingSurface.v1",
            "shape_proposal_content_sha256": shape_content_sha256,
            "decoded_topology_sha256": decoded_topology_sha256,
            "iris_checkpoint_sha256": iris_checkpoint_sha256,
            "tessa_checkpoint_sha256": tessa_checkpoint_sha256,
            "normalization_hash": normalization_hash,
            "policy_sha256": policy_sha256,
            "nodes": [node.to_dict() for node in nodes],
            "relations": [relation.to_dict() for relation in relations],
        }
    )
    metadata = {
        "surface_evidence_profile": PRODUCT_SURFACE_PROFILE_V1,
        "builder_id": PRODUCT_SURFACE_BUILDER_V1,
        "shape_proposal_content_sha256": shape_content_sha256,
        "decoded_topology_sha256": decoded_topology_sha256,
        "iris_checkpoint_sha256": iris_checkpoint_sha256,
        "tessa_checkpoint_sha256": tessa_checkpoint_sha256,
        "normalization_hash": normalization_hash,
        "policy_sha256": policy_sha256,
        "scene_first_signed_geometry": False,
        "signed_zero_surface_claimed": False,
        "decoded_topology_is_proposal_evidence": True,
        "canonical_mesh_authority": False,
        "topology_neighborhood_authority": "DECODED_TESSA_FACE_INCIDENCE_ONLY",
        "normal_orientation_hint_source": "IRIS_SURFACE_WITNESS_NORMALS",
        "normal_orientation_hint_role": "SIGN_ONLY_NOT_NEIGHBORHOOD_AUTHORITY",
        "cross_sheet_euclidean_normal_borrowing": False,
        "Nd_operator": str(intermediate.metadata.get("Nd_operator") or ""),
        "Nd_operator_sha256": str(intermediate.metadata.get("Nd_operator_sha256") or ""),
        "observed_node_count": sum(bool(node.support_views) for node in nodes),
        "completed_node_count": sum(not bool(node.support_views) for node in nodes),
        "raster_coordinate_system": "PIXEL_CENTER_XY",
        "resolution": int(intermediate.metadata.get("resolution") or 0),
        "hint_diagnostics": dict(hint_diagnostics),
        "adequacy_summary": dict(adequacy_summary),
        "teacher_truth_used": False,
        "product_authority_claimed": False,
    }
    return RiggingSurfaceIR(
        surface_nodes=tuple(nodes),
        local_relations=tuple(sorted(relations, key=lambda value: value.relation_id)),
        geometry_lineage_hash=lineage,
        builder_id=PRODUCT_SURFACE_BUILDER_V1,
        schema_version="RealSaS.RiggingSurfaceIR.v1",
        metadata=metadata,
    )


def build_product_shape_surface_v1(
    *,
    shape: Any,
    shape_content_sha256: str,
    cameras: Any,
    normalization: Any,
    policy: Mapping[str, Any],
    policy_sha256: str,
    source_fidelity_document: Mapping[str, Any],
    source_fidelity_sha256: str,
    adequacy_document: Mapping[str, Any],
    adequacy_sha256: str,
) -> ProductSurfaceCourtResultV1:
    if len(str(shape_content_sha256)) != 64:
        raise QualificationError("PRODUCT_SURFACE_SHAPE_CONTENT_SHA_INVALID")
    if shape.metadata.get("schema") != "RealSaS.ShapeProposal.v2":
        raise QualificationError("PRODUCT_SURFACE_SHAPE_SCHEMA_DRIFT")
    if shape.metadata.get("proposal_evidence_frame") != "RealSaS.TESSA.SequenceNormalizedFrame.v1":
        raise QualificationError("PRODUCT_SURFACE_SHAPE_FRAME_DRIFT")
    if shape.metadata.get("decoded_topology_is_proposal_evidence") is not True:
        raise QualificationError("PRODUCT_SURFACE_TOPOLOGY_EVIDENCE_FLAG_REQUIRED")
    if shape.metadata.get("canonical_mesh_authority") is not False:
        raise QualificationError("PRODUCT_SURFACE_SHAPE_MESH_AUTHORITY_FORBIDDEN")

    orientation_policy, source_thresholds, gsa_policy = _validate_policy(
        policy,
        expected_policy_sha256=policy_sha256,
        source_fidelity_document=source_fidelity_document,
        source_fidelity_sha256=source_fidelity_sha256,
        adequacy_document=adequacy_document,
        adequacy_sha256=adequacy_sha256,
    )
    decoded_tessa = _as_points(shape.decoded_vertices_tessa, name="DECODED_VERTICES")
    faces = np.asarray(shape.decoded_faces, dtype=np.int64)
    components = np.asarray(shape.decoded_vertex_component_indices, dtype=np.int64).reshape(-1)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) < 1
        or np.any(faces < 0)
        or np.any(faces >= len(decoded_tessa))
        or len(components) != len(decoded_tessa)
    ):
        raise QualificationError("PRODUCT_SURFACE_DECODED_TOPOLOGY_INVALID")

    witness_tessa = _as_points(shape.surface_xyz_tessa, name="IRIS_WITNESS_XYZ")
    witness_normals = _as_points(shape.surface_normal, name="IRIS_WITNESS_NORMAL")
    hints, hint_diagnostics = _nearest_iris_orientation_hints(
        decoded_tessa,
        witness_tessa,
        witness_normals,
        p95_limit_norm=float(orientation_policy["nearest_hint_p95_distance_norm_max"]),
        max_limit_norm=float(orientation_policy["nearest_hint_max_distance_norm_max"]),
    )

    iris_checkpoint = _shape_model_sha(shape, "IRIS")
    tessa_checkpoint = _shape_model_sha(shape, "TESSA")
    camera_dicts = tuple(asdict(camera) for camera in sorted(cameras.cameras, key=lambda value: value.view_index))
    if len(camera_dicts) != 8:
        raise QualificationError("PRODUCT_SURFACE_EXACT_EIGHT_CAMERAS_REQUIRED")

    # TESSA uses [-0.5,+0.5] while NormalizationDomain uses half-extent units.
    decoded_normalized = decoded_tessa * 2.0
    intermediate, adequacy = select_adequate_rigging_surface_v1(
        decoded_normalized,
        faces,
        hints,
        camera_dicts,
        normalization_center=normalization.center_xyz,
        normalization_half_extent=float(normalization.half_extent),
        authority_label="PRODUCT_SHAPE_TOPOLOGY_NUMERICAL_INTERMEDIATE_ONLY",
        source_run_id="PRODUCT_MESH_CAPABILITY",
        source_checkpoint_sha256=iris_checkpoint,
        source_zero_surface_sha256=shape_content_sha256,
        normal_k=int(gsa_policy["normal_k"]),
        visibility_depth_tolerance_norm=float(gsa_policy["visibility_depth_tolerance_norm"]),
        adequacy_policy=dict(gsa_policy["adequacy_policy"]),
        precomputed_component_labels=components,
        metadata={
            "product_shape_numerical_intermediate_only": True,
            "signed_zero_surface_authority_claimed": False,
            "decoded_topology_sha256": str(shape.metadata.get("decoded_topology_sha256") or ""),
        },
    )
    if intermediate is None or str(adequacy.get("status") or "") != "PASS":
        raise QualificationError("PRODUCT_SURFACE_ADEQUACY_FAILED")

    adequacy_summary = {
        "selected_target_node_cap": adequacy.get("selected_target_node_cap"),
        "selected_actual_node_count": adequacy.get("selected_actual_node_count"),
        "selected_surface_lineage_hash_internal_numeric_only": adequacy.get("selected_surface_lineage_hash"),
        "evaluated_candidates": adequacy.get("evaluated_candidates"),
        "selection_rule": adequacy.get("selection_rule"),
        "teacher_truth_used": False,
    }
    surface = _reprofile_surface(
        intermediate,
        shape_content_sha256=shape_content_sha256,
        decoded_topology_sha256=str(shape.metadata.get("decoded_topology_sha256") or ""),
        iris_checkpoint_sha256=iris_checkpoint,
        tessa_checkpoint_sha256=tessa_checkpoint,
        normalization_hash=str(normalization.normalization_hash),
        policy_sha256=policy_sha256,
        hint_diagnostics=hint_diagnostics,
        adequacy_summary=adequacy_summary,
    )
    qualification = {
        "schema": PRODUCT_SURFACE_QUALIFICATION_SCHEMA_V1,
        "status": "PASS",
        "shape_proposal_content_sha256": shape_content_sha256,
        "decoded_topology_sha256": str(shape.metadata.get("decoded_topology_sha256") or ""),
        "iris_checkpoint_sha256": iris_checkpoint,
        "tessa_checkpoint_sha256": tessa_checkpoint,
        "normalization_hash": str(normalization.normalization_hash),
        "camera_set_hash": str(cameras.camera_set_hash),
        "surface_policy_sha256": policy_sha256,
        "source_fidelity_policy_sha256": source_fidelity_sha256,
        "source_fidelity_thresholds": source_thresholds,
        "surface_adequacy_policy_sha256": adequacy_sha256,
        "hint_diagnostics": hint_diagnostics,
        "adequacy": adequacy_summary,
        "surface_geometry_lineage_hash": surface.geometry_lineage_hash,
        "canonical_mesh_authority_claimed": False,
        "teacher_truth_used": False,
        "qualification_hash": "",
    }
    qualification["qualification_hash"] = content_sha256(
        {key: value for key, value in qualification.items() if key != "qualification_hash"}
    )
    return ProductSurfaceCourtResultV1(surface=surface, qualification=qualification)


__all__ = [
    "PRODUCT_SURFACE_PROFILE_V1",
    "PRODUCT_SURFACE_RELATION_V1",
    "PRODUCT_SURFACE_OBSERVED_V1",
    "PRODUCT_SURFACE_COMPLETED_V1",
    "PRODUCT_SURFACE_BUILDER_V1",
    "ProductSurfaceCourtResultV1",
    "build_product_shape_surface_v1",
]
