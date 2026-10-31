from __future__ import annotations

"""Product Proof over the current modular canonical product graph.

This module deliberately preserves ``RealSaS.ProductProofBundleIR.v1`` as the
single product-pass authority.  The old V4 evaluator consumed one monolithic
``CanonicalPuppetGraphV3``.  The current product graph instead owns exact
Mesh/Rig/Skin/Mechanics/Graphics/Motion authorities.  This evaluator binds those
artifacts into one deterministic product-state identity and executes real domain
courts without projecting them back into the legacy monolith.
"""

from dataclasses import replace
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .appearance_authority_v2 import validate_complete_appearance_asset
from .canonical_graphics_ir_v1 import (
    CanonicalGraphicsIRV1,
    canonical_graphics_from_dict_v1,
)
from .canonical_mechanics_ir_v1 import (
    CanonicalMechanicsIRV1,
    canonical_mechanics_from_dict_v1,
)
from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_from_dict_v1
from .canonical_motion_ir_v2 import CanonicalMotionIRV2, canonical_motion_from_dict_v2
from .canonical_rig_ir_v2 import CanonicalRigIRV2, canonical_rig_from_dict_v2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2, canonical_skin_from_dict_v2
from .control_motion_runtime_v2 import solve_motion_pose_v2
from .hashing import content_sha256
from .mechanics_control_solver_v1 import deform_mesh_lbs_v1, solve_control_pose_v1
from .types import QualificationError
from .v4 import domain_proof_hash, measurement_report_hash, proof_bundle_hash, proof_plan_hash
from .v4_types import DomainProofReportIR, MeasurementReportIR, ProductProofBundleIR, ProofPlanIR

Json = dict[str, Any]

MODULAR_PRODUCT_STATE_SCHEMA_V1 = "RealSaS.ModularProductState.v1"
MODULAR_PRODUCT_PROOF_EVALUATOR_V1 = "RealSaS.ModularProductProofEvaluator.v1"
REQUIRED_PROOF_DOMAINS_V1 = (
    "MECHANICAL_STRUCTURE",
    "MESH_QUALITY",
    "DEFORMATION",
    "DIRECTIONAL_VISUAL",
    "MOTION",
    "RUNTIME_CONSUMPTION",
)

_MESH_AREA_RELATIVE_EPSILON = 1.0e-12
_DEFORMATION_AREA_RATIO_MIN = 1.0e-6
_MECHANICS_CONSTRAINT_RESIDUAL_MAX = 1.0e-3
_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

_OWNER_BY_DOMAIN = {
    "MECHANICAL_STRUCTURE": "MECHANICS",
    "MESH_QUALITY": "MESH",
    "DEFORMATION": "MECHANICS",
    "DIRECTIONAL_VISUAL": "GRAPHICS",
    "MOTION": "MOTION",
    "RUNTIME_CONSUMPTION": "PROOF_RUNTIME",
}


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise QualificationError(code)


def _roundtrip_validate(
    value: Any,
    decoder: Callable[[Mapping[str, Any]], Any],
    code: str,
) -> None:
    try:
        decoded = decoder(value.to_dict())
    except Exception as exc:  # decoder owns typed fail-closed details
        raise QualificationError(code) from exc
    if decoded != value:
        raise QualificationError(code)


def _canonical_motions(motions: Iterable[CanonicalMotionIRV2]) -> tuple[CanonicalMotionIRV2, ...]:
    rows = tuple(motions)
    _require(bool(rows), "MODULAR_PROOF_MOTION_REQUIRED")
    clip_ids = [row.canonical_clip_id for row in rows]
    lineages = [row.motion_lineage_hash for row in rows]
    _require(len(clip_ids) == len(set(clip_ids)), "MODULAR_PROOF_DUPLICATE_MOTION_CLIP")
    _require(len(lineages) == len(set(lineages)), "MODULAR_PROOF_DUPLICATE_MOTION_LINEAGE")
    return tuple(sorted(rows, key=lambda row: (row.canonical_clip_id, row.motion_lineage_hash)))


def modular_product_state_hash_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    graphics: CanonicalGraphicsIRV1,
    motions: Iterable[CanonicalMotionIRV2],
) -> str:
    motions = _canonical_motions(motions)
    _roundtrip_validate(mesh, canonical_mesh_from_dict_v1, "MODULAR_PROOF_MESH_AUTHORITY_DRIFT")
    _roundtrip_validate(rig, canonical_rig_from_dict_v2, "MODULAR_PROOF_RIG_AUTHORITY_DRIFT")
    _roundtrip_validate(skin, canonical_skin_from_dict_v2, "MODULAR_PROOF_SKIN_AUTHORITY_DRIFT")
    _roundtrip_validate(
        mechanics,
        canonical_mechanics_from_dict_v1,
        "MODULAR_PROOF_MECHANICS_AUTHORITY_DRIFT",
    )
    _roundtrip_validate(
        graphics,
        canonical_graphics_from_dict_v1,
        "MODULAR_PROOF_GRAPHICS_AUTHORITY_DRIFT",
    )
    for motion in motions:
        _roundtrip_validate(
            motion,
            canonical_motion_from_dict_v2,
            "MODULAR_PROOF_MOTION_AUTHORITY_DRIFT",
        )

    _require(rig.mesh_lineage_hash == mesh.mesh_lineage_hash, "MODULAR_PROOF_RIG_MESH_DRIFT")
    _require(skin.mesh_lineage_hash == mesh.mesh_lineage_hash, "MODULAR_PROOF_SKIN_MESH_DRIFT")
    _require(
        skin.deform_structure_hash == rig.deform_structure_hash,
        "MODULAR_PROOF_SKIN_DEFORM_STRUCTURE_DRIFT",
    )
    _require(
        mechanics.mesh_lineage_hash == mesh.mesh_lineage_hash,
        "MODULAR_PROOF_MECHANICS_MESH_DRIFT",
    )
    _require(
        mechanics.rig_lineage_hash == rig.rig_lineage_hash,
        "MODULAR_PROOF_MECHANICS_RIG_DRIFT",
    )
    _require(
        mechanics.deform_structure_hash == rig.deform_structure_hash,
        "MODULAR_PROOF_MECHANICS_DEFORM_STRUCTURE_DRIFT",
    )
    _require(
        mechanics.control_program_hash == rig.control_program_hash,
        "MODULAR_PROOF_MECHANICS_CONTROL_PROGRAM_DRIFT",
    )
    _require(
        mechanics.skin_lineage_hash == skin.skin_lineage_hash,
        "MODULAR_PROOF_MECHANICS_SKIN_DRIFT",
    )
    _require(
        graphics.mesh_lineage_hash == mesh.mesh_lineage_hash,
        "MODULAR_PROOF_GRAPHICS_MESH_DRIFT",
    )
    _require(
        graphics.skin_lineage_hash == skin.skin_lineage_hash,
        "MODULAR_PROOF_GRAPHICS_SKIN_DRIFT",
    )
    _require(
        graphics.mechanics_lineage_hash == mechanics.mechanics_lineage_hash,
        "MODULAR_PROOF_GRAPHICS_MECHANICS_DRIFT",
    )
    for motion in motions:
        _require(
            motion.rig_lineage_hash == rig.rig_lineage_hash,
            "MODULAR_PROOF_MOTION_RIG_DRIFT",
        )
        _require(
            motion.control_program_hash == rig.control_program_hash,
            "MODULAR_PROOF_MOTION_CONTROL_PROGRAM_DRIFT",
        )
        _require(
            motion.mechanics_lineage_hash == mechanics.mechanics_lineage_hash,
            "MODULAR_PROOF_MOTION_MECHANICS_DRIFT",
        )

    return content_sha256(
        {
            "schema": MODULAR_PRODUCT_STATE_SCHEMA_V1,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "rig_lineage_hash": rig.rig_lineage_hash,
            "deform_structure_hash": rig.deform_structure_hash,
            "control_program_hash": rig.control_program_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
            "graphics_lineage_hash": graphics.graphics_lineage_hash,
            "motion_lineage_hashes": [motion.motion_lineage_hash for motion in motions],
        }
    )


def _triangle_areas(mesh: CanonicalMeshIR, positions: Mapping[str, Sequence[float]] | None = None) -> np.ndarray:
    rest = {
        vertex.canonical_mesh_vertex_id: np.asarray(vertex.P, dtype=np.float64)
        for vertex in mesh.vertices
    }
    if positions is not None:
        rest.update({key: np.asarray(tuple(value), dtype=np.float64) for key, value in positions.items()})
    values: list[float] = []
    for face in mesh.faces:
        try:
            a, b, c = (rest[str(vertex_id)] for vertex_id in face)
        except KeyError as exc:
            raise QualificationError("MODULAR_PROOF_FACE_VERTEX_UNKNOWN") from exc
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in (a, b, c)):
            raise QualificationError("MODULAR_PROOF_FACE_POSITION_INVALID")
        values.append(0.5 * float(np.linalg.norm(np.cross(b - a, c - a))))
    return np.asarray(values, dtype=np.float64)


def _mesh_quality_measurements(mesh: CanonicalMeshIR) -> Json:
    positions = np.asarray([vertex.P for vertex in mesh.vertices], dtype=np.float64)
    _require(positions.ndim == 2 and positions.shape[1] == 3, "MODULAR_PROOF_MESH_POSITION_SHAPE")
    _require(np.isfinite(positions).all(), "MODULAR_PROOF_MESH_NONFINITE")
    diagonal = float(np.linalg.norm(np.max(positions, axis=0) - np.min(positions, axis=0)))
    _require(diagonal > 0.0 and math.isfinite(diagonal), "MODULAR_PROOF_MESH_ZERO_EXTENT")
    areas = _triangle_areas(mesh)
    threshold = max(diagonal * diagonal * _MESH_AREA_RELATIVE_EPSILON, 1.0e-18)
    _require(bool(len(areas)), "MODULAR_PROOF_MESH_FACE_REQUIRED")
    _require(bool(np.all(areas > threshold)), "MODULAR_PROOF_MESH_DEGENERATE_FACE")
    return {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "bbox_diagonal": diagonal,
        "minimum_triangle_area": float(np.min(areas)),
        "relative_area_epsilon": _MESH_AREA_RELATIVE_EPSILON,
        "absolute_area_threshold": threshold,
    }


def _mechanical_structure_measurements(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
) -> Json:
    _require(bool(rig.joints), "MODULAR_PROOF_RIG_JOINT_REQUIRED")
    _require(bool(rig.controls), "MODULAR_PROOF_RIG_CONTROL_REQUIRED")
    _require(len(rig.local_frames) == len(rig.joints), "MODULAR_PROOF_RIG_FRAME_COVERAGE")
    _require(len(skin.rows) == len(mesh.vertices), "MODULAR_PROOF_SKIN_COVERAGE")
    _require(
        tuple(mechanics.constraint_schedule)
        == tuple(constraint.canonical_constraint_id for constraint in rig.constraints),
        "MODULAR_PROOF_MECHANICS_SCHEDULE_DRIFT",
    )
    pose = solve_control_pose_v1(rig, ())
    deformed = deform_mesh_lbs_v1(mesh, rig, skin, pose)
    _require(
        tuple(deformed.canonical_mesh_vertex_ids)
        == tuple(vertex.canonical_mesh_vertex_id for vertex in mesh.vertices),
        "MODULAR_PROOF_REST_DEFORMATION_VERTEX_DRIFT",
    )
    values = np.asarray(deformed.positions, dtype=np.float64)
    _require(np.isfinite(values).all(), "MODULAR_PROOF_REST_DEFORMATION_NONFINITE")
    max_residual = max((float(value) for _, value in pose.constraint_residuals), default=0.0)
    _require(
        math.isfinite(max_residual) and max_residual <= _MECHANICS_CONSTRAINT_RESIDUAL_MAX,
        "MODULAR_PROOF_REST_CONSTRAINT_RESIDUAL",
    )
    return {
        "joint_count": len(rig.joints),
        "control_count": len(rig.controls),
        "chain_count": len(rig.chains),
        "constraint_count": len(rig.constraints),
        "skin_row_count": len(skin.rows),
        "max_rest_constraint_residual": max_residual,
        "constraint_residual_limit": _MECHANICS_CONSTRAINT_RESIDUAL_MAX,
        "exact_mechanics_solver_executed": True,
        "lbs_executed": True,
    }


def _sample_times(motion: CanonicalMotionIRV2) -> tuple[float, ...]:
    duration = float(motion.duration_seconds)
    _require(duration > 0.0 and math.isfinite(duration), "MODULAR_PROOF_MOTION_DURATION_INVALID")
    return tuple(duration * value for value in _SAMPLE_FRACTIONS)


def _motion_and_deformation_measurements(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    motions: tuple[CanonicalMotionIRV2, ...],
) -> tuple[Json, Json]:
    rest_areas = _triangle_areas(mesh)
    _require(bool(np.all(rest_areas > 0.0)), "MODULAR_PROOF_REST_TRIANGLE_DEGENERATE")
    motion_rows: list[Json] = []
    deformation_rows: list[Json] = []
    for motion in motions:
        _require(bool(motion.control_tracks), "MODULAR_PROOF_MOTION_TRACK_REQUIRED")
        control_ids = {control.canonical_control_id for control in rig.controls}
        seen_tracks: set[str] = set()
        for track in motion.control_tracks:
            _require(track.canonical_control_id in control_ids, "MODULAR_PROOF_MOTION_CONTROL_UNKNOWN")
            _require(track.canonical_control_id not in seen_tracks, "MODULAR_PROOF_MOTION_CONTROL_DUPLICATE")
            seen_tracks.add(track.canonical_control_id)
            _require(bool(track.keys), "MODULAR_PROOF_MOTION_KEY_REQUIRED")
            times = tuple(float(key.time_seconds) for key in track.keys)
            _require(all(math.isfinite(value) for value in times), "MODULAR_PROOF_MOTION_KEY_TIME_NONFINITE")
            _require(times == tuple(sorted(times)), "MODULAR_PROOF_MOTION_KEY_ORDER_DRIFT")
            _require(times[0] >= 0.0 and times[-1] <= motion.duration_seconds, "MODULAR_PROOF_MOTION_KEY_RANGE_DRIFT")

        clip_max_residual = 0.0
        clip_min_area_ratio = math.inf
        clip_max_displacement = 0.0
        for sample_time in _sample_times(motion):
            pose = solve_motion_pose_v2(
                motion,
                rig=rig,
                mechanics=mechanics,
                time_seconds=sample_time,
            )
            max_residual = max((float(value) for _, value in pose.constraint_residuals), default=0.0)
            _require(math.isfinite(max_residual), "MODULAR_PROOF_MOTION_RESIDUAL_NONFINITE")
            _require(
                max_residual <= _MECHANICS_CONSTRAINT_RESIDUAL_MAX,
                "MODULAR_PROOF_MOTION_CONSTRAINT_RESIDUAL",
            )
            clip_max_residual = max(clip_max_residual, max_residual)

            deformed = deform_mesh_lbs_v1(mesh, rig, skin, pose)
            positions = np.asarray(deformed.positions, dtype=np.float64)
            _require(np.isfinite(positions).all(), "MODULAR_PROOF_DEFORMATION_NONFINITE")
            _require(
                tuple(deformed.canonical_mesh_vertex_ids)
                == tuple(vertex.canonical_mesh_vertex_id for vertex in mesh.vertices),
                "MODULAR_PROOF_DEFORMATION_VERTEX_DRIFT",
            )
            by_id = {
                vertex_id: position
                for vertex_id, position in zip(deformed.canonical_mesh_vertex_ids, deformed.positions)
            }
            areas = _triangle_areas(mesh, by_id)
            ratios = np.divide(areas, rest_areas, out=np.zeros_like(areas), where=rest_areas > 0.0)
            minimum_ratio = float(np.min(ratios))
            _require(
                math.isfinite(minimum_ratio) and minimum_ratio >= _DEFORMATION_AREA_RATIO_MIN,
                "MODULAR_PROOF_DEFORMATION_TRIANGLE_COLLAPSE",
            )
            clip_min_area_ratio = min(clip_min_area_ratio, minimum_ratio)
            rest_positions = np.asarray([vertex.P for vertex in mesh.vertices], dtype=np.float64)
            displacement = np.linalg.norm(positions - rest_positions, axis=1)
            clip_max_displacement = max(clip_max_displacement, float(np.max(displacement)))

        motion_rows.append(
            {
                "canonical_clip_id": motion.canonical_clip_id,
                "intent": motion.intent,
                "sample_count": len(_SAMPLE_FRACTIONS),
                "track_count": len(motion.control_tracks),
                "max_constraint_residual": clip_max_residual,
            }
        )
        deformation_rows.append(
            {
                "canonical_clip_id": motion.canonical_clip_id,
                "sample_count": len(_SAMPLE_FRACTIONS),
                "minimum_triangle_area_ratio": clip_min_area_ratio,
                "minimum_allowed_triangle_area_ratio": _DEFORMATION_AREA_RATIO_MIN,
                "max_vertex_displacement": clip_max_displacement,
                "lbs_executed": True,
            }
        )
    return {
        "clips": motion_rows,
        "total_clip_count": len(motion_rows),
        "control_space_authority": True,
        "exact_mechanics_solver_executed": True,
    }, {
        "clips": deformation_rows,
        "total_clip_count": len(deformation_rows),
        "deformation_method": "LINEAR_BLEND_SKINNING",
        "exact_mechanics_pose_consumed": True,
    }


def _has_cycle(edges: Sequence[tuple[str, str]]) -> bool:
    adjacency: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for front, back in edges:
        adjacency.setdefault(front, []).append(back)
        nodes.update((front, back))
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for child in adjacency.get(node, ()):  # front -> back
            if walk(child):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(walk(node) for node in nodes)


def _directional_visual_measurements(graphics: CanonicalGraphicsIRV1, mesh: CanonicalMeshIR) -> Json:
    validate_complete_appearance_asset(graphics.appearance)
    fragments = {fragment.fragment_id: fragment for fragment in graphics.presentation.fragments}
    _require(bool(fragments), "MODULAR_PROOF_GRAPHICS_FRAGMENT_REQUIRED")
    face_count = len(mesh.faces)
    covered_faces: set[int] = set()
    for fragment in fragments.values():
        _require(bool(fragment.face_indices), "MODULAR_PROOF_GRAPHICS_FRAGMENT_EMPTY")
        for index in fragment.face_indices:
            _require(0 <= int(index) < face_count, "MODULAR_PROOF_GRAPHICS_FACE_OUT_OF_RANGE")
            covered_faces.add(int(index))
    _require(covered_faces == set(range(face_count)), "MODULAR_PROOF_GRAPHICS_FACE_COVERAGE")

    views = tuple(graphics.presentation.views)
    _require(len(views) == 8, "MODULAR_PROOF_GRAPHICS_REQUIRES_8_VIEWS")
    _require(
        tuple(sorted(view.view_index for view in views)) == tuple(range(8)),
        "MODULAR_PROOF_GRAPHICS_VIEW_DOMAIN_DRIFT",
    )
    max_edge_count = 0
    for view in views:
        visible = set(view.visible_fragment_ids)
        _require(visible.issubset(fragments), "MODULAR_PROOF_GRAPHICS_VISIBLE_FRAGMENT_UNKNOWN")
        pairs: list[tuple[str, str]] = []
        for edge in view.order_edges:
            _require(edge.view_index == view.view_index, "MODULAR_PROOF_GRAPHICS_EDGE_VIEW_DRIFT")
            _require(edge.front_fragment_id in fragments, "MODULAR_PROOF_GRAPHICS_EDGE_FRONT_UNKNOWN")
            _require(edge.back_fragment_id in fragments, "MODULAR_PROOF_GRAPHICS_EDGE_BACK_UNKNOWN")
            _require(edge.front_fragment_id != edge.back_fragment_id, "MODULAR_PROOF_GRAPHICS_EDGE_SELF")
            pairs.append((edge.front_fragment_id, edge.back_fragment_id))
        _require(not _has_cycle(pairs), "MODULAR_PROOF_GRAPHICS_ORDER_CYCLE")
        max_edge_count = max(max_edge_count, len(pairs))

    texture_views = tuple(sorted(texture.direction_index for texture in graphics.appearance.textures))
    _require(texture_views == tuple(range(8)), "MODULAR_PROOF_APPEARANCE_REQUIRES_8_VIEWS")
    _require(
        graphics.metadata.get("source_art_wins_where_observed") is True,
        "MODULAR_PROOF_GRAPHICS_SOURCE_WINS_DRIFT",
    )
    return {
        "fragment_count": len(fragments),
        "covered_face_count": len(covered_faces),
        "view_count": len(views),
        "appearance_texture_view_count": len(texture_views),
        "maximum_order_edge_count_per_view": max_edge_count,
        "order_graphs_acyclic": True,
        "source_art_wins_where_observed": True,
        "caa_is_graphics_appearance_sublayer": True,
    }


def _runtime_consumption_measurements(
    state_hash: str,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    graphics: CanonicalGraphicsIRV1,
    motions: tuple[CanonicalMotionIRV2, ...],
) -> Json:
    payload = {
        "schema": MODULAR_PRODUCT_STATE_SCHEMA_V1,
        "product_state_hash": state_hash,
        "mesh": mesh.to_dict(),
        "rig": rig.to_dict(),
        "skin": skin.to_dict(),
        "mechanics": mechanics.to_dict(),
        "graphics": graphics.to_dict(),
        "motions": [motion.to_dict() for motion in motions],
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except Exception as exc:
        raise QualificationError("MODULAR_PROOF_RUNTIME_SERIALIZATION_FAILED") from exc
    _require(bool(encoded), "MODULAR_PROOF_RUNTIME_SERIALIZATION_EMPTY")
    return {
        "modular_state_schema": MODULAR_PRODUCT_STATE_SCHEMA_V1,
        "serialized_state_size_bytes": len(encoded),
        "native_runtime_execution_claimed": False,
        "pre_export_contract_compatibility_only": True,
        "post_export_native_interlock_required": True,
    }


def _proof_policy(domain: str) -> Json:
    return {
        "schema": "RealSaS.ModularProductProofPolicy.v1",
        "domain": domain,
        "evaluator": MODULAR_PRODUCT_PROOF_EVALUATOR_V1,
        "mesh_area_relative_epsilon": _MESH_AREA_RELATIVE_EPSILON,
        "deformation_area_ratio_min": _DEFORMATION_AREA_RATIO_MIN,
        "constraint_residual_max": _MECHANICS_CONSTRAINT_RESIDUAL_MAX,
        "motion_sample_fractions": _SAMPLE_FRACTIONS,
    }


def _bind_domain_report(
    state_hash: str,
    domain: str,
    evaluator: Callable[[], Json],
) -> DomainProofReportIR:
    policy = _proof_policy(domain)
    policy_hash = content_sha256(policy)
    plan = ProofPlanIR(
        source_product_state_hash=state_hash,
        proof_domain=domain,
        operator_policy_hashes=(policy_hash,),
        probe_specification={
            "evaluator": MODULAR_PRODUCT_PROOF_EVALUATOR_V1,
            "domain": domain,
            "deterministic": True,
        },
        proof_plan_hash="",
    )
    plan = replace(plan, proof_plan_hash=proof_plan_hash(plan))

    status = "PASS"
    failures: tuple[Json, ...] = ()
    try:
        measurements = evaluator()
        measurements = {"status": "PASS", **measurements}
    except Exception as exc:
        status = "FAIL"
        code = str(exc) or type(exc).__name__
        measurements = {
            "status": "FAIL",
            "failure_code": code,
            "exception_type": type(exc).__name__,
        }
        failures = ({"code": code, "exception_type": type(exc).__name__},)

    measurement = MeasurementReportIR(
        source_product_state_hash=state_hash,
        proof_plan_hash=plan.proof_plan_hash,
        measurements=measurements,
        measurement_report_hash="",
    )
    measurement = replace(
        measurement,
        measurement_report_hash=measurement_report_hash(measurement),
    )
    report = DomainProofReportIR(
        source_product_state_hash=state_hash,
        proof_domain=domain,
        proof_plan_hash=plan.proof_plan_hash,
        measurement_report_hash=measurement.measurement_report_hash,
        status=status,
        failure_signatures=failures,
        owner_attribution=(
            {
                "owner": _OWNER_BY_DOMAIN[domain],
                "authority": "MODULAR_PRODUCT_PROOF_V1",
            },
        ),
        domain_proof_hash="",
        metadata={
            "policy_hash": policy_hash,
            "measurements": measurements,
            "evaluator": MODULAR_PRODUCT_PROOF_EVALUATOR_V1,
        },
    )
    return replace(report, domain_proof_hash=domain_proof_hash(report))


def evaluate_modular_product_proof_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    graphics: CanonicalGraphicsIRV1,
    motions: Iterable[CanonicalMotionIRV2],
) -> ProductProofBundleIR:
    motions = _canonical_motions(motions)
    state_hash = modular_product_state_hash_v1(mesh, rig, skin, mechanics, graphics, motions)
    motion_measurements, deformation_measurements = _motion_and_deformation_measurements(
        mesh,
        rig,
        skin,
        mechanics,
        motions,
    )
    evaluators: dict[str, Callable[[], Json]] = {
        "MECHANICAL_STRUCTURE": lambda: _mechanical_structure_measurements(mesh, rig, skin, mechanics),
        "MESH_QUALITY": lambda: _mesh_quality_measurements(mesh),
        "DEFORMATION": lambda: deformation_measurements,
        "DIRECTIONAL_VISUAL": lambda: _directional_visual_measurements(graphics, mesh),
        "MOTION": lambda: motion_measurements,
        "RUNTIME_CONSUMPTION": lambda: _runtime_consumption_measurements(
            state_hash,
            mesh,
            rig,
            skin,
            mechanics,
            graphics,
            motions,
        ),
    }
    reports = tuple(
        _bind_domain_report(state_hash, domain, evaluators[domain])
        for domain in REQUIRED_PROOF_DOMAINS_V1
    )
    overall = "PASS" if all(report.status == "PASS" for report in reports) else "FAIL"
    value = ProductProofBundleIR(
        source_product_state_hash=state_hash,
        required_domains=REQUIRED_PROOF_DOMAINS_V1,
        domain_reports=reports,
        overall_status=overall,
        proof_bundle_hash="",
        metadata={
            "product_state_schema": MODULAR_PRODUCT_STATE_SCHEMA_V1,
            "evaluator": MODULAR_PRODUCT_PROOF_EVALUATOR_V1,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "rig_lineage_hash": rig.rig_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
            "graphics_lineage_hash": graphics.graphics_lineage_hash,
            "motion_lineage_hashes": [motion.motion_lineage_hash for motion in motions],
            "legacy_canonical_puppet_graph_projection_used": False,
            "product_pass_authority": "PRODUCT_PROOF_BUNDLE_IR_V1",
        },
    )
    return replace(value, proof_bundle_hash=proof_bundle_hash(value))


def product_proof_bundle_from_dict_v1(payload: Mapping[str, Any]) -> ProductProofBundleIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.ProductProofBundleIR.v1":
        raise QualificationError("PRODUCT_PROOF_BUNDLE_SCHEMA_DRIFT")
    reports = tuple(
        DomainProofReportIR(
            source_product_state_hash=str(row["source_product_state_hash"]),
            proof_domain=str(row["proof_domain"]),
            proof_plan_hash=str(row["proof_plan_hash"]),
            measurement_report_hash=str(row["measurement_report_hash"]),
            status=str(row["status"]),
            failure_signatures=tuple(dict(value) for value in row.get("failure_signatures") or ()),
            owner_attribution=tuple(dict(value) for value in row.get("owner_attribution") or ()),
            domain_proof_hash=str(row["domain_proof_hash"]),
            schema_version=str(row.get("schema_version") or "RealSaS.DomainProofReportIR.v1"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("domain_reports") or ()
    )
    value = ProductProofBundleIR(
        source_product_state_hash=str(payload["source_product_state_hash"]),
        required_domains=tuple(map(str, payload.get("required_domains") or ())),
        domain_reports=reports,
        overall_status=str(payload["overall_status"]),
        proof_bundle_hash=str(payload["proof_bundle_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.proof_bundle_hash != proof_bundle_hash(value):
        raise QualificationError("PRODUCT_PROOF_BUNDLE_HASH_DRIFT")
    for report in value.domain_reports:
        if report.domain_proof_hash != domain_proof_hash(report):
            raise QualificationError("PRODUCT_DOMAIN_PROOF_HASH_DRIFT")
        if report.source_product_state_hash != value.source_product_state_hash:
            raise QualificationError("PRODUCT_DOMAIN_PROOF_STATE_DRIFT")
    return value


def require_current_modular_proof_bundle_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    graphics: CanonicalGraphicsIRV1,
    motions: Iterable[CanonicalMotionIRV2],
    bundle: ProductProofBundleIR,
    *,
    require_pass: bool = False,
) -> None:
    motions = _canonical_motions(motions)
    expected = modular_product_state_hash_v1(mesh, rig, skin, mechanics, graphics, motions)
    if bundle.source_product_state_hash != expected:
        raise QualificationError("STALE_MODULAR_PRODUCT_PROOF_BUNDLE")
    if bundle.proof_bundle_hash != proof_bundle_hash(bundle):
        raise QualificationError("PRODUCT_PROOF_BUNDLE_HASH_DRIFT")
    if bundle.required_domains != REQUIRED_PROOF_DOMAINS_V1:
        raise QualificationError("PRODUCT_PROOF_REQUIRED_DOMAIN_DRIFT")
    by_domain = {report.proof_domain: report for report in bundle.domain_reports}
    if set(by_domain) != set(REQUIRED_PROOF_DOMAINS_V1):
        raise QualificationError("PRODUCT_PROOF_DOMAIN_COVERAGE_DRIFT")
    for report in bundle.domain_reports:
        if report.domain_proof_hash != domain_proof_hash(report):
            raise QualificationError("PRODUCT_DOMAIN_PROOF_HASH_DRIFT")
        if report.source_product_state_hash != expected:
            raise QualificationError("PRODUCT_DOMAIN_PROOF_STATE_DRIFT")
    if require_pass and bundle.overall_status != "PASS":
        raise QualificationError("PRODUCT_REQUIRES_PASS_MODULAR_PROOF_BUNDLE")


__all__ = [
    "MODULAR_PRODUCT_PROOF_EVALUATOR_V1",
    "MODULAR_PRODUCT_STATE_SCHEMA_V1",
    "REQUIRED_PROOF_DOMAINS_V1",
    "evaluate_modular_product_proof_v1",
    "modular_product_state_hash_v1",
    "product_proof_bundle_from_dict_v1",
    "require_current_modular_proof_bundle_v1",
]
