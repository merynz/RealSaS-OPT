from __future__ import annotations

"""Qualified current directional evaluator for the present rotation-only preset lane.

The evaluator consumes a Compiler-qualified DirectionalJointViewBindingSetIR. Rest
mesh raster positions come from each mesh vertex's admitted SurfaceSupportBinding;
joint pivots come from the separately qualified mechanical-P -> raster affine map.
Mechanical P.xy is never used as a directional coordinate.

Current operation scope is deliberately narrow and generic: authored joint rotation
is supported; nonzero translation/depth, non-unit scale, and untyped order/visibility
tracks fail closed until their runtime semantics are separately qualified.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from compiler.realsas_compiler_core.directional_binding import (
    DirectionalJointViewBindingSetIR,
    assert_directional_binding_for_product,
    joint_pivot,
    project_mechanical_point,
    projection_for_view,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from .motion_bake import bind_qualification_owned_motion_bake


@dataclass(frozen=True)
class DirectionalMotionEvaluatorPolicyV1:
    uniform_sample_count: int = 9
    max_mesh_projection_residual01: float = 0.05
    operation_scope: str = "ROTATION_ONLY_CURRENT_PRESET_V1"
    schema_version: str = "RealSaS.DirectionalMotionEvaluatorPolicy.v1"

    def validate(self) -> None:
        if self.uniform_sample_count < 3:
            raise ValueError("directional evaluator requires >=3 uniform samples")
        if not (0.0 < self.max_mesh_projection_residual01 < 1.0):
            raise ValueError("invalid mesh projection residual policy")
        if self.operation_scope != "ROTATION_ONLY_CURRENT_PRESET_V1":
            raise ValueError("unsupported directional evaluator operation scope")

    @property
    def policy_hash(self) -> str:
        self.validate()
        return content_sha256(asdict(self))


EVALUATOR_SEMANTIC_VERSION = "RealSaS.DirectionalMotionEvaluator.v1.rotation_only"


def _surface_raster(surface, view_index: int) -> dict[str, tuple[float, float]]:
    out = {}
    for node in surface.surface_nodes:
        rows = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view_index)]
        if len(rows) > 1:
            raise QualificationError(f"DIRECTIONAL_EVALUATOR_DUPLICATE_SURFACE_RASTER:{node.surface_id}:{view_index}")
        if rows:
            out[node.surface_id] = rows[0]
    return out


def _mesh_rest_from_surface(component, surface, view_index: int, binding, policy: DirectionalMotionEvaluatorPolicyV1):
    raster = _surface_raster(surface, view_index)
    projection = projection_for_view(binding, view_index)
    vertices = tuple(component.mesh.vertices)
    vertex_index = {v.canonical_mesh_vertex_id: i for i, v in enumerate(vertices)}
    if len(vertex_index) != len(vertices):
        raise QualificationError("DIRECTIONAL_EVALUATOR_DUPLICATE_MESH_VERTEX")
    rest = []
    residual01 = []
    for vertex in vertices:
        coeffs = tuple(vertex.support_binding.coefficients)
        if not coeffs:
            raise QualificationError("DIRECTIONAL_EVALUATOR_MESH_VERTEX_MISSING_SUPPORT")
        x = y = total = 0.0
        for surface_id, coefficient in coeffs:
            if surface_id not in raster:
                raise QualificationError(f"DIRECTIONAL_EVALUATOR_SUPPORT_NOT_RASTER_BOUND:V{view_index}:{surface_id}")
            c = float(coefficient)
            if not math.isfinite(c) or c < 0.0:
                raise QualificationError("DIRECTIONAL_EVALUATOR_INVALID_SUPPORT_COEFFICIENT")
            x += c * raster[surface_id][0]
            y += c * raster[surface_id][1]
            total += c
        if abs(total - 1.0) > 1e-8:
            raise QualificationError("DIRECTIONAL_EVALUATOR_SUPPORT_SIMPLEX_RESIDUAL")
        xy = (float(x), float(y))
        projected = project_mechanical_point(projection, vertex.P)
        residual01.append(float(np.linalg.norm(np.asarray(xy) - np.asarray(projected))) / max(float(projection.raster_span_px), 1e-12))
        rest.append(xy)
    if residual01 and max(residual01) > float(policy.max_mesh_projection_residual01):
        raise QualificationError(
            f"DIRECTIONAL_EVALUATOR_MESH_PROJECTION_RESIDUAL:V{view_index}:{max(residual01)}"
        )
    faces = []
    for face in component.mesh.faces:
        if len(face) != 3 or any(vertex_id not in vertex_index for vertex_id in face):
            raise QualificationError("DIRECTIONAL_EVALUATOR_REQUIRES_TRIANGULATED_MESH")
        faces.append(tuple(vertex_index[vertex_id] for vertex_id in face))
    if not faces:
        raise QualificationError("DIRECTIONAL_EVALUATOR_EMPTY_MESH")
    return np.asarray(rest, dtype=np.float64), tuple(faces), max(residual01, default=0.0)


def _weights(component, skeleton):
    joint_ids = tuple(sorted(j.canonical_joint_id for j in skeleton.joints))
    joint_index = {jid: i for i, jid in enumerate(joint_ids)}
    vertices = tuple(component.mesh.vertices)
    rows = {row.canonical_mesh_vertex_id: row for row in component.mesh_skin.rows}
    W = np.zeros((len(vertices), len(joint_ids)), dtype=np.float64)
    for vi, vertex in enumerate(vertices):
        row = rows.get(vertex.canonical_mesh_vertex_id)
        if row is None:
            raise QualificationError("DIRECTIONAL_EVALUATOR_MISSING_MESH_SKIN_ROW")
        for jid, weight in row.influences:
            if jid not in joint_index:
                raise QualificationError("DIRECTIONAL_EVALUATOR_UNKNOWN_SKIN_JOINT")
            W[vi, joint_index[jid]] = float(weight)
    if not np.isfinite(W).all() or (W < -1e-8).any() or not np.allclose(W.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
        raise QualificationError("DIRECTIONAL_EVALUATOR_REQUIRES_QUALIFIED_SIMPLEX_SKIN")
    return joint_ids, W


def _topological_skeleton(skeleton):
    by_id = {j.canonical_joint_id: j for j in skeleton.joints}
    if len(by_id) != len(skeleton.joints) or not by_id:
        raise QualificationError("DIRECTIONAL_EVALUATOR_INVALID_SKELETON")
    visiting, done, order = set(), set(), []

    def visit(jid):
        if jid in done:
            return
        if jid in visiting:
            raise QualificationError("DIRECTIONAL_EVALUATOR_SKELETON_CYCLE")
        visiting.add(jid)
        parent = by_id[jid].parent_canonical_id
        if parent is not None:
            if parent not in by_id:
                raise QualificationError("DIRECTIONAL_EVALUATOR_MISSING_PARENT")
            visit(parent)
        visiting.remove(jid)
        done.add(jid)
        order.append(jid)

    for jid in sorted(by_id):
        visit(jid)
    return tuple(order), by_id


def _tracks_for_clip(motion_state, clip_id: str):
    tracks = {}
    for track in motion_state.joint_tracks:
        if track.clip_id != clip_id:
            continue
        if track.canonical_joint_id in tracks:
            raise QualificationError("DIRECTIONAL_EVALUATOR_DUPLICATE_JOINT_TRACK")
        tracks[track.canonical_joint_id] = track
    return tracks


def _assert_rotation_only_scope(product, clip_id: str) -> None:
    if any(track.clip_id == clip_id for track in product.motion_state.order_tracks):
        raise QualificationError("DIRECTIONAL_EVALUATOR_ORDER_TRACK_SCHEMA_NOT_QUALIFIED")
    if any(track.clip_id == clip_id for track in product.motion_state.visibility_tracks):
        raise QualificationError("DIRECTIONAL_EVALUATOR_VISIBILITY_TRACK_SCHEMA_NOT_QUALIFIED")
    for track in product.motion_state.joint_tracks:
        if track.clip_id != clip_id:
            continue
        for key in track.keys:
            if max(abs(float(v)) for v in key.translation_xy) > 1e-12:
                raise QualificationError("DIRECTIONAL_EVALUATOR_TRANSLATION_UNITS_NOT_QUALIFIED")
            if max(abs(float(v) - 1.0) for v in key.scale_xy) > 1e-12:
                raise QualificationError("DIRECTIONAL_EVALUATOR_SCALE_SEMANTICS_NOT_QUALIFIED")
            if abs(float(key.depth_offset)) > 1e-12:
                raise QualificationError("DIRECTIONAL_EVALUATOR_DEPTH_ORDER_SEMANTICS_NOT_QUALIFIED")


def _sample_times(clip, tracks, count: int) -> tuple[float, ...]:
    duration = float(clip.duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise QualificationError("DIRECTIONAL_EVALUATOR_INVALID_CLIP_DURATION")
    times = {float(x) for x in np.linspace(0.0, duration, int(count))}
    for track in tracks.values():
        for key in track.keys:
            t = float(key.time_sec)
            if not math.isfinite(t) or t < -1e-9 or t > duration + 1e-9:
                raise QualificationError("DIRECTIONAL_EVALUATOR_KEY_OUTSIDE_CLIP")
            times.add(min(duration, max(0.0, t)))
    return tuple(sorted(times))


def _rotation_deg(track, time_seconds: float) -> float:
    if track is None or not track.keys:
        return 0.0
    keys = tuple(track.keys)
    if time_seconds <= float(keys[0].time_sec):
        return float(keys[0].rotation_deg)
    if time_seconds >= float(keys[-1].time_sec):
        return float(keys[-1].rotation_deg)
    for a, b in zip(keys[:-1], keys[1:]):
        ta, tb = float(a.time_sec), float(b.time_sec)
        if ta <= time_seconds <= tb:
            if tb <= ta:
                return float(b.rotation_deg)
            u = (time_seconds - ta) / (tb - ta)
            return (1.0 - u) * float(a.rotation_deg) + u * float(b.rotation_deg)
    return float(keys[-1].rotation_deg)


def _T(x: float, y: float) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[0, 2] = float(x); out[1, 2] = float(y)
    return out


def _R(degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    c, s = math.cos(angle), math.sin(angle)
    out = np.eye(3, dtype=np.float64)
    out[0, 0] = c; out[0, 1] = -s
    out[1, 0] = s; out[1, 1] = c
    return out


def _skinning_matrices(product, binding, view_index: int, tracks, time_seconds: float, order, by_id, joint_ids):
    posed = {}
    bind_global = {}
    for jid in order:
        pivot = np.asarray(joint_pivot(binding, view_index, jid), dtype=np.float64)
        parent = by_id[jid].parent_canonical_id
        delta = _R(_rotation_deg(tracks.get(jid), time_seconds))
        bind_global[jid] = _T(*pivot)
        if parent is None:
            posed[jid] = _T(*pivot) @ delta
        else:
            parent_pivot = np.asarray(joint_pivot(binding, view_index, parent), dtype=np.float64)
            posed[jid] = posed[parent] @ _T(*(pivot - parent_pivot)) @ delta
    return np.asarray([posed[jid] @ np.linalg.inv(bind_global[jid]) for jid in joint_ids], dtype=np.float64)


def _apply_lbs_2d(rest: np.ndarray, weights: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    hom = np.concatenate([rest, np.ones((len(rest), 1), dtype=np.float64)], axis=1)
    per_joint = np.stack([(hom @ matrix.T)[:, :2] for matrix in matrices], axis=1)
    out = np.sum(per_joint * weights[:, :, None], axis=1)
    if not np.isfinite(out).all():
        raise QualificationError("DIRECTIONAL_EVALUATOR_NONFINITE_DEFORMATION")
    return out


def evaluate_clip_to_qualification_bake(product, plan, clip, binding: DirectionalJointViewBindingSetIR, *, policy: DirectionalMotionEvaluatorPolicyV1 = DirectionalMotionEvaluatorPolicyV1()):
    policy.validate()
    assert_directional_binding_for_product(product, binding)
    if plan.source_product_state_hash != product.product_state_hash or plan.proof_domain != "MOTION":
        raise QualificationError("DIRECTIONAL_EVALUATOR_PROOF_PLAN_BINDING_MISMATCH")
    if clip.clip_id not in {row.clip_id for row in product.motion_state.clips}:
        raise QualificationError("DIRECTIONAL_EVALUATOR_UNKNOWN_CLIP")
    _assert_rotation_only_scope(product, clip.clip_id)

    skeleton = product.mechanical_state.skeleton
    order, by_id = _topological_skeleton(skeleton)
    tracks = _tracks_for_clip(product.motion_state, clip.clip_id)
    unknown = set(tracks) - set(by_id)
    if unknown:
        raise QualificationError(f"DIRECTIONAL_EVALUATOR_UNKNOWN_TRACK_JOINT:{sorted(unknown)}")
    times = _sample_times(clip, tracks, policy.uniform_sample_count)

    component_rows = []
    rest_by_mesh = {}
    triangles_by_mesh = {}
    render_order = {}
    max_mesh_projection_residual01 = 0.0
    for direction in sorted(product.directional_renderables.directions, key=lambda d: int(d.view_index)):
        view = int(direction.view_index)
        visible_mesh_ids = []
        for component in sorted(direction.components, key=lambda c: (int(c.setup_order), str(c.component_id))):
            mesh_id = f"V{view}:{component.component_id}"
            rest, faces, mesh_residual = _mesh_rest_from_surface(component, product.mechanical_state.surface, view, binding, policy)
            max_mesh_projection_residual01 = max(max_mesh_projection_residual01, float(mesh_residual))
            joint_ids, weights = _weights(component, skeleton)
            component_rows.append((view, mesh_id, rest, faces, joint_ids, weights))
            rest_by_mesh[mesh_id] = tuple(tuple(map(float, xy)) for xy in rest)
            triangles_by_mesh[mesh_id] = faces
            if bool(component.default_visible):
                visible_mesh_ids.append(mesh_id)
        render_order[f"V{view}"] = tuple(visible_mesh_ids)

    evaluator_binding_hash = content_sha256({
        "schema": EVALUATOR_SEMANTIC_VERSION,
        "source_product_state_hash": product.product_state_hash,
        "motion_state_hash": product.motion_state_hash,
        "directional_binding_set_hash": binding.binding_set_hash,
        "proof_plan_hash": plan.proof_plan_hash,
        "clip_id": clip.clip_id,
        "policy_hash": policy.policy_hash,
        "operation_scope": policy.operation_scope,
    })

    frame_rows = []
    for time_seconds in times:
        meshes = {}
        matrix_cache = {}
        for view, mesh_id, rest, _, joint_ids, weights in component_rows:
            key = (view, tuple(joint_ids))
            matrices = matrix_cache.get(key)
            if matrices is None:
                matrices = _skinning_matrices(product, binding, view, tracks, time_seconds, order, by_id, joint_ids)
                matrix_cache[key] = matrices
            deformed = _apply_lbs_2d(rest, weights, matrices)
            meshes[mesh_id] = tuple(tuple(map(float, xy)) for xy in deformed)
        frame_rows.append({
            "time_seconds": float(time_seconds),
            "mesh_vertices_by_id": meshes,
            "render_order_by_view": render_order,
            "metadata": {
                "directional_binding_set_hash": binding.binding_set_hash,
                "evaluator_policy_hash": policy.policy_hash,
                "operation_scope": policy.operation_scope,
            },
        })

    fps = float(max(1, len(times) - 1)) / float(clip.duration_sec)
    return bind_qualification_owned_motion_bake(
        source_product_state_hash=product.product_state_hash,
        proof_plan_hash=plan.proof_plan_hash,
        clip_id=clip.clip_id,
        duration_seconds=float(clip.duration_sec),
        fps=fps,
        loop=bool(clip.loop),
        evaluator_semantic_version=EVALUATOR_SEMANTIC_VERSION,
        evaluator_binding_hash=evaluator_binding_hash,
        sampling_policy=f"UNIFORM_{policy.uniform_sample_count}_PLUS_AUTHORED_KEYS",
        rest_mesh_vertices_by_id=rest_by_mesh,
        triangles_by_mesh_id=triangles_by_mesh,
        frame_rows=tuple(frame_rows),
        metadata={
            "directional_binding_set_hash": binding.binding_set_hash,
            "evaluator_policy_hash": policy.policy_hash,
            "operation_scope": policy.operation_scope,
            "mechanical_xy_used_as_raster_xy": False,
            "mesh_rest_authority": "SURFACE_SUPPORT_BINDING_TO_ADMITTED_TARGET_RASTER",
            "joint_pivot_authority": "QUALIFIED_AFFINE_P_TO_RASTER_BINDING",
            "max_mesh_projection_residual01": max_mesh_projection_residual01,
            "order_visibility_semantics": "DEFAULT_COMPONENT_STATE_ONLY",
        },
    )


def make_qualified_motion_bake_provider(binding: DirectionalJointViewBindingSetIR, *, policy: DirectionalMotionEvaluatorPolicyV1 = DirectionalMotionEvaluatorPolicyV1()):
    """Compatibility alias; returns the typed hashed provider authority, not a closure."""
    from .directional_motion_provider import make_qualified_directional_motion_provider
    return make_qualified_directional_motion_provider(binding, policy=policy)
