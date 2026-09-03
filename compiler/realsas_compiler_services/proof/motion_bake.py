from __future__ import annotations

"""Qualification-owned directional motion-frame evidence.

This module validates and binds frames produced by a separately qualified
current directional evaluator. It never derives a view-space pivot, runs LBS,
or treats mechanical Vec3 / directional mesh P.xy as a shared coordinate frame.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256

Json = dict[str, Any]
Point2 = tuple[float, float]


@dataclass(frozen=True)
class QualificationOwnedMotionFrameIR:
    time_seconds: float
    mesh_vertices_by_id: tuple[tuple[str, tuple[Point2, ...]], ...]
    render_order_by_view: tuple[tuple[str, tuple[str, ...]], ...]
    semantic_sha256: str
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.QualificationOwnedMotionFrameIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class QualificationOwnedMotionBakeIR:
    source_product_state_hash: str
    proof_plan_hash: str
    clip_id: str
    duration_seconds: float
    fps: float
    loop: bool
    evaluator_semantic_version: str
    evaluator_binding_hash: str
    sampling_policy: str
    rest_mesh_vertices_by_id: tuple[tuple[str, tuple[Point2, ...]], ...]
    triangles_by_mesh_id: tuple[tuple[str, tuple[tuple[int, int, int], ...]], ...]
    frames: tuple[QualificationOwnedMotionFrameIR, ...]
    bake_hash: str
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.QualificationOwnedMotionBakeIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


def _points(rows: Iterable[Iterable[float]]) -> tuple[Point2, ...]:
    out = tuple((float(row[0]), float(row[1])) for row in rows)
    if not out:
        raise ValueError("MOTION_BAKE_EMPTY_MESH")
    return out


def _meshes(value: Mapping[str, Iterable[Iterable[float]]]):
    rows = tuple((str(mid), _points(points)) for mid, points in sorted(value.items()))
    if not rows:
        raise ValueError("MOTION_BAKE_REQUIRES_MESH")
    return rows


def _orders(value: Mapping[str, Iterable[str]]):
    return tuple((str(view), tuple(map(str, order))) for view, order in sorted(value.items()))


def _triangles(value: Mapping[str, Iterable[Iterable[int]]]):
    rows = []
    for mid, triangles in sorted(value.items()):
        tris = tuple(tuple(map(int, tri)) for tri in triangles)
        if any(len(tri) != 3 for tri in tris):
            raise ValueError("MOTION_BAKE_REQUIRES_TRIANGLES")
        rows.append((str(mid), tris))
    return tuple(rows)


def _frame_hash(product_hash: str, plan_hash: str, clip_id: str, evaluator_hash: str, t: float, meshes, orders):
    return content_sha256({
        "source_product_state_hash": product_hash,
        "proof_plan_hash": plan_hash,
        "clip_id": clip_id,
        "evaluator_binding_hash": evaluator_hash,
        "time_seconds": float(t),
        "mesh_vertices_by_id": meshes,
        "render_order_by_view": orders,
    })


def bind_qualification_owned_motion_bake(*, source_product_state_hash: str, proof_plan_hash: str, clip_id: str,
        duration_seconds: float, fps: float, loop: bool, evaluator_semantic_version: str,
        evaluator_binding_hash: str, sampling_policy: str, rest_mesh_vertices_by_id,
        triangles_by_mesh_id, frame_rows, metadata: Mapping[str, Any] | None = None) -> QualificationOwnedMotionBakeIR:
    if not source_product_state_hash or not proof_plan_hash:
        raise ValueError("MOTION_BAKE_REQUIRES_EXACT_BINDING")
    if duration_seconds <= 0 or fps <= 0 or not clip_id or not evaluator_semantic_version or not evaluator_binding_hash:
        raise ValueError("MOTION_BAKE_INVALID_EVALUATOR_OR_CLIP")
    rest = _meshes(rest_mesh_vertices_by_id)
    counts = {mid: len(points) for mid, points in rest}
    triangles = _triangles(triangles_by_mesh_id)
    for mid, tris in triangles:
        if mid not in counts or any(min(tri) < 0 or max(tri) >= counts[mid] for tri in tris):
            raise ValueError("MOTION_BAKE_TRIANGLE_OUT_OF_RANGE")
    frames = []
    previous = None
    for raw in frame_rows:
        t = float(raw["time_seconds"])
        if previous is not None and t <= previous:
            raise ValueError("MOTION_BAKE_TIMES_NOT_STRICT")
        previous = t
        meshes = _meshes(dict(raw.get("mesh_vertices_by_id") or {}))
        if {mid: len(points) for mid, points in meshes} != counts:
            raise ValueError("MOTION_BAKE_MESH_IDENTITY_OR_CARDINALITY_DRIFT")
        orders = _orders(dict(raw.get("render_order_by_view") or {}))
        semantic = _frame_hash(source_product_state_hash, proof_plan_hash, clip_id, evaluator_binding_hash, t, meshes, orders)
        frames.append(QualificationOwnedMotionFrameIR(t, meshes, orders, semantic, {"qualification_owned": True, "export_solver_replay_forbidden": True, **dict(raw.get("metadata") or {})}))
    if not frames:
        raise ValueError("MOTION_BAKE_REQUIRES_FRAME")
    provisional = QualificationOwnedMotionBakeIR(
        str(source_product_state_hash), str(proof_plan_hash), str(clip_id), float(duration_seconds), float(fps), bool(loop),
        str(evaluator_semantic_version), str(evaluator_binding_hash), str(sampling_policy), rest, triangles, tuple(frames), "",
        {"authority": "PROOF_MEASUREMENT_SIBLING_DERIVED", "frames_generated_by_this_binder": False,
         "same_frames_required_for_export": True, "export_solver_replay_forbidden": True, **dict(metadata or {})},
    )
    payload = provisional.to_dict(); payload.pop("bake_hash", None)
    return replace(provisional, bake_hash=content_sha256(payload))


def assert_motion_bake_binding(bake: QualificationOwnedMotionBakeIR, *, source_product_state_hash: str, proof_plan_hash: str) -> None:
    if bake.source_product_state_hash != str(source_product_state_hash):
        raise ValueError("STALE_MOTION_BAKE_PRODUCT_BINDING")
    if bake.proof_plan_hash != str(proof_plan_hash):
        raise ValueError("STALE_MOTION_BAKE_PROOF_PLAN_BINDING")
    payload = bake.to_dict(); claimed = payload.pop("bake_hash")
    if content_sha256(payload) != claimed:
        raise ValueError("MOTION_BAKE_HASH_MISMATCH")


def deploy_frame_dicts(bake: QualificationOwnedMotionBakeIR) -> tuple[Json, ...]:
    return tuple({
        "time_seconds": frame.time_seconds,
        "semantic_sha256": frame.semantic_sha256,
        "mesh_vertices_by_id": {mid: points for mid, points in frame.mesh_vertices_by_id},
        "render_order_by_view": {view: order for view, order in frame.render_order_by_view},
    } for frame in bake.frames)
