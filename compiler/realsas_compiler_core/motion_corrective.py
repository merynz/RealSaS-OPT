from __future__ import annotations

"""Recovered bounded corrective-deformation primitives for motion polish.

These operations are compiler-owned post-LBS deformation intents inspired by the
historical SaS v18 deformer layer (Warp/Rotation/SquashStretch/CageHandle/
VelocityLag). They operate only on an explicit qualified 2D vertex set supplied by
the caller and never create mesh topology, weights, ownership or new art pixels.

This module is not yet wired into the current Mage PRODUCT_PASS path. It exists as
a deterministic, testable recovery target for a later qualification-owned
post-deformation bake stage.
"""

from dataclasses import dataclass, field, asdict
import math
from typing import Sequence

from .hashing import content_sha256


Point2 = tuple[float, float]
SCHEMA = "RealSaS.CorrectiveDeformerStack.v1"
PRODUCER = "RealSaS.MotionEngine.HistoricalCorrectiveRecovery.v1"


def _finite_point(p: Point2) -> bool:
    return math.isfinite(float(p[0])) and math.isfinite(float(p[1]))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(float(hi), max(float(lo), float(value)))


@dataclass(frozen=True)
class CorrectiveDeformerSpec:
    deformer_id: str
    kind: str
    center_xy: Point2
    radius: float
    strength: float
    axis_xy: Point2 = (1.0, 0.0)
    parameters: dict = field(default_factory=dict)
    source_authority_hash: str = ""
    schema_version: str = "RealSaS.CorrectiveDeformerSpec.v1"

    def validate(self) -> None:
        if not self.deformer_id or not self.kind:
            raise ValueError("corrective deformer requires id and kind")
        if not _finite_point(self.center_xy) or not _finite_point(self.axis_xy):
            raise ValueError("corrective deformer contains nonfinite coordinate")
        if not (math.isfinite(float(self.radius)) and float(self.radius) > 0.0):
            raise ValueError("corrective deformer radius invalid")
        if not math.isfinite(float(self.strength)):
            raise ValueError("corrective deformer strength invalid")
        if abs(float(self.strength)) > 2.0:
            raise ValueError("corrective deformer strength exceeds bounded recovery policy")
        if not self.source_authority_hash:
            raise ValueError("corrective deformer requires source authority hash")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CorrectiveDeformerStackIR:
    stack_id: str
    mesh_lineage_hash: str
    deformers: tuple[CorrectiveDeformerSpec, ...]
    stack_hash: str
    schema_version: str = SCHEMA
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CorrectiveApplicationReport:
    stack_hash: str
    vertex_count: int
    affected_vertex_count: int
    max_offset: float
    finite: bool
    passed: bool
    report_hash: str
    schema_version: str = "RealSaS.CorrectiveApplicationReport.v1"

    def to_dict(self) -> dict:
        return asdict(self)


def build_corrective_stack(
    *, stack_id: str, mesh_lineage_hash: str, deformers: Sequence[CorrectiveDeformerSpec], metadata: dict | None = None,
) -> CorrectiveDeformerStackIR:
    rows = tuple(deformers)
    if not stack_id or not mesh_lineage_hash or not rows:
        raise ValueError("corrective stack requires id, mesh lineage and deformer")
    for row in rows:
        row.validate()
    payload = {
        "schema": SCHEMA,
        "stack_id": str(stack_id),
        "mesh_lineage_hash": str(mesh_lineage_hash),
        "deformers": [row.to_dict() for row in rows],
        "metadata": dict(metadata or {}),
    }
    return CorrectiveDeformerStackIR(
        str(stack_id), str(mesh_lineage_hash), rows, content_sha256(payload), metadata=dict(metadata or {})
    )


def _falloff(point: Point2, center: Point2, radius: float) -> float:
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    d = math.hypot(dx, dy)
    if d >= float(radius):
        return 0.0
    u = 1.0 - d / float(radius)
    return u * u * (3.0 - 2.0 * u)


def _normalize(v: Point2) -> Point2:
    length = math.hypot(float(v[0]), float(v[1]))
    if length <= 1e-12:
        return (1.0, 0.0)
    return (float(v[0]) / length, float(v[1]) / length)


def _rotate_about(point: Point2, center: Point2, degrees: float) -> Point2:
    a = math.radians(float(degrees))
    c, s = math.cos(a), math.sin(a)
    x, y = float(point[0]) - float(center[0]), float(point[1]) - float(center[1])
    return (float(center[0]) + c * x - s * y, float(center[1]) + s * x + c * y)


def _apply_rotation(point: Point2, spec: CorrectiveDeformerSpec, drive: float) -> Point2:
    w = _falloff(point, spec.center_xy, spec.radius)
    if w <= 0.0:
        return point
    angle = float(spec.parameters.get("max_rotation_deg", 12.0)) * float(spec.strength) * float(drive) * w
    return _rotate_about(point, spec.center_xy, angle)


def _apply_squash_stretch(point: Point2, spec: CorrectiveDeformerSpec, drive: float) -> Point2:
    w = _falloff(point, spec.center_xy, spec.radius)
    if w <= 0.0:
        return point
    axis = _normalize(spec.axis_xy)
    ortho = (-axis[1], axis[0])
    delta = (float(point[0]) - spec.center_xy[0], float(point[1]) - spec.center_xy[1])
    along = delta[0] * axis[0] + delta[1] * axis[1]
    across = delta[0] * ortho[0] + delta[1] * ortho[1]
    amount = _clamp(float(spec.strength) * float(drive) * w, -0.65, 0.65)
    along *= 1.0 + amount
    across *= 1.0 - 0.55 * amount
    return (
        spec.center_xy[0] + along * axis[0] + across * ortho[0],
        spec.center_xy[1] + along * axis[1] + across * ortho[1],
    )


def _apply_warp(point: Point2, spec: CorrectiveDeformerSpec, drive: float) -> Point2:
    w = _falloff(point, spec.center_xy, spec.radius)
    if w <= 0.0:
        return point
    axis = _normalize(spec.axis_xy)
    max_offset = float(spec.parameters.get("max_offset", 0.04))
    offset = max_offset * float(spec.strength) * float(drive) * w
    return (float(point[0]) + axis[0] * offset, float(point[1]) + axis[1] * offset)


def _apply_cage_handle(point: Point2, spec: CorrectiveDeformerSpec, drive: float) -> Point2:
    handle = spec.parameters.get("handle_delta_xy", (0.0, 0.0))
    hx, hy = float(handle[0]), float(handle[1])
    w = _falloff(point, spec.center_xy, spec.radius)
    scale = float(spec.strength) * float(drive) * w
    return (float(point[0]) + hx * scale, float(point[1]) + hy * scale)


def _apply_velocity_lag(point: Point2, spec: CorrectiveDeformerSpec, drive: float) -> Point2:
    velocity = spec.parameters.get("source_velocity_xy", (0.0, 0.0))
    lag_seconds = _clamp(float(spec.parameters.get("lag_seconds", 0.045)), 0.0, 0.25)
    w = _falloff(point, spec.center_xy, spec.radius)
    return (
        float(point[0]) - float(velocity[0]) * lag_seconds * float(spec.strength) * float(drive) * w,
        float(point[1]) - float(velocity[1]) * lag_seconds * float(spec.strength) * float(drive) * w,
    )


def apply_corrective_stack(
    stack: CorrectiveDeformerStackIR,
    *,
    mesh_lineage_hash: str,
    vertices: Sequence[Sequence[float]],
    drive_by_deformer_id: dict[str, float] | None = None,
    max_allowed_offset: float = 0.35,
) -> tuple[tuple[Point2, ...], CorrectiveApplicationReport]:
    if str(mesh_lineage_hash) != stack.mesh_lineage_hash:
        raise ValueError("corrective stack mesh lineage drift")
    source = tuple((float(p[0]), float(p[1])) for p in vertices)
    if not source or not all(_finite_point(p) for p in source):
        raise ValueError("corrective stack requires finite vertices")
    if not (0.0 < float(max_allowed_offset) <= 2.0):
        raise ValueError("corrective max_allowed_offset invalid")
    drives = dict(drive_by_deformer_id or {})
    out = list(source)
    affected = set()

    dispatch = {
        "ROTATION": _apply_rotation,
        "SQUASH_STRETCH": _apply_squash_stretch,
        "WARP": _apply_warp,
        "CAGE_HANDLE": _apply_cage_handle,
        "VELOCITY_LAG": _apply_velocity_lag,
    }
    for spec in stack.deformers:
        spec.validate()
        kind = str(spec.kind).strip().upper()
        if kind not in dispatch:
            raise ValueError(f"unsupported corrective deformer kind: {spec.kind}")
        drive = _clamp(float(drives.get(spec.deformer_id, 1.0)), -1.0, 1.0)
        fn = dispatch[kind]
        for index, point in enumerate(tuple(out)):
            updated = fn(point, spec, drive)
            if not _finite_point(updated):
                raise ValueError("corrective deformer emitted nonfinite vertex")
            if math.hypot(updated[0] - source[index][0], updated[1] - source[index][1]) > 1e-12:
                affected.add(index)
            out[index] = updated

    offsets = tuple(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(out, source))
    max_offset = max(offsets, default=0.0)
    finite = all(_finite_point(p) for p in out)
    passed = finite and max_offset <= float(max_allowed_offset) + 1e-10
    provisional = {
        "schema_version": "RealSaS.CorrectiveApplicationReport.v1",
        "stack_hash": stack.stack_hash,
        "vertex_count": len(source),
        "affected_vertex_count": len(affected),
        "max_offset": max_offset,
        "finite": finite,
        "passed": passed,
    }
    report = CorrectiveApplicationReport(
        stack.stack_hash, len(source), len(affected), float(max_offset), bool(finite), bool(passed), content_sha256(provisional)
    )
    if not passed:
        raise ValueError(f"corrective deformation exceeded bounded policy: {max_offset}")
    return tuple(out), report


__all__ = [
    "CorrectiveApplicationReport", "CorrectiveDeformerSpec", "CorrectiveDeformerStackIR", "PRODUCER",
    "apply_corrective_stack", "build_corrective_stack",
]
