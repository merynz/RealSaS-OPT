from __future__ import annotations

"""Recovered deterministic 2D XPBD secondary-motion kernel.

This is a compiler/runtime-neutral recovery of the historical SaS v18 soft-chain
semantics: substeps, compliance, damping, stretch/bend constraints, root lock,
velocity/acceleration inheritance and a bounded maximum offset.

It does NOT infer soft parts and it does NOT mutate canonical mesh/rig/skin. A
caller must provide an explicit qualified chain and rest geometry. Product use is
forbidden until a component adapter and proof bind the chain to current authority.
"""

from dataclasses import dataclass, field, asdict
import math
from typing import Sequence

from .hashing import content_sha256


SCHEMA = "RealSaS.SecondaryXPBDChain.v1"
PRODUCER = "RealSaS.MotionEngine.HistoricalXPBDRecovery.v1"

Point2 = tuple[float, float]


def _vadd(a: Point2, b: Point2) -> Point2:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]))


def _vsub(a: Point2, b: Point2) -> Point2:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _vmul(a: Point2, s: float) -> Point2:
    return (float(a[0]) * float(s), float(a[1]) * float(s))


def _norm(a: Point2) -> float:
    return math.hypot(float(a[0]), float(a[1]))


def _finite_point(p: Point2) -> bool:
    return math.isfinite(float(p[0])) and math.isfinite(float(p[1]))


@dataclass(frozen=True)
class SecondaryXPBDPolicy:
    substeps: int = 4
    solver_iterations: int = 5
    compliance: float = 2.5e-5
    damping: float = 0.10
    stretch_stiffness: float = 1.0
    bend_stiffness: float = 0.35
    velocity_inheritance: float = 0.70
    acceleration_inheritance: float = 0.18
    max_offset: float = 0.22
    root_locked: bool = True
    schema_version: str = "RealSaS.SecondaryXPBDPolicy.v1"

    def validate(self) -> None:
        if not (1 <= int(self.substeps) <= 32):
            raise ValueError("XPBD substeps out of range")
        if not (1 <= int(self.solver_iterations) <= 64):
            raise ValueError("XPBD solver_iterations out of range")
        if float(self.compliance) < 0.0 or not math.isfinite(float(self.compliance)):
            raise ValueError("XPBD compliance invalid")
        for name in ("damping", "stretch_stiffness", "bend_stiffness", "velocity_inheritance", "acceleration_inheritance"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"XPBD {name} must be in [0,1]")
        if not (0.0 < float(self.max_offset) <= 4.0):
            raise ValueError("XPBD max_offset invalid")

    @property
    def policy_hash(self) -> str:
        self.validate()
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class SecondaryChainIR:
    chain_id: str
    component_id: str
    parent_joint_id: str
    rest_points: tuple[Point2, ...]
    policy: SecondaryXPBDPolicy
    chain_hash: str
    source_authority_hash: str
    schema_version: str = SCHEMA
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SecondaryFrameReport:
    frame_index: int
    max_offset: float
    max_segment_error: float
    root_error: float
    finite: bool
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SecondarySimulationReport:
    chain_hash: str
    policy_hash: str
    frame_reports: tuple[SecondaryFrameReport, ...]
    max_offset: float
    max_segment_error: float
    max_root_error: float
    passed: bool
    report_hash: str
    schema_version: str = "RealSaS.SecondaryXPBDSimulationReport.v1"

    def to_dict(self) -> dict:
        return asdict(self)


def build_secondary_chain(
    *,
    chain_id: str,
    component_id: str,
    parent_joint_id: str,
    rest_points: Sequence[Sequence[float]],
    source_authority_hash: str,
    policy: SecondaryXPBDPolicy = SecondaryXPBDPolicy(),
    metadata: dict | None = None,
) -> SecondaryChainIR:
    policy.validate()
    points = tuple((float(p[0]), float(p[1])) for p in rest_points)
    if len(points) < 3:
        raise ValueError("secondary chain requires at least 3 points")
    if not all(_finite_point(p) for p in points):
        raise ValueError("secondary chain contains nonfinite rest point")
    if any(_norm(_vsub(b, a)) <= 1e-9 for a, b in zip(points, points[1:])):
        raise ValueError("secondary chain contains zero-length segment")
    if not chain_id or not component_id or not parent_joint_id or not source_authority_hash:
        raise ValueError("secondary chain requires exact typed authority binding")
    payload = {
        "schema": SCHEMA,
        "chain_id": str(chain_id),
        "component_id": str(component_id),
        "parent_joint_id": str(parent_joint_id),
        "rest_points": points,
        "policy_hash": policy.policy_hash,
        "source_authority_hash": str(source_authority_hash),
        "metadata": dict(metadata or {}),
    }
    return SecondaryChainIR(
        str(chain_id), str(component_id), str(parent_joint_id), points, policy,
        content_sha256(payload), str(source_authority_hash), metadata=dict(metadata or {}),
    )


def _distance_project(
    positions: list[Point2], inv_mass: list[float], i: int, j: int, rest_length: float,
    compliance: float, dt: float, stiffness: float,
) -> None:
    delta = _vsub(positions[j], positions[i])
    length = _norm(delta)
    if length <= 1e-12:
        return
    c = length - float(rest_length)
    wi, wj = float(inv_mass[i]), float(inv_mass[j])
    wsum = wi + wj
    if wsum <= 0.0:
        return
    alpha = float(compliance) / max(float(dt) * float(dt), 1e-12)
    lam = -c / (wsum + alpha)
    n = _vmul(delta, 1.0 / length)
    correction = _vmul(n, lam * float(stiffness))
    if wi > 0.0:
        positions[i] = _vsub(positions[i], _vmul(correction, wi))
    if wj > 0.0:
        positions[j] = _vadd(positions[j], _vmul(correction, wj))


def _bend_project(
    positions: list[Point2], inv_mass: list[float], i: int, k: int, rest_chord: float,
    compliance: float, dt: float, stiffness: float,
) -> None:
    _distance_project(positions, inv_mass, i, k, rest_chord, compliance * 2.0, dt, stiffness)


def _clamp_offset(point: Point2, rest: Point2, max_offset: float) -> Point2:
    delta = _vsub(point, rest)
    length = _norm(delta)
    if length <= float(max_offset) or length <= 1e-12:
        return point
    return _vadd(rest, _vmul(delta, float(max_offset) / length))


def simulate_secondary_chain(
    chain: SecondaryChainIR,
    *,
    root_positions: Sequence[Sequence[float]],
    frame_dt: float,
    external_accelerations: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[tuple[Point2, ...], ...], SecondarySimulationReport]:
    """Simulate one explicit chain over the supplied root path.

    The root path is caller-owned qualified motion evidence. No root motion is
    invented. Returned coordinates stay in the same caller-defined 2D space.
    """
    policy = chain.policy
    policy.validate()
    dt_frame = float(frame_dt)
    if not (math.isfinite(dt_frame) and dt_frame > 0.0):
        raise ValueError("secondary frame_dt must be positive finite")
    roots = tuple((float(p[0]), float(p[1])) for p in root_positions)
    if len(roots) < 2 or not all(_finite_point(p) for p in roots):
        raise ValueError("secondary simulation requires >=2 finite root samples")
    if external_accelerations is None:
        external = tuple((0.0, 0.0) for _ in roots)
    else:
        external = tuple((float(p[0]), float(p[1])) for p in external_accelerations)
        if len(external) != len(roots) or not all(_finite_point(p) for p in external):
            raise ValueError("secondary external acceleration samples invalid")

    rest = tuple(chain.rest_points)
    base_root = rest[0]
    rel = tuple(_vsub(p, base_root) for p in rest)
    rest_lengths = tuple(_norm(_vsub(b, a)) for a, b in zip(rest, rest[1:]))
    bend_chords = tuple(_norm(_vsub(rest[i + 2], rest[i])) for i in range(len(rest) - 2))
    inv_mass = [0.0 if i == 0 and policy.root_locked else 1.0 for i in range(len(rest))]

    positions = [_vadd(roots[0], r) for r in rel]
    previous = list(positions)
    frames: list[tuple[Point2, ...]] = []
    reports: list[SecondaryFrameReport] = []
    previous_root = roots[0]
    previous_root_velocity = (0.0, 0.0)

    for frame_index, (root, ext_acc) in enumerate(zip(roots, external)):
        root_velocity = _vmul(_vsub(root, previous_root), 1.0 / dt_frame) if frame_index else (0.0, 0.0)
        root_acc = _vmul(_vsub(root_velocity, previous_root_velocity), 1.0 / dt_frame) if frame_index else (0.0, 0.0)
        sub_dt = dt_frame / int(policy.substeps)

        for _substep in range(int(policy.substeps)):
            if policy.root_locked:
                positions[0] = root
                previous[0] = root
            for i in range(1 if policy.root_locked else 0, len(positions)):
                velocity = _vmul(_vsub(positions[i], previous[i]), 1.0 - float(policy.damping))
                inherited_v = _vmul(root_velocity, float(policy.velocity_inheritance) * sub_dt)
                inherited_a = _vmul(_vadd(root_acc, ext_acc), float(policy.acceleration_inheritance) * sub_dt * sub_dt)
                old = positions[i]
                predicted = _vadd(_vadd(_vadd(positions[i], velocity), inherited_v), inherited_a)
                previous[i] = old
                positions[i] = predicted

            for _iteration in range(int(policy.solver_iterations)):
                if policy.root_locked:
                    positions[0] = root
                for i, length in enumerate(rest_lengths):
                    _distance_project(
                        positions, inv_mass, i, i + 1, length,
                        float(policy.compliance), sub_dt, float(policy.stretch_stiffness),
                    )
                for i, chord in enumerate(bend_chords):
                    _bend_project(
                        positions, inv_mass, i, i + 2, chord,
                        float(policy.compliance), sub_dt, float(policy.bend_stiffness),
                    )
                if policy.root_locked:
                    positions[0] = root

            frame_rest = tuple(_vadd(root, r) for r in rel)
            for i in range(1 if policy.root_locked else 0, len(positions)):
                positions[i] = _clamp_offset(positions[i], frame_rest[i], float(policy.max_offset))

        frame = tuple(positions)
        frame_rest = tuple(_vadd(root, r) for r in rel)
        offsets = tuple(_norm(_vsub(a, b)) for a, b in zip(frame, frame_rest))
        segment_errors = tuple(abs(_norm(_vsub(frame[i + 1], frame[i])) - rest_lengths[i]) for i in range(len(rest_lengths)))
        root_error = _norm(_vsub(frame[0], root))
        finite = all(_finite_point(p) for p in frame)
        max_offset = max(offsets, default=0.0)
        max_segment_error = max(segment_errors, default=0.0)
        passed = (
            finite
            and max_offset <= float(policy.max_offset) + 1e-8
            and (not policy.root_locked or root_error <= 1e-9)
            and max_segment_error <= max(0.025, 0.20 * max(rest_lengths))
        )
        reports.append(SecondaryFrameReport(frame_index, max_offset, max_segment_error, root_error, finite, bool(passed)))
        frames.append(frame)
        previous_root = root
        previous_root_velocity = root_velocity

    provisional = {
        "schema_version": "RealSaS.SecondaryXPBDSimulationReport.v1",
        "chain_hash": chain.chain_hash,
        "policy_hash": policy.policy_hash,
        "frame_reports": [r.to_dict() for r in reports],
        "max_offset": max((r.max_offset for r in reports), default=0.0),
        "max_segment_error": max((r.max_segment_error for r in reports), default=0.0),
        "max_root_error": max((r.root_error for r in reports), default=0.0),
        "passed": all(r.passed for r in reports),
    }
    report = SecondarySimulationReport(
        chain.chain_hash, policy.policy_hash, tuple(reports),
        float(provisional["max_offset"]), float(provisional["max_segment_error"]),
        float(provisional["max_root_error"]), bool(provisional["passed"]), content_sha256(provisional),
    )
    return tuple(frames), report


__all__ = [
    "PRODUCER", "SecondaryChainIR", "SecondaryFrameReport", "SecondarySimulationReport",
    "SecondaryXPBDPolicy", "build_secondary_chain", "simulate_secondary_chain",
]
