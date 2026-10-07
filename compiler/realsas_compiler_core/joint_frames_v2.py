from __future__ import annotations

"""Post-bind deterministic joint frames with generic coincident-control handling.

Coincident parent/child positions do not define an orientation. A coincident
control may inherit its parent's frame only when its entire skeleton subtree has
zero mechanical influence in the final carrier-native W_M. Otherwise the Compiler
fails closed and AXIS must expose explicit orientation/tail evidence.

No joint name, raw source index, subject id, or invented epsilon participates.
"""

from dataclasses import dataclass
import numpy as np

from .carrier_skin_v1 import carrier_joint_influence_vectors
from .hashing import content_sha256
from .joint_frames_v1 import DerivedJointFrameV1, object_basis_from_camera_set
from .types import QualificationError


@dataclass(frozen=True)
class CoincidentFrameEventV2:
    joint_id: str
    parent_id: str
    subtree_weight_total: float
    subtree_weight_mean: float
    subtree_weight_max_vertex: float
    resolution: str

    def to_dict(self) -> dict:
        return {
            "joint_id": self.joint_id,
            "parent_id": self.parent_id,
            "subtree_weight_total": self.subtree_weight_total,
            "subtree_weight_mean": self.subtree_weight_mean,
            "subtree_weight_max_vertex": self.subtree_weight_max_vertex,
            "resolution": self.resolution,
        }


def _unit(value, code):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise QualificationError(code)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise QualificationError(code)
    return v / n


def _direct_frame(*, pos, parent_pos, gr, gu, gf):
    y = _unit(np.asarray(pos) - np.asarray(parent_pos), "JOINT_FRAME_V2_BONE_DEGENERATE")
    z = gf - y * float(np.dot(gf, y))
    if float(np.linalg.norm(z)) <= 1e-8:
        z = gu - y * float(np.dot(gu, y))
    if float(np.linalg.norm(z)) <= 1e-8:
        z = gr - y * float(np.dot(gr, y))
    z = _unit(z, "JOINT_FRAME_V2_SECONDARY_AXIS_DEGENERATE")
    x = _unit(np.cross(y, z), "JOINT_FRAME_V2_PRIMARY_AXIS_DEGENERATE")
    if float(np.dot(x, gr)) < 0.0:
        x = -x
        z = -z
    z = _unit(np.cross(x, y), "JOINT_FRAME_V2_ORTHONORMALIZATION_FAIL")
    R = np.column_stack((x, y, z))
    if np.linalg.det(R) < 0.999999:
        raise QualificationError("JOINT_FRAME_V2_NOT_RIGHT_HANDED")
    return R


def derive_joint_frames_post_bind_v2(
    skeleton,
    *,
    carrier_skin,
    cameras,
    inactive_subtree_mean_weight_max: float = 1e-12,
    inactive_subtree_max_vertex_weight_max: float = 1e-10,
    coincident_position_tolerance: float = 1e-12,
):
    joints = {str(j.canonical_joint_id): j for j in skeleton.joints}
    if len(joints) != len(skeleton.joints) or skeleton.root_id not in joints:
        raise QualificationError("JOINT_FRAME_V2_SKELETON_INVALID")
    if carrier_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("JOINT_FRAME_V2_SKIN_SKELETON_DRIFT")

    parent = {
        jid: (
            None
            if joints[jid].parent_canonical_id is None
            else str(joints[jid].parent_canonical_id)
        )
        for jid in joints
    }
    children = {jid: [] for jid in joints}
    for jid, pid in parent.items():
        if pid is not None:
            if pid not in joints:
                raise QualificationError("JOINT_FRAME_V2_PARENT_UNKNOWN")
            children[pid].append(jid)
    roots = [jid for jid, pid in parent.items() if pid is None]
    if roots != [str(skeleton.root_id)]:
        raise QualificationError("JOINT_FRAME_V2_ROOT_DRIFT")

    order: list[str] = []
    queue = [str(skeleton.root_id)]
    while queue:
        jid = queue.pop(0)
        order.append(jid)
        queue.extend(sorted(children[jid]))
    if len(order) != len(joints):
        raise QualificationError("JOINT_FRAME_V2_DISCONNECTED_OR_CYCLIC")

    direct_vectors = carrier_joint_influence_vectors(carrier_skin)
    vertex_count = len(carrier_skin.rows)
    subtree_vectors: dict[str, np.ndarray] = {}
    for jid in reversed(order):
        vector = np.asarray(
            direct_vectors.get(jid, np.zeros(vertex_count, dtype=np.float64)),
            dtype=np.float64,
        ).copy()
        for child in children[jid]:
            vector += subtree_vectors[child]
        subtree_vectors[jid] = vector

    gr, gu, gf = object_basis_from_camera_set(cameras)
    frames: dict[str, DerivedJointFrameV1] = {}
    events: list[CoincidentFrameEventV2] = []
    for jid in order:
        joint = joints[jid]
        pos = np.asarray(joint.position, dtype=np.float64)
        pid = parent[jid]
        if pid is None:
            R = np.column_stack((gr, gu, gf))
        else:
            parent_pos = np.asarray(joints[pid].position, dtype=np.float64)
            distance = float(np.linalg.norm(pos - parent_pos))
            if distance > float(coincident_position_tolerance):
                R = _direct_frame(
                    pos=pos, parent_pos=parent_pos, gr=gr, gu=gu, gf=gf
                )
            else:
                vector = subtree_vectors[jid]
                total = float(vector.sum())
                mean = float(vector.mean()) if len(vector) else 0.0
                max_vertex = float(vector.max()) if len(vector) else 0.0
                inactive = (
                    mean <= float(inactive_subtree_mean_weight_max)
                    and max_vertex <= float(inactive_subtree_max_vertex_weight_max)
                )
                if not inactive:
                    raise QualificationError(
                        "ACTIVE_COINCIDENT_CONTROL_REQUIRES_AXIS_ORIENTATION_EXTENSION:"
                        f"{jid}:{pid}:mean={mean}:max={max_vertex}"
                    )
                if pid not in frames:
                    raise QualificationError("JOINT_FRAME_V2_PARENT_FRAME_UNAVAILABLE")
                R = np.asarray(frames[pid].rotation_matrix, dtype=np.float64)
                events.append(
                    CoincidentFrameEventV2(
                        joint_id=jid,
                        parent_id=pid,
                        subtree_weight_total=total,
                        subtree_weight_mean=mean,
                        subtree_weight_max_vertex=max_vertex,
                        resolution="INHERIT_PARENT_FRAME__MECHANICALLY_UNOBSERVABLE_SUBTREE",
                    )
                )
        payload = {
            "schema": "RealSaS.DerivedJointFrame.v2",
            "joint_id": jid,
            "parent_id": pid,
            "rest_position": tuple(map(float, pos)),
            "rotation_matrix": tuple(
                tuple(map(float, row)) for row in np.asarray(R)
            ),
            "carrier_skin_lineage_hash": carrier_skin.skin_lineage_hash,
        }
        frames[jid] = DerivedJointFrameV1(
            joint_id=jid,
            parent_id=pid,
            rest_position=tuple(map(float, pos)),
            rotation_matrix=payload["rotation_matrix"],
            frame_hash=content_sha256(payload),
        )

    report = {
        "schema": "RealSaS.PostBindJointFrameQualification.v2",
        "status": "PASS",
        "subject_specific_code_used": False,
        "joint_name_semantics_used": False,
        "source_index_semantics_used": False,
        "invented_epsilon": False,
        "inactive_subtree_mean_weight_max": float(inactive_subtree_mean_weight_max),
        "inactive_subtree_max_vertex_weight_max": float(inactive_subtree_max_vertex_weight_max),
        "coincident_position_tolerance": float(coincident_position_tolerance),
        "coincident_edge_count": len(events),
        "events": [event.to_dict() for event in events],
        "carrier_skin_lineage_hash": carrier_skin.skin_lineage_hash,
    }
    return frames, report


def frame_set_hash_v2(frames, *, carrier_skin_lineage_hash: str) -> str:
    return content_sha256(
        {
            "schema": "RealSaS.DerivedJointFrameSet.v2",
            "carrier_skin_lineage_hash": carrier_skin_lineage_hash,
            "frames": tuple(
                (jid, frames[jid].frame_hash) for jid in sorted(frames)
            ),
        }
    )


__all__ = [
    "CoincidentFrameEventV2",
    "derive_joint_frames_post_bind_v2",
    "frame_set_hash_v2",
]
