from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.motion import build_deterministic_preset_motion
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, QualifiedJoint, QualifiedSkinIR, QualifiedSkinRow
from compiler.realsas_compiler_core.v4 import build_mechanical_state, qualified_skeleton_v2_lineage_hash
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from experiments.geppetto_arachne_r6_20260901.verified_lbs_v1 import apply_verified_lbs_v1


WITNESSES = (
    "weighted_child",
    "root_only_with_decoy_child",
    "split_children_with_decoy",
)


def _surface():
    points = (
        (-0.65, -0.20, 0.0),
        (-0.20, 0.45, 0.0),
        (0.35, -0.35, 0.0),
        (0.75, 0.30, 0.0),
    )
    nodes = tuple(
        SurfaceNode(
            f"S:{i}",
            p,
            (),
            ("SYNTHETIC_MOTION_GATE",),
            (),
            (),
            f"PG:{i}",
        )
        for i, p in enumerate(points)
    )
    return RiggingSurfaceIR(nodes, geometry_lineage_hash="S:MOTION:V1")


def _skeleton(joints):
    roots = tuple(j.canonical_joint_id for j in joints if j.parent_canonical_id is None)
    sk = QualifiedSkeletonIRV2(tuple(joints), roots, {}, {"status": "PASS"}, "")
    return replace(sk, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(sk))


def _skin(surface, skeleton, rows_by_sid):
    rows = tuple(
        QualifiedSkinRow(sid, tuple(influences), 0.0, 0.0)
        for sid, influences in sorted(rows_by_sid.items())
    )
    return QualifiedSkinIR(rows, surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash, {"status": "PASS"}, f"W:{len(rows)}")


def _mechanical(name):
    surface = _surface()
    sids = tuple(n.surface_id for n in surface.surface_nodes)
    if name == "weighted_child":
        joints = (
            QualifiedJoint("J:ROOT", (0.0, 0.0, 0.0), None),
            QualifiedJoint("J:CHILD", (0.35, 0.0, 0.0), "J:ROOT"),
        )
        rows = {sid: (("J:ROOT", 0.25), ("J:CHILD", 0.75)) for sid in sids}
    elif name == "root_only_with_decoy_child":
        joints = (
            QualifiedJoint("J:ROOT", (0.0, 0.0, 0.0), None),
            QualifiedJoint("J:DECOY", (0.35, 0.0, 0.0), "J:ROOT"),
        )
        rows = {sid: (("J:ROOT", 1.0), ("J:DECOY", 0.0)) for sid in sids}
    elif name == "split_children_with_decoy":
        joints = (
            QualifiedJoint("J:ROOT", (0.0, 0.0, 0.0), None),
            QualifiedJoint("J:A_DECOY", (-0.25, 0.0, 0.0), "J:ROOT"),
            QualifiedJoint("J:B_WEIGHTED", (0.45, 0.0, 0.0), "J:ROOT"),
        )
        rows = {sid: (("J:ROOT", 0.20), ("J:A_DECOY", 0.0), ("J:B_WEIGHTED", 0.80)) for sid in sids}
    else:
        raise ValueError(name)
    skeleton = _skeleton(joints)
    skin = _skin(surface, skeleton, rows)
    return build_mechanical_state(surface, skeleton, skin)


def _dense_surface_weights(mechanical):
    nodes = tuple(sorted(mechanical.surface.surface_nodes, key=lambda n: n.surface_id))
    joints = tuple(sorted(mechanical.skeleton.joints, key=lambda j: j.canonical_joint_id))
    rows = {r.surface_id: r for r in mechanical.skin.rows}
    ji = {j.canonical_joint_id: i for i, j in enumerate(joints)}
    p = np.asarray([n.P for n in nodes], dtype=np.float64)
    w = np.zeros((len(nodes), len(joints)), dtype=np.float64)
    for ni, node in enumerate(nodes):
        for jid, value in rows[node.surface_id].influences:
            w[ni, ji[jid]] = float(value)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    return p, w, joints


def _pivoted_rotation_z(pivot, degrees):
    angle = math.radians(float(degrees))
    c, s = math.cos(angle), math.sin(angle)
    px, py, _ = map(float, pivot)
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    out[0, 3] = px - c * px + s * py
    out[1, 3] = py - s * px - c * py
    return out


def _probe(mechanical, amplitude_deg):
    motion = build_deterministic_preset_motion(mechanical, amplitude_deg=float(amplitude_deg))
    assert len(motion.joint_tracks) == 1
    track = motion.joint_tracks[0]
    positive_key = max(track.keys, key=lambda k: float(k.rotation_deg))
    assert float(positive_key.rotation_deg) > 0.0

    p, w, joints = _dense_surface_weights(mechanical)
    ji = {j.canonical_joint_id: i for i, j in enumerate(joints)}
    selected = track.canonical_joint_id
    selected_joint = next(j for j in joints if j.canonical_joint_id == selected)
    transforms = np.tile(np.eye(4, dtype=np.float64), (1, len(joints), 1, 1))
    transforms[0, ji[selected]] = _pivoted_rotation_z(selected_joint.position, positive_key.rotation_deg)
    pred = apply_verified_lbs_v1(p, w, transforms)[0].astype(np.float64)
    displacement = np.linalg.norm(pred - p, axis=1)
    return {
        "motion": motion,
        "track": track,
        "selected_skin_mass": float(w[:, ji[selected]].sum()),
        "rms_displacement": float(np.sqrt(np.mean(displacement ** 2))),
        "max_displacement": float(displacement.max()),
    }


@pytest.mark.parametrize("witness", WITNESSES)
def test_deterministic_preset_causes_verified_lbs_motion(witness):
    mechanical = _mechanical(witness)
    replay_a = build_deterministic_preset_motion(mechanical, amplitude_deg=4.0)
    replay_b = build_deterministic_preset_motion(mechanical, amplitude_deg=4.0)
    assert replay_a.motion_state_hash == replay_b.motion_state_hash

    small = _probe(mechanical, 4.0)
    large = _probe(mechanical, 12.0)
    track = small["track"]

    assert track.transform_space == "PUPPET_LOCAL_2D_2P5D"
    assert small["motion"].metadata["authored_joint_names_used"] is False
    assert small["motion"].metadata["full_3d_motion_authority"] is False
    assert all(not hasattr(k, "rotation_xyzw") for k in track.keys)
    assert small["selected_skin_mass"] > 0.0
    assert small["max_displacement"] > 1e-10
    assert large["rms_displacement"] > small["rms_displacement"]
    assert large["motion"].motion_state_hash != small["motion"].motion_state_hash
