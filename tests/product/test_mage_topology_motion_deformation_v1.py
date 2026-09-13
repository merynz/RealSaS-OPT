from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.motion import build_mage_topology_preset_motion
from compiler.realsas_compiler_core.motion_deformation import verified_motion_deformation_report
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedJoint,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    QualifiedMeshVertex,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import build_mechanical_state, qualified_skeleton_v2_lineage_hash
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2


def _fixture():
    joint_rows = (
        ("J:ROOT", (0.0, 0.0, 0.0), None),
        ("J:SPINE", (0.0, 0.0, 0.35), "J:ROOT"),
        ("J:CHEST", (0.0, 0.0, 0.72), "J:SPINE"),
        ("J:HEAD", (0.0, 0.0, 1.05), "J:CHEST"),
        ("J:ARM_L", (-0.42, 0.0, 0.72), "J:CHEST"),
        ("J:ARM_R", (0.42, 0.0, 0.72), "J:CHEST"),
        ("J:LEG_L", (-0.18, 0.0, -0.48), "J:ROOT"),
        ("J:LEG_R", (0.18, 0.0, -0.48), "J:ROOT"),
    )
    joints = tuple(QualifiedJoint(jid, pos, parent) for jid, pos, parent in joint_rows)
    skeleton = QualifiedSkeletonIRV2(joints, ("J:ROOT",), {}, {"status": "PASS"}, "")
    skeleton = replace(skeleton, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(skeleton))

    points = tuple((jid, pos) for jid, pos, _parent in joint_rows)
    nodes = tuple(
        SurfaceNode(f"S:{i}", tuple(map(float, pos)), (), ("MOTION_TEST",), (), persistence_group_id=f"PG:{i}")
        for i, (_jid, pos) in enumerate(points)
    )
    surface = RiggingSurfaceIR(nodes, geometry_lineage_hash="S:MAGE:MOTION:TEST")
    skin_rows = tuple(
        QualifiedSkinRow(node.surface_id, ((joint_rows[i][0], 1.0),), 0.0, 0.0)
        for i, node in enumerate(nodes)
    )
    skin = QualifiedSkinIR(skin_rows, surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash, {"status": "PASS"}, "W:MAGE:MOTION:TEST")
    mechanical = build_mechanical_state(surface, skeleton, skin)

    vertices = tuple(
        QualifiedMeshVertex(
            f"MV:{i}",
            node.P,
            SurfaceSupportBinding("IDENTITY", ((node.surface_id, 1.0),)),
            source_candidate_vertex_id=f"C:{i}",
        )
        for i, node in enumerate(nodes)
    )
    mesh = QualifiedEditableMeshIR(
        vertices,
        (("MV:0", "MV:1", "MV:6"), ("MV:0", "MV:7", "MV:1"), ("MV:1", "MV:2", "MV:4"), ("MV:1", "MV:5", "MV:2")),
        (("MV:0", "MV:1"),),
        surface.geometry_lineage_hash,
        4,
        "CAM:V4",
        {"status": "PASS"},
        "M:MAGE:MOTION:TEST",
    )
    mesh_skin = QualifiedMeshSkinIR(
        tuple(
            QualifiedMeshSkinRow(
                f"MV:{i}",
                ((joint_rows[i][0], 1.0),),
                ((nodes[i].surface_id, 1.0),),
                0.0,
                0.0,
            )
            for i in range(len(nodes))
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,
        "DIRECT_MODEL_QUERY_V1",
        {"status": "PASS"},
        "B:MAGE:MOTION:TEST",
    )
    return mechanical, mesh, mesh_skin


def _is_identity_key(key):
    return key.translation_xy == (0.0, 0.0) and key.rotation_deg == 0.0 and key.scale_xy == (1.0, 1.0) and key.depth_offset == 0.0


def test_mage_idle_run_are_deterministic_identity_safe_and_multijoint():
    mechanical, _mesh, _mesh_skin = _fixture()
    first = build_mage_topology_preset_motion(mechanical)
    second = build_mage_topology_preset_motion(mechanical)
    assert first.motion_state_hash == second.motion_state_hash
    assert {c.clip_id for c in first.clips} == {"mage_fit1_idle_v2", "mage_fit1_run_v2"}
    assert all(c.clip_kind == "PRESET" and c.loop for c in first.clips)
    assert len(first.joint_tracks) >= 8
    assert {t.clip_id for t in first.joint_tracks} == {"mage_fit1_idle_v2", "mage_fit1_run_v2"}
    assert all(_is_identity_key(t.keys[0]) for t in first.joint_tracks)
    assert all(_is_identity_key(t.keys[-1]) for t in first.joint_tracks)
    assert first.metadata["historical_mesh_authority_used"] is False
    assert first.metadata["historical_weight_authority_used"] is False


def test_mage_idle_run_cause_real_lbs_deformation_and_close_at_rest():
    mechanical, mesh, mesh_skin = _fixture()
    motion = build_mage_topology_preset_motion(mechanical)
    idle = verified_motion_deformation_report(mesh, mesh_skin, mechanical.skeleton, motion, "mage_fit1_idle_v2")
    run = verified_motion_deformation_report(mesh, mesh_skin, mechanical.skeleton, motion, "mage_fit1_run_v2")
    assert idle["status"] == "PASS"
    assert run["status"] == "PASS"
    assert idle["frame0_exact_identity_pass"]
    assert run["frame0_exact_identity_pass"]
    assert idle["loop_closure_identity_pass"]
    assert run["loop_closure_identity_pass"]
    assert idle["dynamic_nonzero_pass"]
    assert run["dynamic_nonzero_pass"]
    assert run["peak_max_displacement"] > idle["peak_max_displacement"]
