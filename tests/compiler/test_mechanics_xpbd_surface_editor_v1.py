from __future__ import annotations

"""TESSA surface XPBD in the actual editor/Graphics projection consumer."""

from dataclasses import replace
import json

import pytest

from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanics_xpbd_domain_v1 import (
    DeclaredXPBDSecondaryRequestV1,
    prepare_declared_xpbd_secondary_domain_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_secondary_v1 import (
    XPBDSecondaryPolicyV1, XPBDSecondaryChainPolicyV1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1, XPBDSurfacePolicyV1,
    XPBDSurfaceJointSphereColliderV1,
    prepare_declared_xpbd_surface_domain_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_product_bridge_v1 import _editor_bundle


def _editor():
    bundle,_,_=_editor_bundle()
    domain=prepare_declared_xpbd_surface_domain_v1(
        bundle.mesh,bundle.rig,bundle.skin,bundle.mechanics,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=bundle.mesh.faces,
            pinned_vertex_ids=("MV:0",),
        ),
    )
    policy=XPBDSurfacePolicyV1(domain_hash=domain.domain_hash)
    return bundle,domain,policy


def test_editor_surface_simulation_updates_same_canonical_graphics_vertices():
    bundle,domain,policy=_editor()
    session=ControllablePuppetSessionV1(bundle)
    session.enable_declared_xpbd_surface(domain,policy)
    base=session.frame()
    assert base.xpbd_surface_state.ticks==0
    assert base.xpbd_surface_state.frames==0
    session.set_view_index(2)
    snapshot=session.frame()
    assert snapshot.xpbd_surface_state==base.xpbd_surface_state
    session.set_view_index(0)
    again=session.frame()
    assert again.xpbd_surface_state==base.xpbd_surface_state
    advanced=session.frame(dt_seconds=1./60.)
    assert advanced.xpbd_surface_state.ticks==2
    assert advanced.product_revision_id==base.product_revision_id
    assert advanced.deformed.mesh_lineage_hash==base.deformed.mesh_lineage_hash
    assert advanced.deformed.canonical_mesh_vertex_ids==base.deformed.canonical_mesh_vertex_ids
    assert advanced.pose==base.pose
    assert advanced.deformed.positions!=base.deformed.positions
    assert advanced.meshes!=base.meshes
    assert session.frame().xpbd_surface_state==advanced.xpbd_surface_state
    session.disable_declared_xpbd_surface()
    assert session.frame().xpbd_surface_state is None


def test_editor_surface_state_can_be_exported_reopened_and_replayed_without_drift():
    bundle,domain,policy=_editor()
    session=ControllablePuppetSessionV1(bundle)
    session.enable_declared_xpbd_surface(domain,policy)
    session.frame()
    checkpoint=session.frame(dt_seconds=1./60.)
    wire=json.loads(json.dumps(session.export_xpbd_surface_runtime_state()))
    assert wire["state_hash"]==checkpoint.xpbd_surface_state.state_hash
    reopened=ControllablePuppetSessionV1(bundle)
    reopened.enable_declared_xpbd_surface(domain,policy)
    reopened.restore_xpbd_surface_runtime_state(wire)
    assert reopened.frame().xpbd_surface_state==checkpoint.xpbd_surface_state
    next_original=session.frame(dt_seconds=1./60.)
    next_reopened=reopened.frame(dt_seconds=1./60.)
    assert next_original.deformed==next_reopened.deformed
    assert next_original.xpbd_surface_state==next_reopened.xpbd_surface_state
    assert next_original.meshes==next_reopened.meshes
    tampered=json.loads(json.dumps(wire))
    tampered["particles"][0]["world_P"][0]+=0.3
    with pytest.raises(QualificationError,match="STATE_HASH_DRIFT"):
        reopened.restore_xpbd_surface_runtime_state(tampered)
    assert reopened.frame().xpbd_surface_state==next_reopened.xpbd_surface_state


def test_editor_refuses_rehashed_surface_or_double_owned_joint_particles():
    bundle,domain,policy=_editor()
    # A one-triangle editor fixture has no bend pairs; forge a real
    # accepted stretch constraint to test content-addressed requalification.
    assert domain.stretch_edges
    forged=replace(domain,stretch_edges=())
    forged=replace(forged,domain_hash=content_sha256({
        k:v for k,v in forged.to_dict().items() if k!="domain_hash"
    }))
    session=ControllablePuppetSessionV1(bundle)
    with pytest.raises(QualificationError,match="DOMAIN_REQUALIFICATION_DRIFT"):
        session.enable_declared_xpbd_surface(
            forged,replace(policy,domain_hash=forged.domain_hash),
        )
    session.enable_declared_xpbd_surface(domain,policy)
    joint_domain=prepare_declared_xpbd_secondary_domain_v1(
        bundle.mesh,bundle.rig,bundle.skin,bundle.mechanics,
        DeclaredXPBDSecondaryRequestV1(("K0",)),
    )
    joint_policy=XPBDSecondaryPolicyV1(
        domain_hash=joint_domain.domain_hash,
        chains=(XPBDSecondaryChainPolicyV1("K0"),),
    )
    with pytest.raises(QualificationError,match="SURFACE_DOUBLE_OWNERSHIP"):
        session.enable_declared_xpbd_secondary(joint_domain,joint_policy)
    assert session.frame().xpbd_surface_state.ticks==0


def test_editor_does_not_advance_physics_or_grant_label_authority_implicitly():
    bundle,domain,policy=_editor()
    session=ControllablePuppetSessionV1(bundle)
    with pytest.raises(QualificationError,match="DT_WITHOUT_POLICY"):
        session.frame(dt_seconds=1./60.)
    session.enable_declared_xpbd_surface(domain,policy)
    session.frame()
    with pytest.raises(QualificationError,match="STATE_WIRE"):
        session.restore_xpbd_surface_runtime_state({"random":"payload"})


def test_editor_joint_attached_collider_pose_snapshot_and_reopen():
    bundle,domain,base_policy=_editor()
    jid=bundle.rig.joints[0].canonical_joint_id
    proxy=XPBDSurfaceJointSphereColliderV1(
        "explicit_body_proxy",jid,radius=1.e-5,
    )
    policy=replace(base_policy,joint_spheres=(proxy,))
    editor=ControllablePuppetSessionV1(bundle)
    editor.enable_declared_xpbd_surface(domain,policy)
    first=editor.frame()
    expected_joint=next(row.solved_P for row in first.pose.joints
                        if row.canonical_joint_id==jid)
    expected_centers=[["explicit_body_proxy",list(expected_joint)]]
    assert first.xpbd_surface_state.metadata[
        "kinematic_joint_sphere_centers"
    ]==expected_centers
    editor.set_view_index(2)
    assert editor.frame().xpbd_surface_state==first.xpbd_surface_state
    editor.set_view_index(0)
    advanced=editor.frame(dt_seconds=1./60.)
    assert advanced.xpbd_surface_state.ticks==2
    checkpoint=json.loads(json.dumps(editor.export_xpbd_surface_runtime_state()))
    reopened=ControllablePuppetSessionV1(bundle)
    reopened.enable_declared_xpbd_surface(domain,policy)
    reopened.restore_xpbd_surface_runtime_state(checkpoint)
    assert reopened.frame().xpbd_surface_state==advanced.xpbd_surface_state
    original_next=editor.frame(dt_seconds=1./60.)
    reopened_next=reopened.frame(dt_seconds=1./60.)
    assert original_next.deformed==reopened_next.deformed
    assert original_next.xpbd_surface_state==reopened_next.xpbd_surface_state
