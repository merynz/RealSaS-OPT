from __future__ import annotations

"""Explicit neural OPEN pin review -> editor command -> native XPBD domain.

These synthetic tests prove canonical address/authority isolation, NOT trained
pin quality, native source truth, UI authentication or product Graphics parity.
"""
from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.mechanics_editor_open_pin_approval_v1 import (
    ACTION, EditorOpenPinNeuralReviewV1,
    EditorOpenPinAuthoringCommandV1,
    prepare_editor_approved_open_pin_domain_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    XPBDSurfacePolicyV1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_surface_editor_v1 import _editor


def _bundle_with_open_cloth_and_distinct_holder():
    """Keep the tested runtime's actual first OPEN mesh; add an independent host.

    Every IR is rebound to the SAME enlarged canonical M* lineage. No other
    surface is used as a solver substitute and no second runtime is invented.
    """
    bundle,_,_ = _editor()
    base=bundle.mesh
    held=base.faces[0]
    additions=tuple(
        replace(
            base.vertices[0],
            canonical_mesh_vertex_id=f"MV:{i+3}",
            source_candidate_vertex_id=f"RSSV:{i+3:08d}",
            component_id="HOLDER:BODY",
            P=p,
        )
        for i,p in enumerate(((3.,0.,0.),(4.,0.,0.),(3.,1.,0.)))
    )
    holder=("MV:3","MV:4","MV:5")
    mesh=replace(
        base,
        vertices=base.vertices+additions,
        faces=base.faces+(holder,),
        edges=base.edges+(("MV:3","MV:4"),("MV:4","MV:5"),("MV:3","MV:5")),
        mesh_lineage_hash="",
    )
    mesh=replace(mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))
    rig=replace(bundle.rig,mesh_lineage_hash=mesh.mesh_lineage_hash,
                rig_lineage_hash="")
    rig=replace(rig,rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))
    extra_rows=tuple(
        replace(bundle.skin.rows[0],
                canonical_mesh_vertex_id=f"MV:{i+3}",
                source_carrier_vertex_id=f"RSSV:{i+3:08d}")
        for i in range(3)
    )
    skin=replace(bundle.skin,
                 mesh_lineage_hash=mesh.mesh_lineage_hash,
                 deform_structure_hash=rig.deform_structure_hash,
                 rows=bundle.skin.rows+extra_rows,
                 skin_lineage_hash="")
    skin=replace(skin,skin_lineage_hash=canonical_skin_lineage_hash_v2(skin))
    mechanics=replace(bundle.mechanics,
                      mesh_lineage_hash=mesh.mesh_lineage_hash,
                      rig_lineage_hash=rig.rig_lineage_hash,
                      skin_lineage_hash=skin.skin_lineage_hash,
                      deform_structure_hash=rig.deform_structure_hash,
                      control_program_hash=rig.control_program_hash)
    return replace(bundle,mesh=mesh,rig=rig,skin=skin,mechanics=mechanics),held,holder


def _case():
    bundle,held,holder=_bundle_with_open_cloth_and_distinct_holder()
    review=EditorOpenPinNeuralReviewV1(
        anchor_event_index=7,
        source_mstar_topology_digest="a"*64,
        canonical_mesh_lineage_hash=bundle.mesh.mesh_lineage_hash,
        canonical_rig_lineage_hash=bundle.rig.rig_lineage_hash,
        canonical_skin_lineage_hash=bundle.skin.skin_lineage_hash,
        held_canonical_face=held,
        holder_canonical_face=holder,
        candidate_pin_vertex_ids=(held[0],),
        candidate_component_id=bundle.mesh.vertices[0].component_id,
        suggested_host_chain_id=bundle.rig.chains[0].canonical_chain_id,
        state="REVIEW_READY_NEURAL_PIN_AND_HOST_HYPOTHESIS",
    ).sealed()
    command=EditorOpenPinAuthoringCommandV1(
        review_hash=review.review_hash,
        canonical_mesh_lineage_hash=bundle.mesh.mesh_lineage_hash,
        canonical_rig_lineage_hash=bundle.rig.rig_lineage_hash,
        canonical_skin_lineage_hash=bundle.skin.skin_lineage_hash,
        authoring_revision_id="rsedit:explicit-cape-pin-0001",
        editor_action_id="local-editor-click-0042",
        approved_pin_vertex_ids=(held[0],),
        approved_host_chain_id=bundle.rig.chains[0].canonical_chain_id,
    ).sealed()
    params=dict(mesh=bundle.mesh,rig=bundle.rig,skin=bundle.skin,
                mechanics=bundle.mechanics,review=review,command=command)
    return bundle,review,command,params


def test_editor_review_and_explicit_approval_reuse_native_surface_solver():
    bundle,review,command,args=_case()
    result=prepare_editor_approved_open_pin_domain_v1(**args)
    assert result.domain.selected_faces==(bundle.mesh.faces[0],)
    assert result.domain.pinned_vertex_ids==(review.candidate_pin_vertex_ids[0],)
    assert len(result.domain.dynamic_vertex_ids)==2
    assert result.domain.mesh_lineage_hash==bundle.mesh.mesh_lineage_hash
    assert result.approval_receipt_sha256 and len(result.approval_receipt_sha256)==64
    assert not result.source_pin_qualified
    assert not result.product_release_qualified
    assert not result.automatic_xpbd_activation
    assert not result.source_material_truth_claimed
    editor=ControllablePuppetSessionV1(bundle)
    with pytest.raises(QualificationError,match="DT_WITHOUT_POLICY"):
        editor.frame(dt_seconds=1./60.)
    editor.enable_declared_xpbd_surface(
        result.domain,XPBDSurfacePolicyV1(domain_hash=result.domain.domain_hash),
    )
    start=editor.frame()
    advanced=editor.frame(dt_seconds=1./60.)
    assert advanced.xpbd_surface_state.ticks==2
    assert advanced.deformed.canonical_mesh_vertex_ids==start.deformed.canonical_mesh_vertex_ids
    assert advanced.deformed.positions!=start.deformed.positions
    assert advanced.product_revision_id==start.product_revision_id


def test_forged_review_user_action_and_altered_command_fail_closed():
    _,review,command,args=_case()
    for bad in (
        replace(review,source_pin_qualified=True).sealed(),
        replace(review,compiler_xpbd_activation_qualified=True).sealed(),
        replace(review,state="ABSTAIN_NO_CONFIDENT_OPEN_HEM_PINS").sealed(),
    ):
        with pytest.raises(QualificationError,match="REVIEW_NOT_NEURAL"):
            prepare_editor_approved_open_pin_domain_v1(
                **{**args,"review":bad},
            )
    with pytest.raises(QualificationError,match="EXPLICIT_AUTHORING"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"command":replace(command,editor_action_id="").sealed()},
        )
    with pytest.raises(QualificationError,match="EXPLICIT_AUTHORING"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"command":replace(command,approved_pin_vertex_ids=("MV:1",))},
        )
    with pytest.raises(QualificationError,match="EXPLICIT_AUTHORING"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"command":replace(command,action="AUTO_APPROVE").sealed()},
        )
    with pytest.raises(QualificationError,match="EXPLICIT_AUTHORING"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"command":replace(command,review_hash="f"*64).sealed()},
        )


def test_host_binding_repair_review_cannot_skip_guided_repair():
    _,review,command,args=_case()
    flagged=replace(
        review,state="REVIEW_NEEDS_EXPLICIT_HOST_PIN_BINDING_REPAIR",
    ).sealed()
    attempted=replace(command,review_hash=flagged.review_hash).sealed()
    with pytest.raises(QualificationError,match="HOST_BINDING_REPAIR_REQUIRED"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"review":flagged,"command":attempted},
        )


def test_manual_pin_override_is_user_edit_not_source_or_auto_physics():
    _,review,command,args=_case()
    manual=replace(command,approved_pin_vertex_ids=("MV:1",)).sealed()
    result=prepare_editor_approved_open_pin_domain_v1(
        **{**args,"command":manual}
    )
    assert result.domain.pinned_vertex_ids==("MV:1",)
    assert result.review_hash==review.review_hash
    assert result.command_hash==manual.command_hash
    assert not result.source_pin_qualified
    assert not result.automatic_xpbd_activation


def test_no_closed_wrong_component_or_missing_host_skin_pins():
    bundle,review,command,args=_case()
    # The model's held face has a distinct holder but a pin on the holder
    # cannot be approved as a cloth pin by the canonical geometry court.
    with pytest.raises(QualificationError,match="NOT_OPEN_HEM_OR_OVERPINNED"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"command":replace(
                command,approved_pin_vertex_ids=("MV:3",)
            ).sealed()},
        )
    with pytest.raises(QualificationError,match="HOST_CHAIN_NOT_DECLARED"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"command":replace(
                command,approved_host_chain_id="UNDECLARED"
            ).sealed()},
        )
    # Rebinding a skin revision without regenerating neural review/approval
    # cannot reuse the old candidate on the new canonical identity.
    changed=replace(bundle.skin,skin_lineage_hash="e"*64)
    with pytest.raises(QualificationError,match="CANONICAL_PARENT_LINEAGE_DRIFT"):
        prepare_editor_approved_open_pin_domain_v1(
            **{**args,"skin":changed},
        )


