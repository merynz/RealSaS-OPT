from __future__ import annotations

"""Native-preview-backed OPEN pin authoring: .rsedit save/reopen/XPBD replay.

Regression data is SYNTHETIC ONLY; this does not claim learned pins, source
license verification, product release or Graphics post-physics pixel parity.
"""
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalRigChainIRV2,_hash_deform_structure,_hash_control_program,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanics_editor_open_pin_approval_v1 import (
    EditorOpenPinNeuralReviewV1,EditorOpenPinAuthoringCommandV1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes,read_rss_v2,
)
from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
    _make_pose_binary,reopen_rss_vnext_native_preview_package_v1,
)
from compiler.rss_vnext.workers.rss_vnext_canonical_authoring_revision_v1 import (
    compile_rss_vnext_canonical_user_edit_revision_v1,
    reopen_rss_vnext_canonical_user_edit_revision_v1,
)
from compiler.rss_vnext.workers.rss_vnext_editor_open_pin_revision_v1 import (
    ENTRY,seal_editor_approved_open_pin_rsedit_v1,
)
from tests.compiler.test_rss_vnext_native_preview_package_v1 import (
    _compiled_preview_fixture,
)


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def _two_component_native_preview(tmp_path):
    """Add one independent OPEN holder to a real replayed .rss fixture.

    No source authority is inferred: cloned supports/graphics are regression
    placeholders, and this archive must never be shipped as ProductProof.
    """
    original,_,mesh,rig,skin,_ = _compiled_preview_fixture(tmp_path)
    base=reopen_rss_vnext_native_preview_package_v1(original)
    assert len(mesh.faces)==1
    existing_ids={v.canonical_mesh_vertex_id for v in mesh.vertices}
    assert len(existing_ids)==len(mesh.vertices)
    holder_ids=tuple(f"MV:PIN_TEST_HOLDER:{i}" for i in range(3))
    assert not existing_ids.intersection(holder_ids)
    new_vertices=tuple(
        replace(
            mesh.vertices[0],
            canonical_mesh_vertex_id=holder_ids[i],
            source_candidate_vertex_id=f"RSSV:PIN_TEST_HOLDER:{i}",
            component_id="PIN_TEST_HOLDER",
            P=xyz,
            metadata={"pin_test_generated_holder_not_source_truth":True},
        )
        for i,xyz in enumerate(((4.,0.,0.),(5.,0.,0.),(4.,1.,0.)))
    )
    new_mesh=replace(
        mesh,vertices=mesh.vertices+new_vertices,
        faces=mesh.faces+(holder_ids,),
        edges=mesh.edges+(
            (holder_ids[0],holder_ids[1]),
            (holder_ids[1],holder_ids[2]),
            (holder_ids[0],holder_ids[2]),
        ),mesh_lineage_hash="",
    )
    new_mesh=replace(new_mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(new_mesh))
    root=rig.joints[0].canonical_joint_id
    chains=rig.chains or (
        CanonicalRigChainIRV2(
            "K:TEST_OPEN_PIN_HOST",(root,),"GENERIC_KINEMATIC_CHAIN",
            "TEST_ONLY_NOT_SOURCE_QUALIFIED",
        ),
    )
    new_rig=replace(
        rig,mesh_lineage_hash=new_mesh.mesh_lineage_hash,chains=chains,
        deform_structure_hash="",control_program_hash="",rig_lineage_hash="",
    )
    new_rig=replace(new_rig,deform_structure_hash=_hash_deform_structure(new_rig))
    new_rig=replace(new_rig,control_program_hash=_hash_control_program(new_rig))
    new_rig=replace(new_rig,rig_lineage_hash=canonical_rig_lineage_hash_v2(new_rig))
    root=rig.joints[0].canonical_joint_id
    rows=tuple(
        replace(
            skin.rows[0],
            canonical_mesh_vertex_id=holder_ids[i],
            source_carrier_vertex_id=f"RSSV:PIN_TEST_HOLDER:{i}",
            influences=((root,1.),),
        )
        for i in range(3)
    )
    new_skin=replace(
        skin,rows=skin.rows+rows,
        mesh_lineage_hash=new_mesh.mesh_lineage_hash,
        deform_structure_hash=new_rig.deform_structure_hash,
        skin_lineage_hash="",
    )
    new_skin=replace(new_skin,skin_lineage_hash=canonical_skin_lineage_hash_v2(new_skin))
    pose,hashes=_make_pose_binary(
        new_mesh,new_rig,new_skin,base["control_frames"],base["virtual_camera"],
    )
    entries=read_rss_v2(original)
    controls=json.loads(entries["controls.json"])
    controls["mechanics_pose_hashes"]=hashes
    entries["mesh.json"]=_json(new_mesh.to_dict())
    entries["rig.json"]=_json(new_rig.to_dict())
    entries["skin.json"]=_json(new_skin.to_dict())
    entries["controls.json"]=_json(controls)
    entries["pose.bin"]=pose
    manifest=dict(line.split("=",1) for line in entries["manifest.txt"].decode().splitlines())
    manifest["canonical_mesh_lineage_hash"]=new_mesh.mesh_lineage_hash
    manifest["canonical_rig_lineage_hash"]=new_rig.rig_lineage_hash
    manifest["canonical_skin_lineage_hash"]=new_skin.skin_lineage_hash
    for key in ("mesh.json","rig.json","skin.json","controls.json","pose.bin"):
        manifest["sha256:"+key]=sha256(entries[key]).hexdigest()
    entries["manifest.txt"]=(
        "\n".join(f"{k}={v}" for k,v in sorted(manifest.items()))+"\n"
    ).encode()
    source=tmp_path/"synthetic_source_with_open_holder.rss"
    source.write_bytes(_entry_bytes(entries))
    reopened=reopen_rss_vnext_native_preview_package_v1(source)
    assert reopened["mesh"]==new_mesh
    assert reopened["rig"]==new_rig
    assert reopened["skin"]==new_skin
    return source,new_mesh,new_rig,new_skin


def _case(tmp_path,*,manual_pin_index=0):
    source,mesh,rig,skin=_two_component_native_preview(tmp_path)
    blob,receipt=compile_rss_vnext_canonical_user_edit_revision_v1(
        source_rss_path=source,pin_only_authoring_shell=True,
    )
    shell=tmp_path/"pin_only_authoring_shell.rsedit"
    shell.write_bytes(blob)
    reopened_shell=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=shell,source_rss_path=source,
    )
    assert reopened_shell["editor_open_pin"] is None
    assert reopened_shell["mesh"]==mesh
    assert reopened_shell["rig"]==rig
    assert reopened_shell["skin"]==skin
    held=mesh.faces[0]
    holder=mesh.faces[-1]
    candidate=held[0]
    chosen=held[manual_pin_index]
    review=EditorOpenPinNeuralReviewV1(
        anchor_event_index=8,source_mstar_topology_digest="a"*64,
        canonical_mesh_lineage_hash=mesh.mesh_lineage_hash,
        canonical_rig_lineage_hash=rig.rig_lineage_hash,
        canonical_skin_lineage_hash=skin.skin_lineage_hash,
        held_canonical_face=held,holder_canonical_face=holder,
        candidate_pin_vertex_ids=(candidate,),
        candidate_component_id=mesh.vertices[0].component_id,
        suggested_host_chain_id=rig.chains[0].canonical_chain_id,
        state="REVIEW_READY_NEURAL_PIN_AND_HOST_HYPOTHESIS",
    ).sealed()
    command=EditorOpenPinAuthoringCommandV1(
        review_hash=review.review_hash,
        canonical_mesh_lineage_hash=mesh.mesh_lineage_hash,
        canonical_rig_lineage_hash=rig.rig_lineage_hash,
        canonical_skin_lineage_hash=skin.skin_lineage_hash,
        authoring_revision_id=receipt["authoring_revision_id"],
        editor_action_id="explicit-user-click-test-not-authenticated",
        approved_pin_vertex_ids=(chosen,),
        approved_host_chain_id=rig.chains[0].canonical_chain_id,
    ).sealed()
    return source,shell,review,command,held


def test_open_pin_only_rsedit_persists_and_independently_requalifies_xpbd(tmp_path):
    source,shell,review,command,held=_case(tmp_path)
    before=source.read_bytes()
    blob,row=seal_editor_approved_open_pin_rsedit_v1(
        source_rss_path=source,revision_path=shell,
        review=review,command=command,
    )
    saved=tmp_path/"editor-approved-cape-pin.rsedit"
    saved.write_bytes(blob)
    result=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=saved,source_rss_path=source,
    )
    pinned=result["editor_open_pin"]
    assert pinned is not None
    assert pinned["pin_revision_id"]==row["pin_revision_id"]
    assert pinned["review"]==review
    assert pinned["command"]==command
    assert pinned["prepared"].domain.selected_faces==(held,)
    assert pinned["prepared"].domain.pinned_vertex_ids==(held[0],)
    assert len(pinned["prepared"].domain.dynamic_vertex_ids)==2
    assert not pinned["automatic_xpbd_activation"]
    assert not pinned["product_release_qualified"]
    assert not pinned["source_pin_qualified"]
    assert not result["product_revision_minted"]
    assert not result["graphics_uv_binding_qualified"]
    assert source.read_bytes()==before
    assert read_rss_v2(saved)["mesh.json"]==read_rss_v2(shell)["mesh.json"]
    assert read_rss_v2(saved)["pose.bin"]==read_rss_v2(shell)["pose.bin"]
    with pytest.raises(ValueError,match="ALREADY_SEALED"):
        seal_editor_approved_open_pin_rsedit_v1(
            source_rss_path=source,revision_path=saved,
            review=review,command=command,
        )


def test_manual_editor_pin_override_remains_separate_from_neural_candidate(tmp_path):
    source,shell,review,command,held=_case(tmp_path,manual_pin_index=1)
    blob,row=seal_editor_approved_open_pin_rsedit_v1(
        source_rss_path=source,revision_path=shell,
        review=review,command=command,
    )
    saved=tmp_path/"manual-override.rsedit"
    saved.write_bytes(blob)
    opened=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=saved,source_rss_path=source,
    )["editor_open_pin"]
    assert opened["review"].candidate_pin_vertex_ids==(held[0],)
    assert opened["prepared"].domain.pinned_vertex_ids==(held[1],)
    assert opened["command"].review_hash==review.review_hash
    assert not opened["source_pin_qualified"]


def test_rehashed_auto_activation_flag_fails_on_rsedit_reopen(tmp_path):
    source,shell,review,command,_=_case(tmp_path)
    blob,_=seal_editor_approved_open_pin_rsedit_v1(
        source_rss_path=source,revision_path=shell,
        review=review,command=command,
    )
    path=tmp_path/"forged.rsedit"
    path.write_bytes(blob)
    entries=read_rss_v2(path)
    row=json.loads(entries[ENTRY])
    row["automatic_xpbd_activation"]=True
    row["pin_revision_id"]=content_sha256({
        k:v for k,v in row.items() if k!="pin_revision_id"
    })
    entries[ENTRY]=_json(row)
    manifest=dict(line.split("=",1) for line in entries["manifest.txt"].decode().splitlines())
    manifest["sha256:"+ENTRY]=sha256(entries[ENTRY]).hexdigest()
    manifest["pin_revision_id"]=row["pin_revision_id"]
    entries["manifest.txt"]=(
        "\n".join(f"{k}={v}" for k,v in sorted(manifest.items()))+"\n"
    ).encode()
    path.write_bytes(_entry_bytes(entries))
    with pytest.raises(RuntimeError,match="AUTHORITY_OR_PARENT_DRIFT"):
        reopen_rss_vnext_canonical_user_edit_revision_v1(
            revision_path=path,source_rss_path=source,
        )


def test_wrong_revision_id_and_pending_pin_without_user_action_do_not_activate(tmp_path):
    source,shell,review,command,_=_case(tmp_path)
    with pytest.raises(ValueError,match="EXPLICIT_REVIEW_AND_COMMAND_REQUIRED"):
        seal_editor_approved_open_pin_rsedit_v1(
            source_rss_path=source,revision_path=shell,
            review=review,
            command=replace(command,authoring_revision_id="wrong").sealed(),
        )
    pending=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=shell,source_rss_path=source,
    )
    assert pending["editor_open_pin"] is None
    with pytest.raises(ValueError,match="NO_EDIT"):
        compile_rss_vnext_canonical_user_edit_revision_v1(
            source_rss_path=source,
        )
