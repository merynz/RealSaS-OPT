from __future__ import annotations

"""Exact native Blender authoring inventory (no Blender required for court)."""

from dataclasses import asdict,replace
import numpy as np
import pytest

from tools.truth_foundry.native_blender_attachment_inventory_v1 import (
    compile_blender_native_attachment_inventory_candidate_v1,
)

def _constraint(kind="CHILD_OF",target="Rig",bone="Grip",**overrides):
    row={
        "name":"Constraint","type":kind,"mute":False,"enabled":True,
        "is_valid":True,"influence":1.,
        "target":target,"target_type":"ARMATURE","subtarget":bone,
        "mix_mode":"REPLACE" if kind=="COPY_TRANSFORMS" else None,
    }
    if kind=="CHILD_OF":
        row.update({f"use_{axis}_{dim}":True
            for axis in ("location","rotation","scale")
            for dim in ("x","y","z")})
    row.update(overrides)
    return row


def _owner(index,name,*,bone=None,parent_type="OBJECT",
           parent_name=None,constraints=None,modifiers=None):
    return {
        "owner_index":index,"name":name,
        "parent_type":"BONE" if bone is not None else parent_type,
        "parent_name":"Rig" if bone is not None else parent_name,
        "parent_object_type":"ARMATURE" if bone is not None else (
            "MESH" if parent_name is not None else None
        ),
        "parent_bone":bone or "",
        "constraints":list(constraints or []),
        "modifiers":list(modifiers or []),
    }


def _court(rows,owners=None):
    return compile_blender_native_attachment_inventory_candidate_v1(
        owner_rows=rows,
        source_face_owner=np.asarray(
            owners if owners is not None else [0,0,1,1,1],dtype=np.int32,
        ),
        source_asset_sha256="a"*64,
        native_teacher_receipt_sha256="b"*64,
        native_teacher_npz_sha256="c"*64,
        observation_admission_sha256="d"*64,
    )


def test_native_bone_parent_is_an_authored_link_but_not_holder_or_contact():
    result=_court([
        _owner(0,"Sword",bone="HandSocket"),
        _owner(1,"Body",modifiers=[{"type":"ARMATURE","target":"Rig"}]),
    ])
    sword,body=result.owners
    assert sword.state=="UNQUALIFIED_AUTHORED_BONE_LINK_CANDIDATE"
    assert (sword.parent_armature_name,sword.parent_bone_name)==(
        "Rig","HandSocket",
    )
    assert sword.authoring_mechanism=="BONE_PARENT"
    assert (sword.source_face_start,sword.source_face_count)==(0,2)
    assert sword.source_contact_qualified is False
    assert sword.holder_face_known is False
    assert sword.held_face_selected is False
    assert sword.training_admitted is False
    assert body.state=="UNKNOWN_NO_EXPLICIT_UNIQUE_RIGID_BONE_LINK"
    assert body.armature_deform_modifier_count==1
    assert (body.source_face_start,body.source_face_count)==(2,3)
    assert not body.exhaustive_no_attachment_verified
    assert not result.source_rigid_attachment_qualified
    assert not result.exhaustive_empty_socket_inventory_qualified


@pytest.mark.parametrize("constraint",[
    _constraint("CHILD_OF"),
    _constraint("COPY_TRANSFORMS"),
])
def test_full_active_rigid_constraint_to_existing_bone_is_candidate_only(constraint):
    row=_owner(0,"Weapon",constraints=[constraint])
    result=_court([row,_owner(1,"Body")])
    candidate=result.owners[0]
    assert candidate.state=="UNQUALIFIED_AUTHORED_BONE_LINK_CANDIDATE"
    assert candidate.parent_armature_name=="Rig"
    assert candidate.parent_bone_name=="Grip"
    assert candidate.declared_link_count==1
    assert candidate.unresolved_constraint_count==0
    assert not candidate.source_contact_qualified


@pytest.mark.parametrize("constraint",[
    _constraint("CHILD_OF",use_location_x=False),
    _constraint("CHILD_OF",mute=True),
    _constraint("CHILD_OF",enabled=False),
    _constraint("CHILD_OF",is_valid=False),
    _constraint("COPY_TRANSFORMS",mix_mode="BEFORE_FULL"),
    _constraint("COPY_TRANSFORMS",influence=.5),
    _constraint("COPY_LOCATION"),
    _constraint("CHILD_OF",target_type="EMPTY"),
    _constraint("CHILD_OF",subtarget=""),
])
def test_partial_muted_unknown_or_nonbone_constraints_are_never_rigid_truth(constraint):
    result=_court([
        _owner(0,"Weapon",constraints=[constraint]),
        _owner(1,"Body"),
    ])
    row=result.owners[0]
    assert row.state=="ABSTAIN_AMBIGUOUS_OR_PARTIAL_SOURCE_BINDING"
    assert row.unresolved_constraint_count==1
    assert row.parent_bone_name is None
    assert not row.training_admitted


def test_multiple_incompatible_bone_targets_abstain_not_choose_first():
    result=_court([
        _owner(0,"Sword",bone="LeftGrip",
               constraints=[_constraint("CHILD_OF",bone="RightGrip")]),
        _owner(1,"Body"),
    ])
    row=result.owners[0]
    assert row.declared_link_count==2
    assert row.state=="ABSTAIN_AMBIGUOUS_OR_PARTIAL_SOURCE_BINDING"
    assert row.parent_bone_name is None


def test_parent_and_constraint_agree_on_same_bone_remains_unqualified():
    row=_court([
        _owner(0,"Sword",bone="Grip",
               constraints=[_constraint("CHILD_OF",bone="Grip")]),
        _owner(1,"Body"),
    ]).owners[0]
    assert row.state=="UNQUALIFIED_AUTHORED_BONE_LINK_CANDIDATE"
    assert row.declared_link_count==2
    assert row.authoring_mechanism=="BONE_PARENT+CHILD_OF"
    assert not row.holder_face_known


def test_mesh_object_parent_is_not_false_negative_or_auto_bone():
    result=_court([
        _owner(0,"Sword",parent_name="BodyMesh"),
        _owner(1,"Body"),
    ])
    row=result.owners[0]
    assert row.state=="ABSTAIN_AMBIGUOUS_OR_PARTIAL_SOURCE_BINDING"
    assert row.unresolved_constraint_count==1
    assert not row.exhaustive_no_attachment_verified


def test_nonrigid_skin_modifier_cannot_be_confused_with_grip_attachment():
    row=_court([
        _owner(0,"Cape",modifiers=[{"type":"ARMATURE","target":"Rig"}]),
        _owner(1,"Body"),
    ]).owners[0]
    assert row.armature_deform_modifier_count==1
    assert row.declared_link_count==0
    assert row.state=="UNKNOWN_NO_EXPLICIT_UNIQUE_RIGID_BONE_LINK"
    assert row.parent_bone_name is None


def test_repeated_deterministic_hashes_bind_original_teacher_mesh_face_owner():
    rows=[_owner(0,"Sword",bone="Grip"),_owner(1,"Body")]
    a=_court(rows)
    b=_court(rows)
    assert a==b
    assert len(a.inventory_sha256)==64
    assert len(a.source_face_owner_sha256)==64
    assert a.source_face_owner_sha256!=a.authoring_inventory_sha256
    shifted=_court(rows,owners=[0,1,1,1,1])
    assert shifted.inventory_sha256!=a.inventory_sha256
    assert shifted.owners[0].source_face_count!=a.owners[0].source_face_count


@pytest.mark.parametrize("owners",[
    [0,1,0,1,1],
    [-1,0,1,1,1],
    [0,0,2,2,2],
])
def test_bad_or_noncontiguous_teacher_face_owners_fail_closed(owners):
    rows=[_owner(0,"Sword",bone="Grip"),_owner(1,"Body")]
    with pytest.raises(ValueError,match="TEACHER_FACE_OWNER_DOMAIN|SOURCE_FACE_CONTIGUOUS_OWNER_DRIFT"):
        _court(rows,owners=owners)


def test_duplicate_owner_names_or_wrong_indices_refused():
    with pytest.raises(ValueError,match="TEACHER_OWNER_UNIQUE_OR_COVERAGE"):
        _court([_owner(0,"Sword"),_owner(1,"Sword")])
    with pytest.raises(ValueError,match="SOURCE_OWNER_ROW_INCOMPLETE"):
        _court([_owner(1,"Sword"),_owner(0,"Body")])


def test_incomplete_constraint_inventory_schema_does_not_forge_link():
    malformed=_owner(0,"Sword",constraints=[{"name":"unknown"}])
    with pytest.raises(ValueError,match="SOURCE_CONSTRAINT_TYPED"):
        _court([malformed,_owner(1,"Body")])
    with pytest.raises(ValueError,match="SOURCE_OWNER_ROW_INCOMPLETE"):
        _court([{"owner_index":0,"name":"Sword"},_owner(1,"Body")])


def test_wrong_source_bytes_or_observation_provenance_rejected_before_join():
    rows=[_owner(0,"Sword"),_owner(1,"Body")]
    with pytest.raises(ValueError,match="SOURCE_PROVENANCE_HASH_DOMAIN"):
        compile_blender_native_attachment_inventory_candidate_v1(
            owner_rows=rows,
            source_face_owner=np.asarray([0,0,1,1,1],dtype=np.int32),
            source_asset_sha256="unverified",
            native_teacher_receipt_sha256="b"*64,
            native_teacher_npz_sha256="c"*64,
            observation_admission_sha256="d"*64,
        )


def test_zero_triangle_renderable_owner_preserves_source_identity_without_held_face():
    rows=[
        _owner(0,"InvisibleAttachmentHelper",bone="Grip"),
        _owner(1,"VisibleBody"),
    ]
    got=_court(rows,owners=[1,1,1,1])
    empty=got.owners[0]
    assert empty.source_face_count==0
    assert empty.state=="ABSTAIN_EMPTY_RENDERABLE_MESH_NO_FACE_ADDRESS"
    assert empty.parent_bone_name is None
    assert not empty.held_face_selected
    assert got.owners[1].source_face_start==0
    assert got.owners[1].source_face_count==4


def test_blender_exporter_actual_python_source_parses_and_roundtrips_authoring_fields():
    """Validate the Blender script's real schema extraction without bpy.

    This is NOT a Blender runtime pass: evaluate only two pure extraction
    functions from their actual source AST, with typed fake Blender objects.
    """
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    script=Path(__file__).resolve().parents[2]/(
        "tools/truth_foundry/blender_native_attachment_inventory_v1.py"
    )
    tree=ast.parse(script.read_text("utf8"),filename=str(script))
    selected=[
        node for node in tree.body
        if isinstance(node,ast.FunctionDef)
        and node.name in ("_constraint_row","_source_row")
    ]
    assert len(selected)==2
    scope={}
    exec(compile(ast.Module(body=selected,type_ignores=[]),
                 str(script),"exec"),scope)
    rig=SimpleNamespace(name="Rig",type="ARMATURE")
    constraint=SimpleNamespace(
        name="GripParent",type="CHILD_OF",target=rig,subtarget="Grip",
        mute=False,enabled=True,is_valid=True,influence=1.,
        inverse_matrix=np.eye(4),
        **{f"use_{axis}_{dim}":True
           for axis in ("location","rotation","scale")
           for dim in ("x","y","z")}
    )
    mod=SimpleNamespace(
        name="ArmatureDeform",type="ARMATURE",object=rig,
    )
    sword=SimpleNamespace(
        name="Sword",parent=rig,parent_type="BONE",parent_bone="Grip",
        constraints=[constraint],modifiers=[mod],
    )
    row=scope["_source_row"](sword,0)
    assert row["constraints"][0]["enabled"] is True
    assert row["constraints"][0]["is_valid"] is True
    assert row["constraints"][0]["use_scale_z"] is True
    assert row["constraints"][0]["inverse_matrix"]==np.eye(4).tolist()
    assert row["parent_type"]=="BONE"
    assert row["parent_bone"]=="Grip"
    assert row["modifiers"][0]["type"]=="ARMATURE"
    classifier=_court([row,_owner(1,"Body")]).owners[0]
    assert classifier.state=="UNQUALIFIED_AUTHORED_BONE_LINK_CANDIDATE"
    assert classifier.authoring_mechanism=="BONE_PARENT+CHILD_OF"
    assert not classifier.holder_face_known
    constraint.is_valid=False
    invalid=scope["_source_row"](sword,0)
    classified=_court([invalid,_owner(1,"Body")]).owners[0]
    assert classified.state=="ABSTAIN_AMBIGUOUS_OR_PARTIAL_SOURCE_BINDING"
