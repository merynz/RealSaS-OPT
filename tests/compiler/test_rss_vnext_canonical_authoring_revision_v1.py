from __future__ import annotations

"""Actual M*/Rig/Skin user edit-save-reopen, no invented source ProductProof."""

from collections import OrderedDict
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes,read_rss_v2,
)
from compiler.rss_vnext.workers.rss_vnext_canonical_authoring_revision_v1 import (
    AUTHORITY,SCHEMA,
    compile_rss_vnext_canonical_user_edit_revision_v1,
    reopen_rss_vnext_canonical_user_edit_revision_v1,
)
from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)
from tests.compiler.test_rss_vnext_native_preview_package_v1 import (
    _compiled_preview_fixture,
)


def _multijoint_preview_fixture(tmp_path):
    """Two-view canonical Geometry/Graphics with TEST-ONLY second deform joint.

    The live two-image test pipeline currently qualifies exactly one joint.
    Painting a different valid simplex requires >=2 joints, not fake weights
    such as root=0.3. Add a second independently marked *regression-only*
    deform joint to the already compiled developer .rss and recompute all
    real Rig/Skin/pose hashes. No source rig or training truth is asserted.
    This regression adapter is never a product compiler input.
    """
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
        _hash_deform_structure,_hash_control_program,
        canonical_rig_lineage_hash_v2,validate_canonical_rig_domain_v2,
    )
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
        canonical_skin_lineage_hash_v2,validate_canonical_skin_domain_v2,
    )
    from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
        _make_pose_binary,
    )
    original,manifest,mesh,rig,skin,controls=_compiled_preview_fixture(tmp_path)
    if len(rig.joints)>=2:
        return original,manifest,mesh,rig,skin,controls
    assert len(rig.joints)==1
    base=reopen_rss_vnext_native_preview_package_v1(original)
    root=rig.joints[0]
    child_id="J:USER_EDIT_REGRESSION_ONLY_CHILD"
    child=replace(
        root,canonical_joint_id=child_id,
        parent_canonical_joint_id=root.canonical_joint_id,
        source_proposal_joint_id="TEST_ONLY_NOT_NATIVE_SOURCE",
        P=(root.P[0]+.12,root.P[1]+.01,root.P[2]),
        metadata={**root.metadata,"regression_only_not_source_qualified":True},
    )
    frame=replace(
        rig.local_frames[0],canonical_joint_id=child_id,
        metadata={
            **rig.local_frames[0].metadata,
            "regression_only_not_source_qualified":True,
        },
    )
    revised_rig=replace(
        rig,joints=rig.joints+(child,),
        local_frames=rig.local_frames+(frame,),
        deform_structure_hash="",control_program_hash="",rig_lineage_hash="",
        metadata={
            **rig.metadata,
            "second_joint_regression_only_not_native_truth":True,
        },
    )
    revised_rig=replace(
        revised_rig,
        deform_structure_hash=_hash_deform_structure(revised_rig),
    )
    revised_rig=replace(
        revised_rig,
        control_program_hash=_hash_control_program(revised_rig),
    )
    revised_rig=replace(
        revised_rig,
        rig_lineage_hash=canonical_rig_lineage_hash_v2(revised_rig),
    )
    validate_canonical_rig_domain_v2(revised_rig)
    root_id=root.canonical_joint_id
    rows=list(skin.rows)
    rows[0]=replace(
        rows[0],
        influences=((root_id,.75),(child_id,.25)),
    )
    revised_skin=replace(
        skin,rows=tuple(rows),
        deform_structure_hash=revised_rig.deform_structure_hash,
        skin_lineage_hash="",
        metadata={
            **skin.metadata,
            "multi_joint_weights_regression_only_not_native_truth":True,
        },
    )
    revised_skin=replace(
        revised_skin,
        skin_lineage_hash=canonical_skin_lineage_hash_v2(revised_skin),
    )
    validate_canonical_skin_domain_v2(revised_skin)
    pose,pose_hashes=_make_pose_binary(
        mesh,revised_rig,revised_skin,controls,base["virtual_camera"],
    )
    entries=read_rss_v2(original)
    entries["rig.json"]=json.dumps(
        revised_rig.to_dict(),sort_keys=True,separators=(",",":"),
    ).encode()
    entries["skin.json"]=json.dumps(
        revised_skin.to_dict(),sort_keys=True,separators=(",",":"),
    ).encode()
    controls_payload=json.loads(entries["controls.json"])
    controls_payload["mechanics_pose_hashes"]=pose_hashes
    entries["controls.json"]=json.dumps(
        controls_payload,sort_keys=True,separators=(",",":"),
    ).encode()
    entries["pose.bin"]=pose
    manifest=dict(
        line.split("=",1)
        for line in entries["manifest.txt"].decode().splitlines()
    )
    manifest["canonical_rig_lineage_hash"]=revised_rig.rig_lineage_hash
    manifest["canonical_skin_lineage_hash"]=revised_skin.skin_lineage_hash
    for name in ("rig.json","skin.json","pose.bin","controls.json"):
        manifest["sha256:"+name]=sha256(entries[name]).hexdigest()
    entries["manifest.txt"]=(
        "\n".join(f"{key}={value}" for key,value in sorted(manifest.items()))
        +"\n"
    ).encode()
    revised=tmp_path/"two_source_geometry_test_only_multijoint.rss"
    revised.write_bytes(_entry_bytes(entries))
    reopened=reopen_rss_vnext_native_preview_package_v1(revised)
    assert reopened["rig"]==revised_rig
    assert reopened["skin"]==revised_skin
    assert reopened["mesh"]==mesh
    assert reopened["control_frames"]==controls
    assert reopened["product_revision_minted"] is False
    return revised,manifest,mesh,revised_rig,revised_skin,controls


def _args(mesh,rig,skin):
    vertex=mesh.vertices[0]
    joint=rig.joints[0]
    assert len(rig.joints)>=2
    v0=skin.rows[0].canonical_mesh_vertex_id
    joints=(rig.joints[0].canonical_joint_id,rig.joints[1].canonical_joint_id)
    weights={joints[0]:.65,joints[1]:.35}
    # In rare cases the same numbers are already painted, choose another.
    if tuple(sorted(weights.items())) == tuple(
        sorted(skin.rows[0].influences)
    ):
        weights={joints[0]:.7,joints[1]:.3}
    return {
        "mesh_vertex_position_edits":{
            vertex.canonical_mesh_vertex_id:(
                vertex.P[0],vertex.P[1],vertex.P[2]+.027,
            ),
        },
        "joint_rest_position_edits":{
            joint.canonical_joint_id:(
                joint.P[0]+.023,joint.P[1],joint.P[2],
            ),
        },
        "skin_weight_edits":{v0:weights},
    }


def _write_edit(tmp_path,archive,**kwargs):
    blob,receipt=compile_rss_vnext_canonical_user_edit_revision_v1(
        source_rss_path=archive,**kwargs,
    )
    path=tmp_path/"my-canonical-geometry-rig-skin-edit.rsedit"
    path.write_bytes(blob)
    return path,receipt


def test_all_three_canonical_edits_roundtrip_and_replay_actual_LBS(tmp_path):
    original,_,mesh,rig,skin,controls=_multijoint_preview_fixture(tmp_path)
    unchanged_bytes=original.read_bytes()
    path,receipt=_write_edit(
        tmp_path,original,**_args(mesh,rig,skin),
    )
    result=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=path,source_rss_path=original,
    )
    assert receipt["authority"]==AUTHORITY
    assert receipt["schema"]==SCHEMA
    assert result["authoring_revision_id"]==receipt["authoring_revision_id"]
    assert result["mesh"].mesh_lineage_hash!=mesh.mesh_lineage_hash
    assert result["rig"].rig_lineage_hash!=rig.rig_lineage_hash
    assert result["skin"].skin_lineage_hash!=skin.skin_lineage_hash
    assert tuple(v.canonical_mesh_vertex_id for v in result["mesh"].vertices)==tuple(
        v.canonical_mesh_vertex_id for v in mesh.vertices
    )
    assert tuple(j.canonical_joint_id for j in result["rig"].joints)==tuple(
        j.canonical_joint_id for j in rig.joints
    )
    assert result["mesh"].faces==mesh.faces
    assert result["mesh"].edges==mesh.edges
    assert result["rig"].mesh_lineage_hash==result["mesh"].mesh_lineage_hash
    assert result["skin"].deform_structure_hash==result["rig"].deform_structure_hash
    assert result["source_preview"]["control_frames"]==controls
    assert result["pose_bytes"]!=result["source_preview"]["pose_bytes"]
    assert receipt["source_geometry_qualification_reused"] is False
    assert receipt["source_rig_qualification_reused"] is False
    assert receipt["source_skin_qualification_reused"] is False
    assert receipt["graphics_uv_require_independent_rebind"] is True
    assert result["graphics_uv_binding_qualified"] is False
    assert not result["product_revision_minted"]
    assert original.read_bytes()==unchanged_bytes
    reopened_original=reopen_rss_vnext_native_preview_package_v1(original)
    assert reopened_original["mesh"].mesh_lineage_hash==mesh.mesh_lineage_hash


def test_weight_only_does_not_rename_mesh_or_rig(tmp_path):
    original,_,mesh,rig,skin,_=_multijoint_preview_fixture(tmp_path)
    path,_=_write_edit(
        tmp_path,original,skin_weight_edits=_args(mesh,rig,skin)[
            "skin_weight_edits"
        ],
    )
    result=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=path,source_rss_path=original,
    )
    assert result["mesh"].mesh_lineage_hash==mesh.mesh_lineage_hash
    assert result["rig"].rig_lineage_hash==rig.rig_lineage_hash
    assert result["skin"].skin_lineage_hash!=skin.skin_lineage_hash
    assert result["mesh"]==mesh
    assert result["rig"]==rig


def test_rest_edit_preserves_mesh_but_invalidates_actual_deform_structure(tmp_path):
    original,_,mesh,rig,skin,_=_multijoint_preview_fixture(tmp_path)
    path,_=_write_edit(
        tmp_path,original,joint_rest_position_edits=_args(mesh,rig,skin)[
            "joint_rest_position_edits"
        ],
    )
    result=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=path,source_rss_path=original,
    )
    assert result["mesh"]==mesh
    assert result["rig"].deform_structure_hash!=rig.deform_structure_hash
    assert result["skin"].deform_structure_hash!=skin.deform_structure_hash
    assert result["skin"].skin_lineage_hash!=skin.skin_lineage_hash


def test_invalid_weight_and_source_entity_addresses_fail_closed(tmp_path):
    archive,_,mesh,rig,skin,_=_multijoint_preview_fixture(tmp_path)
    args=_args(mesh,rig,skin)
    with pytest.raises(ValueError,match="WEIGHT_SIMPLEX_OR_JOINT"):
        compile_rss_vnext_canonical_user_edit_revision_v1(
            source_rss_path=archive,
            skin_weight_edits={
                skin.rows[0].canonical_mesh_vertex_id:{
                    rig.joints[0].canonical_joint_id:.3,
                },
            },
        )
    with pytest.raises(ValueError,match="UNKNOWN_RIG_JOINT"):
        compile_rss_vnext_canonical_user_edit_revision_v1(
            source_rss_path=archive,
            joint_rest_position_edits={"made-up-joint":(1.,2.,3.)},
        )
    with pytest.raises(ValueError,match="UNKNOWN_MESH_VERTEX"):
        compile_rss_vnext_canonical_user_edit_revision_v1(
            source_rss_path=archive,
            mesh_vertex_position_edits={"made-up-vertex":(1.,2.,3.)},
        )
    with pytest.raises(ValueError,match="NO_EDIT"):
        compile_rss_vnext_canonical_user_edit_revision_v1(
            source_rss_path=archive,
        )


def test_rehashed_forged_pose_still_fails_independent_mechanics_replay(tmp_path):
    original,_,mesh,rig,skin,_=_multijoint_preview_fixture(tmp_path)
    path,_=_write_edit(tmp_path,original,**_args(mesh,rig,skin))
    entries=read_rss_v2(path)
    mutated=bytearray(entries["pose.bin"])
    mutated[-24]^=1
    entries["pose.bin"]=bytes(mutated)
    manifest=entries["manifest.txt"].decode().splitlines()
    manifest=[
        "sha256:pose.bin="+sha256(entries["pose.bin"]).hexdigest()
        if x.startswith("sha256:pose.bin=") else x for x in manifest
    ]
    entries["manifest.txt"]=("\n".join(manifest)+"\n").encode()
    path.write_bytes(_entry_bytes(entries))
    with pytest.raises(RuntimeError,match="INDEPENDENT_LBS_REPLAY_DRIFT"):
        reopen_rss_vnext_canonical_user_edit_revision_v1(
            revision_path=path,source_rss_path=original,
        )


def test_rehashed_fake_product_release_cannot_claim_authoring_revision(tmp_path):
    original,_,mesh,rig,skin,_=_multijoint_preview_fixture(tmp_path)
    path,_=_write_edit(tmp_path,original,**_args(mesh,rig,skin))
    entries=read_rss_v2(path)
    receipt=json.loads(entries["revision.json"])
    receipt["product_release_admission"]=True
    from compiler.realsas_compiler_core.hashing import content_sha256
    receipt["authoring_revision_id"]=content_sha256({
        k:v for k,v in receipt.items() if k!="authoring_revision_id"
    })
    entries["revision.json"]=json.dumps(
        receipt,sort_keys=True,separators=(",",":"),
    ).encode()
    manifest=entries["manifest.txt"].decode().splitlines()
    manifest=[
        "sha256:revision.json="+sha256(entries["revision.json"]).hexdigest()
        if x.startswith("sha256:revision.json=") else x
        for x in manifest
    ]
    manifest=[
        "authoring_revision_id="+receipt["authoring_revision_id"]
        if x.startswith("authoring_revision_id=") else x for x in manifest
    ]
    entries["manifest.txt"]=("\n".join(manifest)+"\n").encode()
    path.write_bytes(_entry_bytes(entries))
    with pytest.raises(RuntimeError,match="REVISION_OR_PARENT_DRIFT"):
        reopen_rss_vnext_canonical_user_edit_revision_v1(
            revision_path=path,source_rss_path=original,
        )


def test_authoring_revision_requires_exact_original_rss_parent(tmp_path):
    original,_,mesh,rig,skin,_=_multijoint_preview_fixture(tmp_path)
    path,_=_write_edit(tmp_path,original,**_args(mesh,rig,skin))
    from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
        edit_rss_vnext_native_preview_controls_v1,
    )
    from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
        ControlTransformIRV1,
    )
    root=rig.controls[0].canonical_control_id
    other_blob,_=edit_rss_vnext_native_preview_controls_v1(
        source_package_path=original,
        control_frames=(
            (),
            (ControlTransformIRV1(
                canonical_control_id=root,translation=(.32,.03,0.),
            ),),
        ),
    )
    other=tmp_path/"valid-but-different-original.rss"
    other.write_bytes(other_blob)
    assert reopen_rss_vnext_native_preview_package_v1(other)["mesh"]==mesh
    with pytest.raises(RuntimeError,match="REVISION_OR_PARENT_DRIFT"):
        reopen_rss_vnext_canonical_user_edit_revision_v1(
            revision_path=path,source_rss_path=other,
        )
