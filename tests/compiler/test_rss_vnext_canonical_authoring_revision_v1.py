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
    original,_,mesh,rig,skin,controls=_compiled_preview_fixture(tmp_path)
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
    original,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
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
    original,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
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
    archive,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
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
    original,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
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
    original,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
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
    original,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
    path,_=_write_edit(tmp_path,original,**_args(mesh,rig,skin))
    original.write_bytes(original.read_bytes()+b"")
    # Wrong source bytes must fail rather than silently attach revision to
    # another candidate with the same human-facing character name.
    other=tmp_path/"not-original.rss"
    other.write_bytes(b"not-a-puppet")
    with pytest.raises(Exception):
        reopen_rss_vnext_canonical_user_edit_revision_v1(
            revision_path=path,source_rss_path=other,
        )
