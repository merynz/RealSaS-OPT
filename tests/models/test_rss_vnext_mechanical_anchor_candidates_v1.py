from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.rss_vnext_mechanical_anchor_candidates_v1 import (
    read_rss_vnext_mechanical_anchors_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_model_v6 import (
    CONTROL_KINDS_V6, RigProgramEventV6, RigTokenKindV6,
)
from tests.models.test_rss_vnext_surface_decisions_v1 import _real_surface_fixture


@pytest.fixture(autouse=True)
def deterministic_tiny_cpu():
    import torch
    torch.manual_seed(195)
    torch.set_num_threads(1)


def _source(*, anchor_kind=1, anchor_parent=0, control_type=7,
            anchor_coord=(65, 96, 128)):
    _,_,_,_,original,_ = _real_surface_fixture()
    events=[
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT,coord_indices=(128,128,128)),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
        RigProgramEventV6(RigTokenKindV6.CONTROL,
            aux_type_id=control_type,coord_indices=(40,90,140)),
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHOR,
            coord_indices=anchor_coord,anchor_parent_kind=anchor_kind,
            pointer_a=anchor_parent if anchor_kind==0 else -1,
            pointer_b=anchor_parent if anchor_kind==1 else -1),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
        RigProgramEventV6(RigTokenKindV6.EOS),
    ]
    rows=[]
    for ev in events:
        from dataclasses import asdict
        d=asdict(ev);d["kind"]=int(ev.kind);rows.append(d)
    return seal_inference_evidence_v1(
        replace(original,control_events=tuple(rows),evidence_hash=""),
    )


def test_model_control_type_abi_is_explicit_and_pinned_to_compiler():
    assert int(RigTokenKindV6.JOINT)==2
    assert int(RigTokenKindV6.CONTROL)==6
    assert int(RigTokenKindV6.ANCHOR)==15
    assert len(CONTROL_KINDS_V6)==8
    assert CONTROL_KINDS_V6[7]=="SOCKET_CONTROL"


def test_ax_socket_anchored_to_control_preserves_parent_and_unknown_held():
    e=_source()
    r=read_rss_vnext_mechanical_anchors_v1(e)
    assert len(r.socket_controls)==1 and len(r.anchors)==1
    c=r.socket_controls[0];a=r.anchors[0]
    assert c.control_index==0 and c.has_explicit_anchor
    assert c.quantized_xyz==(40,90,140)
    assert a.parent_kind=="CONTROL" and a.parent_index==0
    assert a.socket_control_index==0
    assert a.quantized_xyz==(65,96,128)
    assert r.candidate_mesh_sha256
    assert r.inference_evidence_hash==e.evidence_hash
    assert r.unified_checkpoint_sha256==e.checkpoint_sha256
    assert not r.holder_held_attachment_qualified
    assert not r.executable_socket_authority
    assert c.holder_face_index is None and c.held_face_index is None
    assert a.holder_face_index is None and a.held_face_index is None


def test_joint_anchor_is_not_automatically_a_socket_or_attachment():
    r=read_rss_vnext_mechanical_anchors_v1(
        _source(anchor_kind=0),
    )
    assert len(r.socket_controls)==1
    assert not r.socket_controls[0].has_explicit_anchor
    assert r.anchors[0].parent_kind=="JOINT"
    assert r.anchors[0].socket_control_index is None
    assert not r.source_contact_qualified


def test_non_socket_control_anchor_is_never_semantically_relabelled():
    r=read_rss_vnext_mechanical_anchors_v1(_source(control_type=2))
    assert r.socket_controls==()
    assert len(r.anchors)==1
    assert r.anchors[0].socket_control_index is None


@pytest.mark.parametrize("mod,reason",[
    (dict(anchor_kind=1,anchor_parent=1),"ANCHOR_CONTROL_PARENT_DOMAIN"),
    (dict(anchor_kind=0,anchor_parent=3),"ANCHOR_JOINT_PARENT_DOMAIN"),
    (dict(anchor_kind=1,anchor_coord=(40,90,256)),"QUANTIZED_ANCHOR_COORD_DOMAIN"),
    (dict(control_type=12),"CONTROL_TYPE_DOMAIN"),
])
def test_anchor_candidates_fail_closed_instead_of_reparenting(mod,reason):
    with pytest.raises(QualificationError,match=reason):
        read_rss_vnext_mechanical_anchors_v1(_source(**mod))


def test_evidence_rehash_or_wrong_checkpoint_does_not_mutate_source():
    evidence=_source()
    with pytest.raises(QualificationError,match="HASH_DRIFT"):
        read_rss_vnext_mechanical_anchors_v1(
            replace(evidence,checkpoint_sha256="f"*64),
        )
