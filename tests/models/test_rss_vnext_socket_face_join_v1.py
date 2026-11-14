from __future__ import annotations

from dataclasses import asdict,replace

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_expert_fanout_v1 import (
    open_rss_vnext_expert_fanout_v1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.rss_vnext_mechanical_anchor_candidates_v1 import (
    read_rss_vnext_mechanical_anchors_v1,
)
from compiler.realsas_compiler_core.rss_vnext_socket_face_join_v1 import (
    RSSvNextExplicitSocketFaceRequestV1,
    compile_rss_vnext_socket_face_join_v1,
)
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    compile_rss_vnext_surface_candidates_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_model_v6 import RigProgramEventV6,RigTokenKindV6
from models.rss_vnext.inference_evidence_export_v1 import export_inference_evidence_v1
from models.rss_vnext.neural_scene_archive_v1 import encode_neural_scene_carrier_v1
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
from tests.models.test_rss_vnext_surface_decisions_v1 import (
    _fanout,_real_surface_fixture,
)
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import (
    _network,_images,
)


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(195)
    torch.set_num_threads(1)


def _events(*, socket_kind=7, anchor_kind=1):
    rows=[
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT,
                          coord_indices=(128,128,128)),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
        RigProgramEventV6(RigTokenKindV6.CONTROL,
                          aux_type_id=socket_kind,
                          coord_indices=(50,100,150)),
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHOR,
                          coord_indices=(91,113,138),
                          anchor_parent_kind=anchor_kind,
                          pointer_a=0 if anchor_kind==0 else -1,
                          pointer_b=0 if anchor_kind==1 else -1),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
        RigProgramEventV6(RigTokenKindV6.EOS),
    ]
    return tuple({**asdict(e),"kind":int(e.kind)} for e in rows)


def _with_controls(state,evidence,*,socket_kind=7,anchor_kind=1):
    linked=seal_inference_evidence_v1(replace(
        evidence,control_events=_events(socket_kind=socket_kind,
                                       anchor_kind=anchor_kind),
        evidence_hash="",
    ))
    carrier=encode_neural_scene_carrier_v1(state,evidence=linked)
    return linked,carrier


def _two_components():
    model=_network()
    rgba,cam=_images()
    with torch.no_grad():
        state,_,_=model.encode_observations(rgba,cam)
    # Two truly independent COMPONENT blocks, not two FACE_BREAK strips
    # in a single component. Same real student H(x) and heads as product.
    vertices=np.array([
        [-.34,-.34,0],[-.14,-.34,0],[-.34,-.14,0],
        [.14,.14,0],[.34,.14,0],[.14,.34,0],
    ],dtype=np.float64)
    cfg=TESSAConfigV1(coordinate_bins=model.cfg.tessa.coordinate_bins)
    parts=[]
    for i in (0,1):
        s=encode_adjacent_teacher_mesh_v1(
            vertices[i*3:(i+1)*3],np.array([[0,1,2]]),
            component_id=f"SYNTH_{i}",chart_id=f"SYNTH_{i}",cfg=cfg,
        ).token_ids
        parts.append(s)
    tokens=tuple(parts[0][:-1]+parts[1][1:])
    with torch.no_grad():
        mesh=model.build_mesh_candidate(
            state,tokens,source_authority="LEARNED_PREDICTED_TOPOLOGY",
        )
    assert tuple(mesh.face_component_ids.tolist())==(0,1)
    evidence=export_inference_evidence_v1(dict(
        state=state,mesh_candidate=mesh,mesh_status="UNQUALIFIED_MESH_CANDIDATE",
        mesh_failure=None,observed_view_count=2,
        auxiliary_view_authority="LEARNED_PREDICTED_ONLY",
        qualification_status="UNQUALIFIED_NEURAL_PROPOSAL",
    ),original_rgba=rgba,original_camera_features=cam,
        checkpoint_sha256="a"*64)
    return model.cfg,state,evidence


def _setup(*,two_components=True,socket_kind=7,anchor_kind=1):
    if two_components:
        cfg,state,evidence=_two_components()
    else:
        cfg,state,_,_,evidence,_=_real_surface_fixture(two_faces=True)
    bound,carrier=_with_controls(state,evidence,socket_kind=socket_kind,
                                 anchor_kind=anchor_kind)
    fanout=_fanout(cfg,bound,carrier)
    surface=compile_rss_vnext_surface_candidates_v1(fanout)
    mechanics=read_rss_vnext_mechanical_anchors_v1(bound)
    return fanout,surface,mechanics


def _req(held=0,holder=1,anchor=0):
    return RSSvNextExplicitSocketFaceRequestV1(
        anchor_index=anchor,held_face_index=held,holder_face_index=holder,
    )


def test_real_rss_components_axis_socket_and_directional_relation_join_is_unqualified():
    f,s,m=_setup()
    result=compile_rss_vnext_socket_face_join_v1(
        fanout=f,surface=s,mechanics=m,requests=(_req(),),
    )
    assert len(result.candidates)==1
    row=result.candidates[0]
    assert row.state=="UNQUALIFIED_ADDRESSED_CROSS_EXPERT_CANDIDATE"
    assert (row.held_face_index,row.holder_face_index)==(0,1)
    assert row.held_component_index==0 and row.holder_component_index==1
    assert row.anchor_event_index==m.anchors[0].event_index
    assert row.socket_control_event_index==m.socket_controls[0].event_index
    assert row.anchor_local_coord_bins==(91,113,138)
    assert row.anchor_coordinate_space=="PARENT_LOCAL_AX_V6_NOT_WORLD_SPACE"
    assert 0<=row.directed_attachment_affinity<=1
    assert row.held_unknown_mass>0 and row.holder_unknown_mass>0
    assert "QUALIFIED_PARENT_REST_FRAME" in row.missing_executable_witnesses
    assert "SOURCE_CONTACT_AND_HELDOUT_RELATIVE_TRANSPORT" in row.missing_executable_witnesses
    assert not row.executable_attachment_qualified
    assert not result.attachment_authority
    assert not result.source_contact_authority
    assert not result.canonical_mesh_authority
    assert result.inference_evidence_hash==f.evidence.evidence_hash
    assert result.neural_carrier_sha256==f.neural_carrier_sha256
    assert result.unified_checkpoint_sha256==f.evidence.checkpoint_sha256
    assert result.candidate_mesh_sha256==s.candidate_mesh_sha256
    assert result==compile_rss_vnext_socket_face_join_v1(
        fanout=f,surface=s,mechanics=m,requests=(_req(),),
    )


def test_inverse_direction_is_distinct_and_does_not_become_authorized():
    f,s,m=_setup()
    output=compile_rss_vnext_socket_face_join_v1(
        fanout=f,surface=s,mechanics=m,requests=(_req(),_req(1,0)),
    )
    assert len(output.candidates)==2
    a,b=output.candidates
    assert (a.held_face_index,a.holder_face_index)==(0,1)
    assert (b.held_face_index,b.holder_face_index)==(1,0)
    assert a.held_component_index==b.holder_component_index
    assert a.holder_component_index==b.held_component_index
    assert all(not c.source_contact_qualified for c in output.candidates)
    # A reverse query cannot automatically reverse a causal socket.
    assert all(not c.executable_attachment_qualified for c in output.candidates)


def test_one_tessa_component_is_not_falsely_split_into_held_and_holder():
    f,s,m=_setup(two_components=False)
    assert s.faces[0].candidate_component_index==s.faces[1].candidate_component_index
    output=compile_rss_vnext_socket_face_join_v1(
        fanout=f,surface=s,mechanics=m,requests=(_req(),),
    )
    row=output.candidates[0]
    assert row.state=="ABSTAIN_DISTINCT_MECHANICAL_OWNERSHIP_NOT_ESTABLISHED"
    assert "DISTINCT_MECHANICAL_OWNER_EVIDENCE" in row.missing_executable_witnesses


@pytest.mark.parametrize("kind,anchor,expected",[
    (2,1,"ABSTAIN_ANCHOR_NOT_SOCKET_CONTROL"),
    (7,0,"ABSTAIN_ANCHOR_NOT_SOCKET_CONTROL"),
])
def test_non_socket_or_joint_parent_does_not_become_a_socket(kind,anchor,expected):
    f,s,m=_setup(socket_kind=kind,anchor_kind=anchor)
    output=compile_rss_vnext_socket_face_join_v1(
        fanout=f,surface=s,mechanics=m,requests=(_req(),),
    )
    assert output.candidates[0].state==expected
    assert output.candidates[0].socket_control_index is None
    assert not output.attachment_authority


@pytest.mark.parametrize("requests,reason",[
    ((),"EXPLICIT_RESOURCE_BUDGET_OR_EMPTY_REQUEST"),
    ((_req(0,0),),"DIRECTED_PAIR_OR_ANCHOR_ADDRESS_INVALID"),
    ((_req(anchor=4),),"DIRECTED_PAIR_OR_ANCHOR_ADDRESS_INVALID"),
    ((_req(0,4),),"DIRECTED_PAIR_OR_ANCHOR_ADDRESS_INVALID"),
    ((_req(),_req()),"DUPLICATE_ANCHOR_FACE_PAIR"),
    (({ "held": 0 },),"DIRECTED_PAIR_OR_ANCHOR_ADDRESS_INVALID"),
])
def test_explicit_selection_is_bounded_and_does_not_expand_into_f_sq(requests,reason):
    f,s,m=_setup()
    with pytest.raises(QualificationError,match=reason):
        compile_rss_vnext_socket_face_join_v1(
            fanout=f,surface=s,mechanics=m,requests=requests,
        )


def test_forged_readouts_are_not_accepted_even_when_the_parent_hash_matches():
    f,s,m=_setup()
    bad_m=replace(m,anchors=(replace(m.anchors[0],
                    quantized_xyz=(0,0,0)),))
    with pytest.raises(QualificationError,match="SEALED_EXPERT_READOUT_DRIFT"):
        compile_rss_vnext_socket_face_join_v1(
            fanout=f,surface=s,mechanics=bad_m,requests=(_req(),),
        )
    bad_face=replace(s.faces[0],candidate_component_index=12)
    with pytest.raises(QualificationError,match="SEALED_EXPERT_READOUT_DRIFT"):
        compile_rss_vnext_socket_face_join_v1(
            fanout=f,surface=replace(s,faces=(bad_face,s.faces[1])),
            mechanics=m,requests=(_req(),),
        )


def test_no_unknown_frame_or_source_contact_can_be_promoted_by_a_probability():
    f,s,m=_setup()
    row=compile_rss_vnext_socket_face_join_v1(
        fanout=f,surface=s,mechanics=m,requests=(_req(),),
    ).candidates[0]
    assert row.directed_attachment_affinity is not None
    assert not any((row.source_contact_qualified,row.parent_frame_qualified,
                    row.holder_held_surface_binding_qualified,
                    row.executable_attachment_qualified))
    assert row.same_drawable_affinity is not None
    assert row.cut_affinity is not None
