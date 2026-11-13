from __future__ import annotations

"""Exact single-RSS-checkpoint socket-face head -> immutable archive -> Compiler.

These tests use a valid, explicit synthetic two-component TESSA teacher token
stream with live student H(x), AXIS causal hidden states and Semantic relations.
Nothing proves source-native held/holder identity or a product attachment.
"""
from dataclasses import asdict,replace
import json

import pytest
import torch

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_socket_face_archive_v1 import (
    RSSvNextNeuralSocketFaceArchiveV1,
    encode_rss_vnext_neural_socket_face_archive_v1,
    read_rss_vnext_neural_socket_face_archive_v1,
    seal_rss_vnext_neural_socket_face_archive_v1,
    compile_model_selected_rss_vnext_socket_face_join_v1,
)
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    compile_rss_vnext_surface_candidates_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_model_v6 import (
    AXISSequenceV6,RigProgramEventV6,RigTokenKindV6,
)
from models.rss_vnext.attachment_face_correspondence_v1 import (
    propose_socket_face_pairs_v1,
)
from models.rss_vnext.inference_anchor_face_export_v1 import (
    export_rss_vnext_neural_socket_face_archive_v1,
)
from models.rss_vnext.inference_evidence_export_v1 import (
    export_inference_evidence_v1,
)
from models.rss_vnext.neural_scene_archive_v1 import (
    encode_neural_scene_carrier_v1,
)
from tests.models.test_rss_vnext_attachment_face_correspondence_v1 import (
    _tokens,
)
from tests.models.test_rss_vnext_surface_decisions_v1 import _fanout
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import (
    _network,_images,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(522)
    torch.set_num_threads(1)


def _events():
    return (
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT,coord_indices=(128,128,128)),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
        RigProgramEventV6(RigTokenKindV6.CONTROL,aux_type_id=7,
                          coord_indices=(50,90,120)),
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHOR,
                          anchor_parent_kind=1,pointer_b=0,
                          coord_indices=(80,140,170)),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
        RigProgramEventV6(RigTokenKindV6.EOS),
    )


def _actual_student(*,force_direction=True):
    net=_network().eval()
    rgba,cam=_images()
    events=_events()
    seq=AXISSequenceV6.from_events(events,device=torch.device("cpu"))
    with torch.no_grad():
        state,aux,_=net.encode_observations(rgba,cam)
        mesh=net.build_mesh_candidate(state,_tokens(net))
        memory=net._control_memory(
            state,net._structural_memory(state,aux),mesh,
        )
        control_out=net.axis(memory,seq)
        prediction=net.anchor_face_correspondence(
            mesh=mesh,axis_hidden=control_out["hidden"],
            axis_sequence=seq,
            region_relation_logits=state.region_relation_logits,
        )
        if force_direction:
            # Fixture-only intervention to exercise a POSITIVE unqualified
            # proposal. No such logits are source-truth or claimed learned.
            prediction=replace(
                prediction,
                held_face_logits=torch.tensor([[60.,-60.,-60.]]),
                holder_face_logits=torch.tensor([[-60.,60.,-60.]]),
            )
    predicted=dict(
        state=state,mesh_candidate=mesh,
        mesh_status="UNQUALIFIED_MESH_CANDIDATE",mesh_failure=None,
        control_event_proposal=events,
        qualification_status="UNQUALIFIED_NEURAL_PROPOSAL",
        observed_view_count=2,
        auxiliary_view_authority="LEARNED_PREDICTED_ONLY",
        anchor_face_candidate=prediction,
    )
    evidence=export_inference_evidence_v1(
        predicted,original_rgba=rgba,original_camera_features=cam,
        checkpoint_sha256="a"*64,
    )
    carrier=encode_neural_scene_carrier_v1(state,evidence=evidence)
    reader=_fanout(net.cfg,evidence,carrier)
    surface=compile_rss_vnext_surface_candidates_v1(reader)
    assert len(surface.faces)==2
    return predicted,evidence,reader,surface


def _archive(*,force_direction=True):
    prediction,evidence,reader,surface=_actual_student(
        force_direction=force_direction,
    )
    raw=export_rss_vnext_neural_socket_face_archive_v1(
        prediction,evidence=evidence,
    )
    assert isinstance(raw,bytes)
    archive=read_rss_vnext_neural_socket_face_archive_v1(
        raw,evidence=evidence,
    )
    return prediction,evidence,reader,surface,raw,archive


def test_two_observed_images_real_student_axis_tessa_and_semantic_archive_to_compiler():
    p,e,reader,surface,raw,archive=_archive()
    assert archive.schema=="RealSaS.RSSvNext.NeuralSocketFaceArchive.v1"
    assert archive.checkpoint_sha256==e.checkpoint_sha256
    assert archive.inference_evidence_hash==e.evidence_hash
    assert archive.candidate_mesh_sha256==content_sha256(e.mesh)
    assert len(archive.proposals)==1
    assert archive.proposals[0]["held_face_index"]==0
    assert archive.proposals[0]["holder_face_index"]==1
    assert archive.proposals[0]["anchor_event_index"]==8
    assert not archive.executable_attachment_authority
    result=compile_model_selected_rss_vnext_socket_face_join_v1(
        fanout=reader,surface=surface,archive=archive,
    )
    assert result.candidate_join is not None
    row=result.candidate_join.candidates[0]
    assert row.state=="UNQUALIFIED_ADDRESSED_CROSS_EXPERT_CANDIDATE"
    assert (row.held_face_index,row.holder_face_index)==(0,1)
    assert row.held_component_index != row.holder_component_index
    assert row.anchor_coordinate_space=="PARENT_LOCAL_AX_V6_NOT_WORLD_SPACE"
    assert not row.executable_attachment_qualified
    assert not result.executable_attachment_authority
    assert not result.source_contact_qualified


def test_none_predictions_preserve_neural_abstain_on_full_compiler_read():
    p,e,reader,surface,raw,archive=_archive()
    none=replace(
        p["anchor_face_candidate"],
        held_face_logits=torch.tensor([[0.,0.,100.]]),
    )
    payload=export_rss_vnext_neural_socket_face_archive_v1(
        {**p,"anchor_face_candidate":none},evidence=e,
    )
    archive=read_rss_vnext_neural_socket_face_archive_v1(
        payload,evidence=e,
    )
    result=compile_model_selected_rss_vnext_socket_face_join_v1(
        fanout=reader,surface=surface,archive=archive,
    )
    assert result.candidate_join is None
    assert result.abstained_anchor_event_indices==(8,)
    assert not result.executable_attachment_authority


def test_tampered_bytes_and_cross_checkpoint_cannot_rebind_neural_socket_faces():
    p,e,reader,surface,raw,archive=_archive()
    tampered=raw.replace(b"\"socket_control_index\":0",
                         b"\"socket_control_index\":1")
    assert tampered!=raw
    with pytest.raises(QualificationError,match="ARCHIVE_HASH_DRIFT"):
        read_rss_vnext_neural_socket_face_archive_v1(tampered,evidence=e)
    from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
        seal_inference_evidence_v1,
    )
    different=seal_inference_evidence_v1(replace(
        e,checkpoint_sha256="f"*64,evidence_hash="",
    ))
    with pytest.raises(QualificationError,match="PARENT_AUTHORITY_OR_BYTES_DRIFT"):
        read_rss_vnext_neural_socket_face_archive_v1(raw,evidence=different)


def test_resigned_affinity_forgery_cannot_override_original_semantic_graph():
    p,e,reader,surface,raw,archive=_archive()
    row=dict(archive.proposals[0])
    true_affinity=row["predicted_directed_relation"]
    row["predicted_directed_relation"] = (
        0. if true_affinity>0.5 else 1.
    )
    forged=seal_rss_vnext_neural_socket_face_archive_v1(replace(
        archive,proposals=(row,),archive_hash="",
    ))
    raw=encode_rss_vnext_neural_socket_face_archive_v1(
        forged,evidence=e,
    )
    with pytest.raises(QualificationError,match="ORIGINAL_DIRECTIONAL_RELATION_REPLAY_DRIFT"):
        compile_model_selected_rss_vnext_socket_face_join_v1(
            fanout=reader,surface=surface,
            archive=read_rss_vnext_neural_socket_face_archive_v1(raw,evidence=e),
        )


def test_resealed_wrong_tessa_component_and_same_face_pair_fail_closed():
    p,e,reader,surface,raw,archive=_archive()
    row=dict(archive.proposals[0])
    row["same_tessa_component"]=True
    bad=seal_rss_vnext_neural_socket_face_archive_v1(replace(
        archive,proposals=(row,),archive_hash="",
    ))
    with pytest.raises(QualificationError,match="FACE_INDEX_OR_TESSA_COMPONENT_DRIFT"):
        encode_rss_vnext_neural_socket_face_archive_v1(bad,evidence=e)
    row=dict(archive.proposals[0])
    row["holder_face_index"]=row["held_face_index"]
    bad=seal_rss_vnext_neural_socket_face_archive_v1(replace(
        archive,proposals=(row,),archive_hash="",
    ))
    with pytest.raises(QualificationError,match="FACE_INDEX_OR_TESSA_COMPONENT_DRIFT"):
        encode_rss_vnext_neural_socket_face_archive_v1(bad,evidence=e)


def test_training_teacher_mesh_cannot_be_exported_as_learned_model_face_result():
    p,e,reader,surface,raw,archive=_archive()
    p=dict(p)
    p["mesh_candidate"]=replace(
        p["mesh_candidate"],source_authority="TEACHER_FORCED_TRAINING_ONLY",
    )
    with pytest.raises(ValueError,match="TEACHER_OR_AUTHORITY_FORBIDDEN"):
        export_rss_vnext_neural_socket_face_archive_v1(p,evidence=e)


def test_original_evidence_with_no_generated_mesh_has_no_socket_sidecar():
    p,e,reader,surface,raw,archive=_archive()
    empty={**p,"mesh_candidate":None,"control_event_proposal":(),
           "anchor_face_candidate":None}
    assert export_rss_vnext_neural_socket_face_archive_v1(empty,evidence=e) is None


def test_pinned_checkpoint_two_view_archive_receipt_and_compiler_reader_e2e(
    tmp_path,monkeypatch,
):
    """Forced valid *synthetic* topology is explicitly a smoke-test fixture.

    Production still requires a learned unseen topology and actual source
    native correspondence. All infrastructure is the REAL receipt writer,
    pinned checkpoint loader, original neural carrier and Compiler join.
    """
    from models.rss_vnext.unified_character_model import (
        RSSvNextUnifiedCharacterModel,RSSvNextPatchEncoderForTests,
    )
    from models.rss_vnext.inference_checkpoint_v1 import save_inference_checkpoint_v1
    from models.rss_vnext.inference_pipeline_v1 import (
        run_two_view_to_compiler_proposal_v1,
        read_compiler_proposal_receipt_v1,
        read_compiler_model_selected_socket_face_join_v1,
    )
    net=_network().eval()
    ckpt=tmp_path/"rss_unified.pt"
    digest=save_inference_checkpoint_v1(ckpt,net)
    rgba=torch.rand(1,2,4,28,28)
    _,cam=_images()

    def controlled_synthetic_proposal(
        self,rgba,camera_features,*,
        max_topology_tokens=None,generate_controls=False,
        executable_constraint_operators=None,
    ):
        assert generate_controls
        state,aux,_=self.encode_observations(rgba,camera_features)
        mesh=self.build_mesh_candidate(state,_tokens(self))
        events=_events()
        seq=AXISSequenceV6.from_events(events,device=rgba.device)
        control_memory=self._control_memory(
            state,self._structural_memory(state,aux),mesh,
        )
        axis_hidden=self.axis(control_memory,seq)["hidden"]
        candidate=self.anchor_face_correspondence(
            mesh=mesh,axis_hidden=axis_hidden,axis_sequence=seq,
            region_relation_logits=state.region_relation_logits,
        )
        candidate=replace(
            candidate,
            held_face_logits=torch.tensor([[80.,-80.,-80.]],device=rgba.device),
            holder_face_logits=torch.tensor([[-80.,80.,-80.]],device=rgba.device),
        )
        return dict(
            state=state,mesh_candidate=mesh,
            mesh_status="UNQUALIFIED_MESH_CANDIDATE",
            mesh_failure=None,control_event_proposal=events,
            qualification_status="UNQUALIFIED_NEURAL_PROPOSAL",
            observed_view_count=2,
            auxiliary_view_authority="LEARNED_PREDICTED_ONLY",
            anchor_face_candidate=candidate,
        )

    monkeypatch.setattr(
        RSSvNextUnifiedCharacterModel,"propose_teacher_free",
        controlled_synthetic_proposal,
    )
    output=tmp_path/"model_receipt"
    receipt=run_two_view_to_compiler_proposal_v1(
        checkpoint_path=ckpt,expected_checkpoint_sha256=digest,
        vision_encoder=RSSvNextPatchEncoderForTests(net.cfg.iris.image_token_dim),
        observed_rgba=rgba,camera_features=cam,
        output_directory=output,generate_controls=True,product_mode=False,
    )
    assert receipt["anchor_face_archive_schema"]=="RealSaS.RSSvNext.NeuralSocketFaceArchive.v1"
    assert receipt["anchor_face_executable_attachment_qualified"] is False
    sidecar=output/"rss_vnext_neural_anchor_face.json"
    assert sidecar.is_file()
    actual,evidence=read_compiler_proposal_receipt_v1(
        output,expected_checkpoint_sha256=digest,
    )
    assert actual==receipt
    compiler=read_compiler_model_selected_socket_face_join_v1(
        output,expected_checkpoint_sha256=digest,model_config=net.cfg,
    )
    assert compiler.candidate_join is not None
    assert len(compiler.candidate_join.candidates)==1
    row=compiler.candidate_join.candidates[0]
    assert (row.held_face_index,row.holder_face_index)==(0,1)
    assert not row.executable_attachment_qualified
    assert not compiler.executable_attachment_authority
    # Even the exact declared sidecar bytes must be protected by the
    # source/receipt digest; a rehashed neural JSON alone is insufficient.
    before=sidecar.read_bytes()
    sidecar.write_bytes(before+b"tamper")
    with pytest.raises(RuntimeError,match="RECEIPT_ANCHOR_FACE_ARCHIVE_BYTES_DRIFT"):
        read_compiler_proposal_receipt_v1(
            output,expected_checkpoint_sha256=digest,
        )


def test_undeclared_socket_face_file_cannot_be_spliced_into_abstain_receipt(
    tmp_path,
):
    from models.rss_vnext.unified_character_model import RSSvNextPatchEncoderForTests
    from models.rss_vnext.inference_checkpoint_v1 import save_inference_checkpoint_v1
    from models.rss_vnext.inference_pipeline_v1 import (
        run_two_view_to_compiler_proposal_v1,read_compiler_proposal_receipt_v1,
    )
    net=_network().eval()
    ckpt=tmp_path/"ckpt.pt"
    digest=save_inference_checkpoint_v1(ckpt,net)
    rgba=torch.rand(1,2,4,28,28)
    _,cam=_images()
    root=tmp_path/"no_controls"
    result=run_two_view_to_compiler_proposal_v1(
        checkpoint_path=ckpt,expected_checkpoint_sha256=digest,
        vision_encoder=RSSvNextPatchEncoderForTests(net.cfg.iris.image_token_dim),
        observed_rgba=rgba,camera_features=cam,output_directory=root,
        generate_controls=False,product_mode=False,max_topology_tokens=12,
    )
    assert "anchor_face_archive_file_sha256" not in result
    read_compiler_proposal_receipt_v1(root,expected_checkpoint_sha256=digest)
    (root/"rss_vnext_neural_anchor_face.json").write_text("{}")
    with pytest.raises(RuntimeError,match="RECEIPT_UNDECLARED_ANCHOR_FACE_ARCHIVE"):
        read_compiler_proposal_receipt_v1(root,expected_checkpoint_sha256=digest)
