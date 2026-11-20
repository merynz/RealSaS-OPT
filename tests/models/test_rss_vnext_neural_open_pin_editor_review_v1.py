from __future__ import annotations

"""Neural M* OPEN pin review over exact canonical mesh/rig/skin addresses."""
from dataclasses import replace

import pytest
import torch

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,CanonicalMeshVertexIR,canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2,CanonicalSkinRowIRV2,canonical_skin_lineage_hash_v2,
)
from models.rss_vnext.product_neural_open_pin_editor_review_v1 import (
    propose_open_pin_canonical_editor_review_v1,
)
from tests.compiler.test_mechanics_control_solver_v1 import _rig,_support
from tests.models.test_rss_vnext_open_pin_vertex_head_v1 import _case


def _canonical():
    model,result,seq,mesh,pin_output,teacher,ledger,train=_case()
    vertices=[]
    for i in range(mesh.rest_vertices_iris.shape[1]):
        vertices.append(CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=f"MV:{i}",
            support_binding=_support(i),
            component_id=f"COMP:{int(mesh.vertex_component_ids[i])}",
            P=tuple(float(x) for x in mesh.rest_vertices_iris[0,i]),
            source_candidate_vertex_id=f"RSSV:{i:08d}",
        ))
    faces=tuple(
        tuple(f"MV:{int(j)}" for j in row)
        for row in mesh.faces
    )
    edges=tuple(sorted({
        tuple(sorted((face[i],face[(i+1)%3])))
        for face in faces for i in range(3)
    }))
    canonical=CanonicalMeshIR(
        vertices=tuple(vertices),faces=faces,edges=edges,
        source_evidence_binding_hash="1"*64,
        partition_binding_hash="2"*64,
        carrier_policy_binding_hash="3"*64,
        static_qualification_hash="4"*64,
        carrier_evidence_hash="5"*64,
        carrier_topology_hash="6"*64,
        carrier_geometry_hash="7"*64,
        proposal_evidence_refs=("8"*64,),
        source_candidate_lineage_hash="9"*64,
        mesh_lineage_hash="",
        metadata={"authority":"COMPILER_CANONICAL_MESH"},
    )
    canonical=replace(
        canonical,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(canonical),
    )
    rig=_rig(canonical)
    rig=replace(rig,rig_lineage_hash="")
    rig=replace(rig,rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))
    skin=CanonicalSkinIRV2(
        rows=tuple(CanonicalSkinRowIRV2(
            canonical_mesh_vertex_id=f"MV:{i}",
            influences=(("J0",1.),),
            source_carrier_vertex_id=f"RSSV:{i:08d}",
        ) for i in range(len(vertices))),
        mesh_lineage_hash=canonical.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        mesh_carrier_evidence_hash=canonical.carrier_evidence_hash,
        mesh_carrier_topology_hash=canonical.carrier_topology_hash,
        mesh_carrier_geometry_hash=canonical.carrier_geometry_hash,
        carrier_skin_qualification_hash="f"*64,
        skin_code_proposal_hash="0"*64,
        skin_codec_architecture_id="synthetic-test-codec",
        skin_codec_checkpoint_hash="a"*64,
        skin_lineage_hash="",
        metadata={"authority":"COMPILER_CANONICAL_SKIN"},
    )
    skin=replace(skin,skin_lineage_hash=canonical_skin_lineage_hash_v2(skin))
    face=result.anchor_face_candidate
    # Preserve the exact original face/batch/anchor identity while choosing
    # two independent components as the model's *unqualified* suggestion.
    f_logits=torch.tensor([[15.,-20.,-20.]])
    h_logits=torch.tensor([[-20.,15.,-20.]])
    face=replace(face,held_face_logits=f_logits,holder_face_logits=h_logits)
    pin_logits=torch.full_like(pin_output.pin_logits,-8.)
    pin_logits[0,0]=pin_logits[0,1]=10.
    pin=replace(pin_output,pin_logits=pin_logits)
    return mesh,canonical,rig,skin,face,pin


def _preview(mesh,canonical,rig,skin,face,pin,**kw):
    return propose_open_pin_canonical_editor_review_v1(
        student=mesh,canonical=canonical,rig=rig,skin=skin,
        face_output=face,pin_output=pin,
        max_pinned_fraction=kw.pop("max_pinned_fraction",.8),
        **kw,
    )


def test_neural_open_pins_become_reviewable_canonical_ids_not_solver_authority():
    bundle=_canonical()
    record=_preview(*bundle)[0]
    assert record.state=="REVIEW_READY_NEURAL_PIN_AND_HOST_HYPOTHESIS"
    assert record.held_canonical_face==("MV:0","MV:1","MV:2")
    assert record.holder_canonical_face==("MV:3","MV:4","MV:5")
    assert record.candidate_pin_vertex_ids==("MV:0","MV:1")
    assert record.suggested_host_chain_id=="K0"
    assert record.candidate_component_id=="COMP:0"
    assert not record.source_pin_qualified
    assert not record.compiler_xpbd_activation_qualified
    assert not record.inferred_physical_parameters


def test_exact_mstar_geometry_face_and_parent_hash_cannot_be_silently_rebound():
    mesh,canonical,rig,skin,face,pin=_canonical()
    student_xyz=mesh.rest_vertices_iris.clone()
    student_xyz[0,0,2]+=0.01
    with pytest.raises(ValueError,match="REST_MSTAR_REPLAY_DRIFT"):
        _preview(replace(mesh,rest_vertices_iris=student_xyz),
                 canonical,rig,skin,face,pin)
    with pytest.raises(ValueError,match="MSTAR_OR_PARENT_DRIFT"):
        _preview(mesh,replace(canonical,mesh_lineage_hash="0"*64),
                 rig,skin,face,pin)
    bad=replace(canonical,faces=canonical.faces[::-1],mesh_lineage_hash="")
    bad=replace(bad,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(bad))
    badrig=replace(rig,mesh_lineage_hash=bad.mesh_lineage_hash,
                   rig_lineage_hash="")
    badrig=replace(badrig,rig_lineage_hash=canonical_rig_lineage_hash_v2(badrig))
    badskin=replace(skin,mesh_lineage_hash=bad.mesh_lineage_hash,
                    skin_lineage_hash="")
    badskin=replace(badskin,skin_lineage_hash=canonical_skin_lineage_hash_v2(badskin))
    with pytest.raises(ValueError,match="ORIENTED_FACE_REPLAY_DRIFT"):
        _preview(mesh,bad,badrig,badskin,face,pin)


def test_cross_component_pin_body_or_no_face_abstains_not_fabricates_xpbd():
    mesh,canonical,rig,skin,face,pin=_canonical()
    mixed=torch.full_like(pin.pin_logits,-8.)
    mixed[0,0]=mixed[0,3]=12.
    assert _preview(
        mesh,canonical,rig,skin,face,replace(pin,pin_logits=mixed),
    )[0].state=="ABSTAIN_CROSS_COMPONENT_OR_OVERPINNED"
    assert _preview(
        mesh,canonical,rig,skin,face,pin,max_pinned_fraction=.5,
    )[0].state=="ABSTAIN_CROSS_COMPONENT_OR_OVERPINNED"
    none=replace(face,held_face_logits=torch.tensor([[-20.,-20.,15.]]))
    assert _preview(
        mesh,canonical,rig,skin,none,pin,
    )[0].state=="ABSTAIN_NO_DIRECTED_FACE_PAIR"
