from __future__ import annotations

"""Real trainable AXIS weights must see the exact deployed 47D conditioner."""

import pytest
import torch

from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from models.axis.scene_conditioning_v6 import (
    AXIS_V6_SCENE_ADAPTER_ID, AXIS_V6_SURFACE_SCENE_DIM,
    condition_axis_v6_from_rigging_surface,
)
from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6, AXISRigProgramModelV6, AXISSequenceV6,
    AXIS_V6_PRODUCT_ARCHITECTURE_ID, RigProgramEventV6, RigTokenKindV6,
)
from models.axis.product_training_v6 import (
    axis_product_teacher_sequence_sha256_v6,
    qualified_axis_product_teacher_objective_v6,
)
from models.axis.runtime_checkpoint_v2 import (
    save_axis_runtime_checkpoint_v2, load_axis_runtime_model_v2,
)
from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID


@pytest.fixture(autouse=True)
def deterministic_tiny_training():
    torch.manual_seed(31)
    torch.set_num_threads(1)


def _source_surface():
    return RiggingSurfaceIR(
        surface_nodes=(
            SurfaceNode(
                "S:0", (0., 0., 0.), (0,), ("source-a",), ("ray-a",),
                raster_bindings=((0,(0.,0.)),), derived_normal=(0.,0.,1.),
                validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            ),
            SurfaceNode(
                "S:1", (0., 1., 0.), (0,), ("source-b",), ("ray-b",),
                raster_bindings=((0,(0.,.5)),), derived_normal=(0.,0.,1.),
                validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            ),
        ),
        local_relations=(
            SurfaceRelation("R:0", "S:0", "S:1", "SOURCE_TOPOLOGY_NEIGHBOR", .9),
        ),
        geometry_lineage_hash="a"*64,
        metadata={"raster_coordinate_system":"NORMALIZED_GRID_XY"},
    )


def _product_axis():
    return AXISRigProgramModelV6(AXISRigProgramConfigV6(
        scene_dim=AXIS_V6_SURFACE_SCENE_DIM,
        hidden_dim=32, layers=1, heads=4, ff_mult=2,
        max_sequence_length=64, max_pointer_entities=16,
        skin_vocab_size=625, dropout=0.0,
    ))


def _teacher(*, control: bool = False):
    kinds=[
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT, coord_indices=(128,128,128)),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(
            RigTokenKindV6.FRAME, pointer_a=0,
            frame_6d=(1.,0.,0.,0.,1.,0.),
            dof_mask=(0.,0.,0.,1.,1.,1.,0.,0.,0.),
        ),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
    ]
    if control:
        kinds.extend([
            RigProgramEventV6(
                RigTokenKindV6.CONTROL, coord_indices=(128,128,128),
                pointer_a=-1, aux_type_id=1,
            ),
            RigProgramEventV6(RigTokenKindV6.CONTROL_TARGET, pointer_a=0),
        ])
    kinds.extend([
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
    ])
    kinds.extend(RigProgramEventV6(RigTokenKindV6.SKIN_CODE, skin_code_id=i) for i in range(4))
    kinds.append(RigProgramEventV6(RigTokenKindV6.EOS))
    return AXISSequenceV6.from_events(kinds,device=torch.device("cpu"))


def _receipt(source, teacher):
    t=condition_axis_v6_from_rigging_surface(source,device="cpu").surface_tensor
    return {
        "admission_status":"QUALIFIED_SOURCE_RIG_SKIN_TOKEN_TEACHER",
        "dataset_manifest_sha256":"3"*64,
        "source_qualification_key":"synthetic-unit-only",
        "architecture_id":AXIS_V6_PRODUCT_ARCHITECTURE_ID,
        "scene_adapter_id":AXIS_V6_SCENE_ADAPTER_ID,
        "surface_geometry_lineage_hash":source.geometry_lineage_hash,
        "surface_tensorization_hash":t.tensorization_hash,
        "teacher_sequence_sha256":axis_product_teacher_sequence_sha256_v6(teacher),
        "mira_codec_checkpoint_sha256":"b"*64,
        "teacher_semantics":"SOURCE_RIG_AND_PINNED_FSQ_NOT_INFERRED_CONTROLS",
    }


def _objective(model, source, teacher, receipt):
    return qualified_axis_product_teacher_objective_v6(
        model, source_surface=source, teacher=teacher, receipt=receipt,
        dataset_manifest_sha256="3"*64,
        source_qualification_key="synthetic-unit-only",
        codec_checkpoint_sha256="b"*64,
        tokens_per_joint=4,
    )


def test_product_conditioned_source_rig_skin_training_reloads_same_runtime_logits(tmp_path):
    source=_source_surface()
    teacher=_teacher()
    model=_product_axis().train()
    receipt=_receipt(source,teacher)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    result=_objective(model,source,teacher,receipt)
    result.loss.backward()
    assert model.kind_head.weight.grad is not None
    assert model.kind_head.weight.grad.abs().sum()>0
    assert model.skin_code_head.weight.grad is not None
    assert model.skin_code_head.weight.grad.abs().sum()>0
    assert model.coord_head.weight.grad is not None
    assert model.coord_head.weight.grad.abs().sum()>0
    optimizer.step()
    assert result.model_release_authority_granted is False
    scene=condition_axis_v6_from_rigging_surface(source,device="cpu")
    model.eval()
    with torch.no_grad():
        before=model(scene.scene_memory,teacher)
    path=tmp_path/"trained_axis_runtime_candidate.pt"
    digest=save_axis_runtime_checkpoint_v2(
        model,path,
        skin_codec_architecture_id=MIRA_V6_CODEC_ID,
        skin_codec_checkpoint_sha256="b"*64,
        skin_codebook_size=625,
        tokens_per_joint=4,
    )
    loaded,package=load_axis_runtime_model_v2(path,device=torch.device("cpu"))
    with torch.no_grad():
        after=loaded(scene.scene_memory,teacher)
    assert len(digest)==64
    assert package["scene_adapter"]["adapter_id"]==AXIS_V6_SCENE_ADAPTER_ID
    assert package["packaging_authority"]=="RUNTIME_PACKAGING_ONLY__NO_SCIENTIFIC_PASS"
    assert torch.equal(before["skin_code_logits"], after["skin_code_logits"])
    assert torch.equal(before["kind_logits"], after["kind_logits"])


def test_shared_scene_axis_cannot_be_mistaken_for_product_conditioning():
    source=_source_surface()
    teacher=_teacher()
    model=AXISRigProgramModelV6(AXISRigProgramConfigV6(
        scene_dim=32,hidden_dim=32,layers=1,heads=4,skin_vocab_size=625,
    )).train()
    receipt=_receipt(source,teacher)
    with pytest.raises(ValueError,match="CONDITIONING_WIDTH_NOT_RUNTIME_COMPATIBLE"):
        _objective(model,source,teacher,receipt)


def test_source_rig_teacher_does_not_grant_editable_control_without_intervention_receipt():
    source=_source_surface()
    teacher=_teacher(control=True)
    receipt=_receipt(source,teacher)
    model=_product_axis().train()
    with pytest.raises(ValueError,match="CONTROL_INTERVENTION_REQUIRED"):
        _objective(model,source,teacher,receipt)
    receipt["intervention_qualification_status"]="QUALIFIED_SOURCE_CONTROL_CONSEQUENCE"
    receipt["source_intervention_sha256"]="9"*64
    result=_objective(model,source,teacher,receipt)
    assert result.control_intervention_qualified is True
    assert torch.isfinite(result.loss)


def test_modified_teacher_surface_or_codec_cannot_enter_product_axis_training():
    source=_source_surface()
    teacher=_teacher()
    receipt=_receipt(source,teacher)
    model=_product_axis().train()
    receipt["mira_codec_checkpoint_sha256"]="e"*64
    with pytest.raises(ValueError,match="UNADMITTED_TEACHER_OR_CONDITIONING_DRIFT"):
        _objective(model,source,teacher,receipt)
    receipt=_receipt(source,teacher)
    receipt["surface_tensorization_hash"]="f"*64
    with pytest.raises(ValueError,match="UNADMITTED_TEACHER_OR_CONDITIONING_DRIFT"):
        _objective(model,source,teacher,receipt)
