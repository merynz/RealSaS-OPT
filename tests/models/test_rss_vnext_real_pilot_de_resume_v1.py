from __future__ import annotations

"""Synthetic-only D/E restart acceptance: no fabricated GPU/teacher admission."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn

from models.rss_vnext.mira_embedded_codec_source_training_v1 import (
    RSSvNextEmbeddedMIRASourceCodecTrainerV1,
)

from models.rss_vnext.product_real_pilot_resume_v1 import (
    PilotDEPhaseResumeIdentityV1,
    pilot_phase_plan_sha256_v1,
    read_pilot_de_resume_v1, write_pilot_de_resume_v1,
)
from models.rss_vnext.product_real_pilot_codec_launcher_v1 import (
    PilotEmbeddedCodecPlanV1, PreparedPilotCodecPhaseV1,
    save_real_pilot_embedded_codec_resume_v1,
    load_real_pilot_embedded_codec_resume_v1,
    execute_real_pilot_embedded_codec_phase_v1,
    export_real_pilot_embedded_codec_handoff_v1,
)
from models.rss_vnext.product_real_pilot_structural_launcher_v1 import (
    PilotStructuralEPhasePlanV1, PreparedPilotStructuralPhaseV1,
    save_real_pilot_structural_e_resume_v1,
    load_real_pilot_structural_e_resume_v1,
    execute_real_pilot_structural_e_phase_v1,
    export_real_pilot_structural_e_handoff_v1,
)

COHORT="a"*64
PRIOR="b"*64
OWNER="c"*64
OBJECTIVE="d"*64


class TinyDCodecTrainer:
    def __init__(self, step=0):
        self.step=step
        self.optimizer=SimpleNamespace(state={})
        self.optimizer_ownership=SimpleNamespace(binding_sha256=OWNER)
        self.phase_objective_policy_sha256=OBJECTIVE
        self.exported=[]

    def save_checkpoint(self,path):
        Path(path).write_text(json.dumps({"step":self.step}))

    def load_checkpoint(self,path):
        self.step=json.loads(Path(path).read_text())["step"]

    def export_phase_handoff(self,path):
        self.exported.append(str(path))
        return "D_WEIGHTS_EXPORTED"


class TinyEStructuralTrainer:
    def __init__(self, step=0):
        self.global_step=step
        self.optimizer=SimpleNamespace(state={})
        self.objective_policy_sha256=OBJECTIVE
        self.exported=[]

    def save_checkpoint(self,path,*,sampler_cursor):
        Path(path).write_text(json.dumps({
            "step":self.global_step,"cursor":sampler_cursor,
        }))

    def load_checkpoint(self,path):
        payload=json.loads(Path(path).read_text())
        self.global_step=payload["step"]
        return payload["cursor"]

    def export_phase_handoff(self,path,*,target_kind):
        assert target_kind=="STRUCTURAL"
        self.exported.append(str(path))
        return "E_WEIGHTS_EXPORTED"


def make_d(*, step=0, plan=None, prior=PRIOR):
    return PreparedPilotCodecPhaseV1(
        trainer=TinyDCodecTrainer(step),
        plan=plan or PilotEmbeddedCodecPlanV1(
            optimizer_steps=5,learning_rate=5e-5,
        ),
        prior_source_model_sha256=prior,
        source_cohort_sha256=COHORT,
    )


def make_e(*, step=0, plan=None, prior=PRIOR):
    return PreparedPilotStructuralPhaseV1(
        stage="MSTAR_GEOMETRY_BOOTSTRAP",
        plan=plan or PilotStructuralEPhasePlanV1(
            stage="MSTAR_GEOMETRY_BOOTSTRAP",
            planned_optimizer_steps=7,learning_rate=3e-5,
        ),
        trainer=TinyEStructuralTrainer(step),
        phase_owner_sha256="e"*64,
        initial_optimizer_binding_sha256=OWNER,
        incoming_handoff_model_sha256=prior,
        source_cohort_sha256=COHORT,
    )


def test_receipt_roundtrip_and_all_byte_identity_gates(tmp_path):
    plan=PilotEmbeddedCodecPlanV1(optimizer_steps=5,learning_rate=1e-4)
    identity=PilotDEPhaseResumeIdentityV1(
        "D_EMBEDDED_MIRA_CODEC",COHORT,
        pilot_phase_plan_sha256_v1(plan),PRIOR,OWNER,OBJECTIVE,
    )
    file=tmp_path/"codec.pt"
    file.write_bytes(b"synthetic optimizer checkpoint")
    sidecar=write_pilot_de_resume_v1(
        file,identity=identity,step=2,sampler_cursor=29,
    )
    assert read_pilot_de_resume_v1(
        file,identity=identity,max_steps=5,
    )==(2,29)
    assert sidecar.name.endswith(".pilot-resume.json")
    with pytest.raises(ValueError,match="PROVENANCE_OR_BYTES_DRIFT"):
        read_pilot_de_resume_v1(
            file,identity=replace(identity,incoming_model_state_sha256="f"*64),
            max_steps=5,
        )
    with pytest.raises(ValueError,match="PROVENANCE_OR_BYTES_DRIFT"):
        read_pilot_de_resume_v1(file,identity=identity,max_steps=1)
    file.write_bytes(b"corrupted state")
    with pytest.raises(ValueError,match="PROVENANCE_OR_BYTES_DRIFT"):
        read_pilot_de_resume_v1(file,identity=identity,max_steps=5)


def test_receipt_rejects_symlink_bool_step_and_torn_checkpoint(tmp_path):
    file=tmp_path/"resume.pt"
    file.write_bytes(b"fixture bytes")
    identity=PilotDEPhaseResumeIdentityV1(
        "E_MSTAR_GEOMETRY_BOOTSTRAP",COHORT,"1"*64,PRIOR,OWNER,OBJECTIVE,
    )
    with pytest.raises(ValueError,match="CURSOR_INVALID"):
        write_pilot_de_resume_v1(file,identity=identity,step=True,sampler_cursor=0)
    sidecar=write_pilot_de_resume_v1(
        file,identity=identity,step=1,sampler_cursor=1,
    )
    sidecar.unlink()
    with pytest.raises(ValueError,match="FILE_MISSING_OR_SYMLINK"):
        read_pilot_de_resume_v1(file,identity=identity,max_steps=5)
    other=tmp_path/"symlink.pt"
    try:
        other.symlink_to(file)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError,match="FILE_MISSING_OR_SYMLINK"):
        write_pilot_de_resume_v1(
            other,identity=identity,step=1,sampler_cursor=2,
        )


def test_codec_plan_and_prior_handoff_locked_on_restart(tmp_path):
    source=make_d(step=2)
    checkpoint=tmp_path/"D.pt"
    save_real_pilot_embedded_codec_resume_v1(
        source,checkpoint_path=checkpoint,sampler_cursor=17,
    )
    restored=load_real_pilot_embedded_codec_resume_v1(
        make_d(),checkpoint_path=checkpoint,
    )
    assert (restored.trainer.step,restored.verified_resume_step,
            restored.verified_resume_cursor)==(2,2,17)
    with pytest.raises(ValueError,match="PROVENANCE_OR_BYTES_DRIFT"):
        load_real_pilot_embedded_codec_resume_v1(
            make_d(prior="0"*64),checkpoint_path=checkpoint,
        )
    with pytest.raises(ValueError,match="PROVENANCE_OR_BYTES_DRIFT"):
        load_real_pilot_embedded_codec_resume_v1(
            make_d(plan=PilotEmbeddedCodecPlanV1(
                optimizer_steps=6,learning_rate=5e-5,
            )),checkpoint_path=checkpoint,
        )
    with pytest.raises(RuntimeError,match="DESTINATION_NOT_FRESH"):
        load_real_pilot_embedded_codec_resume_v1(
            source,checkpoint_path=checkpoint,
        )


def test_codec_resume_finishes_remaining_only_and_seals_d_to_e(monkeypatch,tmp_path):
    import models.rss_vnext.product_real_pilot_codec_launcher_v1 as module
    original=make_d(step=3)
    f=tmp_path/"D.pt"
    save_real_pilot_embedded_codec_resume_v1(
        original,checkpoint_path=f,sampler_cursor=21,
    )
    restored=load_real_pilot_embedded_codec_resume_v1(
        make_d(),checkpoint_path=f,
    )
    calls=[]
    def runner(*,trainer,staged_examples,source_index,cohort,max_steps):
        calls.append(max_steps)
        trainer.step+=max_steps
        return [{"step":trainer.step}]
    monkeypatch.setattr(module,"run_source_qualified_pilot_updates_v1",runner)
    with pytest.raises(ValueError,match="RESUME_REQUIRES_CHECKPOINT"):
        execute_real_pilot_embedded_codec_phase_v1(
            replace(restored,verified_resume_step=None),
            examples=(),source_index={},cohort=SimpleNamespace(cohort_sha256=COHORT),
        )
    with pytest.raises(RuntimeError,match="UNFINISHED_D_TO_E"):
        export_real_pilot_embedded_codec_handoff_v1(
            restored,path=tmp_path/"final.pt",
        )
    with pytest.raises(ValueError,match="RESUME_REQUIRES_CHECKPOINT"):
        execute_real_pilot_embedded_codec_phase_v1(
            restored,examples=(),source_index={},
            cohort=SimpleNamespace(cohort_sha256=COHORT),
            resume_sampler_cursor=20,
        )
    report=execute_real_pilot_embedded_codec_phase_v1(
        restored,examples=(),source_index={},
        cohort=SimpleNamespace(cohort_sha256=COHORT),
        resume_sampler_cursor=21,
    )
    assert calls==[2] and report[0]["step"]==5
    assert export_real_pilot_embedded_codec_handoff_v1(
        restored,path=tmp_path/"final.pt",
    )=="D_WEIGHTS_EXPORTED"


def test_structural_restart_pins_plan_optimizers_and_resume_cursor(tmp_path):
    original=make_e(step=4)
    checkpoint=tmp_path/"E.pt"
    save_real_pilot_structural_e_resume_v1(
        original,checkpoint_path=checkpoint,sampler_cursor=123,
    )
    restored=load_real_pilot_structural_e_resume_v1(
        make_e(),checkpoint_path=checkpoint,
    )
    assert (
        restored.trainer.global_step,restored.verified_resume_step,
        restored.verified_resume_cursor,
    )==(4,4,123)
    with pytest.raises(ValueError,match="PROVENANCE_OR_BYTES_DRIFT"):
        load_real_pilot_structural_e_resume_v1(
            make_e(plan=replace(original.plan,gradient_mode="BOUNDED_SHARED_PCGRAD")),
            checkpoint_path=checkpoint,
        )
    with pytest.raises(ValueError,match="PROVENANCE_OR_BYTES_DRIFT"):
        load_real_pilot_structural_e_resume_v1(
            make_e(prior="f"*64),checkpoint_path=checkpoint,
        )


def test_structural_resume_remaining_only_and_finished_handoff(monkeypatch,tmp_path):
    import models.rss_vnext.product_real_pilot_structural_launcher_v1 as module
    original=make_e(step=4)
    file=tmp_path/"E.pt"
    save_real_pilot_structural_e_resume_v1(
        original,checkpoint_path=file,sampler_cursor=11,
    )
    restored=load_real_pilot_structural_e_resume_v1(
        make_e(),checkpoint_path=file,
    )
    calls=[]
    def runner(*,trainer,staged_examples,source_index,cohort,max_steps):
        calls.append(max_steps)
        trainer.global_step+=max_steps
        return [{"step":trainer.global_step}]
    monkeypatch.setattr(module,"run_source_qualified_pilot_updates_v1",runner)
    with pytest.raises(ValueError,match="RESUME_EXPECTS_CHECKPOINT"):
        execute_real_pilot_structural_e_phase_v1(
            replace(restored,verified_resume_step=None),
            examples=(),source_index={},cohort=SimpleNamespace(cohort_sha256=COHORT),
        )
    with pytest.raises(RuntimeError,match="UNFINISHED_PHASE_HANDOFF"):
        export_real_pilot_structural_e_handoff_v1(
            restored,path=tmp_path/"next-e.pt",
        )
    with pytest.raises(ValueError,match="RESUME_EXPECTS_CHECKPOINT"):
        execute_real_pilot_structural_e_phase_v1(
            restored,examples=(),source_index={},
            cohort=SimpleNamespace(cohort_sha256=COHORT),
            resume_sampler_cursor=10,
        )
    report=execute_real_pilot_structural_e_phase_v1(
        restored,examples=(),source_index={},
        cohort=SimpleNamespace(cohort_sha256=COHORT),
        resume_sampler_cursor=11,
    )
    assert calls==[3] and report[0]["step"]==7
    assert export_real_pilot_structural_e_handoff_v1(
        restored,path=tmp_path/"next-e.pt",
    )=="E_WEIGHTS_EXPORTED"

class TinyCodecNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.mira=nn.Module()
        self.mira.codec=nn.Linear(2,1)
        self.mira.spatial_decoder=nn.Linear(2,1)


def test_d_codec_gradient_guard_blocks_disconnected_deployed_decoder():
    net=TinyCodecNetwork()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.01)
    trainer=object.__new__(RSSvNextEmbeddedMIRASourceCodecTrainerV1)
    trainer.model=net
    trainer.optimizer=optimizer
    trainer.step=1  # Avoid periodic unrelated frozen-upstream hash check.
    trainer._trainable_mira_names=frozenset(n for n,_ in net.named_parameters())
    trainer.gradient_court_policy=SimpleNamespace(every_n_steps=0)
    trainer.skin_decoder_mode="POINT_QUERY_CROSS_ATTENTION"
    trainer._assert_scope=lambda:None
    trainer._assert_frozen_state_identity=lambda:None

    trainer.objective=lambda batch:SimpleNamespace(
        loss=net.mira.codec.weight.square().sum(),
    )
    old=net.mira.codec.weight.detach().clone()
    with pytest.raises(RuntimeError,match="REQUIRED_TRAINABLE_GRADIENT_MISSING"):
        trainer.train_step(None)
    assert trainer.step==1
    assert not optimizer.state
    assert torch.equal(net.mira.codec.weight,old)

    trainer.objective=lambda batch:SimpleNamespace(
        loss=(net.mira.codec.weight.square().sum()
              +net.mira.spatial_decoder.weight.square().sum()),
    )
    trainer.train_step(None)
    assert trainer.step==2
    assert not torch.equal(net.mira.codec.weight,old)
