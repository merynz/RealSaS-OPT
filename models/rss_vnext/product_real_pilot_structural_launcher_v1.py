from __future__ import annotations

"""Live RSS.vNext E1–E5 structural phase launcher for a user-split pilot.

This binds the already built product-seeded trainer, curriculum, source-only
registries, same-H trainable-parameter ownership and exact weights-only
handoff into one ordered execution entry. Unlike a "plan" or a dataclass-only
gate, it constructs a REAL optimizer and a REAL structural trainer and can
execute source-qualified training batches through that trainer.

No source is selected/reassigned here. The caller supplies its own cohort,
manifest and independently authorized source receipts. No training runs at
module import or during launcher construction.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping
import math

import torch

from .product_real_pilot_resume_v1 import (
    PilotDEPhaseResumeIdentityV1,
    pilot_phase_plan_sha256_v1,
    read_pilot_de_resume_v1, write_pilot_de_resume_v1,
)
from .product_real_pilot_cohort_v1 import (
    RealPilotAssetV1, RealPilotCohortReceiptV1,
)
from .product_real_pilot_execution_v1 import (
    STAGES, PilotLedgerV1, RealPilotTrainingEnvelopeV1,
    run_source_qualified_pilot_updates_v1, _file_digest,
    requalify_pilot_ledger_artifacts_v1,
)
from .product_seeded_phase_ownership_v1 import (
    ProductSeededPhaseOwnerPolicyV1, PHASES,
)
from .product_multitask_curriculum_v1 import ProductSeededCurriculumV1
from .product_multitask_gradient_control_v1 import (
    ProductMultitaskGradientPolicyV1, MODES,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel
from .training_engine import RSSvNextStructuralTrainer

TRAINING_PROFILE = "PRODUCT_SEEDED_IRIS_TESSA_V1"
STAGE_KINDS = {"D_EMBEDDED_MIRA_CODEC":"MIRA_CODEC",
               **{stage:"STRUCTURAL" for stage in STAGES if stage.startswith("E_")}}


@dataclass(frozen=True)
class PilotStructuralEPhasePlanV1:
    stage:str
    planned_optimizer_steps:int
    learning_rate:float
    gradient_mode:str="WEIGHTED_SUM"
    gradient_projection_strength:float=0.5
    cagrad_conflict_aversion:float=0.5

    def validate(self)->None:
        if (
            self.stage not in PHASES
            or type(self.planned_optimizer_steps) is not int
            or not 1 <= self.planned_optimizer_steps <= 10_000_000
            or type(self.learning_rate) not in (int,float)
            or not math.isfinite(self.learning_rate)
            or not 1e-8<=self.learning_rate<=1e-2
            or self.gradient_mode not in MODES
            or type(self.gradient_projection_strength) not in (int,float)
            or not math.isfinite(self.gradient_projection_strength)
            or not 0<=self.gradient_projection_strength<=1
            or type(self.cagrad_conflict_aversion) not in (int,float)
            or not math.isfinite(self.cagrad_conflict_aversion)
            or not 0<=self.cagrad_conflict_aversion<1
        ):
            raise ValueError("RSS_VNEXT_PILOT_E_PHASE_POLICY_INVALID")


@dataclass(frozen=True)
class PreparedPilotStructuralPhaseV1:
    stage:str
    plan:PilotStructuralEPhasePlanV1
    trainer:RSSvNextStructuralTrainer
    phase_owner_sha256:str
    initial_optimizer_binding_sha256:str
    incoming_handoff_model_sha256:str
    source_cohort_sha256:str
    verified_resume_step:int|None=None
    verified_resume_cursor:int=0


def activate_real_pilot_e_training_mode_v1(model:RSSvNextUnifiedCharacterModel)->None:
    """D's codec-first trainer uses model.eval; E MUST reactivate shared H.

    Frozen vision, FSQ codec and point-query skin decoder stay in eval mode
    to prevent dropout/batch-stat mutation. AXIS/MIRA active heads, Graphics,
    IRIS and seeded TESSA are allowed to train in their selected E phase.
    """
    if not isinstance(model,torch.nn.Module):
        raise TypeError("RSS_VNEXT_PILOT_E_TRAIN_MODE_MODEL_REQUIRED")
    model.train()
    for module in (
        model.vision,
        model.mira.codec,
        model.mira.spatial_decoder,
    ):
        module.eval()
    if (
        not model.training or model.vision.training
        or model.mira.codec.training or model.mira.spatial_decoder.training
    ):
        raise RuntimeError("RSS_VNEXT_PILOT_E_FROZEN_EVAL_MODE_DRIFT")


def build_real_pilot_structural_e_phase_v1(
    *,model:RSSvNextUnifiedCharacterModel,
    cohort:RealPilotCohortReceiptV1,
    ledger:PilotLedgerV1,
    plan:PilotStructuralEPhasePlanV1,
    prior_handoff_file:str|Path,
    source_trainer_kwargs:Mapping[str,Any],
)->PreparedPilotStructuralPhaseV1:
    """Construct a REAL seeded structural trainer after exact phase handoff.

    Requires an already completed audited D or E stage with the actual
    weights-only handoff file as its ledger output. C/D architecture handoff
    remains a separate existing trainer implementation; neither is faked here.

    The source_trainer_kwargs are the *qualified caller-supplied* teacher
    registries, signed/Open layer policy, loss weights and source masks.
    No default teacher or synthetic placeholder is injected.
    """
    plan.validate()
    ledger.validate()
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or not isinstance(source_trainer_kwargs,Mapping)
        or not cohort.source_evidence_bytes_verified
        or cohort.cohort_sha256!=ledger.cohort_sha256
        or ledger.next_stage!="E_"+plan.stage
        or not ledger.completed
    ):
        raise ValueError("RSS_VNEXT_PILOT_E_PHASE_LAUNCH_NOT_ADMITTED")
    requalify_pilot_ledger_artifacts_v1(ledger,cohort=cohort)
    prior=ledger.completed[-1]
    if prior.stage not in STAGE_KINDS:
        raise ValueError("RSS_VNEXT_PILOT_E_PHASE_PREVIOUS_STAGE_NOT_READY")
    prior.validate()
    # The only allowed handoff is the prior stage's sealed artifact, never
    # a differently sourced model whose tensors merely have same shapes.
    source=Path(prior_handoff_file).expanduser().resolve()
    expected=Path(prior.output_checkpoint_path).expanduser().resolve()
    if (
        source!=expected or source.is_symlink() or not source.is_file()
        or _file_digest(source)!=prior.output_checkpoint_sha256
    ):
        raise ValueError("RSS_VNEXT_PILOT_E_PHASE_HANDOFF_PATH_OR_BYTES_DRIFT")
    forbidden={
        "dataset_manifest_sha256","training_profile",
        "scene_conditioning_policy","axis_training_policy",
        "product_seeded_owner_policy","product_curriculum",
        "product_gradient_route_policy",
    }
    if set(source_trainer_kwargs).intersection(forbidden):
        raise ValueError("RSS_VNEXT_PILOT_E_PHASE_RESERVED_TRAINER_FIELD")
    owner=ProductSeededPhaseOwnerPolicyV1.standard(plan.stage)
    owner.activate_before_optimizer_v1(model)
    activate_real_pilot_e_training_mode_v1(model)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(plan.learning_rate),betas=(.9,.999),
        weight_decay=.01,
    )
    binding=owner.assert_qualified_optimizer(model,optimizer)
    grad_policy=ProductMultitaskGradientPolicyV1(
        mode=plan.gradient_mode,
        projection_strength=plan.gradient_projection_strength,
        cagrad_conflict_aversion=plan.cagrad_conflict_aversion,
    )
    grad_policy.validate()
    trainer=RSSvNextStructuralTrainer(
        model,optimizer,
        dataset_manifest_sha256=cohort.cohort_sha256,
        training_profile=TRAINING_PROFILE,
        scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
        axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
        product_seeded_owner_policy=owner,
        product_curriculum=ProductSeededCurriculumV1(plan.stage),
        product_gradient_route_policy=grad_policy,
        **dict(source_trainer_kwargs),
    )
    imported=trainer.import_phase_handoff(
        source,
        expected_source_kind=STAGE_KINDS[prior.stage],
        expected_source_objective_policy_sha256=prior.objective_policy_sha256,
        expected_source_optimizer_binding_sha256=prior.optimizer_ownership_sha256,
    )
    if (
        imported.source_global_step!=prior.trained_steps
        or imported.dataset_manifest_sha256!=cohort.cohort_sha256
        or trainer.global_step!=0
        or optimizer.state
    ):
        raise RuntimeError("RSS_VNEXT_PILOT_E_PHASE_HANDOFF_RESTART_DRIFT")
    return PreparedPilotStructuralPhaseV1(
        stage=plan.stage,plan=plan,trainer=trainer,
        phase_owner_sha256=owner.digest,
        initial_optimizer_binding_sha256=binding,
        incoming_handoff_model_sha256=imported.model_state_sha256,
        source_cohort_sha256=cohort.cohort_sha256,
    )


def _structural_pilot_resume_identity_v1(
    phase:PreparedPilotStructuralPhaseV1,
)->PilotDEPhaseResumeIdentityV1:
    return PilotDEPhaseResumeIdentityV1(
        stage="E_"+phase.stage,
        cohort_sha256=phase.source_cohort_sha256,
        plan_sha256=pilot_phase_plan_sha256_v1(phase.plan),
        incoming_model_state_sha256=phase.incoming_handoff_model_sha256,
        optimizer_binding_sha256=phase.initial_optimizer_binding_sha256,
        objective_policy_sha256=phase.trainer.objective_policy_sha256,
    )


def save_real_pilot_structural_e_resume_v1(
    phase:PreparedPilotStructuralPhaseV1,
    *,checkpoint_path:str|Path,sampler_cursor:int,
)->Path:
    """Save real E optimizer+model+RNG and bind phase plan/source receipt."""
    current=phase.trainer.global_step
    if not 0<=current<=phase.plan.planned_optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_E_RESUME_PROGRESS_INVALID")
    identity=_structural_pilot_resume_identity_v1(phase)
    identity.validate()
    phase.trainer.save_checkpoint(checkpoint_path,sampler_cursor=sampler_cursor)
    return write_pilot_de_resume_v1(
        checkpoint_path,identity=identity,step=current,
        sampler_cursor=sampler_cursor,
    )


def load_real_pilot_structural_e_resume_v1(
    phase:PreparedPilotStructuralPhaseV1,
    *,checkpoint_path:str|Path,
)->PreparedPilotStructuralPhaseV1:
    """Do not import E AdamW moments until lineage, stage and plan match."""
    if phase.trainer.global_step!=0 or phase.trainer.optimizer.state:
        raise RuntimeError("RSS_VNEXT_PILOT_E_RESUME_DESTINATION_NOT_FRESH")
    expected,cursor=read_pilot_de_resume_v1(
        checkpoint_path,identity=_structural_pilot_resume_identity_v1(phase),
        max_steps=phase.plan.planned_optimizer_steps,
    )
    restored_cursor=phase.trainer.load_checkpoint(checkpoint_path)
    if phase.trainer.global_step!=expected or restored_cursor!=cursor:
        raise RuntimeError("RSS_VNEXT_PILOT_E_RESUME_STEP_OR_CURSOR_DRIFT")
    return replace(
        phase,verified_resume_step=expected,verified_resume_cursor=cursor,
    )


def export_real_pilot_structural_e_handoff_v1(
    phase:PreparedPilotStructuralPhaseV1, *,path:str|Path,
):
    """No partial E1..E5 checkpoint may seed the next E stage."""
    if phase.trainer.global_step!=phase.plan.planned_optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_E_UNFINISHED_PHASE_HANDOFF")
    return phase.trainer.export_phase_handoff(path,target_kind="STRUCTURAL")


def execute_real_pilot_structural_e_phase_v1(
    phase:PreparedPilotStructuralPhaseV1,*,
    examples:Iterable[RealPilotTrainingEnvelopeV1],
    source_index:Mapping[str,RealPilotAssetV1],
    cohort:RealPilotCohortReceiptV1,
    resume_sampler_cursor:int|None=None,
)->list[dict[str,object]]:
    """Runs actual PyTorch optimizer steps, but ONLY on user's TRAIN split."""
    current=phase.trainer.global_step
    if (
        phase.source_cohort_sha256!=cohort.cohort_sha256
        or not 0<=current<phase.plan.planned_optimizer_steps
        or (current>0 and (
            phase.verified_resume_step!=current
            or type(resume_sampler_cursor) is not int
            or resume_sampler_cursor!=phase.verified_resume_cursor
        ))
        or (current==0 and resume_sampler_cursor not in (None,0))
    ):
        raise ValueError("RSS_VNEXT_PILOT_E_STAGE_RESUME_EXPECTS_CHECKPOINT")
    outcome=run_source_qualified_pilot_updates_v1(
        trainer=phase.trainer,staged_examples=examples,
        source_index=source_index,cohort=cohort,
        max_steps=phase.plan.planned_optimizer_steps-current,
    )
    if phase.trainer.global_step!=phase.plan.planned_optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_E_STAGE_OPTIMIZER_STEP_DRIFT")
    return outcome


__all__=[
    "TRAINING_PROFILE","STAGE_KINDS",
    "activate_real_pilot_e_training_mode_v1",
    "PilotStructuralEPhasePlanV1",
    "PreparedPilotStructuralPhaseV1",
    "build_real_pilot_structural_e_phase_v1",
    "execute_real_pilot_structural_e_phase_v1",
    "save_real_pilot_structural_e_resume_v1",
    "load_real_pilot_structural_e_resume_v1",
    "export_real_pilot_structural_e_handoff_v1",
]
