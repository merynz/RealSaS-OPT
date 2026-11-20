from __future__ import annotations

"""Live C(S-LRM)->D(embedded MIRA codec) phase bridge for pilot training.

Preserves the SAME unified character model learned by S-LRM. Imports a
byte-sealed weights-only handoff, freezes every upstream IRIS/TESSA/AXIS/Graphics
tensor, trains only the source-qualified deployed MIRA encoder/decoder and
starts a fresh optimizer. Source 3D truth is loss-only, never model input.

No asset selection, no synthetic rig labels and no work at import time.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping
import math

import torch

from .product_real_pilot_resume_v1 import (
    PilotDEPhaseResumeIdentityV1,
    pilot_phase_plan_sha256_v1, read_pilot_de_resume_v1,
    write_pilot_de_resume_v1,
)
from .training_engine import architecture_digest
from .unified_character_model import RSSvNextUnifiedCharacterModel
from .training_phase_handoff_v1 import (
    import_rss_vnext_training_phase_handoff_v1,
)
from .mira_embedded_codec_source_training_v1 import (
    RSSvNextEmbeddedMIRASourceCodecTrainerV1,
)
from .product_real_pilot_cohort_v1 import (
    RealPilotAssetV1,RealPilotCohortReceiptV1,
)
from .product_real_pilot_execution_v1 import (
    PilotLedgerV1,RealPilotTrainingEnvelopeV1,
    run_source_qualified_pilot_updates_v1,_file_digest,
)


@dataclass(frozen=True)
class PilotEmbeddedCodecPlanV1:
    optimizer_steps:int
    learning_rate:float
    pose_consequence_weight:float=0.
    source_area_skin_weight:float=0.
    source_area_skin_simplex_kl_weight:float=0.

    def validate(self)->None:
        values=(
            self.learning_rate,self.pose_consequence_weight,
            self.source_area_skin_weight,self.source_area_skin_simplex_kl_weight,
        )
        if (
            type(self.optimizer_steps) is not int
            or not 1<=self.optimizer_steps<=10_000_000
            or any(type(x) not in (int,float) or not math.isfinite(x)
                   or x<0 for x in values)
            or not 1e-8<=self.learning_rate<=1e-2
        ):
            raise ValueError("RSS_VNEXT_PILOT_CODEC_SCHEDULE_INVALID")


@dataclass(frozen=True)
class PreparedPilotCodecPhaseV1:
    trainer:RSSvNextEmbeddedMIRASourceCodecTrainerV1
    plan:PilotEmbeddedCodecPlanV1
    prior_source_model_sha256:str
    source_cohort_sha256:str
    verified_resume_step:int|None=None
    verified_resume_cursor:int=0


def build_real_pilot_embedded_codec_phase_v1(
    *,model:RSSvNextUnifiedCharacterModel,
    ledger:PilotLedgerV1,cohort:RealPilotCohortReceiptV1,
    prior_handoff_file:str|Path,
    plan:PilotEmbeddedCodecPlanV1,
    trusted_source_skin_registry:Mapping[str,Mapping[str,Any]],
    trusted_source_pose_registry:Mapping[str,Mapping[str,Any]]|None=None,
    trusted_source_area_registry:Mapping[str,Mapping[str,Any]]|None=None,
)->PreparedPilotCodecPhaseV1:
    """One-time D-stage construction only after certified C-SLRM handoff.

    Train/validation/test membership comes entirely from the user's manifest.
    This cannot be called early with untrained A/B or skipped C.
    """
    plan.validate()
    ledger.validate()
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or ledger.next_stage!="D_EMBEDDED_MIRA_CODEC"
        or not ledger.completed
        or ledger.completed[-1].stage!="C_SLRM"
        or not cohort.source_evidence_bytes_verified
        or cohort.cohort_sha256!=ledger.cohort_sha256
        or not trusted_source_skin_registry
    ):
        raise ValueError("RSS_VNEXT_PILOT_CODEC_STAGE_OR_TEACHER_NOT_ADMITTED")
    prev=ledger.completed[-1]
    source=Path(prior_handoff_file).expanduser().resolve()
    expected=Path(prev.output_checkpoint_path).expanduser().resolve()
    if (
        source!=expected or source.is_symlink() or not source.is_file()
        or _file_digest(source)!=prev.output_checkpoint_sha256
    ):
        raise ValueError("RSS_VNEXT_PILOT_CODEC_SOURCE_HANDOFF_DRIFT")
    # Reconstruct EXACT selected deployed decoder scope from the MIRA trainer;
    # the frozen upstream network cannot change while codec FSQ learns.
    mode=model.cfg.skin_decoder_mode
    encoder_prefixes=(
        "mira.codec.vertex_encode.","mira.codec.joint_encode.",
        "mira.codec.latent_head.",
    )
    if mode not in ("DOT_PRODUCT_BASELINE","POINT_QUERY_CROSS_ATTENTION"):
        raise ValueError("RSS_VNEXT_PILOT_CODEC_DECODER_MODE_UNSUPPORTED")
    selected=set()
    for name,p in model.named_parameters():
        allowed=(name.startswith("mira.codec.") if
                 mode=="DOT_PRODUCT_BASELINE" else
                 name.startswith(encoder_prefixes)
                 or name.startswith("mira.spatial_decoder."))
        p.requires_grad_(allowed)
        p.grad=None
        if allowed:selected.add(name)
    if not selected:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_EMPTY_TRAINABLE_OWNER")
    model.eval()
    model.mira.codec.train()
    if mode=="POINT_QUERY_CROSS_ATTENTION":
        model.mira.spatial_decoder.train()
    optimizer=torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=plan.learning_rate,betas=(.9,.999),weight_decay=.01,
    )
    imported=import_rss_vnext_training_phase_handoff_v1(
        source,model=model,optimizer=optimizer,
        target_kind="MIRA_CODEC",expected_source_kind="SLRM",
        expected_source_objective_policy_sha256=prev.objective_policy_sha256,
        expected_source_optimizer_binding_sha256=prev.optimizer_ownership_sha256,
        dataset_manifest_sha256=cohort.cohort_sha256,
        architecture_id=model.architecture_id,
        architecture_digest=architecture_digest(model.cfg),
    )
    if imported.source_global_step!=prev.trained_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_UPSTREAM_STEP_DRIFT")
    trainer=RSSvNextEmbeddedMIRASourceCodecTrainerV1(
        model,optimizer,dataset_manifest_sha256=cohort.cohort_sha256,
        trusted_source_skin_registry=trusted_source_skin_registry,
        trusted_source_pose_registry=trusted_source_pose_registry,
        trusted_source_area_registry=trusted_source_area_registry,
        pose_consequence_weight=plan.pose_consequence_weight,
        source_area_skin_weight=plan.source_area_skin_weight,
        source_area_skin_simplex_kl_weight=
            plan.source_area_skin_simplex_kl_weight,
    )
    if trainer.step!=0 or optimizer.state:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_NOT_FRESH_AFTER_HANDOFF")
    return PreparedPilotCodecPhaseV1(
        trainer=trainer,plan=plan,
        prior_source_model_sha256=imported.model_state_sha256,
        source_cohort_sha256=cohort.cohort_sha256,
    )


def _codec_pilot_resume_identity_v1(
    phase:PreparedPilotCodecPhaseV1,
)->PilotDEPhaseResumeIdentityV1:
    return PilotDEPhaseResumeIdentityV1(
        stage="D_EMBEDDED_MIRA_CODEC",
        cohort_sha256=phase.source_cohort_sha256,
        plan_sha256=pilot_phase_plan_sha256_v1(phase.plan),
        incoming_model_state_sha256=phase.prior_source_model_sha256,
        optimizer_binding_sha256=phase.trainer.optimizer_ownership.binding_sha256,
        objective_policy_sha256=phase.trainer.phase_objective_policy_sha256,
    )


def save_real_pilot_embedded_codec_resume_v1(
    phase:PreparedPilotCodecPhaseV1,
    *,checkpoint_path:str|Path,sampler_cursor:int,
)->Path:
    """Persist real codec optimizer/RNG, plus immutable pilot schedule receipt."""
    current=phase.trainer.step
    if not 0<=current<=phase.plan.optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_RESUME_PROGRESS_INVALID")
    identity=_codec_pilot_resume_identity_v1(phase)
    identity.validate()
    phase.trainer.save_checkpoint(checkpoint_path)
    return write_pilot_de_resume_v1(
        checkpoint_path,identity=identity,step=current,
        sampler_cursor=sampler_cursor,
    )


def load_real_pilot_embedded_codec_resume_v1(
    phase:PreparedPilotCodecPhaseV1,
    *,checkpoint_path:str|Path,
)->PreparedPilotCodecPhaseV1:
    """Verify sidecar & upstream C lineage BEFORE touching D's trainable state."""
    if phase.trainer.step!=0 or phase.trainer.optimizer.state:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_RESUME_DESTINATION_NOT_FRESH")
    target,cursor=read_pilot_de_resume_v1(
        checkpoint_path,identity=_codec_pilot_resume_identity_v1(phase),
        max_steps=phase.plan.optimizer_steps,
    )
    phase.trainer.load_checkpoint(checkpoint_path)
    if phase.trainer.step!=target:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_RESUME_STEP_DRIFT")
    return replace(
        phase,verified_resume_step=target,verified_resume_cursor=cursor,
    )


def export_real_pilot_embedded_codec_handoff_v1(
    phase:PreparedPilotCodecPhaseV1, *,path:str|Path,
):
    """D -> E cannot happen until all planned D optimizer updates finished."""
    if phase.trainer.step!=phase.plan.optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_UNFINISHED_D_TO_E")
    return phase.trainer.export_phase_handoff(path)


def execute_real_pilot_embedded_codec_phase_v1(
    phase:PreparedPilotCodecPhaseV1,*,
    examples:Iterable[RealPilotTrainingEnvelopeV1],
    source_index:Mapping[str,RealPilotAssetV1],
    cohort:RealPilotCohortReceiptV1,
    resume_sampler_cursor:int|None=None,
)->list[dict[str,object]]:
    current=phase.trainer.step
    if (
        phase.source_cohort_sha256!=cohort.cohort_sha256
        or not 0<=current<phase.plan.optimizer_steps
        or (current>0 and (
            phase.verified_resume_step!=current
            or type(resume_sampler_cursor) is not int
            or resume_sampler_cursor!=phase.verified_resume_cursor
        ))
        or (current==0 and resume_sampler_cursor not in (None,0))
    ):
        raise ValueError("RSS_VNEXT_PILOT_CODEC_RESUME_REQUIRES_CHECKPOINT")
    results=run_source_qualified_pilot_updates_v1(
        trainer=phase.trainer,staged_examples=examples,
        source_index=source_index,cohort=cohort,
        max_steps=phase.plan.optimizer_steps-current,
    )
    if phase.trainer.step!=phase.plan.optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_CODEC_OPTIMIZER_STEP_DRIFT")
    return results


__all__=[
    "PilotEmbeddedCodecPlanV1","PreparedPilotCodecPhaseV1",
    "build_real_pilot_embedded_codec_phase_v1",
    "execute_real_pilot_embedded_codec_phase_v1",
    "save_real_pilot_embedded_codec_resume_v1",
    "load_real_pilot_embedded_codec_resume_v1",
    "export_real_pilot_embedded_codec_handoff_v1",
]
