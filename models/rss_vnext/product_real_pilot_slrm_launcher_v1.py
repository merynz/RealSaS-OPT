from __future__ import annotations

"""Source-qualified C (S-LRM) pilot launcher between trained B and embedded D.

This is an actual optimizer entry point, not an output/quality receipt.
The user supplies the already approved cohort/splits, the B weights artifact,
and independently source-rendered teacher registries. Neither generated
multiview outputs nor B's predicted normals become ground truth.

The three S-LRM subphases are mandatory and ordered; the *same* model and
optimizer carry learning through C1 -> C2 -> C3. The resulting C trainer can
export its full unified-model weights to the existing D handoff importer.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import math

import torch
from torch import Tensor

from .product_real_pilot_cohort_v1 import RealPilotAssetV1, RealPilotCohortReceiptV1
from .product_real_pilot_execution_v1 import (
    PilotLedgerV1, RealPilotTrainingEnvelopeV1,
    _file_digest, run_source_qualified_pilot_updates_v1,
)
from .product_seeded_phase_ownership_v1 import (
    PREFIXES_GEOMETRY, PREFIXES_SEMANTIC_GRAPHICS,
)
from .realsas_joint_slrm_render_v1 import STAGE_1, STAGE_2, STAGE_3
from .realsas_sdxl_mvadapter_construction_v2 import (
    two_stage_sdxl_mv_binding_v2,
)
from .realsas_slrm_training_driver_v1 import (
    REALSAS_SIX_VIEW_TRAINING_PROFILE,
    RealSaSSLRMBatchV1, RealSaSSLRMTrainerV1,
    RealSaSSLRMTrainingLossWeightsV1,
)
from .training_optimizer_ownership_v1 import (
    qualify_rss_vnext_optimizer_ownership_v1,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel


_C_TRAINABLE = tuple(dict.fromkeys(
    PREFIXES_GEOMETRY + PREFIXES_SEMANTIC_GRAPHICS
))
_C_PROTECTED = ("vision.", "axis.", "mira.", "mesh_face_to_control.")
_C_STAGES = (STAGE_1, STAGE_2, STAGE_3)


@dataclass(frozen=True)
class PilotSLRMPhasePlanV1:
    nerf_single_layer_steps: int
    nerf_outside_in_steps: int
    flexicubes_multilayer_steps: int
    learning_rate: float
    clip_gradient_norm: float = 1.0

    def validate(self) -> None:
        counts = self.stage_steps
        if (
            any(type(n) is not int or not 1 <= n <= 10_000_000 for n in counts)
            or type(self.learning_rate) not in (float, int)
            or not math.isfinite(self.learning_rate)
            or not 1e-8 <= self.learning_rate <= 1e-2
            or type(self.clip_gradient_norm) not in (float, int)
            or not math.isfinite(self.clip_gradient_norm)
            or not 0 < self.clip_gradient_norm <= 1_000
        ):
            raise ValueError("RSS_VNEXT_PILOT_SLRM_PHASE_PLAN_INVALID")

    @property
    def digest(self) -> str:
        self.validate()
        return sha256(json.dumps(asdict(self), sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()

    @property
    def stage_steps(self) -> tuple[int, int, int]:
        return (self.nerf_single_layer_steps, self.nerf_outside_in_steps,
                self.flexicubes_multilayer_steps)

    @property
    def total_steps(self) -> int:
        return sum(self.stage_steps)

    def required_stage(self, completed: int) -> str:
        if type(completed) is not int or not 0 <= completed < self.total_steps:
            raise ValueError("RSS_VNEXT_PILOT_SLRM_CURSOR_OUT_OF_RANGE")
        if completed < self.stage_steps[0]:
            return STAGE_1
        if completed < self.stage_steps[0] + self.stage_steps[1]:
            return STAGE_2
        return STAGE_3


class _ProvenanceBoundSLRMTrainerV1:
    """Bind B's *actual exported weights bytes* to every generated C input."""

    def __init__(
        self, trainer: RealSaSSLRMTrainerV1, *,
        plan: PilotSLRMPhasePlanV1,
        a_checkpoint_sha256: str, b_checkpoint_sha256: str,
        pose_program_sha256: str, cohort_sha256: str,
    ) -> None:
        self.trainer = trainer
        self.plan = plan
        self.a_checkpoint_sha256 = a_checkpoint_sha256
        self.b_checkpoint_sha256 = b_checkpoint_sha256
        self.pose_program_sha256 = pose_program_sha256
        self.dataset_manifest_sha256 = cohort_sha256
        self._resume_attested = False

    @property
    def global_step(self) -> int:
        return self.trainer.global_step

    def train_step(self, batch: RealSaSSLRMBatchV1) -> dict[str, Tensor]:
        before = self.trainer.global_step
        if before >= self.plan.total_steps:
            raise RuntimeError("RSS_VNEXT_PILOT_SLRM_UNSCHEDULED_STEP")
        if (
            type(batch) is not RealSaSSLRMBatchV1
            or batch.original_observed_view_count != 2
            or batch.generating_checkpoint_sha256 != two_stage_sdxl_mv_binding_v2(
                stage_a_checkpoint_sha256=self.a_checkpoint_sha256,
                stage_b_checkpoint_sha256=self.b_checkpoint_sha256,
                source_binding_sha256=batch.source_binding_sha256,
                pose_program_sha256=self.pose_program_sha256,
            )
            or batch.source_3d_render_target.stage
                != self.plan.required_stage(before)
        ):
            raise ValueError("RSS_VNEXT_PILOT_SLRM_B_PROVENANCE_OR_STAGE_DRIFT")
        result = self.trainer.train_step(batch)
        if self.trainer.global_step != before + 1:
            raise RuntimeError("RSS_VNEXT_PILOT_SLRM_OPTIMIZER_STEP_NOT_RECORDED")
        live = [
            p.grad for name, p in self.trainer.model.named_parameters()
            if name.startswith("iris.") and p.requires_grad and p.grad is not None
        ]
        if not live or any(
            g.is_sparse or not bool(torch.isfinite(g).all()) for g in live
        ):
            raise RuntimeError("RSS_VNEXT_PILOT_SLRM_SHARED_H_GRADIENT_MISSING")
        return result


    def export_phase_handoff(self, path: str | Path):
        """Only the finished three-stage C model may seed D's fresh optimizer."""
        if self.global_step != self.plan.total_steps:
            raise RuntimeError("RSS_VNEXT_PILOT_SLRM_INCOMPLETE_C_TO_D_HANDOFF")
        return self.trainer.export_phase_handoff(path, target_kind="MIRA_CODEC")

    def save_resume_checkpoint(self, path: str | Path, *, sampler_cursor: int) -> None:
        """Seal the C subphase schedule and trained-B identity beside model/AdamW."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(file, sampler_cursor=sampler_cursor)
        policy = {
            "schema": "RealSaS.RSSvNext.PilotSLRMResume.v1",
            "checkpoint_sha256": _file_digest(file),
            "cohort_sha256": self.dataset_manifest_sha256,
            "trained_a_task_weights_sha256": self.a_checkpoint_sha256,
            "trained_b_task_weights_sha256": self.b_checkpoint_sha256,
            "canonical_pose_program_sha256": self.pose_program_sha256,
            "c_phase_plan_sha256": self.plan.digest,
            "step": self.global_step,
            "sampler_cursor": sampler_cursor,
        }
        sidecar = Path(str(file) + ".pilot.json")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=file.parent,
            prefix="rss-slrm-policy-", suffix=".json", delete=False,
        ) as stream:
            tmp = Path(stream.name)
            json.dump(policy, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(tmp, sidecar)
        finally:
            tmp.unlink(missing_ok=True)

    def load_resume_checkpoint(self, path: str | Path) -> int:
        file = Path(path)
        sidecar = Path(str(file) + ".pilot.json")
        if (
            file.is_symlink() or sidecar.is_symlink()
            or not file.is_file() or not sidecar.is_file()
        ):
            raise ValueError("RSS_VNEXT_PILOT_SLRM_RESUME_FILES_MISSING")
        data = json.loads(sidecar.read_text())
        if (
            type(data) is not dict
            or data.get("schema") != "RealSaS.RSSvNext.PilotSLRMResume.v1"
            or data.get("checkpoint_sha256") != _file_digest(file)
            or data.get("cohort_sha256") != self.dataset_manifest_sha256
            or data.get("trained_a_task_weights_sha256") != self.a_checkpoint_sha256
            or data.get("trained_b_task_weights_sha256") != self.b_checkpoint_sha256
            or data.get("canonical_pose_program_sha256") != self.pose_program_sha256
            or data.get("c_phase_plan_sha256") != self.plan.digest
            or type(data.get("step")) is not int
            or not 0 <= data["step"] <= self.plan.total_steps
            or type(data.get("sampler_cursor")) is not int
            or data["sampler_cursor"] < 0
        ):
            raise ValueError("RSS_VNEXT_PILOT_SLRM_RESUME_PROVENANCE_DRIFT")
        cursor = self.trainer.load_checkpoint(file)
        if cursor != data["sampler_cursor"] or self.global_step != data["step"]:
            raise ValueError("RSS_VNEXT_PILOT_SLRM_RESUME_PROGRESS_DRIFT")
        self._resume_attested = True
        return cursor


@dataclass(frozen=True)
class PreparedPilotSLRMPhaseV1:
    trainer: _ProvenanceBoundSLRMTrainerV1
    plan: PilotSLRMPhasePlanV1
    source_cohort_sha256: str
    trained_a_checkpoint_sha256: str
    trained_b_checkpoint_sha256: str
    pose_program_sha256: str
    optimizer_binding_sha256: str


def build_real_pilot_slrm_phase_v1(
    *,
    model: RSSvNextUnifiedCharacterModel,
    ledger: PilotLedgerV1,
    cohort: RealPilotCohortReceiptV1,
    a_task_weights_path: str | Path,
    b_task_weights_path: str | Path,
    pose_program_sha256: str,
    plan: PilotSLRMPhasePlanV1,
    trusted_source_3d_render_registry: Mapping[str, Mapping[str, object]],
    trusted_original_observed_ray_registry:
        Mapping[str, Mapping[str, object]] | None = None,
    weights: RealSaSSLRMTrainingLossWeightsV1,
    lpips_loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> PreparedPilotSLRMPhaseV1:
    plan.validate()
    ledger.validate()
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or not cohort.source_evidence_bytes_verified
        or cohort.cohort_sha256 != ledger.cohort_sha256
        or ledger.next_stage != "C_SLRM"
        or not ledger.completed
        or ledger.completed[-1].stage != "B_PAIRED_RGB_NORMAL_MVADAPTER"
        or len(ledger.completed) != 2
        or ledger.completed[0].stage != "A_CANONICAL_SDXL"
        or not trusted_source_3d_render_registry
        or type(weights) is not RealSaSSLRMTrainingLossWeightsV1
    ):
        raise ValueError("RSS_VNEXT_PILOT_SLRM_B_NOT_ADMITTED")
    a, b = ledger.completed
    # B and A are distinct pretrained/task weights. The V2 generation
    # receipt is a four-part construction pin, not the raw B SHA.
    two_stage_sdxl_mv_binding_v2(
        stage_a_checkpoint_sha256=a.output_checkpoint_sha256,
        stage_b_checkpoint_sha256=b.output_checkpoint_sha256,
        source_binding_sha256="0" * 64,
        pose_program_sha256=pose_program_sha256,
    )
    a_source = Path(a_task_weights_path).expanduser()
    a_expected = Path(a.output_checkpoint_path).expanduser()
    if (
        a_source.is_symlink() or not a_source.is_file()
        or a_source.resolve() != a_expected.resolve()
        or _file_digest(a_source) != a.output_checkpoint_sha256
    ):
        raise ValueError("RSS_VNEXT_PILOT_SLRM_A_CHECKPOINT_BYTES_DRIFT")
    source = Path(b_task_weights_path).expanduser()
    expected = Path(b.output_checkpoint_path).expanduser()
    if (
        source.is_symlink() or not source.is_file()
        or source.resolve() != expected.resolve()
        or _file_digest(source) != b.output_checkpoint_sha256
    ):
        raise ValueError("RSS_VNEXT_PILOT_SLRM_B_CHECKPOINT_BYTES_DRIFT")

    # Real S-LRM C changes the SAME H(x), Geometry and Graphics that D/E
    # will inherit. B's SDXL UNet is a conditioning dependency, not a
    # transplant of unrelated diffusion weights into this student.
    selected = []
    for name, param in model.named_parameters():
        allowed = (
            name.startswith(_C_TRAINABLE)
            and not name.startswith(_C_PROTECTED)
        )
        param.requires_grad_(allowed)
        param.grad = None
        if allowed:
            selected.append(param)
    if not selected or not any(
        n.startswith("iris.") and p.requires_grad
        for n, p in model.named_parameters()
    ):
        raise RuntimeError("RSS_VNEXT_PILOT_SLRM_IRIS_OWNER_MISSING")
    model.train()
    model.vision.eval()
    model.mira.eval()
    model.axis.eval()
    optimizer = torch.optim.AdamW(
        selected, lr=float(plan.learning_rate),
        betas=(0.9, 0.999), weight_decay=0.01,
    )
    binding = qualify_rss_vnext_optimizer_ownership_v1(
        model, optimizer
    ).binding_sha256
    if optimizer.state:
        raise RuntimeError("RSS_VNEXT_PILOT_SLRM_OPTIMIZER_NOT_FRESH")
    trainer = RealSaSSLRMTrainerV1(
        model, optimizer,
        dataset_manifest_sha256=cohort.cohort_sha256,
        trusted_source_3d_render_registry=trusted_source_3d_render_registry,
        trusted_original_observed_ray_registry=
            trusted_original_observed_ray_registry,
        weights=weights,
        lpips_loss_fn=lpips_loss_fn,
        clip_gradient_norm=plan.clip_gradient_norm,
        input_profile=REALSAS_SIX_VIEW_TRAINING_PROFILE,
    )
    return PreparedPilotSLRMPhaseV1(
        trainer=_ProvenanceBoundSLRMTrainerV1(
            trainer, plan=plan,
            a_checkpoint_sha256=a.output_checkpoint_sha256,
            b_checkpoint_sha256=b.output_checkpoint_sha256,
            pose_program_sha256=pose_program_sha256,
            cohort_sha256=cohort.cohort_sha256,
        ),
        plan=plan, source_cohort_sha256=cohort.cohort_sha256,
        trained_a_checkpoint_sha256=a.output_checkpoint_sha256,
        trained_b_checkpoint_sha256=b.output_checkpoint_sha256,
        pose_program_sha256=pose_program_sha256,
        optimizer_binding_sha256=binding,
    )


def execute_real_pilot_slrm_phase_v1(
    phase: PreparedPilotSLRMPhaseV1,
    *, examples: Iterable[RealPilotTrainingEnvelopeV1],
    source_index: Mapping[str, RealPilotAssetV1],
    cohort: RealPilotCohortReceiptV1,
) -> list[dict[str, object]]:
    if (
        phase.source_cohort_sha256 != cohort.cohort_sha256
        or (phase.trainer.global_step != 0
            and not phase.trainer._resume_attested)
        or phase.trainer.global_step >= phase.plan.total_steps
    ):
        raise ValueError("RSS_VNEXT_PILOT_SLRM_RESUME_REQUIRES_CHECKPOINT")
    remaining = phase.plan.total_steps - phase.trainer.global_step
    result = run_source_qualified_pilot_updates_v1(
        trainer=phase.trainer, staged_examples=examples,
        source_index=source_index, cohort=cohort,
        max_steps=remaining,
    )
    if phase.trainer.global_step != phase.plan.total_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_SLRM_PHASE_PROGRESS_DRIFT")
    return result


__all__ = [
    "PilotSLRMPhasePlanV1", "PreparedPilotSLRMPhaseV1",
    "build_real_pilot_slrm_phase_v1", "execute_real_pilot_slrm_phase_v1",
]
