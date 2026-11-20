from __future__ import annotations

"""Real A/B task-training pilot lifecycle over existing pretrained SDXL/MV.

All native SDXL/MV-Adapter forward, backward, source authority, optimizer,
scheduler and trained-weight serialization live in SourceSDXLStageABOptimizerV2.
This launcher adds the missing *cohort-stage order*, immutable planned steps,
source identity, exact restart sampler cursor, and refusal of early release.

Neither source splits nor canonical/multiview teachers are synthesized here.
A is the paired SDXL canonicalizer. B uses its own actual pretrained native
MV-Adapter and receives A's already-trained weights only as a dependency,
not as a transplanted UNet or a teacher image.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math
import os
import tempfile

from .product_real_pilot_cohort_v1 import (
    RealPilotAssetV1, RealPilotCohortReceiptV1,
)
from .product_real_pilot_execution_v1 import (
    PilotLedgerV1, RealPilotTrainingEnvelopeV1,
    _file_digest, run_source_qualified_pilot_updates_v1,
)
from .product_real_pilot_resume_v1 import pilot_phase_plan_sha256_v1
from .realsas_sdxl_mvadapter_stageab_trainer_v2 import (
    SourceSDXLStageABOptimizerV2,
)

SCHEMA = "RealSaS.RSSvNext.PilotABStageResume.v1"
_A = "A_CANONICAL_SDXL"
_B = "B_PAIRED_RGB_NORMAL_MVADAPTER"


def _sha(v: object) -> bool:
    return type(v) is str and len(v) == 64 and all(
        c in "0123456789abcdef" for c in v
    )


def _digest(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf8")).hexdigest()


@dataclass(frozen=True)
class PilotABPhasePlanV1:
    stage: str
    optimizer_steps: int
    learning_rate: float
    warmup_steps: int = 2000
    amp_dtype: str = "BF16"

    def validate(self) -> None:
        if (
            self.stage not in (_A, _B)
            or type(self.optimizer_steps) is not int
            or not 1 <= self.optimizer_steps <= 10_000_000
            or type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.learning_rate)
            or not 1e-8 <= self.learning_rate <= 1e-2
            or type(self.warmup_steps) is not int
            or not 0 <= self.warmup_steps <= 100_000
            or self.amp_dtype not in ("BF16", "FP16")
        ):
            raise ValueError("RSS_VNEXT_PILOT_AB_PHASE_PLAN_INVALID")


@dataclass(frozen=True)
class PreparedPilotABPhaseV1:
    trainer: SourceSDXLStageABOptimizerV2
    plan: PilotABPhasePlanV1
    source_cohort_sha256: str
    previous_stage_task_sha256: str | None
    optimizer_binding_sha256: str
    verified_resume_step: int | None = None
    verified_resume_cursor: int = 0


def _actual_trainer_identity_v1(trainer: SourceSDXLStageABOptimizerV2) -> str:
    return _digest({
        "stage": trainer.stage,
        "dataset_manifest_sha256": trainer.dataset_manifest_sha256,
        "source_registry_sha256": trainer.source_registry_sha256,
        "pretrained_source_sha256": trainer.pretrained_source_sha256,
        "pretrained_mv_adapter_sha256": trainer.pretrained_mv_adapter_sha256,
        "source_vae_sha256": trainer.source_vae_sha256,
        "optimizer_binding": trainer.optimizer_binding,
        "training_noise_schedule_sha256": trainer.training_noise_schedule_sha256,
        "guidance_policy_sha256": trainer.guidance_policy_sha256,
        "base_learning_rate": trainer.base_learning_rate,
        "warmup_steps": trainer.warmup_steps,
        "amp_dtype": trainer.amp_dtype,
    })


def build_real_pilot_ab_phase_v1(
    *, trainer: SourceSDXLStageABOptimizerV2,
    plan: PilotABPhasePlanV1,
    cohort: RealPilotCohortReceiptV1,
    ledger: PilotLedgerV1,
    prior_a_task_weights_path: str | Path | None = None,
) -> PreparedPilotABPhaseV1:
    """Admit a REAL pretrained, registry-bound stage trainer once per phase."""
    plan.validate()
    ledger.validate()
    if (
        type(trainer) is not SourceSDXLStageABOptimizerV2
        or trainer.stage != plan.stage
        or trainer.dataset_manifest_sha256 != cohort.cohort_sha256
        or not cohort.source_evidence_bytes_verified
        or ledger.cohort_sha256 != cohort.cohort_sha256
        or ledger.next_stage != plan.stage
        or trainer.step != 0
        or trainer.optimizer.state
        or trainer.base_learning_rate != float(plan.learning_rate)
        or trainer.warmup_steps != plan.warmup_steps
        or trainer.amp_dtype != plan.amp_dtype
        or not _sha(trainer.optimizer_binding)
        or not _sha(trainer.source_registry_sha256)
    ):
        raise ValueError("RSS_VNEXT_PILOT_AB_PHASE_TRAINER_NOT_ADMITTED")
    previous = None
    if plan.stage == _A:
        if ledger.completed or prior_a_task_weights_path is not None:
            raise ValueError("RSS_VNEXT_PILOT_AB_A_HAS_UNEXPECTED_PARENT")
    else:
        if (
            len(ledger.completed) != 1
            or ledger.completed[0].stage != _A
            or prior_a_task_weights_path is None
        ):
            raise ValueError("RSS_VNEXT_PILOT_AB_TRAINED_A_REQUIRED")
        a = ledger.completed[-1]
        path = Path(prior_a_task_weights_path).expanduser()
        expected = Path(a.output_checkpoint_path).expanduser()
        if (
            path.is_symlink() or not path.is_file()
            or path.resolve() != expected.resolve()
            or _file_digest(path) != a.output_checkpoint_sha256
        ):
            raise ValueError("RSS_VNEXT_PILOT_AB_A_TASK_BYTES_DRIFT")
        previous = a.output_checkpoint_sha256
    return PreparedPilotABPhaseV1(
        trainer=trainer, plan=plan,
        source_cohort_sha256=cohort.cohort_sha256,
        previous_stage_task_sha256=previous,
        optimizer_binding_sha256=trainer.optimizer_binding,
    )


def _resume_identity_v1(phase: PreparedPilotABPhaseV1) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "stage": phase.plan.stage,
        "cohort_sha256": phase.source_cohort_sha256,
        "plan_sha256": pilot_phase_plan_sha256_v1(phase.plan),
        "trainer_identity_sha256": _actual_trainer_identity_v1(phase.trainer),
        "previous_stage_task_sha256": phase.previous_stage_task_sha256,
        "optimizer_binding_sha256": phase.optimizer_binding_sha256,
    }


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".pilot-ab.json")


def save_real_pilot_ab_resume_v1(
    phase: PreparedPilotABPhaseV1,
    *, checkpoint_path: str | Path, sampler_cursor: int,
) -> Path:
    """A/B optimizer+native LoRA/adapter checkpoint plus pilot cursor SHA."""
    if (
        type(sampler_cursor) is not int or sampler_cursor < 0
        or not 1 <= phase.trainer.step <= phase.plan.optimizer_steps
    ):
        raise ValueError("RSS_VNEXT_PILOT_AB_SAVE_PROGRESS_INVALID")
    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.is_symlink() or _sidecar(checkpoint).is_symlink():
        raise ValueError("RSS_VNEXT_PILOT_AB_SAVE_SYMLINK")
    phase.trainer.save_resume_checkpoint(checkpoint)
    identity = _resume_identity_v1(phase)
    payload = {
        **identity,
        "checkpoint_sha256": _file_digest(checkpoint),
        "step": phase.trainer.step,
        "sampler_cursor": sampler_cursor,
    }
    sidecar = _sidecar(checkpoint)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", dir=sidecar.parent,
        prefix="rss-pilot-ab-", suffix=".json", delete=False,
    ) as stream:
        tmp = Path(stream.name)
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(tmp, sidecar)
    finally:
        tmp.unlink(missing_ok=True)
    return sidecar


def load_real_pilot_ab_resume_v1(
    phase: PreparedPilotABPhaseV1,
    *, checkpoint_path: str | Path,
) -> PreparedPilotABPhaseV1:
    """Verify BOTH byte and stage identity before mutating LoRA or AdamW."""
    if phase.trainer.step != 0 or phase.trainer.optimizer.state:
        raise RuntimeError("RSS_VNEXT_PILOT_AB_RESUME_DESTINATION_NOT_FRESH")
    checkpoint = Path(checkpoint_path).expanduser()
    sidecar = _sidecar(checkpoint)
    if (
        checkpoint.is_symlink() or not checkpoint.is_file()
        or sidecar.is_symlink() or not sidecar.is_file()
        or sidecar.stat().st_size > 32 * 1024
    ):
        raise ValueError("RSS_VNEXT_PILOT_AB_RESUME_FILES_INVALID")
    try:
        data = json.loads(sidecar.read_text(encoding="utf8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("RSS_VNEXT_PILOT_AB_RESUME_RECEIPT_UNREADABLE") from exc
    expected = _resume_identity_v1(phase)
    if (
        type(data) is not dict
        or any(data.get(k) != v for k, v in expected.items())
        or not _sha(data.get("checkpoint_sha256"))
        or data["checkpoint_sha256"] != _file_digest(checkpoint)
        or type(data.get("step")) is not int
        or not 1 <= data["step"] <= phase.plan.optimizer_steps
        or type(data.get("sampler_cursor")) is not int
        or data["sampler_cursor"] < 0
    ):
        raise ValueError("RSS_VNEXT_PILOT_AB_RESUME_PROVENANCE_DRIFT")
    # Native optimizer loader separately confirms exact in-memory ancestor,
    # training-noise schedule, CFG policy and all trainable module names.
    phase.trainer.load_resume_checkpoint(checkpoint)
    if phase.trainer.step != data["step"]:
        raise RuntimeError("RSS_VNEXT_PILOT_AB_RESUME_STEP_DRIFT")
    return replace(
        phase, verified_resume_step=phase.trainer.step,
        verified_resume_cursor=data["sampler_cursor"],
    )


def execute_real_pilot_ab_phase_v1(
    phase: PreparedPilotABPhaseV1,
    *, examples: Iterable[RealPilotTrainingEnvelopeV1],
    source_index: Mapping[str, RealPilotAssetV1],
    cohort: RealPilotCohortReceiptV1,
    resume_sampler_cursor: int | None = None,
) -> list[dict[str, object]]:
    """Actual optimizer updates, TRAIN-only, from exact stage cursor."""
    current = phase.trainer.step
    if (
        phase.source_cohort_sha256 != cohort.cohort_sha256
        or not 0 <= current < phase.plan.optimizer_steps
        or (current > 0 and (
            phase.verified_resume_step != current
            or type(resume_sampler_cursor) is not int
            or resume_sampler_cursor != phase.verified_resume_cursor
        ))
        or (current == 0 and resume_sampler_cursor not in (None, 0))
    ):
        raise ValueError("RSS_VNEXT_PILOT_AB_STAGE_RESUME_REQUIRES_CHECKPOINT")
    results = run_source_qualified_pilot_updates_v1(
        trainer=phase.trainer, staged_examples=examples,
        source_index=source_index, cohort=cohort,
        max_steps=phase.plan.optimizer_steps - current,
    )
    if phase.trainer.step != phase.plan.optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_AB_OPTIMIZER_STEP_DRIFT")
    return results


def export_real_pilot_ab_trained_task_weights_v1(
    phase: PreparedPilotABPhaseV1, *, output_path: str | Path,
):
    """Refuse a one-step 'trained' A/B release before the PILOT plan ends."""
    if phase.trainer.step != phase.plan.optimizer_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_AB_UNFINISHED_WEIGHTS_EXPORT")
    return phase.trainer.export_trained_task_weights(output_path)


__all__ = [
    "SCHEMA", "PilotABPhasePlanV1", "PreparedPilotABPhaseV1",
    "build_real_pilot_ab_phase_v1", "save_real_pilot_ab_resume_v1",
    "load_real_pilot_ab_resume_v1", "execute_real_pilot_ab_phase_v1",
    "export_real_pilot_ab_trained_task_weights_v1",
]
