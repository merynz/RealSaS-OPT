from __future__ import annotations

"""Policy-only launch courts; no actual pilot source is fabricated here."""

from dataclasses import replace

import pytest

from models.rss_vnext.product_real_pilot_codec_launcher_v1 import (
    PilotEmbeddedCodecPlanV1,
    build_real_pilot_embedded_codec_phase_v1,
)

from models.rss_vnext.product_real_pilot_structural_launcher_v1 import (
    PilotStructuralEPhasePlanV1,
    build_real_pilot_structural_e_phase_v1,
)
from models.rss_vnext.product_real_pilot_execution_v1 import (
    PilotLedgerV1,SCHEMA as LEDGER_SCHEMA,
)
from models.rss_vnext.product_real_pilot_cohort_v1 import (
    RealPilotCohortReceiptV1,SCHEMA as COHORT_SCHEMA,
)


def test_phase_plan_keeps_optimizer_choice_open_without_guessing_winner():
    for mode in ("WEIGHTED_SUM","BOUNDED_SHARED_PCGRAD",
                 "BOUNDED_SHARED_CAGRAD_K"):
        plan=PilotStructuralEPhasePlanV1(
            stage="FULL_SHARED_H_FINETUNE",
            planned_optimizer_steps=12,
            learning_rate=5e-5,
            gradient_mode=mode,
        )
        plan.validate()
    with pytest.raises(ValueError,match="POLICY_INVALID"):
        replace(plan,gradient_mode="UNVERIFIED_OPTIMIZER").validate()
    with pytest.raises(ValueError,match="POLICY_INVALID"):
        replace(plan,planned_optimizer_steps=0).validate()


def test_launch_cannot_skip_untrained_abcd_or_manufacture_source_splits():
    cohort=RealPilotCohortReceiptV1(
        schema=COHORT_SCHEMA,cohort_sha256="a"*64,
        total_assets=300,train_assets=280,val_assets=10,
        unseen_test_assets=10,lineage_groups=300,
        teacher_family_coverage_train={},
        source_evidence_bytes_verified=True,
    )
    ledger=PilotLedgerV1(
        schema=LEDGER_SCHEMA,cohort_sha256=cohort.cohort_sha256,
        completed=(),
    )
    plan=PilotStructuralEPhasePlanV1(
        stage="MSTAR_GEOMETRY_BOOTSTRAP",
        planned_optimizer_steps=10,learning_rate=1e-4,
    )
    # Even a sealed synthetic cohort cannot jump past untrained A,B,C,D.
    with pytest.raises(ValueError,match="LAUNCH_NOT_ADMITTED"):
        build_real_pilot_structural_e_phase_v1(
            model=None,cohort=cohort,ledger=ledger,plan=plan,
            prior_handoff_file="/tmp/nonexistent-structural-source.pt",
            source_trainer_kwargs={},
        )

def test_slrm_codec_phase_requires_proven_training_origin_before_any_optimizer():
    cohort=RealPilotCohortReceiptV1(
        schema=COHORT_SCHEMA,cohort_sha256="a"*64,
        total_assets=300,train_assets=285,val_assets=10,
        unseen_test_assets=5,lineage_groups=300,
        teacher_family_coverage_train={},
        source_evidence_bytes_verified=True,
    )
    ledger=PilotLedgerV1(
        schema=LEDGER_SCHEMA,cohort_sha256=cohort.cohort_sha256,
        completed=(),
    )
    plan=PilotEmbeddedCodecPlanV1(
        optimizer_steps=100,learning_rate=5e-5,
    )
    plan.validate()
    with pytest.raises(ValueError,match="CODEC_STAGE_OR_TEACHER_NOT_ADMITTED"):
        build_real_pilot_embedded_codec_phase_v1(
            model=None,cohort=cohort,ledger=ledger,
            prior_handoff_file="/tmp/no-real-slrm-checkpoint.pt",
            plan=plan,trusted_source_skin_registry={"fictional":"not-used"},
        )
    with pytest.raises(ValueError,match="CODEC_SCHEDULE_INVALID"):
        replace(plan,source_area_skin_weight=-1.).validate()
