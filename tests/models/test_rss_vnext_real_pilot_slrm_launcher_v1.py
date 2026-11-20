from __future__ import annotations

"""Only synthetic policy fixtures; never counted as real-source GPU training."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.rss_vnext.product_real_pilot_slrm_launcher_v1 import (
    PilotSLRMPhasePlanV1, _ProvenanceBoundSLRMTrainerV1,
    build_real_pilot_slrm_phase_v1,
)
from models.rss_vnext.product_real_pilot_execution_v1 import (
    PilotLedgerV1, SCHEMA as LEDGER_SCHEMA,
)
from models.rss_vnext.product_real_pilot_cohort_v1 import (
    RealPilotCohortReceiptV1, SCHEMA as COHORT_SCHEMA,
)
from models.rss_vnext.realsas_joint_slrm_render_v1 import (
    STAGE_1, STAGE_2, STAGE_3,
)
from models.rss_vnext.realsas_slrm_training_driver_v1 import (
    RealSaSSLRMBatchV1, RealSaSSLRMTrainingLossWeightsV1,
)


def plan():
    return PilotSLRMPhasePlanV1(
        nerf_single_layer_steps=2, nerf_outside_in_steps=2,
        flexicubes_multilayer_steps=1, learning_rate=1e-4,
    )


def test_slrm_c_three_subphases_are_ordered_and_nonempty():
    p = plan()
    p.validate()
    assert p.total_steps == 5
    assert tuple(p.required_stage(i) for i in range(5)) == (
        STAGE_1, STAGE_1, STAGE_2, STAGE_2, STAGE_3,
    )
    with pytest.raises(ValueError, match="CURSOR_OUT_OF_RANGE"):
        p.required_stage(5)
    for invalid in (
        replace(p, nerf_single_layer_steps=0),
        replace(p, flexicubes_multilayer_steps=0),
        replace(p, learning_rate=float("nan")),
        replace(p, clip_gradient_norm=0),
    ):
        with pytest.raises(ValueError, match="PHASE_PLAN_INVALID"):
            invalid.validate()


def test_c_must_have_real_completed_b_before_model_or_optimizer_created():
    cohort = RealPilotCohortReceiptV1(
        schema=COHORT_SCHEMA, cohort_sha256="a" * 64,
        total_assets=300, train_assets=280, val_assets=10,
        unseen_test_assets=10, lineage_groups=300,
        teacher_family_coverage_train={},
        source_evidence_bytes_verified=True,
    )
    ledger = PilotLedgerV1(
        schema=LEDGER_SCHEMA, cohort_sha256=cohort.cohort_sha256,
        completed=(),
    )
    with pytest.raises(ValueError, match="B_NOT_ADMITTED"):
        build_real_pilot_slrm_phase_v1(
            model=None, ledger=ledger, cohort=cohort,
            b_task_weights_path="/tmp/nonexistent-b.safetensors",
            plan=plan(),
            trusted_source_3d_render_registry={"not": "a source"},
            weights=RealSaSSLRMTrainingLossWeightsV1(),
        )


class _FakeSLRM:
    def __init__(self):
        self.global_step = 0
        self.model = nn.Module()
        self.model.iris = nn.Linear(1, 1)
        self.called = 0

    def train_step(self, batch):
        self.called += 1
        self.global_step += 1
        self.model.iris.weight.grad = torch.ones_like(self.model.iris.weight)
        return {"total": torch.ones((), requires_grad=True)}


def _batch(stage, b_hash):
    # The actual S-LRM trainer, unlike this policy double, independently
    # checks the tensor bytes and source 3D render admission registry.
    return RealSaSSLRMBatchV1(
        conditioning_rgba=torch.empty(1, 6, 4, 8, 8),
        conditioning_cameras=torch.empty(1, 6, 13),
        conditioning_authority="SIX_VIEW_PRIOR",
        source_binding_sha256="c" * 64,
        original_observed_view_count=2,
        generating_checkpoint_sha256=b_hash,
        ray_xyz=torch.empty(1, 2, 2, 3),
        ray_deltas=torch.empty(1, 2, 2),
        ray_depths=torch.empty(1, 2, 2),
        ray_directions=torch.empty(1, 2, 3),
        keep_roles=torch.empty(1, 2, 1),
        source_3d_render_target=SimpleNamespace(stage=stage),
        admitted_3d_render_key="fixture-only",
    )


def test_slrm_provenance_and_order_rejected_before_optimizer_step():
    fake = _FakeSLRM()
    bound = _ProvenanceBoundSLRMTrainerV1(
        fake, plan=plan(), b_checkpoint_sha256="b" * 64,
        cohort_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="B_PROVENANCE_OR_STAGE_DRIFT"):
        bound.train_step(_batch(STAGE_1, "f" * 64))
    with pytest.raises(ValueError, match="B_PROVENANCE_OR_STAGE_DRIFT"):
        bound.train_step(_batch(STAGE_3, "b" * 64))
    assert fake.called == 0
    for stage in (STAGE_1, STAGE_1, STAGE_2, STAGE_2, STAGE_3):
        bound.train_step(_batch(stage, "b" * 64))
    assert fake.global_step == 5
    with pytest.raises(RuntimeError, match="UNSCHEDULED_STEP"):
        bound.train_step(_batch(STAGE_3, "b" * 64))


def test_slrm_shared_h_reachability_is_not_optional():
    fake = _FakeSLRM()
    original = fake.train_step
    def disconnected(batch):
        result = original(batch)
        fake.model.iris.weight.grad = None
        return result
    fake.train_step = disconnected
    bound = _ProvenanceBoundSLRMTrainerV1(
        fake, plan=plan(), b_checkpoint_sha256="b" * 64,
        cohort_sha256="a" * 64,
    )
    with pytest.raises(RuntimeError, match="SHARED_H_GRADIENT_MISSING"):
        bound.train_step(_batch(STAGE_1, "b" * 64))
