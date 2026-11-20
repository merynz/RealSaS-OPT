from __future__ import annotations

"""Fail-closed per-batch source identity gates; synthetic policy tests only."""

from types import SimpleNamespace
import pytest

from models.rss_vnext.realsas_sdxl_mvadapter_stageab_trainer_v2 import (
    SourceSDXLStageABOptimizerV2,
)
from models.rss_vnext.product_real_pilot_slrm_launcher_v1 import (
    PilotSLRMPhasePlanV1, _ProvenanceBoundSLRMTrainerV1,
)
from models.rss_vnext.mira_embedded_codec_source_training_v1 import (
    RSSvNextEmbeddedMIRASourceCodecTrainerV1,
)
from models.rss_vnext.training_engine import RSSvNextStructuralTrainer


TRAIN_SHA = "a" * 64
VAL_SHA = "b" * 64
COHORT_SHA = "f" * 64


def test_stage_ab_receipt_asset_identity_is_required_before_training():
    for stage in ("A_CANONICAL_SDXL","B_PAIRED_RGB_NORMAL_MVADAPTER"):
        trainer=object.__new__(SourceSDXLStageABOptimizerV2)
        trainer.stage=stage
        trainer.dataset_manifest_sha256=COHORT_SHA
        trainer.trusted_source_registry={
            "registered":{
                "source_asset_sha256":TRAIN_SHA,
                "dataset_manifest_sha256":COHORT_SHA,
                "stage":stage,
            },
        }
        source=SimpleNamespace(source_render_qualification_key="registered")
        trainer.assert_pilot_batch_source_asset_v1(source,TRAIN_SHA)
        with pytest.raises(RuntimeError,match="BATCH_ASSET_IDENTITY_DRIFT"):
            trainer.assert_pilot_batch_source_asset_v1(source,VAL_SHA)
        with pytest.raises(RuntimeError,match="BATCH_ASSET_IDENTITY_DRIFT"):
            trainer.assert_pilot_batch_source_asset_v1(
                SimpleNamespace(source_render_qualification_key="unknown"),
                TRAIN_SHA,
            )


def test_stage_c_source_render_receipt_must_bind_train_asset():
    fake=SimpleNamespace(
        trusted_source_3d_render_registry={
            "accepted":{
                "source_asset_sha256":TRAIN_SHA,
                "dataset_manifest_sha256":COHORT_SHA,
            },
        },
        global_step=0,
    )
    plan=PilotSLRMPhasePlanV1(
        nerf_single_layer_steps=1,nerf_outside_in_steps=1,
        flexicubes_multilayer_steps=1,learning_rate=1e-4,
    )
    trainer=_ProvenanceBoundSLRMTrainerV1(
        fake,plan=plan,a_checkpoint_sha256="1"*64,
        b_checkpoint_sha256="2"*64,
        pose_program_sha256="3"*64,cohort_sha256=COHORT_SHA,
    )
    batch=SimpleNamespace(admitted_3d_render_key="accepted")
    trainer.assert_pilot_batch_source_asset_v1(batch,TRAIN_SHA)
    with pytest.raises(RuntimeError,match="BATCH_ASSET_IDENTITY_DRIFT"):
        trainer.assert_pilot_batch_source_asset_v1(batch,VAL_SHA)


def test_stage_d_source_skin_receipt_must_bind_train_asset():
    trainer=object.__new__(RSSvNextEmbeddedMIRASourceCodecTrainerV1)
    trainer.dataset_manifest_sha256=COHORT_SHA
    trainer.trusted_registry={
        "accepted":{
            "source_asset_sha256":TRAIN_SHA,
            "dataset_manifest_sha256":COHORT_SHA,
        },
    }
    batch=SimpleNamespace(admission_key="accepted")
    trainer.assert_pilot_batch_source_asset_v1(batch,TRAIN_SHA)
    with pytest.raises(RuntimeError,match="BATCH_ASSET_IDENTITY_DRIFT"):
        trainer.assert_pilot_batch_source_asset_v1(batch,VAL_SHA)


def test_stage_e_matched_mstar_teacher_source_is_not_a_val_disguise():
    trainer=object.__new__(RSSvNextStructuralTrainer)
    trainer.training_profile="PRODUCT_SEEDED_IRIS_TESSA_V1"
    batch=SimpleNamespace(
        seeded_tessa_source_teacher=SimpleNamespace(
            source_asset_sha256=TRAIN_SHA,
        ),
    )
    trainer.assert_pilot_batch_source_asset_v1(batch,TRAIN_SHA)
    with pytest.raises(RuntimeError,match="BATCH_ASSET_IDENTITY_DRIFT"):
        trainer.assert_pilot_batch_source_asset_v1(batch,VAL_SHA)
