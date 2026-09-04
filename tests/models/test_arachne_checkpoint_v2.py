from __future__ import annotations

import pytest
import torch

from models.arachne.v2.arachne_candidate_v2 import ArachneCandidateConfigV2, ArachneCandidateV2
from models.arachne.v2.arachne_checkpoint_v2 import load_arachne_checkpoint_v2, save_arachne_checkpoint_v2
from models.skin_field_codec.v1.skin_field_codec_v1 import SkinFieldCodecV1
from models.skin_field_codec.v1.train_codec_r6_a0_v1 import CodecA0QualificationTokenV1


A = "a" * 64
B = "b" * 64
C = "c" * 64


def _model() -> ArachneCandidateV2:
    codec = SkinFieldCodecV1()
    model = ArachneCandidateV2(codec, ArachneCandidateConfigV2(model_dim=32, attention_heads=4, feedforward_dim=64, surface_encoder_layers=1), freeze_codec=True)
    return model


def _token(model: ArachneCandidateV2) -> CodecA0QualificationTokenV1:
    return CodecA0QualificationTokenV1(
        status="PASS",
        codec_config_hash=model.codec.config.config_hash,
        source_gate="UNIT",
        optimizer_steps=3,
        criteria_hash=A,
        metrics_hash=B,
    )


def test_arachne_checkpoint_binds_rung_teacher_access_codec_and_claim(tmp_path) -> None:
    model = _model(); token = _token(model); path = tmp_path / "arachne.pt"
    save_arachne_checkpoint_v2(
        path,
        model,
        architecture_base_commit=A,
        training_data_manifest_sha256=B,
        optimizer_steps=7,
        rung="ORACLE_S__CURRENT_QUALIFIED_G",
        teacher_access_level="L3_CONDITIONING_INPUT",
        codec_checkpoint_sha256=C,
        codec_a0_token=token,
        upstream_surface_checkpoint_sha256=None,
        upstream_skeleton_checkpoint_sha256=None,
        authorized_claims=("DOWNSTREAM_CONSUMER_CEILING",),
        forbidden_claims=("SHIPPING_OBSERVATION_ONLY_INFERENCE",),
        shipping_authority=False,
    )
    authority = load_arachne_checkpoint_v2(
        path,
        model,
        expected_architecture_base_commit=A,
        expected_training_data_manifest_sha256=B,
        expected_codec_checkpoint_sha256=C,
        expected_rung="ORACLE_S__CURRENT_QUALIFIED_G",
        expected_teacher_access_level="L3_CONDITIONING_INPUT",
        requested_claim="DOWNSTREAM_CONSUMER_CEILING",
    )
    assert authority["shipping_authority"] is False
    with pytest.raises(ValueError, match="claim not authorized"):
        load_arachne_checkpoint_v2(
            path,
            model,
            expected_architecture_base_commit=A,
            expected_training_data_manifest_sha256=B,
            expected_codec_checkpoint_sha256=C,
            expected_rung="ORACLE_S__CURRENT_QUALIFIED_G",
            expected_teacher_access_level="L3_CONDITIONING_INPUT",
            requested_claim="SHIPPING_OBSERVATION_ONLY_INFERENCE",
        )


def test_arachne_checkpoint_rejects_wrong_upstream_codec(tmp_path) -> None:
    model = _model(); token = _token(model); path = tmp_path / "arachne.pt"
    save_arachne_checkpoint_v2(
        path,
        model,
        architecture_base_commit=A,
        training_data_manifest_sha256=B,
        optimizer_steps=7,
        rung="CURRENT_S__CURRENT_QUALIFIED_G",
        teacher_access_level="L1_LOSS_ONLY",
        codec_checkpoint_sha256=C,
        codec_a0_token=token,
        upstream_surface_checkpoint_sha256=A,
        upstream_skeleton_checkpoint_sha256=B,
        authorized_claims=("SINGLE_FAMILY_FIT",),
        forbidden_claims=("SHIPPING_OBSERVATION_ONLY_INFERENCE",),
        shipping_authority=False,
    )
    with pytest.raises(ValueError, match="codec_checkpoint_sha256"):
        load_arachne_checkpoint_v2(
            path,
            model,
            expected_architecture_base_commit=A,
            expected_training_data_manifest_sha256=B,
            expected_codec_checkpoint_sha256=A,
            expected_rung="CURRENT_S__CURRENT_QUALIFIED_G",
            expected_teacher_access_level="L1_LOSS_ONLY",
            requested_claim="SINGLE_FAMILY_FIT",
        )
