from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import torch
from torch import nn

from models.rss_vnext.stdgen_instantnerf_pretrain_court_v1 import (
    StdGENInstantNeRFArchitectureV1,
    inspect_stdgen_instantnerf_checkpoint_compatibility_v1,
    load_verified_stdgen_instantnerf_weights_v1,
)
from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,RSSvNextUnifiedCharacterModel,
)


class _NativeTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=nn.Module()
        self.transformer.pos_embed=nn.Parameter(torch.randn(1,6,4))
        self.transformer.deconv=nn.ConvTranspose2d(4,2,2,stride=2)


def _fixture(tmp_path):
    model=_NativeTiny()
    file=tmp_path/"tiny.ckpt"
    torch.save({
        "state_dict":{"lrm_generator."+k:v.clone()
                      for k,v in model.state_dict().items()},
        "epoch":1,
    },file)
    digest=sha256(file.read_bytes()).hexdigest()
    return model,file,digest


def test_checkpoint_model_exact_matching_accepts_only_non_prod_fixture(tmp_path):
    native,file,digest=_fixture(tmp_path)
    receipt=inspect_stdgen_instantnerf_checkpoint_compatibility_v1(
        file,native,expected_sha256=digest,unit_test_override=True,
    )
    assert receipt.exact_key_and_shape_match
    assert receipt.test_fixture_only
    assert not receipt.pretrained_model_loaded
    assert not receipt.source_code_imported
    assert not receipt.approved_as_source_observation
    with pytest.raises(ValueError,match="REAL_CHECKPOINT_ADMISSION_REQUIRED"):
        load_verified_stdgen_instantnerf_weights_v1(
            file,native,compatibility=receipt,
        )


def test_unrelated_iris_model_never_gets_partial_instantnerf_weights(tmp_path):
    native,file,digest=_fixture(tmp_path)
    cfg=RSSvNextConfig.tiny_for_test()
    iris=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    with pytest.raises(ValueError,match="NOT_LOAD_COMPATIBLE_WITH_TARGET_MODEL"):
        inspect_stdgen_instantnerf_checkpoint_compatibility_v1(
            file,iris,expected_sha256=digest,unit_test_override=True,
        )


def test_checkpoint_digest_tamper_and_published_architecture_fail_closed(tmp_path):
    native,file,digest=_fixture(tmp_path)
    with pytest.raises(ValueError,match="PINNED_SHA256_MISMATCH"):
        inspect_stdgen_instantnerf_checkpoint_compatibility_v1(file,native)
    with pytest.raises(ValueError,match="TEST_HASH_CONTRACT"):
        inspect_stdgen_instantnerf_checkpoint_compatibility_v1(
            file,native,expected_sha256="1"*64,unit_test_override=True,
        )
    from dataclasses import replace
    with pytest.raises(ValueError,match="PUBLISHED_ARCHITECTURE_DRIFT"):
        replace(StdGENInstantNeRFArchitectureV1(),transformer_width=768).check_published()
