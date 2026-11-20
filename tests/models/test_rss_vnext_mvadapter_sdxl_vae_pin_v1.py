from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from models.rss_vnext.realsas_mvadapter_sdxl_vae_pin_v1 import (
    PinnedMVAdapterSDXLVAEV1,WEIGHT,UPSTREAM_REPO,
)


def digest(data:bytes)->str:
    return sha256(data).hexdigest()


def _fixture(tmp_path):
    path=tmp_path/"vae"
    path.mkdir()
    weights=b"regression-only-not-a-real-fp16-vae"
    cfg=json.dumps({
        "_class_name":"AutoencoderKL",
        "in_channels":3,"out_channels":3,"latent_channels":4,
    }).encode()
    (path/WEIGHT).write_bytes(weights)
    (path/"config.json").write_bytes(cfg)
    return PinnedMVAdapterSDXLVAEV1(
        snapshot_dir=str(path),pinned_revision="a"*40,
        checkpoint_sha256=digest(weights),
        model_config_sha256=digest(cfg),
        independent_license_review_sha256="b"*64,
    )


def test_exact_official_sdxl_fp16_vae_pin_required(tmp_path):
    pin=_fixture(tmp_path)
    proof=pin.verify()
    assert proof["vae_ancestry"]==UPSTREAM_REPO
    assert proof["vae_weight_sha256"]==pin.checkpoint_sha256
    assert proof["license_human_review_not_automatically_proven"]=="true"


def test_vae_weight_or_config_tamper_and_rights_drift_refused(tmp_path):
    pin=_fixture(tmp_path)
    path=tmp_path/"vae"
    (path/WEIGHT).write_bytes(b"tampered")
    with pytest.raises(RuntimeError,match="CHECKPOINT_BYTES_DRIFT"):
        pin.verify()
    (path/WEIGHT).write_bytes(b"regression-only-not-a-real-fp16-vae")
    with pytest.raises(ValueError,match="RIGHTS_OR_REVISION_PIN"):
        replace(pin,license_id="UNVERIFIED_LICENSE").verify()
    cfg=path/"config.json"
    contents=json.loads(cfg.read_text())
    contents["latent_channels"]=8
    cfg.write_text(json.dumps(contents))
    changed=replace(pin,model_config_sha256=digest(cfg.read_bytes()))
    with pytest.raises(RuntimeError,match="INCOMPATIBLE_SDXL_CODEC"):
        changed.verify()
