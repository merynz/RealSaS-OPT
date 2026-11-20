from __future__ import annotations

"""Offline bootstrap ABI checks only; never pass these off as model quality."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.rss_vnext import realsas_sdxl_mvadapter_pretrain_v1 as m


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _fixture(tmp_path):
    base = tmp_path / "sdxl"
    adapter = tmp_path / "adapter"
    code = tmp_path / "source"
    (base / "unet").mkdir(parents=True)
    for n in ("vae", "text_encoder", "text_encoder_2"):
        (base / n).mkdir()
        (base / n / "config.json").write_text("{}")
    (base / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
    }))
    (base / "unet" / "config.json").write_text(json.dumps({
        "in_channels": 4,
        "out_channels": 4,
        "cross_attention_dim": 2048,
        "addition_embed_type": "text_time",
    }))
    unet = base / m.BASE_UNET_WEIGHT
    unet.write_bytes(b"fake-test-only-unet-not-a-trained-checkpoint")
    adapter.mkdir()
    weight = adapter / m.ADAPTER_WEIGHT
    weight.write_bytes(b"fake-test-only-adapter-not-a-trained-checkpoint")
    (code / "mvadapter").mkdir(parents=True)
    (code / "mvadapter" / "__init__.py").write_text("# test source\n")
    pin = m.StrongSDXLMVAdapterPinV1(
        base_snapshot_dir=str(base), base_revision="a" * 40,
        base_tree_sha256=m.pinned_source_tree_sha256_v1(
            base, suffixes=(".safetensors", ".json", ".txt", ".md", ".model"),
        ),
        base_unet_sha256=m._sha256_file(unet),
        adapter_snapshot_dir=str(adapter), adapter_revision="b" * 40,
        adapter_weight_sha256=m._sha256_file(weight),
        upstream_code_dir=str(code), upstream_code_revision="c" * 40,
        upstream_code_tree_sha256=m.pinned_source_tree_sha256_v1(
            code / "mvadapter", suffixes=(".py",),
        ),
        commercial_review_record_sha256="d" * 64,
    )
    return pin, base, adapter, code


def test_pins_exact_sdxl_architecture_and_upstream_adapter_bytes(tmp_path):
    pin, _, _, _ = _fixture(tmp_path)
    proof = m.verify_strong_pretrained_snapshots_v1(pin)
    assert proof["actual_adapter_sha256"] == pin.adapter_weight_sha256
    assert proof["pretrained_stage_b_single_reference_rgb_only"] is True
    assert proof["trained_two_source_rgb_normal_multiview_proven"] is False
    assert proof["product_release_qualified"] is False


def test_changed_weight_or_source_code_is_rejected(tmp_path):
    pin, base, adapter, code = _fixture(tmp_path)
    (adapter / m.ADAPTER_WEIGHT).write_bytes(b"other model")
    with pytest.raises(RuntimeError, match="CHECKPOINT_BYTES_DRIFT"):
        m.verify_strong_pretrained_snapshots_v1(pin)
    pin, base, adapter, code = _fixture(tmp_path / "modified-source")
    (code / "mvadapter" / "__init__.py").write_text("# silently modified")
    with pytest.raises(RuntimeError, match="UPSTREAM_CODE_DRIFT"):
        m.verify_strong_pretrained_snapshots_v1(pin)


def test_sdxl_wrong_architecture_fails_even_if_honestly_rehashed(tmp_path):
    pin, base, _, _ = _fixture(tmp_path)
    cfg = base / "unet" / "config.json"
    value = json.loads(cfg.read_text())
    value["in_channels"] = 8
    cfg.write_text(json.dumps(value))
    pin = replace(pin, base_tree_sha256=m.pinned_source_tree_sha256_v1(
        base, suffixes=(".safetensors", ".json", ".txt", ".md", ".model"),
    ))
    with pytest.raises(RuntimeError, match="NOT_REAL_SDXL_BASE"):
        m.verify_strong_pretrained_snapshots_v1(pin)


def test_false_training_provenance_and_license_injection_rejected(tmp_path):
    pin, _, _, _ = _fixture(tmp_path)
    for patched in (
        replace(pin, pretrained_adapter_is_two_source=True),
        replace(pin, pretrained_adapter_generates_normals=True),
        replace(pin, product_release_qualified=True),
        replace(pin, adapter_license="AGPL-3.0"),
        replace(pin, base_repo="example/other-model"),
        replace(pin, adapter_revision="main"),
    ):
        with pytest.raises(ValueError, match="SOURCE_OR_LICENSE_PIN_INVALID"):
            m.verify_strong_pretrained_snapshots_v1(patched)


def test_loader_calls_official_pipeline_and_genuine_adapter_slots(tmp_path, monkeypatch):
    pin, _, _, _ = _fixture(tmp_path)
    record = []

    class FakeModule:
        def to(self, **kwargs):
            record.append(("to", kwargs))
            return self

        def eval(self):
            record.append(("eval", None))
            return self

    class FakePipe:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            record.append(("from_pretrained", (path, kwargs)))
            return cls()

        def __init__(self):
            self.unet = FakeModule()
            self.unet.config = SimpleNamespace(
                in_channels=4, out_channels=4, cross_attention_dim=2048,
            )
            self.cond_encoder = FakeModule()

        def init_custom_adapter(self, **kwargs):
            record.append(("init_custom_adapter", kwargs))

        def load_custom_adapter(self, path, **kwargs):
            record.append(("load_custom_adapter", (path, kwargs)))

        def to(self, **kwargs):
            record.append(("pipeline_to", kwargs))
            return self

        def enable_vae_slicing(self):
            record.append(("enable_vae_slicing", None))

    monkeypatch.setattr(
        m, "_verified_upstream_module",
        lambda *_args, **_kwargs: SimpleNamespace(
            MVAdapterI2MVSDXLPipeline=FakePipe,
        ),
    )
    loaded = m.load_official_pretrained_i2mv_stage_b_v1(
        pin, device="cpu", dtype="float32",
    )
    assert isinstance(loaded, FakePipe)
    assert ("init_custom_adapter", {"num_views": 6}) in record
    assert any(k == "load_custom_adapter"
               and v[1] == {"weight_name": m.ADAPTER_WEIGHT}
               for k, v in record)
    assert next(v for k, v in record if k == "from_pretrained")[1] == {
        "local_files_only": True,
        "use_safetensors": True,
        "torch_dtype": "float32",
    }


def test_reference_loader_cannot_promote_to_trained_paired_stage_ab():
    with pytest.raises(RuntimeError, match="NOT_TRAINED_PAIRED_RGB_NORMAL"):
        m.assert_pretrained_is_not_final_stage_ab_v1(
            canonical_trained_for_two_sources=True,
            multiview_trained_for_two_sources=True,
            multiview_trained_for_rgb_and_normals=True,
        )
    with pytest.raises(TypeError, match="FLAGS_INVALID"):
        m.assert_pretrained_is_not_final_stage_ab_v1(
            canonical_trained_for_two_sources=1,
            multiview_trained_for_two_sources=False,
            multiview_trained_for_rgb_and_normals=False,
        )
