from __future__ import annotations

import binascii
import hashlib
from pathlib import Path
import struct
import zlib

import numpy as np

from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3FrameComposition,
    RuntimeV3Slot,
    TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    RuntimeV4AttachmentAsset,
    RuntimeV4Clip,
    RuntimeV4Frame,
    RuntimeV4PlaybackContract,
    RuntimeV4ViewAssetOverlay,
    RuntimeV4ViewOverlay,
    provenance_code,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4_cache import (
    materialize_runtime_v4_archive_cached,
    runtime_v4_cache_identity,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_png(path: Path) -> None:
    rgba = bytes((255, 255, 255, 255))
    raw = b"\x00" + rgba
    payload = (
        PNG_MAGIC
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 1))
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture(root: Path):
    rest = np.asarray(((-0.5, 0.5, 0.3), (0.5, 0.5, 0.3), (-0.5, -0.5, 0.3)), dtype=np.float32)
    tri = np.asarray(((0, 1, 2),), dtype=np.uint32)
    asset = RuntimeV4AttachmentAsset(
        "body_asset",
        "body_slot",
        "body_attachment",
        AttachmentKind.DEFORMABLE_BODY,
        TopologyClass.STATIC,
        rest,
        tri,
        "a" * 64,
    )

    views = []
    textures = []
    uv = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float32)
    prov = np.asarray((provenance_code(AppearanceProvenance.DIRECT_SOURCE),), dtype=np.uint8)
    for i in range(8):
        view_id = f"V{i}"
        camera = CameraProjectionV3(
            view_id,
            i,
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            1,
        )
        views.append(RuntimeV4ViewOverlay(
            view_id,
            i,
            camera,
            (RuntimeV4ViewAssetOverlay(
                "body_asset",
                uv,
                prov,
                np.asarray((i,), dtype=np.int16),
            ),),
        ))
        rel = f"textures/{view_id}.png"
        path = root / rel
        _write_png(path)
        raw = path.read_bytes()
        textures.append(RuntimeV3TexturePayload(
            view_id,
            rel,
            hashlib.sha256(raw).hexdigest(),
            zlib.crc32(raw) & 0xFFFFFFFF,
            1,
            1,
        ))

    contract = RuntimeV4PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        assets=(asset,),
        views=tuple(views),
    )
    compositions = {
        f"V{i}": RuntimeV3FrameComposition(
            f"V{i}",
            ("body_slot",),
            {"body_slot": "body_attachment"},
        )
        for i in range(8)
    }
    frame0 = RuntimeV4Frame(0.0, {"body_asset": rest}, compositions)
    frame1 = RuntimeV4Frame(1.0, {"body_asset": rest + np.asarray((0.0, 0.1, 0.0), np.float32)}, compositions)
    clip = RuntimeV4Clip("idle", "Idle", "idle", 1.0, 30.0, False, (frame0, frame1), True)
    return contract, tuple(textures), (clip,)


def _call(tmp_path: Path, *, output_name: str, source_hash: str = "c" * 64):
    texture_root = tmp_path / "textures-root"
    contract, textures, clips = _fixture(texture_root)
    return materialize_runtime_v4_archive_cached(
        out_path=tmp_path / output_name,
        texture_root=texture_root,
        contract=contract,
        textures=textures,
        clips=clips,
        source_product_state_hash=source_hash,
        source_proof_bundle_hash="d" * 64,
        cache_root=tmp_path / "cache",
    )


def test_runtime_v4_cache_cold_then_warm_restore(tmp_path: Path):
    cold = _call(tmp_path, output_name="cold.rss")
    assert cold["cache_hit"] is False

    warm = _call(tmp_path, output_name="warm.rss")
    assert warm["cache_hit"] is True
    assert warm["cache_reason"] == "HIT_RESTORED"
    assert warm["cache_key"] == cold["cache_key"]
    assert warm["archive_sha256"] == cold["archive_sha256"]
    assert (tmp_path / "warm.rss").read_bytes() == (tmp_path / "cold.rss").read_bytes()

    same = _call(tmp_path, output_name="warm.rss")
    assert same["cache_hit"] is True
    assert same["cache_output_already_current"] is True
    assert same["cache_reason"] == "HIT_OUTPUT_ALREADY_CURRENT"


def test_runtime_v4_cache_invalidates_on_product_state_change(tmp_path: Path):
    first = _call(tmp_path, output_name="a.rss", source_hash="c" * 64)
    changed = _call(tmp_path, output_name="b.rss", source_hash="e" * 64)
    assert first["cache_key"] != changed["cache_key"]
    assert changed["cache_hit"] is False


def test_runtime_v4_cache_identity_rejects_texture_byte_drift(tmp_path: Path):
    texture_root = tmp_path / "textures-root"
    contract, textures, clips = _fixture(texture_root)
    (texture_root / textures[0].texture_path).write_bytes(b"not-the-sealed-texture")
    try:
        runtime_v4_cache_identity(
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="c" * 64,
            source_proof_bundle_hash="d" * 64,
            cache_root=tmp_path / "cache",
        )
    except Exception as exc:
        assert "TEXTURE_SHA_MISMATCH" in str(exc)
    else:
        raise AssertionError("texture byte drift was accepted by cache identity")



def test_runtime_v4_cache_identity_separates_product_proof_from_admission_authority(tmp_path: Path):
    texture_root = tmp_path / "textures-root"
    contract, textures, clips = _fixture(texture_root)
    common = dict(
        texture_root=texture_root,
        contract=contract,
        textures=textures,
        clips=clips,
        source_product_state_hash="c" * 64,
        source_proof_bundle_hash="d" * 64,
        cache_root=tmp_path / "cache",
    )
    _cache_a, product_key, product_inputs = runtime_v4_cache_identity(
        **common,
        source_authority_kind="PRODUCT_PROOF_BUNDLE",
    )
    _cache_b, admission_key, admission_inputs = runtime_v4_cache_identity(
        **common,
        source_authority_kind="RUNTIME_V4_ADMISSION_CERTIFICATE",
    )
    assert product_key != admission_key
    assert product_inputs["source_authority_kind"] == "PRODUCT_PROOF_BUNDLE"
    assert admission_inputs["source_authority_kind"] == "RUNTIME_V4_ADMISSION_CERTIFICATE"
