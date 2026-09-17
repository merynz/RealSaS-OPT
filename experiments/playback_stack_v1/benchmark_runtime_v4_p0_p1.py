from __future__ import annotations

"""Reference-workstation P0/P1 measurement for the preregistered dense BODY scale.

This is intentionally synthetic representation/performance evidence. It does not claim
product appearance, semantic motion, or Founder Visual PASS.
"""

import binascii
import hashlib
import json
from pathlib import Path
import struct
import tempfile
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
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive

VERTEX_COUNT = 257_505
FACE_COUNT = 500_000
FRAME_COUNT = 18
VIEW_COUNT = 8

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)


def _write_rgba_png(path: Path) -> None:
    width = height = 2
    rgba = bytes((255, 255, 255, 255) * 4)
    raw = b"".join(b"\x00" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))
    payload = PNG_MAGIC + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + _png_chunk(b"IDAT", zlib.compress(raw, 1)) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _camera(view_id: str, index: int) -> CameraProjectionV3:
    angle = (2.0 * np.pi * index) / VIEW_COUNT
    right = (float(np.cos(angle)), 0.0, float(-np.sin(angle)))
    forward = (float(np.sin(angle)), 0.0, float(np.cos(angle)))
    return CameraProjectionV3(
        view_id=view_id,
        view_index=index,
        origin=(0.0, 0.0, -2.0),
        right=right,
        screen_up=(0.0, 1.0, 0.0),
        forward=forward,
        half_extent=2.0,
        resolution=1024,
    )


def _fixture(root: Path):
    idx = np.arange(VERTEX_COUNT, dtype=np.float32)
    rest = np.empty((VERTEX_COUNT, 3), dtype=np.float32)
    rest[:, 0] = ((idx % 719.0) / 719.0 - 0.5) * 2.0
    rest[:, 1] = (((idx // 719.0) % 359.0) / 359.0 - 0.5) * 2.0
    rest[:, 2] = np.sin(idx * np.float32(0.001)) * np.float32(0.1)

    base = np.arange(FACE_COUNT, dtype=np.uint32) % np.uint32(VERTEX_COUNT - 2)
    faces = np.stack((base, base + 1, base + 2), axis=1).astype(np.uint32, copy=False)
    asset = RuntimeV4AttachmentAsset(
        asset_id="body_asset",
        slot_id="body_slot",
        attachment_id="body_attachment",
        attachment_kind=AttachmentKind.DEFORMABLE_BODY,
        topology_class=TopologyClass.STATIC,
        rest_xyz=rest,
        triangles=faces,
        sealed_source_hash="a" * 64,
    )

    uv = np.empty((VERTEX_COUNT, 2), dtype=np.float32)
    uv[:, 0] = (idx % 1024.0) / 1023.0
    uv[:, 1] = ((idx // 1024.0) % 1024.0) / 1023.0
    provenance = np.full((FACE_COUNT,), provenance_code(AppearanceProvenance.DIRECT_SOURCE), dtype=np.uint8)
    views = []
    textures = []
    for i in range(VIEW_COUNT):
        view_id = f"V{i}"
        donors = np.full((FACE_COUNT,), i, dtype=np.int16)
        views.append(RuntimeV4ViewOverlay(
            view_id=view_id,
            view_index=i,
            camera=_camera(view_id, i),
            assets=(RuntimeV4ViewAssetOverlay("body_asset", uv, provenance, donors),),
        ))
        rel = f"textures/{view_id}.png"
        path = root / rel
        _write_rgba_png(path)
        raw = path.read_bytes()
        textures.append(RuntimeV3TexturePayload(
            view_id=view_id,
            texture_path=rel,
            texture_sha256=hashlib.sha256(raw).hexdigest(),
            texture_crc32=zlib.crc32(raw) & 0xFFFFFFFF,
            width=2,
            height=2,
        ))

    contract = RuntimeV4PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        assets=(asset,),
        views=tuple(views),
    )
    compositions = {
        f"V{i}": RuntimeV3FrameComposition(
            view_id=f"V{i}",
            draw_order_slot_ids=("body_slot",),
            active_attachment_by_slot={"body_slot": "body_attachment"},
        )
        for i in range(VIEW_COUNT)
    }
    frames = []
    for fi in range(FRAME_COUNT):
        posed = rest.copy()
        posed[:, 1] += np.float32(0.01 * np.sin((2.0 * np.pi * fi) / (FRAME_COUNT - 1)))
        frames.append(RuntimeV4Frame(
            time_seconds=fi / float(FRAME_COUNT - 1),
            canonical_posed_xyz_by_asset={"body_asset": posed},
            composition_by_view=compositions,
        ))
    clip = RuntimeV4Clip(
        clip_id="dense_probe",
        display_name="Dense Probe",
        intent="representation_benchmark",
        duration_seconds=1.0,
        fps=17.0,
        loop=False,
        frames=tuple(frames),
        runtime_qualified=False,
    )
    return contract, tuple(textures), (clip,)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_v4_p0_p1_bench_") as temp:
        root = Path(temp)
        texture_root = root / "texture_root"
        contract, textures, clips = _fixture(texture_root)
        out = root / "dense_probe.rss"
        result = materialize_runtime_v4_archive(
            out_path=out,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="c" * 64,
            source_proof_bundle_hash="d" * 64,
        )
        v4_per_frame = VERTEX_COUNT * 3 * 4
        v3_equivalent_per_frame = v4_per_frame * VIEW_COUNT
        report = {
            "schema": "RealSaS.RuntimeV4P0P1Measurement.v1",
            "claim_scope": "SYNTHETIC_REPRESENTATION_AND_WRITER_ONLY",
            "vertex_count": VERTEX_COUNT,
            "face_count": FACE_COUNT,
            "frame_count": FRAME_COUNT,
            "view_count": VIEW_COUNT,
            "v4_canonical_xyz_bytes_per_frame": v4_per_frame,
            "v3_equivalent_xyz_bytes_per_frame": v3_equivalent_per_frame,
            "xyz_reduction_factor": v3_equivalent_per_frame / v4_per_frame,
            "v4_canonical_xyz_bytes_total": v4_per_frame * FRAME_COUNT,
            "v3_equivalent_xyz_bytes_total": v3_equivalent_per_frame * FRAME_COUNT,
            **{k: v for k, v in result.items() if k != "archive_path"},
            "performance_pass_claimed": False,
            "product_pass_claimed": False,
        }
        print("RUNTIME_V4_P0_P1_MEASUREMENT=" + json.dumps(report, sort_keys=True))
        if report["xyz_reduction_factor"] != 8.0:
            raise SystemExit("Runtime-v4 shared canonical XYZ reduction is not exactly 8x")


if __name__ == "__main__":
    main()
