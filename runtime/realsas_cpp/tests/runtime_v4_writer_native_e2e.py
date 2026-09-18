from __future__ import annotations

import argparse
import binascii
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3  # noqa: E402
from compiler.realsas_compiler_core.playback_runtime_v3 import (  # noqa: E402
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3FrameComposition,
    RuntimeV3Slot,
    TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (  # noqa: E402
    RuntimeV4AttachmentAsset,
    RuntimeV4Clip,
    RuntimeV4Frame,
    RuntimeV4PlaybackContract,
    RuntimeV4ViewAssetOverlay,
    RuntimeV4ViewOverlay,
    provenance_code,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload  # noqa: E402
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)


def _write_rgba_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    raw = b"".join(b"\x00" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))
    payload = PNG_MAGIC + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + _png_chunk(b"IDAT", zlib.compress(raw, 9)) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != PNG_MAGIC or data[12:16] != b"IHDR":
        raise AssertionError("PNG header missing")
    return struct.unpack(">II", data[16:24])


def _decode_first_pixel(path: Path) -> tuple[int, int, int, int]:
    data = path.read_bytes()
    pos = len(PNG_MAGIC)
    width = height = None
    idat = bytearray()
    while pos < len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + size]
        pos += 12 + size
        if kind == b"IHDR": width, height = struct.unpack(">II", payload[:8])
        elif kind == b"IDAT": idat.extend(payload)
        elif kind == b"IEND": break
    packed = zlib.decompress(bytes(idat))
    if width is None or height is None:
        raise AssertionError("PNG dimensions missing")
    stride = width * 4
    rows = []
    offset = 0
    for _ in range(height):
        f = packed[offset]; offset += 1
        scan = bytearray(packed[offset:offset + stride]); offset += stride
        prior = rows[-1] if rows else bytearray(stride)
        for x in range(stride):
            left = scan[x - 4] if x >= 4 else 0; up = prior[x]; ul = prior[x - 4] if x >= 4 else 0
            if f == 1: scan[x] = (scan[x] + left) & 255
            elif f == 2: scan[x] = (scan[x] + up) & 255
            elif f == 3: scan[x] = (scan[x] + ((left + up) // 2)) & 255
            elif f == 4:
                p = left + up - ul; pa,pb,pc = abs(p-left),abs(p-up),abs(p-ul)
                pred = left if pa <= pb and pa <= pc else (up if pb <= pc else ul)
                scan[x] = (scan[x] + pred) & 255
            elif f != 0: raise AssertionError(f"unsupported PNG filter {f}")
        rows.append(scan)
    return tuple(rows[0][:4])


def build_fixture(texture_root: Path, *, separate_atlas_resolution: bool = True):
    views = tuple(f"V{i}" for i in range(8))
    # Writer E2E deliberately exercises a 4x2 atlas feeding a 2x2 framebuffer.
    # Cross-version v3/v4 parity must opt out because Runtime-v3 has no separate
    # framebuffer-resolution contract.
    texture_width = 4 if separate_atlas_resolution else 2
    row = (
        bytes((255, 0, 0, 255) * 2 + (0, 255, 0, 255) * 2)
        if separate_atlas_resolution
        else bytes((255, 0, 0, 255) + (0, 255, 0, 255))
    )
    rgba = row * 2
    textures = []
    for view_id in views:
        rel = f"textures/{view_id}.png"
        path = texture_root / rel
        _write_rgba_png(path, texture_width, 2, rgba)
        raw = path.read_bytes()
        textures.append(RuntimeV3TexturePayload(
            view_id,
            rel,
            _sha(path),
            zlib.crc32(raw) & 0xFFFFFFFF,
            texture_width,
            2,
        ))

    tri = np.asarray(((0, 1, 2),), dtype=np.uint32)
    far_xyz = np.asarray(((-1, 1, 0.8), (1, 1, 0.8), (-1, -1, 0.8)), dtype=np.float32)
    near_xyz = np.asarray(((-1, 1, 0.2), (1, 1, 0.2), (-1, -1, 0.2)), dtype=np.float32)
    assets = (
        RuntimeV4AttachmentAsset("far_asset", "far_slot", "far_attachment", AttachmentKind.DEFORMABLE_BODY, TopologyClass.STATIC, far_xyz, tri, "a" * 64),
        RuntimeV4AttachmentAsset("near_asset", "near_slot", "near_attachment", AttachmentKind.RIGID_COMPONENT, TopologyClass.STATIC, near_xyz, tri, "b" * 64),
    )
    overlays = []
    for i, view_id in enumerate(views):
        camera = CameraProjectionV3(view_id, i, (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0, 2)
        overlays.append(RuntimeV4ViewOverlay(
            view_id,
            i,
            camera,
            (
                RuntimeV4ViewAssetOverlay("far_asset", np.asarray(((0, .5), (0, .5), (0, .5)), np.float32), np.asarray((provenance_code(AppearanceProvenance.DIRECT_SOURCE),), np.uint8), np.asarray((i,), np.int16)),
                RuntimeV4ViewAssetOverlay("near_asset", np.asarray(((1, .5), (1, .5), (1, .5)), np.float32), np.asarray((provenance_code(AppearanceProvenance.DIRECT_SOURCE),), np.uint8), np.asarray((i,), np.int16)),
            ),
        ))
    contract = RuntimeV4PlaybackContract(
        slots=(RuntimeV3Slot("far_slot", "root", 0, "far_attachment"), RuntimeV3Slot("near_slot", "root", 1, "near_attachment")),
        assets=assets,
        views=tuple(overlays),
    )
    compositions = {view_id: RuntimeV3FrameComposition(view_id, ("near_slot", "far_slot"), {"far_slot": "far_attachment", "near_slot": "near_attachment"}) for view_id in views}
    frame0 = RuntimeV4Frame(0.0, {"far_asset": far_xyz, "near_asset": near_xyz}, compositions)
    frame1 = RuntimeV4Frame(1.0, {"far_asset": far_xyz + np.asarray((0,0,-.05),np.float32), "near_asset": near_xyz + np.asarray((0,0,.05),np.float32)}, compositions)
    clip = RuntimeV4Clip("idle", "Idle", "idle", 1.0, 30.0, False, (frame0, frame1), True)
    return contract, tuple(textures), (clip,)


def run(runtime_demo: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_runtime_v4_e2e_") as tmp:
        root = Path(tmp); texture_root = root / "texture_root"
        contract, textures, clips = build_fixture(texture_root)
        package = root / "fixture.rss"
        result = materialize_runtime_v4_archive(out_path=package, texture_root=texture_root, contract=contract, textures=textures, clips=clips, source_product_state_hash="c"*64, source_proof_bundle_hash="d"*64)
        if result["deform_array_bytes_per_frame"] != 2 * 3 * 3 * 4:
            raise AssertionError("Runtime-v4 frame array accounting drift")
        out = root / "frame.png"
        proc = subprocess.run([str(runtime_demo), str(package), "--clip", "idle", "--view", "V0", "--time", "0.5", "--out", str(out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0: raise AssertionError(proc.stdout)
        if "renderer=RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE" not in proc.stdout: raise AssertionError(proc.stdout)
        if _png_dimensions(out) != (2, 2):
            raise AssertionError(f"Runtime-v4 framebuffer incorrectly followed atlas size: {_png_dimensions(out)}")
        pixel = _decode_first_pixel(out)
        if not (pixel[1] >= 240 and pixel[0] <= 15 and pixel[2] <= 15 and pixel[3] >= 240):
            raise AssertionError(f"Runtime-v4 posed depth failed: {pixel}\n{proc.stdout}")
        print("RUNTIME_V4_WRITER_NATIVE_DEPTH_E2E_PASS")
        print(proc.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--runtime-demo", required=True); args = ap.parse_args()
    run(Path(args.runtime_demo).resolve())


if __name__ == "__main__": main()
