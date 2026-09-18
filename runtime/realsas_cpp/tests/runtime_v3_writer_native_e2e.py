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

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compiler.realsas_compiler_core.playback_runtime_v3 import (  # noqa: E402
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3AppearancePatch,
    RuntimeV3FrameComposition,
    RuntimeV3Mesh,
    RuntimeV3PlaybackContract,
    RuntimeV3Slot,
    RuntimeV3Vertex,
    TopologyClass,
)
from compiler.realsas_compiler_services.export.runtime_v3 import (  # noqa: E402
    RuntimeV3Clip,
    RuntimeV3Frame,
    RuntimeV3TexturePayload,
    materialize_runtime_v3_archive,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)


def _write_rgba_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    assert len(rgba) == width * height * 4
    raw = b"".join(
        b"\x00" + rgba[y * width * 4:(y + 1) * width * 4]
        for y in range(height)
    )
    payload = (
        PNG_MAGIC
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _decode_rgba_png(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(PNG_MAGIC):
        raise AssertionError("native output is not PNG")
    pos = len(PNG_MAGIC)
    width = height = None
    idat = bytearray()
    while pos < len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + size]
        pos += 12 + size
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if (bit_depth, color_type, compression, filtering, interlace) != (8, 6, 0, 0, 0):
                raise AssertionError("unexpected native PNG encoding")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise AssertionError("native PNG missing IHDR")

    packed = zlib.decompress(bytes(idat))
    stride = width * 4
    expected = height * (stride + 1)
    if len(packed) != expected:
        raise AssertionError("native PNG scanline size mismatch")

    rows: list[bytearray] = []
    offset = 0
    bpp = 4
    for _ in range(height):
        filter_type = packed[offset]
        offset += 1
        scan = bytearray(packed[offset:offset + stride])
        offset += stride
        prior = rows[-1] if rows else bytearray(stride)
        for x in range(stride):
            left = scan[x - bpp] if x >= bpp else 0
            up = prior[x]
            up_left = prior[x - bpp] if x >= bpp else 0
            if filter_type == 0:
                value = scan[x]
            elif filter_type == 1:
                value = (scan[x] + left) & 0xFF
            elif filter_type == 2:
                value = (scan[x] + up) & 0xFF
            elif filter_type == 3:
                value = (scan[x] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                value = (scan[x] + predictor) & 0xFF
            else:
                raise AssertionError(f"unsupported PNG filter {filter_type}")
            scan[x] = value
        rows.append(scan)
    return width, height, b"".join(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh(view_id: str, role: str, slot_id: str, attachment_id: str, z: float, u: float, kind: AttachmentKind) -> RuntimeV3Mesh:
    return RuntimeV3Mesh(
        view_id=view_id,
        mesh_id=f"{view_id}:{role}",
        slot_id=slot_id,
        attachment_id=attachment_id,
        attachment_kind=kind,
        topology_class=TopologyClass.STATIC,
        vertices=(
            RuntimeV3Vertex(0.0, 0.0, z, u, 0.5),
            RuntimeV3Vertex(2.0, 0.0, z, u, 0.5),
            RuntimeV3Vertex(0.0, 2.0, z, u, 0.5),
        ),
        triangles=((0, 1, 2),),
    )


def _build_fixture(texture_root: Path):
    views = tuple(f"V{i}" for i in range(8))
    textures = []
    # Left texel = red, right texel = green. Rows are identical, so V is irrelevant.
    rgba = bytes((255, 0, 0, 255, 0, 255, 0, 255) * 2)
    for view_id in views:
        rel = f"textures/{view_id}.png"
        path = texture_root / rel
        _write_rgba_png(path, 2, 2, rgba)
        raw = path.read_bytes()
        textures.append(RuntimeV3TexturePayload(
            view_id=view_id,
            texture_path=rel,
            texture_sha256=_sha256(path),
            texture_crc32=zlib.crc32(raw) & 0xFFFFFFFF,
            width=2,
            height=2,
        ))

    slots = (
        RuntimeV3Slot("far_slot", "root", 0, "far_attachment"),
        RuntimeV3Slot("near_slot", "root", 1, "near_attachment"),
    )
    meshes = []
    patches = []
    posed0 = {}
    posed1 = {}
    for view_index, view_id in enumerate(views):
        far = _mesh(view_id, "far", "far_slot", "far_attachment", 0.80, 0.0, AttachmentKind.DEFORMABLE_BODY)
        near = _mesh(view_id, "near", "near_slot", "near_attachment", 0.20, 1.0, AttachmentKind.RIGID_COMPONENT)
        meshes.extend((far, near))
        patches.extend((
            RuntimeV3AppearancePatch(
                patch_id=f"{far.mesh_id}:face0",
                mesh_id=far.mesh_id,
                face_indices=(0,),
                provenance=AppearanceProvenance.DIRECT_SOURCE,
                donor_view_index=view_index,
                atlas_id=view_id,
            ),
            RuntimeV3AppearancePatch(
                patch_id=f"{near.mesh_id}:face0",
                mesh_id=near.mesh_id,
                face_indices=(0,),
                provenance=AppearanceProvenance.DIRECT_SOURCE,
                donor_view_index=view_index,
                atlas_id=view_id,
            ),
        ))
        posed0[far.mesh_id] = tuple((v.x, v.y, v.z) for v in far.vertices)
        posed0[near.mesh_id] = tuple((v.x, v.y, v.z) for v in near.vertices)
        # Frame 1 changes depth slightly to exercise XYZ interpolation while
        # preserving the same near/far relation.
        posed1[far.mesh_id] = tuple((v.x, v.y, v.z - 0.05) for v in far.vertices)
        posed1[near.mesh_id] = tuple((v.x, v.y, v.z + 0.05) for v in near.vertices)

    contract = RuntimeV3PlaybackContract(
        slots=slots,
        meshes=tuple(meshes),
        appearance_patches=tuple(patches),
    )

    compositions = {
        view_id: RuntimeV3FrameComposition(
            view_id=view_id,
            # Intentionally render the near slot first. A painter-only runtime
            # would then let the later far/red surface overwrite it. Correct
            # posed depth keeps the near/green surface visible.
            draw_order_slot_ids=("near_slot", "far_slot"),
            active_attachment_by_slot={
                "far_slot": "far_attachment",
                "near_slot": "near_attachment",
            },
        )
        for view_id in views
    }
    frames = (
        RuntimeV3Frame(0.0, posed0, compositions),
        RuntimeV3Frame(1.0, posed1, compositions),
    )
    clip = RuntimeV3Clip(
        clip_id="idle",
        display_name="Idle",
        intent="idle",
        duration_seconds=1.0,
        fps=30.0,
        loop=False,
        frames=frames,
        runtime_qualified=True,
    )
    return contract, tuple(textures), (clip,)


def run(runtime_demo: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_runtime_v3_e2e_") as tmp:
        root = Path(tmp)
        texture_root = root / "texture_root"
        contract, textures, clips = _build_fixture(texture_root)
        package = root / "fixture.rss"
        result = materialize_runtime_v3_archive(
            out_path=package,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="a" * 64,
            source_proof_bundle_hash="b" * 64,
        )
        if result["view_count"] != 8 or result["clip_count"] != 1:
            raise AssertionError("runtime-v3 writer fixture count mismatch")

        out_png = root / "frame.png"
        proc = subprocess.run(
            [
                str(runtime_demo), str(package),
                "--clip", "idle",
                "--view", "V0",
                "--time", "0.5",
                "--out", str(out_png),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(f"native runtime-v3 render failed\n{proc.stdout}")
        if "renderer=RUNTIME_V3_POSED_DEPTH_REFERENCE" not in proc.stdout:
            raise AssertionError(f"native runtime-v3 marker missing\n{proc.stdout}")

        width, height, rgba = _decode_rgba_png(out_png)
        if (width, height) != (2, 2):
            raise AssertionError("native runtime-v3 output size mismatch")
        r, g, b, a = rgba[0:4]
        if not (g >= 240 and r <= 15 and b <= 15 and a >= 240):
            raise AssertionError(
                "posed-depth occlusion failed: expected near green surface to win "
                f"despite adverse draw order, got rgba={(r, g, b, a)}\n{proc.stdout}"
            )
        print("RUNTIME_V3_WRITER_NATIVE_DEPTH_E2E_PASS")
        print(proc.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-demo", required=True)
    args = ap.parse_args()
    runtime_demo = Path(args.runtime_demo).resolve()
    if not runtime_demo.is_file():
        raise SystemExit(f"runtime demo missing: {runtime_demo}")
    run(runtime_demo)


if __name__ == "__main__":
    main()
