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
    RuntimeV3ClipInterval,
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


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _write_rgba_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    raw = b"".join(
        b"\x00" + rgba[y * width * 4:(y + 1) * width * 4]
        for y in range(height)
    )
    payload = (
        PNG_MAGIC
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
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
                raise AssertionError("unexpected PNG encoding")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise AssertionError("PNG missing IHDR")

    packed = zlib.decompress(bytes(idat))
    stride = width * 4
    rows: list[bytearray] = []
    offset = 0
    for _ in range(height):
        filter_type = packed[offset]
        offset += 1
        scan = bytearray(packed[offset:offset + stride])
        offset += stride
        prior = rows[-1] if rows else bytearray(stride)
        for x in range(stride):
            left = scan[x - 4] if x >= 4 else 0
            up = prior[x]
            up_left = prior[x - 4] if x >= 4 else 0
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
                pred = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                value = (scan[x] + pred) & 0xFF
            else:
                raise AssertionError(f"unsupported PNG filter {filter_type}")
            scan[x] = value
        rows.append(scan)
    return width, height, b"".join(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quad(
    *,
    view_id: str,
    mesh_id: str,
    slot_id: str,
    attachment_id: str,
    kind: AttachmentKind,
    x0: float,
    x1: float,
    z: float,
) -> RuntimeV3Mesh:
    return RuntimeV3Mesh(
        view_id=view_id,
        mesh_id=mesh_id,
        slot_id=slot_id,
        attachment_id=attachment_id,
        attachment_kind=kind,
        topology_class=(TopologyClass.CLIP_DYNAMIC if kind == AttachmentKind.CLIPPING else TopologyClass.STATIC),
        vertices=(
            RuntimeV3Vertex(x0, 0.0, z, 0.5, 0.5),
            RuntimeV3Vertex(x1, 0.0, z, 0.5, 0.5),
            RuntimeV3Vertex(x0, 4.0, z, 0.5, 0.5),
            RuntimeV3Vertex(x1, 4.0, z, 0.5, 0.5),
        ),
        triangles=((0, 1, 2), (1, 3, 2)),
    )


def _fixture(texture_root: Path):
    views = tuple(f"V{i}" for i in range(8))
    texture_rows = []
    green = bytes((0, 255, 0, 255) * 16)
    for view_id in views:
        rel = f"textures/{view_id}.png"
        path = texture_root / rel
        _write_rgba_png(path, 4, 4, green)
        payload = path.read_bytes()
        texture_rows.append(RuntimeV3TexturePayload(
            view_id=view_id,
            texture_path=rel,
            texture_sha256=_sha(path),
            texture_crc32=zlib.crc32(payload) & 0xFFFFFFFF,
            width=4,
            height=4,
        ))

    slots = (
        RuntimeV3Slot("clip_slot", "root", 0, "clip_attachment"),
        RuntimeV3Slot("body_slot", "root", 1, "body_attachment"),
    )
    meshes = []
    patches = []
    posed = {}
    compositions = {}
    for view_index, view_id in enumerate(views):
        clip_mesh = _quad(
            view_id=view_id,
            mesh_id=f"{view_id}:clip",
            slot_id="clip_slot",
            attachment_id="clip_attachment",
            kind=AttachmentKind.CLIPPING,
            x0=0.0,
            x1=2.0,
            z=0.1,
        )
        body_mesh = _quad(
            view_id=view_id,
            mesh_id=f"{view_id}:body",
            slot_id="body_slot",
            attachment_id="body_attachment",
            kind=AttachmentKind.DEFORMABLE_BODY,
            x0=0.0,
            x1=4.0,
            z=0.5,
        )
        meshes.extend((clip_mesh, body_mesh))
        patches.extend((
            RuntimeV3AppearancePatch(
                patch_id=f"{view_id}:clip_patch",
                mesh_id=clip_mesh.mesh_id,
                face_indices=(0, 1),
                provenance=AppearanceProvenance.UNSEEN,
                donor_view_index=None,
                atlas_id=None,
            ),
            RuntimeV3AppearancePatch(
                patch_id=f"{view_id}:body_patch",
                mesh_id=body_mesh.mesh_id,
                face_indices=(0, 1),
                provenance=AppearanceProvenance.DIRECT_SOURCE,
                donor_view_index=view_index,
                atlas_id=view_id,
            ),
        ))
        posed[clip_mesh.mesh_id] = tuple((v.x, v.y, v.z) for v in clip_mesh.vertices)
        posed[body_mesh.mesh_id] = tuple((v.x, v.y, v.z) for v in body_mesh.vertices)
        compositions[view_id] = RuntimeV3FrameComposition(
            view_id=view_id,
            draw_order_slot_ids=("clip_slot", "body_slot"),
            active_attachment_by_slot={
                "clip_slot": "clip_attachment",
                "body_slot": "body_attachment",
            },
            clip_intervals=(
                RuntimeV3ClipInterval(
                    clip_attachment_id="clip_attachment",
                    start_slot_id="clip_slot",
                    end_slot_id="body_slot",
                    inverse=False,
                ),
            ),
        )

    contract = RuntimeV3PlaybackContract(
        slots=slots,
        meshes=tuple(meshes),
        appearance_patches=tuple(patches),
    )
    frame = RuntimeV3Frame(0.0, posed, compositions)
    clip = RuntimeV3Clip(
        clip_id="idle",
        display_name="Idle",
        intent="idle",
        duration_seconds=1.0,
        fps=30.0,
        loop=False,
        frames=(frame,),
        runtime_qualified=True,
    )
    return contract, tuple(texture_rows), (clip,)


def run(runtime_demo: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_runtime_v3_clip_") as tmp:
        root = Path(tmp)
        texture_root = root / "texture_root"
        contract, textures, clips = _fixture(texture_root)
        package = root / "clip_fixture.rss"
        materialize_runtime_v3_archive(
            out_path=package,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="c" * 64,
            source_proof_bundle_hash="d" * 64,
        )

        out_png = root / "frame.png"
        proc = subprocess.run(
            [
                str(runtime_demo), str(package),
                "--clip", "idle",
                "--view", "V0",
                "--time", "0.0",
                "--out", str(out_png),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(f"native runtime-v3 clipping render failed\n{proc.stdout}")

        width, height, rgba = _decode_rgba_png(out_png)
        if (width, height) != (4, 4):
            raise AssertionError("clipping output dimensions mismatch")

        def pixel(x: int, y: int) -> tuple[int, int, int, int]:
            i = (y * width + x) * 4
            return tuple(rgba[i:i + 4])

        inside = pixel(0, 0)
        inside_second = pixel(1, 2)
        outside = pixel(3, 0)
        outside_second = pixel(2, 2)
        if inside != (0, 255, 0, 255) or inside_second != (0, 255, 0, 255):
            raise AssertionError(f"clip mask failed to preserve inside pixels: {inside} {inside_second}")
        if outside[3] != 0 or outside_second[3] != 0:
            raise AssertionError(f"clip mask failed to reject outside pixels: {outside} {outside_second}")

        print("RUNTIME_V3_NATIVE_CLIPPING_E2E_PASS")
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
