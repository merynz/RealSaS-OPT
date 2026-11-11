from __future__ import annotations

"""Real .rss-v4 -> C++ 3D clipping -> PNG court across eight camera views.

A synthetic source fixture tests mechanics/presentation execution, not actual
neural appearance or source admission. Clipping masks never consume source
texture nor write to camera depth.
"""
import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance, AttachmentKind, RuntimeV3ClipInterval,
    RuntimeV3FrameComposition, RuntimeV3Slot, TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    RuntimeV4AttachmentAsset, RuntimeV4Clip, RuntimeV4Frame,
    RuntimeV4PlaybackContract, RuntimeV4ViewAssetOverlay,
    RuntimeV4ViewOverlay, provenance_code,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive
from runtime_v3_clipping_native_e2e import _decode_rgba_png, _write_rgba_png


def _quad(x0: float, x1: float, y0: float, y1: float, z: float) -> np.ndarray:
    return np.asarray((
        (x0, y1, z), (x1, y1, z),
        (x0, y0, z), (x1, y0, z),
    ), dtype=np.float32)


def _camera(i: int) -> CameraProjectionV3:
    theta = i * math.pi / 4
    s, c = math.sin(theta), math.cos(theta)
    return CameraProjectionV3(
        f"V{i}", i, (2.0 * s, 0.0, -2.0 * c),
        (c, 0.0, s), (0.0, 1.0, 0.0),
        (-s, 0.0, c), 1.0, 8,
    )


def _fixture(root: Path, *, inverse: bool):
    mesh_faces = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.uint32)
    mask_rest = _quad(-1.0, 0.0, -1, 1, .10)
    body_rest = _quad(-1, 1, -1, 1, .25)
    # Third slot intentionally OUTSIDE the mask scope; this must remain
    # visible even when its pixels are outside the geometry clip.
    after_rest = _quad(.35, 1, -1, -.35, .05)
    specs = [
        ("mask", "clip_slot", "clip_attachment", AttachmentKind.CLIPPING, mask_rest),
        ("body", "body_slot", "body_attachment", AttachmentKind.DEFORMABLE_BODY, body_rest),
        ("after", "after_slot", "after_attachment", AttachmentKind.RIGID_COMPONENT, after_rest),
    ]
    assets = tuple(
        RuntimeV4AttachmentAsset(
            aid, slot, attachment, kind,
            TopologyClass.CLIP_DYNAMIC if kind == AttachmentKind.CLIPPING
            else TopologyClass.STATIC,
            rest, mesh_faces, hashlib.sha256(aid.encode()).hexdigest(),
        )
        for aid, slot, attachment, kind, rest in specs
    )
    textures = []
    views = []
    tex_root = root / "textures"
    for i in range(8):
        view_id = f"V{i}"
        texture = tex_root / (view_id + ".png")
        # Green in left half, blue in right half of the source atlas.
        rgba = b"".join(
            bytes((0, 255, 0, 255)) * 2 + bytes((0, 0, 255, 255)) * 2
            for _ in range(4)
        )
        _write_rgba_png(texture, 4, 4, rgba)
        raw = texture.read_bytes()
        textures.append(RuntimeV3TexturePayload(
            view_id, str(Path("textures") / texture.name),
            hashlib.sha256(raw).hexdigest(), zlib.crc32(raw) & 0xffffffff, 4, 4,
        ))
        overlays = []
        for aid, _, _, kind, _ in specs:
            uv_value = (1., .5) if aid == "after" else (0., .5)
            uv = np.asarray([uv_value] * 4, dtype=np.float32)
            provenance = (
                AppearanceProvenance.UNSEEN if kind == AttachmentKind.CLIPPING
                else AppearanceProvenance.DIRECT_SOURCE
            )
            overlays.append(RuntimeV4ViewAssetOverlay(
                aid, uv,
                np.asarray([provenance_code(provenance)] * 2, dtype=np.uint8),
                np.asarray([-1 if kind == AttachmentKind.CLIPPING else i] * 2,
                           dtype=np.int16),
            ))
        views.append(RuntimeV4ViewOverlay(view_id, i, _camera(i), tuple(overlays)))
    slots = (
        RuntimeV3Slot("clip_slot", "root", 0, "clip_attachment"),
        RuntimeV3Slot("body_slot", "root", 1, "body_attachment"),
        RuntimeV3Slot("after_slot", "root", 2, "after_attachment"),
    )
    contract = RuntimeV4PlaybackContract(slots, assets, tuple(views))
    compositions = {
        f"V{i}": RuntimeV3FrameComposition(
            f"V{i}", ("clip_slot", "body_slot", "after_slot"),
            {
                "clip_slot": "clip_attachment",
                "body_slot": "body_attachment",
                "after_slot": "after_attachment",
            },
            (RuntimeV3ClipInterval(
                "clip_attachment", "clip_slot", "body_slot", inverse,
            ),),
        ) for i in range(8)
    }
    first = {name: rest for name, _, _, _, rest in specs}
    second = dict(first)
    second["mask"] = mask_rest + np.asarray((1., 0., 0.), dtype=np.float32)
    frames = (
        RuntimeV4Frame(0., first, compositions),
        RuntimeV4Frame(1., second, compositions),
    )
    clip = RuntimeV4Clip("idle", "Idle", "idle", 1., 30., False, frames, True)
    package = root / ("inverse.rss" if inverse else "normal.rss")
    materialize_runtime_v4_archive(
        out_path=package, texture_root=root,
        contract=contract, textures=tuple(textures), clips=(clip,),
        source_product_state_hash="c"*64, source_proof_bundle_hash="d"*64,
    )
    return package


def _render(demo: Path, package: Path, *, view: str, t: float, out: Path):
    proc = subprocess.run([
        str(demo), str(package), "--clip", "idle", "--view", view,
        "--time", str(t), "--out", str(out),
    ], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(f"native clipping failed: {view}, t={t}: {proc.stdout}\n{proc.stderr}")
    width, height, pixels = _decode_rgba_png(out)
    assert (width, height) == (8, 8)
    return np.frombuffer(pixels, np.uint8).reshape(height, width, 4)


def run(demo: Path):
    with tempfile.TemporaryDirectory(prefix="rss-v4-3d-clip-") as tmp:
        root = Path(tmp)
        normal = _fixture(root, inverse=False)
        inverse = _fixture(root, inverse=True)
        # Both packages have independent clip states. Eight views share the
        # exact same animated mechanical vertex arrays.
        for view_index in range(8):
            _render(demo, normal, view=f"V{view_index}", t=0.,
                    out=root / f"all-{view_index}.png")
        before = _render(demo, normal, view="V0", t=0., out=root/"before.png")
        moved = _render(demo, normal, view="V0", t=1., out=root/"moved.png")
        inverted = _render(demo, inverse, view="V0", t=0., out=root/"inverse.png")
        back = _render(demo, normal, view="V4", t=0., out=root/"back.png")

        green = (0, 255, 0, 255)
        blue = (0, 0, 255, 255)
        assert tuple(before[1, 1]) == green, before[1, 1]
        assert before[1, 6, 3] == 0, before[1, 6]
        assert tuple(before[6, 6]) == blue, before[6, 6]
        assert moved[1, 1, 3] == 0, moved[1, 1]
        assert tuple(moved[1, 6]) == green, moved[1, 6]
        assert tuple(inverted[1, 6]) == green, inverted[1, 6]
        assert inverted[1, 1, 3] == 0, inverted[1, 1]
        assert tuple(inverted[6, 6]) == blue, inverted[6, 6]
        assert tuple(back[1, 6]) == green, back[1, 6]
        assert back[1, 1, 3] == 0, back[1, 1]
        print("RUNTIME_V4_EIGHT_VIEW_CANONICAL_CLIPPING_NATIVE_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-demo", required=True)
    args = parser.parse_args()
    run(Path(args.runtime_demo).resolve())


if __name__ == "__main__":
    main()
