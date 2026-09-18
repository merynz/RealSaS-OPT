from __future__ import annotations

import argparse
from pathlib import Path
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
from runtime.realsas_cpp.tests.runtime_v4_writer_native_e2e import (  # noqa: E402
    _decode_first_pixel,
    _sha,
    _write_rgba_png,
)


def _solid_rgba(r: int, g: int, b: int, a: int = 255) -> bytes:
    return bytes((r, g, b, a) * 4)


def build_fixture(texture_root: Path):
    views = tuple(f"V{i}" for i in range(8))
    textures = []
    for i, view_id in enumerate(views):
        rel = f"textures/{view_id}.png"
        path = texture_root / rel
        if i == 0:
            rgba = _solid_rgba(255, 0, 0)  # target texture: red everywhere
        elif i == 1:
            # donor texture: bottom-right is yellow; all other texels are blue.
            rgba = bytes((
                0, 0, 255, 255,   0, 0, 255, 255,
                0, 0, 255, 255,   255, 255, 0, 255,
            ))
        else:
            rgba = _solid_rgba(0, 255, 0)
        _write_rgba_png(path, 2, 2, rgba)
        raw = path.read_bytes()
        textures.append(RuntimeV3TexturePayload(
            view_id, rel, _sha(path), zlib.crc32(raw) & 0xFFFFFFFF, 2, 2
        ))

    tri = np.asarray(((0, 1, 2),), dtype=np.uint32)
    xyz = np.asarray(((-1, 1, 0.2), (1, 1, 0.2), (-1, -1, 0.2)), dtype=np.float32)
    asset = RuntimeV4AttachmentAsset(
        "body_asset",
        "body_slot",
        "body_attachment",
        AttachmentKind.DEFORMABLE_BODY,
        TopologyClass.STATIC,
        xyz,
        tri,
        "a" * 64,
    )

    overlays = []
    for i, view_id in enumerate(views):
        camera = CameraProjectionV3(
            view_id, i, (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0, 2
        )
        if i == 0:
            provenance = AppearanceProvenance.OTHER_VIEW_SOURCE
            donor = 1
            uv = np.asarray(((0, 0), (0, 0), (0, 0)), dtype=np.float32)
        elif i == 1:
            provenance = AppearanceProvenance.DIRECT_SOURCE
            donor = 1
            uv = np.asarray(((1, 1), (1, 1), (1, 1)), dtype=np.float32)
        else:
            provenance = AppearanceProvenance.DIRECT_SOURCE
            donor = i
            uv = np.asarray(((0, 0), (0, 0), (0, 0)), dtype=np.float32)
        overlays.append(RuntimeV4ViewOverlay(
            view_id,
            i,
            camera,
            (
                RuntimeV4ViewAssetOverlay(
                    "body_asset",
                    uv,
                    np.asarray((provenance_code(provenance),), dtype=np.uint8),
                    np.asarray((donor,), dtype=np.int16),
                ),
            ),
        ))

    contract = RuntimeV4PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        assets=(asset,),
        views=tuple(overlays),
    )
    composition = {
        view_id: RuntimeV3FrameComposition(
            view_id,
            ("body_slot",),
            {"body_slot": "body_attachment"},
        )
        for view_id in views
    }
    frame0 = RuntimeV4Frame(0.0, {"body_asset": xyz}, composition)
    frame1 = RuntimeV4Frame(1.0, {"body_asset": xyz}, composition)
    clip = RuntimeV4Clip("idle", "Idle", "idle", 1.0, 30.0, False, (frame0, frame1), True)
    return contract, tuple(textures), (clip,)


def run(runtime_demo: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_runtime_v4_other_view_") as tmp:
        root = Path(tmp)
        texture_root = root / "texture_root"
        contract, textures, clips = build_fixture(texture_root)
        package = root / "fixture.rss"
        materialize_runtime_v4_archive(
            out_path=package,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="c" * 64,
            source_proof_bundle_hash="d" * 64,
        )
        out = root / "frame.png"
        proc = subprocess.run(
            [str(runtime_demo), str(package), "--clip", "idle", "--view", "V0", "--time", "0.5", "--out", str(out)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stdout)
        pixel = _decode_first_pixel(out)
        # Correct implementation must sample V1 bottom-right = yellow.
        # Legacy target-view/target-UV sampling returns V0 top-left = red.
        if not (pixel[0] >= 240 and pixel[1] >= 240 and pixel[2] <= 15 and pixel[3] >= 240):
            raise AssertionError(f"Runtime-v4 donor texture/UV sampling failed: {pixel}\n{proc.stdout}")
        print("RUNTIME_V4_OTHER_VIEW_SOURCE_NATIVE_E2E_PASS")
        print(proc.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-demo", required=True)
    args = ap.parse_args()
    run(Path(args.runtime_demo).resolve())


if __name__ == "__main__":
    main()
