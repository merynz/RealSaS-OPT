from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from runtime_v3_writer_native_e2e import _decode_rgba_png, _sha256, _write_rgba_png  # noqa: E402
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


def run(runtime_demo: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_runtime_v3_unseen_") as tmp:
        root = Path(tmp)
        texture_root = root / "texture_root"
        views = tuple(f"V{i}" for i in range(8))
        textures = []
        meshes = []
        patches = []
        posed = {}

        # Fully opaque red source texture. If the runtime ignores provenance,
        # the output will visibly turn red and this test will fail.
        rgba = bytes((255, 0, 0, 255) * 4)
        for view_index, view_id in enumerate(views):
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
            mesh = RuntimeV3Mesh(
                view_id=view_id,
                mesh_id=f"{view_id}:body",
                slot_id="body_slot",
                attachment_id="body_attachment",
                attachment_kind=AttachmentKind.DEFORMABLE_BODY,
                topology_class=TopologyClass.STATIC,
                vertices=(
                    RuntimeV3Vertex(0.0, 0.0, 0.2, 0.5, 0.5),
                    RuntimeV3Vertex(2.0, 0.0, 0.2, 0.5, 0.5),
                    RuntimeV3Vertex(0.0, 2.0, 0.2, 0.5, 0.5),
                ),
                triangles=((0, 1, 2),),
            )
            meshes.append(mesh)
            patches.append(RuntimeV3AppearancePatch(
                patch_id=f"{view_id}:unseen",
                mesh_id=mesh.mesh_id,
                face_indices=(0,),
                provenance=AppearanceProvenance.UNSEEN,
                donor_view_index=None,
                atlas_id=None,
            ))
            posed[mesh.mesh_id] = tuple((v.x, v.y, v.z) for v in mesh.vertices)

        contract = RuntimeV3PlaybackContract(
            slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
            meshes=tuple(meshes),
            appearance_patches=tuple(patches),
        )
        compositions = {
            view_id: RuntimeV3FrameComposition(
                view_id=view_id,
                draw_order_slot_ids=("body_slot",),
                active_attachment_by_slot={"body_slot": "body_attachment"},
            )
            for view_id in views
        }
        clip = RuntimeV3Clip(
            clip_id="idle",
            display_name="Idle",
            intent="idle",
            duration_seconds=1.0,
            fps=30.0,
            loop=False,
            frames=(
                RuntimeV3Frame(0.0, posed, compositions),
                RuntimeV3Frame(1.0, posed, compositions),
            ),
            runtime_qualified=True,
        )
        package = root / "fixture.rss"
        materialize_runtime_v3_archive(
            out_path=package,
            texture_root=texture_root,
            contract=contract,
            textures=tuple(textures),
            clips=(clip,),
            source_product_state_hash="c" * 64,
            source_proof_bundle_hash="d" * 64,
        )

        out_png = root / "frame.png"
        proc = subprocess.run(
            [str(runtime_demo), str(package), "--clip", "idle", "--view", "V0", "--time", "0.5", "--out", str(out_png)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(f"native runtime-v3 unseen render failed\n{proc.stdout}")
        width, height, pixels = _decode_rgba_png(out_png)
        if (width, height) != (2, 2):
            raise AssertionError("unexpected native output size")
        if any(pixels):
            raise AssertionError(
                "UNSEEN appearance leaked source texels into product render; "
                f"first_pixel={tuple(pixels[:4])}\n{proc.stdout}"
            )
        print("RUNTIME_V3_UNSEEN_SUPPRESSION_E2E_PASS")


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
