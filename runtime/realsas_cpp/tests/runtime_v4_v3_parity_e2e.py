from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from compiler.realsas_compiler_core.playback_full_surface_v3 import project_points_xyz_v3  # noqa: E402
from compiler.realsas_compiler_core.playback_runtime_v3 import (  # noqa: E402
    AppearanceProvenance,
    RuntimeV3AppearancePatch,
    RuntimeV3Mesh,
    RuntimeV3PlaybackContract,
    RuntimeV3Vertex,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3Clip, RuntimeV3Frame, materialize_runtime_v3_archive  # noqa: E402
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive  # noqa: E402
from runtime_v4_writer_native_e2e import build_fixture  # noqa: E402


def _v3_from_v4(contract4, clip4):
    meshes = []
    patches = []
    for view in contract4.views:
        for row in view.assets:
            asset = next(x for x in contract4.assets if x.asset_id == row.asset_id)
            xyz = project_points_xyz_v3(asset.rest_xyz, view.camera)
            vertices = tuple(RuntimeV3Vertex(float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]), float(row.uv[i,0]), float(row.uv[i,1])) for i in range(asset.vertex_count))
            mesh_id = f"{view.view_id}:{asset.attachment_id}"
            meshes.append(RuntimeV3Mesh(view.view_id, mesh_id, asset.slot_id, asset.attachment_id, asset.attachment_kind, asset.topology_class, vertices, tuple(tuple(map(int,t)) for t in asset.triangles.tolist())))
            patches.append(RuntimeV3AppearancePatch(f"{mesh_id}:p0", mesh_id, tuple(range(asset.face_count)), AppearanceProvenance.DIRECT_SOURCE, view.view_index, view.view_id))
    contract3 = RuntimeV3PlaybackContract(contract4.slots, tuple(meshes), tuple(patches), contract4.visibility, contract4.raster, contract4.allow_completion)
    frames = []
    for frame4 in clip4.frames:
        posed = {}
        for view in contract4.views:
            for asset in contract4.assets:
                posed[f"{view.view_id}:{asset.attachment_id}"] = tuple(tuple(map(float,row)) for row in project_points_xyz_v3(frame4.canonical_posed_xyz_by_asset[asset.asset_id], view.camera))
        frames.append(RuntimeV3Frame(frame4.time_seconds, posed, frame4.composition_by_view))
    clip3 = RuntimeV3Clip(clip4.clip_id, clip4.display_name, clip4.intent, clip4.duration_seconds, clip4.fps, clip4.loop, tuple(frames), clip4.runtime_qualified)
    return contract3, (clip3,)


def _run_demo(demo: Path, package: Path, out: Path):
    proc = subprocess.run([str(demo), str(package), "--clip", "idle", "--view", "V0", "--time", "0.5", "--out", str(out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0: raise AssertionError(proc.stdout)


def run(v3_demo: Path, v4_demo: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_v4_v3_parity_") as tmp:
        root = Path(tmp); texture_root = root / "texture_root"
        contract4, textures, clips4 = build_fixture(texture_root)
        package4 = root / "fixture_v4.rss"
        materialize_runtime_v4_archive(out_path=package4, texture_root=texture_root, contract=contract4, textures=textures, clips=clips4, source_product_state_hash="c"*64, source_proof_bundle_hash="d"*64)
        contract3, clips3 = _v3_from_v4(contract4, clips4[0])
        package3 = root / "fixture_v3.rss"
        materialize_runtime_v3_archive(out_path=package3, texture_root=texture_root, contract=contract3, textures=textures, clips=clips3, source_product_state_hash="c"*64, source_proof_bundle_hash="d"*64)
        out3 = root / "v3.png"; out4 = root / "v4.png"
        _run_demo(v3_demo, package3, out3); _run_demo(v4_demo, package4, out4)
        if out3.read_bytes() != out4.read_bytes():
            raise AssertionError("Runtime-v3 / Runtime-v4 native projection parity failed")
        print("RUNTIME_V4_V3_NATIVE_PROJECTION_PARITY_PASS")


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--runtime-v3-demo",required=True);ap.add_argument("--runtime-v4-demo",required=True);a=ap.parse_args();run(Path(a.runtime_v3_demo).resolve(),Path(a.runtime_v4_demo).resolve())
if __name__=="__main__":main()
