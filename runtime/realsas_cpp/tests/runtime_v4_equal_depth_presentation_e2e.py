from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

import numpy as np

REPO=Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,AttachmentKind,RuntimeV3FrameComposition,RuntimeV3Slot,TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    RuntimeV4AttachmentAsset,RuntimeV4Clip,RuntimeV4Frame,RuntimeV4PlaybackContract,
    RuntimeV4ViewAssetOverlay,RuntimeV4ViewOverlay,provenance_code,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive
from runtime.realsas_cpp.tests.runtime_v4_writer_native_e2e import _decode_first_pixel,_sha,_write_rgba_png


def _textures(root:Path):
    rows=[]
    rgba=bytes((
        255,0,0,255, 0,255,0,255,
        255,0,0,255, 0,255,0,255,
    ))
    for i in range(8):
        rel=f"textures/V{i}.png"; path=root/rel
        _write_rgba_png(path,2,2,rgba)
        raw=path.read_bytes()
        rows.append(RuntimeV3TexturePayload(f"V{i}",rel,_sha(path),zlib.crc32(raw)&0xffffffff,2,2))
    return tuple(rows)


def _fixture(texture_root:Path,*,draw_order,green_other_view:bool):
    textures=_textures(texture_root)
    tri=np.asarray(((0,1,2),),dtype=np.uint32)
    xyz=np.asarray(((-1,1,.2),(1,1,.2),(-1,-1,.2)),dtype=np.float32)
    assets=(
        RuntimeV4AttachmentAsset("red_asset","red_slot","red_att",AttachmentKind.DEFORMABLE_BODY,TopologyClass.STATIC,xyz,tri,"a"*64),
        RuntimeV4AttachmentAsset("green_asset","green_slot","green_att",AttachmentKind.DEFORMABLE_BODY,TopologyClass.STATIC,xyz,tri,"b"*64),
    )
    views=[]
    for i in range(8):
        camera=CameraProjectionV3(f"V{i}",i,(0,0,-1),(1,0,0),(0,1,0),(0,0,1),1.0,2)
        green_prov=AppearanceProvenance.OTHER_VIEW_SOURCE if (i==0 and green_other_view) else AppearanceProvenance.DIRECT_SOURCE
        green_donor=1 if (i==0 and green_other_view) else i
        views.append(RuntimeV4ViewOverlay(
            f"V{i}",i,camera,(
                RuntimeV4ViewAssetOverlay(
                    "red_asset",
                    np.asarray(((0,0),(0,0),(0,0)),np.float32),
                    np.asarray((provenance_code(AppearanceProvenance.DIRECT_SOURCE),),np.uint8),
                    np.asarray((i,),np.int16),
                ),
                RuntimeV4ViewAssetOverlay(
                    "green_asset",
                    np.asarray(((1,0),(1,0),(1,0)),np.float32),
                    np.asarray((provenance_code(green_prov),),np.uint8),
                    np.asarray((green_donor,),np.int16),
                ),
            )
        ))
    contract=RuntimeV4PlaybackContract(
        slots=(RuntimeV3Slot("red_slot","root",0,"red_att"),RuntimeV3Slot("green_slot","root",1,"green_att")),
        assets=assets,views=tuple(views),
    )
    comp={f"V{i}":RuntimeV3FrameComposition(
        f"V{i}",tuple(draw_order),{"red_slot":"red_att","green_slot":"green_att"}
    ) for i in range(8)}
    f0=RuntimeV4Frame(0.0,{"red_asset":xyz,"green_asset":xyz},comp)
    f1=RuntimeV4Frame(1.0,{"red_asset":xyz,"green_asset":xyz},comp)
    clip=RuntimeV4Clip("idle","Idle","idle",1.0,30.0,False,(f0,f1),True)
    return contract,textures,(clip,)


def _render(runtime_demo:Path,root:Path,*,order,green_other_view:bool):
    texture_root=root/"texture_root"
    contract,textures,clips=_fixture(texture_root,draw_order=order,green_other_view=green_other_view)
    package=root/"fixture.rss"
    materialize_runtime_v4_archive(
        out_path=package,texture_root=texture_root,contract=contract,textures=textures,clips=clips,
        source_product_state_hash="c"*64,source_proof_bundle_hash="d"*64,
    )
    out=root/"frame.png"
    proc=subprocess.run(
        [str(runtime_demo),str(package),"--clip","idle","--view","V0","--time","0.5","--out",str(out)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
    )
    if proc.returncode!=0:
        raise AssertionError(proc.stdout)
    return _decode_first_pixel(out)


def run(runtime_demo:Path):
    with tempfile.TemporaryDirectory(prefix="realsas_runtime_v4_tie_") as tmp:
        base=Path(tmp)
        # Equal depth + equal provenance: later presentation slot wins.
        p=_render(runtime_demo,base/"slot_green",order=("red_slot","green_slot"),green_other_view=False)
        if not (p[1]>=240 and p[0]<=15):
            raise AssertionError(f"slot-order equal-depth tie failed: {p}")
        p=_render(runtime_demo,base/"slot_red",order=("green_slot","red_slot"),green_other_view=False)
        if not (p[0]>=240 and p[1]<=15):
            raise AssertionError(f"reversed slot-order equal-depth tie failed: {p}")
        # Source-view authority outranks slot order: direct target-view red wins
        # even though cross-view green is later in presentation order.
        p=_render(runtime_demo,base/"source_priority",order=("red_slot","green_slot"),green_other_view=True)
        if not (p[0]>=240 and p[1]<=15):
            raise AssertionError(f"source-provenance equal-depth priority failed: {p}")
        print("RUNTIME_V4_EQUAL_DEPTH_PRESENTATION_TIEBREAK_NATIVE_E2E_PASS")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--runtime-demo",required=True); a=ap.parse_args()
    run(Path(a.runtime_demo).resolve())

if __name__=="__main__":
    main()
