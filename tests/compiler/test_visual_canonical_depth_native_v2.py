"""Adversarial native overlap ownership; same sealed package, reversed face order."""
from collections import OrderedDict
import io
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
from PIL import Image
import pytest
from compiler.realsas_compiler_core.runtime_package_v2 import write_rss_v2
from compiler.realsas_compiler_core.visual_depth_v2 import render_visual_depth
from compiler.realsas_compiler_core.visual_domain_v2 import OPERATOR_ID, DEPTH_CONTRACT


def render(tmp_path, depths, order=None):
    player = os.environ.get("REALSAS_RUNTIME_V2_PLAYER")
    if not player: pytest.skip("native consumer binary required")
    layers = len(depths)
    positions = np.tile([[0.,0.],[3.,0.],[0.,3.]], (layers,1))
    faces = np.arange(layers*3, dtype="<u4").reshape(-1,3)
    if order is not None: faces = faces[list(order)]
    uv = np.repeat(np.array([[0.,0.] if i==0 else [1.,1.] for i in range(layers)]),3,axis=0)
    tex = np.zeros((4,4,4),dtype=np.uint8); tex[:]=(0,0,255,255);tex[0,0]=(255,0,0,128)
    png = io.BytesIO();Image.fromarray(tex).save(png,format="PNG")
    manifest = {
        "schema":"RealSaS.RuntimePackage.v2", "presentation_geometry_mode":"SOURCE_OWNED_VISUAL_PRESENTATION_V1",
        "mechanical_mesh_render_authority":"0", "runtime_visual_mesh_rebuild":"0",
        "runtime_binding_solve":"0", "runtime_generation":"0", "donor_search_at_runtime":"0",
        "texture_sampling_contract":"SOURCE_RGBA8_BILINEAR_STRAIGHT_TO_PM_V1", "mip_generation_authorized":"0",
        "view_count":"8", "clip_count":"1", "clip.0.id":"depth", "clip.0.frame_count":"1",
        "playback_sampling_contract":"SEALED_FRAME_INDEX_ONLY", "view_selection_contract":"SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        "cross_direction_blending_authorized":"0", "host_interpolation_authorized":"0",
        "presentation_state_execution_authorized":"0", "clipping_authorized":"0", "tint_order_visibility_authorized":"0",
        "view.0.id":"V0", "view.0.mesh_entry":"mesh", "view.0.texture_entry":"texture", "view.0.resolution":"4",
        "clip.0.view.0.positions_entry":"positions", "clip.0.view.0.depths_entry":"depths",
        "visual_deformation_operator_id":OPERATOR_ID, "depth_ownership_contract":DEPTH_CONTRACT,
        "depth_tie_epsilon":"1e-9", "maximum_fragment_layers":"32",
    }
    z=np.repeat(depths,3).astype("<f8")
    entries=OrderedDict([
        ("mesh",b"RSVM1\0\0\0"+struct.pack("<IIII",4,4,len(positions),layers)+uv.astype("<f8").tobytes()+faces.tobytes()),
        ("positions",b"RSVP1\0\0\0"+struct.pack("<II",1,len(positions))+positions.astype("<f8").tobytes()),
        ("depths",b"RSVD1\0\0\0"+struct.pack("<II",1,len(positions))+z.tobytes()),
        ("texture",png.getvalue()), ("manifest.txt",("\n".join(k+"="+v for k,v in manifest.items())+"\n").encode()),
    ])
    archive=tmp_path/"depth.rss";write_rss_v2(archive,entries)
    paths=[tmp_path/name for name in ("rgba","prov","view","owner")]
    args=[player,str(archive),"--clip","depth","--view","V0","--frame","0"]
    for flag,path in zip(("--out-rgba","--out-provenance","--out-source-view","--out-owner"),paths):args.extend((flag,str(path)))
    proc=subprocess.run(args,capture_output=True,text=True)
    reference=None if proc.returncode else render_visual_depth(positions=positions,depths=z,faces=faces,uv=uv,texture=tex,resolution=4)
    return proc, paths, reference


def test_native_translucent_overlap_uses_depth_and_survives_face_permutation(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b";a.mkdir();b.mkdir()
    first,paths,ref=render(a,[1.,2.]); second,reverse,ref2=render(b,[1.,2.],order=[1,0])
    assert first.returncode==second.returncode==0,(first.stderr,second.stderr)
    assert paths[0].read_bytes()==reverse[0].read_bytes()==ref.straight_rgba_u8.tobytes()
    assert paths[3].read_bytes()==ref.owner_face_index.astype("<i4").tobytes()
    assert reverse[3].read_bytes()==ref2.owner_face_index.astype("<i4").tobytes()


@pytest.mark.parametrize("depths,reason",[
    ([1.,1.],"UNRESOLVED_TIE"),
    (list(range(1,34)),"FRAGMENT_OVERFLOW"),
    ([-1.],"BEHIND_CAMERA"),
])
def test_native_ambiguous_or_invalid_depth_fails_closed(tmp_path,depths,reason):
    proc,_,_=render(tmp_path,depths)
    assert proc.returncode!=0 and reason in proc.stderr
