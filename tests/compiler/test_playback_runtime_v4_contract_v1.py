from pathlib import Path
import binascii, hashlib, struct, zlib
import numpy as np
import pytest

from compiler.realsas_compiler_core.playback_runtime_v3 import AttachmentKind, RuntimeV3FrameComposition, RuntimeV3Slot, TopologyClass
from compiler.realsas_compiler_core.playback_runtime_v4 import RuntimeV4Asset, RuntimeV4Camera, RuntimeV4PlaybackContract, RuntimeV4ViewOverlay, validate_playback_runtime_v4_contract
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4 import RuntimeV4Clip, RuntimeV4Frame, materialize_runtime_v4_archive
from compiler.realsas_compiler_core.types import QualificationError

PNG_MAGIC=b'\x89PNG\r\n\x1a\n'
def _chunk(k,p): return struct.pack('>I',len(p))+k+p+struct.pack('>I',binascii.crc32(k+p)&0xffffffff)
def _png(path):
    raw=b''.join([b'\x00'+bytes([255,255,255,255])*2 for _ in range(2)])
    path.write_bytes(PNG_MAGIC+_chunk(b'IHDR',struct.pack('>IIBBBBB',2,2,8,6,0,0,0))+_chunk(b'IDAT',zlib.compress(raw))+_chunk(b'IEND',b''))
def _sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def _fixture(root: Path):
    views=tuple(f'V{i}' for i in range(8)); xyz=np.array([[-1,1,1],[1,1,1],[-1,-1,1]],np.float32); tri=np.array([[0,1,2]],np.uint32)
    asset=RuntimeV4Asset('body_asset','body_slot','body_attachment',AttachmentKind.DEFORMABLE_BODY,TopologyClass.STATIC,xyz,tri)
    cams=tuple(RuntimeV4Camera(v,i,(0,0,-1),(1,0,0),(0,1,0),(0,0,1),1.0,2) for i,v in enumerate(views))
    overlays=tuple(RuntimeV4ViewOverlay(v,'body_asset',np.array([[0,0],[1,0],[0,1]],np.float32),np.array([0],np.uint8)) for v in views)
    contract=RuntimeV4PlaybackContract((RuntimeV3Slot('body_slot','root',0,'body_attachment'),),(asset,),cams,overlays)
    comp={v:RuntimeV3FrameComposition(v,('body_slot',),{'body_slot':'body_attachment'}) for v in views}
    clips=(RuntimeV4Clip('idle','Idle','idle',1.0,30.0,False,(RuntimeV4Frame(0.0,{'body_asset':xyz},comp),RuntimeV4Frame(1.0,{'body_asset':xyz+np.array([0,0,.1],np.float32)},comp)),True),)
    tex=[]
    for v in views:
        p=root/f'{v}.png'; _png(p); raw=p.read_bytes(); tex.append(RuntimeV3TexturePayload(v,p.name,_sha(p),zlib.crc32(raw)&0xffffffff,2,2))
    return contract,tuple(tex),clips

def test_v4_contract_shared_geometry_hash_is_deterministic(tmp_path):
    c,_,_=_fixture(tmp_path); h1=validate_playback_runtime_v4_contract(c); h2=validate_playback_runtime_v4_contract(c); assert h1==h2 and len(h1)==64

def test_v4_rejects_missing_overlay(tmp_path):
    c,_,_=_fixture(tmp_path); bad=RuntimeV4PlaybackContract(c.slots,c.assets,c.cameras,c.overlays[:-1])
    with pytest.raises(QualificationError,match='OVERLAY_VIEW_ASSET_COVERAGE_INCOMPLETE'): validate_playback_runtime_v4_contract(bad)

def test_v4_rejects_completion_when_disabled(tmp_path):
    c,_,_=_fixture(tmp_path); o=c.overlays[0]; bad_o=RuntimeV4ViewOverlay(o.view_id,o.asset_id,o.uv,np.array([4],np.uint8)); bad=RuntimeV4PlaybackContract(c.slots,c.assets,c.cameras,(bad_o,)+c.overlays[1:])
    with pytest.raises(QualificationError,match='COMPLETION_NOT_ALLOWED'): validate_playback_runtime_v4_contract(bad)

def test_v4_writer_reports_shared_frame_payload(tmp_path):
    c,t,clips=_fixture(tmp_path); out=materialize_runtime_v4_archive(out_path=tmp_path/'x.rss',texture_root=tmp_path,contract=c,textures=t,clips=clips,source_product_state_hash='a'*64,source_proof_bundle_hash='b'*64)
    assert out['deform_array_bytes_per_frame']==3*3*4
    assert out['deform_array_bytes_total']==2*3*3*4
    assert out['view_count']==8 and out['clip_count']==1
    assert Path(out['archive_path']).is_file()

def test_v4_p0_arithmetic_removes_view_multiplier():
    n=257_505; frame=n*3*4; assert frame*8==24_720_480 and frame==3_090_060
