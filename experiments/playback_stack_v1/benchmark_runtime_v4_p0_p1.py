from __future__ import annotations
import json, tempfile, time, zlib, hashlib, struct, binascii
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.playback_runtime_v3 import AttachmentKind, RuntimeV3FrameComposition, RuntimeV3Slot, TopologyClass
from compiler.realsas_compiler_core.playback_runtime_v4 import RuntimeV4Asset, RuntimeV4Camera, RuntimeV4PlaybackContract, RuntimeV4ViewOverlay
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4 import RuntimeV4Clip, RuntimeV4Frame, materialize_runtime_v4_archive
PNG=b'\x89PNG\r\n\x1a\n'
def ch(k,p): return struct.pack('>I',len(p))+k+p+struct.pack('>I',binascii.crc32(k+p)&0xffffffff)
def png(p):
    raw=b'\x00'+bytes([255,255,255,255]); p.write_bytes(PNG+ch(b'IHDR',struct.pack('>IIBBBBB',1,1,8,6,0,0,0))+ch(b'IDAT',zlib.compress(raw))+ch(b'IEND',b''))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    n=257_505; frames_n=18; views=tuple(f'V{i}' for i in range(8))
    xyz=np.zeros((n,3),np.float32); xyz[:,0]=np.linspace(-1,1,n,dtype=np.float32)
    tri=np.column_stack([np.arange(0,n-2,3,dtype=np.uint32),np.arange(1,n-1,3,dtype=np.uint32),np.arange(2,n,3,dtype=np.uint32)])
    asset=RuntimeV4Asset('body','body_slot','body_attachment',AttachmentKind.DEFORMABLE_BODY,TopologyClass.STATIC,xyz,tri)
    cams=tuple(RuntimeV4Camera(v,i,(0,0,-2),(1,0,0),(0,1,0),(0,0,1),1,1024) for i,v in enumerate(views))
    uv=np.zeros((n,2),np.float32); prov=np.zeros((len(tri),),np.uint8)
    ovs=tuple(RuntimeV4ViewOverlay(v,'body',uv,prov) for v in views)
    c=RuntimeV4PlaybackContract((RuntimeV3Slot('body_slot','root',0,'body_attachment'),),(asset,),cams,ovs)
    comp={v:RuntimeV3FrameComposition(v,('body_slot',),{'body_slot':'body_attachment'}) for v in views}
    frames=tuple(RuntimeV4Frame(i/(frames_n-1),{'body':xyz},comp) for i in range(frames_n)); clip=(RuntimeV4Clip('bench','Bench','bench',1,30,False,frames,False),)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); tex=[]
        for v in views:
            p=root/f'{v}.png'; png(p); b=p.read_bytes(); tex.append(RuntimeV3TexturePayload(v,p.name,sha(p),zlib.crc32(b)&0xffffffff,1,1))
        t=time.perf_counter(); r=materialize_runtime_v4_archive(out_path=root/'bench.rss',texture_root=root,contract=c,textures=tuple(tex),clips=clip,source_product_state_hash='a'*64,source_proof_bundle_hash='b'*64); wall=time.perf_counter()-t
        report={'schema':'RealSaS.RuntimeV4P0P1Benchmark.v1','vertex_count':n,'frame_count':frames_n,'view_count':8,'v3_equivalent_deform_bytes':n*3*4*8*frames_n,'v4_deform_bytes':r['deform_array_bytes_total'],'deform_reduction_ratio':(n*3*4*8*frames_n)/r['deform_array_bytes_total'],'runtime_binary_bytes':r['binary_bytes'],'package_bytes':r['package_bytes'],'writer_seconds':r['write_seconds'],'materialize_seconds':wall,'performance_pass_claimed':False}
        print('RUNTIME_V4_P0_P1_BENCHMARK='+json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
