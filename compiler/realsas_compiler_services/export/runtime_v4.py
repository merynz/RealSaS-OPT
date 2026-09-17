from __future__ import annotations

"""Runtime-v4 compact package writer: shared canonical assets + bulk arrays."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Iterable, Mapping
import json, os, struct, tempfile, zipfile, zlib
import numpy as np

from compiler.realsas_compiler_core.playback_runtime_v3 import RuntimeV3FrameComposition
from compiler.realsas_compiler_core.playback_runtime_v4 import RuntimeV4PlaybackContract, contract_manifest_v4, validate_playback_runtime_v4_contract, validate_runtime_v4_frame_composition
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload

RUNTIME_V4_BINARY_MAGIC=b"RSRT\x00\x04\x00\x00"
RUNTIME_V4_BINARY_VERSION=4
RUNTIME_V4_BINARY_SCHEMA="RealSaS.RuntimeBinary.rsr.v4"
RUNTIME_V4_PACKAGE_SCHEMA="RealSaS.RuntimePackage.rss.v4"
RUNTIME_V4_COORDINATE_SYSTEM="canonical_3d.shared_asset_plus_orthographic_view_projection"
RUNTIME_V4_FEATURE_FLAGS=(1<<1)|(1<<4)|(1<<5)|(1<<6)|(1<<7)|(1<<10)|(1<<12)
_KIND={"DEFORMABLE_BODY":0,"RIGID_COMPONENT":1,"CLIPPING":2}
_TOPO={"STATIC":0,"ATTACHMENT_DYNAMIC":1,"CLIP_DYNAMIC":2,"DRAW_ORDER_DYNAMIC":3}

@dataclass(frozen=True)
class RuntimeV4Frame:
    time_seconds: float
    posed_xyz_by_asset: Mapping[str,np.ndarray]
    composition_by_view: Mapping[str,RuntimeV3FrameComposition]

@dataclass(frozen=True)
class RuntimeV4Clip:
    clip_id: str; display_name: str; intent: str
    duration_seconds: float; fps: float; loop: bool
    frames: tuple[RuntimeV4Frame,...]
    runtime_qualified: bool=False

class _StreamWriter:
    def __init__(self,f): self.f=f; self.crc=0; self.sha=sha256(); self.bytes=0
    def raw(self,b):
        if not b: return
        self.f.write(b); self.crc=zlib.crc32(b,self.crc); self.sha.update(b); self.bytes+=len(b)
    def u8(self,v): self.raw(struct.pack('<B',int(v)))
    def u32(self,v): self.raw(struct.pack('<I',int(v)))
    def f32(self,v): self.raw(struct.pack('<f',float(v)))
    def string(self,s):
        b=str(s).encode(); self.u32(len(b)); self.raw(b)
    def f32_array(self,a):
        arr=np.asarray(a,dtype='<f4',order='C')
        if not np.isfinite(arr).all(): raise QualificationError('RUNTIME_V4_NONFINITE_FLOAT_ARRAY')
        self.raw(memoryview(arr).cast('B'))
    def u32_array(self,a): self.raw(memoryview(np.asarray(a,dtype='<u4',order='C')).cast('B'))
    def u8_array(self,a): self.raw(memoryview(np.asarray(a,dtype=np.uint8,order='C')).cast('B'))

def _sha_file(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _validate_textures(textures,root,views):
    rows=tuple(textures); by={x.view_id:x for x in rows}
    if set(by)!=set(views) or len(rows)!=len(views): raise QualificationError('RUNTIME_V4_TEXTURE_VIEW_SET_MISMATCH')
    for t in rows:
        p=(root/t.texture_path).resolve()
        try: p.relative_to(root)
        except ValueError as exc: raise QualificationError('RUNTIME_V4_TEXTURE_PATH_ESCAPE') from exc
        if not p.is_file() or _sha_file(p)!=t.texture_sha256: raise QualificationError('RUNTIME_V4_TEXTURE_HASH_MISMATCH')
        raw=p.read_bytes()
        if (zlib.crc32(raw)&0xffffffff)!=int(t.texture_crc32): raise QualificationError('RUNTIME_V4_TEXTURE_CRC_MISMATCH')
    return tuple(by[v] for v in views)

def _validate_clips(contract,clips,views):
    assets={a.asset_id:a for a in contract.assets}; out=tuple(clips)
    if not out: raise QualificationError('RUNTIME_V4_REQUIRES_CLIP')
    for clip in out:
        if not clip.frames or clip.duration_seconds<=0 or clip.fps<=0: raise QualificationError('RUNTIME_V4_CLIP_INVALID')
        for frame in clip.frames:
            if set(frame.posed_xyz_by_asset)!=set(assets): raise QualificationError('RUNTIME_V4_FRAME_ASSET_SET_MISMATCH')
            if set(frame.composition_by_view)!=set(views): raise QualificationError('RUNTIME_V4_FRAME_VIEW_SET_MISMATCH')
            for aid,a in assets.items():
                arr=np.asarray(frame.posed_xyz_by_asset[aid])
                if arr.shape!=(len(a.canonical_xyz),3) or not np.isfinite(arr).all(): raise QualificationError('RUNTIME_V4_FRAME_POSED_XYZ_INVALID')
            for v in views:
                validate_runtime_v4_frame_composition(contract,frame.composition_by_view[v])
                if frame.composition_by_view[v].clip_intervals: raise QualificationError('RUNTIME_V4_INITIAL_DIAGNOSTIC_CLIPPING_FAIL_CLOSED')
    return out

def _write_binary(path,contract,textures,clips,binding_sha,proof_sha,views):
    slot_index={s.slot_id:i for i,s in enumerate(contract.slots)}; asset_index={a.asset_id:i for i,a in enumerate(contract.assets)}
    overlays={(o.view_id,o.asset_id):o for o in contract.overlays}; texture_by={t.view_id:t for t in textures}; cameras={c.view_id:c for c in contract.cameras}
    started=monotonic(); static_bytes=overlay_bytes=deform_total=0
    deform_per_frame=sum(len(a.canonical_xyz)*3*4 for a in contract.assets)
    with Path(path).open('wb') as f:
        w=_StreamWriter(f); w.raw(RUNTIME_V4_BINARY_MAGIC); w.u32(4); w.string(RUNTIME_V4_BINARY_SCHEMA); w.string(binding_sha); w.string(proof_sha); w.string(RUNTIME_V4_COORDINATE_SYSTEM); w.u32(RUNTIME_V4_FEATURE_FLAGS); w.string(contract.contract_hash); w.string(contract.raster.contract_hash); w.f32(contract.visibility.alpha_cutout_threshold)
        w.u32(len(contract.slots))
        for s in contract.slots: w.string(s.slot_id); w.string(s.bone_id); w.u32(s.setup_order); w.string(s.default_attachment_id or '')
        w.u32(len(contract.assets))
        for a in contract.assets:
            xyz=np.ascontiguousarray(a.canonical_xyz,dtype=np.float32); tri=np.ascontiguousarray(a.triangles,dtype=np.uint32)
            w.string(a.asset_id); w.u32(slot_index[a.slot_id]); w.string(a.attachment_id); w.u8(_KIND[a.attachment_kind.value]); w.u8(_TOPO[a.topology_class.value]); w.u8(0); w.u8(0); w.u32(len(xyz)); w.u32(len(tri)); w.f32_array(xyz); w.u32_array(tri); static_bytes+=xyz.nbytes+tri.nbytes
        w.u32(len(views))
        for v in views:
            t=texture_by[v]; c=cameras[v]; w.string(v); w.string(t.texture_path.replace('\\','/')); w.u32(t.texture_crc32); w.u32(t.width); w.u32(t.height); w.f32_array(np.asarray([*c.origin,*c.right,*c.screen_up,*c.forward,c.half_extent],np.float32)); w.u32(c.resolution); w.u32(len(contract.assets))
            for a in contract.assets:
                o=overlays[(v,a.asset_id)]; uv=np.ascontiguousarray(o.uv,dtype=np.float32); prov=np.ascontiguousarray(o.face_provenance,dtype=np.uint8); w.u32(asset_index[a.asset_id]); w.f32_array(uv); w.u8_array(prov); overlay_bytes+=uv.nbytes+prov.nbytes
        w.u32(len(clips))
        for clip in clips:
            w.string(clip.clip_id); w.string(clip.display_name); w.string(clip.intent); w.f32(clip.duration_seconds); w.f32(clip.fps); w.u8(clip.loop); w.u8(clip.runtime_qualified); w.u8(0); w.u8(0); w.u32(len(clip.frames))
            for frame in clip.frames:
                w.f32(frame.time_seconds)
                for a in contract.assets:
                    arr=np.ascontiguousarray(frame.posed_xyz_by_asset[a.asset_id],dtype=np.float32); w.f32_array(arr); deform_total+=arr.nbytes
                for v in views:
                    comp=frame.composition_by_view[v]; w.u32(len(comp.draw_order_slot_ids)); w.u32_array([slot_index[x] for x in comp.draw_order_slot_ids]); w.u32(len(contract.slots))
                    for s in contract.slots: w.string(comp.active_attachment_by_slot[s.slot_id] or '')
                    w.u32(0)
        body_crc=w.crc&0xffffffff; body_sha=w.sha.hexdigest(); body_bytes=w.bytes; f.write(struct.pack('<I',body_crc))
    return {'body_bytes':body_bytes,'binary_bytes':body_bytes+4,'body_crc32':f'{body_crc:08x}','body_sha256':body_sha,'binary_sha256':_sha_file(path),'static_asset_array_bytes':static_bytes,'view_overlay_array_bytes':overlay_bytes,'deform_array_bytes_per_frame':deform_per_frame,'deform_array_bytes_total':deform_total,'write_seconds':monotonic()-started}

def materialize_runtime_v4_archive(*,out_path:str|Path,texture_root:str|Path,contract:RuntimeV4PlaybackContract,textures:Iterable[RuntimeV3TexturePayload],clips:Iterable[RuntimeV4Clip],source_product_state_hash:str,source_proof_bundle_hash:str,required_views:tuple[str,...]=tuple(f'V{i}' for i in range(8)))->dict:
    contract_hash=validate_playback_runtime_v4_contract(contract,required_view_ids=required_views)
    if len(source_product_state_hash)!=64 or len(source_proof_bundle_hash)!=64: raise QualificationError('RUNTIME_V4_SOURCE_IDENTITIES_MUST_BE_SHA256')
    root=Path(texture_root).resolve(); tex=_validate_textures(textures,root,required_views); clip_rows=_validate_clips(contract,clips,required_views)
    contract_bytes=json.dumps(contract_manifest_v4(contract),sort_keys=True,separators=(',',':')).encode(); binding={'schema':'RealSaS.RuntimeV4SourceBinding.v1','source_product_state_hash':source_product_state_hash,'source_proof_bundle_hash':source_proof_bundle_hash,'playback_contract_hash':contract_hash,'reference_raster_contract_hash':contract.raster.contract_hash,'texture_hashes':{t.view_id:t.texture_sha256 for t in tex},'clip_ids':[c.clip_id for c in clip_rows]}; binding_bytes=json.dumps(binding,sort_keys=True,separators=(',',':')).encode(); binding_sha=sha256(binding_bytes).hexdigest(); target=Path(out_path).resolve(); target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='realsas_runtime_v4_',dir=str(target.parent)) as td:
        rsr=Path(td)/'realsas_runtime.rsr'; stats=_write_binary(rsr,contract,tex,clip_rows,binding_sha,source_proof_bundle_hash,required_views); manifest={'schema_version':RUNTIME_V4_PACKAGE_SCHEMA,'binary':{'path':'runtime/realsas_runtime.rsr','schema':RUNTIME_V4_BINARY_SCHEMA,'sha256':stats['binary_sha256']},'source_binding':{**binding,'content_sha256':binding_sha},'playback_contract':{'path':'runtime/playback_contract.json','content_sha256':sha256(contract_bytes).hexdigest()},'representation':{'canonical_geometry_shared_across_views':True,'canonical_posed_xyz_once_per_asset_frame':True,'view_projection_runtime_owned':True,'dense_array_serialization':'CONTIGUOUS_BULK_WRITE'},'measurements':stats,'founder_visual_pass_claimed':False}; tmp=target.with_suffix(target.suffix+'.tmp')
        with zipfile.ZipFile(tmp,'w') as z:
            z.writestr('manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n',compress_type=zipfile.ZIP_DEFLATED,compresslevel=6); z.writestr('source_binding.json',binding_bytes,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6); z.writestr('runtime/playback_contract.json',contract_bytes,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6); z.write(rsr,'runtime/realsas_runtime.rsr',compress_type=zipfile.ZIP_STORED)
            for t in tex: z.write(root/t.texture_path,t.texture_path,compress_type=zipfile.ZIP_STORED)
        os.replace(tmp,target)
    return {'archive_path':str(target),'archive_sha256':_sha_file(target),'runtime_binary_sha256':stats['binary_sha256'],'source_binding_sha256':binding_sha,'playback_contract_hash':contract_hash,'reference_raster_contract_hash':contract.raster.contract_hash,'view_count':len(tex),'clip_count':len(clip_rows),**stats,'package_bytes':target.stat().st_size}
