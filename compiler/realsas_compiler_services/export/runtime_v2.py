from __future__ import annotations

"""Pure RealSaS native runtime-v2 deployment writer.

Consumes only already-qualified/projected runtime views and already-baked frames.
It never invokes models, solvers, deformation, proof mutation, or repair.
"""

from dataclasses import dataclass, asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping
import json
import struct
import zipfile
import zlib

RUNTIME_BINARY_MAGIC = b"RSRT\x00\x02\x00\x00"
RUNTIME_BINARY_VERSION = 2
RUNTIME_BINARY_SCHEMA_VERSION = "realSaS.RuntimeBinary.rsr.v2"
RUNTIME_PACKAGE_SCHEMA_VERSION = "RealSaS.RuntimePackage.current_v4_to_native_v2.v1"
RUNTIME_COORDINATE_SYSTEM = "screen_2d.top_left.x_right.y_down.uv_top_left"
RUNTIME_UNITS_PER_PIXEL = 1.0

RUNTIME_FEATURE_RGBA8_TEXTURES = 1 << 0
RUNTIME_FEATURE_BAKED_MESH_FRAMES = 1 << 1
RUNTIME_FEATURE_LINEAR_FRAME_INTERPOLATION = 1 << 2
RUNTIME_FEATURE_DYNAMIC_DRAW_ORDER = 1 << 3
RUNTIME_FEATURE_MULTI_VIEW = 1 << 4
RUNTIME_FEATURE_SOFTWARE_REFERENCE_RENDERER = 1 << 5
RUNTIME_FEATURE_CLIP_QUALIFICATION = 1 << 6
RUNTIME_FEATURE_TEXTURE_CRC32 = 1 << 7
RUNTIME_FEATURE_REFERENCE_POST_PROCESS = 1 << 8
RUNTIME_FEATURE_CLIP_MIXING = 1 << 9
RUNTIME_FEATURE_FLAGS = sum(1 << i for i in range(10))


@dataclass(frozen=True)
class RuntimeV2Mesh:
    mesh_id: str
    xyuv: tuple[tuple[float, float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class RuntimeV2View:
    view_id: str
    texture_path: str
    texture_sha256: str
    texture_crc32: int
    width: int
    height: int
    meshes: tuple[RuntimeV2Mesh, ...]


@dataclass(frozen=True)
class RuntimeV2Frame:
    time_seconds: float
    mesh_xy: Mapping[str, tuple[tuple[float, float], ...]]
    draw_order_by_view: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class RuntimeV2Clip:
    clip_id: str
    display_name: str
    intent: str
    duration_seconds: float
    fps: float
    loop: bool
    frames: tuple[RuntimeV2Frame, ...]
    runtime_qualified: bool = True
    runtime_envelope_factor: float = 1.0


class _Writer:
    def __init__(self): self.b = BytesIO()
    def raw(self, x: bytes): self.b.write(x)
    def u8(self, x: int): self.raw(struct.pack("<B", int(x)))
    def u32(self, x: int): self.raw(struct.pack("<I", int(x)))
    def f32(self, x: float): self.raw(struct.pack("<f", float(x)))
    def string(self, x: str):
        p = str(x).encode("utf-8"); self.u32(len(p)); self.raw(p)
    def finish(self): return self.b.getvalue()


def _sha_file(path: Path) -> str:
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def _png_size(path: Path) -> tuple[int,int]:
    raw=path.read_bytes()[:24]
    if len(raw)<24 or raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR':
        raise ValueError(f"runtime texture is not PNG:{path}")
    return struct.unpack('>II', raw[16:24])


def _validate(views: tuple[RuntimeV2View,...], clips: tuple[RuntimeV2Clip,...], texture_root: Path):
    if not views: raise ValueError('runtime v2 requires views')
    if not clips: raise ValueError('runtime v2 requires clips')
    view_ids=[v.view_id for v in views]
    if len(set(view_ids))!=len(view_ids): raise ValueError('duplicate runtime view id')
    all_mesh={}
    for v in views:
        p=(texture_root / v.texture_path).resolve()
        if texture_root.resolve() not in p.parents and p != texture_root.resolve(): raise ValueError('texture path escapes root')
        if not p.is_file(): raise ValueError(f'missing texture:{v.texture_path}')
        if _sha_file(p) != v.texture_sha256: raise ValueError(f'texture sha mismatch:{v.view_id}')
        if (zlib.crc32(p.read_bytes()) & 0xffffffff) != int(v.texture_crc32): raise ValueError(f'texture crc mismatch:{v.view_id}')
        if _png_size(p)!=(int(v.width),int(v.height)): raise ValueError(f'texture dimensions mismatch:{v.view_id}')
        mids=[m.mesh_id for m in v.meshes]
        if len(set(mids))!=len(mids) or not mids: raise ValueError(f'invalid mesh ids:{v.view_id}')
        for m in v.meshes:
            if m.mesh_id in all_mesh: raise ValueError('runtime mesh ids must be globally unique')
            all_mesh[m.mesh_id]=(v,m)
            if not m.xyuv: raise ValueError('runtime mesh has no vertices')
            for tri in m.triangles:
                if len(tri)!=3 or min(tri)<0 or max(tri)>=len(m.xyuv): raise ValueError('invalid runtime triangle')
            for x,y,u,w in m.xyuv:
                if not (0.0 <= u <= 1.0 and 0.0 <= w <= 1.0): raise ValueError('runtime uv outside [0,1]')
    for c in clips:
        if not c.frames or c.duration_seconds <= 0 or c.fps <= 0: raise ValueError('invalid runtime clip')
        last=-1.0
        for f in c.frames:
            if f.time_seconds < last: raise ValueError('frame times must be ordered')
            last=f.time_seconds
            for mid,(v,m) in all_mesh.items():
                pts=tuple(f.mesh_xy.get(mid,()))
                if len(pts)!=len(m.xyuv): raise ValueError(f'frame vertex count mismatch:{mid}')
            for vid in view_ids:
                order=tuple(f.draw_order_by_view.get(vid,()))
                allowed={m.mesh_id for m in next(v for v in views if v.view_id==vid).meshes}
                if set(order)!=allowed or len(order)!=len(allowed): raise ValueError(f'draw order is not exact permutation:{vid}')


def write_runtime_v2_binary(*, views: tuple[RuntimeV2View,...], clips: tuple[RuntimeV2Clip,...], source_binding_sha256: str, source_proof_bundle_hash: str) -> bytes:
    if len(source_binding_sha256)!=64 or len(source_proof_bundle_hash)!=64: raise ValueError('runtime source identities must be sha256 hex')
    w=_Writer(); w.raw(RUNTIME_BINARY_MAGIC); w.u32(RUNTIME_BINARY_VERSION)
    w.string(RUNTIME_BINARY_SCHEMA_VERSION); w.string(source_binding_sha256); w.string(source_proof_bundle_hash)
    w.string(RUNTIME_COORDINATE_SYSTEM); w.f32(RUNTIME_UNITS_PER_PIXEL); w.u32(RUNTIME_FEATURE_FLAGS)
    w.u32(len(views))
    for v in views:
        p=Path(v.texture_path)
        w.string(v.view_id); w.string(str(p).replace('\\','/')); w.u32(int(v.texture_crc32)); w.u32(v.width); w.u32(v.height); w.u32(len(v.meshes))
        for m in v.meshes:
            w.string(m.mesh_id); w.u32(len(m.xyuv)); w.u32(len(m.triangles))
            for x,y,u,q in m.xyuv: w.f32(x); w.f32(y); w.f32(u); w.f32(q)
            for a,b,c in m.triangles: w.u32(a); w.u32(b); w.u32(c)
    w.u32(len(clips))
    for c in clips:
        w.string(c.clip_id); w.string(c.display_name); w.string(c.intent); w.f32(c.duration_seconds); w.f32(c.fps)
        w.u8(1 if c.loop else 0); w.u8(1 if c.runtime_qualified else 0); w.u8(0); w.u8(0); w.f32(c.runtime_envelope_factor); w.u32(len(c.frames))
        for f in c.frames:
            w.f32(f.time_seconds)
            for v in views:
                for m in v.meshes:
                    for x,y in f.mesh_xy[m.mesh_id]: w.f32(x); w.f32(y)
                index={m.mesh_id:i for i,m in enumerate(v.meshes)}
                order=f.draw_order_by_view[v.view_id]; w.u32(len(order))
                for mid in order: w.u32(index[mid])
    raw=w.finish(); return raw + struct.pack('<I', zlib.crc32(raw)&0xffffffff)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(str(name).replace("\\", "/"), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100644 & 0xFFFF) << 16
    info.create_system = 3
    return info


def _zip_write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    archive.writestr(_zip_info(name), payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def materialize_runtime_v2_archive(*, out_path: str|Path, texture_root: str|Path, views: Iterable[RuntimeV2View], clips: Iterable[RuntimeV2Clip], source_product_state_hash: str, source_proof_bundle_hash: str, projection_hash: str) -> dict:
    views=tuple(views); clips=tuple(clips); root=Path(texture_root).resolve(); _validate(views,clips,root)
    binding={
        'schema':'RealSaS.RuntimeV2SourceBinding.v1',
        'source_product_state_hash':source_product_state_hash,
        'source_proof_bundle_hash':source_proof_bundle_hash,
        'projection_hash':projection_hash,
        'view_texture_hashes':{v.view_id:v.texture_sha256 for v in views},
        'clip_ids':[c.clip_id for c in clips],
    }
    binding_bytes=json.dumps(binding,sort_keys=True,separators=(',',':')).encode(); binding_sha=sha256(binding_bytes).hexdigest()
    runtime=write_runtime_v2_binary(views=views,clips=clips,source_binding_sha256=binding_sha,source_proof_bundle_hash=source_proof_bundle_hash)
    manifest={
        'schema_version':RUNTIME_PACKAGE_SCHEMA_VERSION,
        'source_binding':{**binding,'content_sha256':binding_sha},
        'runtime_binary':{'path':'runtime/realsas_runtime.rsr','content_sha256':sha256(runtime).hexdigest(),'schema_version':RUNTIME_BINARY_SCHEMA_VERSION},
        'views':[{'view_id':v.view_id,'texture_path':v.texture_path,'texture_sha256':v.texture_sha256,'texture_crc32':f'{int(v.texture_crc32):08x}','width':v.width,'height':v.height,'mesh_ids':[m.mesh_id for m in v.meshes]} for v in views],
        'clips':[{'clip_id':c.clip_id,'duration_seconds':c.duration_seconds,'fps':c.fps,'loop':c.loop,'runtime_qualified':c.runtime_qualified,'frame_count':len(c.frames)} for c in clips],
        'representation_class':'DIRECTIONAL_2D_2P5D_PUPPET',
        'full_3d_reconstruction_authority':False,
        'runtime_export_solver_replay':False,
    }
    target=Path(out_path).resolve(); target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        _zip_write_bytes(z, 'manifest.json', json.dumps(manifest,indent=2,sort_keys=True).encode())
        _zip_write_bytes(z, 'source_binding.json', binding_bytes)
        _zip_write_bytes(z, 'runtime/realsas_runtime.rsr', runtime)
        for v in sorted(views, key=lambda row: row.texture_path):
            _zip_write_bytes(z, v.texture_path, (root / v.texture_path).read_bytes())
    return {'archive_path':str(target),'archive_sha256':_sha_file(target),'runtime_binary_sha256':sha256(runtime).hexdigest(),'source_binding_sha256':binding_sha,'view_count':len(views),'clip_count':len(clips)}
