from hashlib import sha256
from pathlib import Path
import json
import struct
import zipfile
import zlib

from compiler.realsas_compiler_services.export.runtime_v2 import (
    RUNTIME_BINARY_MAGIC,
    RuntimeV2Clip,
    RuntimeV2Frame,
    RuntimeV2Mesh,
    RuntimeV2View,
    materialize_runtime_v2_archive,
)


def _png_rgba(path: Path, width: int = 8, height: int = 8) -> None:
    sig = b"\x89PNG\r\n\x1a\n"
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width): row.extend((255 if x < 4 else 0, 255 if y < 4 else 0, 128, 255))
        rows.append(bytes(row))
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def test_runtime_v2_writer_binds_product_proof_and_never_replays_solver(tmp_path: Path) -> None:
    texture_root = tmp_path / "payload"; (texture_root / "textures").mkdir(parents=True)
    views=[]; frame0={}; frame1={}; orders={}
    for i in range(8):
        rel=f"textures/V{i}.png"; path=texture_root/rel; _png_rgba(path); raw=path.read_bytes(); mesh_id=f"V{i}:body"
        mesh=RuntimeV2Mesh(mesh_id,((1.,1.,.125,.125),(6.,1.,.75,.125),(1.,6.,.125,.75)),((0,1,2),))
        views.append(RuntimeV2View(f"V{i}",rel,sha256(raw).hexdigest(),zlib.crc32(raw)&0xffffffff,8,8,(mesh,)))
        frame0[mesh_id]=((1.,1.),(6.,1.),(1.,6.)); frame1[mesh_id]=((1.25,1.),(6.25,1.),(1.25,6.)); orders[f"V{i}"]=(mesh_id,)
    clip=RuntimeV2Clip("idle","Idle","PRESET",1.,1.,False,(RuntimeV2Frame(0.,frame0,orders),RuntimeV2Frame(1.,frame1,orders)))
    out=tmp_path/"fixture.rss"
    result=materialize_runtime_v2_archive(out_path=out,texture_root=texture_root,views=tuple(views),clips=(clip,),source_product_state_hash="a"*64,source_proof_bundle_hash="b"*64,projection_hash="c"*64)
    assert result["view_count"]==8 and result["clip_count"]==1
    with zipfile.ZipFile(out) as archive:
        manifest=json.loads(archive.read("manifest.json")); binding=json.loads(archive.read("source_binding.json")); binary=archive.read("runtime/realsas_runtime.rsr")
    assert binary.startswith(RUNTIME_BINARY_MAGIC)
    assert manifest["runtime_export_solver_replay"] is False
    assert manifest["source_binding"]["source_product_state_hash"]=="a"*64
    assert manifest["source_binding"]["source_proof_bundle_hash"]=="b"*64
    assert binding["projection_hash"]=="c"*64
    assert manifest["full_3d_reconstruction_authority"] is False
