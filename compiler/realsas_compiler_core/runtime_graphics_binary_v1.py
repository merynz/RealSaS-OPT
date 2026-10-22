from __future__ import annotations
from collections import OrderedDict
import hashlib,struct
from .graphics_runtime_v1 import RuntimeGraphicsV1,runtime_graphics_hash_v1
from .runtime_package_v2 import build_rss_v2_entries
from .types import QualificationError

MAGIC=b"RSGFX001"

def _u32(out,value): out.extend(struct.pack("<I",int(value)))
def _f64(out,value): out.extend(struct.pack("<d",float(value)))
def _text(out,value):
    raw=str(value).encode("utf-8");_u32(out,len(raw));out.extend(raw)

def encode_runtime_graphics_v1(value:RuntimeGraphicsV1)->bytes:
    if value.runtime_graphics_hash!=runtime_graphics_hash_v1(value): raise QualificationError("RUNTIME_GRAPHICS_HASH_DRIFT")
    out=bytearray(MAGIC);_u32(out,1);_u32(out,len(value.face_fragment_index));_u32(out,len(value.fragment_ids));_u32(out,len(value.views));_u32(out,len(value.attachments));_text(out,value.canonical_mesh_binding_hash);_text(out,value.compiled_graphics_binding_hash);_text(out,value.runtime_graphics_hash)
    for x in value.face_fragment_index:_u32(out,x)
    for fid,support in zip(value.fragment_ids,value.fragment_support_band_pixels):_text(out,fid);_u32(out,support)
    for view in sorted(value.views,key=lambda r:r.view_index):
        _u32(out,view.view_index);_u32(out,len(view.visible_fragment_indices));
        for x in view.visible_fragment_indices:_u32(out,x)
        _u32(out,len(view.topological_fragment_order));
        for x in view.topological_fragment_order:_u32(out,x)
        _u32(out,len(view.order_edges));
        for a,b in view.order_edges:_u32(out,a);_u32(out,b)
    for row in value.attachments:
        _u32(out,row["fragment_index"]);_text(out,row["parent_mechanical_domain_id"]);_u32(out,row["anchor_face_index"])
        bary=tuple(row["barycentric_anchor"])
        if len(bary)!=3:raise QualificationError("RUNTIME_GRAPHICS_ATTACHMENT_BARY_INVALID")
        for x in bary:_f64(out,x)
        _f64(out,row["confidence"])
    return bytes(out)

def build_rss_vnext_graphics_entries_v1(projection,runtime_graphics:RuntimeGraphicsV1)->OrderedDict[str,bytes]:
    """Extend the existing RSS transport while making Graphics CONTENT mandatory."""
    if projection.mesh_binding_hash!=runtime_graphics.canonical_mesh_binding_hash:raise QualificationError("RSS_GRAPHICS_MESH_BINDING_DRIFT")
    entries=build_rss_v2_entries(projection);payload=encode_runtime_graphics_v1(runtime_graphics);manifest=entries.pop("manifest.txt").decode("utf-8").splitlines();repl={"presentation_state_execution_authorized":"1","tint_order_visibility_authorized":"1"};manifest=[f"{k}={repl[k]}" if (k:=row.split("=",1)[0]) in repl else row for row in manifest];manifest.extend(["graphics_state_execution_authorized=1","graphics_state_contract=ROLE_FREE_CANONICAL_M_GRAPHICS_V1","graphics_entry=graphics.bin",f"graphics_sha256={hashlib.sha256(payload).hexdigest()}",f"graphics_binding_hash={runtime_graphics.runtime_graphics_hash}","graphics_face_ownership_required=1","graphics_partial_order_required=1","triangle_array_order_as_presentation_authority=0","independent_visual_mesh_authorized=0"]);entries["graphics.bin"]=payload;entries["manifest.txt"]=("\n".join(manifest)+"\n").encode("utf-8");return entries
