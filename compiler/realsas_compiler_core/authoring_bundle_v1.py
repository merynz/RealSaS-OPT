from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json=dict[str,Any]

@dataclass(frozen=True)
class EditablePuppetBundleSealIR:
    product_state_binding_hash:str
    skeleton_binding_hash:str
    mesh_binding_hash:str
    mesh_skin_binding_hash:str
    presentation_binding_hash:str
    motion_binding_hash:str
    appearance_binding_hash:str
    runtime_projection_binding_hash:str
    payload_hash:str
    archive_path:str
    archive_sha256:str
    texture_sha256_by_view:tuple[tuple[int,str],...]
    qualification_report:Json
    authoring_bundle_hash:str
    schema_version:str="RealSaS.EditablePuppetBundleSealIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

def editable_puppet_bundle_hash(value:EditablePuppetBundleSealIR)->str:
    payload=value.to_dict(); payload.pop("authoring_bundle_hash",None)
    return content_sha256(payload)

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def _zip_write_bytes(zf,name,data):
    info=zipfile.ZipInfo(str(name),date_time=(1980,1,1,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o100644<<16
    zf.writestr(info,data,compresslevel=1)

def materialize_editable_puppet_bundle_v1(
    *,out_path:Path,state_payload:Mapping[str,Any],skeleton_payload:Mapping[str,Any],
    mesh_payload:Mapping[str,Any],mesh_skin_payload:Mapping[str,Any],
    presentation_payload:Mapping[str,Any],appearance_payload:Mapping[str,Any],
    motion_payload:Mapping[str,Any],projection,
)->EditablePuppetBundleSealIR:
    bindings={
        "product_state":str(state_payload["product_state_hash"]),
        "skeleton":str(skeleton_payload["skeleton_lineage_hash"]),
        "mesh":str(mesh_payload["mesh_lineage_hash"]),
        "mesh_skin":str(mesh_skin_payload["mesh_skin_lineage_hash"]),
        "presentation":str(presentation_payload["presentation_lineage_hash"]),
        "motion":str(motion_payload["motion_lineage_hash"]),
        "appearance":str(appearance_payload["appearance_set_hash"]),
        "projection":str(projection.projection_hash),
    }
    if str(state_payload["skeleton_lineage_hash"])!=bindings["skeleton"]:
        raise QualificationError("AUTHORING_BUNDLE_STATE_SKELETON_DRIFT")
    if str(state_payload["mesh_lineage_hash"])!=bindings["mesh"]:
        raise QualificationError("AUTHORING_BUNDLE_STATE_MESH_DRIFT")
    if str(state_payload["mesh_skin_lineage_hash"])!=bindings["mesh_skin"]:
        raise QualificationError("AUTHORING_BUNDLE_STATE_MESH_SKIN_DRIFT")
    if str(presentation_payload["product_state_binding_hash"])!=bindings["product_state"]:
        raise QualificationError("AUTHORING_BUNDLE_PRESENTATION_STATE_DRIFT")
    if str(motion_payload["product_state_binding_hash"])!=bindings["product_state"]:
        raise QualificationError("AUTHORING_BUNDLE_MOTION_STATE_DRIFT")
    if str(appearance_payload["mesh_binding_hash"])!=bindings["mesh"]:
        raise QualificationError("AUTHORING_BUNDLE_APPEARANCE_MESH_DRIFT")
    if str(projection.product_state_binding_hash)!=bindings["product_state"]:
        raise QualificationError("AUTHORING_BUNDLE_PROJECTION_STATE_DRIFT")

    textures=[]
    for row in sorted(projection.textures,key=lambda x:x.view_index):
        path=Path(row.transport_png_path).resolve()
        if not path.is_file() or _sha256(path)!=row.transport_png_sha256:
            raise QualificationError("AUTHORING_BUNDLE_TEXTURE_DRIFT")
        textures.append((int(row.view_index),str(row.transport_png_sha256),path))

    payload={
        "schema":"RealSaS.EditablePuppetAuthoringPayload.v1",
        "status":"EDITABLE_CANONICAL_AUTHORING_STATE",
        "input_contract":"CONTROLLED_FULL_SUBJECT_8VIEW_V1",
        "canonical_authority":{
            "skeleton":dict(skeleton_payload),
            "mesh":dict(mesh_payload),
            "mesh_skin":dict(mesh_skin_payload),
            "presentation":dict(presentation_payload),
            "appearance":dict(appearance_payload),
            "motion_curves":dict(motion_payload),
            "product_state":dict(state_payload),
        },
        "source_textures":tuple({
            "view_index":vi,"path":f"textures/V{vi}.png","sha256":sha,
            "authority":"SOURCE_RASTER_TRANSPORT",
        } for vi,sha,_ in textures),
        "edit_contract":{
            "bones":"editable; parent/rest edits invalidate Stage25+",
            "mesh":"editable vertices/topology; edits invalidate Stage27+",
            "weights":"editable per-mesh-vertex joint simplex; edits invalidate Stage28+",
            "presentation":"editable groups/slots/order/visibility/tint; edits invalidate Stage30+",
            "appearance":"source-authority changes invalidate Stage31+",
            "motion_curves":"editable Stage34 keyframes; edits invalidate Stage34+",
            "runtime":"Runtime-v4/.rss is derived deployment data, never authoring authority",
            "reseal_required_after_any_edit":True,
        },
        "invalidation_matrix":{
            "SKELETON_EDIT":"25-40","MESH_EDIT":"27-40","WEIGHT_EDIT":"28-40",
            "PRESENTATION_EDIT":"30-40","APPEARANCE_EDIT":"31-40","MOTION_CURVE_EDIT":"34-40",
        },
        "unsupported_from_flattened_input":{
            "recover_original_artist_layer_names_or_semantics":True,
            "infer_physically_separate_translucent_layer_stack":True,
        },
    }
    payload_hash=content_sha256(payload)
    encoded=(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode("utf-8")
    out_path=Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_path,"w") as zf:
        _zip_write_bytes(zf,"puppet.json",encoded)
        for vi,_sha,path in textures:
            _zip_write_bytes(zf,f"textures/V{vi}.png",path.read_bytes())
    value=EditablePuppetBundleSealIR(
        bindings["product_state"],bindings["skeleton"],bindings["mesh"],bindings["mesh_skin"],
        bindings["presentation"],bindings["motion"],bindings["appearance"],bindings["projection"],
        payload_hash,str(out_path),_sha256(out_path),tuple((vi,sha) for vi,sha,_ in textures),
        {
            "status":"PASS_EDITABLE_AUTHORING_BUNDLE","self_contained":True,
            "bones_editable":True,"mesh_editable":True,"weights_editable":True,
            "presentation_editable":True,"motion_curves_editable":True,
            "source_appearance_preserved":True,"runtime_is_derived_not_authority":True,
            "reseal_after_edit_required":True,
        },"",
        metadata={
            "archive_extension":".rsedit",
            "original_artist_layer_semantics_claimed":False,
            "stacked_translucency_recovery_claimed":False,
        },
    )
    return replace(value,authoring_bundle_hash=editable_puppet_bundle_hash(value))

def editable_puppet_bundle_seal_from_dict(payload:Mapping[str,Any])->EditablePuppetBundleSealIR:
    if str(payload.get("schema_version") or payload.get("schema") or "")!="RealSaS.EditablePuppetBundleSealIR.v1":
        raise ValueError("AUTHORING_BUNDLE_SEAL_SCHEMA_MISMATCH")
    value=EditablePuppetBundleSealIR(
        str(payload["product_state_binding_hash"]),str(payload["skeleton_binding_hash"]),
        str(payload["mesh_binding_hash"]),str(payload["mesh_skin_binding_hash"]),
        str(payload["presentation_binding_hash"]),str(payload["motion_binding_hash"]),
        str(payload["appearance_binding_hash"]),str(payload["runtime_projection_binding_hash"]),
        str(payload["payload_hash"]),str(payload["archive_path"]),str(payload["archive_sha256"]),
        tuple((int(a),str(b)) for a,b in payload.get("texture_sha256_by_view") or ()),
        dict(payload.get("qualification_report") or {}),str(payload["authoring_bundle_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.EditablePuppetBundleSealIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.authoring_bundle_hash!=editable_puppet_bundle_hash(value):
        raise ValueError("AUTHORING_BUNDLE_SEAL_HASH_MISMATCH")
    return value
