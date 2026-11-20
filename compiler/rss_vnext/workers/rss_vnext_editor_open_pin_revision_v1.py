from __future__ import annotations

"""Persisted, editor-authored OPEN pin approval in an existing .rsedit archive.

An approval is an additional *user-authored* revision over one immutable
canonical Mesh/Rig/Skin triplet. It does not change the base authoring revision
ID, claim source-native pin witnesses, change the inferred Graphics binding,
or automatically run XPBD. This module replays the complete structural
acceptance court when the archive is reopened. Hashes are integrity checks,
not authorization, identity verification or source observations.
"""

from collections import OrderedDict
from dataclasses import asdict, fields
from pathlib import Path
import json
from hashlib import sha256

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    mint_canonical_mechanics_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanics_editor_open_pin_approval_v1 import (
    EditorOpenPinNeuralReviewV1,
    EditorOpenPinAuthoringCommandV1,
    prepare_editor_approved_open_pin_domain_v1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes, read_rss_v2,
)
from .rss_vnext_canonical_authoring_revision_v1 import (
    MAX_BYTES, REQUIRED, reopen_rss_vnext_canonical_user_edit_revision_v1,
)

ENTRY = "editor_open_pin.json"
SCHEMA = "RealSaS.RSSvNext.EditorOpenPinUserRevision.v1"
AUTHORITY = "USER_AUTHORED_PIN_NOT_NATIVE_SOURCE_OR_RELEASE"


def _sha(blob:bytes)->str:
    return sha256(blob).hexdigest()


def _json(value:object)->bytes:
    return json.dumps(
        value,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False,
    ).encode("utf-8")


def _manifest(entries:dict[str,bytes])->dict[str,str]:
    result={}
    for line in entries["manifest.txt"].decode("utf-8").splitlines():
        if "=" not in line:
            raise RuntimeError("RSS_PIN_REVISION_MANIFEST_INVALID")
        key,value=line.split("=",1)
        if key in result:
            raise RuntimeError("RSS_PIN_REVISION_MANIFEST_DUPLICATE")
        result[key]=value
    return result


def _typed_dataclass(cls,payload:object,tuple_fields:tuple[str,...]):
    if type(payload) is not dict or set(payload)!={
        row.name for row in fields(cls)
    }:
        raise RuntimeError("RSS_PIN_REVISION_TYPED_PAYLOAD_INVALID")
    content=dict(payload)
    for key in tuple_fields:
        if type(content[key]) is not list:
            raise RuntimeError("RSS_PIN_REVISION_TYPED_PAYLOAD_INVALID")
        content[key]=tuple(content[key])
    return cls(**content)


def _receipt_dict(*,review,command,prepared,authoring_revision_id:str):
    return {
        "schema":SCHEMA,
        "authority":AUTHORITY,
        "authoring_revision_id":authoring_revision_id,
        # Canonical JSON converts dataclass tuples to JSON arrays before
        # in-memory validation; reopen sees the same wire representation.
        "review":json.loads(_json(asdict(review))),
        "command":json.loads(_json(asdict(command))),
        "prepared_domain_hash":prepared.domain.domain_hash,
        "prepared_canonical_open_hem_sha256":prepared.canonical_open_hem_sha256,
        "prepared_qualification_receipt_sha256":prepared.approval_receipt_sha256,
        "source_pin_qualified":False,
        "automatic_xpbd_activation":False,
        "product_release_qualified":False,
        "source_material_truth_claimed":False,
        "graphics_uv_binding_qualified":False,
    }


def validate_persisted_editor_open_pin_record_v1(
    *,payload:object,
    authoring_revision_id:str,mesh,rig,skin,
)->dict:
    """Recompute the REAL Canonical Mechanics/XPBD domain from saved addresses.

    No cached domain is trusted. It must survive a fresh Compiler executable
    mechanics mint and an independent full OPEN surface qualification.
    """
    if type(payload) is not dict or set(payload)!={
        "schema","authority","authoring_revision_id","review","command",
        "prepared_domain_hash","prepared_canonical_open_hem_sha256",
        "prepared_qualification_receipt_sha256",
        "source_pin_qualified","automatic_xpbd_activation",
        "product_release_qualified","source_material_truth_claimed",
        "graphics_uv_binding_qualified","pin_revision_id",
    }:
        raise RuntimeError("RSS_PIN_REVISION_ENTRY_SCHEMA_DRIFT")
    if (
        payload["schema"]!=SCHEMA
        or payload["authority"]!=AUTHORITY
        or payload["authoring_revision_id"]!=authoring_revision_id
        or any(payload[k] is not False for k in (
            "source_pin_qualified","automatic_xpbd_activation",
            "product_release_qualified","source_material_truth_claimed",
            "graphics_uv_binding_qualified",
        ))
        or payload["pin_revision_id"]!=content_sha256({
            k:v for k,v in payload.items() if k!="pin_revision_id"
        })
    ):
        raise RuntimeError("RSS_PIN_REVISION_AUTHORITY_OR_PARENT_DRIFT")
    review=_typed_dataclass(
        EditorOpenPinNeuralReviewV1,payload["review"],
        ("held_canonical_face","holder_canonical_face",
         "candidate_pin_vertex_ids"),
    )
    command=_typed_dataclass(
        EditorOpenPinAuthoringCommandV1,payload["command"],
        ("approved_pin_vertex_ids",),
    )
    if command.authoring_revision_id!=authoring_revision_id:
        raise RuntimeError("RSS_PIN_REVISION_EDITOR_COMMAND_PARENT_DRIFT")
    try:
        mechanics=mint_canonical_mechanics_v1(mesh,rig,skin)
        prepared=prepare_editor_approved_open_pin_domain_v1(
            mesh=mesh,rig=rig,skin=skin,mechanics=mechanics,
            review=review,command=command,
        )
    except (ValueError,RuntimeError) as exc:
        raise RuntimeError("RSS_PIN_REVISION_INDEPENDENT_MECHANICS_REPLAY_REJECTED") from exc
    if (
        prepared.domain.domain_hash!=payload["prepared_domain_hash"]
        or prepared.canonical_open_hem_sha256
            !=payload["prepared_canonical_open_hem_sha256"]
        or prepared.approval_receipt_sha256
            !=payload["prepared_qualification_receipt_sha256"]
    ):
        raise RuntimeError("RSS_PIN_REVISION_REPLAY_DOMAIN_DRIFT")
    return {
        "review":review,"command":command,"prepared":prepared,
        "pin_revision_id":payload["pin_revision_id"],
        "automatic_xpbd_activation":False,
        "source_pin_qualified":False,
        "product_release_qualified":False,
    }


def seal_editor_approved_open_pin_rsedit_v1(
    *,source_rss_path:Path,revision_path:Path,
    review:EditorOpenPinNeuralReviewV1,
    command:EditorOpenPinAuthoringCommandV1,
)->tuple[bytes,dict]:
    """Add one approved pin action to the EXACT existing .rsedit revision.

    The original Mesh/Rig/Skin/pose bytes remain unchanged. The returned bytes
    can be written atomically as a new .rsedit file by the caller.
    """
    original=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=revision_path,source_rss_path=source_rss_path,
    )
    if original.get("editor_open_pin") is not None:
        raise ValueError("RSS_PIN_REVISION_ALREADY_SEALED")
    if (
        type(review) is not EditorOpenPinNeuralReviewV1
        or type(command) is not EditorOpenPinAuthoringCommandV1
        or command.authoring_revision_id!=original["authoring_revision_id"]
    ):
        raise ValueError("RSS_PIN_REVISION_EXPLICIT_REVIEW_AND_COMMAND_REQUIRED")
    entries=read_rss_v2(revision_path)
    if set(entries)!=REQUIRED:
        raise RuntimeError("RSS_PIN_REVISION_UNEXPECTED_BASE_ENTRY_SET")
    mechanics=mint_canonical_mechanics_v1(
        original["mesh"],original["rig"],original["skin"],
    )
    prepared=prepare_editor_approved_open_pin_domain_v1(
        mesh=original["mesh"],rig=original["rig"],skin=original["skin"],
        mechanics=mechanics,review=review,command=command,
    )
    row=_receipt_dict(
        review=review,command=command,prepared=prepared,
        authoring_revision_id=original["authoring_revision_id"],
    )
    row["pin_revision_id"]=content_sha256(row)
    validated=validate_persisted_editor_open_pin_record_v1(
        payload=row,authoring_revision_id=original["authoring_revision_id"],
        mesh=original["mesh"],rig=original["rig"],skin=original["skin"],
    )
    if validated["pin_revision_id"]!=row["pin_revision_id"]:
        raise RuntimeError("RSS_PIN_REVISION_VALIDATION_DRIFT")
    manifest=_manifest(entries)
    manifest["sha256:"+ENTRY]=_sha(_json(row))
    manifest["pin_revision_id"]=row["pin_revision_id"]
    entries[ENTRY]=_json(row)
    entries["manifest.txt"]=(
        "\n".join(f"{k}={v}" for k,v in sorted(manifest.items()))+"\n"
    ).encode("utf-8")
    packed=_entry_bytes(OrderedDict((k,entries[k]) for k in (
        "mesh.json","rig.json","skin.json","pose.bin","revision.json",
        ENTRY,"manifest.txt",
    )))
    if len(packed)>MAX_BYTES:
        raise ValueError("RSS_PIN_REVISION_ARCHIVE_BUDGET")
    return packed,row


__all__=[
    "ENTRY","SCHEMA","AUTHORITY",
    "seal_editor_approved_open_pin_rsedit_v1",
    "validate_persisted_editor_open_pin_record_v1",
]
