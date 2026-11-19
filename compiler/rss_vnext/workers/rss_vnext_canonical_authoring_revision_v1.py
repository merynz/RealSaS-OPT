from __future__ import annotations

"""Canonical M*/AXIS/MIRA authoring edits with immutable source .rss identity.

This is an editable *user-authored* .rsedit revision, not a reminted source
teacher, model release, or ProductRevision. It preserves stable original
canonical entity addresses, recomputes the actual edited Mesh/Rig/Skin
content hashes and Mechanics LBS frame bytes, and independently verifies
everything when reopening against the exact immutable base .rss.

The revision deliberately never reuses the base native Graphics/UV bindings
as if edited geometry had source-certified RGB. A downstream presentation
rebind is required to render this revision. This file does not silently
rewrite the source .rss, infer new material truth, or invent a second mesh.
"""

from collections import OrderedDict
from dataclasses import asdict,replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,canonical_mesh_from_dict_v1,
    canonical_mesh_lineage_hash_v1,validate_canonical_mesh_domain_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalRigIRV2,_hash_deform_structure,_hash_control_program,
    canonical_rig_from_dict_v2,canonical_rig_lineage_hash_v2,
    validate_canonical_rig_domain_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2,canonical_skin_from_dict_v2,
    canonical_skin_lineage_hash_v2,validate_canonical_skin_domain_v2,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes,read_rss_v2,
)
from .rss_vnext_native_preview_package_v1 import (
    _make_pose_binary,reopen_rss_vnext_native_preview_package_v1,
)

SCHEMA="RealSaS.RSSvNext.CanonicalUserAuthoringRevision.v1"
AUTHORITY="USER_EDITED_CANONICAL_TRIPLET_NOT_SOURCE_PRODUCT_OR_GRAPHICS_PROOF"
REQUIRED=frozenset((
    "manifest.txt","mesh.json","rig.json","skin.json",
    "pose.bin","revision.json",
))
MAX_BYTES=64*1024*1024


def _json(value) -> bytes:
    return json.dumps(
        value,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False,
    ).encode("utf-8")


def _sha(blob: bytes) -> str:
    return sha256(blob).hexdigest()


def _coordinates(value, label: str) -> tuple[float,float,float]:
    if (
        not isinstance(value,(list,tuple)) or len(value)!=3
        or any(type(x) not in (int,float) or not math.isfinite(x)
               for x in value)
    ):
        raise ValueError(label+"_INVALID")
    return tuple(float(x) for x in value)


def _edited_mesh(
    mesh:CanonicalMeshIR, moves:Mapping[str,tuple[float,float,float]],
    base_sha:str,
) -> CanonicalMeshIR:
    known={v.canonical_mesh_vertex_id for v in mesh.vertices}
    if not set(moves).issubset(known):
        raise ValueError("RSS_AUTHORING_EDIT_UNKNOWN_MESH_VERTEX")
    vertices=tuple(
        replace(v,P=_coordinates(moves[v.canonical_mesh_vertex_id],
                                "RSS_AUTHORING_EDIT_MESH_POSITION"))
        if v.canonical_mesh_vertex_id in moves else v
        for v in mesh.vertices
    )
    pos={v.canonical_mesh_vertex_id:np.asarray(v.P,dtype=np.float64)
         for v in vertices}
    if any(
        float(np.linalg.norm(np.cross(
            pos[b]-pos[a],pos[c]-pos[a],
        ))) <= 1e-10
        for a,b,c in mesh.faces
    ):
        raise ValueError("RSS_AUTHORING_EDIT_DEGENERATE_TRIANGLE")
    result=replace(
        mesh,vertices=vertices,mesh_lineage_hash="",
        metadata={
            **mesh.metadata,
            "authoring_revision_schema":SCHEMA,
            "user_authored_geometry_not_source_qualified":True,
            "original_canonical_mesh_lineage_hash":mesh.mesh_lineage_hash,
            "original_rss_sha256":base_sha,
        },
    )
    result=replace(result,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(result))
    validate_canonical_mesh_domain_v1(result)
    return result


def _edited_rig(
    rig:CanonicalRigIRV2, mesh:CanonicalMeshIR,
    moves:Mapping[str,tuple[float,float,float]],
    base_sha:str,
) -> CanonicalRigIRV2:
    known={j.canonical_joint_id for j in rig.joints}
    if not set(moves).issubset(known):
        raise ValueError("RSS_AUTHORING_EDIT_UNKNOWN_RIG_JOINT")
    joints=tuple(
        replace(j,P=_coordinates(moves[j.canonical_joint_id],
                                "RSS_AUTHORING_EDIT_JOINT_REST"))
        if j.canonical_joint_id in moves else j
        for j in rig.joints
    )
    positions={j.canonical_joint_id:np.asarray(j.P,dtype=np.float64)
               for j in joints}
    if any(
        j.parent_canonical_joint_id is not None
        and float(np.linalg.norm(
            positions[j.canonical_joint_id]
            -positions[j.parent_canonical_joint_id],
        ))<=1e-9
        for j in joints
    ):
        raise ValueError("RSS_AUTHORING_EDIT_COLLAPSED_BONE")
    result=replace(
        rig,joints=joints,mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash="",control_program_hash="",rig_lineage_hash="",
        metadata={
            **rig.metadata,
            "authoring_revision_schema":SCHEMA,
            "user_authored_rig_rest_not_source_qualified":True,
            "original_canonical_rig_lineage_hash":rig.rig_lineage_hash,
            "original_rss_sha256":base_sha,
        },
    )
    result=replace(result,deform_structure_hash=_hash_deform_structure(result))
    result=replace(result,control_program_hash=_hash_control_program(result))
    result=replace(result,rig_lineage_hash=canonical_rig_lineage_hash_v2(result))
    validate_canonical_rig_domain_v2(result)
    return result


def _edited_skin(
    skin:CanonicalSkinIRV2, mesh:CanonicalMeshIR,rig:CanonicalRigIRV2,
    weights:Mapping[str,Mapping[str,float]],base_sha:str,
) -> CanonicalSkinIRV2:
    known={row.canonical_mesh_vertex_id for row in skin.rows}
    valid_joints={j.canonical_joint_id for j in rig.joints}
    if not set(weights).issubset(known):
        raise ValueError("RSS_AUTHORING_EDIT_UNKNOWN_SKIN_VERTEX")
    overrides={}
    for vid,w in weights.items():
        if (
            not isinstance(w,Mapping) or not 1<=len(w)<=8
            or any(
                jid not in valid_joints or type(v) not in (float,int)
                or not math.isfinite(v) or v<0
                for jid,v in w.items()
            )
            or not math.isclose(
                sum(float(value) for value in w.values()),
                1.,rel_tol=0.,abs_tol=1e-6,
            )
        ):
            raise ValueError("RSS_AUTHORING_EDIT_WEIGHT_SIMPLEX_OR_JOINT")
        overrides[vid]=tuple(sorted((jid,float(v))
                                   for jid,v in w.items() if v>0))
        if not overrides[vid]:
            raise ValueError("RSS_AUTHORING_EDIT_WEIGHT_ZERO")
    rows=tuple(
        replace(row,influences=overrides[row.canonical_mesh_vertex_id])
        if row.canonical_mesh_vertex_id in overrides else row
        for row in skin.rows
    )
    result=replace(
        skin,rows=rows,mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        skin_lineage_hash="",
        metadata={
            **skin.metadata,
            "authoring_revision_schema":SCHEMA,
            "user_authored_skin_weights_not_source_qualified":True,
            "original_canonical_skin_lineage_hash":skin.skin_lineage_hash,
            "original_rss_sha256":base_sha,
        },
    )
    result=replace(result,skin_lineage_hash=canonical_skin_lineage_hash_v2(result))
    validate_canonical_skin_domain_v2(result)
    return result


def compile_rss_vnext_canonical_user_edit_revision_v1(
    *,source_rss_path:Path,
    mesh_vertex_position_edits:Mapping[str,tuple[float,float,float]]|None=None,
    joint_rest_position_edits:Mapping[str,tuple[float,float,float]]|None=None,
    skin_weight_edits:Mapping[str,Mapping[str,float]]|None=None,
) -> tuple[bytes,dict]:
    base=reopen_rss_vnext_native_preview_package_v1(source_rss_path)
    data=source_rss_path.read_bytes()
    base_sha=_sha(data)
    mesh_edits=dict(mesh_vertex_position_edits or {})
    rest_edits=dict(joint_rest_position_edits or {})
    weight_edits=dict(skin_weight_edits or {})
    if not (mesh_edits or rest_edits or weight_edits):
        raise ValueError("RSS_AUTHORING_REVISION_NO_EDIT")
    mesh=_edited_mesh(base["mesh"],mesh_edits,base_sha)
    rig=_edited_rig(base["rig"],mesh,rest_edits,base_sha)
    skin=_edited_skin(base["skin"],mesh,rig,weight_edits,base_sha)
    if (
        rig.mesh_lineage_hash!=mesh.mesh_lineage_hash
        or skin.mesh_lineage_hash!=mesh.mesh_lineage_hash
        or skin.deform_structure_hash!=rig.deform_structure_hash
        or tuple(x.canonical_mesh_vertex_id for x in mesh.vertices)!=
            tuple(x.canonical_mesh_vertex_id for x in base["mesh"].vertices)
    ):
        raise RuntimeError("RSS_AUTHORING_REVISION_CANONICAL_PARENT_DRIFT")
    pose,pose_hashes=_make_pose_binary(
        mesh,rig,skin,base["control_frames"],base["virtual_camera"],
    )
    edit_receipt={
        "schema":SCHEMA,
        "authority":AUTHORITY,
        "base_rss_sha256":base_sha,
        "base_mesh_lineage_hash":base["mesh"].mesh_lineage_hash,
        "base_rig_lineage_hash":base["rig"].rig_lineage_hash,
        "base_skin_lineage_hash":base["skin"].skin_lineage_hash,
        "edited_mesh_lineage_hash":mesh.mesh_lineage_hash,
        "edited_rig_lineage_hash":rig.rig_lineage_hash,
        "edited_skin_lineage_hash":skin.skin_lineage_hash,
        "edit_scope":{
            "mesh_positions":sorted(mesh_edits),
            "joint_rest_positions":sorted(rest_edits),
            "skin_weights":sorted(weight_edits),
        },
        "mechanics_pose_hashes":pose_hashes,
        "original_canonical_vertex_and_joint_ids_preserved":True,
        "original_canonical_face_topology_preserved":True,
        "graphics_uv_require_independent_rebind":True,
        "source_geometry_qualification_reused":False,
        "source_rig_qualification_reused":False,
        "source_skin_qualification_reused":False,
        "source_appearance_qualified":False,
        "source_camera_observed":False,
        "product_revision_minted":False,
        "product_release_admission":False,
    }
    edit_receipt["authoring_revision_id"]=content_sha256(edit_receipt)
    entries=OrderedDict([
        ("mesh.json",_json(mesh.to_dict())),
        ("rig.json",_json(rig.to_dict())),
        ("skin.json",_json(skin.to_dict())),
        ("pose.bin",pose),
        ("revision.json",_json(edit_receipt)),
    ])
    manifest={
        "schema":SCHEMA,
        "authority":AUTHORITY,
        "base_rss_sha256":base_sha,
        "authoring_revision_id":edit_receipt["authoring_revision_id"],
        "product_revision_minted":"0",
        "graphics_uv_binding_qualified":"0",
    }
    for name,blob in entries.items():
        manifest["sha256:"+name]=_sha(blob)
    entries["manifest.txt"]=(
        "\n".join(f"{k}={v}" for k,v in sorted(manifest.items()))+"\n"
    ).encode()
    packed=_entry_bytes(entries)
    if len(packed)>MAX_BYTES:
        raise ValueError("RSS_AUTHORING_REVISION_ARCHIVE_BUDGET")
    return packed,edit_receipt


def reopen_rss_vnext_canonical_user_edit_revision_v1(
    *,revision_path:Path,source_rss_path:Path,
) -> dict:
    if revision_path.stat().st_size>MAX_BYTES:
        raise RuntimeError("RSS_AUTHORING_REOPEN_ARCHIVE_BUDGET")
    entries=read_rss_v2(revision_path)
    if set(entries)!=REQUIRED:
        raise RuntimeError("RSS_AUTHORING_REOPEN_ENTRY_SET")
    try:
        manifest=dict(row.split("=",1)
                      for row in entries["manifest.txt"].decode().splitlines())
        receipt=json.loads(entries["revision.json"])
        mesh=canonical_mesh_from_dict_v1(json.loads(entries["mesh.json"]))
        rig=canonical_rig_from_dict_v2(json.loads(entries["rig.json"]))
        skin=canonical_skin_from_dict_v2(json.loads(entries["skin.json"]))
    except (ValueError,KeyError,UnicodeDecodeError,TypeError) as exc:
        raise RuntimeError("RSS_AUTHORING_REOPEN_TYPED_ARCHIVE_INVALID") from exc
    base=reopen_rss_vnext_native_preview_package_v1(source_rss_path)
    base_sha=_sha(source_rss_path.read_bytes())
    if (
        manifest.get("schema")!=SCHEMA
        or manifest.get("authority")!=AUTHORITY
        or manifest.get("product_revision_minted")!="0"
        or manifest.get("graphics_uv_binding_qualified")!="0"
        or receipt.get("schema")!=SCHEMA
        or receipt.get("authority")!=AUTHORITY
        or receipt.get("base_rss_sha256")!=base_sha
        or manifest.get("base_rss_sha256")!=base_sha
        or receipt.get("base_mesh_lineage_hash")
            !=base["mesh"].mesh_lineage_hash
        or receipt.get("base_rig_lineage_hash")
            !=base["rig"].rig_lineage_hash
        or receipt.get("base_skin_lineage_hash")
            !=base["skin"].skin_lineage_hash
        or any(
            receipt.get(key) is not flag for key,flag in (
                ("original_canonical_vertex_and_joint_ids_preserved",True),
                ("original_canonical_face_topology_preserved",True),
                ("graphics_uv_require_independent_rebind",True),
                ("source_geometry_qualification_reused",False),
                ("source_rig_qualification_reused",False),
                ("source_skin_qualification_reused",False),
                ("source_appearance_qualified",False),
                ("source_camera_observed",False),
                ("product_revision_minted",False),
                ("product_release_admission",False),
            )
        )
        or any(
            manifest.get("sha256:"+name)!=_sha(entries[name])
            for name in REQUIRED-{"manifest.txt"}
        )
        or receipt.get("authoring_revision_id")!=content_sha256({
            k:v for k,v in receipt.items() if k!="authoring_revision_id"
        })
        or manifest.get("authoring_revision_id")!=receipt.get("authoring_revision_id")
        or receipt.get("edited_mesh_lineage_hash")!=mesh.mesh_lineage_hash
        or receipt.get("edited_rig_lineage_hash")!=rig.rig_lineage_hash
        or receipt.get("edited_skin_lineage_hash")!=skin.skin_lineage_hash
        or mesh.mesh_lineage_hash!=canonical_mesh_lineage_hash_v1(mesh)
        or rig.rig_lineage_hash!=canonical_rig_lineage_hash_v2(rig)
        or skin.skin_lineage_hash!=canonical_skin_lineage_hash_v2(skin)
        or rig.mesh_lineage_hash!=mesh.mesh_lineage_hash
        or skin.mesh_lineage_hash!=mesh.mesh_lineage_hash
        or skin.deform_structure_hash!=rig.deform_structure_hash
        or mesh.faces!=base["mesh"].faces
        or mesh.edges!=base["mesh"].edges
        or tuple(v.canonical_mesh_vertex_id for v in mesh.vertices)!=tuple(
            v.canonical_mesh_vertex_id for v in base["mesh"].vertices
        )
        or tuple(j.canonical_joint_id for j in rig.joints)!=tuple(
            j.canonical_joint_id for j in base["rig"].joints
        )
        or tuple(row.canonical_mesh_vertex_id for row in skin.rows)!=tuple(
            row.canonical_mesh_vertex_id for row in base["skin"].rows
        )
        or tuple(c.canonical_control_id for c in rig.controls)!=tuple(
            c.canonical_control_id for c in base["rig"].controls
        )
        or mesh.metadata.get("original_canonical_mesh_lineage_hash")
            !=base["mesh"].mesh_lineage_hash
        or rig.metadata.get("original_canonical_rig_lineage_hash")
            !=base["rig"].rig_lineage_hash
        or skin.metadata.get("original_canonical_skin_lineage_hash")
            !=base["skin"].skin_lineage_hash
        or any(obj.metadata.get("original_rss_sha256")!=base_sha
               for obj in (mesh,rig,skin))
    ):
        raise RuntimeError("RSS_AUTHORING_REOPEN_REVISION_OR_PARENT_DRIFT")
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    validate_canonical_skin_domain_v2(skin)
    fresh_pose,fresh_pose_hashes=_make_pose_binary(
        mesh,rig,skin,base["control_frames"],base["virtual_camera"],
    )
    if (
        fresh_pose!=entries["pose.bin"]
        or fresh_pose_hashes!=receipt.get("mechanics_pose_hashes")
    ):
        raise RuntimeError("RSS_AUTHORING_REOPEN_INDEPENDENT_LBS_REPLAY_DRIFT")
    return {
        "source_preview":base,
        "authoring_revision_id":receipt["authoring_revision_id"],
        "mesh":mesh,"rig":rig,"skin":skin,
        "pose_bytes":fresh_pose,
        "receipt":receipt,
        "product_revision_minted":False,
        "graphics_uv_binding_qualified":False,
    }


__all__=[
    "SCHEMA","AUTHORITY",
    "compile_rss_vnext_canonical_user_edit_revision_v1",
    "reopen_rss_vnext_canonical_user_edit_revision_v1",
]
