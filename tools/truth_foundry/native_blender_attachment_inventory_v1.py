from __future__ import annotations

"""Blender native attachment *inventory*, not an attachment truth generator.

Authoring bone-parent and active rigid constraints are typed evidence for an
object-to-bone link. They DO NOT establish holder triangle, barycentric grip,
source contact, editable drawable, or any verified NO_FACE negative. Absence
of a recognized Blender constraint is explicitly UNKNOWN, never NO_FACE.
ARMATURE deform modifiers are recorded but do not imply rigid ownership.

All mesh owner indices come from the existing source teacher exporter; this
pure-Python court neither imports Blender nor runs model inference.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

SCHEMA = "RealSaS.TruthFoundry.BlenderNativeAttachmentInventoryCandidate.v1"
AUTHORITY = "SOURCE_AUTHORED_OBJECT_BONE_LINK_CANDIDATE_ONLY"
_UNKNOWN = "UNKNOWN_NO_EXPLICIT_UNIQUE_RIGID_BONE_LINK"
_CANDIDATE = "UNQUALIFIED_AUTHORED_BONE_LINK_CANDIDATE"
_AMBIGUOUS = "ABSTAIN_AMBIGUOUS_OR_PARTIAL_SOURCE_BINDING"


def _fail(reason: str) -> None:
    raise ValueError("RSS_VNEXT_BLENDER_ATTACHMENT_INVENTORY_" + reason)


def _hash(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf8")).hexdigest()


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


@dataclass(frozen=True)
class BlenderNativeOwnerLinkCandidateV1:
    owner_index: int
    source_object_name: str
    source_face_start: int
    source_face_count: int
    parent_armature_name: str | None
    parent_bone_name: str | None
    authoring_mechanism: str | None
    declared_link_count: int
    unresolved_constraint_count: int
    armature_deform_modifier_count: int
    state: str
    missing_witnesses: tuple[str, ...]
    source_contact_qualified: bool = False
    holder_face_known: bool = False
    held_face_selected: bool = False
    exhaustive_no_attachment_verified: bool = False
    training_admitted: bool = False


@dataclass(frozen=True)
class BlenderNativeAttachmentInventoryCandidateV1:
    source_asset_sha256: str
    native_teacher_receipt_sha256: str
    native_teacher_npz_sha256: str
    observation_admission_sha256: str
    source_face_owner_sha256: str
    authoring_inventory_sha256: str
    owners: tuple[BlenderNativeOwnerLinkCandidateV1, ...]
    inventory_sha256: str
    schema: str = SCHEMA
    authority: str = AUTHORITY
    trainer_admission_qualified: bool = False
    exhaustive_empty_socket_inventory_qualified: bool = False
    source_rigid_attachment_qualified: bool = False


def _rigid_constraint_link(row: Mapping[str, object]) -> tuple[str,str,str] | None:
    if (
        row.get("type") not in ("CHILD_OF", "COPY_TRANSFORMS")
        or row.get("mute") is not False
        or type(row.get("influence")) not in (int, float)
        or abs(float(row["influence"]) - 1.) > 1.e-7
        or row.get("target_type") != "ARMATURE"
        or not isinstance(row.get("target"), str)
        or not isinstance(row.get("subtarget"), str)
        or not row["subtarget"]
    ):
        return None
    if row["type"] == "CHILD_OF" and any(
        row.get(k) is not True
        for k in (
            "use_location_x","use_location_y","use_location_z",
            "use_rotation_x","use_rotation_y","use_rotation_z",
            "use_scale_x","use_scale_y","use_scale_z",
        )
    ):
        return None
    return (str(row["target"]),str(row["subtarget"]),str(row["type"]))


def compile_blender_native_attachment_inventory_candidate_v1(
    *, owner_rows: Sequence[Mapping[str,object]],
    source_face_owner: np.ndarray,
    source_asset_sha256: str,
    native_teacher_receipt_sha256: str,
    native_teacher_npz_sha256: str,
    observation_admission_sha256: str,
) -> BlenderNativeAttachmentInventoryCandidateV1:
    """Classify exact authored object/bone links, never classify held→holder.

    Every source triangle is owned by exactly one teacher-selected renderable
    object, and owner rows are in the teacher's stable order. The input
    constraint list must be a COMPLETE original native Blender inventory,
    including muted and unsupported constraints; omitting constraints cannot
    manufacture an authored exhaustive-negative certificate.
    """
    if not all(_sha(s) for s in (
        source_asset_sha256,native_teacher_receipt_sha256,
        native_teacher_npz_sha256,observation_admission_sha256,
    )):
        _fail("SOURCE_PROVENANCE_HASH_DOMAIN")
    if (
        not isinstance(owner_rows,(tuple,list)) or not 1<=len(owner_rows)<=65536
        or not isinstance(source_face_owner,np.ndarray)
        or source_face_owner.ndim!=1
        or source_face_owner.dtype.kind not in "iu"
        or not 1<=len(source_face_owner)<=20_000_000
        or np.any(source_face_owner<0)
        or np.any(source_face_owner>=len(owner_rows))
        or np.any(source_face_owner[1:]<source_face_owner[:-1])
    ):
        _fail("TEACHER_FACE_OWNER_DOMAIN")
    candidates=[]
    normalized=[]
    offset=0
    for index,record in enumerate(owner_rows):
        if not isinstance(record,Mapping):
            _fail("SOURCE_OWNER_ROW_TYPED")
        constraints=record.get("constraints")
        modifiers=record.get("modifiers")
        if (
            record.get("owner_index")!=index
            or not isinstance(record.get("name"),str)
            or not record["name"]
            or not isinstance(constraints,(tuple,list))
            or not isinstance(modifiers,(tuple,list))
            or not isinstance(record.get("parent_type"),str)
            or record.get("parent_name") is not None
            and not isinstance(record["parent_name"],str)
            or record.get("parent_bone") is not None
            and not isinstance(record["parent_bone"],str)
            or record.get("parent_object_type") is not None
            and not isinstance(record["parent_object_type"],str)
        ):
            _fail("SOURCE_OWNER_ROW_INCOMPLETE")
        count=int(np.sum(source_face_owner==index))
        if count<1 or np.any(source_face_owner[offset:offset+count]!=index):
            _fail("SOURCE_FACE_CONTIGUOUS_OWNER_DRIFT")
        links=[]
        unresolved=0
        if record["parent_type"]=="BONE":
            if (
                record.get("parent_object_type")=="ARMATURE"
                and record.get("parent_name") and record.get("parent_bone")
            ):
                links.append((record["parent_name"],record["parent_bone"],"BONE_PARENT"))
            else:
                unresolved+=1
        elif record["parent_type"] not in ("OBJECT","", "NONE"):
            unresolved+=1
        for item in constraints:
            if not isinstance(item,Mapping) or not isinstance(item.get("type"),str):
                _fail("SOURCE_CONSTRAINT_TYPED")
            link=_rigid_constraint_link(item)
            if link is None:
                # Muted or unknown constraints are retained as *uncertainty*.
                # Some constraint families are deform-only, so even if no
                # candidate rigid parent exists this is NOT a negative.
                unresolved+=1
            else:
                links.append(link)
        for item in modifiers:
            if not isinstance(item,Mapping) or not isinstance(item.get("type"),str):
                _fail("SOURCE_MODIFIER_TYPED")
        deform=sum(str(m["type"])=="ARMATURE" for m in modifiers)
        pairs={(a,b) for a,b,_ in links}
        if len(pairs)==1 and unresolved==0:
            parent,bone=next(iter(pairs))
            state=_CANDIDATE
            mechanism="+".join(sorted({kind for _,_,kind in links}))
        elif len(links)>0 or unresolved:
            parent=bone=mechanism=None
            state=_AMBIGUOUS
        else:
            parent=bone=mechanism=None
            state=_UNKNOWN
        missing=[
            "AUTHORED_SOCKET_ID_AND_HELD_HOLDER_SURFACE_PAIR",
            "SOURCE_NATIVE_PARENT_ROOT_INTERVENTION_RESPONSE",
            "EXTERNAL_SOURCE_ASSET_AND_ACTION_ADMISSION",
            "HELDOUT_CONTACT_AND_ATTACH_DETACH_CONSEQUENCE",
        ]
        if state!=_CANDIDATE:
            missing.append("UNIQUE_SUPPORTED_AUTHORED_BONE_LINK")
        candidates.append(BlenderNativeOwnerLinkCandidateV1(
            owner_index=index,source_object_name=record["name"],
            source_face_start=offset,source_face_count=count,
            parent_armature_name=parent,parent_bone_name=bone,
            authoring_mechanism=mechanism,
            declared_link_count=len(links),
            unresolved_constraint_count=unresolved,
            armature_deform_modifier_count=deform,
            state=state,missing_witnesses=tuple(missing),
        ))
        normalized.append({
            "owner_index":index,"name":record["name"],
            "parent_name":record.get("parent_name"),
            "parent_object_type":record.get("parent_object_type"),
            "parent_type":record.get("parent_type"),
            "parent_bone":record.get("parent_bone"),
            "constraints":[dict(x) for x in constraints],
            "modifiers":[dict(x) for x in modifiers],
        })
        offset+=count
    if offset!=len(source_face_owner) or len({r["name"] for r in normalized})!=len(normalized):
        _fail("TEACHER_OWNER_UNIQUE_OR_COVERAGE")
    native_hash=_hash(normalized)
    face_payload=np.ascontiguousarray(source_face_owner.astype("<i8"))
    face_hash=sha256(json.dumps({
        "dtype":"int64","shape":list(face_payload.shape),
    },sort_keys=True,separators=(",",":")).encode()+b"\x00"+face_payload.tobytes()).hexdigest()
    payload={
        "source_asset_sha256":source_asset_sha256,
        "native_teacher_receipt_sha256":native_teacher_receipt_sha256,
        "native_teacher_npz_sha256":native_teacher_npz_sha256,
        "observation_admission_sha256":observation_admission_sha256,
        "source_face_owner_sha256":face_hash,
        "authoring_inventory_sha256":native_hash,
        "owners":[asdict(o) for o in candidates],
        "schema":SCHEMA,"authority":AUTHORITY,
        "trainer_admission_qualified":False,
        "exhaustive_empty_socket_inventory_qualified":False,
        "source_rigid_attachment_qualified":False,
    }
    return BlenderNativeAttachmentInventoryCandidateV1(
        source_asset_sha256=source_asset_sha256,
        native_teacher_receipt_sha256=native_teacher_receipt_sha256,
        native_teacher_npz_sha256=native_teacher_npz_sha256,
        observation_admission_sha256=observation_admission_sha256,
        source_face_owner_sha256=face_hash,
        authoring_inventory_sha256=native_hash,
        owners=tuple(candidates),
        inventory_sha256=_hash(payload),
    )


__all__=[
    "SCHEMA","AUTHORITY","BlenderNativeOwnerLinkCandidateV1",
    "BlenderNativeAttachmentInventoryCandidateV1",
    "compile_blender_native_attachment_inventory_candidate_v1",
]
