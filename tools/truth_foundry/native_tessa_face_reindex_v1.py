from __future__ import annotations

"""Source-only exact quantized native-face -> TESSA teacher-face reindex court.

Never used by the two-view model inference. TESSA's adjacent-strip grammar
changes face ordering; its per-component quantized coordinate welding can also
collapse distinct native vertices. Therefore identical list offsets, nearest
vertex, source component label, and component proximity CANNOT grant a native
held/holder training label. This court tests an exact BIJECTION in the
TESSA quantized face namespace and fails if it cannot be proven.

This is a structural-index candidate, *NOT* source-contact, edit/drawable,
mechanical ownership, authoring rig or source-rights authority. An independent
native source/pose/intervention court must still qualify the annotation before
RSSvNextStructuralTrainer can receive any positive/negative supervision.
"""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json

import numpy as np

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v1.tokenization_v1 import quantize_tessa_xyz_v1


SCHEMA="RealSaS.TruthFoundry.NativeTESSAFaceReindexCandidate.v1"
AUTHORITY="STRUCTURAL_FACE_IDENTITY_IN_EXACT_TESSA_QUANTIZED_DOMAIN_ONLY"
_NOT_SOURCE="SOURCE_NATIVE_ATTACHMENT_AND_MOTION_AUTHORITY_NOT_VERIFIED"


@dataclass(frozen=True)
class NativeTESSAFaceReindexCandidateV1:
    source_face_to_tessa_face:tuple[int,...]
    tessa_face_to_source_face:tuple[int,...]
    source_corner_to_tessa_corner:tuple[tuple[int,int,int],...]
    source_face_winding_parity:tuple[int,...]  # 0 even / 1 odd; not render winding authority
    source_face_components:tuple[int,...]
    tessa_face_components:tuple[int,...]
    normalized_source_mesh_canonical_sha256:str
    tessa_teacher_token_sha256:str
    tessa_mesh_topology_digest:str
    candidate_sha256:str
    schema:str=SCHEMA
    authority:str=AUTHORITY
    source_mesh_geometry_qualified:bool=False
    source_contact_qualified:bool=False
    source_drawable_qualified:bool=False
    native_rig_attachment_qualified:bool=False
    trainer_admission_qualified:bool=False
    missing_authority:str=_NOT_SOURCE


def _fail(code:str)->None:
    raise ValueError("RSS_VNEXT_NATIVE_TESSA_REINDEX_"+code)


def _json_hash(obj:object)->str:
    return sha256(json.dumps(
        obj,sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode("utf8")).hexdigest()


def _source_arrays_digest(*,xyz:np.ndarray,faces:np.ndarray,components:np.ndarray)->str:
    h=sha256()
    for name,array in (
        ("source_xyz",xyz.astype("<f8",copy=False)),
        ("source_faces",faces.astype("<i8",copy=False)),
        ("source_component_ids",components.astype("<i8",copy=False)),
    ):
        payload=np.ascontiguousarray(array)
        h.update(_json_hash({
            "name":name,"dtype":str(payload.dtype),"shape":list(payload.shape)
        }).encode())
        h.update(b"\x00")
        h.update(payload.tobytes())
    return h.hexdigest()


def _torch_int64_digest(value:np.ndarray)->str:
    """Replicate the student face/component tensor hash without torch/Blender."""
    a=np.ascontiguousarray(np.asarray(value,dtype="<i8"))
    header=json.dumps({"dtype":"torch.int64","shape":list(a.shape)},
                      sort_keys=True,separators=(",",":")).encode()
    return sha256(header+b"\x00"+a.tobytes()).hexdigest()


def _student_topology_digest(
    tokens:tuple[int,...], faces:np.ndarray,
    vcomponents:np.ndarray,fcomponents:np.ndarray,
)->str:
    return _json_hash({
        "topology_tokens_sha256":sha256(json.dumps(
            list(tokens),separators=(",",":"),
        ).encode()).hexdigest(),
        "faces_sha256":_torch_int64_digest(faces),
        "vertices_sha256":_torch_int64_digest(vcomponents),
        "components_sha256":_torch_int64_digest(fcomponents),
    })


def _permutation_parity(indices:tuple[int,int,int])->int:
    return sum(indices[i]>indices[j] for i in range(3) for j in range(i+1,3))%2


def reindex_source_faces_to_tessa_teacher_v1(
    *,
    source_vertices_normalized:np.ndarray,
    source_faces:np.ndarray,
    source_face_component_indices:np.ndarray,
    tessa_teacher_token_ids:tuple[int,...],
    coordinate_bins:int=256,
)->NativeTESSAFaceReindexCandidateV1:
    """Require one-to-one source *quantized faces*, not an approximate fit.

    The source coordinates MUST already be exactly normalized to TESSA
    [-0.5,0.5]. This court does not invent a normalization, remesh, source
    component partition, contact label, mesh topology or triangulation.
    Every native face is matched under a source component + 3-vertex
    quantized-key signature, independent of TESSA face ordering.
    """
    x=np.asarray(source_vertices_normalized)
    faces=np.asarray(source_faces)
    comp=np.asarray(source_face_component_indices)
    if (
        type(coordinate_bins) is not int or not 2<=coordinate_bins<=65536
        or type(tessa_teacher_token_ids) is not tuple
        or len(tessa_teacher_token_ids)<4
        or any(type(t) is not int for t in tessa_teacher_token_ids)
    ):
        _fail("TEACHER_SCHEMA_OR_COORDINATE_BINS")
    if (
        x.ndim!=2 or x.shape[1]!=3 or x.dtype.kind!="f"
        or len(x)<3 or not np.isfinite(x).all()
        or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in "iu"
        or comp.shape!=(len(faces),) or comp.dtype.kind not in "iu"
        or len(faces)<1 or np.any(faces<0) or np.any(faces>=len(x))
        or np.any(comp<0)
    ):
        _fail("SOURCE_MESH_ARRAY_DOMAIN")
    if (
        len(set(map(int,comp))) != int(comp.max())+1
        or np.any(np.sort(faces,axis=1)[:,1:]==
                  np.sort(faces,axis=1)[:,:-1])
    ):
        _fail("SOURCE_FACE_OR_COMPONENT_NOT_INDEXED")
    if x.min() < -.5000001 or x.max() > .5000001:
        _fail("SOURCE_NOT_ALREADY_IN_TESSA_NORMALIZED_FRAME")

    cfg=TESSAConfigV1(coordinate_bins=coordinate_bins)
    decoded=decode_tessa_asset_sequence_v1(tessa_teacher_token_ids,cfg=cfg)
    if (
        decoded.component_count!=int(comp.max())+1
        or len(decoded.faces)!=len(faces)
        or not decoded.faces.size
    ):
        _fail("STUDENT_SOURCE_FACE_COMPONENT_COUNTS_MISMATCH")
    source_q=quantize_tessa_xyz_v1(x.astype(np.float64),coordinate_bins)
    decoded_q=quantize_tessa_xyz_v1(
        decoded.vertices_normalized,coordinate_bins,
    )
    # Exact quantized vertex collisions within a component destroy a
    # source-native vertex address, even if the faces happen to look valid.
    # Across TESSA components they are intentionally distinct.
    used_vertex_map:dict[tuple[int,tuple[int,int,int]],int]={}
    for fi,row in enumerate(faces):
        for vi in row:
            vi=int(vi)
            key=(int(comp[fi]),tuple(map(int,source_q[vi])))
            former=used_vertex_map.setdefault(key,vi)
            if former!=vi:
                _fail("QUANTIZATION_VERTEX_COLLISION_UNDER_COMPONENT_WELD")

    def face_signature(
        q:np.ndarray, row:np.ndarray, owner:int,
    )->tuple[int,tuple[tuple[int,int,int],...]]:
        return (owner,tuple(sorted(tuple(map(int,q[int(vi)])) for vi in row)))

    source_keys={}
    for fi,row in enumerate(faces):
        key=face_signature(source_q,row,int(comp[fi]))
        if key in source_keys:
            _fail("SOURCE_FACE_COLLISION_AFTER_QUANTIZATION")
        source_keys[key]=fi
    student_keys={}
    for fi,row in enumerate(decoded.faces):
        key=face_signature(decoded_q,row,int(decoded.face_component_indices[fi]))
        if key in student_keys:
            _fail("STUDENT_FACE_COLLISION_AFTER_QUANTIZATION")
        student_keys[key]=fi
    if source_keys.keys()!=student_keys.keys():
        _fail("NATIVE_TESSA_TRIANGLE_KEY_BIJECTION_MISSING")

    forward=tuple(int(student_keys[
        face_signature(source_q,faces[fi],int(comp[fi]))
    ]) for fi in range(len(faces)))
    reverse=tuple(int(source_keys[
        face_signature(decoded_q,decoded.faces[fi],
                       int(decoded.face_component_indices[fi]))
    ]) for fi in range(len(decoded.faces)))
    corner_perm=[]
    parity=[]
    for source_fi,teacher_fi in enumerate(forward):
        old=[tuple(map(int,source_q[int(i)])) for i in faces[source_fi]]
        new=[tuple(map(int,decoded_q[int(i)])) for i in decoded.faces[teacher_fi]]
        if len(set(old))!=3 or len(set(new))!=3:
            _fail("DEGENERATE_QUANTIZED_FACE")
        permutation=tuple(new.index(c) for c in old)
        if len(set(permutation))!=3:
            _fail("CORNER_MAP_AMBIGUOUS")
        corner_perm.append(permutation)
        parity.append(_permutation_parity(permutation))

    tessa_face_comp=tuple(int(i) for i in decoded.face_component_indices)
    source_face_comp=tuple(int(i) for i in comp)
    original_digest=_source_arrays_digest(xyz=x,faces=faces,components=comp)
    student_digest=_student_topology_digest(
        tessa_teacher_token_ids,decoded.faces,
        np.asarray(decoded.vertex_component_indices,dtype=np.int64),
        np.asarray(decoded.face_component_indices,dtype=np.int64),
    )
    payload={
        "source_face_to_tessa_face":forward,
        "tessa_face_to_source_face":reverse,
        "source_corner_to_tessa_corner":tuple(corner_perm),
        "source_face_winding_parity":tuple(parity),
        "source_face_components":source_face_comp,
        "tessa_face_components":tessa_face_comp,
        "normalized_source_mesh_canonical_sha256":original_digest,
        "tessa_teacher_token_sha256":sha256(json.dumps(
            list(tessa_teacher_token_ids),separators=(",",":"),
        ).encode()).hexdigest(),
        "tessa_mesh_topology_digest":student_digest,
        "schema":SCHEMA,"authority":AUTHORITY,
        "source_mesh_geometry_qualified":False,
        "source_contact_qualified":False,
        "source_drawable_qualified":False,
        "native_rig_attachment_qualified":False,
        "trainer_admission_qualified":False,
        "missing_authority":_NOT_SOURCE,
    }
    return NativeTESSAFaceReindexCandidateV1(
        **{**payload,"candidate_sha256":_json_hash(payload)},
    )


def read_native_tessa_face_reindex_candidate_v1(
    raw:dict,
)->NativeTESSAFaceReindexCandidateV1:
    """Portable face-mapping record; never an untrusted admission receipt."""
    if type(raw) is not dict or set(raw)!={f.name for f in __import__("dataclasses").fields(
        NativeTESSAFaceReindexCandidateV1
    )}:
        _fail("CANDIDATE_FIELDS_INVALID")
    data=dict(raw)
    for name in (
        "source_face_to_tessa_face","tessa_face_to_source_face",
        "source_face_winding_parity","source_face_components",
        "tessa_face_components",
    ):
        if not isinstance(data[name],(tuple,list)):
            _fail("CANDIDATE_INDEX_ARRAY_INVALID")
        data[name]=tuple(data[name])
    name="source_corner_to_tessa_corner"
    data[name]=tuple(tuple(x) for x in data[name])
    obj=NativeTESSAFaceReindexCandidateV1(**data)
    original=asdict(obj)
    digest=original.pop("candidate_sha256")
    if (
        _json_hash(original)!=digest
        or obj.schema!=SCHEMA or obj.authority!=AUTHORITY
        or any((
            obj.source_mesh_geometry_qualified,obj.source_contact_qualified,
            obj.source_drawable_qualified,obj.native_rig_attachment_qualified,
            obj.trainer_admission_qualified,
        ))
        or obj.missing_authority!=_NOT_SOURCE
        or len(obj.source_face_to_tessa_face)!=len(obj.tessa_face_to_source_face)
        or sorted(obj.source_face_to_tessa_face)!=list(range(len(obj.tessa_face_to_source_face)))
        or sorted(obj.tessa_face_to_source_face)!=list(range(len(obj.source_face_to_tessa_face)))
    ):
        _fail("CANDIDATE_HASH_OR_AUTHORITY_DRIFT")
    return obj


__all__=[
    "SCHEMA","AUTHORITY","NativeTESSAFaceReindexCandidateV1",
    "reindex_source_faces_to_tessa_teacher_v1",
    "read_native_tessa_face_reindex_candidate_v1",
]
