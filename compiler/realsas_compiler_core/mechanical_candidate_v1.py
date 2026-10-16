from __future__ import annotations
from dataclasses import asdict,dataclass,field,replace
import math
from typing import Any,Mapping
from .hashing import content_sha256
from .types import QualificationError
Json=dict[str,Any]

@dataclass(frozen=True)
class MeshCandidateRefV1:
    candidate_id:str; candidate_mesh_binding_hash:str; source_scene_binding_hash:str; topology_sequence_hash:str; static_penalty:float; metadata:Json=field(default_factory=dict); schema_version:str="RealSaS.MeshCandidateRef.v1"
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class MeshCandidateSetV1:
    candidates:tuple[MeshCandidateRefV1,...]; candidate_set_hash:str; schema_version:str="RealSaS.MeshCandidateSet.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class AdmittedMeshCandidateV1:
    candidate_id:str; candidate_mesh_binding_hash:str; static_qualification_hash:str; static_penalty:float; schema_version:str="RealSaS.AdmittedMeshCandidate.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class CandidateMechanicsProbeV1:
    candidate_id:str; mesh_binding_hash:str; skeleton_binding_hash:str; skin_binding_hash:str; dynamic_penalty:float; passed:bool; probe_hash:str; schema_version:str="RealSaS.CandidateMechanicsProbe.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class CanonicalMechanicsSelectionV1:
    selected_candidate_id:str; canonical_mesh_binding_hash:str; canonical_skeleton_binding_hash:str; canonical_skin_binding_hash:str; candidate_set_binding_hash:str; selection_hash:str; schema_version:str="RealSaS.CanonicalMechanicsSelection.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)

def _hash_without(payload,key):
    body=dict(payload);body.pop(key,None);return content_sha256(body)
def mesh_candidate_set_hash_v1(v):return _hash_without(v.to_dict(),"candidate_set_hash")
def mechanics_probe_hash_v1(v):return _hash_without(v.to_dict(),"probe_hash")
def canonical_mechanics_selection_hash_v1(v):return _hash_without(v.to_dict(),"selection_hash")

def build_mesh_candidate_set_v1(candidates:tuple[MeshCandidateRefV1,...],*,source_scene_binding_hash:str):
    if len(candidates)<2:raise QualificationError("MESH_CANDIDATE_SET_REQUIRES_MULTIPLE_PROPOSALS")
    ids=[r.candidate_id for r in candidates]; hashes=[r.candidate_mesh_binding_hash for r in candidates]
    if len(set(ids))!=len(ids):raise QualificationError("MESH_CANDIDATE_ID_DUPLICATE")
    if len(set(hashes))!=len(hashes):raise QualificationError("MESH_CANDIDATE_MESH_DUPLICATE")
    for r in candidates:
        if r.source_scene_binding_hash!=source_scene_binding_hash:raise QualificationError("MESH_CANDIDATE_SCENE_BINDING_DRIFT")
        if not math.isfinite(float(r.static_penalty)):raise QualificationError("MESH_CANDIDATE_STATIC_PENALTY_NONFINITE")
        if bool(r.metadata.get("canonical")):raise QualificationError("MESH_CANDIDATE_PREMATURE_CANONICAL_AUTHORITY")
    value=MeshCandidateSetV1(tuple(sorted(candidates,key=lambda r:r.candidate_id)),"",metadata={"authority_class":"PROPOSAL_EVIDENCE_ONLY","single_canonical_mesh_authority_minted":False,"independent_visual_mesh_allowed":False});return replace(value,candidate_set_hash=mesh_candidate_set_hash_v1(value))

def build_candidate_mechanics_probe_v1(*,candidate_id,mesh_binding_hash,skeleton_binding_hash,skin_binding_hash,dynamic_penalty,passed,metadata:Mapping[str,Any]|None=None):
    if not all(map(str,(candidate_id,mesh_binding_hash,skeleton_binding_hash,skin_binding_hash))):raise QualificationError("CANDIDATE_MECHANICS_PROBE_BINDING_EMPTY")
    if not math.isfinite(float(dynamic_penalty)) or float(dynamic_penalty)<0:raise QualificationError("CANDIDATE_MECHANICS_DYNAMIC_PENALTY_INVALID")
    value=CandidateMechanicsProbeV1(str(candidate_id),str(mesh_binding_hash),str(skeleton_binding_hash),str(skin_binding_hash),float(dynamic_penalty),bool(passed),"",metadata=dict(metadata or {}));return replace(value,probe_hash=mechanics_probe_hash_v1(value))

def select_canonical_mechanics_v1(candidate_set,admitted,probes):
    """Mint the first and only canonical M/G/W after static+dynamic candidate evidence."""
    if candidate_set.candidate_set_hash!=mesh_candidate_set_hash_v1(candidate_set):raise QualificationError("MESH_CANDIDATE_SET_HASH_DRIFT")
    by={r.candidate_id:r for r in candidate_set.candidates}; adm={r.candidate_id:r for r in admitted}; probe={r.candidate_id:r for r in probes}
    if len(adm)!=len(admitted) or len(probe)!=len(probes):raise QualificationError("CANDIDATE_SELECTION_DUPLICATE_EVIDENCE")
    if not adm:raise QualificationError("CANDIDATE_SELECTION_NO_STATIC_SURVIVOR")
    ranked=[]
    for cid,row in adm.items():
        c,p=by.get(cid),probe.get(cid)
        if c is None or p is None:raise QualificationError("CANDIDATE_SELECTION_EVIDENCE_INCOMPLETE")
        if row.candidate_mesh_binding_hash!=c.candidate_mesh_binding_hash:raise QualificationError("CANDIDATE_SELECTION_STATIC_MESH_DRIFT")
        if p.mesh_binding_hash!=c.candidate_mesh_binding_hash:raise QualificationError("CANDIDATE_SELECTION_DYNAMIC_MESH_DRIFT")
        if p.probe_hash!=mechanics_probe_hash_v1(p):raise QualificationError("CANDIDATE_SELECTION_PROBE_HASH_DRIFT")
        if p.passed:ranked.append((float(p.dynamic_penalty),float(row.static_penalty),str(cid),c,p))
    if not ranked:raise QualificationError("CANDIDATE_SELECTION_NO_DYNAMIC_SURVIVOR")
    ranked.sort(key=lambda x:(x[0],x[1],x[2]));_,_,cid,c,p=ranked[0]
    value=CanonicalMechanicsSelectionV1(cid,c.candidate_mesh_binding_hash,p.skeleton_binding_hash,p.skin_binding_hash,candidate_set.candidate_set_hash,"",metadata={"authority_class":"SOLE_CANONICAL_MECHANICS","selection_semantics":"DYNAMIC_PENALTY_THEN_STATIC_PENALTY_THEN_ID","candidate_count":len(candidate_set.candidates),"static_survivor_count":len(admitted),"dynamic_survivor_count":len(ranked),"independent_visual_mesh_allowed":False});return replace(value,selection_hash=canonical_mechanics_selection_hash_v1(value))
