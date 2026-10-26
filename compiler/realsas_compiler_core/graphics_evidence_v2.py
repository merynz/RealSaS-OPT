"""Qualified, scope-bound ray evidence on the sole canonical mesh.

Caller supplies the Registry-verified admission map and address witnesses. A
measurement_valid flag or a free confidence scalar cannot admit a record. The
exact inner row hash is bound to its artifact's qualification to prevent payload
substitution while keeping the outer container hash non-recursive.
"""
from dataclasses import asdict, dataclass
import math
import re
from .hashing import content_sha256
from .types import QualificationError


def require(condition, code):
    if not condition:
        raise QualificationError(code)


def checked_hash(value):
    require(isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None,'GRAPHICS_V2_SHA256_REQUIRED')


@dataclass(frozen=True)
class CanonicalAddressV2:
    mesh_hash: str
    face_index: int
    barycentric: tuple[float,float,float]
    binding_witness_hash: str

    def validate(self, *, mesh_hash, face_count, address_witnesses):
        checked_hash(self.mesh_hash);checked_hash(self.binding_witness_hash)
        require(self.mesh_hash==mesh_hash,'GRAPHICS_V2_CANONICAL_M_DRIFT')
        require(type(self.face_index) is int and 0<=self.face_index<face_count,'GRAPHICS_V2_FACE_INVALID')
        b=self.barycentric
        require(len(b)==3 and all(math.isfinite(x) and 0<=x<=1 for x in b)
                and math.isclose(sum(b),1,rel_tol=0,abs_tol=1e-9),'GRAPHICS_V2_BARYCENTRIC_INVALID')
        require(self.binding_witness_hash in address_witnesses,'GRAPHICS_V2_ADDRESS_WITNESS_UNADMITTED')


@dataclass(frozen=True)
class RayHitEvidenceV2:
    address: CanonicalAddressV2
    depth: float
    effective_alpha: float|None
    linear_rgb: tuple[float,float,float]|None


@dataclass(frozen=True)
class AdmittedRayArtifactV2:
    qualification_hash: str
    record_hashes: frozenset[str]


@dataclass(frozen=True)
class RayStackEvidenceV2:
    scope_hash: str
    ray_key: str
    view_index: int
    hits: tuple[RayHitEvidenceV2,...]
    complete: bool
    blend_law: str
    tie_epsilon: float
    evidence_artifact_hash: str
    qualification_hash: str
    measurement_valid: bool
    source_artistic_order_status: str = 'UNKNOWN'

    def record_hash(self):
        value=asdict(self)
        del value['evidence_artifact_hash'];del value['qualification_hash']
        return content_sha256(value)

    def validate(self, *, mesh_hash, face_count, address_witnesses, admitted_artifacts):
        for digest in (self.scope_hash,self.evidence_artifact_hash,self.qualification_hash):checked_hash(digest)
        require(isinstance(self.ray_key,str) and bool(self.ray_key),'GRAPHICS_V2_RAY_KEY_REQUIRED')
        require(type(self.view_index) is int and self.view_index>=0,'GRAPHICS_V2_VIEW_INVALID')
        require(type(self.complete) is bool and type(self.measurement_valid) is bool,'GRAPHICS_V2_BOOLEAN_MASK_REQUIRED')
        require(math.isfinite(self.tie_epsilon) and self.tie_epsilon>=0,'GRAPHICS_V2_TIE_POLICY_REQUIRED')
        require(isinstance(self.blend_law,str) and bool(self.blend_law),'GRAPHICS_V2_BLEND_REQUIRED')
        require(self.source_artistic_order_status in {'NO_AUTHORED_ORDER__DEPTH_QUALIFIED','UNSUPPORTED_AUTHORED_ORDER','UNKNOWN'},
                'GRAPHICS_V2_SOURCE_ORDER_STATUS_REQUIRED')
        faces=[];depths=[]
        for hit in self.hits:
            hit.address.validate(mesh_hash=mesh_hash,face_count=face_count,address_witnesses=address_witnesses)
            require(math.isfinite(hit.depth),'GRAPHICS_V2_DEPTH_INVALID')
            require(hit.effective_alpha is None or (math.isfinite(hit.effective_alpha) and 0<=hit.effective_alpha<=1),'GRAPHICS_V2_ALPHA_INVALID')
            require(hit.linear_rgb is None or (len(hit.linear_rgb)==3 and all(math.isfinite(x) for x in hit.linear_rgb)),'GRAPHICS_V2_COLOUR_INVALID')
            faces.append(hit.address.face_index);depths.append(hit.depth)
        require(len(set(faces))==len(faces),'GRAPHICS_V2_DUPLICATE_FACE_IN_RAY')
        require(all(a<=b for a,b in zip(depths,depths[1:])),'GRAPHICS_V2_UNSORTED_RAY')
        admitted=admitted_artifacts.get(self.evidence_artifact_hash)
        require(isinstance(admitted,AdmittedRayArtifactV2) and admitted.qualification_hash==self.qualification_hash,'GRAPHICS_V2_ARTIFACT_UNADMITTED')
        require(self.record_hash() in admitted.record_hashes,'GRAPHICS_V2_RECORD_NOT_IN_QUALIFIED_ARTIFACT')
