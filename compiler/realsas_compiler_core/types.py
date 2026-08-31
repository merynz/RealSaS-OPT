from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
Json=dict[str,Any]
Vec2=tuple[float,float]
Vec3=tuple[float,float,float]

@dataclass(frozen=True)
class ObservationSample:
    observation_id:str
    view_index:int
    raster_xy:Vec2
    ray_origin:Vec3
    ray_forward:Vec3
    depth:float
    support:bool=True
    provenance_ref:str=""
    validity_flags:tuple[str,...]=()
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ObservationEvidenceIR:
    samples:tuple[ObservationSample,...]
    camera_model:str="KNOWN_ORTHOGRAPHIC_8VIEW"
    observation_frame:str="REALSAS_OBJECT_FRAME"
    evidence_version:str="RealSaS.ObservationEvidenceIR.v2"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class PersistenceGroup:
    group_id:str
    observation_ids:tuple[str,...]
    method:str="MUTUAL_P003"
    diagnostics:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SurfaceNode:
    surface_id:str
    P:Vec3
    support_views:tuple[int,...]
    provenance_refs:tuple[str,...]
    source_observation_ids:tuple[str,...]
    raster_bindings:tuple[tuple[int,Vec2],...]=()
    persistence_group_id:str|None=None
    derived_normal:Vec3|None=None
    validity_flags:tuple[str,...]=()
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SurfaceRelation:
    relation_id:str
    a_surface_id:str
    b_surface_id:str
    relation_kind:str
    score:float=1.0
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class RiggingSurfaceIR:
    surface_nodes:tuple[SurfaceNode,...]
    local_relations:tuple[SurfaceRelation,...]=()
    geometry_lineage_hash:str=""
    builder_id:str="RealSaS.SurfaceBuilder.current"
    schema_version:str="RealSaS.RiggingSurfaceIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SkeletonProposalJoint:
    proposal_id:str
    position:Vec3
    root_score:float=0.0
    confidence:float=1.0
    support_surface_ids:tuple[str,...]=()
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SkeletonProposalEdge:
    edge_id:str
    parent_proposal_id:str
    child_proposal_id:str
    score:float
    confidence:float=1.0
    hard_required:bool=False
    hard_forbidden:bool=False
    reason:str=""
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SkeletonProposalIR:
    joints:tuple[SkeletonProposalJoint,...]
    edges:tuple[SkeletonProposalEdge,...]
    surface_binding_hash:str
    model_provenance:str=""
    schema_version:str="RealSaS.SkeletonProposalIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedJoint:
    canonical_joint_id:str
    position:Vec3
    parent_canonical_id:str|None
    support_surface_ids:tuple[str,...]=()
    source_proposal_id:str=""
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedSkeletonIR:
    joints:tuple[QualifiedJoint,...]
    root_id:str
    qualification_report:Json
    skeleton_lineage_hash:str
    schema_version:str="RealSaS.QualifiedSkeletonIR.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SkinInfluenceProposal:
    surface_id:str
    canonical_joint_id:str
    weight:float
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SkinProposalIR:
    influences:tuple[SkinInfluenceProposal,...]
    surface_binding_hash:str
    skeleton_binding_hash:str
    model_provenance:str=""
    schema_version:str="RealSaS.SkinProposalIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedSkinRow:
    surface_id:str
    influences:tuple[tuple[str,float],...]
    simplex_residual_before:float
    correction_l1:float
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedSkinIR:
    rows:tuple[QualifiedSkinRow,...]
    surface_binding_hash:str
    skeleton_binding_hash:str
    qualification_report:Json
    skin_lineage_hash:str
    schema_version:str="RealSaS.QualifiedSkinIR.v1"
    def to_dict(self): return asdict(self)

# MWB-0: typed editable-mesh / mesh-weight seam. These types intentionally
# contain no triangulation, BBW, ARAP or XPBD implementation authority.
@dataclass(frozen=True)
class SurfaceSupportBinding:
    mode:str
    coefficients:tuple[tuple[str,float],...]
    metadata:Json=field(default_factory=dict)
    schema_version:str="RealSaS.SurfaceSupportBinding.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MeshVertexCandidate:
    candidate_vertex_id:str
    P:Vec3
    support_binding:SurfaceSupportBinding
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MeshDiscretizationCandidateIR:
    vertices:tuple[MeshVertexCandidate,...]
    faces:tuple[tuple[str,...],...]
    edges:tuple[tuple[str,str],...]
    surface_binding_hash:str
    view_index:int
    camera_binding_hash:str
    candidate_lineage_hash:str
    boundary_constraints:tuple[Json,...]=()
    coverage_classification:str=""
    solver_provenance:Json=field(default_factory=dict)
    residual_report:Json=field(default_factory=dict)
    schema_version:str="RealSaS.MeshDiscretizationCandidateIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedMeshVertex:
    canonical_mesh_vertex_id:str
    P:Vec3
    support_binding:SurfaceSupportBinding
    source_candidate_vertex_id:str=""
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedEditableMeshIR:
    vertices:tuple[QualifiedMeshVertex,...]
    faces:tuple[tuple[str,...],...]
    edges:tuple[tuple[str,str],...]
    surface_binding_hash:str
    view_index:int
    camera_binding_hash:str
    qualification_report:Json
    mesh_lineage_hash:str
    boundary_constraints:tuple[Json,...]=()
    support_coverage_classification:str=""
    schema_version:str="RealSaS.QualifiedEditableMeshIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedMeshSkinRow:
    canonical_mesh_vertex_id:str
    influences:tuple[tuple[str,float],...]
    source_support_coefficients:tuple[tuple[str,float],...]
    simplex_residual_before:float
    correction_l1:float
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedMeshSkinIR:
    rows:tuple[QualifiedMeshSkinRow,...]
    surface_binding_hash:str
    skeleton_binding_hash:str
    skin_binding_hash:str
    mesh_binding_hash:str
    transfer_method:str
    qualification_report:Json
    mesh_skin_lineage_hash:str
    schema_version:str="RealSaS.QualifiedMeshSkinIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class CanonicalPuppetGraph:
    """Legacy/current V1 product retained for compatibility during MWB rollout."""
    product_lineage_id:str
    product_state_hash:str
    parent_state_hash:str|None
    admitted_surface_hash:str
    admitted_skeleton_hash:str
    admitted_skin_hash:str
    qualification_ledger:tuple[Json,...]
    deformation_state:Json=field(default_factory=dict)
    contact_state:Json=field(default_factory=dict)
    motion_bindings:Json=field(default_factory=dict)
    editable_metadata:Json=field(default_factory=dict)
    export_contract_version:str="RealSaS.RuntimePackageIR.v1"
    schema_version:str="RealSaS.CanonicalPuppetGraph.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class CanonicalPuppetGraphV2:
    product_lineage_id:str
    product_state_hash:str
    parent_state_hash:str|None
    admitted_surface_hash:str
    admitted_skeleton_hash:str
    admitted_skin_hash:str
    admitted_mesh_hash:str
    admitted_mesh_skin_hash:str
    qualification_ledger:tuple[Json,...]
    deformation_state:Json=field(default_factory=dict)
    contact_state:Json=field(default_factory=dict)
    motion_bindings:Json=field(default_factory=dict)
    editable_metadata:Json=field(default_factory=dict)
    export_contract_version:str="RealSaS.RuntimePackageIR.v1"
    schema_version:str="RealSaS.CanonicalPuppetGraph.v2"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ProofFrame:
    product_state_hash:str
    probe_plan_hash:str
    measurement_hash:str
    passed:bool
    proof_payload:Json=field(default_factory=dict)
    schema_version:str="RealSaS.ProofFrame.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class RepairDirective:
    source_product_state_hash:str
    owner_domain:str
    target_refs:tuple[str,...]
    allowed_operation_class:str
    bounded_budget:int
    expected_failure_signature:str=""
    schema_version:str="RealSaS.RepairDirective.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class RuntimePackageIR:
    source_product_state_hash:str
    source_proof_hash:str
    manifest:Json
    runtime_payload_ref:str
    abi_schema_version:str="RealSaS.CPP17.CABI.v0.5"
    schema_version:str="RealSaS.RuntimePackageIR.v1"
    def to_dict(self): return asdict(self)

class QualificationError(RuntimeError):
    pass
