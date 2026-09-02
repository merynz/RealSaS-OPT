from __future__ import annotations
from dataclasses import dataclass, field, asdict
from .types import Json, Vec2, Vec3, RiggingSurfaceIR, QualifiedJoint, QualifiedSkinIR, QualifiedEditableMeshIR, QualifiedMeshSkinIR

Quat=tuple[float,float,float,float]

@dataclass(frozen=True)
class QualifiedSkeletonIRV2:
    joints: tuple[QualifiedJoint, ...]
    deform_root_ids: tuple[str, ...]
    assembly_root_binding: Json
    qualification_report: Json
    skeleton_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedSkeletonIR.v2"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MechanicalStateIR:
    surface: RiggingSurfaceIR
    skeleton: QualifiedSkeletonIRV2
    skin: QualifiedSkinIR
    mechanical_state_hash: str
    schema_version: str = "RealSaS.MechanicalStateIR.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class AppearanceCornerBinding:
    face_index: int
    corner_index: int
    material_uv: Vec2
    donor_view_index: int
    donor_raster_xy: Vec2
    source_observation_hash: str
    authority_class: str
    completion_id: str = ""
    confidence: float = 1.0
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class AppearanceBindingIR:
    target_view_index: int
    mesh_binding_hash: str
    camera_binding_hash: str
    corner_bindings: tuple[AppearanceCornerBinding, ...]
    appearance_lineage_hash: str
    atlas_payload_hash: str = ""
    schema_version: str = "RealSaS.AppearanceBindingIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class VisualCompletionProposalIR:
    completion_id: str
    target_view_index: int
    target_region_id: str
    support_surface_ids: tuple[str, ...]
    donor_view_indices: tuple[int, ...]
    proposal_payload_hash: str
    required: bool = False
    schema_version: str = "RealSaS.VisualCompletionProposalIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QualifiedVisualCompletionIR:
    completion_id: str
    target_view_index: int
    proposal_payload_hash: str
    support_surface_ids: tuple[str, ...]
    qualification_report: Json
    completion_lineage_hash: str
    required: bool = False
    authority_class: str = "QUALIFIED_COMPLETION"
    schema_version: str = "RealSaS.QualifiedVisualCompletionIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class RenderableComponentIR:
    component_id: str
    view_index: int
    mesh: QualifiedEditableMeshIR
    mesh_skin: QualifiedMeshSkinIR
    appearance: AppearanceBindingIR
    setup_order: int
    coverage_classification: str
    component_state_hash: str
    completions: tuple[QualifiedVisualCompletionIR, ...] = ()
    default_visible: bool = True
    schema_version: str = "RealSaS.RenderableComponentIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class DirectionalRenderableIR:
    view_index: int
    camera_binding_hash: str
    components: tuple[RenderableComponentIR, ...]
    direction_state_hash: str
    schema_version: str = "RealSaS.DirectionalRenderableIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class DirectionalRenderableSetIR:
    directions: tuple[DirectionalRenderableIR, ...]
    directional_visual_state_hash: str
    exact_cardinality: int = 8
    schema_version: str = "RealSaS.DirectionalRenderableSetIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class CapabilityRequirement:
    capability_id: str
    activation: str
    implementation_binding_hash: str
    policy_hash: str
    required_proof_domains: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class CapabilityContractIR:
    profile_id: str
    requirements: tuple[CapabilityRequirement, ...]
    capability_contract_hash: str
    schema_version: str = "RealSaS.CapabilityContractIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MotionClipIR:
    clip_id: str
    clip_kind: str
    clip_payload_hash: str
    required_capabilities: tuple[str, ...] = ()
    source_ref: str = ""
    duration_sec: float = 1.0
    loop: bool = False
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class JointTransformKeyIR:
    time_sec: float
    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation_xyzw: Quat = (0.0, 0.0, 0.0, 1.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class JointTransformTrackIR:
    track_id: str
    clip_id: str
    canonical_joint_id: str
    keys: tuple[JointTransformKeyIR, ...]
    track_hash: str
    schema_version: str = "RealSaS.JointTransformTrackIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ComponentOrderTrackIR:
    track_id: str
    clip_id: str
    view_index: int
    keys: tuple[Json, ...]
    track_hash: str
    schema_version: str = "RealSaS.ComponentOrderTrackIR.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ComponentVisibilityTrackIR:
    track_id: str
    clip_id: str
    view_index: int
    keys: tuple[Json, ...]
    track_hash: str
    schema_version: str = "RealSaS.ComponentVisibilityTrackIR.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MotionStateIR:
    clips: tuple[MotionClipIR, ...]
    joint_tracks: tuple[JointTransformTrackIR, ...]
    order_tracks: tuple[ComponentOrderTrackIR, ...]
    visibility_tracks: tuple[ComponentVisibilityTrackIR, ...]
    motion_state_hash: str
    schema_version: str = "RealSaS.MotionStateIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class CanonicalPuppetGraphV3:
    product_lineage_id: str
    product_state_hash: str
    parent_state_hash: str | None
    mechanical_state: MechanicalStateIR
    directional_renderables: DirectionalRenderableSetIR
    capability_contract: CapabilityContractIR
    motion_state: MotionStateIR
    mechanical_state_hash: str
    directional_visual_state_hash: str
    motion_state_hash: str
    capability_contract_hash: str
    qualification_ledger: tuple[Json, ...]
    editable_metadata: Json = field(default_factory=dict)
    runtime_policy: Json = field(default_factory=dict)
    export_contract_version: str = "RealSaS.RuntimePackageIR.v1"
    schema_version: str = "RealSaS.CanonicalPuppetGraph.v3"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ProofPlanIR:
    source_product_state_hash: str
    proof_domain: str
    operator_policy_hashes: tuple[str, ...]
    probe_specification: Json
    proof_plan_hash: str
    schema_version: str = "RealSaS.ProofPlanIR.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MeasurementReportIR:
    source_product_state_hash: str
    proof_plan_hash: str
    measurements: Json
    measurement_report_hash: str
    schema_version: str = "RealSaS.MeasurementReportIR.v1"
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class DomainProofReportIR:
    source_product_state_hash: str
    proof_domain: str
    proof_plan_hash: str
    measurement_report_hash: str
    status: str
    failure_signatures: tuple[Json, ...]
    owner_attribution: tuple[Json, ...]
    domain_proof_hash: str
    schema_version: str = "RealSaS.DomainProofReportIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ProductProofBundleIR:
    source_product_state_hash: str
    required_domains: tuple[str, ...]
    domain_reports: tuple[DomainProofReportIR, ...]
    overall_status: str
    proof_bundle_hash: str
    schema_version: str = "RealSaS.ProductProofBundleIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class CapabilityQualificationIR:
    source_product_state_hash: str
    capability_id: str
    status: str
    proof_report_hashes: tuple[str, ...]
    qualification_hash: str
    schema_version: str = "RealSaS.CapabilityQualificationIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)
