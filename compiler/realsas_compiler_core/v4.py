from __future__ import annotations

import math
from dataclasses import replace
from .hashing import content_sha256
from .mesh_binding import validate_qualified_mesh, validate_qualified_mesh_skin
from .types import QualifiedSkeletonIR, RuntimePackageIR, QualificationError
from .v4_types import (
    QualifiedSkeletonIRV2, MechanicalStateIR,
    AppearanceBindingIR, AppearanceCornerBinding,
    VisualCompletionProposalIR, QualifiedVisualCompletionIR,
    RenderableComponentIR, DirectionalRenderableIR, DirectionalRenderableSetIR,
    CapabilityRequirement, CapabilityContractIR,
    MotionClipIR, JointTransformKeyIR, JointTransformTrackIR,
    ComponentOrderTrackIR, ComponentVisibilityTrackIR, MotionStateIR,
    CanonicalPuppetGraphV3,
    ProofPlanIR, MeasurementReportIR, DomainProofReportIR, ProductProofBundleIR,
    CapabilityQualificationIR,
)

_ALLOWED_APPEARANCE = {"OBSERVED_LOCAL", "OBSERVED_CROSS_VIEW", "QUALIFIED_COMPLETION"}
_ALLOWED_CAPABILITY_ACTIVATION = {"REQUIRED", "OPTIONAL", "DISABLED"}
_ALLOWED_PROOF_STATUS = {"PASS", "FAIL", "ABSTAIN"}
_ALLOWED_CAPABILITY_STATUS = {"PROVEN", "FAIL", "ABSTAIN", "UNPROVEN"}
_REQUIRED_VIEWS = tuple(range(8))


def _hash_without(value, field_name: str) -> str:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return content_sha256(payload)


def qualified_skeleton_v2_lineage_hash(value: QualifiedSkeletonIRV2) -> str: return _hash_without(value, "skeleton_lineage_hash")
def mechanical_state_hash(value: MechanicalStateIR) -> str: return _hash_without(value, "mechanical_state_hash")
def appearance_lineage_hash(value: AppearanceBindingIR) -> str: return _hash_without(value, "appearance_lineage_hash")
def completion_lineage_hash(value: QualifiedVisualCompletionIR) -> str: return _hash_without(value, "completion_lineage_hash")
def component_state_hash(value: RenderableComponentIR) -> str: return _hash_without(value, "component_state_hash")
def direction_state_hash(value: DirectionalRenderableIR) -> str: return _hash_without(value, "direction_state_hash")
def directional_visual_state_hash(value: DirectionalRenderableSetIR) -> str: return _hash_without(value, "directional_visual_state_hash")
def capability_contract_hash(value: CapabilityContractIR) -> str: return _hash_without(value, "capability_contract_hash")
def track_hash(value) -> str: return _hash_without(value, "track_hash")
def motion_state_hash(value: MotionStateIR) -> str: return _hash_without(value, "motion_state_hash")
def proof_plan_hash(value: ProofPlanIR) -> str: return _hash_without(value, "proof_plan_hash")
def measurement_report_hash(value: MeasurementReportIR) -> str: return _hash_without(value, "measurement_report_hash")
def domain_proof_hash(value: DomainProofReportIR) -> str: return _hash_without(value, "domain_proof_hash")
def proof_bundle_hash(value: ProductProofBundleIR) -> str: return _hash_without(value, "proof_bundle_hash")
def capability_qualification_hash(value: CapabilityQualificationIR) -> str: return _hash_without(value, "qualification_hash")


def upgrade_qualified_skeleton_v2(skeleton: QualifiedSkeletonIR, *, assembly_root_binding: dict | None = None) -> QualifiedSkeletonIRV2:
    roots = tuple(sorted(j.canonical_joint_id for j in skeleton.joints if j.parent_canonical_id is None))
    if not roots:
        if skeleton.root_id: roots = (skeleton.root_id,)
        else: raise QualificationError("SKELETON_V2_NO_DEFORM_ROOT")
    binding = dict(assembly_root_binding or {})
    if binding: binding.setdefault("non_deforming", True)
    report = dict(skeleton.qualification_report)
    report["upgraded_from_schema"] = skeleton.schema_version
    report["forest_representation"] = "DEFORM_ROOT_SET"
    value = QualifiedSkeletonIRV2(skeleton.joints, roots, binding, report, "")
    value = replace(value, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(value))
    validate_qualified_skeleton_v2(value)
    return value


def validate_qualified_skeleton_v2(skeleton: QualifiedSkeletonIRV2) -> None:
    ids = {j.canonical_joint_id for j in skeleton.joints}
    if len(ids) != len(skeleton.joints) or not ids: raise QualificationError("SKELETON_V2_INVALID_JOINT_IDS")
    roots = set(skeleton.deform_root_ids)
    if not roots or not roots.issubset(ids): raise QualificationError("SKELETON_V2_INVALID_DEFORM_ROOT_SET")
    actual_roots = {j.canonical_joint_id for j in skeleton.joints if j.parent_canonical_id is None}
    if actual_roots != roots: raise QualificationError("SKELETON_V2_DEFORM_ROOT_PARENT_MISMATCH")
    for joint in skeleton.joints:
        if joint.parent_canonical_id is not None and joint.parent_canonical_id not in ids: raise QualificationError("SKELETON_V2_ILLEGAL_PARENT")
    if skeleton.assembly_root_binding and not bool(skeleton.assembly_root_binding.get("non_deforming", False)):
        raise QualificationError("SKELETON_V2_ASSEMBLY_ROOT_MUST_BE_NON_DEFORMING")
    if skeleton.skeleton_lineage_hash != qualified_skeleton_v2_lineage_hash(skeleton): raise QualificationError("SKELETON_V2_LINEAGE_HASH_MISMATCH")


def build_mechanical_state(surface, skeleton: QualifiedSkeletonIRV2, skin) -> MechanicalStateIR:
    validate_qualified_skeleton_v2(skeleton)
    if skin.surface_binding_hash != surface.geometry_lineage_hash: raise QualificationError("MECHANICAL_SKIN_SURFACE_LINEAGE_MISMATCH")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash: raise QualificationError("MECHANICAL_SKIN_SKELETON_LINEAGE_MISMATCH")
    value = MechanicalStateIR(surface, skeleton, skin, "")
    return replace(value, mechanical_state_hash=mechanical_state_hash(value))


def build_appearance_binding(*, target_view_index: int, mesh_binding_hash: str, camera_binding_hash: str, corner_bindings: tuple[AppearanceCornerBinding, ...], atlas_payload_hash: str = "", metadata: dict | None = None) -> AppearanceBindingIR:
    value = AppearanceBindingIR(target_view_index, mesh_binding_hash, camera_binding_hash, corner_bindings, "", atlas_payload_hash, metadata=metadata or {})
    value = replace(value, appearance_lineage_hash=appearance_lineage_hash(value))
    validate_appearance_binding(value)
    return value


def validate_appearance_binding(value: AppearanceBindingIR) -> None:
    if value.target_view_index not in _REQUIRED_VIEWS: raise QualificationError("APPEARANCE_INVALID_TARGET_VIEW")
    seen: set[tuple[int, int]] = set()
    for corner in value.corner_bindings:
        key = (corner.face_index, corner.corner_index)
        if corner.face_index < 0 or corner.corner_index < 0 or key in seen: raise QualificationError("APPEARANCE_INVALID_OR_DUPLICATE_CORNER_INDEX")
        seen.add(key)
        if corner.authority_class not in _ALLOWED_APPEARANCE: raise QualificationError("APPEARANCE_INVALID_AUTHORITY_CLASS")
        if corner.donor_view_index not in _REQUIRED_VIEWS: raise QualificationError("APPEARANCE_INVALID_DONOR_VIEW")
        if not (0.0 <= float(corner.confidence) <= 1.0): raise QualificationError("APPEARANCE_INVALID_CONFIDENCE")
        if any(not math.isfinite(float(x)) for x in (*corner.material_uv, *corner.donor_raster_xy)): raise QualificationError("APPEARANCE_NONFINITE_COORDINATE")
        if not corner.source_observation_hash: raise QualificationError("APPEARANCE_MISSING_OBSERVATION_AUTHORITY")
        if corner.authority_class == "OBSERVED_LOCAL":
            if corner.donor_view_index != value.target_view_index or corner.completion_id: raise QualificationError("APPEARANCE_LOCAL_AUTHORITY_MISMATCH")
        elif corner.authority_class == "OBSERVED_CROSS_VIEW":
            if corner.donor_view_index == value.target_view_index or corner.completion_id: raise QualificationError("APPEARANCE_CROSS_VIEW_AUTHORITY_MISMATCH")
        elif not corner.completion_id:
            raise QualificationError("APPEARANCE_COMPLETION_ID_REQUIRED")
    if value.appearance_lineage_hash != appearance_lineage_hash(value): raise QualificationError("APPEARANCE_LINEAGE_HASH_MISMATCH")


def qualify_visual_completion(proposal: VisualCompletionProposalIR, surface, *, qualification_report: dict) -> QualifiedVisualCompletionIR:
    if proposal.target_view_index not in _REQUIRED_VIEWS: raise QualificationError("COMPLETION_INVALID_TARGET_VIEW")
    known_surface_ids = {node.surface_id for node in surface.surface_nodes}
    if not proposal.support_surface_ids or not set(proposal.support_surface_ids).issubset(known_surface_ids):
        raise QualificationError("COMPLETION_REQUIRES_VALID_OBSERVED_SURFACE_SUPPORT")
    if any(v not in _REQUIRED_VIEWS for v in proposal.donor_view_indices): raise QualificationError("COMPLETION_INVALID_DONOR_VIEW")
    if not bool(qualification_report.get("passed", False)): raise QualificationError("COMPLETION_QUALIFICATION_FAILED")
    value = QualifiedVisualCompletionIR(proposal.completion_id, proposal.target_view_index, proposal.proposal_payload_hash, proposal.support_surface_ids, dict(qualification_report), "", proposal.required, metadata={"target_region_id": proposal.target_region_id, **proposal.metadata})
    return replace(value, completion_lineage_hash=completion_lineage_hash(value))


def build_renderable_component(*, component_id: str, view_index: int, mesh, mesh_skin, appearance: AppearanceBindingIR, setup_order: int, coverage_classification: str, completions: tuple[QualifiedVisualCompletionIR, ...] = (), default_visible: bool = True, metadata: dict | None = None) -> RenderableComponentIR:
    value = RenderableComponentIR(component_id, view_index, mesh, mesh_skin, appearance, setup_order, coverage_classification, "", completions, default_visible, metadata=metadata or {})
    return replace(value, component_state_hash=component_state_hash(value))


def validate_renderable_component(component: RenderableComponentIR, mechanical: MechanicalStateIR) -> None:
    if component.view_index not in _REQUIRED_VIEWS: raise QualificationError("RENDERABLE_COMPONENT_INVALID_VIEW")
    validate_qualified_mesh(component.mesh, mechanical.surface)
    validate_qualified_mesh_skin(component.mesh_skin, surface=mechanical.surface, skeleton=mechanical.skeleton, skin=mechanical.skin, mesh=component.mesh)
    if component.mesh.view_index != component.view_index: raise QualificationError("RENDERABLE_COMPONENT_MESH_VIEW_MISMATCH")
    validate_appearance_binding(component.appearance)
    if component.appearance.target_view_index != component.view_index: raise QualificationError("RENDERABLE_COMPONENT_APPEARANCE_VIEW_MISMATCH")
    if component.appearance.mesh_binding_hash != component.mesh.mesh_lineage_hash: raise QualificationError("RENDERABLE_COMPONENT_APPEARANCE_MESH_MISMATCH")
    if component.appearance.camera_binding_hash != component.mesh.camera_binding_hash: raise QualificationError("RENDERABLE_COMPONENT_APPEARANCE_CAMERA_MISMATCH")
    expected = {(fi, ci) for fi, face in enumerate(component.mesh.faces) for ci in range(len(face))}
    actual = {(c.face_index, c.corner_index) for c in component.appearance.corner_bindings}
    if actual != expected: raise QualificationError("RENDERABLE_COMPONENT_APPEARANCE_COVERAGE_INCOMPLETE")
    completion_by_id = {c.completion_id: c for c in component.completions}
    if len(completion_by_id) != len(component.completions): raise QualificationError("RENDERABLE_COMPONENT_DUPLICATE_COMPLETION_ID")
    for completion in component.completions:
        if completion.target_view_index != component.view_index: raise QualificationError("RENDERABLE_COMPONENT_COMPLETION_VIEW_MISMATCH")
        if completion.completion_lineage_hash != completion_lineage_hash(completion): raise QualificationError("RENDERABLE_COMPONENT_COMPLETION_HASH_MISMATCH")
    referenced = {c.completion_id for c in component.appearance.corner_bindings if c.authority_class == "QUALIFIED_COMPLETION"}
    if not referenced.issubset(completion_by_id): raise QualificationError("RENDERABLE_COMPONENT_UNQUALIFIED_COMPLETION_REFERENCE")
    required = {c.completion_id for c in component.completions if c.required}
    if not required.issubset(referenced): raise QualificationError("RENDERABLE_COMPONENT_REQUIRED_COMPLETION_NOT_BOUND")
    if component.component_state_hash != component_state_hash(component): raise QualificationError("RENDERABLE_COMPONENT_STATE_HASH_MISMATCH")


def build_directional_renderable(*, view_index: int, camera_binding_hash: str, components: tuple[RenderableComponentIR, ...], metadata: dict | None = None) -> DirectionalRenderableIR:
    value = DirectionalRenderableIR(view_index, camera_binding_hash, components, "", metadata=metadata or {})
    return replace(value, direction_state_hash=direction_state_hash(value))


def validate_directional_renderable(direction: DirectionalRenderableIR, mechanical: MechanicalStateIR) -> None:
    if direction.view_index not in _REQUIRED_VIEWS: raise QualificationError("DIRECTION_INVALID_VIEW")
    if not direction.components: raise QualificationError("DIRECTION_REQUIRES_RENDERABLE_COMPONENT")
    ids = [c.component_id for c in direction.components]; orders = [c.setup_order for c in direction.components]
    if len(ids) != len(set(ids)): raise QualificationError("DIRECTION_DUPLICATE_COMPONENT_ID")
    if len(orders) != len(set(orders)): raise QualificationError("DIRECTION_DUPLICATE_SETUP_ORDER")
    for component in direction.components:
        validate_renderable_component(component, mechanical)
        if component.view_index != direction.view_index: raise QualificationError("DIRECTION_COMPONENT_VIEW_MISMATCH")
        if component.mesh.camera_binding_hash != direction.camera_binding_hash: raise QualificationError("DIRECTION_CAMERA_BINDING_MISMATCH")
    if direction.direction_state_hash != direction_state_hash(direction): raise QualificationError("DIRECTION_STATE_HASH_MISMATCH")


def build_directional_renderable_set(directions: tuple[DirectionalRenderableIR, ...], *, metadata: dict | None = None) -> DirectionalRenderableSetIR:
    value = DirectionalRenderableSetIR(tuple(sorted(directions, key=lambda d: d.view_index)), "", metadata=metadata or {})
    return replace(value, directional_visual_state_hash=directional_visual_state_hash(value))


def validate_directional_renderable_set(value: DirectionalRenderableSetIR, mechanical: MechanicalStateIR) -> None:
    if value.exact_cardinality != 8 or len(value.directions) != 8: raise QualificationError("DIRECTIONAL_RENDERABLE_SET_REQUIRES_EXACTLY_8")
    if tuple(d.view_index for d in value.directions) != _REQUIRED_VIEWS: raise QualificationError("DIRECTIONAL_RENDERABLE_SET_VIEW_ORDER_MUST_BE_0_TO_7")
    for direction in value.directions: validate_directional_renderable(direction, mechanical)
    if value.directional_visual_state_hash != directional_visual_state_hash(value): raise QualificationError("DIRECTIONAL_RENDERABLE_SET_HASH_MISMATCH")


def build_capability_contract(profile_id: str, requirements: tuple[CapabilityRequirement, ...], *, metadata: dict | None = None) -> CapabilityContractIR:
    value = CapabilityContractIR(profile_id, tuple(sorted(requirements, key=lambda r: r.capability_id)), "", metadata=metadata or {})
    value = replace(value, capability_contract_hash=capability_contract_hash(value)); validate_capability_contract(value); return value


def validate_capability_contract(value: CapabilityContractIR) -> None:
    ids = [r.capability_id for r in value.requirements]
    if len(ids) != len(set(ids)): raise QualificationError("CAPABILITY_DUPLICATE_ID")
    for requirement in value.requirements:
        if requirement.activation not in _ALLOWED_CAPABILITY_ACTIVATION: raise QualificationError("CAPABILITY_INVALID_ACTIVATION")
        if requirement.activation != "DISABLED" and (not requirement.implementation_binding_hash or not requirement.policy_hash): raise QualificationError("CAPABILITY_ACTIVE_REQUIRES_BINDINGS")
    if value.capability_contract_hash != capability_contract_hash(value): raise QualificationError("CAPABILITY_CONTRACT_HASH_MISMATCH")


def make_single_family_e2e_capability_contract(*, base_lbs_hash: str, mesh_hash: str, skinning_hash: str, visual_hash: str, preset_motion_hash: str, runtime_hash: str, policy_hash: str) -> CapabilityContractIR:
    requirements = (
        CapabilityRequirement("BASE_LBS", "REQUIRED", base_lbs_hash, policy_hash, ("DEFORMATION",)),
        CapabilityRequirement("EDITABLE_MESH", "REQUIRED", mesh_hash, policy_hash, ("MESH_QUALITY",)),
        CapabilityRequirement("QUALIFIED_SKINNING", "REQUIRED", skinning_hash, policy_hash, ("MECHANICAL_STRUCTURE", "DEFORMATION")),
        CapabilityRequirement("VISUAL_8_DIRECTION", "REQUIRED", visual_hash, policy_hash, ("DIRECTIONAL_VISUAL",)),
        CapabilityRequirement("PRESET_MOTION", "REQUIRED", preset_motion_hash, policy_hash, ("MOTION",)),
        CapabilityRequirement("RUNTIME_BACKEND", "REQUIRED", runtime_hash, policy_hash, ("RUNTIME_CONSUMPTION",)),
    )
    return build_capability_contract("SINGLE_FAMILY_E2E_FIT_V1", requirements, metadata={"generalization_claim": False})


def build_joint_track(track_id: str, clip_id: str, canonical_joint_id: str, keys: tuple[JointTransformKeyIR, ...], *, metadata: dict | None = None) -> JointTransformTrackIR:
    value = JointTransformTrackIR(track_id, clip_id, canonical_joint_id, keys, "", metadata=metadata or {})
    return replace(value, track_hash=track_hash(value))


def build_order_track(track_id: str, clip_id: str, view_index: int, keys: tuple[dict, ...]) -> ComponentOrderTrackIR:
    value = ComponentOrderTrackIR(track_id, clip_id, view_index, keys, ""); return replace(value, track_hash=track_hash(value))


def build_visibility_track(track_id: str, clip_id: str, view_index: int, keys: tuple[dict, ...]) -> ComponentVisibilityTrackIR:
    value = ComponentVisibilityTrackIR(track_id, clip_id, view_index, keys, ""); return replace(value, track_hash=track_hash(value))


def build_motion_state(clips: tuple[MotionClipIR, ...], joint_tracks: tuple[JointTransformTrackIR, ...] = (), order_tracks: tuple[ComponentOrderTrackIR, ...] = (), visibility_tracks: tuple[ComponentVisibilityTrackIR, ...] = (), *, metadata: dict | None = None) -> MotionStateIR:
    value = MotionStateIR(clips, joint_tracks, order_tracks, visibility_tracks, "", metadata=metadata or {})
    value = replace(value, motion_state_hash=motion_state_hash(value)); validate_motion_state(value); return value


def _validate_joint_key(key: JointTransformKeyIR, duration_sec: float) -> None:
    vals = (key.time_sec, *key.translation, *key.rotation_xyzw, *key.scale)
    if any(not math.isfinite(float(v)) for v in vals): raise QualificationError("MOTION_JOINT_KEY_NONFINITE")
    if key.time_sec < 0.0 or key.time_sec > duration_sec or any(float(s) <= 0.0 for s in key.scale): raise QualificationError("MOTION_JOINT_KEY_INVALID")
    qnorm = math.sqrt(sum(float(q) * float(q) for q in key.rotation_xyzw))
    if abs(qnorm - 1.0) > 1e-6: raise QualificationError("MOTION_ROTATION_QUATERNION_NOT_UNIT")


def validate_motion_state(value: MotionStateIR) -> None:
    clips = {clip.clip_id: clip for clip in value.clips}
    if len(clips) != len(value.clips): raise QualificationError("MOTION_DUPLICATE_CLIP_ID")
    for clip in value.clips:
        if not math.isfinite(float(clip.duration_sec)) or clip.duration_sec <= 0.0: raise QualificationError("MOTION_CLIP_INVALID_DURATION")
    track_ids: set[str] = set()
    for track in (*value.joint_tracks, *value.order_tracks, *value.visibility_tracks):
        if track.track_id in track_ids: raise QualificationError("MOTION_DUPLICATE_TRACK_ID")
        track_ids.add(track.track_id)
        if track.clip_id not in clips: raise QualificationError("MOTION_TRACK_UNKNOWN_CLIP")
        if track.track_hash != track_hash(track): raise QualificationError("MOTION_TRACK_HASH_MISMATCH")
    for track in value.joint_tracks:
        if not track.keys: raise QualificationError("MOTION_JOINT_TRACK_EMPTY")
        duration = float(clips[track.clip_id].duration_sec); times=[]
        for key in track.keys: _validate_joint_key(key, duration); times.append(float(key.time_sec))
        if times != sorted(times) or len(times) != len(set(times)): raise QualificationError("MOTION_JOINT_KEYS_MUST_BE_STRICTLY_TIME_ORDERED")
    for track in (*value.order_tracks, *value.visibility_tracks):
        if track.view_index not in _REQUIRED_VIEWS: raise QualificationError("MOTION_TRACK_INVALID_VIEW")
    if value.motion_state_hash != motion_state_hash(value): raise QualificationError("MOTION_STATE_HASH_MISMATCH")


def _joint_track_has_effective_motion(track: JointTransformTrackIR) -> bool:
    if len(track.keys) < 2: return False
    first = track.keys[0]; first_state=(first.translation, first.rotation_xyzw, first.scale)
    return any((key.translation, key.rotation_xyzw, key.scale) != first_state for key in track.keys[1:])


def validate_motion_against_mechanical(value: MotionStateIR, mechanical: MechanicalStateIR) -> None:
    valid_joint_ids = {joint.canonical_joint_id for joint in mechanical.skeleton.joints}
    for track in value.joint_tracks:
        if track.canonical_joint_id not in valid_joint_ids: raise QualificationError("MOTION_TRACK_UNKNOWN_CANONICAL_JOINT")


def _required_capability_ids(contract: CapabilityContractIR) -> set[str]: return {r.capability_id for r in contract.requirements if r.activation == "REQUIRED"}

def required_proof_domains(contract: CapabilityContractIR) -> tuple[str, ...]:
    domains=set()
    for requirement in contract.requirements:
        if requirement.activation == "REQUIRED": domains.update(requirement.required_proof_domains)
    return tuple(sorted(domains))


def assemble_product_v3(mechanical: MechanicalStateIR, directional_renderables: DirectionalRenderableSetIR, capability_contract: CapabilityContractIR, motion_state: MotionStateIR, *, parent_state_hash: str | None = None, editable_metadata: dict | None = None, runtime_policy: dict | None = None) -> CanonicalPuppetGraphV3:
    if mechanical.mechanical_state_hash != mechanical_state_hash(mechanical): raise QualificationError("MECHANICAL_STATE_HASH_MISMATCH")
    validate_qualified_skeleton_v2(mechanical.skeleton)
    if mechanical.skin.surface_binding_hash != mechanical.surface.geometry_lineage_hash: raise QualificationError("MECHANICAL_SKIN_SURFACE_LINEAGE_MISMATCH")
    if mechanical.skin.skeleton_binding_hash != mechanical.skeleton.skeleton_lineage_hash: raise QualificationError("MECHANICAL_SKIN_SKELETON_LINEAGE_MISMATCH")
    validate_directional_renderable_set(directional_renderables, mechanical); validate_capability_contract(capability_contract); validate_motion_state(motion_state); validate_motion_against_mechanical(motion_state, mechanical)
    required = _required_capability_ids(capability_contract)
    if "VISUAL_8_DIRECTION" in required and len(directional_renderables.directions) != 8: raise QualificationError("REQUIRED_VISUAL_8_DIRECTION_NOT_SATISFIED")
    if "PRESET_MOTION" in required:
        preset_ids = {clip.clip_id for clip in motion_state.clips if clip.clip_kind == "PRESET"}
        if not preset_ids: raise QualificationError("REQUIRED_PRESET_MOTION_NOT_SATISFIED")
        if not any(track.clip_id in preset_ids and _joint_track_has_effective_motion(track) for track in motion_state.joint_tracks): raise QualificationError("REQUIRED_PRESET_MOTION_HAS_NO_EFFECTIVE_JOINT_MOTION")
    ledger=(
        {"stage":"MECHANICAL_STATE","hash":mechanical.mechanical_state_hash},
        {"stage":"DIRECTIONAL_VISUAL_STATE","hash":directional_renderables.directional_visual_state_hash},
        {"stage":"CAPABILITY_CONTRACT","hash":capability_contract.capability_contract_hash},
        {"stage":"MOTION_STATE","hash":motion_state.motion_state_hash},
    )
    payload={"schema":"RealSaS.CanonicalPuppetGraph.v3","parent":parent_state_hash,"mechanical":mechanical.mechanical_state_hash,"directional_visual":directional_renderables.directional_visual_state_hash,"capability_contract":capability_contract.capability_contract_hash,"motion":motion_state.motion_state_hash,"ledger":ledger,"editable":editable_metadata or {},"runtime_policy":runtime_policy or {}}
    state=content_sha256(payload); lineage="PUPPETV3:"+content_sha256({"genesis":state,"parent":parent_state_hash})[:24]
    return CanonicalPuppetGraphV3(lineage,state,parent_state_hash,mechanical,directional_renderables,capability_contract,motion_state,mechanical.mechanical_state_hash,directional_renderables.directional_visual_state_hash,motion_state.motion_state_hash,capability_contract.capability_contract_hash,ledger,editable_metadata or {},runtime_policy or {})


def bind_proof_plan(product: CanonicalPuppetGraphV3, *, proof_domain: str, operator_policy_hashes: tuple[str, ...], probe_specification: dict) -> ProofPlanIR:
    value=ProofPlanIR(product.product_state_hash,proof_domain,operator_policy_hashes,probe_specification,""); return replace(value,proof_plan_hash=proof_plan_hash(value))


def bind_measurement_report(product: CanonicalPuppetGraphV3, plan: ProofPlanIR, *, measurements: dict) -> MeasurementReportIR:
    if plan.source_product_state_hash != product.product_state_hash: raise QualificationError("STALE_PROOF_PLAN")
    value=MeasurementReportIR(product.product_state_hash,plan.proof_plan_hash,measurements,""); return replace(value,measurement_report_hash=measurement_report_hash(value))


def bind_domain_proof(product: CanonicalPuppetGraphV3, plan: ProofPlanIR, measurements: MeasurementReportIR, *, status: str, failure_signatures: tuple[dict, ...] = (), owner_attribution: tuple[dict, ...] = (), metadata: dict | None = None) -> DomainProofReportIR:
    if status not in _ALLOWED_PROOF_STATUS: raise QualificationError("INVALID_PROOF_STATUS")
    if plan.source_product_state_hash != product.product_state_hash or measurements.source_product_state_hash != product.product_state_hash: raise QualificationError("STALE_PROOF_INPUT")
    if measurements.proof_plan_hash != plan.proof_plan_hash: raise QualificationError("PROOF_MEASUREMENT_PLAN_MISMATCH")
    value=DomainProofReportIR(product.product_state_hash,plan.proof_domain,plan.proof_plan_hash,measurements.measurement_report_hash,status,failure_signatures,owner_attribution,"",metadata=metadata or {})
    return replace(value,domain_proof_hash=domain_proof_hash(value))


def bind_product_proof_bundle(product: CanonicalPuppetGraphV3, reports: tuple[DomainProofReportIR, ...], *, metadata: dict | None = None) -> ProductProofBundleIR:
    required=required_proof_domains(product.capability_contract); by_domain={}
    for report in reports:
        if report.source_product_state_hash != product.product_state_hash: raise QualificationError("STALE_DOMAIN_PROOF")
        if report.domain_proof_hash != domain_proof_hash(report): raise QualificationError("DOMAIN_PROOF_HASH_MISMATCH")
        if report.proof_domain in by_domain: raise QualificationError("DUPLICATE_DOMAIN_PROOF")
        by_domain[report.proof_domain]=report
    missing=set(required)-set(by_domain)
    if missing: overall="ABSTAIN"
    elif any(by_domain[d].status == "FAIL" for d in required): overall="FAIL"
    elif any(by_domain[d].status != "PASS" for d in required): overall="ABSTAIN"
    else: overall="PASS"
    value=ProductProofBundleIR(product.product_state_hash,required,tuple(sorted(reports,key=lambda r:r.proof_domain)),overall,"",metadata=metadata or {"missing_required_domains":sorted(missing)})
    return replace(value,proof_bundle_hash=proof_bundle_hash(value))


def require_current_proof_bundle(product: CanonicalPuppetGraphV3, bundle: ProductProofBundleIR, *, require_pass: bool = False) -> None:
    if bundle.source_product_state_hash != product.product_state_hash: raise QualificationError("STALE_PRODUCT_PROOF_BUNDLE")
    if bundle.proof_bundle_hash != proof_bundle_hash(bundle): raise QualificationError("PRODUCT_PROOF_BUNDLE_HASH_MISMATCH")
    if require_pass and bundle.overall_status != "PASS": raise QualificationError("RUNTIME_REQUIRES_PASS_PRODUCT_PROOF_BUNDLE")


def qualify_capability(product: CanonicalPuppetGraphV3, capability_id: str, *, status: str, proof_report_hashes: tuple[str, ...], metadata: dict | None = None) -> CapabilityQualificationIR:
    if status not in _ALLOWED_CAPABILITY_STATUS: raise QualificationError("INVALID_CAPABILITY_QUALIFICATION_STATUS")
    if capability_id not in {r.capability_id for r in product.capability_contract.requirements}: raise QualificationError("UNKNOWN_CAPABILITY_ID")
    value=CapabilityQualificationIR(product.product_state_hash,capability_id,status,proof_report_hashes,"",metadata=metadata or {})
    return replace(value,qualification_hash=capability_qualification_hash(value))


def project_runtime_package_v3(product: CanonicalPuppetGraphV3, proof_bundle: ProductProofBundleIR, *, manifest: dict, runtime_payload_ref: str) -> RuntimePackageIR:
    require_current_proof_bundle(product,proof_bundle,require_pass=True)
    payload=dict(manifest); payload["source_product_state_hash"]=product.product_state_hash; payload["source_proof_hash"]=proof_bundle.proof_bundle_hash; payload["canonical_product_schema"]=product.schema_version
    return RuntimePackageIR(product.product_state_hash,proof_bundle.proof_bundle_hash,payload,runtime_payload_ref)
