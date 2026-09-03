from __future__ import annotations

"""Typed failure diagnostics without causal-owner invention.

Promoted/rebound from the historical v0.5 motion-failure diagnostic semantics.
The historical implementation correctly separated *failure localization* from
*owner attribution*.  This service preserves that invariant while consuming the
current compiler's JSON measurement reports instead of historical contracts.

It is deliberately subordinate: it does not decide product truth, change proof
status, mint canonical rig/part identity, or emit repair directives.
"""

from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

SERVICE_SCHEMA = "RealSaS.CompilerServices.ProofFailureSignatures.v1"
HISTORICAL_SOURCE = "compiler/realsas_deformation/failure_signatures.py"
HISTORICAL_SOURCE_SHA256 = "08ce49b91887a1cc39cdac340802ed924b61240da9b33efb928e8ab5d2bbbc36"


def _stable_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"PROOF_FAILURE:{sha256(encoded).hexdigest()[:24]}"


def _number(value: Any, default: float = 0.0) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float(default)


def _ratio_severity(observed: float, allowed: float) -> float:
    if allowed <= 1.0e-12: return 1.0 if observed > allowed else 0.0
    return max(0.0, min(1.0, observed / allowed - 1.0))


def _deficit_severity(observed: float, required: float) -> float:
    if required <= 1.0e-12: return 0.0
    return max(0.0, min(1.0, (required - observed) / required))


def _signature(*, proof_domain: str, failure_family: str, invariant: str, observed: Any, allowed: Any, severity01: float, localization: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    identity={"proof_domain":proof_domain,"failure_family":failure_family,"invariant":invariant,"observed":observed,"allowed":allowed,"localization":dict(localization or {})}
    return {"schema_version":SERVICE_SCHEMA,"signature_id":_stable_id(identity),"proof_domain":proof_domain,"failure_family":failure_family,"invariant":invariant,"observed_value":observed,"allowed_value":allowed,"severity01":max(0.0,min(1.0,float(severity01))),"localization":dict(localization or {}),"owner_attribution_status":"not_performed","metadata":{"promotion_origin":HISTORICAL_SOURCE,"promotion_origin_sha256":HISTORICAL_SOURCE_SHA256,"causal_owner_not_inferred_from_failure":True,"repair_not_authorized_by_signature":True,**dict(metadata or {})}}


def no_owner_attribution() -> tuple[dict[str, Any], ...]: return ()


def _mechanical(measurements):
    illegal=int(measurements.get("illegal_parent_count",0) or 0); unsupported=int(measurements.get("unsupported_joint_count",0) or 0); roots=int(measurements.get("deform_root_count",0) or 0)
    if illegal>0: yield _signature(proof_domain="MECHANICAL_STRUCTURE",failure_family="illegal_parent_reference",invariant="valid_parent_references",observed=illegal,allowed=0,severity01=min(1.0,illegal),metadata={"localization_source":"mechanical_summary"})
    if unsupported>0: yield _signature(proof_domain="MECHANICAL_STRUCTURE",failure_family="unsupported_joint",invariant="surface_supported_joints",observed=unsupported,allowed=0,severity01=min(1.0,unsupported),metadata={"localization_source":"mechanical_summary"})
    if roots<=0: yield _signature(proof_domain="MECHANICAL_STRUCTURE",failure_family="missing_deform_root",invariant="deform_root_present",observed=roots,allowed=1,severity01=1.0,metadata={"localization_source":"mechanical_summary"})


def _mesh(measurements):
    degenerate=int(measurements.get("degenerate_faces",0) or 0); min_area=_number(measurements.get("min_area"),0.0)
    if degenerate>0: yield _signature(proof_domain="MESH_QUALITY",failure_family="degenerate_triangle",invariant="nondegenerate_mesh",observed=degenerate,allowed=0,severity01=min(1.0,degenerate),metadata={"localization_source":"mesh_summary"})
    if min_area<=1e-12: yield _signature(proof_domain="MESH_QUALITY",failure_family="minimum_triangle_area_violated",invariant="positive_triangle_area",observed=min_area,allowed=1e-12,severity01=1.0,metadata={"localization_source":"mesh_summary"})


def _visual(measurements):
    direction_count=int(measurements.get("direction_count",0) or 0); corner_count=int(measurements.get("corner_binding_count",0) or 0); order=list(measurements.get("view_order",[]) or [])
    if direction_count!=8: yield _signature(proof_domain="DIRECTIONAL_VISUAL",failure_family="direction_cardinality_mismatch",invariant="exact_8_direction_renderable_set",observed=direction_count,allowed=8,severity01=min(1.0,abs(direction_count-8)/8.0))
    if order!=list(range(8)): yield _signature(proof_domain="DIRECTIONAL_VISUAL",failure_family="direction_order_mismatch",invariant="canonical_view_order",observed=order,allowed=list(range(8)),severity01=1.0)
    if corner_count<=0: yield _signature(proof_domain="DIRECTIONAL_VISUAL",failure_family="missing_appearance_corner_binding",invariant="appearance_binding_present",observed=corner_count,allowed=1,severity01=1.0)


def _motion(measurements):
    clip_count=int(measurements.get("clip_count",0) or 0); effective=int(measurements.get("effective_joint_track_count",0) or 0)
    if clip_count<=0: yield _signature(proof_domain="MOTION",failure_family="missing_motion_clip",invariant="motion_clip_present",observed=clip_count,allowed=1,severity01=1.0)
    if effective<=0: yield _signature(proof_domain="MOTION",failure_family="required_mobility_not_achieved",invariant="effective_motion_track_present",observed=effective,allowed=1,severity01=1.0,metadata={"historical_family_rebound":True})
    rich=(("max_edge_stretch_ratio","max_edge_stretch_ratio",1.55,"edge_stretch_exceeded","bounded_edge_stretch","ratio"),("max_area_change_ratio","max_area_change_ratio",2.10,"area_change_exceeded","bounded_area_change","ratio"),("flipped_triangles",None,0.0,"triangle_flip","no_triangle_flip","ratio"),("max_attachment_drift01","max_attachment_drift01",0.25,"attachment_drift_exceeded","bounded_attachment_drift","ratio"),("return_to_rest_error01","max_return_to_rest_error01",0.02,"return_to_rest_exceeded","return_to_rest","ratio"))
    thresholds=dict(measurements.get("frozen_policy_thresholds",{}) or {})
    for observed_key,threshold_key,default_limit,family,invariant,mode in rich:
        if observed_key not in measurements: continue
        observed=_number(measurements.get(observed_key),0.0); allowed=_number(thresholds.get(threshold_key,default_limit),default_limit) if threshold_key else default_limit
        if observed>allowed: yield _signature(proof_domain="MOTION",failure_family=family,invariant=invariant,observed=observed,allowed=allowed,severity01=_ratio_severity(observed,allowed) if mode=="ratio" else _deficit_severity(observed,allowed),localization=dict(measurements.get("failure_localization",{}) or {}),metadata={"historical_family_rebound":True,"full_amplitude_authority_requires_probe_contract":True})


def _deformation(measurements):
    for metric in ("rms","p95"):
        if metric not in measurements: continue
        observed=_number(measurements.get(metric),float("inf")); allowed=_number(measurements.get(f"{metric}_threshold"),1e-6)
        if observed>allowed: yield _signature(proof_domain="DEFORMATION",failure_family=f"{metric}_deformation_error_exceeded",invariant=f"bounded_{metric}_deformation_error",observed=observed,allowed=allowed,severity01=_ratio_severity(observed,allowed),metadata={"localization_source":"verified_deformation_measurement"})


def _runtime(measurements):
    representation=measurements.get("representation_class"); authority=bool(measurements.get("full_3d_reconstruction_authority")); directions=int(measurements.get("direction_count",0) or 0); motion_representation=measurements.get("motion_representation")
    if representation!="DIRECTIONAL_2D_2P5D_PUPPET": yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_representation_mismatch",invariant="directional_2d_2p5d_runtime_contract",observed=representation,allowed="DIRECTIONAL_2D_2P5D_PUPPET",severity01=1.0)
    if authority: yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_authority_escalation",invariant="runtime_is_subordinate_consumer",observed=True,allowed=False,severity01=1.0)
    if directions!=8: yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_direction_cardinality_mismatch",invariant="runtime_exact_8_directions",observed=directions,allowed=8,severity01=min(1.0,abs(directions-8)/8.0))
    if motion_representation!="DIRECTIONAL_2D_2P5D_PUPPET_MOTION": yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_motion_representation_mismatch",invariant="runtime_directional_motion_contract",observed=motion_representation,allowed="DIRECTIONAL_2D_2P5D_PUPPET_MOTION",severity01=1.0)
    bake_status=str(measurements.get("proof_owned_runtime_bake_status") or "MISSING")
    if not bake_status.startswith("PASS_"): yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_proof_owned_bake_unavailable",invariant="proof_owned_frames_no_export_replay",observed=bake_status,allowed="PASS_*",severity01=1.0)
    if measurements.get("runtime_v2_single_atlas_per_direction") is False: yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_v2_atlas_projection_incompatible",invariant="one_exact_atlas_per_direction",observed=False,allowed=True,severity01=1.0)
    if measurements.get("runtime_v2_material_uv_projection_supported") is False: yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_v2_uv_projection_unsupported",invariant="explicit_material_uv_convention",observed=False,allowed=True,severity01=1.0)
    if measurements.get("runtime_export_solver_replay") is not False: yield _signature(proof_domain="RUNTIME_CONSUMPTION",failure_family="runtime_export_replay_forbidden",invariant="export_is_projection_only",observed=measurements.get("runtime_export_solver_replay"),allowed=False,severity01=1.0)


def derive_failure_signatures(proof_domain: str, measurements: Mapping[str, Any], *, status: str) -> tuple[dict[str, Any], ...]:
    if status=="PASS": return ()
    if status not in {"FAIL","ABSTAIN"}: raise ValueError(f"unsupported proof status:{status}")
    m=dict(measurements or {})
    if status=="ABSTAIN":
        if proof_domain=="RUNTIME_CONSUMPTION":
            specialized=tuple(_runtime(m))
            if specialized: return specialized
        return (_signature(proof_domain=proof_domain,failure_family="insufficient_proof_evidence",invariant="proof_evidence_available",observed=m.get("status","MISSING"),allowed="SUFFICIENT_EVIDENCE",severity01=1.0,metadata={"abstention":True}),)
    mapper={"MECHANICAL_STRUCTURE":_mechanical,"MESH_QUALITY":_mesh,"DIRECTIONAL_VISUAL":_visual,"MOTION":_motion,"DEFORMATION":_deformation,"RUNTIME_CONSUMPTION":_runtime}.get(proof_domain)
    signatures=tuple(mapper(m)) if mapper is not None else ()
    if signatures: return signatures
    return (_signature(proof_domain=proof_domain,failure_family="proof_domain_failed_unlocalized",invariant="required_domain_pass",observed=m,allowed="PASS",severity01=1.0,metadata={"localization_pending":True}),)
