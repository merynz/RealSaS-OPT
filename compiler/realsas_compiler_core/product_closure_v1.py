from __future__ import annotations

"""Stage-40 product closure seal.

PRODUCT_PASS is minted only after exact rest preservation, canonical dynamic proof,
compact runtime materialization, native package playback, and native visible artist
motion have all passed and their lineages still agree.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .motion_dynamic_proof_v1 import qualified_dynamic_motion_hash
from .rest_preservation_policy_v1 import qualified_rest_preservation_hash
from .runtime_native_proof_v1 import native_playback_hash, visual_motion_hash
from .runtime_package_v1 import runtime_v4_package_seal_hash
from .runtime_projection_v1 import runtime_v4_projection_hash
from .types import QualificationError

Json=dict[str,Any]


@dataclass(frozen=True)
class ProductClosureSealIR:
    product_state_binding_hash:str
    rest_preservation_binding_hash:str
    dynamic_motion_binding_hash:str
    runtime_projection_binding_hash:str
    runtime_package_binding_hash:str
    native_playback_binding_hash:str
    visual_motion_binding_hash:str
    authoring_bundle_binding_hash:str
    professional_motion_clip_ids:tuple[str,...]
    qualification_report:Json
    product_closure_hash:str
    schema_version:str="RealSaS.ProductClosureSealIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def product_closure_hash(value:ProductClosureSealIR)->str:
    payload=value.to_dict(); payload.pop("product_closure_hash",None)
    return content_sha256(payload)


def build_product_closure_seal(
    *,product_state_hash:str,rest,dynamic,projection,package,native,visual,authoring,
)->ProductClosureSealIR:
    if rest.preservation_lineage_hash!=qualified_rest_preservation_hash(rest):
        raise QualificationError("PRODUCT_CLOSURE_REST_HASH_DRIFT")
    if dynamic.dynamic_motion_hash!=qualified_dynamic_motion_hash(dynamic):
        raise QualificationError("PRODUCT_CLOSURE_DYNAMIC_HASH_DRIFT")
    if projection.projection_hash!=runtime_v4_projection_hash(projection):
        raise QualificationError("PRODUCT_CLOSURE_PROJECTION_HASH_DRIFT")
    if package.package_seal_hash!=runtime_v4_package_seal_hash(package):
        raise QualificationError("PRODUCT_CLOSURE_PACKAGE_HASH_DRIFT")
    if native.native_playback_hash!=native_playback_hash(native):
        raise QualificationError("PRODUCT_CLOSURE_NATIVE_HASH_DRIFT")
    if visual.visual_motion_hash!=visual_motion_hash(visual):
        raise QualificationError("PRODUCT_CLOSURE_VISUAL_HASH_DRIFT")
    from .authoring_bundle_v1 import editable_puppet_bundle_hash
    if authoring.authoring_bundle_hash!=editable_puppet_bundle_hash(authoring):
        raise QualificationError("PRODUCT_CLOSURE_AUTHORING_HASH_DRIFT")
    if authoring.qualification_report.get("status")!="PASS_EDITABLE_AUTHORING_BUNDLE":
        raise QualificationError("PRODUCT_CLOSURE_AUTHORING_NOT_PASS")
    if authoring.product_state_binding_hash!=str(product_state_hash):
        raise QualificationError("PRODUCT_CLOSURE_AUTHORING_PRODUCT_DRIFT")
    if not bool(rest.qualification_report.get("every_view_passed_every_rule",False)):
        raise QualificationError("PRODUCT_CLOSURE_REST_NOT_PASS")
    if not bool(rest.qualification_report.get("motion_authorization_precondition_satisfied",False)):
        raise QualificationError("PRODUCT_CLOSURE_MOTION_NOT_AUTHORIZED")
    if not bool(dynamic.qualification_report.get("dynamic_proof_passed",False)):
        raise QualificationError("PRODUCT_CLOSURE_DYNAMIC_NOT_PASS")
    if projection.product_state_binding_hash!=str(product_state_hash):
        raise QualificationError("PRODUCT_CLOSURE_PROJECTION_PRODUCT_DRIFT")
    if package.product_state_binding_hash!=str(product_state_hash):
        raise QualificationError("PRODUCT_CLOSURE_PACKAGE_PRODUCT_DRIFT")
    if package.projection_binding_hash!=projection.projection_hash:
        raise QualificationError("PRODUCT_CLOSURE_PACKAGE_PROJECTION_DRIFT")
    if package.dynamic_motion_binding_hash!=dynamic.dynamic_motion_hash:
        raise QualificationError("PRODUCT_CLOSURE_PACKAGE_DYNAMIC_DRIFT")
    if native.package_seal_binding_hash!=package.package_seal_hash or native.projection_binding_hash!=projection.projection_hash:
        raise QualificationError("PRODUCT_CLOSURE_NATIVE_BINDING_DRIFT")
    if visual.package_seal_binding_hash!=package.package_seal_hash or visual.projection_binding_hash!=projection.projection_hash:
        raise QualificationError("PRODUCT_CLOSURE_VISUAL_BINDING_DRIFT")
    if visual.native_playback_binding_hash!=native.native_playback_hash:
        raise QualificationError("PRODUCT_CLOSURE_VISUAL_NATIVE_DRIFT")
    if native.qualification_report.get("status")!="PASS_NATIVE_PACKAGE_OPEN_PLAYBACK":
        raise QualificationError("PRODUCT_CLOSURE_NATIVE_NOT_PASS")
    if visual.qualification_report.get("status")!="PASS_NATIVE_VISUAL_BAKE":
        raise QualificationError("PRODUCT_CLOSURE_VISUAL_NOT_PASS")

    professional=tuple(sorted(c.clip_id for c in dynamic.clips if c.professional_motion_evidence))
    if not professional:
        raise QualificationError("PRODUCT_CLOSURE_REQUIRES_NONZERO_ARTIST_MOTION")
    visible=set(visual.professional_visible_motion_clip_ids)
    missing=tuple(cid for cid in professional if cid not in visible)
    if missing:
        raise QualificationError(f"PRODUCT_CLOSURE_ARTIST_MOTION_NOT_VISIBLY_PROVED:{missing}")

    value=ProductClosureSealIR(
        str(product_state_hash),rest.preservation_lineage_hash,dynamic.dynamic_motion_hash,
        projection.projection_hash,package.package_seal_hash,native.native_playback_hash,
        visual.visual_motion_hash,authoring.authoring_bundle_hash,professional,
        {
            "status":"PRODUCT_PASS",
            "product_pass":True,
            "rest_source_preservation_passed":True,
            "canonical_dynamic_motion_passed":True,
            "professional_artist_motion_present":True,
            "runtime_v4_materialized":True,
            "native_package_open_playback_passed":True,
            "native_visible_artist_motion_passed":True,
            "editable_authoring_export_passed":True,
            "json_only_product_pass_forbidden":True,
            "founder_visual_pass_claimed":False,
        },"",
        metadata={
            "single_canonical_3d_mesh_truth":True,
            "completion_used":False,
            "mechanical_probe_alone_cannot_mint_product_pass":True,
            "closure_authority":"EXACT_STAGE32_35_36_37_38_39_PLUS_EDITABLE_AUTHORING_LINEAGE",
        },
    )
    return replace(value,product_closure_hash=product_closure_hash(value))


def product_closure_from_dict(payload:Mapping[str,Any])->ProductClosureSealIR:
    if str(payload.get("schema_version") or payload.get("schema") or "")!="RealSaS.ProductClosureSealIR.v1":
        raise ValueError("PRODUCT_CLOSURE_SCHEMA_MISMATCH")
    value=ProductClosureSealIR(
        str(payload["product_state_binding_hash"]),str(payload["rest_preservation_binding_hash"]),
        str(payload["dynamic_motion_binding_hash"]),str(payload["runtime_projection_binding_hash"]),
        str(payload["runtime_package_binding_hash"]),str(payload["native_playback_binding_hash"]),
        str(payload["visual_motion_binding_hash"]),str(payload["authoring_bundle_binding_hash"]),
        tuple(map(str,payload.get("professional_motion_clip_ids") or ())),
        dict(payload.get("qualification_report") or {}),str(payload["product_closure_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.ProductClosureSealIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.product_closure_hash!=product_closure_hash(value):
        raise ValueError("PRODUCT_CLOSURE_HASH_MISMATCH")
    return value
