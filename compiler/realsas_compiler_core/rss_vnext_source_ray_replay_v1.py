from __future__ import annotations

"""CPU source-ray replay against exact original two-view RGBA and cameras.

Separate image-ray sampling from depth, color ownership, layer splitting, and
mechanical control. This is a source-integrity court, not a texture/mesh court.
The producer uses torch.grid_sample(bilinear, align_corners=False,
padding_mode=zeros) on TESSA rest XYZ; compiler recomputes independently.
"""

from dataclasses import asdict, dataclass, replace
import math
import numpy as np

from .hashing import content_sha256
from .rss_vnext_source_observation_v1 import RSSvNextTwoViewSourceObservationV1
from .rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1,
)
from .rss_vnext_source_evidence_binding_v1 import (
    RSSvNextSourceEvidenceBindingCourtV1,
    validate_rss_vnext_source_evidence_binding_receipt_v1,
)
from .types import QualificationError

SCHEMA = "RealSaS.RSSvNext.SourceRayReplayCourt.v1"
_MAX_VERTICES = 250_000
_ABS_TOLERANCE = 0.001
_BOUNDARY_NDC_EPS = 2e-6


def _reject(reason: str) -> None:
    raise QualificationError("RSS_VNEXT_RAY_REPLAY_" + reason)


@dataclass(frozen=True)
class SourceRayReplayCourtV1:
    source_binding_sha256: str
    inference_evidence_hash: str
    source_identity_court_sha256: str
    state: str
    checked_ray_count: int
    boundary_ambiguous_count: int
    max_channel_error: float
    court_sha256: str
    schema: str = SCHEMA
    source_surface_ownership: bool = False
    source_depth_or_visibility: bool = False
    material_or_drawable_authority: bool = False
    canonical_mesh_authority: bool = False


def _seal(value: SourceRayReplayCourtV1) -> SourceRayReplayCourtV1:
    payload=asdict(value)
    payload.pop("court_sha256")
    return replace(value,court_sha256=content_sha256(payload))


def _bilinear(raw: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Match align_corners=False and zero padding, no learned appearance."""
    _,h,w=raw.shape
    x=(u+np.float32(1.))*np.float32(w/2.)-np.float32(.5)
    y=(v+np.float32(1.))*np.float32(h/2.)-np.float32(.5)
    x0=np.floor(x).astype(np.int64)
    y0=np.floor(y).astype(np.int64)
    fx=x-x0
    fy=y-y0
    result=np.zeros((len(x),4),dtype=np.float32)
    for dx in (0,1):
        ix=x0+dx
        wx=fx if dx else (1.-fx)
        for dy in (0,1):
            iy=y0+dy
            wy=fy if dy else (1.-fy)
            use=(ix>=0)&(iy>=0)&(ix<w)&(iy<h)
            if not np.any(use):
                continue
            sample=raw[:,iy[use],ix[use]].T.astype(np.float32)/np.float32(255.)
            result[use]+=sample*(wx[use]*wy[use])[:,None]
    return result


def replay_original_source_rays_v1(
    *, source: RSSvNextTwoViewSourceObservationV1,
    evidence: RSSvNextInferenceEvidenceIRV1,
    source_identity_court: RSSvNextSourceEvidenceBindingCourtV1,
) -> SourceRayReplayCourtV1:
    if type(source) is not RSSvNextTwoViewSourceObservationV1:
        _reject("SOURCE_TYPE_INVALID")
    validate_inference_evidence_v1(evidence)
    validate_rss_vnext_source_evidence_binding_receipt_v1(source_identity_court)
    if (
        source_identity_court.source_binding_sha256 != source.source_binding_sha256
        or source_identity_court.inference_evidence_hash != evidence.evidence_hash
        or source_identity_court.checkpoint_sha256 != evidence.checkpoint_sha256
    ):
        _reject("PARENT_LINEAGE_MISMATCH")
    parent=dict(
        source_binding_sha256=source.source_binding_sha256,
        inference_evidence_hash=evidence.evidence_hash,
        source_identity_court_sha256=source_identity_court.court_hash,
        court_sha256="",
    )
    if evidence.mesh is None:
        return _seal(SourceRayReplayCourtV1(
            **parent,state="ABSTAIN_NO_TOPOLOGY",checked_ray_count=0,
            boundary_ambiguous_count=0,max_channel_error=0.,
        ))
    mesh=evidence.mesh
    rest=np.asarray(mesh["rest_xyz"],dtype=np.float32)
    n=len(rest)
    if not np.isfinite(rest).all():
        _reject("REST_FLOAT32_NOT_REPRESENTABLE")
    if n>_MAX_VERTICES:
        return _seal(SourceRayReplayCourtV1(
            **parent,state="ABSTAIN_VERTEX_BUDGET",checked_ray_count=0,
            boundary_ambiguous_count=0,max_channel_error=0.,
        ))
    ray=mesh["source_rays"]
    maxerr=0.
    checked=0
    ambiguous=0
    for view in (0,1):
        cam=source.camera_features[view]
        for start in range(0,n,8192):
            end=min(n,start+8192)
            d=rest[start:end]-cam[None,0:3]
            u=np.sum(d*cam[None,3:6],axis=1,dtype=np.float32)/cam[12]
            v=-np.sum(d*cam[None,6:9],axis=1,dtype=np.float32)/cam[12]
            valid=(np.abs(u)<=1.)&(np.abs(v)<=1.)
            near=(np.abs(np.abs(u)-1.)<=_BOUNDARY_NDC_EPS)|(
                np.abs(np.abs(v)-1.)<=_BOUNDARY_NDC_EPS
            )
            is_valid=np.asarray([row[view] for row in ray["valid"][start:end]],dtype=bool)
            if np.any((valid!=is_valid)&~near):
                _reject("PROJECTION_VALID_MASK_DRIFT")
            # torch's producer masks all off-screen rays to zero. Avoid
            # computing floor/casting huge off-screen coordinates to int64;
            # those values were never observable image pixels.
            safe_u=np.where(valid,u,np.float32(2.)).astype(np.float32)
            safe_v=np.where(valid,v,np.float32(2.)).astype(np.float32)
            rgba=_bilinear(source.native_rgba[view],safe_u,safe_v)
            rgba[~valid]=0.
            stored=np.asarray([row[view] for row in ray["rgba"][start:end]],dtype=np.float32)
            compare=~near
            ambiguous+=int(np.count_nonzero(near))
            if not np.any(compare):
                continue
            error=float(np.max(np.abs(rgba[compare]-stored[compare])))
            if not math.isfinite(error) or error>_ABS_TOLERANCE:
                _reject("ORIGINAL_PIXEL_SAMPLE_DRIFT")
            checked+=int(np.count_nonzero(compare))
            maxerr=max(maxerr,error)
    return _seal(SourceRayReplayCourtV1(
        **parent,
        state="REPLAY_WITH_NDC_EDGE_UNCERTAINTY" if ambiguous else "REPLAY_VERIFIED_NO_SURFACE_AUTHORITY",
        checked_ray_count=checked,boundary_ambiguous_count=ambiguous,
        max_channel_error=maxerr,
    ))


def validate_source_ray_replay_court_v1(
    value: SourceRayReplayCourtV1,
    *,
    source: RSSvNextTwoViewSourceObservationV1,
    evidence: RSSvNextInferenceEvidenceIRV1,
    source_identity_court: RSSvNextSourceEvidenceBindingCourtV1,
) -> None:
    """Read-side receipt validation, never a source-surface authorization.

    This checks exact authority, parent identity, counters, state, and digest.
    Independent consumers may additionally replay original pixels; a CAS hash
    or replay receipt alone is not permission to infer source depth.
    """
    if (
        type(value) is not SourceRayReplayCourtV1
        or value.schema != SCHEMA
        or type(value.canonical_mesh_authority) is not bool
        or any((
            value.source_surface_ownership,
            value.source_depth_or_visibility,
            value.material_or_drawable_authority,
            value.canonical_mesh_authority,
        ))
        or not all(
            type(x) is bool
            for x in (
                value.source_surface_ownership,
                value.source_depth_or_visibility,
                value.material_or_drawable_authority,
                value.canonical_mesh_authority,
            )
        )
        or type(value.checked_ray_count) is not int
        or type(value.boundary_ambiguous_count) is not int
        or value.checked_ray_count < 0 or value.boundary_ambiguous_count < 0
        or type(value.max_channel_error) not in (int,float)
        or not math.isfinite(value.max_channel_error)
        or value.max_channel_error < 0 or value.max_channel_error > _ABS_TOLERANCE
    ):
        _reject("RECEIPT_AUTHORITY_OR_DOMAIN_DRIFT")
    validate_rss_vnext_source_evidence_binding_receipt_v1(source_identity_court)
    validate_inference_evidence_v1(evidence)
    if (
        type(source) is not RSSvNextTwoViewSourceObservationV1
        or value.source_binding_sha256 != source.source_binding_sha256
        or value.inference_evidence_hash != evidence.evidence_hash
        or value.source_identity_court_sha256 != source_identity_court.court_hash
        or source_identity_court.source_binding_sha256 != source.source_binding_sha256
        or source_identity_court.inference_evidence_hash != evidence.evidence_hash
    ):
        _reject("RECEIPT_PARENT_DRIFT")
    if evidence.mesh_status == "ABSTAIN":
        if (
            value.state != "ABSTAIN_NO_TOPOLOGY"
            or value.checked_ray_count != 0 or value.boundary_ambiguous_count != 0
            or value.max_channel_error != 0.
        ):
            _reject("RECEIPT_ABSTAIN_DRIFT")
    else:
        vertex_count=len(evidence.mesh["rest_xyz"])
        if value.state == "ABSTAIN_VERTEX_BUDGET":
            if (
                vertex_count <= _MAX_VERTICES
                or value.checked_ray_count != 0 or value.boundary_ambiguous_count != 0
                or value.max_channel_error != 0.
            ):
                _reject("RECEIPT_VERTEX_BUDGET_DRIFT")
        elif (
            value.state not in (
                "REPLAY_WITH_NDC_EDGE_UNCERTAINTY",
                "REPLAY_VERIFIED_NO_SURFACE_AUTHORITY",
            )
            or value.checked_ray_count+value.boundary_ambiguous_count != 2*vertex_count
            or bool(value.boundary_ambiguous_count) != (
                value.state == "REPLAY_WITH_NDC_EDGE_UNCERTAINTY"
            )
        ):
            _reject("RECEIPT_COVERAGE_OR_STATE_DRIFT")
    fields=asdict(value)
    claimed=fields.pop("court_sha256")
    if claimed != content_sha256(fields):
        _reject("RECEIPT_HASH_DRIFT")


__all__ = [
    "SourceRayReplayCourtV1","replay_original_source_rays_v1",
    "validate_source_ray_replay_court_v1",
]
