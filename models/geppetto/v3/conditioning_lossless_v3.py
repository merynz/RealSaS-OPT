from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable

import numpy as np

from compiler.realsas_compiler_core.substrate.validation import (
    rigging_surface_boundary_audit_hash_v1,
    rigging_surface_topology_fingerprint_v1,
    validate_rigging_surface_ir_v1,
)
from models.geppetto.v2.geppetto_conditioning_v2 import (
    FEATURE_CONTRACT_V2,
    GeppettoConditioningAdapterV2,
    GeometryNormalizationV2,
)


SCHEMA_V3 = "RealSaS.GeppettoLosslessConditioningBatch.v3"
PRODUCT_CONDITIONING_AUTHORITY_V3 = "PRODUCT_VALIDATED_LOSSLESS_RIGGING_EVIDENCE_V3"
LOSSLESS_EVIDENCE_CONTRACT_V1 = "RIGGING_SURFACE_IR_FULL_PAYLOAD_PLUS_VIEW_TENSORS_V1"
LEGACY_SUMMARY_AUTHORITY_V2 = "PRODUCT_VALIDATED_SCENE_FIRST_SIGNED_V1"
AUTHORITY_REVOCATION_ID_V1 = "REALSAS_20260906_CAUSAL_REPAIR_233ec3bcd77e0f02"
AUTHORITY_REVOCATION_REASON_V1 = (
    "New causal-repair evidence showed that the 24D summary-only seam cannot remain "
    "the sole product-authoritative evidence boundary. RiggingSurfaceIR already "
    "contains view-indexed raster/support and typed topology; product conditioning "
    "must preserve that source evidence without lossy aggregation. The historical "
    "24D tensor remains replayable only as an explicitly derived compatibility view."
)


def _sha(payload: object) -> str:
    def norm(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): norm(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (tuple, list)):
            return [norm(x) for x in v]
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    return sha256(
        json.dumps(norm(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalized_raster_xy(surface, xy) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.shape != (2,) or not np.isfinite(arr).all():
        raise ValueError("raster binding must be finite Vec2")
    mode = str(surface.metadata.get("raster_coordinate_system", ""))
    if mode == "PIXEL_CENTER_XY":
        resolution = int(surface.metadata.get("resolution", 0))
        if resolution <= 0:
            raise ValueError("PIXEL_CENTER_XY requires positive resolution")
        out = np.asarray(
            [
                2.0 * (arr[0] + 0.5) / float(resolution) - 1.0,
                2.0 * (arr[1] + 0.5) / float(resolution) - 1.0,
            ],
            dtype=np.float32,
        )
    elif mode in {"GRID_XY", "GRID_SAMPLE_XY", "NORMALIZED_GRID_XY"}:
        out = arr.astype(np.float32)
    else:
        raise ValueError("explicit raster_coordinate_system required")
    if not np.isfinite(out).all():
        raise ValueError("normalized raster binding is non-finite")
    return out


def _lossless_view_tensors(surface, ordered_nodes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(ordered_nodes)
    raster_xy = np.zeros((n, 8, 2), dtype=np.float32)
    raster_valid = np.zeros((n, 8), dtype=bool)
    support = np.zeros((n, 8), dtype=bool)

    for i, node in enumerate(ordered_nodes):
        support_views = {int(v) for v in node.support_views}
        if any(v < 0 or v > 7 for v in support_views):
            raise ValueError("support view outside 0..7")
        for v in support_views:
            support[i, v] = True

        seen = set()
        for view, xy in node.raster_bindings:
            v = int(view)
            if v < 0 or v > 7 or v in seen:
                raise ValueError("invalid or duplicate raster view")
            seen.add(v)
            raster_xy[i, v] = _normalized_raster_xy(surface, xy)
            raster_valid[i, v] = True

        if seen != support_views:
            raise ValueError("raster/support view mismatch")

    return raster_xy, raster_valid, support


def _pad(items: list[np.ndarray], trailing: tuple[int, ...], dtype) -> np.ndarray:
    max_nodes = max(len(x) for x in items)
    out = np.zeros((len(items), max_nodes, *trailing), dtype=dtype)
    for b, x in enumerate(items):
        if x.shape[1:] != trailing:
            raise ValueError("lossless evidence tensor shape mismatch")
        out[b, : len(x)] = x
    return out


@dataclass(frozen=True)
class GeppettoLosslessConditioningBatchV3:
    surface_ids: tuple[tuple[str, ...], ...]
    positions_normalized: np.ndarray
    valid_mask: np.ndarray
    normalizations: tuple[GeometryNormalizationV2, ...]
    raster_xy_by_view: np.ndarray
    raster_valid_by_view: np.ndarray
    support_by_view: np.ndarray
    local_relation_pairs: tuple[tuple[tuple[int, int], ...], ...]
    raw_surface_payloads: tuple[dict[str, Any], ...]
    raw_surface_payload_hashes: tuple[str, ...]
    source_surface_hashes: tuple[str, ...]
    surface_boundary_audit_hashes: tuple[str, ...]
    local_relation_hashes: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    derived_features_24d: np.ndarray
    derived_feature_contract: tuple[str, ...] = FEATURE_CONTRACT_V2
    derived_feature_authority: bool = False
    conditioning_authority: str = PRODUCT_CONDITIONING_AUTHORITY_V3
    evidence_contract: str = LOSSLESS_EVIDENCE_CONTRACT_V1
    authority_revocation_id: str = AUTHORITY_REVOCATION_ID_V1
    authority_revocation_reason: str = AUTHORITY_REVOCATION_REASON_V1
    schema_version: str = SCHEMA_V3

    def as_legacy_v2_diagnostic(self):
        """Explicit opt-in only; never grants product authority."""
        raise RuntimeError(
            "V3 product evidence may not be silently collapsed back to the V2 "
            "summary-only product seam. Use the historical V2 adapter directly "
            "for replay/diagnostics."
        )

    def product_certificate(self, sample_index: int) -> dict[str, Any]:
        i = int(sample_index)
        if i < 0 or i >= len(self.surface_ids):
            raise IndexError("conditioning sample index out of range")
        payload = {
            "schema": "RealSaS.GeppettoLosslessProductConditioningCertificate.v3",
            "conditioning_authority": self.conditioning_authority,
            "evidence_contract": self.evidence_contract,
            "source_surface_hash": self.source_surface_hashes[i],
            "raw_surface_payload_hash": self.raw_surface_payload_hashes[i],
            "surface_boundary_audit_hash": self.surface_boundary_audit_hashes[i],
            "local_relation_hash": self.local_relation_hashes[i],
            "evidence_hash": self.evidence_hashes[i],
            "derived_feature_contract": self.derived_feature_contract,
            "derived_feature_authority": False,
            "revoked_legacy_summary_authority": LEGACY_SUMMARY_AUTHORITY_V2,
            "authority_revocation_id": self.authority_revocation_id,
            "authority_revocation_reason": self.authority_revocation_reason,
        }
        return {**payload, "certificate_sha256": _sha(payload)}


class GeppettoProductConditioningAdapterV3:
    """Current product seam: preserve RiggingSurfaceIR before learner-specific views."""

    product_boundary_authority = True
    lossy_summary_authority = False
    evidence_contract = LOSSLESS_EVIDENCE_CONTRACT_V1

    def __call__(self, surfaces: Iterable) -> GeppettoLosslessConditioningBatchV3:
        surfaces = tuple(surfaces)
        if not surfaces:
            raise ValueError("at least one surface required")

        # Historical 24D derivation remains available only as a compatibility tensor.
        # It is deliberately built through the diagnostic adapter, not the revoked
        # product-authority path.
        derived = GeppettoConditioningAdapterV2()(surfaces)

        raster_items = []
        raster_valid_items = []
        support_items = []
        raw_payloads = []
        raw_hashes = []
        boundary_hashes = []
        relation_hashes = []
        evidence_hashes = []

        for b, surface in enumerate(surfaces):
            audit = validate_rigging_surface_ir_v1(
                surface,
                require_scene_first_signed_contract=True,
            )
            boundary_hash = rigging_surface_boundary_audit_hash_v1(audit)
            relation_hash = rigging_surface_topology_fingerprint_v1(surface)

            ordered_nodes = tuple(sorted(surface.surface_nodes, key=lambda n: n.surface_id))
            ids = tuple(n.surface_id for n in ordered_nodes)
            if ids != derived.surface_ids[b]:
                raise ValueError("canonical surface row order drift between V2 and V3")

            raster_xy, raster_valid, support = _lossless_view_tensors(
                surface, ordered_nodes
            )
            raw_payload = surface.to_dict()
            raw_hash = _sha(raw_payload)
            evidence_hash = _sha(
                {
                    "contract": LOSSLESS_EVIDENCE_CONTRACT_V1,
                    "raw_surface_payload_hash": raw_hash,
                    "raster_xy_by_view": raster_xy,
                    "raster_valid_by_view": raster_valid,
                    "support_by_view": support,
                    "local_relation_hash": relation_hash,
                }
            )

            raster_items.append(raster_xy)
            raster_valid_items.append(raster_valid)
            support_items.append(support)
            raw_payloads.append(raw_payload)
            raw_hashes.append(raw_hash)
            boundary_hashes.append(boundary_hash)
            relation_hashes.append(relation_hash)
            evidence_hashes.append(evidence_hash)

        return GeppettoLosslessConditioningBatchV3(
            surface_ids=derived.surface_ids,
            positions_normalized=derived.positions_normalized,
            valid_mask=derived.valid_mask,
            normalizations=derived.normalizations,
            raster_xy_by_view=_pad(raster_items, (8, 2), np.float32),
            raster_valid_by_view=_pad(raster_valid_items, (8,), bool),
            support_by_view=_pad(support_items, (8,), bool),
            local_relation_pairs=derived.local_relation_pairs,
            raw_surface_payloads=tuple(raw_payloads),
            raw_surface_payload_hashes=tuple(raw_hashes),
            source_surface_hashes=derived.source_surface_hashes,
            surface_boundary_audit_hashes=tuple(boundary_hashes),
            local_relation_hashes=tuple(relation_hashes),
            evidence_hashes=tuple(evidence_hashes),
            derived_features_24d=derived.features,
        )


__all__ = [
    "GeppettoLosslessConditioningBatchV3",
    "GeppettoProductConditioningAdapterV3",
    "PRODUCT_CONDITIONING_AUTHORITY_V3",
    "LOSSLESS_EVIDENCE_CONTRACT_V1",
    "LEGACY_SUMMARY_AUTHORITY_V2",
    "AUTHORITY_REVOCATION_ID_V1",
    "AUTHORITY_REVOCATION_REASON_V1",
]
