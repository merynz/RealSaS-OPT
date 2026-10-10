"""Read sealed canonical CAA evidence without mistaking it for visual support.

A canonical surface colour is not an amodal visual layer. This inspection is
used by the real Stage42 consumer and the sealed-input preflight. It deliberately
does not manufacture a mapping, a grip, a contact denominator or drawing order.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .hashing import content_sha256
from .types import QualificationError


def inspect_visual_completion_field(asset):
    metadata = dict(asset.metadata or {})
    if metadata.get("canonical_completion_field_bound") is not True:
        return {"field_present": False, "field_sha256": None,
                "validation_read": False, "transport_sample_count": 0}
    path = Path(str(metadata.get("canonical_completion_field_path") or ""))
    expected = str(metadata.get("canonical_completion_field_sha256") or "")
    if (metadata.get("canonical_completion_field_schema") != "RealSaS.CAACompileArrays.v3"
            or metadata.get("canonical_completion_support_domain") != "CANONICAL_SURFACE_ADDRESSING"
            or len(expected) != 64 or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise QualificationError("VISUAL_COMPLETION_FIELD_BINDING_OR_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        required = {"sample_positions", "sample_face_index", "rgba", "provenance", "source_view"}
        if not required.issubset(data.files):
            raise QualificationError("VISUAL_COMPLETION_FIELD_ARRAYS_MISSING")
        positions, faces = data["sample_positions"], data["sample_face_index"]
        rgba, provenance, donors = data["rgba"], data["provenance"], data["source_view"]
        samples = len(positions)
        if (not samples or positions.shape != (samples, 3)
                or not np.isfinite(positions).all() or faces.shape != (samples,)
                or faces.dtype.kind not in "iu" or np.any(faces < 0)
                or rgba.shape != (8, samples, 4) or rgba.dtype != np.uint8
                or provenance.shape != (8, samples) or provenance.dtype != np.uint8
                or donors.shape != provenance.shape or donors.dtype.kind != "i"
                or not np.isin(provenance, [0, 1, 2, 3, 4]).all()):
            raise QualificationError("VISUAL_COMPLETION_FIELD_ARRAY_CONTRACT_INVALID")
        view = np.arange(8)[:, None]
        valid = ((provenance == 0) & (donors == view))
        valid |= (provenance == 1) & (donors >= 0) & (donors < 8) & (donors != view)
        valid |= (provenance == 2) & (donors == -2)
        valid |= (provenance == 3) & (donors == -4)
        valid |= (provenance == 4) & (donors == -3)
        if not valid.all() or np.any(rgba[provenance == 3] != 0):
            raise QualificationError("VISUAL_COMPLETION_FIELD_PROVENANCE_DRIFT")
        counts = {str(code): int(np.count_nonzero(provenance == code)) for code in range(5)}
        per_view = [{str(code): int(np.count_nonzero(row == code)) for code in range(5)}
                    for row in provenance]
    return {"field_present": True, "field_sha256": expected,
            "compile_hash": str(metadata.get("canonical_completion_field_compile_hash") or ""),
            "validation_read": True, "canonical_sample_count": samples,
            "direction_sample_count": 8 * samples,
            "provenance_counts": counts, "provenance_counts_by_view": per_view,
            "support_domain": "CANONICAL_SURFACE_ADDRESSING",
            "visual_layer_addressing_produced": False,
            "hidden_visual_support_qualified": False,
            "transport_sample_count": 0,
            "validation_is_material_transport": False}


def source_visual_boundary_evidence(*, asset, appearance, presentation, completion):
    """Inventory the CURRENT implemented path, never accept future metadata flags.

    The missing relation domain has unknown cardinality, not zero. This module
    must change when an actual typed producer/consumer is implemented; setting
    `qualified=true` on an input cannot give the current operators that ability.
    """
    rows = [
        {"relation": "visual_contact", "producer_stage": "37_QUALIFIED_PRESENTATION_STRUCTURE",
         "consumer_stage": "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
         "status": "MISSING_RELATION_AUTHORITY", "required_relation_count": None,
         "checked_relation_count": 0,
         "missing": ["REQUIRED_CONTACT_DOMAIN", "CANONICAL_CONTACT_ADDRESSES", "MOTION_CONTACT_PROOF"]},
        {"relation": "equipment_grip", "producer_stage": "37_QUALIFIED_PRESENTATION_STRUCTURE",
         "consumer_stage": "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
         "status": "MISSING_RELATION_AUTHORITY", "required_relation_count": None,
         "checked_relation_count": 0,
         "missing": ["TARGET_ATTACHMENT_OWNERS", "TARGET_GRIP_SLOT_AND_LOCAL_ADDRESS", "GRIP_RESIDUAL_PROOF"]},
        {"relation": "hidden_material", "producer_stage": "23_COMPLETE_APPEARANCE_ASSET_BAKED",
         "consumer_stage": "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
         "status": ("PRODUCED_CANONICAL_FIELD_WITHOUT_VISUAL_TRANSPORT"
                    if completion["field_present"] else "MISSING_CANONICAL_COMPLETION_FIELD"),
         "canonical_completion_field": completion,
         "missing": ["HIDDEN_VISUAL_LAYER_SUPPORT", "VISUAL_LAYER_TO_CANONICAL_SURFACE_ADDRESSES",
                     "QUALIFIED_COMPLETION_MATERIAL_TRANSPORT", "DYNAMIC_EXPOSURE_COVERAGE_PROOF"]},
        {"relation": "semantic_order", "producer_stage": "37_QUALIFIED_PRESENTATION_STRUCTURE",
         "consumer_stage": "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
         "status": "MISSING_RELATION_AUTHORITY", "required_relation_count": None,
         "checked_relation_count": 0,
         "missing": ["VISUAL_PART_AND_LAYER_OWNERS", "INTENDED_OCCLUSION_RELATIONS", "MOTION_ORDER_PROOF"]},
    ]
    result = {"schema": "RealSaS.SourceVisualBoundaryEvidence.v1",
              "appearance_asset_hash": asset.asset_hash,
              "appearance_qualification_hash": appearance.qualification_hash,
              "qualified_visual_presentation_hash": presentation.set_hash if presentation else None,
              "scope": "CURRENT_SOURCE_RASTER_TRANSPORT__NOT_FULL_VISUAL_ACCEPTANCE",
              "relationships": rows, "full_visual_acceptance_passed": False,
              "missing_relation_domain_is_empty": False,
              "render_authorized_as_product": False}
    result["evidence_hash"] = content_sha256(result)
    return result


def validate_source_visual_boundary_evidence(evidence):
    payload = dict(evidence)
    sealed = payload.pop("evidence_hash", None)
    if (payload.get("schema") != "RealSaS.SourceVisualBoundaryEvidence.v1"
            or sealed != content_sha256(payload)):
        raise QualificationError("SOURCE_VISUAL_BOUNDARY_EVIDENCE_HASH_DRIFT")
    # The current implementation has no full-relation producer or consumer.
    # Even a correctly resealed optimistic flag must fail, not unlock closure.
    if (payload.get("full_visual_acceptance_passed") is not False
            or payload.get("missing_relation_domain_is_empty") is not False
            or payload.get("render_authorized_as_product") is not False):
        raise QualificationError("SOURCE_VISUAL_BOUNDARY_UNIMPLEMENTED_CAPABILITY_CLAIM")
    rows = payload.get("relationships")
    expected = {"visual_contact", "equipment_grip", "hidden_material", "semantic_order"}
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or {row.get("relation") for row in rows} != expected
            or any(not row.get("missing") or not row.get("producer_stage")
                   or row.get("consumer_stage") != "42_RUNTIME_PROJECTION_AND_CAA_BINDING"
                   for row in rows)):
        raise QualificationError("SOURCE_VISUAL_BOUNDARY_REQUIRED_RELATION_DOMAIN_MISSING")
    return payload
