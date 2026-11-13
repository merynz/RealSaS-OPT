"""Per-subject source-equivalence court for PresentationResponseTeacherV1.

This is the current-observation-rest counterpart of the sealed historical sparse
source-ray court.  It compares two packed evaluator measurements and two fresh
admitted-source material references on the exact same face/barycentric ray domain.
The result is still a candidate: student observability/leakage and post-M binding
remain separate gates.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

import numpy as np

from models.iris.truth.qualification_v1 import FamilyQualificationInputV1, qualify_family


PASS = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _packed_core(value: Mapping[str, object]) -> dict:
    return {
        key: value.get(key)
        for key in (
            "schema",
            "subject_id",
            "teacher_scene_sha256",
            "geometry_npz_sha256",
            "candidate_sha256",
            "camera_bundle_sha256",
            "packed_source_scene_blend_sha256",
            "sampling_contract",
            "rgb_carrier_calibration",
            "material_policy_replay",
            "selected_ray_count",
            "pass_ray_count",
            "fail_ray_count",
            "abstain_ray_count",
            "rays",
            "authority",
            "court_status",
        )
    }


def _source_core(value: Mapping[str, object]) -> dict:
    return {
        key: value.get(key)
        for key in (
            "schema",
            "subject_id",
            "source_sha256",
            "candidate_sha256",
            "camera_bundle_sha256",
            "appearance_sha256",
            "geometry_npz_sha256",
            "source_material_transform",
            "rgb_carrier_calibration",
            "equivalence_scope",
            "hit_count",
            "hits",
            "authority",
        )
    }


def _hit_key(ray: Mapping[str, object], hit_index: int, face_index: int) -> tuple[int, int, int, int, int, int]:
    return (
        int(ray["ray_index"]),
        int(ray["view_index"]),
        int(ray["pixel_x"]),
        int(ray["pixel_y"]),
        int(hit_index),
        int(face_index),
    )


def _packed_hits(value: Mapping[str, object]) -> dict[tuple[int, int, int, int, int, int], tuple[float, np.ndarray]]:
    out = {}
    for ray in value.get("rays") or ():
        measurement = dict(ray.get("measurement") or {})
        if measurement.get("status") != PASS or measurement.get("valid") is not True:
            continue
        hits = list(ray.get("canonical_hits") or ())
        alpha = np.asarray(measurement.get("effective_alpha"), dtype=np.float64)
        straight = np.asarray(measurement.get("straight_linear_rgb"), dtype=np.float64)
        if alpha.shape != (len(hits),) or straight.shape != (len(hits), 3):
            raise ValueError("PRESENTATION_EQUIVALENCE_PACKED_LAYOUT")
        for index, hit in enumerate(hits):
            key = _hit_key(ray, index, int(hit["face_index"]))
            if key in out:
                raise ValueError("PRESENTATION_EQUIVALENCE_PACKED_DUPLICATE_HIT")
            out[key] = (float(alpha[index]), straight[index] * alpha[index])
    return out


def _source_hits(value: Mapping[str, object]) -> dict[tuple[int, int, int, int, int, int], tuple[float, np.ndarray, float]]:
    out = {}
    for row in value.get("hits") or ():
        pseudo_ray = {
            "ray_index": row["ray_index"],
            "view_index": row["view_index"],
            "pixel_x": row["pixel_x"],
            "pixel_y": row["pixel_y"],
        }
        key = _hit_key(pseudo_ray, int(row["hit_index"]), int(row["face_index"]))
        if key in out:
            raise ValueError("PRESENTATION_EQUIVALENCE_SOURCE_DUPLICATE_HIT")
        rgb = np.asarray(row["premultiplied_linear_rgb"], dtype=np.float64)
        alpha = float(row["effective_alpha"])
        convergence = float(row["footprint_convergence_delta"])
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or not all(map(math.isfinite, (alpha, convergence))):
            raise ValueError("PRESENTATION_EQUIVALENCE_SOURCE_NONFINITE")
        out[key] = (alpha, rgb, convergence)
    return out


def qualify_subject_presentation_v1(
    packed_a: Mapping[str, object],
    packed_b: Mapping[str, object],
    source_a: Mapping[str, object],
    source_b: Mapping[str, object],
    *,
    appearance_tolerance: float = 5.0e-3,
    convergence_tolerance: float = 5.0e-3,
) -> dict:
    subject = str(packed_a.get("subject_id") or "")
    if not subject or any(str(row.get("subject_id") or "") != subject for row in (packed_b, source_a, source_b)):
        raise ValueError("PRESENTATION_EQUIVALENCE_SUBJECT_BINDING")
    if packed_a.get("authority") != "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS" or packed_b.get("authority") != "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS":
        raise ValueError("PRESENTATION_EQUIVALENCE_PACKED_AUTHORITY")
    if source_a.get("authority") != "SOURCE_OBSERVATION_APPEARANCE_REFERENCE_ONLY" or source_b.get("authority") != "SOURCE_OBSERVATION_APPEARANCE_REFERENCE_ONLY":
        raise ValueError("PRESENTATION_EQUIVALENCE_SOURCE_AUTHORITY")
    if not all(math.isfinite(v) and v >= 0 for v in (appearance_tolerance, convergence_tolerance)):
        raise ValueError("PRESENTATION_EQUIVALENCE_TOLERANCE")

    packed_exact = _packed_core(packed_a) == _packed_core(packed_b)
    source_exact = _source_core(source_a) == _source_core(source_b)
    packed_hits = _packed_hits(packed_a)
    source_hits = _source_hits(source_a)
    same_domain = bool(packed_hits) and set(packed_hits) == set(source_hits)
    maximum_alpha_error = 0.0
    maximum_rgb_error = 0.0
    maximum_convergence = 0.0
    equivalent = same_domain
    if same_domain:
        for key in sorted(packed_hits):
            pa, prgb = packed_hits[key]
            sa, srgb, convergence = source_hits[key]
            alpha_error = abs(pa - sa)
            rgb_error = float(np.max(np.abs(prgb - srgb)))
            maximum_alpha_error = max(maximum_alpha_error, alpha_error)
            maximum_rgb_error = max(maximum_rgb_error, rgb_error)
            maximum_convergence = max(maximum_convergence, convergence)
            equivalent = equivalent and alpha_error <= appearance_tolerance and rgb_error <= appearance_tolerance and convergence <= convergence_tolerance

    witness_payload = {
        "subject_id": subject,
        "packed_core_sha256": _digest(_packed_core(packed_a)),
        "source_core_sha256": _digest(_source_core(source_a)),
        "hit_keys": [list(key) for key in sorted(packed_hits)],
        "appearance_tolerance": appearance_tolerance,
        "convergence_tolerance": convergence_tolerance,
    }
    witness = _digest(witness_payload)
    candidate = FamilyQualificationInputV1(
        family="RAY_COMPOSITING",
        measurement_status="MEASURED",
        observability="MECHANISM_CONDITIONAL",
        witness_sha256=witness,
        valid_count=len(packed_hits) if equivalent else 0,
        total_count=len(packed_hits),
        source_equivalent=bool(equivalent),
        coverage_qualified=bool(packed_hits),
        replay_deterministic=bool(packed_exact and source_exact),
        leakage_free=False,
        conditioning_key="MULTIVIEW_RGBA_CAMERA_PLUS_TYPED_QUERY_ENDPOINTS",
    )
    decision = qualify_family(candidate)
    passed = bool(equivalent and packed_exact and source_exact and decision.oracle_eligible)
    return {
        "schema": "RealSaS.PresentationRaySubjectQualificationCandidate.v1",
        "subject_id": subject,
        "witness_sha256": witness,
        "witness_payload": witness_payload,
        "packed_replay_exact": packed_exact,
        "source_reference_replay_exact": source_exact,
        "same_hit_domain": same_domain,
        "source_equivalent": bool(equivalent),
        "hit_count": len(packed_hits),
        "maximum_alpha_error": maximum_alpha_error,
        "maximum_premultiplied_linear_rgb_error": maximum_rgb_error,
        "maximum_source_reference_footprint_convergence_delta": maximum_convergence,
        "family_qualification_input": {
            "family": candidate.family,
            "measurement_status": candidate.measurement_status,
            "observability": candidate.observability,
            "witness_sha256": candidate.witness_sha256,
            "valid_count": candidate.valid_count,
            "total_count": candidate.total_count,
            "source_equivalent": candidate.source_equivalent,
            "coverage_qualified": candidate.coverage_qualified,
            "replay_deterministic": candidate.replay_deterministic,
            "leakage_free": candidate.leakage_free,
            "conditioning_key": candidate.conditioning_key,
        },
        "local_qualification_decision": {
            "oracle_eligible": decision.oracle_eligible,
            "deterministic_student_supervision_eligible": decision.deterministic_student_supervision_eligible,
            "conditional_student_supervision_eligible": decision.conditional_student_supervision_eligible,
            "reasons": list(decision.reasons),
        },
        "student_supervision_authorized": False,
        "canonical_mesh_binding_minted": False,
        "product_authority_changed": False,
        "status": "PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE" if passed else "FAIL_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE",
    }
