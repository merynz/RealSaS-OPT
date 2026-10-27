#!/usr/bin/env python3
from __future__ import annotations

"""Qualify the sealed sparse measured-ray domain for teacher-oracle use only.

The audit combines four independently executed receipts per subject:
- packed measurement replay A;
- packed measurement replay B;
- admitted source-observation material reference A;
- admitted source-observation material reference B.

It requires exact replay of the measured cores and compares packed per-hit
alpha/premultiplied-linear-RGB against the source-observation evaluator on the
same teacher triangle, UV and micro-camera footprint.  Coverage is intentionally
only the sealed FIT8 16-ray / 32-hit court domain.

The resulting ``qualification_v1`` decision is a local candidate.  It does not
mint Registry admission, a TruthQualificationReport authority, student labels,
training authority or product activation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from models.iris.truth.qualification_v1 import FamilyQualificationInputV1, qualify_family

SCHEMA = "RealSaS.SourceRayAppearanceSourceEquivalenceCourt.v1"
POLICY = {
    "family": "RAY_COMPOSITING",
    "equivalence_scope": "ADMITTED_UNLIT_SOURCE_OBSERVATION_MATERIAL_POLICY__SAME_TEACHER_TRIANGLE_UV_AND_MICRO_CAMERA",
    "subject_count": 8,
    "ray_count": 16,
    "hit_count": 32,
    "appearance_error_tolerance": 0.005,
    "source_reference_footprint_convergence_tolerance": 0.005,
    "packed_measurement_status_required": "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION",
    "packed_replay_core_equality": "EXACT_JSON_VALUE_EQUALITY",
    "source_reference_replay_core_equality": "EXACT_JSON_VALUE_EQUALITY",
    "coverage_scope": "SEALED_FIT8_8_SUBJECT__16_GEOMETRY_ONLY_RAYS__32_HITS_ONLY",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _packed_core(value: dict) -> dict:
    return {
        "schema": value.get("schema"),
        "subject_id": value.get("subject_id"),
        "sampling_contract": value.get("sampling_contract"),
        "rgb_carrier_calibration": value.get("rgb_carrier_calibration"),
        "material_policy_replay": value.get("material_policy_replay"),
        "selected_ray_count": value.get("selected_ray_count"),
        "pass_ray_count": value.get("pass_ray_count"),
        "fail_ray_count": value.get("fail_ray_count"),
        "abstain_ray_count": value.get("abstain_ray_count"),
        "maximum_reconstructed_full_to_original_error": value.get("maximum_reconstructed_full_to_original_error"),
        "maximum_footprint_convergence_delta": value.get("maximum_footprint_convergence_delta"),
        "rays": value.get("rays"),
        "authority": value.get("authority"),
    }


def _source_core(value: dict) -> dict:
    return {
        "schema": value.get("schema"),
        "subject_id": value.get("subject_id"),
        "candidate_sha256": value.get("candidate_sha256"),
        "camera_bundle_sha256": value.get("camera_bundle_sha256"),
        "appearance_sha256": value.get("appearance_sha256"),
        "geometry_npz_sha256": value.get("geometry_npz_sha256"),
        "rgb_carrier_calibration": value.get("rgb_carrier_calibration"),
        "equivalence_scope": value.get("equivalence_scope"),
        "geometry_lane_equivalence_required": value.get("geometry_lane_equivalence_required"),
        "native_lit_parity_claimed": value.get("native_lit_parity_claimed"),
        "hit_count": value.get("hit_count"),
        "hits": value.get("hits"),
        "authority": value.get("authority"),
    }


def _hit_key(row: dict) -> tuple[int, int, int, int, int, int]:
    return (
        int(row["ray_index"]),
        int(row["view_index"]),
        int(row["pixel_x"]),
        int(row["pixel_y"]),
        int(row["hit_index"]),
        int(row["face_index"]),
    )


def _packed_hits(value: dict) -> dict[tuple[int, int, int, int, int, int], dict]:
    out = {}
    for ray in value.get("rays") or ():
        measurement = dict(ray.get("measurement") or {})
        if measurement.get("status") != POLICY["packed_measurement_status_required"] or measurement.get("valid") is not True:
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_PACKED_NONPASS_RAY:" + str(value.get("subject_id")))
        alpha = np.asarray(measurement.get("effective_alpha"), np.float64)
        straight = np.asarray(measurement.get("straight_linear_rgb"), np.float64)
        hits = list(ray.get("canonical_hits") or ())
        if alpha.shape != (len(hits),) or straight.shape != (len(hits), 3) or not np.isfinite(alpha).all() or not np.isfinite(straight).all():
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_PACKED_MEASUREMENT_LAYOUT")
        for index, hit in enumerate(hits):
            row = {
                "ray_index": int(ray["ray_index"]),
                "view_index": int(ray["view_index"]),
                "pixel_x": int(ray["pixel_x"]),
                "pixel_y": int(ray["pixel_y"]),
                "hit_index": index,
                "face_index": int(hit["face_index"]),
                "effective_alpha": float(alpha[index]),
                "premultiplied_linear_rgb": (straight[index] * alpha[index]).tolist(),
            }
            key = _hit_key(row)
            if key in out:
                raise RuntimeError("SOURCE_RAY_EQUIVALENCE_PACKED_DUPLICATE_HIT")
            out[key] = row
    return out


def _source_hits(value: dict) -> dict[tuple[int, int, int, int, int, int], dict]:
    rows = list(value.get("hits") or ())
    out = {}
    for row in rows:
        key = _hit_key(row)
        if key in out:
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_SOURCE_DUPLICATE_HIT")
        alpha = float(row["effective_alpha"])
        rgb = np.asarray(row["premultiplied_linear_rgb"], np.float64)
        convergence = float(row["footprint_convergence_delta"])
        if not math.isfinite(alpha) or not np.isfinite(rgb).all() or rgb.shape != (3,) or not math.isfinite(convergence):
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_SOURCE_NONFINITE")
        out[key] = row
    return out


def audit(root: Path) -> dict:
    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(subject_dirs) != POLICY["subject_count"]:
        raise RuntimeError("SOURCE_RAY_EQUIVALENCE_SUBJECT_COUNT")

    rows = []
    all_parent_hashes = []
    total_hits = total_rays = 0
    source_equivalent = True
    replay_deterministic = True
    max_alpha_error = 0.0
    max_rgb_error = 0.0
    max_source_convergence = 0.0

    for folder in subject_dirs:
        packed_a_path = folder / "PACKED_A.json"
        packed_b_path = folder / "PACKED_B.json"
        source_a_path = folder / "SOURCE_REFERENCE_A.json"
        source_b_path = folder / "SOURCE_REFERENCE_B.json"
        for path in (packed_a_path, packed_b_path, source_a_path, source_b_path):
            if not path.is_file():
                raise RuntimeError("SOURCE_RAY_EQUIVALENCE_RECEIPT_MISSING:" + str(path))
        packed_a, packed_b = _load(packed_a_path), _load(packed_b_path)
        source_a, source_b = _load(source_a_path), _load(source_b_path)
        subject = str(packed_a.get("subject_id") or "")
        if not subject or any(str(v.get("subject_id") or "") != subject for v in (packed_b, source_a, source_b)):
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_SUBJECT_BINDING")
        if packed_a.get("authority") != "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS" or packed_b.get("authority") != "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS":
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_PACKED_AUTHORITY")
        for source in (source_a, source_b):
            if source.get("authority") != "SOURCE_OBSERVATION_APPEARANCE_REFERENCE_ONLY":
                raise RuntimeError("SOURCE_RAY_EQUIVALENCE_SOURCE_AUTHORITY")
            if source.get("equivalence_scope") != POLICY["equivalence_scope"] or source.get("native_lit_parity_claimed") is not False:
                raise RuntimeError("SOURCE_RAY_EQUIVALENCE_SCOPE_DRIFT")
            if source.get("geometry_lane_equivalence_required") is not False:
                raise RuntimeError("SOURCE_RAY_EQUIVALENCE_GEOMETRY_LANE_REGAINED_AUTHORITY")

        packed_exact = _packed_core(packed_a) == _packed_core(packed_b)
        source_exact = _source_core(source_a) == _source_core(source_b)
        replay_deterministic = replay_deterministic and packed_exact and source_exact

        packed_hits = _packed_hits(packed_a)
        source_hits = _source_hits(source_a)
        if set(packed_hits) != set(source_hits):
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_HIT_DOMAIN_DRIFT:" + subject)
        if len(packed_hits) != int(source_a.get("hit_count", -1)):
            raise RuntimeError("SOURCE_RAY_EQUIVALENCE_HIT_COUNT_DRIFT:" + subject)
        ray_count = len({key[:4] for key in packed_hits})
        total_rays += ray_count
        total_hits += len(packed_hits)

        subject_alpha_error = 0.0
        subject_rgb_error = 0.0
        subject_convergence = 0.0
        subject_equivalent = True
        comparison_rows = []
        for key in sorted(packed_hits):
            p = packed_hits[key]
            s = source_hits[key]
            alpha_error = abs(float(p["effective_alpha"]) - float(s["effective_alpha"]))
            rgb_error = float(np.max(np.abs(np.asarray(p["premultiplied_linear_rgb"], np.float64) - np.asarray(s["premultiplied_linear_rgb"], np.float64))))
            convergence = float(s["footprint_convergence_delta"])
            passed = (
                alpha_error <= POLICY["appearance_error_tolerance"] + 1e-15
                and rgb_error <= POLICY["appearance_error_tolerance"] + 1e-15
                and convergence <= POLICY["source_reference_footprint_convergence_tolerance"] + 1e-15
            )
            subject_equivalent = subject_equivalent and passed
            subject_alpha_error = max(subject_alpha_error, alpha_error)
            subject_rgb_error = max(subject_rgb_error, rgb_error)
            subject_convergence = max(subject_convergence, convergence)
            comparison_rows.append({
                "key": list(key),
                "alpha_error": alpha_error,
                "premultiplied_linear_rgb_max_abs_error": rgb_error,
                "source_reference_footprint_convergence_delta": convergence,
                "passed": passed,
            })
        source_equivalent = source_equivalent and subject_equivalent
        max_alpha_error = max(max_alpha_error, subject_alpha_error)
        max_rgb_error = max(max_rgb_error, subject_rgb_error)
        max_source_convergence = max(max_source_convergence, subject_convergence)
        parent_hashes = {
            "packed_a_sha256": _sha(packed_a_path),
            "packed_b_sha256": _sha(packed_b_path),
            "source_reference_a_sha256": _sha(source_a_path),
            "source_reference_b_sha256": _sha(source_b_path),
        }
        all_parent_hashes.append({"subject_id": subject, **parent_hashes})
        rows.append({
            "subject_id": subject,
            "ray_count": ray_count,
            "hit_count": len(packed_hits),
            "packed_replay_core_exact": packed_exact,
            "source_reference_replay_core_exact": source_exact,
            "source_equivalent_on_declared_hits": subject_equivalent,
            "maximum_alpha_error": subject_alpha_error,
            "maximum_premultiplied_linear_rgb_error": subject_rgb_error,
            "maximum_source_reference_footprint_convergence_delta": subject_convergence,
            "parents": parent_hashes,
            "comparisons": comparison_rows,
        })

    coverage_qualified = total_rays == POLICY["ray_count"] and total_hits == POLICY["hit_count"]
    witness_payload = {"policy": POLICY, "parents": all_parent_hashes}
    witness_sha256 = _digest(witness_payload)
    candidate = FamilyQualificationInputV1(
        family="RAY_COMPOSITING",
        measurement_status="MEASURED",
        observability="OBSERVATION_CONDITIONED",
        witness_sha256=witness_sha256,
        valid_count=total_hits if source_equivalent else 0,
        total_count=total_hits,
        source_equivalent=bool(source_equivalent),
        coverage_qualified=bool(coverage_qualified),
        replay_deterministic=bool(replay_deterministic),
        leakage_free=False,
        conditioning_key=None,
    )
    decision = qualify_family(candidate)
    if not (
        source_equivalent
        and coverage_qualified
        and replay_deterministic
        and decision.oracle_eligible
        and not decision.deterministic_student_supervision_eligible
    ):
        status = "FAIL_SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE"
    else:
        status = "PASS_SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE"

    return {
        "schema": SCHEMA,
        "policy": POLICY,
        "policy_sha256": _digest(POLICY),
        "witness_sha256": witness_sha256,
        "subject_count": len(rows),
        "ray_count": total_rays,
        "hit_count": total_hits,
        "source_equivalent_on_declared_sparse_domain": bool(source_equivalent),
        "coverage_qualified_on_declared_sparse_domain": bool(coverage_qualified),
        "replay_deterministic_on_declared_sparse_domain": bool(replay_deterministic),
        "maximum_alpha_error": max_alpha_error,
        "maximum_premultiplied_linear_rgb_error": max_rgb_error,
        "maximum_source_reference_footprint_convergence_delta": max_source_convergence,
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
            "family": decision.family,
            "oracle_eligible": decision.oracle_eligible,
            "deterministic_student_supervision_eligible": decision.deterministic_student_supervision_eligible,
            "conditional_student_supervision_eligible": decision.conditional_student_supervision_eligible,
            "unconditional_abstention_required": decision.unconditional_abstention_required,
            "valid_fraction": decision.valid_fraction,
            "reasons": list(decision.reasons),
        },
        "native_lit_parity_claimed": False,
        "global_sampler_qualification_claimed": False,
        "registry_admitted": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "product_pass_claimed": False,
        "court_status": status,
        "subjects": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subjects-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = audit(a.subjects_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "subjects"}, sort_keys=True), flush=True)
    if result["court_status"] != "PASS_SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE":
        raise RuntimeError("SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE_FAILED")


if __name__ == "__main__":
    main()
