"""Corpus370 semantic-prior candidate contract.

Teacher semantics are DESIGN CHOICES, not material physics or drawable truth.
The producer below cannot mint training admission; it only prepares sparse,
reviewed, source-addressed candidates for an independent family court.

No dependency on PyTorch, Blender, or a trained model.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

POLICY_SCHEMA = "RealSaS.SemanticPriorPolicy.v1"
ANNOTATION_SCHEMA = "RealSaS.Corpus370SemanticAnnotationInput.v1"
ROSTER_SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v5"
SUBJECT_SCHEMA = "RealSaS.Corpus370SemanticCandidate.v1"
COHORT_SCHEMA = "RealSaS.Corpus370ThreeHeadSemanticCandidateCohort.v1"
REVIEW_KINDS = frozenset({"HUMAN_CURATED", "REVIEWED_SOURCE_RULE"})
AFFORDANCE_KEYS = frozenset({
    "RIGID_PREFERRED",
    "FLEXIBLE_DEFORMATION",
    "SECONDARY_MOTION_ELIGIBLE",
    "ATTACHMENT_ELIGIBLE",
    "INDEPENDENT_CONTROL_ELIGIBLE",
    "PRESENTATION_SEPARABLE",
})


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("SEMANTIC_PRIOR_V1_" + reason)


def _hex64(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_policy(policy: dict) -> None:
    _require(isinstance(policy, dict) and policy.get("schema") == POLICY_SCHEMA, "POLICY_SCHEMA")
    _require(policy.get("authority") == "DESIGNED_PRIOR__NOT_SOURCE_PHYSICS", "POLICY_AUTHORITY")
    _require(policy.get("product_target") == "CONTROLLABLE_PUPPET", "POLICY_PRODUCT")
    _require(isinstance(policy.get("version"), str) and bool(policy["version"]), "POLICY_VERSION")
    keys = policy.get("functional_attributes")
    _require(isinstance(keys, list) and len(keys) == len(set(keys)) and set(keys) == AFFORDANCE_KEYS, "POLICY_ATTRIBUTES")
    concepts = policy.get("concepts")
    _require(isinstance(concepts, dict) and concepts and "UNKNOWN" in concepts, "POLICY_CONCEPTS")
    for name, payload in concepts.items():
        _require(isinstance(name, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is not None, "CONCEPT_ID")
        prior = payload.get("prior") if isinstance(payload, dict) else None
        _require(isinstance(prior, dict) and set(prior) <= AFFORDANCE_KEYS, "CONCEPT_PRIOR")
        _require(all(type(v) is bool for v in prior.values()), "CONCEPT_PRIOR_BOOL")
        _require(not (prior.get("RIGID_PREFERRED") is True and prior.get("FLEXIBLE_DEFORMATION") is True),
                 "CONFLICTING_RIGID_FLEXIBLE")
    _require(concepts["UNKNOWN"]["prior"] == {}, "UNKNOWN_MUST_ABSTAIN")
    excluded = policy.get("student_inputs_exclude")
    _require(isinstance(excluded, list) and "teacher_component_refs" in excluded
             and "material_names" in excluded and "bone_names" in excluded, "TEACHER_FIREWALL")


def _validate_roster(roster: dict, *, require_370: bool) -> tuple[dict, set]:
    _require(isinstance(roster, dict) and roster.get("schema") == ROSTER_SCHEMA, "ROSTER_SCHEMA")
    good_rows = roster.get("subjects")
    bad_rows = roster.get("failures")
    _require(isinstance(good_rows, list) and isinstance(bad_rows, list), "ROSTER_ROWS")
    good: dict[str, str] = {}
    excluded: set[str] = set()
    for row in good_rows:
        _require(isinstance(row, dict) and isinstance(row.get("subject_id"), str)
                 and _hex64(row.get("source_sha256")), "ROSTER_ADMITTED_ROW")
        sid = row["subject_id"]
        _require(sid not in good, "ROSTER_DUPLICATE_ADMITTED")
        good[sid] = row["source_sha256"]
    for row in bad_rows:
        _require(isinstance(row, dict) and isinstance(row.get("subject_id"), str), "ROSTER_EXCLUDED_ROW")
        sid = row["subject_id"]
        _require(sid not in good and sid not in excluded, "ROSTER_OVERLAP")
        excluded.add(sid)
    count = len(good) + len(excluded)
    if require_370:
        _require(count == 370 and len(good) == 352 and len(excluded) == 18, "ROSTER_V5_CENSUS")
    if "subject_count" in roster:
        _require(roster["subject_count"] == count, "ROSTER_SUBJECT_COUNT")
    if "prepared_subject_count" in roster:
        _require(roster["prepared_subject_count"] == len(good), "ROSTER_GOOD_COUNT")
    if "failure_count" in roster:
        _require(roster["failure_count"] == len(excluded), "ROSTER_FAIL_COUNT")
    _require(roster.get("training_authorized") is not True, "ROSTER_TRAINING_AUTHORITY")
    return good, excluded


def build_semantic_candidate(*, policy: dict, annotation: dict, source_sha256: str) -> dict:
    """Return explicitly NON-ADMITTED teacher rows without any invented negatives."""
    validate_policy(policy)
    _require(isinstance(annotation, dict) and isinstance(annotation.get("subject_id"), str), "SUBJECT_ID")
    _require(_hex64(source_sha256) and annotation.get("source_sha256") == source_sha256, "SOURCE_SHA")
    n = annotation.get("source_face_count")
    _require(type(n) is int and n > 0, "SOURCE_FACE_COUNT")
    components = annotation.get("components")
    _require(isinstance(components, list) and bool(components), "COMPONENTS_REQUIRED")
    _require(annotation.get("training_authorized") is not True, "ANNOTATION_FORGED_ADMISSION")
    seen_refs: set[str] = set()
    occupied: set[int] = set()
    output: list[dict] = []
    for component in sorted(components, key=lambda row: str(row.get("teacher_component_ref", ""))):
        _require(isinstance(component, dict), "COMPONENT_LAYOUT")
        ref = component.get("teacher_component_ref")
        _require(isinstance(ref, str) and bool(ref.strip()) and ref not in seen_refs, "COMPONENT_REF")
        seen_refs.add(ref)
        concept = component.get("concept")
        _require(concept in policy["concepts"], "UNKNOWN_CONCEPT")
        evidence = component.get("review")
        _require(isinstance(evidence, dict) and evidence.get("kind") in REVIEW_KINDS
                 and _hex64(evidence.get("receipt_sha256")), "INDEPENDENT_REVIEW_REQUIRED")
        face_ids = component.get("source_face_ids")
        _require(isinstance(face_ids, list) and bool(face_ids), "FACE_SUPPORT_REQUIRED")
        _require(all(type(face) is int and 0 <= face < n for face in face_ids), "FACE_RANGE")
        local = set(face_ids)
        _require(len(local) == len(face_ids) and not (occupied & local), "FACE_OVERLAP")
        occupied.update(local)
        # No material/bone/object names or private teacher IDs are projected to the student.
        output.append({
            "teacher_slot": len(output),
            "concept_target": concept,
            "functional_prior_targets": dict(sorted(policy["concepts"][concept]["prior"].items())),
            "support_source_face_ids": sorted(local),
            "source_review_kind": evidence["kind"],
            "source_review_sha256": evidence["receipt_sha256"],
            "target_origin": "DESIGNED_PRIOR_WITH_REVIEWED_SOURCE_SUPPORT",
            "eligible_for_training": False,
        })
    return {
        "schema": SUBJECT_SCHEMA,
        "subject_id": annotation["subject_id"],
        "source_sha256": source_sha256,
        "semantic_policy_version": policy["version"],
        "semantic_policy_sha256": _digest(policy),
        "annotation_sha256": _digest(annotation),
        "source_face_count_claim": n,
        "covered_face_count": len(occupied),
        "unlabeled_face_count": n - len(occupied),
        "component_candidates": output,
        "unlabeled_faces_are_negative": False,
        "teacher_component_ref_exposed_to_student": False,
        "source_face_addresses_are_student_features": False,
        "source_geometry_binding_verified": False,
        "semantic_family_independent_court_passed": False,
        "training_authorized": False,
        "status": "SEMANTIC_PRIOR_SOURCE_SUPPORT_CANDIDATE_ONLY",
    }


def build_corpus370_candidate_cohort(*, policy: dict, roster: dict, annotations: dict,
                                      require_370: bool = True) -> dict:
    """Full V5 census; an absent annotation is UNKNOWN, not a negative example."""
    validate_policy(policy)
    admitted, excluded = _validate_roster(roster, require_370=require_370)
    _require(isinstance(annotations, dict) and annotations.get("schema") == ANNOTATION_SCHEMA, "ANNOTATION_SCHEMA")
    rows = annotations.get("subjects")
    _require(isinstance(rows, list), "ANNOTATION_ROWS")
    lookup: dict[str, dict] = {}
    for row in rows:
        _require(isinstance(row, dict) and isinstance(row.get("subject_id"), str), "ANNOTATION_SUBJECT")
        sid = row["subject_id"]
        _require(sid not in lookup and sid not in excluded and sid in admitted, "ANNOTATION_NOT_ADMITTED")
        lookup[sid] = row
    census: list[dict] = []
    stats: Counter[str] = Counter()
    for sid in sorted(set(admitted) | excluded):
        if sid in excluded:
            item = {"subject_id": sid, "status": "OBSERVATION_EXCLUDED", "candidate": None}
        elif sid not in lookup:
            item = {"subject_id": sid, "status": "SEMANTIC_UNKNOWN", "candidate": None}
        else:
            proposal = build_semantic_candidate(policy=policy, annotation=lookup[sid],
                                                source_sha256=admitted[sid])
            item = {"subject_id": sid, "status": "SEMANTIC_CANDIDATE_ONLY", "candidate": proposal}
        census.append(item)
        stats[item["status"]] += 1
    return {
        "schema": COHORT_SCHEMA,
        "policy_sha256": _digest(policy),
        "observation_roster_sha256": _digest(roster),
        "semantic_annotations_sha256": _digest(annotations),
        "subjects": census,
        "counts": dict(sorted(stats.items())),
        "family_states_minted": False,
        "semantic_training_authorized": False,
        "tessa_training_authorized": False,
        "downstream_compiler_authorized": False,
        "target_matrix_status": "CANDIDATE_AND_UNKNOWN_ONLY__NO_TRAINING",
    }
