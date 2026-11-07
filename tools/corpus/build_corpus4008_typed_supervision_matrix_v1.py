#!/usr/bin/env python3
"""Master source × evidence-kind × model-consumer matrix. NO admission.

This is deliberately conservative:
* Exact original SHA and imported control graph are observations.
* Importing zero IK constraints does NOT show authored IK absence.
* Imported motion channel counts do NOT establish functional animator control.
* Candidate *routing* is not training or label authority.
* No unknown target is converted to PASS, including historical ADMIT.
* No private original files or images are copied to outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.corpus.validate_corpus4008_deep_diagnostic_v1 import verify_control

SCHEMA = "RealSaS.Corpus4008TypedSupervisionMatrix.v1"

OBS = "IMPORTED_SOURCE_OBSERVATION"
UNKNOWN = "UNKNOWN"
# The source verified court is different from importer observation.
TRUTH = "INDEPENDENT_SOURCE_TRUTH"
DECISION = "NOT_ADMITTED"

HEADS = (
    "IRIS_GEOMETRY", "IRIS_SEMANTIC", "IRIS_GRAPHICS",
    "TESSA", "AXIS", "MIRA",
)
EVIDENCE = {
    # kind: (head consumers, optionality)
    "source_identity_sha": (HEADS, "REQUIRED_ALL"),
    "commercial_ml_rights": (HEADS, "REQUIRED_ALL"),
    "canonical_dedup_split": (HEADS, "REQUIRED_ALL"),
    "character_vs_prop": (HEADS, "REQUIRED_ALL"),
    "morphology_functional_class": (HEADS, "REQUIRED_ALL"),
    "mesh_component_topology": ("IRIS_GEOMETRY", "TESSA", "IRIS_SEMANTIC"),
    "native_rest_and_pose": ("IRIS_GEOMETRY", "TESSA", "AXIS", "MIRA"),
    "anatomical_frame_camera": ("IRIS_GEOMETRY", "IRIS_SEMANTIC", "IRIS_GRAPHICS"),
    "native_material_texture_rgba": ("IRIS_GRAPHICS", "IRIS_SEMANTIC"),
    "native_depth_normals_face_ids": ("IRIS_GEOMETRY", "IRIS_SEMANTIC", "IRIS_GRAPHICS"),
    "semantic_source_face_regions": ("IRIS_SEMANTIC", "IRIS_GRAPHICS"),
    "semantic_affordance_prior": ("IRIS_SEMANTIC", "AXIS"),
    "visibility_draw_order": ("IRIS_GRAPHICS", "IRIS_SEMANTIC"),
    "layer_edit_reveal_consequence": ("IRIS_GRAPHICS", "IRIS_SEMANTIC"),
    "rig_hierarchy_and_bind": ("AXIS", "MIRA", "TESSA"),
    "deform_skin_vertex_influences": ("MIRA", "TESSA"),
    "controller_bone_collections": ("AXIS",),
    "ik_constraint_topology": ("AXIS",),
    "fk_controller_channels": ("AXIS",),
    "ik_fk_switch_and_snap_response": ("AXIS",),
    "constraint_target_graph": ("AXIS",),
    "driver_graph": ("AXIS",),
    "animation_curve_channels": ("AXIS",),
    "nla_action_bindings": ("AXIS",),
    "animation_pose_kinematics": ("AXIS", "TESSA", "MIRA"),
    "per_control_causal_response": ("AXIS", "TESSA", "MIRA"),
    "skin_deformation_consequence": ("MIRA", "TESSA"),
    "attachments_sockets_consequence": ("AXIS", "IRIS_GRAPHICS", "MIRA"),
    "stylized_augmented_presentation": ("IRIS_GRAPHICS",),
    "human_visual_review": (HEADS, "REQUIRED_ALL"),
}
# Correct the tuple container for non-global evidence kinds on access.
CANDIDATE_LABELS = {
    "rig_hierarchy_and_bind": "IMPORT_BONE_GRAPH_ONLY",
    "controller_bone_collections": "IMPORT_COLLECTION_NAMES_ONLY",
    "ik_constraint_topology": "IMPORT_IK_CONSTRAINTS_ONLY",
    "constraint_target_graph": "IMPORT_CONSTRAINTS_ONLY",
    "driver_graph": "IMPORT_DRIVER_CHANNELS_ONLY",
    "animation_curve_channels": "IMPORT_FCURVE_CHANNELS_ONLY",
    "nla_action_bindings": "IMPORT_NLA_STRIPS_ONLY",
}


def fail(c, why):
    if not c:
        raise ValueError("TYPED_SUPERVISION_" + why)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def observations(control):
    d = control["imported_control"]
    return {
        "rig_hierarchy_and_bind": {
            "armatures": d["imported_armature_count"],
            "pose_bones": d["imported_pose_bone_count"],
            "claim_boundary": "IMPORTED_BONES_NOT_INDEPENDENT_BIND_PARITY",
        },
        "controller_bone_collections": {
            "nondeform_bone_flags": d["imported_nondeform_bone_count"],
            "claim_boundary": "NONDEFORM_IS_NOT_VERIFIED_ANIMATOR_CONTROLLER",
        },
        "ik_constraint_topology": {
            "imported_ik_constraint_count": d["imported_ik_constraint_count"],
            "claim_boundary": "ZERO_IMPORT_IS_NOT_AUTHORED_IK_ABSENCE",
        },
        "constraint_target_graph": {
            "imported_constraint_count": d["imported_constraint_count"],
            "claim_boundary": "IMPORTED_CONSTRAINT_IS_NOT_SOLVER_BEHAVIOR",
        },
        "driver_graph": {
            "imported_driver_count": d["imported_driver_count"],
            "claim_boundary": "EXPRESSION_HASH_AND_CHANNEL_NOT_EXECUTION_PROOF",
        },
        "animation_curve_channels": {
            "imported_action_count": d["imported_action_count"],
            "imported_action_curve_count": d["imported_action_curve_count"],
            "claim_boundary": "ACTION_CHANNELS_NOT_VALID_SOURCE_ACTION_EVALUATION",
        },
        "nla_action_bindings": {
            "imported_nla_strip_count": d["imported_nla_strip_count"],
            "claim_boundary": "IMPORTED_NLA_NOT_NATIVE_TIME_PARITY",
        },
    }


def build(selection, profiles, output, expected_count=3993):
    master = json.loads(selection.read_text())
    fail(isinstance(master, dict) and len(master) == expected_count,
         "MASTER_SELECTION_SIZE")
    docs = {}
    if profiles.exists():
        for path in sorted(profiles.rglob("*.control.json")):
            d = json.loads(path.read_text())
            verify_control(d)
            aid = d.get("asset_id")
            fail(isinstance(aid, str) and aid not in docs,
                 "DUPLICATE_CONTROL_RECEIPT")
            docs[aid] = (d, digest(path))
    fail(set(docs).issubset(master), "FOREIGN_RECEIPT")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    source_count = Counter()
    family_observations = Counter()
    head_route = Counter()
    all_evidence = {}
    for aid, m in sorted(master.items()):
        fail(m.get("canonical_asset_id") == aid, "ASSET_ID")
        split = m.get("split")
        fail(split in {"FIT", "TUNE", "DEV", "CAL", "EXTERNAL_HOLDOUT"},
             "SPLIT")
        provider = m.get("source_registry_id")
        candidate = m.get("candidate_id")
        fail(isinstance(provider, str) and isinstance(candidate, str)
             and candidate.startswith(provider + ":"), "PROVIDER")
        c = None
        rec_hash = None
        if aid in docs:
            c, rec_hash = docs[aid]
            fail(c.get("candidate_id") == candidate, "CANDIDATE_MISMATCH")
            original_sha = c.get("original_sha256", "")
            token = candidate.split(":", 1)[1]
            fail(len(token) in {24, 64} and original_sha.startswith(token),
                 "SOURCE_SHA_MISMATCH")
            source_count["IMPORTED_PROFILE_PRESENT"] += 1
        else:
            source_count["NOT_IMPORTED"] += 1
        observed = observations(c) if c else {}
        # Immutable manifest itself only proves a historical row, not rights.
        identity = {
            "asset_id": aid,
            "candidate_id": candidate,
            "provider": provider,
            "split": split,
            "holdout_excluded_from_training": split == "EXTERNAL_HOLDOUT",
            "historic_acceptance_not_authority": m.get("decision"),
            "source_control_receipt_sha256": rec_hash,
            "source_sha256": c["original_sha256"] if c else None,
        }
        evidence = {}
        for family, mapping in EVIDENCE.items():
            consumers = mapping[0] if len(mapping) == 2 and isinstance(mapping[0], tuple) else mapping
            if family == "source_identity_sha" and c is not None:
                status = OBS
                value = {"original_sha256_verified_during_import": c["original_sha256"]}
                boundary = "SOURCE_BYTE_HASH_NOT_LICENSE_CHARACTER_OR_TRUTH"
            elif family in observed:
                status = OBS
                value = observed[family]
                boundary = CANDIDATE_LABELS[family]
            else:
                status = UNKNOWN
                value = None
                boundary = "NOT_MEASURED_OR_INDEPENDENTLY_COURTED"
            evidence[family] = {
                "observation_status": status,
                "independent_truth_status": UNKNOWN,
                "model_admission_status": DECISION,
                "evidence_origin": "imported_scene" if status == OBS else "unobserved",
                "observed_value": value,
                "limit": boundary,
                "eligible_consumers_after_independent_courts": list(consumers),
            }
            if status == OBS:
                family_observations[family] += 1
        head = {}
        for h in HEADS:
            present = [kind for kind in evidence if
                       evidence[kind]["observation_status"] == OBS and
                       h in evidence[kind]["eligible_consumers_after_independent_courts"]]
            head[h] = {
                "imported_evidence_kinds": present,
                "source_truth_court": UNKNOWN,
                "per_target_truth_court": UNKNOWN,
                "semantic_and_capability_policy_review": UNKNOWN,
                "supervision_suitable": False,
                "training_admitted": False,
                "candidate_only": bool(present),
            }
            if present:
                head_route[h] += 1
        all_evidence[aid] = {
            "schema": "RealSaS.Corpus4008PerAssetEvidenceLedger.v1",
            "identity": identity,
            "evidence": evidence,
            "head_routes": head,
            "source_vs_prior_vs_representation_policy_not_conflated": True,
            "training_authorized": False,
        }
        rows.append({
            "asset_id": aid,
            "provider": provider,
            "candidate_id": candidate,
            "split": split,
            "holdout": identity["holdout_excluded_from_training"],
            "imported_control_profile": c is not None,
            "source_sha256": identity["source_sha256"],
            "observed_bone_count": (c["imported_control"]["imported_pose_bone_count"] if c else None),
            "observed_ik_constraint_count": (c["imported_control"]["imported_ik_constraint_count"] if c else None),
            "observed_action_count": (c["imported_control"]["imported_action_count"] if c else None),
            "ik_functionality": UNKNOWN,
            "semantic_label_truth": UNKNOWN,
            "graphics_native_rgba_truth": UNKNOWN,
            "head_suitability": UNKNOWN,
            "training_admitted": False,
        })
    # Bound model input to small summary CSV; evidence JSON remains canonical.
    with (output / "TYPED_SUPERVISION_INDEX_V1.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "TYPED_SUPERVISION_EVIDENCE_V1.jsonl").open(
            "w", encoding="utf-8") as f:
        for aid in sorted(all_evidence):
            f.write(json.dumps(all_evidence[aid], sort_keys=True) + "\n")
    report = {
        "schema": SCHEMA,
        "master_selection_sha256": digest(selection),
        "master_rows": len(master),
        "source_profile_counts": dict(sorted(source_count.items())),
        "family_imported_observation_counts": dict(sorted(family_observations.items())),
        "per_head_import_observation_candidate_counts": dict(sorted(head_route.items())),
        "per_head_truth_pass": {h: 0 for h in HEADS},
        "per_head_training_admitted": {h: 0 for h in HEADS},
        "explicit_not_verified": [
            "source_license", "character_class", "semantic_labels",
            "skin_binding_deformation", "renderer_rgb",
            "ik_fk_functionality", "driver_runtime", "camera_anatomy",
            "rig_attachment_interventions", "head_consumers",
        ],
        "training_authorized": False,
    }
    (output / "TYPED_SUPERVISION_SUMMARY_V1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--control-profiles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-count", type=int, default=3993)
    a = p.parse_args()
    print(json.dumps(build(a.selection, a.control_profiles,
                           a.output, a.expected_count), sort_keys=True))


if __name__ == "__main__":
    main()
