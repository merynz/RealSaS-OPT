#!/usr/bin/env python3
"""Objective-exact FIT8 Stage48 bundle with Platform UUID -> logical-key binding.

Platform Attempt/Registry APIs use immutable subject UUIDs, while the FIT8
scientific contract uses stable logical keys such as ``fit8_snake_v1``.  The
exact Stage48 cohort payload already carries that logical key.  This builder
binds the two identities without rewriting CAS bytes:

* Platform UUID remains the Registry/CAS lookup identity.
* Cohort logical key is read from the exact exported Stage48 summary bytes.
* The single regression role is bound to the single preregistered regression
  key while preserving the producer's own subject id as provenance.

It then builds the same objective-exact V3.1 projection sealed by the prior V2
builder.  No training, model promotion or product authority is created here.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from models.iris.v31.presentation_objective_v1 import build_presentation_targets_v1
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256
from tools.truth_foundry.build_fit8_presentation_truth_bundle_v1 import (
    cas_path,
    output_ref,
    require,
    stage_manifest,
    write_json,
)

STAGE08 = "08_NORMALIZATION_DOMAIN_QUALIFIED"
STAGE47 = "47_SOURCE_RESPONSE_CALIBRATION"
STAGE48 = "48_PRESENTATION_RESPONSE_TRUTH"
SUMMARY_SCHEMA = "RealSaS.PresentationResponseTruthCandidate.v1"
EXCLUSION_SCHEMA = "RealSaS.PresentationTruthRegressionExclusion.v1"


def _export_payload(export: dict, schema: str) -> dict:
    rows = [row for row in export.get("outputs") or () if row.get("schema") == schema]
    require(len(rows) == 1, "FIT8_PRESENTATION_V3_EXPORT_SCHEMA_CARDINALITY:" + schema)
    ref = rows[0]
    path = Path(str(ref.get("path") or "")).resolve()
    require(
        path.is_file()
        and path.stat().st_size == int(ref["size_bytes"])
        and sha256(path) == ref["sha256"],
        "FIT8_PRESENTATION_V3_EXPORTED_BYTES_DRIFT:" + schema,
    )
    return json.loads(path.read_text())


def bind_logical_attempts_v3(observation: dict, expected: tuple[str, ...], regression: tuple[str, ...]):
    attempts_by_id = {row["attempt_id"]: dict(row) for row in observation.get("attempts") or ()}
    exports = list(observation.get("exports") or ())
    require(len(attempts_by_id) == len(exports), "FIT8_PRESENTATION_V3_ATTEMPT_EXPORT_CARDINALITY")
    require(len(attempts_by_id) == len(expected) + len(regression), "FIT8_PRESENTATION_V3_MEMBER_CARDINALITY")
    require(all(row.get("state") in {"COMPLETED", "PASS"} for row in attempts_by_id.values()), "FIT8_PRESENTATION_BUNDLE_ATTEMPT_NOT_COMPLETE")

    cohort = {}
    regression_exports = []
    seen_platform_subjects = set()
    for export in exports:
        attempt_id = export.get("attempt_id")
        require(attempt_id in attempts_by_id, "FIT8_PRESENTATION_V3_EXPORT_ATTEMPT_UNKNOWN")
        attempt = attempts_by_id[attempt_id]
        require(
            export.get("subject_id") == attempt.get("subject_id")
            and export.get("role") == attempt.get("role"),
            "FIT8_PRESENTATION_V3_EXPORT_ATTEMPT_BINDING_DRIFT",
        )
        platform_subject_id = str(attempt["subject_id"])
        require(platform_subject_id not in seen_platform_subjects, "FIT8_PRESENTATION_V3_PLATFORM_SUBJECT_DUPLICATE")
        seen_platform_subjects.add(platform_subject_id)
        role = str(attempt["role"])
        if role == "COHORT":
            payload = _export_payload(export, SUMMARY_SCHEMA)
            logical = str(payload.get("subject_id") or "")
            require(logical in expected, "FIT8_PRESENTATION_V3_LOGICAL_SUBJECT_OUT_OF_SCOPE")
            require(logical not in cohort, "FIT8_PRESENTATION_V3_LOGICAL_SUBJECT_DUPLICATE")
            cohort[logical] = {
                "logical_subject_id": logical,
                "platform_subject_id": platform_subject_id,
                "attempt_id": attempt_id,
                "role": role,
            }
        elif role == "REGRESSION":
            payload = _export_payload(export, EXCLUSION_SCHEMA)
            require(
                payload.get("new_truth_produced") is False
                and payload.get("training_authorized") is False
                and bool(payload.get("subject_id")),
                "FIT8_PRESENTATION_BUNDLE_REGRESSION_EXCLUSION_DRIFT",
            )
            regression_exports.append(
                {
                    "platform_subject_id": platform_subject_id,
                    "attempt_id": attempt_id,
                    "producer_subject_id": str(payload["subject_id"]),
                    "role": role,
                }
            )
        else:
            raise ValueError("FIT8_PRESENTATION_V3_UNKNOWN_MEMBER_ROLE:" + role)

    require(set(cohort) == set(expected), "FIT8_PRESENTATION_BUNDLE_SUBJECT_SET_DRIFT")
    require(len(regression) == 1 and len(regression_exports) == 1, "FIT8_PRESENTATION_V3_REGRESSION_CARDINALITY")
    regression_bindings = {
        regression[0]: {
            "logical_subject_id": regression[0],
            **regression_exports[0],
        }
    }
    return cohort, regression_bindings


def build(observation, *, api_root, artifact_root, repo_root, output):
    require(
        observation.get("schema") == "RealSaS.PresentationTruthCampaignObservation.v1"
        and observation.get("all_stage48_outputs_exported") is True
        and observation.get("goal_complete") is True
        and not observation.get("errors"),
        "FIT8_PRESENTATION_BUNDLE_CAMPAIGN_NOT_COMPLETE",
    )
    repo = Path(repo_root).resolve()
    contract = json.loads((repo / "canonical/FIT8_PRESENTATION_TRUTH_CAMPAIGN_V1.json").read_text())
    expected = tuple(contract["cohort_subject_keys"])
    regression = tuple(contract["regression_subject_keys"])
    attempts, regression_bindings = bind_logical_attempts_v3(observation, expected, regression)

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    subject_rows = []
    aggregate = {
        "surface_faces": 0,
        "actions": 0,
        "appearance_hits": 0,
        "observable_pairs": 0,
        "pose_order_pairs": 0,
    }

    for subject in expected:
        binding = attempts[subject]
        attempt_id = binding["attempt_id"]
        platform_subject_id = binding["platform_subject_id"]
        _, record08, manifest08 = stage_manifest(api_root, artifact_root, platform_subject_id, attempt_id, STAGE08)
        _, record47, manifest47 = stage_manifest(api_root, artifact_root, platform_subject_id, attempt_id, STAGE47)
        _, record48, manifest48 = stage_manifest(api_root, artifact_root, platform_subject_id, attempt_id, STAGE48)
        normalization_ref = output_ref(manifest08, schema="RealSaS.NormalizationDomainIR.v1")
        geometry_ref = output_ref(manifest47, schema="RealSaS.TeacherCalibrationArrays.v2")
        summary_ref = output_ref(manifest48, schema=SUMMARY_SCHEMA)
        arrays_ref = output_ref(manifest48, schema="RealSaS.PresentationResponseArrays.v1")
        packed_ref = output_ref(manifest48, schema="RealSaS.SourceRayAppearanceMeasurementCandidate.v1", name="PACKED_A.json")

        normalization = json.loads(cas_path(artifact_root, normalization_ref).read_text())
        geometry_path = cas_path(artifact_root, geometry_ref)
        summary = json.loads(cas_path(artifact_root, summary_ref).read_text())
        arrays_path = cas_path(artifact_root, arrays_ref)
        packed = json.loads(cas_path(artifact_root, packed_ref).read_text())
        require(summary.get("subject_id") == subject, "FIT8_PRESENTATION_BUNDLE_SUMMARY_SUBJECT_DRIFT")
        require(summary.get("array_sha256") == arrays_ref["content_sha256"], "FIT8_PRESENTATION_BUNDLE_ARRAY_BINDING_DRIFT")
        require(summary.get("source_sha256"), "FIT8_PRESENTATION_BUNDLE_SOURCE_BINDING_MISSING")
        require(summary.get("policy", {}).get("semantic_part_labels_used") is False, "FIT8_PRESENTATION_BUNDLE_SEMANTIC_LABEL_FORBIDDEN")
        require(summary.get("policy", {}).get("material_or_owner_ids_used_as_student_targets") is False, "FIT8_PRESENTATION_BUNDLE_PRIVILEGED_TARGET_FORBIDDEN")
        qualification = summary.get("renderer_subject_qualification") or {}
        require(
            qualification.get("status") == "PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE"
            and qualification.get("source_equivalent") is True
            and qualification.get("packed_replay_exact") is True
            and qualification.get("source_reference_replay_exact") is True,
            "FIT8_PRESENTATION_BUNDLE_RENDERER_QUALIFICATION_FAILED",
        )

        with np.load(geometry_path, allow_pickle=False) as geometry, np.load(arrays_path, allow_pickle=False) as presentation:
            for name in ("rest_vertices_source_world", "faces"):
                require(name in geometry.files, "FIT8_PRESENTATION_BUNDLE_GEOMETRY_ARRAY_MISSING:" + name)
            targets = build_presentation_targets_v1(
                rest_vertices=np.asarray(geometry["rest_vertices_source_world"]),
                faces=np.asarray(geometry["faces"]),
                normalization=normalization,
                presentation_arrays={name: np.asarray(presentation[name]) for name in presentation.files},
                packed_measurement=packed,
                contribution_floor=float(summary["policy"]["student_observable_contribution_floor"]),
            )

        target_path = output / subject / "V31_PRESENTATION_TARGETS_V1.npz"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target_path, **{field.name: getattr(targets, field.name) for field in fields(targets)})
        counts = dict(summary["counts"])
        aggregate["surface_faces"] += counts["surface_face_count"]
        aggregate["actions"] += counts["action_count"]
        aggregate["appearance_hits"] += int(len(targets.appearance_xyz))
        aggregate["observable_pairs"] += int(np.count_nonzero(targets.pair_valid))
        aggregate["pose_order_pairs"] += counts["pose_order_pair_count"]
        subject_rows.append(
            {
                "subject_id": subject,
                "platform_subject_id": platform_subject_id,
                "attempt_id": attempt_id,
                "source_sha256": summary["source_sha256"],
                "stage08_manifest_sha256": record08["content_sha256"],
                "stage47_manifest_sha256": record47["content_sha256"],
                "stage48_manifest_sha256": record48["content_sha256"],
                "normalization_sha256": normalization_ref["content_sha256"],
                "teacher_calibration_arrays_sha256": geometry_ref["content_sha256"],
                "presentation_truth_sha256": summary_ref["content_sha256"],
                "presentation_arrays_sha256": arrays_ref["content_sha256"],
                "packed_renderer_measurement_sha256": packed_ref["content_sha256"],
                "v31_target_file": str(target_path.relative_to(output)),
                "v31_target_sha256": sha256(target_path),
                "counts": {
                    **counts,
                    "v31_appearance_hit_count": int(len(targets.appearance_xyz)),
                    "v31_pair_count": int(len(targets.pair_xyz_a)),
                    "v31_observable_pair_count": int(np.count_nonzero(targets.pair_valid)),
                    "v31_mechanical_face_count": int(np.count_nonzero(targets.mechanical_valid)),
                },
                "mechanical_observability": "MECHANISM_CONDITIONAL__HETEROSCEDASTIC_OBJECTIVE_REQUIRED",
                "appearance_observability": "DIRECT_VISIBLE_SOURCE_EQUIVALENT_HIT_DOMAIN",
                "pair_probe_observability": "DIRECT_CONTRIBUTING_SAME_RAY_DOMAIN",
            }
        )

    regression_rows = []
    for subject in regression:
        binding = regression_bindings[subject]
        _, _, manifest48 = stage_manifest(
            api_root,
            artifact_root,
            binding["platform_subject_id"],
            binding["attempt_id"],
            STAGE48,
        )
        exclusion_ref = output_ref(manifest48, schema=EXCLUSION_SCHEMA)
        exclusion = json.loads(cas_path(artifact_root, exclusion_ref).read_text())
        require(
            exclusion.get("subject_id") == binding["producer_subject_id"]
            and exclusion.get("new_truth_produced") is False
            and exclusion.get("training_authorized") is False,
            "FIT8_PRESENTATION_BUNDLE_REGRESSION_EXCLUSION_DRIFT",
        )
        regression_rows.append(dict(binding))

    require(len(subject_rows) == 8 and len({row["subject_id"] for row in subject_rows}) == 8, "FIT8_PRESENTATION_BUNDLE_EIGHT_REQUIRED")
    require(aggregate["surface_faces"] > 0 and aggregate["appearance_hits"] > 0, "FIT8_PRESENTATION_BUNDLE_EMPTY_CORE_TRUTH")
    bundle = {
        "schema": "RealSaS.Fit8PresentationTruthBundle.v1",
        "campaign_id": observation["campaign_id"],
        "experiment_id": observation["experiment_id"],
        "source_code_sha": observation["source_code_sha"],
        "cohort": subject_rows,
        "regression_subjects": list(regression),
        "regression_bindings": regression_rows,
        "aggregate_counts": aggregate,
        "mechanical_objective_semantics": "CONDITIONAL_DISTRIBUTION_PARAMETERIZATION__NOT_DETERMINISTIC_HIDDEN_MECHANISM_CLASS",
        "semantic_part_taxonomy_used": False,
        "privileged_source_ids_are_student_inputs": False,
        "knight_is_training_subject": False,
        "training_launched": False,
        "model_promoted": False,
        "product_authority_changed": False,
        "status": "READY_FOR_LOFO_RECIPE_BINDING",
    }

    # V2 objective-exact student observability census.
    aggregate2 = dict(bundle["aggregate_counts"])
    aggregate2.update({"appearance_rows_total": 0, "observable_appearance_hits": 0, "pair_rows_total": 0, "observable_pairs": 0})
    for row in bundle["cohort"]:
        path = output / row["v31_target_file"]
        require(path.is_file() and sha256(path) == row["v31_target_sha256"], "FIT8_PRESENTATION_V2_TARGET_DRIFT")
        with np.load(path, allow_pickle=False) as z:
            appearance_valid = np.asarray(z["appearance_valid"], dtype=np.bool_)
            pair_valid = np.asarray(z["pair_valid"], dtype=np.bool_)
            mechanical_valid = np.asarray(z["mechanical_valid"], dtype=np.bool_)
        observable_appearance = int(np.count_nonzero(appearance_valid))
        observable_pair = int(np.count_nonzero(pair_valid))
        require(observable_appearance > 0, "FIT8_PRESENTATION_V2_NO_OBSERVABLE_APPEARANCE:" + row["subject_id"])
        counts = dict(row.get("counts") or {})
        counts["v31_appearance_row_count"] = int(len(appearance_valid))
        counts["v31_observable_appearance_hit_count"] = observable_appearance
        counts["v31_pair_count"] = int(len(pair_valid))
        counts["v31_observable_pair_count"] = observable_pair
        counts["v31_mechanical_face_count"] = int(np.count_nonzero(mechanical_valid))
        row["counts"] = counts
        aggregate2["appearance_rows_total"] += int(len(appearance_valid))
        aggregate2["observable_appearance_hits"] += observable_appearance
        aggregate2["pair_rows_total"] += int(len(pair_valid))
        aggregate2["observable_pairs"] += observable_pair

    require(aggregate2["observable_appearance_hits"] > 0, "FIT8_PRESENTATION_V2_EMPTY_OBSERVABLE_APPEARANCE")
    bundle["aggregate_counts"] = aggregate2
    bundle["bundle_builder_revision"] = "V2_OBJECTIVE_EXACT_OBSERVABILITY_COUNTS"
    bundle["subject_binding_revision"] = "V3_PLATFORM_UUID_TO_LOGICAL_KEY_FROM_EXACT_STAGE48_PAYLOAD"
    bundle["objective_projection"] = {
        "mechanical": ["nonrigid_response_conditional", "normalized_response_energy_conditional"],
        "appearance": ["source_equivalent_rgba__SINGLE_MEASURABLE_CONTRIBUTOR_RAYS_ONLY"],
        "pair_probe": ["occlusion_attenuation_probe__JOINTLY_CONTRIBUTING_SAME_RAY_ONLY"],
        "compiler_oracle_only": ["pose_conditioned_complete_ray_order", "geometric_front_order", "hit_removal_response", "full_source_over_stack"],
        "explicit_abstain": ["semantic_part_class", "same_drawable_owner", "attachment_class", "grip_class", "physical_stiffness", "physical_impulse"],
    }
    bundle["appearance_observability"] = "DETERMINISTIC_TARGET_VALID_IFF_EXACTLY_ONE_HIT_EXCEEDS_SEALED_SOURCE_OVER_CONTRIBUTION_FLOOR"
    bundle["pair_observability"] = "MECHANISM_CONDITIONAL_TARGET_VALID_IFF_BOTH_SAME_RAY_ENDPOINTS_MEASURABLY_CONTRIBUTE"
    bundle["hidden_or_overlap_factored_teacher_appearance_becomes_negative_or_valid"] = False

    manifest_path = output / "FIT8_PRESENTATION_TRUTH_BUNDLE_V1.json"
    write_json(manifest_path, bundle)
    print(
        json.dumps(
            {
                "bundle": str(manifest_path),
                "sha256": sha256(manifest_path),
                "observable_appearance_hits": aggregate2["observable_appearance_hits"],
                "observable_pairs": aggregate2["observable_pairs"],
                "builder_revision": bundle["bundle_builder_revision"],
                "subject_binding_revision": bundle["subject_binding_revision"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation", "artifact-root", "repo-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    build(
        json.loads(Path(args.observation).read_text()),
        api_root=args.api_root,
        artifact_root=args.artifact_root,
        repo_root=args.repo_root,
        output=args.output,
    )
