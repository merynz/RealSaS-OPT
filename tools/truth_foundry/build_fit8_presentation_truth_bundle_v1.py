#!/usr/bin/env python3
"""Build an immutable FIT8 presentation/response truth bundle from Platform CAS.

The input is a completed Stage48 campaign observation.  Every subject is rebound
to its exact Attempt and the exact Stage08/Stage47/Stage48 StageResultManifest.
The bundle contains rich raw truth references plus a V3.1 objective projection.
It does not launch training or promote a model.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path, PurePosixPath
import shutil

import numpy as np

from models.iris.v31.presentation_objective_v1 import build_presentation_targets_v1
from tools.platform_cohort_bootstrap_v1 import api
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256

STAGE08 = "08_NORMALIZATION_DOMAIN_QUALIFIED"
STAGE47 = "47_SOURCE_RESPONSE_CALIBRATION"
STAGE48 = "48_PRESENTATION_RESPONSE_TRUTH"


def require(value, code):
    if not value:
        raise ValueError(code)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def cas_path(root, ref):
    key = PurePosixPath(ref["storage_key"])
    require(not key.is_absolute() and ".." not in key.parts, "FIT8_PRESENTATION_BUNDLE_CAS_PATH_ESCAPE")
    path = (Path(root) / key).resolve()
    require(path.is_relative_to(Path(root).resolve()), "FIT8_PRESENTATION_BUNDLE_CAS_PATH_ESCAPE")
    require(
        path.is_file()
        and path.stat().st_size == ref["size_bytes"]
        and sha256(path) == ref["content_sha256"],
        "FIT8_PRESENTATION_BUNDLE_CAS_BYTES_DRIFT",
    )
    return path


def stage_manifest(api_root, artifact_root, subject_id, attempt_id, stage_id):
    records = api(
        api_root,
        "GET",
        f"/v1/subjects/{subject_id}/artifacts?artifact_type=RealSaS.StageResultManifest&schema_version=v1&limit=500",
    )
    rows = [
        row
        for row in records
        if row["source_kind"] == "ATTEMPT"
        and row["source_id"] == attempt_id
        and row["role"] == "stage:" + stage_id
    ]
    require(len(rows) == 1, "FIT8_PRESENTATION_BUNDLE_STAGE_BINDING_CARDINALITY:" + stage_id)
    binding = rows[0]
    record = api(api_root, "GET", "/v1/artifacts/" + binding["artifact_id"])
    require(
        record["content_sha256"] == binding["content_sha256"]
        and record["producer_contract"] == stage_id,
        "FIT8_PRESENTATION_BUNDLE_STAGE_REGISTRY_DRIFT:" + stage_id,
    )
    manifest_path = cas_path(artifact_root, record)
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema") == "RealSaS.StageResultManifest.v1"
        and manifest.get("stage_id") == stage_id
        and manifest.get("compiler_status") == "PASS",
        "FIT8_PRESENTATION_BUNDLE_STAGE_MANIFEST_DRIFT:" + stage_id,
    )
    return binding, record, manifest


def output_ref(manifest, *, schema=None, name=None):
    rows = list(manifest.get("outputs") or ())
    if schema is not None:
        rows = [row for row in rows if row.get("payload_schema") == schema]
    if name is not None:
        rows = [row for row in rows if PurePosixPath(row.get("relative_path") or "").name == name]
    require(len(rows) == 1, "FIT8_PRESENTATION_BUNDLE_OUTPUT_CARDINALITY:" + str(schema or name))
    return rows[0]


def copy_ref(root, ref, destination):
    source = cas_path(root, ref)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, temp)
    require(sha256(temp) == ref["content_sha256"], "FIT8_PRESENTATION_BUNDLE_COPY_DRIFT")
    temp.replace(destination)
    return {"path": str(destination.resolve()), "sha256": ref["content_sha256"], "size_bytes": ref["size_bytes"], "schema": ref["payload_schema"]}


def build(observation, *, api_root, artifact_root, repo_root, output):
    require(
        observation.get("schema") == "RealSaS.PresentationTruthCampaignObservation.v1"
        and observation.get("all_stage48_outputs_exported") is True
        and observation.get("goal_complete") is True
        and not observation.get("errors"),
        "FIT8_PRESENTATION_BUNDLE_CAMPAIGN_NOT_COMPLETE",
    )
    repo = Path(repo_root).resolve()
    campaign_contract = json.loads((repo / "canonical/FIT8_PRESENTATION_TRUTH_CAMPAIGN_V1.json").read_text())
    expected = tuple(campaign_contract["cohort_subject_keys"])
    regression = tuple(campaign_contract["regression_subject_keys"])
    attempts = {row["subject_id"]: row for row in observation["attempts"]}
    require(set(attempts) == set(expected) | set(regression), "FIT8_PRESENTATION_BUNDLE_SUBJECT_SET_DRIFT")
    require(all(row["state"] in {"COMPLETED", "PASS"} for row in attempts.values()), "FIT8_PRESENTATION_BUNDLE_ATTEMPT_NOT_COMPLETE")

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
        attempt_id = attempts[subject]["attempt_id"]
        _, record08, manifest08 = stage_manifest(api_root, artifact_root, subject, attempt_id, STAGE08)
        _, record47, manifest47 = stage_manifest(api_root, artifact_root, subject, attempt_id, STAGE47)
        _, record48, manifest48 = stage_manifest(api_root, artifact_root, subject, attempt_id, STAGE48)
        normalization_ref = output_ref(manifest08, schema="RealSaS.NormalizationDomainIR.v1")
        geometry_ref = output_ref(manifest47, schema="RealSaS.TeacherCalibrationArrays.v2")
        summary_ref = output_ref(manifest48, schema="RealSaS.PresentationResponseTruthCandidate.v1")
        arrays_ref = output_ref(manifest48, schema="RealSaS.PresentationResponseArrays.v1")
        packed_ref = output_ref(manifest48, schema="RealSaS.SourceRayAppearanceMeasurementCandidate.v1", name="PACKED_A.json")

        norm_path = cas_path(artifact_root, normalization_ref)
        geometry_path = cas_path(artifact_root, geometry_ref)
        summary_path = cas_path(artifact_root, summary_ref)
        arrays_path = cas_path(artifact_root, arrays_ref)
        packed_path = cas_path(artifact_root, packed_ref)
        normalization = json.loads(norm_path.read_text())
        summary = json.loads(summary_path.read_text())
        packed = json.loads(packed_path.read_text())
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
        target_sha = sha256(target_path)
        counts = dict(summary["counts"])
        aggregate["surface_faces"] += counts["surface_face_count"]
        aggregate["actions"] += counts["action_count"]
        aggregate["appearance_hits"] += int(len(targets.appearance_xyz))
        aggregate["observable_pairs"] += int(np.count_nonzero(targets.pair_valid))
        aggregate["pose_order_pairs"] += counts["pose_order_pair_count"]
        subject_rows.append(
            {
                "subject_id": subject,
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
                "v31_target_sha256": target_sha,
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

    for subject in regression:
        attempt_id = attempts[subject]["attempt_id"]
        _, _, manifest48 = stage_manifest(api_root, artifact_root, subject, attempt_id, STAGE48)
        exclusion_ref = output_ref(manifest48, schema="RealSaS.PresentationTruthRegressionExclusion.v1")
        exclusion = json.loads(cas_path(artifact_root, exclusion_ref).read_text())
        require(
            exclusion.get("subject_id") == subject
            and exclusion.get("new_truth_produced") is False
            and exclusion.get("training_authorized") is False,
            "FIT8_PRESENTATION_BUNDLE_REGRESSION_EXCLUSION_DRIFT",
        )

    require(len(subject_rows) == 8 and len({row["subject_id"] for row in subject_rows}) == 8, "FIT8_PRESENTATION_BUNDLE_EIGHT_REQUIRED")
    require(aggregate["surface_faces"] > 0 and aggregate["appearance_hits"] > 0, "FIT8_PRESENTATION_BUNDLE_EMPTY_CORE_TRUTH")
    bundle = {
        "schema": "RealSaS.Fit8PresentationTruthBundle.v1",
        "campaign_id": observation["campaign_id"],
        "experiment_id": observation["experiment_id"],
        "source_code_sha": observation["source_code_sha"],
        "cohort": subject_rows,
        "regression_subjects": list(regression),
        "aggregate_counts": aggregate,
        "objective_projection": {
            "mechanical": ["nonrigid_response_conditional", "response_energy_conditional"],
            "appearance": ["source_equivalent_rgba"],
            "pair_probe": ["front_probe", "occlusion_attenuation_probe"],
            "compiler_oracle_only": ["pose_conditioned_complete_ray_order", "hit_removal_response", "full_source_over_stack"],
            "explicit_abstain": ["semantic_part_class", "same_drawable_owner", "attachment_class", "grip_class", "physical_stiffness", "physical_impulse"],
        },
        "mechanical_objective_semantics": "CONDITIONAL_DISTRIBUTION_PARAMETERIZATION__NOT_DETERMINISTIC_HIDDEN_MECHANISM_CLASS",
        "semantic_part_taxonomy_used": False,
        "privileged_source_ids_are_student_inputs": False,
        "knight_is_training_subject": False,
        "training_launched": False,
        "model_promoted": False,
        "product_authority_changed": False,
        "status": "READY_FOR_LOFO_RECIPE_BINDING",
    }
    manifest_path = output / "FIT8_PRESENTATION_TRUTH_BUNDLE_V1.json"
    write_json(manifest_path, bundle)
    print(json.dumps({"bundle": str(manifest_path), "sha256": sha256(manifest_path), "aggregate_counts": aggregate}, sort_keys=True), flush=True)
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
