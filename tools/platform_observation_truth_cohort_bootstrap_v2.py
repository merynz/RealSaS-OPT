#!/usr/bin/env python3
from __future__ import annotations

"""Bootstrap the observation-qualified projection of the Corpus370 candidate universe.

The parent V2 family bundle always binds all 370 raw sources/provenance rows plus
explicit observation preparation PASS/FAIL outcomes. This bootstrap submits only
subjects whose sealed parent row says observation_preparation.status == PASS to the
observation-dependent Stage01..48 research DAG. Excluded subjects are reported
explicitly and remain available to observation-independent family courts.

No family authority, split, training readiness, model promotion, or product
authority is minted here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from tools import platform_truth_cohort_bootstrap_v1 as v1
from tools.platform_artifact_store import (
    fetch_google_drive_with_rclone,
    hydrate_row,
    sha256_file,
)
from tools.platform_cohort_bootstrap_v1 import (
    api,
    external_inventory_row,
    file_ref,
    json_bytes,
    receipt_index,
    require,
)
from tools.platform_release_snapshot import snapshot

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_SCHEMA = "RealSaS.PlatformObservationTruthCohortBootstrapDescriptor.v2"
BUNDLE_SCHEMA = "RealSaS.Corpus370PlatformFamilyInputBundle.v2"
RECEIPT_SCHEMA = "RealSaS.Corpus370PlatformFamilyInputBundleReceipt.v2"
CAMPAIGN_SCHEMA = "RealSaS.Corpus370ObservationTruthCampaign.v2"
SOURCE_SCENE_PROFILE = "SOURCE_SCENE_BIND_V2"
SOURCE_SCENE_ADMISSION = "CONTROLLED_FULL_SUBJECT_8VIEW_V2_SOURCE_SCENE"
SOURCE_LICENSE_PROFILE = "EXACT_SOURCE_PROVENANCE_V2"


def load(path: Path):
    return json.loads(path.read_text())


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_extract_verified_v2(zip_path: Path, descriptor: dict, out_root: Path):
    """Extract only an exact Corpus370 V2 family bundle.

    The historical V1 helper intentionally accepts only the V1 Stage01 bundle
    schema. Reusing it here would make the V2 bootstrap fail at runtime despite
    successful static compilation. Keep the V1 boundary immutable and verify the
    new transport contract explicitly instead.
    """
    bundle = dict(descriptor.get("cohort_bundle") or {})
    manifest_member = str(bundle.get("bootstrap_manifest_member") or "")
    receipt_member = str(bundle.get("bundle_receipt_member") or "")
    require(manifest_member == "BOOTSTRAP_MANIFEST.json", "OBS_TRUTH_V2_MANIFEST_MEMBER")
    require(receipt_member == "BUNDLE_RECEIPT.json", "OBS_TRUTH_V2_RECEIPT_MEMBER")
    require(len(str(bundle.get("sha256") or "")) == 64, "OBS_TRUTH_V2_BUNDLE_SHA")

    zip_path = zip_path.resolve()
    require(zip_path.is_file(), "OBS_TRUTH_V2_BUNDLE_MISSING")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "OBS_TRUTH_V2_DUPLICATE_MEMBER")
        require(manifest_member in names and receipt_member in names, "OBS_TRUTH_V2_REQUIRED_MEMBERS")

        receipt = json.loads(archive.read(receipt_member))
        require(receipt.get("schema") == RECEIPT_SCHEMA, "OBS_TRUTH_V2_RECEIPT_SCHEMA")
        require(receipt.get("training_authorized") is False, "OBS_TRUTH_V2_RECEIPT_PREAUTHORIZED")
        rows = list(receipt.get("files") or ())
        expected = {str(row["path"]): row for row in rows}
        require(int(receipt.get("file_count", -1)) == len(rows), "OBS_TRUTH_V2_RECEIPT_COUNT")
        require(len(expected) == len(rows), "OBS_TRUTH_V2_RECEIPT_DUPLICATE_PATH")
        require(set(names) == set(expected) | {receipt_member}, "OBS_TRUTH_V2_MEMBER_SET_DRIFT")
        require(manifest_member in expected, "OBS_TRUTH_V2_MANIFEST_NOT_RECEIPTED")
        require(
            receipt.get("manifest_sha256") == expected[manifest_member]["sha256"],
            "OBS_TRUTH_V2_MANIFEST_RECEIPT_DRIFT",
        )

        for name, row in expected.items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts, "OBS_TRUTH_V2_PATH_ESCAPE")
            data = archive.read(name)
            require(len(data) == int(row["size_bytes"]), "OBS_TRUTH_V2_SIZE_DRIFT:" + name)
            require(digest_bytes(data) == row["sha256"], "OBS_TRUTH_V2_SHA_DRIFT:" + name)

        manifest = json.loads(archive.read(manifest_member))
        require(manifest.get("schema") == BUNDLE_SCHEMA, "OBS_TRUTH_V2_BUNDLE_SCHEMA")
        require(manifest.get("training_authorized") is False, "OBS_TRUTH_V2_BUNDLE_PREAUTHORIZED")
        require(int(manifest.get("candidate_subject_count", -1)) == 370, "OBS_TRUTH_V2_BUNDLE_CANDIDATE_COUNT")
        require(
            int(manifest.get("observation_qualified_count", -1))
            + int(manifest.get("observation_failed_count", -1)) == 370,
            "OBS_TRUTH_V2_BUNDLE_OBSERVATION_PARTITION",
        )

        target = (out_root / str(bundle["sha256"])).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for name, row in expected.items():
            path = (target / name).resolve()
            require(path.is_relative_to(target), "OBS_TRUTH_V2_EXTRACT_ESCAPE")
            path.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if path.exists():
                require(
                    path.is_file()
                    and path.stat().st_size == len(data)
                    and sha256_file(path) == row["sha256"],
                    "OBS_TRUTH_V2_EXISTING_OBJECT_DRIFT:" + name,
                )
                continue
            fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                temporary.write_bytes(data)
                require(sha256_file(temporary) == row["sha256"], "OBS_TRUTH_V2_EXTRACT_HASH_DRIFT:" + name)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

        receipt_path = target / receipt_member
        receipt_data = archive.read(receipt_member)
        if receipt_path.exists():
            require(
                receipt_path.is_file() and digest_bytes(receipt_path.read_bytes()) == digest_bytes(receipt_data),
                "OBS_TRUTH_V2_EXISTING_RECEIPT_DRIFT",
            )
        else:
            receipt_path.write_bytes(receipt_data)
        return manifest, receipt, target


def subject_sections(subject: dict, bundle_root: Path, index: dict, overlay: dict) -> dict:
    sid = str(subject["subject_id"])
    observation_state = dict(subject.get("observation_preparation") or {})
    require(observation_state.get("status") == "PASS", "OBS_TRUTH_V2_SUBJECT_NOT_QUALIFIED:" + sid)

    camera_ref = file_ref(bundle_root, index, observation_state["camera_bundle"]["path"])
    camera = load(Path(camera_ref["path"]))
    raster_rows = []
    mask_rows = []
    for view in range(8):
        raster_rows.append(
            {"view_index": view, "image": file_ref(bundle_root, index, f"observations/{sid}/V{view}.png")}
        )
        mask_rows.append(
            {"view_index": view, "mask": file_ref(bundle_root, index, f"observations/{sid}/V{view}.mask.bin")}
        )

    license_cfg = subject["license"]
    sections = {
        "source_files": v1.source_files(subject, bundle_root, index),
        "source_license": {
            "profile": SOURCE_LICENSE_PROFILE,
            "source_pack": subject["pack_id"],
            "license_name": license_cfg["name"],
            "license_ref": license_cfg["ref"],
            "source_url_or_locator": license_cfg.get("source_url_or_locator", ""),
            "license_file_role": "license_file",
            "source_page_role": "source_page",
            **({"license_legalcode_role": "license_legalcode"} if subject.get("license_legalcode") else {}),
            "notes": "Corpus370 exact provenance bytes; projection and bootstrap mint no family authority.",
        },
        "source_audit": {
            "profile": SOURCE_SCENE_PROFILE,
            "subject_key": sid,
            "primary_source_role": "primary_source",
            "minimum_border_margin_px": 8,
            "matrix_tolerance": 1e-6,
            "source_rest_candidate": file_ref(bundle_root, index, observation_state["source_rest_candidate"]["path"]),
            "source_scene_snapshot": file_ref(bundle_root, index, observation_state["source_scene_snapshot"]["path"]),
            "iris_observation_admission": file_ref(bundle_root, index, observation_state["iris_observation_admission"]["path"]),
            "iris_observation_admission_required": True,
        },
        "admission": {"mode": SOURCE_SCENE_ADMISSION},
        "observation": {
            "camera_bundle": camera_ref,
            "source_rasters": raster_rows,
            "source_foreground_masks": mask_rows,
        },
        "normalization": {
            "center_xyz": camera["common_center_xyz"],
            "half_extent": camera["common_half_extent"],
            "coordinate_frame": "REALSAS_OBJECT_FRAME",
        },
    }
    require(not set(sections).intersection(overlay), "OBS_TRUTH_V2_OVERLAY_SHADOWS_INPUT")
    sections.update(overlay)
    return sections


def main(args) -> dict:
    repo = args.repo.resolve()
    descriptor_path = args.descriptor.resolve()
    descriptor = load(descriptor_path)
    require(descriptor.get("schema") == DESCRIPTOR_SCHEMA, "OBS_TRUTH_V2_DESCRIPTOR_SCHEMA")
    require(
        descriptor.get("purpose") == "RESEARCH" and descriptor.get("training_allowed") is False,
        "OBS_TRUTH_V2_RESEARCH_SCOPE",
    )

    campaign_contract = load((repo / descriptor["campaign_contract"]).resolve())
    require(campaign_contract.get("schema") == CAMPAIGN_SCHEMA, "OBS_TRUTH_V2_CAMPAIGN_SCHEMA")
    require(campaign_contract.get("canonical_branch") == "main", "OBS_TRUTH_V2_MAIN_REQUIRED")
    plan = load((repo / campaign_contract["compiler_plan"]).resolve())
    require(int(plan["stage_count"]) == int(campaign_contract["required_stage_count"]), "OBS_TRUTH_V2_PLAN_STAGE_COUNT")
    require(plan["stages"][-1]["id"] == campaign_contract["required_target_stage_id"], "OBS_TRUTH_V2_TARGET_STAGE")
    require("iris_fit" not in set(v1.plan_manifest_keys(plan)), "OBS_TRUTH_V2_TRAINING_PLAN_FORBIDDEN")
    require(
        set(v1.plan_manifest_keys(plan)).issubset(
            {
                "source_files", "source_license", "source_audit", "admission",
                "observation", "normalization", "truth_foundry", "presentation_truth",
            }
        ),
        "OBS_TRUTH_V2_UNDECLARED_PLAN_INPUT",
    )

    actor = str(descriptor.get("actor") or "corpus370-observation-truth-bootstrap-v2")
    overlay = dict(descriptor.get("additional_manifest_sections") or {})
    allowed = set(v1.plan_manifest_keys(plan))
    require(set(overlay).issubset(allowed), "OBS_TRUTH_V2_OVERLAY_UNDECLARED")

    hydration_root = (args.platform_root / "hydration" / "truth_cohorts").resolve()
    hydrated = hydrate_row(
        external_inventory_row(descriptor["cohort_bundle"]),
        cas_root=hydration_root,
        fetcher=fetch_google_drive_with_rclone,
        rclone_remote=args.rclone_remote,
        allow_remote=True,
    )
    require(hydrated["status"] in {"CAS_HIT", "HYDRATED"}, "OBS_TRUTH_V2_BUNDLE_HYDRATION")
    bundle_manifest, receipt, bundle_root = safe_extract_verified_v2(
        Path(hydrated["path"]), descriptor, args.authority_root / "truth_cohort_bundles_v2"
    )
    require(
        int(bundle_manifest.get("candidate_subject_count", -1))
        == int(campaign_contract["candidate_universe_subject_count"]) == 370,
        "OBS_TRUTH_V2_CANDIDATE_COUNT",
    )
    require(
        (bundle_manifest.get("parents") or {}).get("exact_source_roster", {}).get("sha256")
        == descriptor["exact_source_roster_sha256"],
        "OBS_TRUTH_V2_ROSTER_IDENTITY",
    )

    candidates = list(bundle_manifest.get("subjects") or [])
    require(len(candidates) == 370, "OBS_TRUTH_V2_CANDIDATE_ROWS")
    candidate_ids = [str(row.get("subject_id") or "") for row in candidates]
    require(all(candidate_ids) and len(set(candidate_ids)) == 370, "OBS_TRUTH_V2_CANDIDATE_KEYS")

    selected = [row for row in candidates if (row.get("observation_preparation") or {}).get("status") == "PASS"]
    excluded = [row for row in candidates if (row.get("observation_preparation") or {}).get("status") == "FAIL"]
    require(len(selected) + len(excluded) == 370, "OBS_TRUTH_V2_PROJECTION_PARTITION")
    require(len(selected) == int(bundle_manifest.get("observation_qualified_count", -1)), "OBS_TRUTH_V2_QUALIFIED_COUNT")
    require(len(excluded) == int(bundle_manifest.get("observation_failed_count", -1)), "OBS_TRUTH_V2_EXCLUDED_COUNT")
    projection = campaign_contract["cohort_projection"]
    require(
        int(projection["minimum_subject_count"]) <= len(selected) <= int(projection["maximum_subject_count"]),
        "OBS_TRUTH_V2_PROJECTED_COUNT_BOUNDS",
    )

    index = receipt_index(receipt)
    state = []
    for subject in selected:
        sid = str(subject["subject_id"])
        sections = subject_sections(subject, bundle_root, index, overlay)
        platform_id = v1.ensure_subject(args.api, sid, str(subject.get("display_name") or sid), actor)
        input_id = v1.seal_sections(
            args.api,
            platform_id,
            sections,
            actor,
            f"urn:realsas:corpus370-observation-truth-v2:{descriptor['campaign_name']}:{sid}",
        )
        state.append({"key": sid, "platform_subject_id": platform_id, "subject_input_id": input_id, "sections": sections})

    stage_params = v1.release_stage_parameters(descriptor, plan)
    release_request = snapshot(
        plan,
        name=descriptor["campaign_name"] + "__OBSERVATION_TRUTH_RESEARCH_RELEASE_V2",
        purpose="RESEARCH",
        created_by=actor,
        stage_parameters=stage_params,
    )
    release = api(args.api, "POST", "/v1/releases", release_request)
    release_id = release["ReleaseID"]
    campaign = api(
        args.api,
        "POST",
        "/v1/research/campaigns",
        {
            "name": descriptor["campaign_name"],
            "baseline_engine_release_id": release_id,
            "candidate_engine_release_id": release_id,
            "target_stage_id": campaign_contract["required_target_stage_id"],
            "cohort_subject_ids": [row["platform_subject_id"] for row in state],
            "regression_subject_ids": [],
            "created_by": actor,
        },
    )
    campaign_id = campaign["campaign_id"]

    run_bindings = []
    for row in state:
        run_id = f"CAMPAIGN_{campaign_id}_{row['platform_subject_id']}"
        run_root = args.authority_root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = {"run_id": run_id, "subject_id": row["key"], "execution_class": "WITNESS", **row["sections"]}
        manifest_path = run_root / "run_manifest.json"
        raw = json_bytes(manifest)
        if manifest_path.exists():
            require(manifest_path.read_bytes() == raw, "OBS_TRUTH_V2_RUN_MANIFEST_DRIFT:" + row["key"])
        else:
            manifest_path.write_bytes(raw)
        run_bindings.append(
            {
                "subject_id": row["platform_subject_id"],
                "subject_input_id": row["subject_input_id"],
                "run_manifest_path": str(manifest_path),
                "run_ledger_path": str(run_root / "ACTIVE_RUN_V2.json"),
            }
        )

    excluded_rows = []
    for subject in excluded:
        obs = dict(subject["observation_preparation"])
        excluded_rows.append(
            {
                "subject_id": subject["subject_id"],
                "source_sha256": subject["raw_source"]["sha256"],
                "error_type": obs.get("error_type"),
                "reason": obs.get("reason"),
                "failure_receipt": obs.get("failure_receipt"),
                "candidate_universe_membership_preserved": True,
            }
        )

    result = {
        "schema": "RealSaS.PlatformObservationTruthCohortBootstrapPreflight.v2",
        "descriptor": str(descriptor_path),
        "campaign_name": descriptor["campaign_name"],
        "campaign_id": campaign_id,
        "release_id": release_id,
        "target_stage_id": campaign_contract["required_target_stage_id"],
        "pipeline_plan_sha256": release_request["manifest"]["pipeline_plan_sha256"],
        "candidate_subject_count": 370,
        "cohort_subject_count": len(state),
        "excluded_subject_count": len(excluded_rows),
        "excluded_subjects": excluded_rows,
        "bundle_sha256": descriptor["cohort_bundle"]["sha256"],
        "exact_source_roster_sha256": descriptor["exact_source_roster_sha256"],
        "training_authorized": False,
        "truth_qualification_minted_by_bootstrap": False,
        "run_bindings": run_bindings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(json_bytes(result))
    if not args.no_submit:
        run = api(
            args.api,
            "POST",
            "/v1/research/campaigns/run",
            {
                "campaign_id": campaign_id,
                "members": run_bindings,
                "pipeline_plan_sha256": result["pipeline_plan_sha256"],
                "requested_by": actor,
            },
        )
        result["campaign_run"] = run
        args.out.write_bytes(json_bytes(result))

    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "release_id": release_id,
                "target_stage_id": result["target_stage_id"],
                "candidate_subject_count": 370,
                "cohort_subject_count": len(state),
                "excluded_subject_count": len(excluded_rows),
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--platform-root", type=Path, default=Path("/home/monster/realsas_platform"))
    parser.add_argument("--authority-root", type=Path, default=Path("/home/monster/realsas_authority"))
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--rclone-remote", default=os.environ.get("REALSAS_RCLONE_REMOTE", "gdrive"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()
    main(args)
