#!/usr/bin/env python3
from __future__ import annotations

"""Bootstrap a sealed truth-only cohort through the public Platform API.

Unlike the historical training cohort bootstrap, this path has no shared-IRIS/model
preregistration dependency. It exists for finite source/observation truth campaigns
whose released plan ends at Stage47 or Stage48. Input preparation still mints no
truth; only Platform Attempts and independent downstream courts may do so.
"""

import argparse
import json
import os
from pathlib import Path

from tools.platform_artifact_store import fetch_google_drive_with_rclone, hydrate_row, sha256_file
from tools.platform_release_snapshot import snapshot
from tools.platform_cohort_bootstrap_v1 import (
    api,
    external_inventory_row,
    file_ref,
    json_bytes,
    put_import,
    receipt_index,
    require,
    safe_extract_verified,
)

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_SCHEMA = "RealSaS.PlatformTruthCohortBootstrapDescriptor.v1"
SOURCE_SCENE_PROFILE = "SOURCE_SCENE_BIND_V2"
SOURCE_SCENE_ADMISSION = "CONTROLLED_FULL_SUBJECT_8VIEW_V2_SOURCE_SCENE"
SOURCE_LICENSE_PROFILE = "EXACT_SOURCE_PROVENANCE_V2"


def load(path: Path):
    return json.loads(path.read_text())


def ensure_subject(api_root: str, slug: str, display: str, actor: str):
    return api(api_root, "POST", "/v1/subjects", {"slug": slug, "display_name": display, "created_by": actor})["subject_id"]


def seal_sections(api_root: str, platform_subject_id: str, sections: dict, actor: str, origin: str):
    bindings = []
    for key in sorted(sections):
        imported = put_import(
            api_root,
            json_bytes(sections[key]),
            artifact_type="RealSaS.RunManifestSection",
            schema="v1",
            source_uri=f"{origin}:manifest:{key}",
            actor=actor,
        )
        bindings.append({"role": "manifest:" + key, "artifact_id": imported["artifact_id"]})
    return api(api_root, "POST", "/v1/subject-inputs", {"subject_id": platform_subject_id, "bindings": bindings, "created_by": actor})["subject_input_id"]


def plan_manifest_keys(plan: dict) -> list[str]:
    result = []
    for stage in plan.get("stages") or ():
        for key in stage.get("manifest_keys") or ():
            if key not in result:
                result.append(key)
    return result


def source_files(subject: dict, bundle_root: Path, index: dict) -> list[dict]:
    rows = []
    primary = subject["raw_source"]
    rows.append({
        "role": "primary_source",
        "path": str((bundle_root / primary["path"]).resolve()),
        "expected_sha256": primary["sha256"],
        "expected_size_bytes": int(primary["size_bytes"]),
        "source_kind": "PRIMARY",
    })
    for number, dep in enumerate(subject.get("source_dependencies") or ()):
        rows.append({
            "role": f"source_dependency_{number:04d}",
            "path": str((bundle_root / dep["path"]).resolve()),
            "expected_sha256": dep["sha256"],
            "expected_size_bytes": int(dep["size_bytes"]),
            "source_kind": "DEPENDENCY",
        })
    license_row = subject["license_bytes"]
    rows.append({
        "role": "license_file",
        "path": str((bundle_root / license_row["path"]).resolve()),
        "expected_sha256": license_row["sha256"],
        "expected_size_bytes": int(license_row["size_bytes"]),
        "source_kind": "LICENSE",
    })
    page = subject["source_page_snapshot"]
    rows.append({
        "role": "source_page",
        "path": str((bundle_root / page["path"]).resolve()),
        "expected_sha256": page["sha256"],
        "expected_size_bytes": int(page["size_bytes"]),
        "source_kind": "PROVENANCE",
    })
    legal = subject.get("license_legalcode")
    if legal:
        rows.append({
            "role": "license_legalcode",
            "path": str((bundle_root / legal["path"]).resolve()),
            "expected_sha256": legal["sha256"],
            "expected_size_bytes": int(legal["size_bytes"]),
            "source_kind": "LICENSE",
        })
    prefix = f"observations/{subject['subject_id']}/"
    for relative, item in sorted(index.items()):
        if not relative.startswith(prefix):
            continue
        name = relative[len(prefix):]
        rows.append({
            "role": "observation_evidence_" + name.replace(".", "_").lower(),
            "path": str((bundle_root / relative).resolve()),
            "expected_sha256": item["sha256"],
            "expected_size_bytes": int(item["size_bytes"]),
            "source_kind": "EVIDENCE",
        })
    roles = [row["role"] for row in rows]
    require(len(set(roles)) == len(roles), "TRUTH_BOOTSTRAP_SOURCE_ROLE_DUPLICATE")
    return rows


def subject_sections(subject: dict, bundle_root: Path, index: dict, overlay: dict) -> dict:
    sid = str(subject["subject_id"])
    scene = subject["source_scene_evidence"]
    camera_ref = file_ref(bundle_root, index, scene["camera_bundle"]["path"])
    camera = load(Path(camera_ref["path"]))
    raster_rows = []
    mask_rows = []
    for view in range(8):
        raster_rows.append({"view_index": view, "image": file_ref(bundle_root, index, f"observations/{sid}/V{view}.png")})
        mask_rows.append({"view_index": view, "mask": file_ref(bundle_root, index, f"observations/{sid}/V{view}.mask.bin")})
    license_cfg = subject["license"]
    sections = {
        "source_files": source_files(subject, bundle_root, index),
        "source_license": {
            "profile": SOURCE_LICENSE_PROFILE,
            "source_pack": subject["pack_id"],
            "license_name": license_cfg["name"],
            "license_ref": license_cfg["ref"],
            "source_url_or_locator": license_cfg.get("source_url_or_locator", ""),
            "license_file_role": "license_file",
            "source_page_role": "source_page",
            **({"license_legalcode_role": "license_legalcode"} if subject.get("license_legalcode") else {}),
            "notes": "Corpus370 exact provenance bytes; preparation does not mint training or family authority.",
        },
        "source_audit": {
            "profile": SOURCE_SCENE_PROFILE,
            "subject_key": sid,
            "primary_source_role": "primary_source",
            "minimum_border_margin_px": 8,
            "matrix_tolerance": 1e-6,
            "source_rest_candidate": file_ref(bundle_root, index, scene["source_rest_candidate"]["path"]),
            "source_scene_snapshot": file_ref(bundle_root, index, scene["source_scene_snapshot"]["path"]),
            "iris_observation_admission": file_ref(bundle_root, index, scene["iris_observation_admission"]["path"]),
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
    require(not set(sections).intersection(overlay), "TRUTH_BOOTSTRAP_OVERLAY_SHADOWS_INPUT")
    sections.update(overlay)
    return sections


def release_stage_parameters(descriptor: dict, plan: dict) -> dict:
    known = {stage["id"] for stage in plan["stages"]}
    result = {"schema": "RealSaS.StageReleaseParameters.v1", "stages": {}}
    for stage_id, parameters in (descriptor.get("additional_stage_parameters") or {}).items():
        require(stage_id in known, "TRUTH_BOOTSTRAP_STAGE_PARAMETER_UNKNOWN_STAGE")
        require(isinstance(parameters, dict), "TRUTH_BOOTSTRAP_STAGE_PARAMETER_OBJECT")
        result["stages"][stage_id] = parameters
    return result


def main(args) -> dict:
    repo = args.repo.resolve()
    descriptor_path = args.descriptor.resolve()
    descriptor = load(descriptor_path)
    require(descriptor.get("schema") == DESCRIPTOR_SCHEMA, "TRUTH_BOOTSTRAP_DESCRIPTOR_SCHEMA")
    require(descriptor.get("purpose") == "RESEARCH" and descriptor.get("training_allowed") is False, "TRUTH_BOOTSTRAP_RESEARCH_SCOPE")
    campaign_contract = load((repo / descriptor["campaign_contract"]).resolve())
    require(campaign_contract.get("canonical_branch") == "main", "TRUTH_BOOTSTRAP_MAIN_REQUIRED")
    plan = load((repo / campaign_contract["compiler_plan"]).resolve())
    require(int(plan["stage_count"]) == int(campaign_contract["required_stage_count"]), "TRUTH_BOOTSTRAP_PLAN_STAGE_COUNT")
    require(plan["stages"][-1]["id"] == campaign_contract["required_target_stage_id"], "TRUTH_BOOTSTRAP_TARGET_STAGE")
    require("iris_fit" not in set(plan_manifest_keys(plan)), "TRUTH_BOOTSTRAP_TRAINING_PLAN_FORBIDDEN")
    require(set(plan_manifest_keys(plan)).issubset({"source_files", "source_license", "source_audit", "admission", "observation", "normalization", "truth_foundry", "presentation_truth"}), "TRUTH_BOOTSTRAP_UNDECLARED_PLAN_INPUT")

    actor = str(descriptor.get("actor") or "corpus370-truth-bootstrap")
    overlay = dict(descriptor.get("additional_manifest_sections") or {})
    allowed = set(plan_manifest_keys(plan))
    require(set(overlay).issubset(allowed), "TRUTH_BOOTSTRAP_OVERLAY_UNDECLARED")
    hydration_root = (args.platform_root / "hydration" / "truth_cohorts").resolve()
    hydrated = hydrate_row(
        external_inventory_row(descriptor["cohort_bundle"]),
        cas_root=hydration_root,
        fetcher=fetch_google_drive_with_rclone,
        rclone_remote=args.rclone_remote,
        allow_remote=True,
    )
    require(hydrated["status"] in {"CAS_HIT", "HYDRATED"}, "TRUTH_BOOTSTRAP_BUNDLE_HYDRATION")
    bundle_manifest, receipt, bundle_root = safe_extract_verified(
        Path(hydrated["path"]), descriptor, args.authority_root / "truth_cohort_bundles"
    )
    require(int(bundle_manifest.get("subject_count", -1)) == int(campaign_contract["expected_subject_count"]), "TRUTH_BOOTSTRAP_BUNDLE_SUBJECT_COUNT")
    require((bundle_manifest.get("parents") or {}).get("exact_source_roster", {}).get("sha256") == descriptor["exact_source_roster_sha256"], "TRUTH_BOOTSTRAP_ROSTER_IDENTITY")
    subjects = list(bundle_manifest.get("subjects") or [])
    keys = [str(row.get("subject_id") or "") for row in subjects]
    require(all(keys) and len(keys) == len(set(keys)), "TRUTH_BOOTSTRAP_SUBJECT_KEYS")
    index = receipt_index(receipt)

    state = []
    for subject in subjects:
        sid = str(subject["subject_id"])
        sections = subject_sections(subject, bundle_root, index, overlay)
        platform_id = ensure_subject(args.api, sid, str(subject.get("display_name") or sid), actor)
        input_id = seal_sections(args.api, platform_id, sections, actor, f"urn:realsas:corpus370-truth:{descriptor['campaign_name']}:{sid}")
        state.append({"key": sid, "platform_subject_id": platform_id, "subject_input_id": input_id, "sections": sections})

    stage_params = release_stage_parameters(descriptor, plan)
    release_request = snapshot(
        plan,
        name=descriptor["campaign_name"] + "__TRUTH_RESEARCH_RELEASE",
        purpose="RESEARCH",
        created_by=actor,
        stage_parameters=stage_params,
    )
    release = api(args.api, "POST", "/v1/releases", release_request)
    release_id = release["ReleaseID"]
    campaign = api(args.api, "POST", "/v1/research/campaigns", {
        "name": descriptor["campaign_name"],
        "baseline_engine_release_id": release_id,
        "candidate_engine_release_id": release_id,
        "target_stage_id": campaign_contract["required_target_stage_id"],
        "cohort_subject_ids": [row["platform_subject_id"] for row in state],
        "regression_subject_ids": [],
        "created_by": actor,
    })
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
            require(manifest_path.read_bytes() == raw, "TRUTH_BOOTSTRAP_RUN_MANIFEST_DRIFT:" + row["key"])
        else:
            manifest_path.write_bytes(raw)
        run_bindings.append({
            "subject_id": row["platform_subject_id"],
            "subject_input_id": row["subject_input_id"],
            "run_manifest_path": str(manifest_path),
            "run_ledger_path": str(run_root / "ACTIVE_RUN_V2.json"),
        })

    result = {
        "schema": "RealSaS.PlatformTruthCohortBootstrapPreflight.v1",
        "descriptor": str(descriptor_path),
        "campaign_name": descriptor["campaign_name"],
        "campaign_id": campaign_id,
        "release_id": release_id,
        "target_stage_id": campaign_contract["required_target_stage_id"],
        "pipeline_plan_sha256": release_request["manifest"]["pipeline_plan_sha256"],
        "cohort_subject_count": len(state),
        "bundle_sha256": descriptor["cohort_bundle"]["sha256"],
        "exact_source_roster_sha256": descriptor["exact_source_roster_sha256"],
        "training_authorized": False,
        "truth_qualification_minted_by_bootstrap": False,
        "run_bindings": run_bindings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(json_bytes(result))
    if not args.no_submit:
        run = api(args.api, "POST", "/v1/research/campaigns/run", {
            "campaign_id": campaign_id,
            "members": run_bindings,
            "pipeline_plan_sha256": result["pipeline_plan_sha256"],
            "requested_by": actor,
        })
        result["campaign_run"] = run
        args.out.write_bytes(json_bytes(result))
    print(json.dumps({k: result[k] for k in ("campaign_id", "release_id", "target_stage_id", "cohort_subject_count", "training_authorized")}, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--platform-root", type=Path, default=Path("/home/monster/realsas_platform"))
    parser.add_argument("--authority-root", type=Path, default=Path("/home/monster/realsas_authority"))
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--rclone-remote", default=os.environ.get("REALSAS_RCLONE_DRIVE_REMOTE", "gdrive"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-submit", action="store_true")
    main(parser.parse_args())
