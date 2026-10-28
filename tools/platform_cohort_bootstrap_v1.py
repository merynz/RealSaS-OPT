#!/usr/bin/env python3
"""Bootstrap a sealed multi-subject research campaign through the public Platform API.

The descriptor is data. This tool contains no subject-name or cohort-size branches.
It verifies external bundle bytes, materializes immutable run-manifest inputs, seals
SubjectInputs and one subject-agnostic EngineRelease, starts the campaign, then
writes exact campaign-bound run manifests and submits the campaign.

Raw import/hydration never mints qualification. Compiler stages remain authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from compiler.realsas_compiler_services.orchestrator.adapters.shared_iris_execution_v1 import value_hash
from tools.platform_artifact_store import fetch_google_drive_with_rclone, hydrate_row, sha256_file
from tools.platform_release_snapshot import snapshot

DESCRIPTOR_SCHEMA = "RealSaS.PlatformCohortBootstrapDescriptor.v1"
BUNDLE_SCHEMA_SUFFIX = "PlatformStage01InputBundle.v1"
RECEIPT_SCHEMA_SUFFIX = "PlatformStage01InputBundleReceipt.v1"
SOURCE_SCENE_PROFILE = "SOURCE_SCENE_BIND_V2"
SOURCE_SCENE_ADMISSION = "CONTROLLED_FULL_SUBJECT_8VIEW_V2_SOURCE_SCENE"
SOURCE_LICENSE_PROFILE = "EXACT_SOURCE_PROVENANCE_V2"
SHARED_PROFILE = "SHARED_IRIS_IMAGE_DERIVED_FSTAR_V1"
ACTOR_DEFAULT = "platform-cohort-bootstrap"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def api(api_root: str, method: str, path: str, payload=None, raw: bytes | None = None):
    if raw is not None and payload is not None:
        raise RuntimeError("API_BODY_AMBIGUOUS")
    body = raw
    headers = {}
    if payload is not None:
        body = json_bytes(payload)
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        headers["Content-Type"] = "application/octet-stream"
    req = urllib.request.Request(api_root.rstrip("/") + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"PLATFORM_API_{error.code}:{path}:{detail}") from error
    return {} if not data else json.loads(data)


def put_import(api_root: str, data: bytes, *, artifact_type: str, schema: str, source_uri: str, actor: str):
    digest = digest_bytes(data)
    obj = api(api_root, "PUT", f"/v1/artifacts/bytes?sha256={digest}", raw=data)
    return api(api_root, "POST", "/v1/artifacts/import", {
        "artifact_type": artifact_type,
        "schema_version": schema,
        "object": obj,
        "source_uri": source_uri,
        "created_by": actor,
    })


def verify_descriptor(descriptor: dict, campaign: dict):
    require(descriptor.get("schema") == DESCRIPTOR_SCHEMA, "COHORT_BOOTSTRAP_DESCRIPTOR_SCHEMA")
    require(descriptor.get("purpose") == "RESEARCH", "COHORT_BOOTSTRAP_RESEARCH_PURPOSE_REQUIRED")
    require(campaign.get("canonical_branch") == "main", "COHORT_BOOTSTRAP_CAMPAIGN_MAIN_REQUIRED")
    rules = descriptor.get("rules") or {}
    for key in (
        "descriptor_is_data_not_subject_specific_code",
        "subject_specific_code_forbidden",
        "all_bundle_members_verified_against_bundle_receipt",
        "observation_binding_must_match_shared_cohort_manifest",
    ):
        require(rules.get(key) is True, "COHORT_BOOTSTRAP_RULE_REQUIRED:" + key)
    defaults = descriptor.get("source_license_defaults") or {}
    require(bool(defaults.get("license_name")) and bool(defaults.get("license_ref")),
            "COHORT_BOOTSTRAP_LICENSE_IDENTITY_REQUIRED")


def external_inventory_row(ref: dict) -> dict:
    require(ref.get("provider") == "google_drive", "COHORT_BOOTSTRAP_PROVIDER_UNSUPPORTED")
    return {
        "name": str(ref["name"]),
        "sha256": str(ref["sha256"]),
        "size_bytes": int(ref["size_bytes"]),
        "source": {"provider": str(ref["provider"]), "file_id": str(ref["file_id"])},
    }


def safe_extract_verified(zip_path: Path, descriptor: dict, out_root: Path):
    bundle = descriptor["cohort_bundle"]
    with zipfile.ZipFile(zip_path) as archive:
        manifest_member = str(bundle["bootstrap_manifest_member"])
        receipt_member = str(bundle["bundle_receipt_member"])
        names = archive.namelist()
        require(len(names) == len(set(names)), "COHORT_BUNDLE_DUPLICATE_MEMBER")
        receipt = json.loads(archive.read(receipt_member))
        require(str(receipt.get("schema") or "").endswith(RECEIPT_SCHEMA_SUFFIX),
                "COHORT_BUNDLE_RECEIPT_SCHEMA")
        rows = list(receipt.get("files") or ())
        expected = {str(row["path"]): row for row in rows}
        require(int(receipt.get("file_count", -1)) == len(rows), "COHORT_BUNDLE_RECEIPT_COUNT")
        require(set(names) == set(expected) | {receipt_member}, "COHORT_BUNDLE_MEMBER_SET_DRIFT")
        require(receipt.get("manifest_sha256") == expected[manifest_member]["sha256"],
                "COHORT_BUNDLE_MANIFEST_RECEIPT_DRIFT")
        for name, row in expected.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "COHORT_BUNDLE_PATH_ESCAPE")
            data = archive.read(name)
            require(len(data) == int(row["size_bytes"]), "COHORT_BUNDLE_SIZE_DRIFT:" + name)
            require(digest_bytes(data) == row["sha256"], "COHORT_BUNDLE_SHA_DRIFT:" + name)
        manifest = json.loads(archive.read(manifest_member))
        require(str(manifest.get("schema") or "").endswith(BUNDLE_SCHEMA_SUFFIX),
                "COHORT_BUNDLE_MANIFEST_SCHEMA")
        target = out_root / str(bundle["sha256"])
        target.mkdir(parents=True, exist_ok=True)
        for name, row in expected.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if path.exists():
                require(path.is_file() and path.stat().st_size == len(data) and sha256_file(path) == row["sha256"],
                        "COHORT_BUNDLE_EXISTING_OBJECT_DRIFT:" + name)
            else:
                fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
                os.close(fd)
                tmp = Path(temp)
                try:
                    tmp.write_bytes(data)
                    require(sha256_file(tmp) == row["sha256"], "COHORT_BUNDLE_EXTRACT_HASH_DRIFT:" + name)
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
        receipt_path = target / receipt_member
        if receipt_path.exists():
            require(sha256_file(receipt_path) == digest_bytes(archive.read(receipt_member)),
                    "COHORT_BUNDLE_RECEIPT_EXISTING_DRIFT")
        else:
            receipt_path.write_bytes(archive.read(receipt_member))
        return manifest, receipt, target


def receipt_index(receipt: dict):
    return {str(row["path"]): row for row in receipt["files"]}


def file_ref(root: Path, index: dict, relative: str):
    row = index[relative]
    path = (root / relative).resolve()
    require(path.is_file(), "COHORT_FILE_MISSING:" + relative)
    require(sha256_file(path) == row["sha256"], "COHORT_FILE_HASH_DRIFT:" + relative)
    return {"path": str(path), "sha256": row["sha256"]}


def derive_observation_binding(proof: dict, proof_sha: str) -> str:
    payload = {
        "proof_sha256": proof_sha,
        "source_sha256": proof["source_sha256"],
        "camera_sha256": proof["camera_sha256"],
        "views": proof["views"],
    }
    return digest_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def validate_shared_inputs(bundle_manifest: dict, bundle_root: Path, cohort_manifest: dict, policy: dict):
    require(cohort_manifest.get("schema") == "RealSaS.SharedIrisCohortShards.v1",
            "COHORT_SHARED_IRIS_MANIFEST_SCHEMA")
    require(policy.get("schema") == "RealSaS.SharedIrisTrainingPolicy.v1",
            "COHORT_SHARED_IRIS_POLICY_SCHEMA")
    require(cohort_manifest.get("policy_sha256") == value_hash(policy),
            "COHORT_SHARED_IRIS_POLICY_VALUE_DRIFT")
    subjects = list(bundle_manifest.get("subjects") or ())
    shared = {row["subject_id"]: row for row in cohort_manifest.get("subjects") or ()}
    require(len(shared) == len(subjects) == int(policy["subject_count"]),
            "COHORT_SHARED_IRIS_SUBJECT_COUNT_DRIFT")
    for subject in subjects:
        key = str(subject["subject_id"])
        evidence = subject["source_scene_evidence"]["iris_observation_admission"]
        proof = load(bundle_root / evidence["path"])
        binding = derive_observation_binding(proof, str(evidence["sha256"]))
        require(key in shared and shared[key]["observation_binding"] == binding,
                "COHORT_SHARED_IRIS_OBSERVATION_BINDING_DRIFT:" + key)


def section_source_files(subject: dict, root: Path, index: dict):
    key = str(subject["subject_id"])
    rows = []
    primary = subject["raw_source"]
    rows.append({"role": "primary_source", "path": str((root / primary["path"]).resolve()),
                 "expected_sha256": primary["sha256"], "expected_size_bytes": primary["size_bytes"],
                 "source_kind": "PRIMARY"})
    license_row = subject["license_bytes"]
    rows.append({"role": "license_file", "path": str((root / license_row["path"]).resolve()),
                 "expected_sha256": license_row["sha256"], "expected_size_bytes": license_row["size_bytes"],
                 "source_kind": "LICENSE"})
    page = subject["source_page_snapshot"]
    rows.append({"role": "source_page", "path": str((root / page["path"]).resolve()),
                 "expected_sha256": page["sha256"], "expected_size_bytes": page["size_bytes"],
                 "source_kind": "PROVENANCE"})
    prefix = f"observations/{key}/"
    fixed = {
        "camera_bundle.json": "observation_camera_bundle",
        "IRIS_OBSERVATION_ADMISSION.json": "iris_observation_admission",
        "SOURCE_REST_RENDER_CANDIDATE.json": "source_rest_candidate",
        "SOURCE_SCENE_SNAPSHOT.json": "source_scene_snapshot",
    }
    for relative, row in sorted(index.items()):
        if not relative.startswith(prefix):
            continue
        name = relative[len(prefix):]
        role = fixed.get(name)
        if role is None:
            if name.endswith(".png"):
                role = "observation_raster_" + name[:-4].lower()
            elif name.endswith(".mask.bin"):
                role = "observation_mask_" + name[:-9].lower()
            else:
                role = "observation_evidence_" + name.replace(".", "_").lower()
        rows.append({"role": role, "path": str((root / relative).resolve()),
                     "expected_sha256": row["sha256"], "expected_size_bytes": row["size_bytes"],
                     "source_kind": "EVIDENCE"})
    require(len({row["role"] for row in rows}) == len(rows), "COHORT_SOURCE_ROLE_DUPLICATE:" + key)
    return rows


def make_cohort_manifest_sections(subject: dict, root: Path, index: dict,
                                  cohort_manifest_path: Path, cohort_manifest_sha: str,
                                  policy_path: Path, policy_sha: str,
                                  prereg_path: Path, prereg_sha: str,
                                  license_defaults: dict):
    key = str(subject["subject_id"])
    scene = subject["source_scene_evidence"]
    camera_ref = file_ref(root, index, scene["camera_bundle"]["path"])
    camera = load(Path(camera_ref["path"]))
    raster_rows, mask_rows = [], []
    for view in range(8):
        raster_rows.append({"view_index": view, "image": file_ref(root, index, f"observations/{key}/V{view}.png")})
        mask_rows.append({"view_index": view, "mask": file_ref(root, index, f"observations/{key}/V{view}.mask.bin")})
    return {
        "source_files": section_source_files(subject, root, index),
        "source_license": {
            "profile": SOURCE_LICENSE_PROFILE,
            "source_pack": subject["pack_id"],
            "license_name": license_defaults["license_name"],
            "license_ref": license_defaults["license_ref"],
            "source_url_or_locator": subject["raw_source"].get("origin", ""),
            "license_file_role": "license_file",
            "source_page_role": "source_page",
            "notes": "Exact bundle-sealed source/license/page bytes; inventory does not mint qualification.",
        },
        "source_audit": {
            "profile": SOURCE_SCENE_PROFILE,
            "subject_key": key,
            "primary_source_role": "primary_source",
            "minimum_border_margin_px": 8,
            "matrix_tolerance": 1e-6,
            "source_rest_candidate": file_ref(root, index, scene["source_rest_candidate"]["path"]),
            "source_scene_snapshot": file_ref(root, index, scene["source_scene_snapshot"]["path"]),
            "iris_observation_admission": file_ref(root, index, scene["iris_observation_admission"]["path"]),
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
        "iris_fit": {
            "training_profile": SHARED_PROFILE,
            "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
            "shared_cohort_manifest": {"path": str(cohort_manifest_path), "sha256": cohort_manifest_sha},
            "shared_training_policy": {"path": str(policy_path), "sha256": policy_sha},
        },
    }


def ensure_subject(api_root: str, slug: str, display: str, actor: str):
    return api(api_root, "POST", "/v1/subjects",
               {"slug": slug, "display_name": display, "created_by": actor})["subject_id"]


def seal_sections(api_root: str, platform_subject_id: str, sections: dict, actor: str, origin: str):
    bindings = []
    for key in sorted(sections):
        imported = put_import(api_root, json_bytes(sections[key]),
                              artifact_type="RealSaS.RunManifestSection", schema="v1",
                              source_uri=f"{origin}:manifest:{key}", actor=actor)
        bindings.append({"role": "manifest:" + key, "artifact_id": imported["artifact_id"]})
    result = api(api_root, "POST", "/v1/subject-inputs",
                 {"subject_id": platform_subject_id, "bindings": bindings, "created_by": actor})
    return result["subject_input_id"]


def nested(value: dict, dotted: str):
    current = value
    for key in dotted.split("."):
        require(isinstance(current, dict) and key in current, "COHORT_REGRESSION_WITNESS_POINTER_INVALID")
        current = current[key]
    return current


def clone_regression_manifest(repo: Path, authority_root: Path, cfg: dict):
    witness_path = (repo / str(cfg["inventory"])).resolve()
    witness = load(witness_path)
    run = str(nested(witness, str(cfg["manifest_source"])))
    path = authority_root / "runs" / run / "run_manifest.json"
    require(path.is_file(), "COHORT_REGRESSION_RUN_MANIFEST_MISSING:" + str(path))
    return load(path), str(witness.get("subject_key") or "regression")


def plan_manifest_keys(plan: dict):
    keys = []
    for stage in plan["stages"]:
        for key in stage.get("manifest_keys") or ():
            if key not in keys:
                keys.append(key)
    return keys


def preregistration_spec(descriptor: dict, policy: dict, policy_path: Path, repo: Path):
    cfg = descriptor["shared_iris"]["preregistration"]
    model_source = (repo / str(cfg["model_source_repo_path"])).resolve()
    require(model_source.is_file(), "COHORT_IRIS_MODEL_SOURCE_MISSING")
    return {
        "schema": "RealSaS.ModelFitPreregistrationSpec.v1",
        "lane": "IRIS",
        "architecture_id": str(cfg["architecture_id"]),
        "executor_kind": str(cfg["executor_kind"]),
        "random_seed": int(cfg["random_seed"]),
        "model_source": {"path": str(model_source), "sha256": sha256_file(model_source)},
        "teacher_inference_inputs_allowed": False,
        "upstream_bindings": {},
        "expected_output_contract": list(cfg["expected_output_contract"]),
        "qualification_policy": {
            "shared_cohort_required": True,
            "shared_training_policy_sha256": sha256_file(policy_path),
            "training_profile": str(descriptor["shared_iris"]["training_profile"]),
            "training_loss_can_mint_fit8_pass": False,
            "compiler_qualification_required": True,
        },
    }


def release_stage_parameters(descriptor, plan):
    """Only released model nodes receive external-worker role requirements."""
    stage_params = {"schema": "RealSaS.StageReleaseParameters.v1", "stages": {}}
    fit = descriptor["shared_iris"]
    if fit["stage_id"] in {stage["id"] for stage in plan["stages"]}:
        stage_params["stages"][fit["stage_id"]] = {
            "external_evidence_roles": list(fit["required_external_roles"]),
            "optional_external_evidence_roles": list(fit["optional_external_roles"]),
        }
    for stage_id, parameters in (descriptor.get("additional_stage_parameters") or {}).items():
        require(stage_id in {s["id"] for s in plan["stages"]}, "COHORT_EXTRA_STAGE_PARAMETER_UNKNOWN_STAGE")
        require(stage_id not in stage_params["stages"], "COHORT_EXTRA_STAGE_PARAMETER_SHADOWS_EXISTING_CONTRACT")
        require(isinstance(parameters, dict), "COHORT_EXTRA_STAGE_PARAMETERS_OBJECT_REQUIRED")
        stage_params["stages"][stage_id] = parameters
    return stage_params


def main(args):
    repo = args.repo.resolve()
    descriptor_path = args.descriptor.resolve()
    descriptor = load(descriptor_path)
    campaign_contract = load((repo / descriptor["campaign_contract"]).resolve())
    verify_descriptor(descriptor, campaign_contract)
    plan = load((repo / campaign_contract["compiler_plan"]).resolve())
    require(int(plan["stage_count"]) == int(campaign_contract["required_stage_count"]),
            "COHORT_BOOTSTRAP_PLAN_STAGE_COUNT_DRIFT")
    require(campaign_contract["required_target_stage_id"] in {stage["id"] for stage in plan["stages"]},
            "COHORT_BOOTSTRAP_RELEASED_TARGET_REQUIRED")
    actor = str(descriptor.get("actor") or ACTOR_DEFAULT)
    overlay = descriptor.get("additional_manifest_sections") or {}
    regression_overlay = descriptor.get("regression_additional_manifest_sections") or {}
    allowed_overlay_keys = set(plan_manifest_keys(plan))
    for value in (overlay, regression_overlay):
        require(isinstance(value, dict) and set(value).issubset(allowed_overlay_keys),
                "COHORT_ADDITIONAL_MANIFEST_READ_SET_UNDECLARED")
    hydration_root = (args.platform_root / "hydration" / "cohorts").resolve()

    bundle_hydrated = hydrate_row(
        external_inventory_row(descriptor["cohort_bundle"]), cas_root=hydration_root,
        fetcher=fetch_google_drive_with_rclone, rclone_remote=args.rclone_remote, allow_remote=True)
    require(bundle_hydrated["status"] in {"CAS_HIT", "HYDRATED"}, "COHORT_BUNDLE_HYDRATION_FAIL")
    bundle_manifest, receipt, bundle_root = safe_extract_verified(
        Path(bundle_hydrated["path"]), descriptor, args.authority_root / "cohort_bundles")
    index = receipt_index(receipt)
    bundle_subjects = list(bundle_manifest.get("subjects") or ())
    bundle_keys = [str(row["subject_id"]) for row in bundle_subjects]
    require(bundle_keys == list(campaign_contract["cohort_subject_keys"]), "COHORT_BOOTSTRAP_ROSTER_DRIFT")

    shared_ref = descriptor["shared_iris"]["cohort_manifest"]
    shared_hydrated = hydrate_row(
        external_inventory_row(shared_ref), cas_root=hydration_root,
        fetcher=fetch_google_drive_with_rclone, rclone_remote=args.rclone_remote, allow_remote=True)
    shared_path = Path(shared_hydrated["path"])
    require(sha256_file(shared_path) == shared_ref["sha256"], "COHORT_SHARED_IRIS_MANIFEST_FILE_DRIFT")
    shared_manifest = load(shared_path)
    policy_path = (repo / descriptor["shared_iris"]["training_policy"]["repo_path"]).resolve()
    policy = load(policy_path)
    validate_shared_inputs(bundle_manifest, bundle_root, shared_manifest, policy)

    prereg = preregistration_spec(descriptor, policy, policy_path, repo)
    prereg_dir = args.authority_root / "campaign_bootstrap" / sha256_file(descriptor_path)
    prereg_dir.mkdir(parents=True, exist_ok=True)
    prereg_path = prereg_dir / "IRIS_MODEL_FIT_PREREGISTRATION.json"
    prereg_raw = json_bytes(prereg)
    if prereg_path.exists():
        require(prereg_path.read_bytes() == prereg_raw, "COHORT_IRIS_PREREG_EXISTING_DRIFT")
    else:
        prereg_path.write_bytes(prereg_raw)
    prereg_sha = sha256_file(prereg_path)

    license_defaults = descriptor["source_license_defaults"]
    cohort_state = []
    for subject in bundle_subjects:
        key = str(subject["subject_id"])
        sections = make_cohort_manifest_sections(
            subject, bundle_root, index, shared_path, str(shared_ref["sha256"]),
            policy_path, sha256_file(policy_path), prereg_path, prereg_sha, license_defaults)
        require(not set(overlay).intersection(sections), "COHORT_ADDITIONAL_MANIFEST_SHADOWS_EXISTING_INPUT")
        sections.update(overlay)
        platform_id = ensure_subject(args.api, key, str(subject["display_name"]), actor)
        input_id = seal_sections(args.api, platform_id, sections, actor,
                                 f"urn:realsas:cohort-bootstrap:{descriptor['campaign_name']}:{key}")
        cohort_state.append({"key": key, "platform_subject_id": platform_id,
                             "subject_input_id": input_id, "sections": sections})

    regression_state = []
    witness_cfgs = descriptor.get("regression_witnesses") or {}
    allowed_keys = set(plan_manifest_keys(plan))
    for key in campaign_contract.get("regression_subject_keys") or ():
        require(key in witness_cfgs, "COHORT_REGRESSION_WITNESS_DESCRIPTOR_REQUIRED:" + key)
        cfg = witness_cfgs[key]
        manifest, evidence_key = clone_regression_manifest(repo, args.authority_root, cfg)
        sections = {name: manifest[name] for name in allowed_keys if name in manifest}
        require(not set(regression_overlay).intersection(sections), "COHORT_REGRESSION_OVERLAY_SHADOWS_EXISTING_INPUT")
        sections.update(regression_overlay)
        manifest.update(regression_overlay)
        require(sections, "COHORT_REGRESSION_MANIFEST_SECTIONS_EMPTY:" + key)
        platform_id = ensure_subject(args.api, key, str(cfg.get("display_name") or key), actor)
        input_id = seal_sections(args.api, platform_id, sections, actor,
                                 f"urn:realsas:cohort-regression:{descriptor['campaign_name']}:{key}")
        regression_state.append({"key": key, "evidence_subject_id": evidence_key,
                                 "platform_subject_id": platform_id, "subject_input_id": input_id,
                                 "template_manifest": manifest, "sections": sections})

    stage_params = release_stage_parameters(descriptor, plan)
    release_request = snapshot(
        plan, name=descriptor["campaign_name"] + "__RESEARCH_RELEASE", purpose="RESEARCH",
        created_by=actor, stage_parameters=stage_params)
    release = api(args.api, "POST", "/v1/releases", release_request)
    release_id = release["ReleaseID"]
    repair_members = descriptor.get("repair_expected_campaign_members")
    if repair_members is not None:
        actual = {(r["platform_subject_id"], role) for rows, role in ((cohort_state,"COHORT"),(regression_state,"REGRESSION")) for r in rows}
        require(actual == {(r["subject_id"],r["role"]) for r in repair_members}, "COHORT_REPAIR_MEMBER_SCOPE_DRIFT")
    campaign = api(args.api, "POST", "/v1/research/campaigns", {
        "name": descriptor["campaign_name"],
        "baseline_engine_release_id": descriptor.get("baseline_engine_release_id") or release_id,
        "candidate_engine_release_id": release_id,
        "target_stage_id": campaign_contract["required_target_stage_id"],
        "cohort_subject_ids": [row["platform_subject_id"] for row in cohort_state],
        "regression_subject_ids": [row["platform_subject_id"] for row in regression_state],
        "created_by": actor,
    })
    campaign_id = campaign["campaign_id"]

    run_bindings = []
    for row in cohort_state + regression_state:
        run_id = f"CAMPAIGN_{campaign_id}_{row['platform_subject_id']}"
        root = args.authority_root / "runs" / run_id
        root.mkdir(parents=True, exist_ok=True)
        if "template_manifest" in row:
            manifest = dict(row["template_manifest"])
            manifest["run_id"] = run_id
        else:
            manifest = {"run_id": run_id, "subject_id": row["key"],
                        "execution_class": "WITNESS", **row["sections"]}
        manifest_path = root / "run_manifest.json"
        raw = json_bytes(manifest)
        if manifest_path.exists():
            require(manifest_path.read_bytes() == raw, "COHORT_RUN_MANIFEST_EXISTING_DRIFT:" + row["key"])
        else:
            manifest_path.write_bytes(raw)
        run_bindings.append({
            "subject_id": row["platform_subject_id"],
            "subject_input_id": row["subject_input_id"],
            "run_manifest_path": str(manifest_path),
            "run_ledger_path": str(root / "ACTIVE_RUN_V2.json"),
        })

    preflight = {
        "schema": "RealSaS.PlatformCohortBootstrapPreflight.v1",
        "descriptor": str(descriptor_path),
        "campaign_name": descriptor["campaign_name"],
        "campaign_id": campaign_id,
        "release_id": release_id,
        "target_stage_id": campaign_contract["required_target_stage_id"],
        "pipeline_plan_sha256": release_request["manifest"]["pipeline_plan_sha256"],
        "cohort_subject_count": len(cohort_state),
        "regression_subject_count": len(regression_state),
        "bundle_sha256": descriptor["cohort_bundle"]["sha256"],
        "shared_cohort_manifest_sha256": shared_ref["sha256"],
        "shared_policy_sha256": sha256_file(policy_path),
        "all_observation_bindings_verified": True,
        "qualification_minted": False,
        "fit8_pass_claimed": False,
        "run_bindings": run_bindings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(json_bytes(preflight))
    if args.no_submit:
        print(json.dumps(preflight, sort_keys=True))
        return

    run = api(args.api, "POST", "/v1/research/campaigns/run", {
        "campaign_id": campaign_id,
        "members": run_bindings,
        "pipeline_plan_sha256": release_request["manifest"]["pipeline_plan_sha256"],
        "requested_by": actor,
    })
    result = {**preflight, "campaign_run": run}
    args.out.write_bytes(json_bytes(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--platform-root", type=Path, default=Path("/home/monster/realsas_platform"))
    parser.add_argument("--authority-root", type=Path, default=Path("/home/monster/realsas_authority"))
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--rclone-remote", default=os.environ.get("REALSAS_RCLONE_DRIVE_REMOTE", "realsas-drive"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-submit", action="store_true",
                        help="Seal inputs/release/campaign and write run manifests, but do not submit campaign execution")
    main(parser.parse_args())
