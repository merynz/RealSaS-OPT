#!/usr/bin/env python3
"""Prepare portable Stage01-08 manifest fragments from the exact Fit8 input bundle.

The tool does not register platform state or mint qualification. It verifies the
canonical roster, sealed bundle locator/receipt, rehashes every referenced byte
and emits one subject-neutral manifest fragment plus an external-artifact
inventory per member.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT_ENV = "${REALSAS_FIT8_INPUT_ROOT}"
RAW_ROLE = "RAW_SOURCE"
LICENSE_ROLE = "LICENSE_BYTES"
PAGE_ROLE = "SOURCE_PAGE_SNAPSHOT"
CORPUS_ROLE = "CORPUS_SEAL"
CANDIDATE_ROLE = "SOURCE_REST_CANDIDATE"
SNAPSHOT_ROLE = "SOURCE_SCENE_SNAPSHOT"
PROOF_ROLE = "IRIS_OBSERVATION_ADMISSION"
CAMERA_ROLE = "CAMERA_BUNDLE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def ref(root: Path, rel: str, receipt: dict[str, dict]) -> dict:
    rel = str(rel).replace("\\", "/")
    require(rel in receipt, f"BUNDLE_RECEIPT_PATH_MISSING:{rel}")
    path = (root / rel).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"BUNDLE_FILE_MISSING:{rel}")
    row = receipt[rel]
    require(path.stat().st_size == int(row["size_bytes"]), f"BUNDLE_SIZE_DRIFT:{rel}")
    digest = sha256(path)
    require(digest == row["sha256"], f"BUNDLE_SHA_DRIFT:{rel}")
    return {
        "path": f"{ROOT_ENV}/{rel}",
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "relative_path": rel,
    }


def source_row(role: str, value: dict) -> dict:
    return {
        "role": role,
        "path": value["path"],
        "expected_sha256": value["sha256"],
        "expected_size_bytes": value["size_bytes"],
    }


def validate_selection(bundle_subjects: list[dict], selection: dict) -> None:
    require(selection.get("schema") == "RealSaS.FIT8DiversitySelection.v1", "FIT8_SELECTION_SCHEMA")
    selected = list(selection.get("selected") or ())
    require(len(selected) == len(bundle_subjects) == 8, "FIT8_SELECTION_CARDINALITY")
    by_key = {str(row.get("subject_key")): row for row in selected}
    require(len(by_key) == 8 and "" not in by_key, "FIT8_SELECTION_SUBJECT_IDENTITY")
    for subject in bundle_subjects:
        subject_id = str(subject.get("subject_id") or "")
        require(subject_id in by_key, f"FIT8_SELECTION_SUBJECT_MISSING:{subject_id}")
        selected_row = by_key[subject_id]
        source = dict(selected_row.get("source") or {})
        require(subject.get("pack_id") == selected_row.get("pack_id"),
                f"FIT8_SELECTION_PACK_DRIFT:{subject_id}")
        raw = dict(subject.get("raw_source") or {})
        require(raw.get("path") == source.get("corpus_relative_path"),
                f"FIT8_SELECTION_SOURCE_PATH_DRIFT:{subject_id}")
        require(raw.get("sha256") == source.get("sha256"),
                f"FIT8_SELECTION_SOURCE_SHA_DRIFT:{subject_id}")
        require(int(raw.get("size_bytes", -1)) == int(source.get("byte_size", -2)),
                f"FIT8_SELECTION_SOURCE_SIZE_DRIFT:{subject_id}")
        page = dict(subject.get("source_page_snapshot") or {})
        require(page.get("sha256") == source.get("source_page_sha256"),
                f"FIT8_SELECTION_SOURCE_PAGE_DRIFT:{subject_id}")
        require(source.get("license") == "CC0-1.0", f"FIT8_SELECTION_LICENSE_DRIFT:{subject_id}")


def build_subject(root: Path, subject: dict, receipt: dict[str, dict], corpus_ref: dict) -> tuple[dict, dict]:
    subject_id = str(subject["subject_id"])
    raw = ref(root, subject["raw_source"]["path"], receipt)
    license_bytes = ref(root, subject["license_bytes"]["path"], receipt)
    page = ref(root, subject["source_page_snapshot"]["path"], receipt)
    evidence = subject["source_scene_evidence"]
    candidate = ref(root, evidence["source_rest_candidate"]["path"], receipt)
    snapshot = ref(root, evidence["source_scene_snapshot"]["path"], receipt)
    proof = ref(root, evidence["iris_observation_admission"]["path"], receipt)
    camera = ref(root, evidence["camera_bundle"]["path"], receipt)

    obs_root = f"observations/{subject_id}"
    rasters, masks = [], []
    source_files = [
        source_row(RAW_ROLE, raw),
        source_row(LICENSE_ROLE, license_bytes),
        source_row(PAGE_ROLE, page),
        source_row(CORPUS_ROLE, corpus_ref),
        source_row(CANDIDATE_ROLE, candidate),
        source_row(SNAPSHOT_ROLE, snapshot),
        source_row(PROOF_ROLE, proof),
        source_row(CAMERA_ROLE, camera),
    ]
    inventory = [
        {"role": row["role"], "relative_path": value["relative_path"],
         "sha256": value["sha256"], "size_bytes": value["size_bytes"]}
        for row, value in zip(source_files,
                              (raw, license_bytes, page, corpus_ref, candidate, snapshot, proof, camera))
    ]
    for view in range(8):
        raster = ref(root, f"{obs_root}/V{view}.png", receipt)
        mask = ref(root, f"{obs_root}/V{view}.mask.bin", receipt)
        raster_role = f"SOURCE_RASTER_V{view}"
        mask_role = f"SOURCE_MASK_V{view}"
        source_files += [source_row(raster_role, raster), source_row(mask_role, mask)]
        inventory += [
            {"role": raster_role, "relative_path": raster["relative_path"],
             "sha256": raster["sha256"], "size_bytes": raster["size_bytes"]},
            {"role": mask_role, "relative_path": mask["relative_path"],
             "sha256": mask["sha256"], "size_bytes": mask["size_bytes"]},
        ]
        rasters.append({"view_index": view, "image": {"path": raster["path"], "sha256": raster["sha256"]}})
        masks.append({"view_index": view, "mask": {"path": mask["path"], "sha256": mask["sha256"]}})

    camera_payload = load_json(root / camera["relative_path"])
    require(camera_payload.get("schema") == "RealSaS.CameraProjectionBundle.v1", "CAMERA_BUNDLE_SCHEMA")
    require(camera_payload.get("per_view_recenter") is False, "CAMERA_PER_VIEW_RECENTER_FORBIDDEN")
    require(camera_payload.get("per_view_pose_change") is False, "CAMERA_PER_VIEW_POSE_CHANGE_FORBIDDEN")
    require(len(camera_payload.get("cameras") or ()) == 8, "CAMERA_COUNT")
    center = camera_payload.get("common_center_xyz")
    half = camera_payload.get("common_half_extent")
    require(isinstance(center, list) and len(center) == 3, "CAMERA_COMMON_CENTER")
    require(isinstance(half, (int, float)) and float(half) > 0, "CAMERA_COMMON_HALF_EXTENT")

    fragment = {
        "schema": "RealSaS.Fit8Stage01To08ManifestFragment.v1",
        "subject_id": subject_id,
        "source_files": source_files,
        "source_license": {
            "profile": "EXACT_SOURCE_PROVENANCE_V2",
            "source_pack": subject["pack_id"],
            "license_name": "CC0-1.0",
            "license_ref": "https://creativecommons.org/publicdomain/zero/1.0/",
            "source_url_or_locator": f"sealed-source-page:{page['sha256']}",
            "primary_source_sha256": raw["sha256"],
            "expected_source_page_sha256": page["sha256"],
            "license_bytes": {"path": license_bytes["path"], "sha256": license_bytes["sha256"]},
            "source_page_snapshot": {"path": page["path"], "sha256": page["sha256"]},
            "corpus_seal": {"path": corpus_ref["path"], "sha256": corpus_ref["sha256"]},
            "notes": "Exact bundle bytes; transport and provenance do not mint qualification.",
        },
        "source_audit": {
            "profile": "SOURCE_SCENE_BIND_V2",
            "subject_key": subject_id,
            "primary_source_role": RAW_ROLE,
            "minimum_border_margin_px": 8,
            "matrix_tolerance": 1e-6,
            "source_rest_candidate": {"path": candidate["path"], "sha256": candidate["sha256"]},
            "source_scene_snapshot": {"path": snapshot["path"], "sha256": snapshot["sha256"]},
            "iris_observation_admission": {"path": proof["path"], "sha256": proof["sha256"]},
            "iris_observation_admission_required": True,
        },
        "admission": {"mode": "CONTROLLED_FULL_SUBJECT_8VIEW_V2_SOURCE_SCENE"},
        "observation": {
            "camera_bundle": {"path": camera["path"], "sha256": camera["sha256"]},
            "source_rasters": rasters,
            "source_foreground_masks": masks,
        },
        "normalization": {
            "center_xyz": [float(x) for x in center],
            "half_extent": float(half),
            "coordinate_frame": "REALSAS_OBJECT_FRAME",
        },
        "iris_fit": {"training_profile": "SHARED_IRIS_IMAGE_DERIVED_FSTAR_V1"},
        "authority": {
            "scope": "STAGE01_TO_08_INPUT_FRAGMENT_ONLY",
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
            "later_stage_manifest_sections_included": False,
        },
    }
    artifact_inventory = {
        "schema": "RealSaS.Fit8ExternalArtifactInventory.v1",
        "subject_id": subject_id,
        "root_env": "REALSAS_FIT8_INPUT_ROOT",
        "artifacts": inventory,
        "authority": "IMPORT_AND_SUBJECT_INPUT_PREPARATION_ONLY__NO_QUALIFICATION",
    }
    return fragment, artifact_inventory


def prepare(bundle_root: Path, locator_path: Path, selection_path: Path, out_root: Path) -> dict:
    bundle_root = bundle_root.expanduser().resolve()
    locator = load_json(locator_path)
    require(locator.get("schema") == "RealSaS.Fit8PlatformStage01InputBundleLocator.v1", "LOCATOR_SCHEMA")
    selection_path = selection_path.expanduser().resolve()
    require(selection_path.is_file(), "FIT8_SELECTION_FILE_MISSING")
    require(sha256(selection_path) == locator["selection"]["sha256"], "FIT8_SELECTION_SHA_DRIFT")
    selection = load_json(selection_path)
    manifest_path = bundle_root / "BOOTSTRAP_MANIFEST.json"
    receipt_path = bundle_root / "BUNDLE_RECEIPT.json"
    require(sha256(manifest_path) == locator["bundle"]["manifest_sha256"], "BOOTSTRAP_MANIFEST_SHA")
    require(sha256(receipt_path) == locator["bundle"]["receipt_sha256"], "BUNDLE_RECEIPT_SHA")
    manifest = load_json(manifest_path)
    receipt_payload = load_json(receipt_path)
    require(manifest.get("schema") == "RealSaS.Fit8PlatformStage01InputBundle.v1", "BOOTSTRAP_MANIFEST_SCHEMA")
    require(receipt_payload.get("schema") == "RealSaS.Fit8PlatformStage01InputBundleReceipt.v1", "BUNDLE_RECEIPT_SCHEMA")
    receipt = {row["path"]: row for row in receipt_payload.get("files") or ()}
    require(len(receipt) == int(receipt_payload["file_count"]), "BUNDLE_RECEIPT_CARDINALITY")
    subjects = list(manifest.get("subjects") or ())
    require(len(subjects) == int(manifest.get("subject_count", -1)) == int(locator.get("subject_count", -2)) == 8,
            "FIT8_SUBJECT_COUNT")
    require(len({row["subject_id"] for row in subjects}) == len(subjects), "FIT8_SUBJECT_DUPLICATE")
    validate_selection(subjects, selection)
    corpus_ref = ref(bundle_root, "manifest/CORPUS_SEAL.json", receipt)

    out_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for subject in subjects:
        subject_id = subject["subject_id"]
        fragment, inventory = build_subject(bundle_root, subject, receipt, corpus_ref)
        subject_root = out_root / subject_id
        subject_root.mkdir(parents=True, exist_ok=True)
        fragment_path = subject_root / "stage01_08_manifest_fragment.json"
        inventory_path = subject_root / "external_artifact_inventory.json"
        fragment_path.write_text(json.dumps(fragment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        index_rows.append({
            "subject_id": subject_id,
            "manifest_fragment": {"path": str(fragment_path), "sha256": sha256(fragment_path)},
            "artifact_inventory": {"path": str(inventory_path), "sha256": sha256(inventory_path)},
        })
    output = {
        "schema": "RealSaS.Fit8Stage01To08BootstrapIndex.v1",
        "selection_sha256": locator["selection"]["sha256"],
        "bundle_sha256": locator["drive"]["sha256"],
        "bundle_drive_file_id": locator["drive"]["file_id"],
        "subject_count": len(index_rows),
        "subjects": index_rows,
        "root_env_required": "REALSAS_FIT8_INPUT_ROOT",
        "later_stage_manifest_sections_included": False,
        "fit8_pass_claimed": False,
        "product_pass_claimed": False,
    }
    index_path = out_root / "FIT8_STAGE01_08_BOOTSTRAP_INDEX.json"
    index_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**output, "index_path": str(index_path), "index_sha256": sha256(index_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True,
                        help="Extracted FIT8_PLATFORM_STAGE01_INPUT_BUNDLE_20261013.zip root")
    parser.add_argument("--locator", type=Path,
                        default=Path("canonical/FIT8_PLATFORM_STAGE01_INPUT_BUNDLE_20261013.json"))
    parser.add_argument("--selection", type=Path,
                        default=Path("canonical/FIT8_DIVERSITY_SELECTION_V1.json"))
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.bundle_root, args.locator, args.selection, args.out_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
