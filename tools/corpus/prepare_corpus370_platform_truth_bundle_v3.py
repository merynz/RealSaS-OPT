#!/usr/bin/env python3
from __future__ import annotations

"""Pack the full Corpus370 candidate universe for family-aware Platform truth.

V3 deliberately does *not* require every candidate to own a qualified observation
packet.  Raw source bytes and provenance remain mandatory for all 370 subjects.
Observation preparation is represented as an explicit per-subject outcome:

- PASS: exact qualified packet bytes are included and may feed the observation /
  Stage47/48 lane;
- FAIL: the exact failure receipt is included and the subject remains in the
  candidate universe for observation-independent source/rig/skin/animation courts.

Each V4 observation PASS includes the *original admitted import JSON bytes* and all
8 hash-checked original rasters/masks for Stage50 source-surface replay. A failed
observation stays FAIL and retains raw source/typed failure provenance.

The bundle is transport only.  It mints no family authority, split, model roster,
readiness, or training authorization.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
import zipfile

from tools.observation.source_scene_contract_v2 import _json_and_chunks
from tools.corpus.prepare_corpus370_observation_packets_v4 import _packet_complete
from tools.corpus.audit_corpus370_historical_observation_reuse_v1 import inspect as inspect_legacy

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json"

SCHEMA = "RealSaS.Corpus370PlatformFamilyInputBundle.v3"
RECEIPT_SCHEMA = "RealSaS.Corpus370PlatformFamilyInputBundleReceipt.v3"
OBSERVATION_SCHEMAS = {
    "RealSaS.Corpus370PreparedObservationPackets.v4",
    "RealSaS.Corpus370PreparedObservationPackets.v5",
}
REQUIRED_PACKET_FILES = (
    "SOURCE_REST_RENDER_CANDIDATE.json",
    "SOURCE_SCENE_SNAPSHOT.json",
    "camera_bundle.json",
    "IRIS_OBSERVATION_ADMISSION.json",
    "imported.json",
    *tuple(f"V{i}.png" for i in range(8)),
    *tuple(f"V{i}.mask.bin" for i in range(8)),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def add_file(
    files: dict[str, dict], archive_path: str, path: Path, expected_sha: str | None = None
) -> dict:
    path = path.resolve()
    require(path.is_file(), "CORPUS370_PLATFORM_V3_FILE_MISSING:" + str(path))
    digest = sha256(path)
    if expected_sha is not None:
        require(digest == expected_sha, "CORPUS370_PLATFORM_V3_FILE_DRIFT:" + str(path))
    row = {"path": archive_path, "sha256": digest, "size_bytes": path.stat().st_size}
    prior = files.get(archive_path)
    if prior is not None:
        require(
            prior["sha256"] == digest and prior["size_bytes"] == row["size_bytes"],
            "CORPUS370_PLATFORM_V3_PATH_COLLISION:" + archive_path,
        )
    else:
        files[archive_path] = {**row, "source_path": str(path)}
    return row


def gltf_dependencies(source: Path) -> list[Path]:
    if source.suffix.lower() not in {".gltf", ".glb"}:
        return []
    doc, _ = _json_and_chunks(source)
    rows: list[Path] = []
    for item in [*doc.get("buffers", []), *doc.get("images", [])]:
        uri = str(item.get("uri") or "")
        if not uri or uri.startswith("data:"):
            continue
        parsed = urlparse(uri)
        require(
            not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
            "CORPUS370_PLATFORM_V3_REMOTE_GLTF_DEPENDENCY",
        )
        target = (source.parent / unquote(parsed.path)).resolve()
        require(
            target.is_relative_to(source.parent.resolve()) and target.is_file(),
            "CORPUS370_PLATFORM_V3_GLTF_DEPENDENCY_MISSING",
        )
        rows.append(target)
    return sorted(set(rows))


def _observation_partition(observations: dict, roster_ids: set[str]) -> tuple[dict, dict]:
    prepared = {str(row["subject_id"]): dict(row) for row in observations.get("subjects") or ()}
    failed = {str(row["subject_id"]): dict(row) for row in observations.get("failures") or ()}
    require(not (set(prepared) & set(failed)), "CORPUS370_PLATFORM_V3_OBSERVATION_OVERLAP")
    require(
        set(prepared) | set(failed) == roster_ids,
        "CORPUS370_PLATFORM_V3_OBSERVATION_PARTITION",
    )
    require(
        int(observations.get("prepared_subject_count", len(prepared))) == len(prepared),
        "CORPUS370_PLATFORM_V3_PREPARED_DECLARATION",
    )
    require(
        int(observations.get("failure_count", len(failed))) == len(failed),
        "CORPUS370_PLATFORM_V3_FAILURE_DECLARATION",
    )
    return prepared, failed


def _source_dependencies(
    *,
    source: Path,
    source_rel: str,
    source_file_index: dict[str, dict],
    files: dict[str, dict],
) -> list[dict]:
    raw_archive = "sources/" + source_rel
    dependencies: list[dict] = []
    for dep in gltf_dependencies(source):
        relative_to_source = dep.relative_to(source.parent).as_posix()
        archive_path = str(Path(raw_archive).parent / relative_to_source)
        known_source_relative = str(Path(source_rel).parent / relative_to_source)
        known = source_file_index.get(known_source_relative)
        require(known is not None, "CORPUS370_PLATFORM_V3_GLTF_DEPENDENCY_NOT_SOURCE_BUNDLE")
        dependencies.append(add_file(files, archive_path, dep, known["sha256"]))

    # FBX external textures are pack-local evidence. Include exact source-bundle
    # bytes without interpreting material semantics here.
    if source.suffix.lower() == ".fbx" and source_rel.startswith("assets/publishers/"):
        prefix = "/".join(source_rel.split("/")[:4]) + "/"
        for rel, known in sorted(source_file_index.items()):
            if not rel.startswith(prefix):
                continue
            if Path(rel).suffix.lower() not in {
                ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp", ".tif", ".tiff", ".exr"
            }:
                continue
            dep = (source.parent.parents[3] / rel).resolve() if False else None
            # The caller's extracted root is not derivable from source without
            # assumptions. The concrete FBX pack-local scan is therefore done in
            # prepare(), where the exact extracted root is available.
    return list({row["path"]: row for row in dependencies}.values())


def prepare(
    roster_path: Path,
    source_bundle_manifest_path: Path,
    extracted_source_root: Path,
    observation_index_path: Path,
    provenance_index_path: Path,
    output: Path,
) -> tuple[dict, dict, dict]:
    roster_path = roster_path.resolve()
    source_bundle_manifest_path = source_bundle_manifest_path.resolve()
    extracted_source_root = extracted_source_root.resolve()
    observation_index_path = observation_index_path.resolve()
    provenance_index_path = provenance_index_path.resolve()

    roster = json.loads(roster_path.read_text())
    source_manifest = json.loads(source_bundle_manifest_path.read_text())
    observations = json.loads(observation_index_path.read_text())
    provenance = json.loads(provenance_index_path.read_text())

    # Current V4 is the only observation producer permitted to mint new
    # source-surface input bundles. Old V2/V3 results remain immutable evidence.
    policy = json.loads(POLICY_PATH.read_text())
    require(policy.get("schema") == "RealSaS.IrisObservationPreparationPolicy.v1"
            and policy.get("training_authorized") is False
            and policy.get("view_count") == 8,
            "CORPUS370_PLATFORM_V3_SOURCE_POLICY")
    # V5 is an explicitly typed composite: 294 old strict PASS rows were
    # independently rehashed in their original place; only old FAIL rows were
    # eligible for V4 repair. Do not accept an unbound synthetic composite.
    if observations.get("schema") == "RealSaS.Corpus370PreparedObservationPackets.v5":
        parent = observations.get("parent") or {}
        require(parent.get("schema") == "RealSaS.Corpus370PreparedObservationPackets.v3"
                and parent.get("prepared_subject_count") == 294
                and parent.get("failure_count") == 76
                and sha256(Path(str(parent.get("path") or ""))) == parent.get("sha256"),
                "CORPUS370_PLATFORM_V3_V5_PARENT_BINDING")
        require((observations.get("authority") or {}).get("old_pass_status_upgraded_to_v4") is False,
                "CORPUS370_PLATFORM_V3_V5_FALSE_RECLASSIFICATION")
        require(observations.get("historical_v3_pass_preserved_count") == 294,
                "CORPUS370_PLATFORM_V3_V5_294_PARENT_SUCCESS_REQUIRED")
    worker_binding = observations.get("binding") or {}
    require((worker_binding.get("policy") or {}).get("sha256") == sha256(POLICY_PATH),
            "CORPUS370_PLATFORM_V3_WORKER_POLICY_DRIFT")
    blender_sha = worker_binding.get("blender_binary_sha256")
    require(isinstance(blender_sha, str) and len(blender_sha) == 64,
            "CORPUS370_PLATFORM_V3_BLENDER_BINDING")


    require(
        roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1"
        and int(roster.get("subject_count", -1)) == 370,
        "CORPUS370_PLATFORM_V3_ROSTER",
    )
    require(
        source_manifest.get("schema") == "RealSaS.ExpandedCorpusSourceInputBundle.v1"
        and int(source_manifest.get("subject_count", -1)) == 370,
        "CORPUS370_PLATFORM_V3_SOURCE_MANIFEST",
    )
    require(
        observations.get("schema") in OBSERVATION_SCHEMAS
        and int(observations.get("subject_count", -1)) == 370,
        "CORPUS370_PLATFORM_V3_OBSERVATIONS",
    )
    require(
        provenance.get("schema") == "RealSaS.Corpus370SourceProvenanceIndex.v1"
        and int(provenance.get("subject_count", -1)) == 370,
        "CORPUS370_PLATFORM_V3_PROVENANCE",
    )
    require(
        all(doc.get("training_authorized") is False for doc in (roster, source_manifest, observations, provenance)),
        "CORPUS370_PLATFORM_V3_PREAUTHORIZED",
    )

    roster_rows = {str(row["subject_id"]): row for row in roster["subjects"]}
    source_rows = {str(row["subject_id"]): row for row in source_manifest["subjects"]}
    provenance_rows = {str(row["subject_id"]): row for row in provenance["subjects"]}
    require(
        set(roster_rows) == set(source_rows) == set(provenance_rows) and len(roster_rows) == 370,
        "CORPUS370_PLATFORM_V3_SUBJECT_SET",
    )
    observation_rows, observation_failures = _observation_partition(observations, set(roster_rows))

    bundle_files: dict[str, dict] = {}
    subjects: list[dict] = []
    source_file_index = {str(row["archive_path"]): row for row in source_manifest.get("files") or []}

    for sid in sorted(roster_rows):
        roster_row = roster_rows[sid]
        source_row = source_rows[sid]
        provenance_row = provenance_rows[sid]
        source_rel = str(roster_row["source_relative_path"])
        require(source_rel == str(source_row["source_relative_path"]), "CORPUS370_PLATFORM_V3_SOURCE_PATH:" + sid)
        require(
            roster_row["source_sha256"] == source_row["source_sha256"] == provenance_row["source_sha256"],
            "CORPUS370_PLATFORM_V3_SOURCE_IDENTITY:" + sid,
        )
        source = (extracted_source_root / source_rel).resolve()
        require(source.is_relative_to(extracted_source_root), "CORPUS370_PLATFORM_V3_SOURCE_ESCAPE")
        raw_archive = "sources/" + source_rel
        raw = add_file(bundle_files, raw_archive, source, roster_row["source_sha256"])

        dependencies = []
        for dep in gltf_dependencies(source):
            relative_to_source = dep.relative_to(source.parent).as_posix()
            archive_path = str(Path(raw_archive).parent / relative_to_source)
            known_source_relative = str(Path(source_rel).parent / relative_to_source)
            known = source_file_index.get(known_source_relative)
            require(
                known is not None,
                "CORPUS370_PLATFORM_V3_GLTF_DEPENDENCY_NOT_SOURCE_BUNDLE:" + sid,
            )
            dependencies.append(add_file(bundle_files, archive_path, dep, known["sha256"]))

        if source.suffix.lower() == ".fbx" and source_rel.startswith("assets/publishers/"):
            prefix = "/".join(source_rel.split("/")[:4]) + "/"
            for rel, known in sorted(source_file_index.items()):
                if not rel.startswith(prefix):
                    continue
                if Path(rel).suffix.lower() not in {
                    ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp", ".tif", ".tiff", ".exr"
                }:
                    continue
                dep = (extracted_source_root / rel).resolve()
                dependencies.append(add_file(bundle_files, "sources/" + rel, dep, known["sha256"]))
        dependencies = list({row["path"]: row for row in dependencies}.values())

        license_path = Path(str((provenance_row.get("license_bytes") or {}).get("path") or "")).resolve()
        page_path = Path(str((provenance_row.get("source_page_snapshot") or {}).get("path") or "")).resolve()
        license_row = add_file(
            bundle_files,
            f"provenance/{sid}/LICENSE.txt",
            license_path,
            provenance_row["license_bytes"]["sha256"],
        )
        page_row = add_file(
            bundle_files,
            f"provenance/{sid}/SOURCE_PAGE.snapshot",
            page_path,
            provenance_row["source_page_snapshot"]["sha256"],
        )
        legalcode_row = None
        if provenance_row.get("license_legalcode"):
            legalcode_path = Path(str(provenance_row["license_legalcode"]["path"])).resolve()
            legalcode_row = add_file(
                bundle_files,
                f"provenance/{sid}/LICENSE_LEGALCODE.snapshot",
                legalcode_path,
                provenance_row["license_legalcode"]["sha256"],
            )

        subject = {
            "subject_id": sid,
            "display_name": provenance_row.get("display_name") or sid,
            "pack_id": provenance_row["source_pack"],
            "raw_source": {**raw, "origin": provenance_row.get("source_url_or_locator", "")},
            "source_dependencies": dependencies,
            "license_bytes": license_row,
            "source_page_snapshot": page_row,
            "license_legalcode": legalcode_row,
            "license": {
                "name": provenance_row["license_name"],
                "ref": provenance_row["license_ref"],
                "source_url_or_locator": provenance_row.get("source_url_or_locator", ""),
            },
            "training_authorized": False,
        }

        if sid in observation_rows:
            observation = observation_rows[sid]
            require(
                observation["source_sha256"] == roster_row["source_sha256"],
                "CORPUS370_PLATFORM_V3_OBSERVATION_SOURCE_IDENTITY:" + sid,
            )
            packet = Path(str(observation["packet_path"])).resolve()
            producer = observation.get("evidence_producer")
            composite = observations["schema"] == "RealSaS.Corpus370PreparedObservationPackets.v5"
            if composite:
                require(producer in (
                    "HISTORICAL_V3_PASS_BYTES_RECHECKED",
                    "NEW_V4_REPAIR_FROM_ORIGINAL_FAIL",
                ), "CORPUS370_PLATFORM_V3_UNTYPED_V5_SOURCE:" + sid)
            else:
                require(producer is None, "CORPUS370_PLATFORM_V3_V4_CANNOT_REDECLARE_PRODUCER:" + sid)

            legacy = producer == "HISTORICAL_V3_PASS_BYTES_RECHECKED"
            if legacy:
                court = inspect_legacy(observation, policy)
                require(court["status"] == "EXACT_HISTORICAL_V3_PASS_BYTES_REUSABLE"
                        and court["legacy_admission_sha256"] == observation.get("legacy_admission_sha256"),
                        "CORPUS370_PLATFORM_V3_HISTORICAL_PASS_BYTES_NOT_PROVEN:" + sid)
                original = Path(str((observation.get("admitted_import_geometry") or {}).get("path") or "")).resolve()
                require(original == (packet.parent.parent / "imported" / sid / "imported.json").resolve()
                        and court["original_import_sha256"]
                            == (observation.get("admitted_import_geometry") or {}).get("sha256"),
                        "CORPUS370_PLATFORM_V3_HISTORICAL_IMPORT_IDENTITY:" + sid)
            else:
                require(_packet_complete(packet, sid, roster_row["source_sha256"],
                                        policy, blender_sha),
                        "CORPUS370_PLATFORM_V3_EIGHT_VIEW_PACKET_OR_IMPORT_NOT_PROVEN:" + sid)
                original = packet / "imported.json"
                if composite:
                    prior = observation.get("parent_failure_receipt") or {}
                    prior_path = Path(str(prior.get("path") or "")).resolve()
                    require(prior_path.is_file() and sha256(prior_path) == prior.get("sha256"),
                            "CORPUS370_PLATFORM_V3_V4_REPAIR_ORIGINAL_FAILURE_DRIFT:" + sid)

            packet_rows = {
                name: add_file(bundle_files, f"observations/{sid}/{name}", packet / name)
                for name in REQUIRED_PACKET_FILES if name != "imported.json"
            }
            packet_rows["imported.json"] = add_file(
                bundle_files, f"observations/{sid}/imported.json",
                original, observation.get("admitted_import_geometry", {}).get("sha256")
                          if legacy else None,
            )
            admission = json.loads((packet / "IRIS_OBSERVATION_ADMISSION.json").read_text())
            require(
                admission.get("status") == "PASS_IRIS_BIND_OBSERVATIONS"
                and admission.get("source_sha256") == roster_row["source_sha256"]
                and admission.get("imported_geometry_sha256")
                    == packet_rows["imported.json"]["sha256"],
                "CORPUS370_PLATFORM_V3_OBSERVATION_ADMISSION:" + sid,
            )
            if not legacy:
                bindings = observation.get("bindings") or {}
                for binding_role, filename in (
                    ("admitted_import_geometry", "imported.json"),
                    ("observation_admission", "IRIS_OBSERVATION_ADMISSION.json"),
                    ("source_rest_candidate", "SOURCE_REST_RENDER_CANDIDATE.json"),
                    ("camera_bundle", "camera_bundle.json"),
                    ("source_scene_snapshot", "SOURCE_SCENE_SNAPSHOT.json"),
                ):
                    expected = bindings.get(binding_role) or {}
                    require(expected.get("sha256") == packet_rows[filename]["sha256"]
                            and Path(str(expected.get("path") or "")).resolve()
                                == (packet / filename).resolve(),
                            "CORPUS370_PLATFORM_V3_WORKER_PACKET_REFERENCE_DRIFT:" + sid + ":" + binding_role)
            subject["observation_preparation"] = {
                "status": "PASS",
                "producer_scope": producer or "CURRENT_V4_ONLY",
                "historical_v3_pass_reattested": legacy,
                "new_v4_admission_claimed": not legacy,
                "packet_schema": observation.get("packet_schema"),
                "packet_identity": observation.get("packet_identity"),
                "source_rest_candidate": packet_rows["SOURCE_REST_RENDER_CANDIDATE.json"],
                "source_scene_snapshot": packet_rows["SOURCE_SCENE_SNAPSHOT.json"],
                "camera_bundle": packet_rows["camera_bundle.json"],
                "iris_observation_admission": packet_rows["IRIS_OBSERVATION_ADMISSION.json"],
                "admitted_import_geometry": packet_rows["imported.json"],
            }
        else:
            failure = observation_failures[sid]
            require(
                failure["source_sha256"] == roster_row["source_sha256"],
                "CORPUS370_PLATFORM_V3_FAILURE_SOURCE_IDENTITY:" + sid,
            )
            failure_ref = dict(failure.get("receipt") or {})
            failure_path = Path(str(failure_ref.get("path") or "")).resolve()
            failure_row = add_file(
                bundle_files,
                f"observations/{sid}/PREPARATION_FAILURE.json",
                failure_path,
                str(failure_ref.get("sha256") or ""),
            )
            subject["observation_preparation"] = {
                "status": "FAIL",
                "error_type": failure.get("error_type"),
                "reason": failure.get("reason"),
                "source_format": failure.get("source_format"),
                "source_id": failure.get("source_id"),
                "failure_receipt": failure_row,
            }
        subjects.append(subject)

    qualified = sum(row["observation_preparation"]["status"] == "PASS" for row in subjects)
    failed = len(subjects) - qualified
    require(qualified == len(observation_rows) and failed == len(observation_failures), "CORPUS370_PLATFORM_V3_ROLLUP")

    manifest = {
        "schema": SCHEMA,
        "status": "FULL_CANDIDATE_UNIVERSE_BOUND__FAMILY_ADMISSION_PENDING",
        "candidate_subject_count": 370,
        "subject_count": 370,
        "observation_qualified_count": qualified,
        "observation_failed_count": failed,
        "subjects": subjects,
        "parents": {
            "exact_source_roster": {"path": str(roster_path), "sha256": sha256(roster_path)},
            "source_bundle_manifest": {
                "path": str(source_bundle_manifest_path),
                "sha256": sha256(source_bundle_manifest_path),
            },
            "observation_preparation": {
                "schema": observations["schema"],
                "path": str(observation_index_path),
                "sha256": sha256(observation_index_path),
            },
            "source_provenance": {"path": str(provenance_index_path), "sha256": sha256(provenance_index_path)},
        },
        "rules": {
            "all_370_raw_sources_and_provenance_bound": True,
            "observation_failure_does_not_remove_subject_from_candidate_universe": True,
            "observation_failed_subject_forbidden_from_observation_dependent_lane": True,
            "exact_admitted_import_bytes_required_for_each_observation_pass": True,
            "historical_v3_pass_has_its_own_identity_never_v4_promoted": True,
            "unqualified_observation_cannot_mint_source_surface": True,
            "family_authority_minted": False,
            "split_minted": False,
        },
        "raw_source_mutated": False,
        "training_authorized": False,
    }

    manifest_bytes = json_bytes(manifest)
    manifest_path = "BOOTSTRAP_MANIFEST.json"
    bundle_files[manifest_path] = {
        "path": manifest_path,
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size_bytes": len(manifest_bytes),
        "inline_bytes": manifest_bytes,
    }
    receipt_files = [
        {k: row[k] for k in ("path", "sha256", "size_bytes")}
        for _, row in sorted(bundle_files.items())
    ]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "file_count": len(receipt_files),
        "files": receipt_files,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "training_authorized": False,
    }
    receipt_bytes = json_bytes(receipt)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for archive_path, row in sorted(bundle_files.items()):
            if "inline_bytes" in row:
                archive.writestr(archive_path, row["inline_bytes"])
            else:
                archive.write(row["source_path"], archive_path)
        archive.writestr("BUNDLE_RECEIPT.json", receipt_bytes)
    with zipfile.ZipFile(temp) as archive:
        require(
            set(archive.namelist()) == {row["path"] for row in receipt_files} | {"BUNDLE_RECEIPT.json"},
            "CORPUS370_PLATFORM_V3_ARCHIVE_MEMBER_SET",
        )
        for row in receipt_files:
            data = archive.read(row["path"])
            require(
                len(data) == row["size_bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                "CORPUS370_PLATFORM_V3_ARCHIVE_RECHECK:" + row["path"],
            )
    temp.replace(output)
    archive_ref = {"path": str(output), "sha256": sha256(output), "size_bytes": output.stat().st_size}
    return manifest, receipt, archive_ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--source-bundle-manifest", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--provenance-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, receipt, archive = prepare(
        args.exact_source_roster,
        args.source_bundle_manifest,
        args.extracted_source_root,
        args.observation_index,
        args.provenance_index,
        args.output,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "candidate_subject_count": manifest["candidate_subject_count"],
                "observation_qualified_count": manifest["observation_qualified_count"],
                "observation_failed_count": manifest["observation_failed_count"],
                "file_count": receipt["file_count"],
                "archive": archive,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
