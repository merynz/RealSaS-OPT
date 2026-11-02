#!/usr/bin/env python3
from __future__ import annotations

"""Pack exact Corpus370 source/provenance/observation inputs for Platform truth.

This is transport preparation only. Every included byte is hash-bound and reopened
from the final archive. Missing provenance or observation evidence fails closed.
The bundle creates no family authority, split, or training authorization.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
import zipfile

SCHEMA = "RealSaS.Corpus370PlatformStage01InputBundle.v1"
RECEIPT_SCHEMA = "RealSaS.Corpus370PlatformStage01InputBundleReceipt.v1"
OBSERVATION_SCHEMAS = {
    "RealSaS.Corpus370PreparedObservationPackets.v1",
    "RealSaS.Corpus370PreparedObservationPackets.v2",
}


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


def add_file(files: dict[str, dict], archive_path: str, path: Path, expected_sha: str | None = None) -> dict:
    path = path.resolve()
    require(path.is_file(), "CORPUS370_PLATFORM_BUNDLE_FILE_MISSING:" + str(path))
    digest = sha256(path)
    if expected_sha is not None:
        require(digest == expected_sha, "CORPUS370_PLATFORM_BUNDLE_FILE_DRIFT:" + str(path))
    row = {"path": archive_path, "sha256": digest, "size_bytes": path.stat().st_size}
    prior = files.get(archive_path)
    if prior is not None:
        require(prior["sha256"] == digest and prior["size_bytes"] == row["size_bytes"], "CORPUS370_PLATFORM_BUNDLE_PATH_COLLISION:" + archive_path)
    else:
        files[archive_path] = {**row, "source_path": str(path)}
    return row


def gltf_dependencies(source: Path) -> list[Path]:
    if source.suffix.lower() != ".gltf":
        return []
    doc = json.loads(source.read_text())
    rows = []
    for item in [*doc.get("buffers", []), *doc.get("images", [])]:
        uri = str(item.get("uri") or "")
        if not uri or uri.startswith("data:"):
            continue
        parsed = urlparse(uri)
        require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment, "CORPUS370_PLATFORM_BUNDLE_REMOTE_GLTF_DEPENDENCY")
        target = (source.parent / unquote(parsed.path)).resolve()
        require(target.is_relative_to(source.parent.resolve()) and target.is_file(), "CORPUS370_PLATFORM_BUNDLE_GLTF_DEPENDENCY_MISSING")
        rows.append(target)
    return sorted(set(rows))


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

    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1" and int(roster.get("subject_count", -1)) == 370, "CORPUS370_PLATFORM_BUNDLE_ROSTER")
    require(source_manifest.get("schema") == "RealSaS.ExpandedCorpusSourceInputBundle.v1" and int(source_manifest.get("subject_count", -1)) == 370, "CORPUS370_PLATFORM_BUNDLE_SOURCE_MANIFEST")
    require(observations.get("schema") in OBSERVATION_SCHEMAS and int(observations.get("subject_count", -1)) == 370, "CORPUS370_PLATFORM_BUNDLE_OBSERVATIONS")
    require(provenance.get("schema") == "RealSaS.Corpus370SourceProvenanceIndex.v1" and int(provenance.get("subject_count", -1)) == 370, "CORPUS370_PLATFORM_BUNDLE_PROVENANCE")
    require(all(doc.get("training_authorized") is False for doc in (roster, source_manifest, observations, provenance)), "CORPUS370_PLATFORM_BUNDLE_PREAUTHORIZED")
    require(int(observations.get("prepared_subject_count", len(observations.get("subjects") or ()))) == 370, "CORPUS370_PLATFORM_BUNDLE_OBSERVATION_PREPARED_COUNT")
    require(int(observations.get("failure_count", 0)) == 0, "CORPUS370_PLATFORM_BUNDLE_OBSERVATION_FAILURES")

    roster_rows = {str(row["subject_id"]): row for row in roster["subjects"]}
    source_rows = {str(row["subject_id"]): row for row in source_manifest["subjects"]}
    observation_rows = {str(row["subject_id"]): row for row in observations["subjects"]}
    provenance_rows = {str(row["subject_id"]): row for row in provenance["subjects"]}
    require(set(roster_rows) == set(source_rows) == set(observation_rows) == set(provenance_rows) and len(roster_rows) == 370, "CORPUS370_PLATFORM_BUNDLE_SUBJECT_SET")

    bundle_files: dict[str, dict] = {}
    subjects = []
    source_file_index = {str(row["archive_path"]): row for row in source_manifest.get("files") or []}
    for sid in sorted(roster_rows):
        roster_row = roster_rows[sid]
        source_row = source_rows[sid]
        observation = observation_rows[sid]
        provenance_row = provenance_rows[sid]
        source_rel = str(roster_row["source_relative_path"])
        require(source_rel == str(source_row["source_relative_path"]), "CORPUS370_PLATFORM_BUNDLE_SOURCE_PATH:" + sid)
        require(roster_row["source_sha256"] == source_row["source_sha256"] == observation["source_sha256"] == provenance_row["source_sha256"], "CORPUS370_PLATFORM_BUNDLE_SOURCE_IDENTITY:" + sid)
        source = (extracted_source_root / source_rel).resolve()
        require(source.is_relative_to(extracted_source_root), "CORPUS370_PLATFORM_BUNDLE_SOURCE_ESCAPE")
        raw_archive = "sources/" + source_rel
        raw = add_file(bundle_files, raw_archive, source, roster_row["source_sha256"])

        dependencies = []
        for dep in gltf_dependencies(source):
            relative_to_source = dep.relative_to(source.parent).as_posix()
            archive_path = str(Path(raw_archive).parent / relative_to_source)
            known_source_relative = str(Path(source_rel).parent / relative_to_source)
            known = source_file_index.get(known_source_relative)
            require(known is not None, "CORPUS370_PLATFORM_BUNDLE_GLTF_DEPENDENCY_NOT_SOURCE_BUNDLE:" + sid)
            dependencies.append(add_file(bundle_files, archive_path, dep, known["sha256"]))

        if source.suffix.lower() == ".fbx" and source_rel.startswith("assets/publishers/"):
            prefix = "/".join(source_rel.split("/")[:4]) + "/"
            for rel, known in sorted(source_file_index.items()):
                if not rel.startswith(prefix):
                    continue
                if Path(rel).suffix.lower() not in {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp", ".tif", ".tiff", ".exr"}:
                    continue
                dep = (extracted_source_root / rel).resolve()
                dependencies.append(add_file(bundle_files, "sources/" + rel, dep, known["sha256"]))
        dependencies = list({row["path"]: row for row in dependencies}.values())

        packet = Path(str(observation["packet_path"])).resolve()
        required_packet = [
            "SOURCE_REST_RENDER_CANDIDATE.json", "SOURCE_SCENE_SNAPSHOT.json", "camera_bundle.json",
            "IRIS_OBSERVATION_ADMISSION.json",
            *[f"V{i}.png" for i in range(8)], *[f"V{i}.mask.bin" for i in range(8)],
        ]
        packet_rows = {}
        for name in required_packet:
            packet_rows[name] = add_file(bundle_files, f"observations/{sid}/{name}", packet / name)
        admission = json.loads((packet / "IRIS_OBSERVATION_ADMISSION.json").read_text())
        require(admission.get("status") == "PASS_IRIS_BIND_OBSERVATIONS" and admission.get("source_sha256") == roster_row["source_sha256"], "CORPUS370_PLATFORM_BUNDLE_OBSERVATION_ADMISSION:" + sid)

        license_path = Path(str((provenance_row.get("license_bytes") or {}).get("path") or "")).resolve()
        page_path = Path(str((provenance_row.get("source_page_snapshot") or {}).get("path") or "")).resolve()
        license_row = add_file(bundle_files, f"provenance/{sid}/LICENSE.txt", license_path, provenance_row["license_bytes"]["sha256"])
        page_row = add_file(bundle_files, f"provenance/{sid}/SOURCE_PAGE.snapshot", page_path, provenance_row["source_page_snapshot"]["sha256"])
        legalcode_row = None
        if provenance_row.get("license_legalcode"):
            legalcode_path = Path(str(provenance_row["license_legalcode"]["path"])).resolve()
            legalcode_row = add_file(bundle_files, f"provenance/{sid}/LICENSE_LEGALCODE.snapshot", legalcode_path, provenance_row["license_legalcode"]["sha256"])

        subjects.append(
            {
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
                "source_scene_evidence": {
                    "source_rest_candidate": packet_rows["SOURCE_REST_RENDER_CANDIDATE.json"],
                    "source_scene_snapshot": packet_rows["SOURCE_SCENE_SNAPSHOT.json"],
                    "camera_bundle": packet_rows["camera_bundle.json"],
                    "iris_observation_admission": packet_rows["IRIS_OBSERVATION_ADMISSION.json"],
                },
                "training_authorized": False,
            }
        )

    manifest = {
        "schema": SCHEMA,
        "status": "EXACT_PLATFORM_TRUTH_INPUT_BUNDLE__AUTHORITY_MINTED_ONLY_BY_ATTEMPT_COURTS",
        "subject_count": len(subjects),
        "subjects": subjects,
        "parents": {
            "exact_source_roster": {"path": str(roster_path), "sha256": sha256(roster_path)},
            "source_bundle_manifest": {"path": str(source_bundle_manifest_path), "sha256": sha256(source_bundle_manifest_path)},
            "observation_packets": {"schema": observations["schema"], "path": str(observation_index_path), "sha256": sha256(observation_index_path)},
            "source_provenance": {"path": str(provenance_index_path), "sha256": sha256(provenance_index_path)},
        },
        "raw_source_mutated": False,
        "training_authorized": False,
    }
    manifest_bytes = json_bytes(manifest)
    manifest_path = "BOOTSTRAP_MANIFEST.json"
    bundle_files[manifest_path] = {"path": manifest_path, "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "size_bytes": len(manifest_bytes), "inline_bytes": manifest_bytes}
    receipt_files = [{k: row[k] for k in ("path", "sha256", "size_bytes")} for _, row in sorted(bundle_files.items())]
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
        require(set(archive.namelist()) == {row["path"] for row in receipt_files} | {"BUNDLE_RECEIPT.json"}, "CORPUS370_PLATFORM_BUNDLE_ARCHIVE_MEMBER_SET")
        for row in receipt_files:
            data = archive.read(row["path"])
            require(len(data) == row["size_bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "CORPUS370_PLATFORM_BUNDLE_ARCHIVE_RECHECK:" + row["path"])
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
    print(json.dumps({"schema": manifest["schema"], "subject_count": manifest["subject_count"], "file_count": receipt["file_count"], "archive": archive, "training_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
