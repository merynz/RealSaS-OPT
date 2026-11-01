#!/usr/bin/env python3
"""Stage exact canonical source bytes and resources for the Run-All worker.

Paths are portable locators. Subject IDs and source byte hashes are preserved;
this package is neither source/truth admission nor a training readiness seal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1 << 20): h.update(block)
    return h.hexdigest()


def create(candidates_path, base_root, publisher_root, output):
    candidates_path, base_root, publisher_root = (Path(p).resolve() for p in (candidates_path, base_root, publisher_root))
    data = json.loads(candidates_path.read_text())
    files, subjects = {}, []
    def locate(path):
        path = Path(path).resolve()
        if path.is_relative_to(base_root):
            return "assets/base/" + path.relative_to(base_root).as_posix()
        if path.is_relative_to(publisher_root):
            return "assets/publishers/" + path.relative_to(publisher_root).as_posix()
        raise ValueError("SOURCE_BUNDLE_INPUT_OUTSIDE_DECLARED_ROOTS")
    def include(path):
        path = Path(path).resolve()
        archive_path = locate(path)
        if not path.is_file():
            raise ValueError("SOURCE_BUNDLE_RESOURCE_MISSING:" + str(path))
        record = {"archive_path": archive_path, "sha256": sha(path), "size_bytes": path.stat().st_size}
        if archive_path in files and files[archive_path][1] != record:
            raise ValueError("SOURCE_BUNDLE_PATH_COLLISION")
        files[archive_path] = (path, record)
        return record
    for row in data["subjects"]:
        source = Path(row["source"]["path"]).resolve()
        record = include(source)
        if record["sha256"] != row["source"]["sha256"]:
            raise ValueError("SOURCE_BUNDLE_CANONICAL_BYTES_DRIFT")
        if source.suffix.lower() == ".gltf":
            document = json.loads(source.read_text())
            for resource in [*document.get("buffers", []), *document.get("images", [])]:
                uri = resource.get("uri", "")
                if not uri or uri.startswith("data:"): continue
                parsed = urlparse(uri)
                if parsed.scheme or parsed.netloc:
                    raise ValueError("SOURCE_BUNDLE_REMOTE_RESOURCE_NOT_PINNED")
                include(source.parent / unquote(parsed.path))
        subjects.append({"subject_id": row["subject_id"], "origin": row["origin"], "source_id": row["source_id"],
                         "source_relative_path": record["archive_path"], "source_sha256": record["sha256"],
                         "source_format": source.suffix.lower(), "training_authorized": False})
    # Native FBX texture references can point into sibling folders. Preserve all
    # publisher raster resources with their original relative paths; no repair or
    # inference of missing assets is performed.
    for sid in sorted({r["source_id"] for r in subjects}):
        raw = publisher_root / sid / "raw"
        if raw.is_dir():
            for path in sorted(raw.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp", ".tif", ".tiff", ".exr"}:
                    include(path)
    if len({r["subject_id"] for r in subjects}) != len(subjects):
        raise ValueError("SOURCE_BUNDLE_DUPLICATE_SUBJECT_ID")
    manifest = {"schema": "RealSaS.ExpandedCorpusSourceInputBundle.v1", "status": "EXACT_RAW_INPUT_CACHE__TRUTH_AND_ADMISSION_SEPARATE",
                "candidate_manifest": {"path": str(candidates_path), "sha256": sha(candidates_path)},
                "subject_count": len(subjects), "subjects": subjects,
                "files": [files[k][1] for k in sorted(files)], "producer_sha256": sha(__file__),
                "native_resource_completeness_still_requires_import_court": True,
                "raw_sources_mutated": False, "training_authorized": False}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".partial.zip")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("SOURCE_BUNDLE_MANIFEST_V1.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        for key in sorted(files): z.write(files[key][0], key)
    # Every canonical subject entry is reopened from the actual archive.
    with zipfile.ZipFile(temp) as z:
        for row in subjects:
            h = hashlib.sha256()
            with z.open(row["source_relative_path"]) as f:
                while block := f.read(1 << 20): h.update(block)
            if h.hexdigest() != row["source_sha256"]:
                raise ValueError("SOURCE_BUNDLE_ARCHIVE_RECHECK_DRIFT")
    temp.replace(output)
    return manifest, {"path": str(output), "sha256": sha(output), "size_bytes": output.stat().st_size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "base-root", "publisher-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    manifest, archive = create(a.candidates, a.base_root, a.publisher_root, a.output)
    receipt = {"schema": "RealSaS.SourceInputBundleReceipt.v1", "subject_count": manifest["subject_count"],
               "file_count": len(manifest["files"]), "archive": archive, "training_authorized": False}
    a.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
