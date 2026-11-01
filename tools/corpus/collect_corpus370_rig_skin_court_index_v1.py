#!/usr/bin/env python3
from __future__ import annotations

"""Collect current GLTF2/FBX rig-skin court artifacts without granting authority."""

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.Corpus370RigSkinCourtIndex.v1"
KNOWN = {
    "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1": "GLTF2_RAW_PLUS_IMPORT_EQUIVALENCE",
    "RealSaS.FbxImportedRigSkinAuthorityCourt.v1": "FBX_PINNED_IMPORT_REPLAY_EQUIVALENCE",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_sha(doc: dict) -> str:
    if doc.get("schema") == "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1":
        return str(((doc.get("inputs") or {}).get("raw_source") or {}).get("sha256") or "")
    return str((doc.get("source") or {}).get("sha256") or "")


def collect(root: Path) -> dict:
    root = root.resolve()
    paths = sorted({
        *root.rglob("GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE_COURT_V1.json"),
        *root.rglob("FBX_IMPORTED_RIG_SKIN_AUTHORITY_COURT_V1.json"),
    })
    rows = []
    seen = set()
    for path in paths:
        doc = json.loads(path.read_text())
        schema = str(doc.get("schema") or "")
        if schema not in KNOWN:
            continue
        subject_id = str(doc.get("subject_id") or "")
        if not subject_id or subject_id in seen:
            raise ValueError("CORPUS370_RIG_SKIN_COURT_DUPLICATE_OR_EMPTY_SUBJECT:" + subject_id)
        digest = source_sha(doc)
        if len(digest) != 64:
            raise ValueError("CORPUS370_RIG_SKIN_COURT_SOURCE_SHA:" + subject_id)
        seen.add(subject_id)
        rows.append({
            "subject_id": subject_id,
            "source_sha256": digest,
            "lane": KNOWN[schema],
            "court": {"path": str(path.resolve()), "sha256": sha256(path), "schema": schema},
        })
    return {
        "schema": SCHEMA,
        "status": "COURT_ARTIFACT_INDEX_ONLY__NO_FAMILY_AUTHORITY",
        "subject_count": len(rows),
        "subjects": sorted(rows, key=lambda row: row["subject_id"]),
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.execution_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": result["schema"], "subject_count": result["subject_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
