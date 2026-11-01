#!/usr/bin/env python3
"""Reconcile historical base rows and publisher actor/export identities.

The output is a candidate denominator. It never authorizes training or infers
closedness, modular semantics, rig truth or skin truth from a mesh/file count.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.observation.source_scene_contract_v2 import _json_and_chunks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


EXPORT_LANES = {"fbx", "gltf", "glb", "obj", "blend", "blender", "models-fbx", "models-gltf",
                "models-glb", "fbx-format", "gltf-format", "glb-format", "obj-format", "fbx-unity",
                "gltf-godot-unreal", "gltf-godot-ue", "godot-ue", "unity"}


def authored_key(path):
    p = Path(path)
    parts = [slug(x) for x in p.parts[:-1]]
    parts = [x for x in parts if x not in EXPORT_LANES]
    parts = ["normal" if x == "with-knife-and-pan" else x for x in parts]
    return "/".join(parts + [slug(p.stem)])


def actor_scope(row):
    sid, path = row["source_id"], row["relative_source_path"].lower()
    if sid == "quaternius-universal-base-characters":
        return "/base characters/" in "/" + path
    if sid in {"kenney-blocky-characters", "kenney-cube-pets"}:
        return Path(path).stem.startswith(("character-", "animal-"))
    return bool(row.get("full_static_truth_ready"))


def resource_closure(path):
    path = Path(path)
    if path.suffix.lower() not in {".gltf", ".glb"}:
        return {"status": "FORMAT_NATIVE_RESOURCE_AUDIT_REQUIRED", "missing": []}
    document, _ = _json_and_chunks(path)
    missing, resources = [], []
    for row in [*document.get("buffers", []), *document.get("images", [])]:
        uri = row.get("uri", "")
        if not uri or uri.startswith("data:"):
            continue
        parsed = urlparse(uri)
        target = (path.parent / unquote(parsed.path)).resolve()
        if parsed.scheme or parsed.netloc or not target.is_relative_to(path.parent.resolve()) or not target.is_file():
            missing.append(uri)
        else:
            resources.append({"uri": uri, "sha256": sha(target)})
    return {"status": "INCOMPLETE_OR_UNSUPPORTED_DECLARED_RESOURCES" if missing else "COMPLETE_DECLARED_GLTF_RESOURCES",
            "missing": missing, "resources": resources}


def compile_candidates(base_manifest, base_root, structure, admission, workspace, additional_receipt):
    base_rows = base_manifest["subjects"]
    if len({x["subject_id"] for x in base_rows}) != len(base_rows) or base_manifest["subject_count"] != len(base_rows):
        raise ValueError("EXPANDED_BASE_IDENTITY_ROWS")
    original, failures = [], []
    for row in base_rows:
        source = (Path(base_root) / row["corpus_relative_path"]).resolve()
        if sha(source) != row["expected_sha256"]:
            raise ValueError("EXPANDED_BASE_SOURCE_DRIFT:" + row["subject_id"])
        original.append({"subject_id": row["subject_id"], "origin": "ESTABLISHED_BASE",
                         "source": {"path": str(source), "sha256": sha(source)},
                         "source_id": Path(row["corpus_relative_path"]).parts[2],
                         "source_representation_count": 1, "legacy_base_id": row["subject_id"],
                         "training_authorized": False})
    decisions = {x["subject_id"]: x for x in admission["subjects"]}
    rows = [x for x in structure["subjects"] if x["role"] != "REFERENCE_BASE139"]
    rows = [x for x in rows if actor_scope(x) and decisions[x["subject_id"]]["status"] not in {
        "EXCLUDED_BY_SEMANTIC_POLICY", "QUARANTINE_PROVENANCE", "DUPLICATE_OF_BASE139_STRUCTURE"}]
    by_id = {x["subject_id"]: x for x in rows}
    parent = {x: x for x in by_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    by_authored = {}
    for row in rows:
        key = row["source_id"], authored_key(row["relative_source_path"])
        if key in by_authored:
            union(row["subject_id"], by_authored[key])
        by_authored[key] = row["subject_id"]
        alias = decisions[row["subject_id"]].get("canonical_subject_id")
        if alias in by_id:
            union(row["subject_id"], alias)
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["subject_id"])].append(row)
    candidates = []
    for members in groups.values():
        sources = {x["source_id"] for x in members}
        if len(sources) != 1:
            failures.append({"code": "CROSS_SOURCE_ALIAS_REVIEW", "subjects": [x["subject_id"] for x in members]})
            continue
        sid = members[0]["source_id"]
        representations = {}
        for row in members:
            for rep in row["representations"]:
                p = (Path(workspace) / sid / "raw" / rep["path"]).resolve()
                if p.suffix.lower() not in {".gltf", ".glb", ".fbx"}:
                    continue
                if not p.is_file() or sha(p) != rep["sha256"]:
                    failures.append({"code": "REPRESENTATION_BYTE_DRIFT", "path": str(p)})
                    continue
                representations[(str(p), rep["sha256"])] = {"path": str(p), "sha256": rep["sha256"],
                    "publisher_relative_path": rep["path"], "resource_closure": resource_closure(p)}
        if not representations:
            continue
        ranked = sorted(representations.values(), key=lambda x: (
            bool(x["resource_closure"]["missing"]), "with knife and pan" in x["publisher_relative_path"].lower(),
            {".gltf": 0, ".glb": 1, ".fbx": 2}[Path(x["path"]).suffix.lower()], x["path"]))
        key = min(authored_key(x["relative_source_path"]) for x in members)
        candidates.append({"subject_id": "corpus_" + digest([sid, key])[:20], "origin": "EXPANSION",
            "source_id": sid, "authored_model_key": key, "source": {k: ranked[0][k] for k in ("path", "sha256")},
            "representations": ranked, "source_representation_count": len(ranked),
            "prior_subject_ids": sorted(x["subject_id"] for x in members),
            "source_capabilities_are_prior_observations_only": True, "training_authorized": False})
    recovered = defaultdict(list)
    for row in additional_receipt["files"]:
        if Path(row["relative_path"]).suffix.lower() in {".gltf", ".glb"}:
            recovered[(row["source_id"], authored_key(row["relative_path"]))].append(row)
    for (sid, key), members in sorted(recovered.items()):
        members.sort(key=lambda x: ("with knife and pan" in x["relative_path"].lower(), x["relative_path"]))
        reps = []
        for row in members:
            p = (Path(workspace) / sid / "raw" / row["relative_path"]).resolve()
            if sha(p) != row["sha256"]:
                raise ValueError("RECOVERED_SOURCE_BYTE_DRIFT")
            reps.append({"path": str(p), "sha256": row["sha256"], "publisher_relative_path": row["relative_path"],
                         "provider_file_id": row["file_id"], "resource_closure": resource_closure(p)})
        candidates.append({"subject_id": "corpus_" + digest([sid, key])[:20], "origin": "EXPANSION", "source_id": sid,
            "authored_model_key": key, "source": {k: reps[0][k] for k in ("path", "sha256")},
            "representations": reps, "source_representation_count": len(reps), "training_authorized": False})
    subjects = sorted(original + candidates, key=lambda x: x["subject_id"])
    if len({x["subject_id"] for x in subjects}) != len(subjects):
        raise ValueError("EXPANDED_SUBJECT_ID_COLLISION")
    return {"schema": "RealSaS.ExpandedSubjectCandidates.v1", "status": "SOURCE_CANDIDATES__FAMILY_ADMISSION_PENDING",
            "base_row_count": len(original), "expansion_candidate_count": len(candidates), "subject_row_count": len(subjects),
            "expansion_source_counts": dict(sorted(Counter(x["source_id"] for x in candidates).items())), "subjects": subjects,
            "failures": failures, "training_authorized": False, "split_created": False,
            "base_row_count_is_unique_structural_count": False,
            "definition": "ESTABLISHED_BASE_ROWS_PLUS_AUTHORED_EXPANSION_CANDIDATES__NOT_QUALIFIED_TRAINING_N"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-manifest", "base-root", "structure", "admission", "workspace", "additional-receipt", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = compile_candidates(json.loads(a.base_manifest.read_text()), a.base_root, json.loads(a.structure.read_text()),
                                json.loads(a.admission.read_text()), a.workspace, json.loads(a.additional_receipt.read_text()))
    result["input_file_sha256"] = {str(x.resolve()): sha(x) for x in (a.base_manifest, a.structure, a.admission, a.additional_receipt)}
    result["producer_sha256"] = sha(__file__)
    result["candidate_identity_sha256"] = digest(result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"subjects", "input_file_sha256"}}, sort_keys=True))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
