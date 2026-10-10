"""Read-only, exact-byte Knight input/context inspection; no stage execution.

The capsule preserves original bytes and paths in a content-addressed catalogue.
It is input transport, never an Attempt, qualification or reusable stage result.
Only pinned inputs and their explicitly referenced files are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
import zipfile

from tools.platform_render_checkpoint import atomic_json, checkout_identity, digest

CAA_RUN = "KNIGHT_LATEST_CURRENT_CAA_VISUAL_20261010"
MAT_RUN = "KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010"


class SealedInputCatalogue:
    def __init__(self, roots, *, byte_budget=200 << 20):
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.byte_budget = byte_budget
        self.rows = {}
        self.unbound_references = []
        self.total_bytes = 0

    def add(self, raw_path, *, expected=None, origin):
        path = Path(raw_path).resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise RuntimeError("KNIGHT_INSPECTION_PATH_OUTSIDE_DECLARED_ROOTS:" + str(path))
        if not path.is_file():
            raise RuntimeError("KNIGHT_INSPECTION_REFERENCED_FILE_MISSING:" + str(path))
        actual = digest(path)
        if expected is not None and actual != expected:
            raise RuntimeError("KNIGHT_INSPECTION_REFERENCED_BYTES_DRIFT:" + str(path))
        key = str(path)
        if key in self.rows:
            if self.rows[key]["sha256"] != actual:
                raise RuntimeError("KNIGHT_INSPECTION_SOURCE_CHANGED_DURING_READ:" + key)
            return
        size = path.stat().st_size
        if self.total_bytes + size > self.byte_budget:
            raise RuntimeError("KNIGHT_INSPECTION_BYTE_BUDGET_EXCEEDED")
        self.total_bytes += size
        self.rows[key] = {"path": key, "sha256": actual, "size_bytes": size,
                          "origin": origin, "expected_hash_present": expected is not None}
        if path.suffix.lower() == ".json":
            self.references(json.loads(path.read_text()), origin=key)

    def references(self, value, *, origin):
        if isinstance(value, list):
            for row in value:
                self.references(row, origin=origin)
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and (key == "path" or key.endswith("_path")):
                    # JSON path strings are not automatically scientific bindings.
                    # Only an explicit sibling SHA permits recursive transport.
                    hash_key = "sha256" if key == "path" else key[:-5] + "_sha256"
                    expected = value.get(hash_key)
                    if isinstance(expected, str) and len(expected) == 64:
                        self.add(item, expected=expected, origin=origin + "#" + key)
                    else:
                        self.unbound_references.append({"origin": origin, "key": key, "path": item})
                elif isinstance(item, (dict, list)):
                    self.references(item, origin=origin)

    def write(self, out, *, context):
        out.mkdir(parents=True, exist_ok=True)
        report = {"schema": "RealSaS.SealedInputInspection.v1", "context": context,
                  "files": list(self.rows.values()), "total_bytes": self.total_bytes,
                  "unbound_references": self.unbound_references,
                  "qualification_minted": False, "model_inference_executed": False,
                  "input_payloads_rewritten": False, "full_visual_acceptance_passed": False}
        atomic_json(out / "SEALED_INPUT_INSPECTION.json", report)
        temporary = out / "SEALED_INPUTS.zip.partial"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(out / "SEALED_INPUT_INSPECTION.json", "SEALED_INPUT_INSPECTION.json")
            seen = set()
            for row in self.rows.values():
                path = Path(row["path"])
                if digest(path) != row["sha256"]:
                    raise RuntimeError("KNIGHT_INSPECTION_SOURCE_CHANGED_BEFORE_EXPORT:" + str(path))
                if row["sha256"] not in seen:
                    copied_hash = hashlib.sha256()
                    copied_size = 0
                    with path.open("rb") as source, archive.open("objects/" + row["sha256"], "w") as target:
                        for chunk in iter(lambda: source.read(8 << 20), b""):
                            copied_hash.update(chunk)
                            copied_size += len(chunk)
                            target.write(chunk)
                    if copied_hash.hexdigest() != row["sha256"] or copied_size != row["size_bytes"]:
                        raise RuntimeError("KNIGHT_INSPECTION_SOURCE_CHANGED_DURING_EXPORT:" + str(path))
                    seen.add(row["sha256"])
        temporary.replace(out / "SEALED_INPUTS.zip")
        return report


def go_context(platform_root):
    from tools.platform_deployment_smoke import registry_read, request
    rows = registry_read(platform_root,
        "SELECT id::text||'|'||slug FROM subjects "
        "WHERE lower(slug) LIKE '%knight%' ORDER BY slug LIMIT 20;")
    subjects = []
    for line in rows.splitlines():
        subject_id, slug = line.split("|", 1)
        subject_id = str(uuid.UUID(subject_id))
        subjects.append({"subject_id": subject_id, "slug": slug,
                         "agent_context": request("/v1/agents/context/" + subject_id)})
    return {"operation": "READ_ONLY_GO_AGENT_CONTEXT", "subjects": subjects,
            "subject_query_limit": 20, "subject_selection_is_execution_authority": False,
            "no_matching_subject_is_qualified_baseline": False}


def main(args):
    repo = Path(__file__).resolve().parents[2]
    code_sha = checkout_identity(repo, args.expected_main_sha)
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                     cwd=repo, text=True).split()[0]
    deployment = json.loads((args.platform_root / "deployment.json").read_text())
    if remote != code_sha or deployment["code_sha"] != code_sha:
        raise RuntimeError("KNIGHT_INSPECTION_REQUIRES_EXACT_DEPLOYED_LIVE_MAIN")
    catalogue = SealedInputCatalogue((args.authority_root, repo))
    caa, material = [args.authority_root / "runs" / name for name in (CAA_RUN, MAT_RUN)]
    for path in (caa / "ACTIVE_RUN_V2.json", caa / "run_manifest.json", caa / "CAA_RECEIPT.json",
                 material / "RECEIPT.json", material / "rebound_candidate.json",
                 material / "mechanical_carrier_evidence.json", material / "qualified_carrier_skin.json"):
        catalogue.add(path, origin="EXPLICIT_SEALED_KNIGHT_ROOT")
    inventory = json.loads((repo / "canonical/PLATFORM_KNIGHT_INPUT_INVENTORY_V1.json").read_text())
    for row in inventory["required_files"]:
        catalogue.add(args.input_root / row["name"], expected=row["sha256"], origin="PINNED_EXTERNAL_INVENTORY")
    for name in ("demo_idle_v1.motion.json", "demo_run_v1.motion.json", "demo_slash_v1.motion.json"):
        catalogue.add(args.input_root / name, origin="EXPLICIT_MOTION_INPUT__NOT_REUSE_QUALIFICATION")
    context = {"code_sha": code_sha, "go": go_context(args.platform_root),
               "source": "SEALED_KNIGHT_BYTES_AND_GO_READ_ONLY_STATE"}
    report = catalogue.write(args.out, context=context)
    print(json.dumps({"code_sha": code_sha, "file_count": len(report["files"]),
                      "byte_count": report["total_bytes"],
                      "go_subject_count": len(context["go"]["subjects"]),
                      "unbound_reference_count": len(report["unbound_references"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authority-root", "input-root", "platform-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-main-sha", required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out / "INSPECTION_FAILURE.json", {
            "schema": "RealSaS.SealedInputInspectionFailure.v1", "status": "FAILED_INSPECTION",
            "requested_code_sha": args.expected_main_sha, "error_type": type(error).__name__,
            "error": str(error), "qualification_minted": False, "model_inference_executed": False})
        raise
