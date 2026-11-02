#!/usr/bin/env python3
"""Finite, resumable source-inspection worker; no corpus/training authority."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools/corpus/blender_teacher_source_inspection_v1.py"
BLENDER_SHA256 = "6fe78c9e0a1cca51a598eb37c3288b975357efbe86dcee18d5e6fdecaaf1847e"
PRODUCER_FILES = {"tools/corpus/blender_teacher_source_inspection_v1.py",
    "tools/observation/blender_controlled_8view_v2.py", "tools/observation/blender_controlled_8view_v1.py",
    "tools/observation/source_scene_contract_v2.py"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reusable_report(path, row):
    """A stale or incomplete cache is a miss, never a successful inspection."""
    try:
        m = json.loads(Path(path).read_text())
        return bool(m.get("subject_id") == row["subject_id"] and m.get("source") == row["source"]
            and sha(row["source"]["path"]) == row["source"]["sha256"]
            and m.get("runtime", {}).get("blender") == "4.0.2"
            and set(m.get("producer_source_sha256", {})) == PRODUCER_FILES
            and set(m.get("outputs", {})) == {"rest_geometry", "snapshot"}
            and all(sha(ROOT / p) == h for p, h in m["producer_source_sha256"].items())
            and all(sha(r["path"]) == r["sha256"] for r in m["outputs"].values()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--blender", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.blender = a.blender.resolve()
    if sha(a.blender) != BLENDER_SHA256:
        raise ValueError("SOURCE_INSPECTION_BLENDER_BINARY_DRIFT")
    source = json.loads(a.candidates.read_text())
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs, resumed = [], []
    for row in source["subjects"]:
        old = root / row["subject_id"] / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json"
        valid = reusable_report(old, row)
        if valid:
            resumed.append(row["subject_id"])
        else:
            jobs.append(row)
    batches = []
    for i in range(0, len(jobs), 8):
        path = root / f"source_job_{i:04d}.json"
        path.write_text(json.dumps(jobs[i:i + 8]))
        batches.append(path)
    def run(path):
        with path.with_suffix(".log").open("w") as log:
            proc = subprocess.run([str(a.blender), "--background", "--factory-startup", "--threads", "1",
                "--python", str(WORKER), "--", str(path), str(root)], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        return {"job": path.name, "exit_code": proc.returncode}
    outcomes = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run, batches):
            outcomes.append(result)
            print(json.dumps({"jobs_done": len(outcomes), "jobs_total": len(batches), "last": result}), flush=True)
    invalid = [r["subject_id"] for r in source["subjects"] if not reusable_report(
        root / r["subject_id"] / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json", r)]
    receipt = {"schema": "RealSaS.SourceInspectionExecutionReceipt.v1", "jobs": outcomes,
               "subject_count": len(source["subjects"]), "resumed_subject_ids": resumed,
               "invalid_or_missing_subject_ids": invalid,
               "candidate_manifest": {"path": str(a.candidates.resolve()), "sha256": sha(a.candidates)},
               "blender_sha256": BLENDER_SHA256, "training_authorized": False}
    (root / "IMPORT_EXECUTION_RECEIPT.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    if invalid or any(x["exit_code"] != 0 for x in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
