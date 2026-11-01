#!/usr/bin/env python3
"""Rebind existing source/observation controls; do not rerender or grant admission."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.corpus.build_source_observation_authority_v1 import build
from tools.corpus.validate_source_observation_authority_v1 import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readiness", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    subjects = json.loads(a.readiness.read_text())["subjects"]
    sources = {r["subject_id"]: r["source"] for r in json.loads(a.candidates.read_text())["subjects"]}
    a.output_root.mkdir(parents=True, exist_ok=True)
    def run(item):
        sid, refs = item
        try:
            ref = refs["geometry"]
            if sha(ref["path"]) != ref["sha256"]:
                raise ValueError("SOURCE_OBSERVATION_REVALIDATION_GEOMETRY_DRIFT")
            gm = json.loads(Path(ref["path"]).read_text())
            source = sources[sid]
            if sha(source["path"]) != source["sha256"] or gm["parents"]["source_sha256"] != source["sha256"]:
                raise ValueError("SOURCE_OBSERVATION_REVALIDATION_SOURCE_DRIFT")
            packet = Path(gm["parents"]["camera_bundle"]["path"]).parent
            out = a.output_root / sid
            build(sid, Path(source["path"]), packet, out)
            manifest = out / "SOURCE_OBSERVATION_AUTHORITY_MANIFEST_V1.json"
            court = validate(manifest)
            cp = out / "SOURCE_OBSERVATION_AUTHORITY_COURT_V1.json"
            cp.write_text(json.dumps(court, sort_keys=True, indent=2) + "\n")
            return {"subject_id": sid, "status": "PASS_EXISTING_SOURCE_OBSERVATION_REBIND",
                    "manifest": {"path": str(manifest.resolve()), "sha256": sha(manifest)},
                    "court": {"path": str(cp.resolve()), "sha256": sha(cp)}}
        except Exception as exc:
            return {"subject_id": sid, "status": "BLOCKED", "error_type": type(exc).__name__, "reason": str(exc)}
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run, sorted(subjects.items())):
            results.append(result)
            if len(results) % 20 == 0:
                print(json.dumps({"done": len(results), "total": len(subjects)}), flush=True)
    receipt = {"schema": "RealSaS.ExistingSourceObservationRevalidation.v1", "subject_count": len(subjects),
               "outcomes": dict(Counter(r.get("reason", r["status"]) for r in results)), "subjects": results,
               "producer_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                   Path(__file__).resolve(), ROOT / "tools/corpus/build_source_observation_authority_v1.py",
                   ROOT / "tools/corpus/validate_source_observation_authority_v1.py")},
               "inputs": {str(p.resolve()): sha(p) for p in (a.readiness, a.candidates)},
               "source_observations_regenerated": False, "training_authorized": False, "qualification_minted": False}
    (a.output_root / "EXISTING_SOURCE_OBSERVATION_REVALIDATION_V1.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "subjects"}, sort_keys=True))
    if any(r["status"] == "BLOCKED" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
