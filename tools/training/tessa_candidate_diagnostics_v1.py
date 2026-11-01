"""Retain measured rejection evidence without granting teacher authority."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_candidate_diagnostics(*, subject_id, output_dir, candidates, parents, producer_files):
    nonfinite = []
    def clean(value, path=""):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append(path)
            return None
        if isinstance(value, dict):
            return {k: clean(v, path + "." + k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v, path + f"[{i}]") for i, v in enumerate(value)]
        return value
    rows = clean(candidates, "candidates")
    passed = sum(r.get("mechanical_pass") is True for r in rows)
    code = {str(Path(p).resolve()): sha(p) for p in [*producer_files, Path(__file__)]}
    payload = {"schema": "RealSaS.TessaCandidateEvaluationDiagnostics.v1", "subject_id": subject_id,
               "status": "MEASURED_CANDIDATES__INDEPENDENT_COURT_SEPARATE" if passed else "NO_MECHANICALLY_ADMISSIBLE_CANDIDATE",
               "parents": parents, "producer_source_sha256": code, "candidates": rows,
               "candidate_count": len(rows), "mechanical_pass_count": passed,
               "nonfinite_metric_paths": nonfinite, "training_authorized": False, "qualification_minted": False}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    payload["diagnostic_identity_sha256"] = identity
    output = Path(output_dir) / ("TESSA_CANDIDATE_DIAGNOSTICS_" + identity[:16] + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if output.exists() and output.read_text() != text:
        raise ValueError("TESSA_DIAGNOSTIC_IMMUTABLE_OUTPUT_COLLISION")
    temp = output.with_suffix(".partial.json")
    temp.write_text(text)
    temp.replace(output)
    return output
