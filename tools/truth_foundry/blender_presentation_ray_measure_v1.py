#!/usr/bin/env python3
"""Run the proven source-ray renderer intervention court on TeacherSceneIR.v2.

The historical measurement implementation is intentionally left byte-for-byte
unchanged because its V1 FIT8 court is already sealed.  This adapter changes only
the accepted *envelope schema*: TeacherSceneIR.v2 carries the same geometry and
appearance payload bindings plus the stronger exact observation-rest proof.

``_sha`` remains untouched, so the measurement receipt binds the real V2 teacher
bytes rather than a rewritten compatibility document.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Blender executes --python entrypoints with its own embedded Python path.  The
# released repository root is not implicitly importable there, so this adapter
# must establish the same standalone bootstrap used by the other Blender truth
# entrypoints before importing `tools.*`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.iris.truth.presentation_pass_domain_v1 import admit_presentation_pass_domain_v1
from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as measured


_original_load = measured._load


def _load_with_v2_teacher(path):
    value = _original_load(path)
    if isinstance(value, dict) and value.get("schema") == "RealSaS.TeacherSceneIR.v2":
        if (
            value.get("authority") != "UNQUALIFIED_DERIVED_MEASUREMENTS"
            or value.get("qualification_minted") is not False
            or value.get("rest_contract") != "EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY"
            or (value.get("observation_rest_binding") or {}).get("status")
            != "PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY"
        ):
            raise RuntimeError("PRESENTATION_RAY_V2_TEACHER_NOT_EXACT_OBSERVATION_REST")
        compatible = dict(value)
        compatible["schema"] = "RealSaS.TeacherSceneIR.v1"
        return compatible
    return value


def _argument_path(flag: str) -> Path:
    try:
        index = sys.argv.index(flag)
        value = sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("PRESENTATION_RAY_ARGUMENT_MISSING:" + flag) from exc
    return Path(value).resolve()


def _admit_preregistered_pass_domain_after_raw_failures() -> dict:
    output = _argument_path("--output")
    # Platform layout is <run_root>/artifacts/48_PRESENTATION_RESPONSE_TRUTH/PACKED_X.json.
    # Fail closed outside that exact Attempt-bound layout rather than consulting
    # a mutable repository-global policy file.
    try:
        run_root = output.parents[2]
    except IndexError as exc:
        raise RuntimeError("PRESENTATION_RAY_RUN_ROOT_UNRESOLVED") from exc
    manifest_path = run_root / "run_manifest.json"
    if not output.is_file() or not manifest_path.is_file():
        raise RuntimeError("PRESENTATION_RAY_PASS_DOMAIN_EVIDENCE_MISSING")
    candidate = json.loads(output.read_text())
    run_manifest = json.loads(manifest_path.read_text())
    try:
        receipt = admit_presentation_pass_domain_v1(candidate, run_manifest)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    measured._load = _load_with_v2_teacher
    try:
        measured.main()
    except RuntimeError as exc:
        # The sealed historical court writes the complete raw candidate before
        # raising when any individual ray fails scalar factorization.  Stage48
        # preregistered a subject-level minimum PASS-ray domain, so preserve the
        # failed rays as evidence and return success only when the exact Attempt
        # manifest's minimum is independently satisfied.
        if str(exc) != "SOURCE_RAY_APPEARANCE_INTERVENTION_REPLAY_FAILED":
            raise
        _admit_preregistered_pass_domain_after_raw_failures()


if __name__ == "__main__":
    main()
