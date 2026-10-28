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

from pathlib import Path
import sys

# Blender executes --python entrypoints with its own embedded Python path.  The
# released repository root is not implicitly importable there, so this adapter
# must establish the same standalone bootstrap used by the other Blender truth
# entrypoints before importing `tools.*`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main() -> None:
    measured._load = _load_with_v2_teacher
    measured.main()


if __name__ == "__main__":
    main()
