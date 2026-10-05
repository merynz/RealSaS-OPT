from __future__ import annotations

"""Robust Blender-name adapter for the exact Knight source RUN frame0 render."""

import re
import sys
from pathlib import Path

import bpy

# Blender's embedded Python does not honor the workflow's PYTHONPATH reliably.
# Re-add the repository root explicitly before importing the shared renderer.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.research import blender_render_knight_source_run_frame0_v1 as base

DESIRED_CANONICAL = {"1H_Sword", "Round_Shield"}
ALL_CANONICAL = tuple(sorted(base.ALL_EQUIPMENT, key=len, reverse=True))


def canonical_equipment_name(raw_name: str) -> str | None:
    clean = re.sub(r"\.\d{3}$", "", str(raw_name))
    tail = re.split(r"[|:/\\]", clean)[-1]
    for canonical in ALL_CANONICAL:
        if tail == canonical or clean == canonical or clean.endswith(canonical):
            return canonical
    return None


def visible_meshes():
    out = []
    selected_actual = {}
    inventory = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        inventory.append(obj.name)
        canonical = canonical_equipment_name(obj.name)
        if canonical is not None:
            if canonical not in DESIRED_CANONICAL:
                obj.hide_render = True
                obj.hide_viewport = True
                continue
            selected_actual[canonical] = obj.name
        obj.hide_render = False
        obj.hide_viewport = False
        out.append(obj)

    missing = sorted(DESIRED_CANONICAL - set(selected_actual))
    if missing:
        raise RuntimeError(
            "SOURCE_COMPARE_LOADOUT_MISSING_NORMALIZED:"
            + ",".join(missing)
            + ":inventory="
            + ",".join(sorted(inventory))
        )
    base.KEEP_EQUIPMENT = set(selected_actual.values())
    print("SOURCE_COMPARE_EQUIPMENT_RESOLUTION", selected_actual, flush=True)
    return tuple(out)


def main():
    base.visible_meshes = visible_meshes
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return base.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
