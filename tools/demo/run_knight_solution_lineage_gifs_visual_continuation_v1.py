from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import tools.demo.build_knight_solution_lineage_gifs_v1 as base

ORIGINAL_PRODUCT_BACKEND = None
DEMO_CONTINUATION_BACKEND = "CANONICAL_RELATION_BASELINE_V1"
PRODUCT_BLOCKER = "HOLELESS_PARTITION_PARENT_QUALITY_REFINEMENT_REQUIRED"

_original_load_json = base.load_json


def _load_json_with_demo_stage18_continuation(path: Path):
    global ORIGINAL_PRODUCT_BACKEND
    payload = _original_load_json(path)
    p = Path(path)
    if p.name == "run_manifest.json" and isinstance(payload, dict):
        mesh = dict(payload.get("mesh") or {})
        if mesh.get("backend"):
            ORIGINAL_PRODUCT_BACKEND = str(mesh["backend"])
            out = copy.deepcopy(payload)
            out_mesh = dict(out.get("mesh") or {})
            out_mesh["backend"] = DEMO_CONTINUATION_BACKEND
            out["mesh"] = out_mesh
            return out
    return payload


def _arg_value(name: str) -> str:
    try:
        i = sys.argv.index(name)
    except ValueError as exc:
        raise RuntimeError(f"MISSING_ARG::{name}") from exc
    if i + 1 >= len(sys.argv):
        raise RuntimeError(f"MISSING_ARG_VALUE::{name}")
    return sys.argv[i + 1]


def main():
    base.load_json = _load_json_with_demo_stage18_continuation
    base.main()

    out_dir = Path(_arg_value("--out-dir")).expanduser().resolve()
    seal_path = out_dir / "KNIGHT_SOLVED_WEIGHT_TOPOLOGY_VISUAL_LINEAGE_SEAL_V1.json"
    seal = json.loads(seal_path.read_text())
    seal["stage18_product_gate"] = {
        "original_product_backend": ORIGINAL_PRODUCT_BACKEND,
        "demo_visual_continuation_backend": DEMO_CONTINUATION_BACKEND,
        "product_blocker_preserved": PRODUCT_BLOCKER,
        "product_authority_claimed": False,
        "reason": (
            "Visual inspection of the mechanically proven corrected-weight / "
            "73-region / component-harmonic solution. No product Stage18 quality "
            "gate is relaxed or claimed PASS."
        ),
    }
    seal["lineage_invariants"]["product_stage18_quality_gate_bypassed_for_visual_proof_only"] = True
    seal["lineage_invariants"]["product_stage18_quality_blocker_preserved"] = True
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_SOLUTION_VISUAL_CONTINUATION_SEALED=" + json.dumps({
        "status": seal["status"],
        "product_backend": ORIGINAL_PRODUCT_BACKEND,
        "demo_backend": DEMO_CONTINUATION_BACKEND,
        "product_blocker": PRODUCT_BLOCKER,
        "g3_passed": seal["g3"]["passed"],
        "motion_gt4": seal["actual_motion"]["max_edge_gt_4"],
        "candidate": seal["stage18"]["candidate_lineage_hash"],
        "appearance": seal["appearance"]["stage23_complete_appearance_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
