from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.motion_compile_v2 import _is_ancestor, _tree
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)


def _context(*, repo_root: Path, authority_root: Path, run_id: str):
    run_root = (authority_root / "runs" / run_id).resolve()
    ledger = json.loads((run_root / "ACTIVE_RUN_V2.json").read_text())
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    return {
        "repo_root": repo_root.resolve(),
        "authority_root": authority_root.resolve(),
        "run_root": run_root,
        "run_id": run_id,
        "run_manifest_path": run_root / "run_manifest.json",
        "run_manifest": manifest,
        "stage": {"id": "MOTION_RETARGET_ANCESTRY_DIAGNOSTIC"},
        "ledger": ledger,
    }


def diagnose(*, repo_root: Path, authority_root: Path, run_id: str, clip_id: str) -> dict:
    ctx = _context(
        repo_root=repo_root,
        authority_root=authority_root,
        run_id=run_id,
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    clip_path = (
        ctx["run_root"]
        / "inputs"
        / "motion"
        / "quaternius_knight_v1"
        / f"{clip_id}.motion.json"
    )
    payload = json.loads(clip_path.read_text())

    src_rows = tuple(payload.get("source_skeleton") or ())
    sids, spar, schildren, spos, sfeat, sroot = _tree(
        src_rows,
        id_key="source_joint_id",
        parent_key="parent_source_joint_id",
        pos_key="rest_position",
    )
    tgt_rows = tuple(
        {
            "joint_id": j.canonical_joint_id,
            "parent_id": j.parent_canonical_id,
            "position": j.position,
        }
        for j in skeleton.joints
    )
    tids, tpar, tchildren, tpos, tfeat, troot = _tree(
        tgt_rows,
        id_key="joint_id",
        parent_key="parent_id",
        pos_key="position",
    )

    cost = np.zeros((len(tids), len(sids)), dtype=np.float64)
    for i, tid in enumerate(tids):
        tf = tfeat[tid]
        for j, sid in enumerate(sids):
            sf = sfeat[sid]
            c = 2.0 * float(np.linalg.norm(tf[:3] - sf[:3]))
            c += 0.45 * abs(float(tf[3] - sf[3]))
            c += 0.35 * abs(float(tf[4] - sf[4]))
            c += 0.35 * abs(float(tf[5] - sf[5]))
            if (tid == troot) != (sid == sroot):
                c += 1000.0
            if (
                abs(float(tf[0])) > 0.05
                and abs(float(sf[0])) > 0.05
                and math.copysign(1.0, float(tf[0]))
                != math.copysign(1.0, float(sf[0]))
            ):
                c += 4.0
            cost[i, j] = c

    target_rows_idx, source_cols = linear_sum_assignment(cost)
    target_to_source = {
        tids[int(target_row)]: sids[int(source_col)]
        for target_row, source_col in zip(target_rows_idx, source_cols)
    }

    violations = []
    for tid in tids:
        target_parent = tpar[tid]
        if target_parent is None:
            continue
        source_child = target_to_source[tid]
        source_parent = target_to_source[target_parent]
        if _is_ancestor(spar, source_parent, source_child):
            continue
        alternatives = []
        target_index = tids.index(tid)
        for source_index, candidate in enumerate(sids):
            if candidate in target_to_source.values():
                continue
            if _is_ancestor(spar, source_parent, candidate):
                alternatives.append(
                    {
                        "source_joint_id": candidate,
                        "cost": float(cost[target_index, source_index]),
                    }
                )
        alternatives.sort(key=lambda row: (row["cost"], row["source_joint_id"]))
        violations.append(
            {
                "target_parent_joint_id": target_parent,
                "target_child_joint_id": tid,
                "mapped_source_parent_joint_id": source_parent,
                "mapped_source_child_joint_id": source_child,
                "mapped_source_parent_is_ancestor": False,
                "target_child_cost": float(
                    cost[target_index, sids.index(source_child)]
                ),
                "lowest_cost_unused_valid_descendants": alternatives[:8],
            }
        )

    assignments = []
    for tid in tids:
        sid = target_to_source[tid]
        row = cost[tids.index(tid)]
        order = np.argsort(row, kind="stable")[:5]
        assignments.append(
            {
                "target_joint_id": tid,
                "target_parent_id": tpar[tid],
                "source_joint_id": sid,
                "source_parent_id": spar[sid],
                "cost": float(row[sids.index(sid)]),
                "five_lowest_cost_sources": [
                    {
                        "source_joint_id": sids[int(index)],
                        "cost": float(row[int(index)]),
                    }
                    for index in order
                ],
            }
        )

    return {
        "schema": "RealSaS.MotionRetargetAncestryDiagnostic.v1",
        "status": "MEASURED",
        "run_id": run_id,
        "clip_id": clip_id,
        "source_joint_count": len(sids),
        "target_joint_count": len(tids),
        "source_root": sroot,
        "target_root": troot,
        "unconstrained_assignment_total_cost": float(
            sum(
                cost[int(target_row), int(source_col)]
                for target_row, source_col in zip(target_rows_idx, source_cols)
            )
        ),
        "root_mapping_correct": target_to_source.get(troot) == sroot,
        "ancestry_violation_count": len(violations),
        "ancestry_violations": violations,
        "assignments": assignments,
        "diagnostic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clip-id", default="demo_idle_v1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = diagnose(
        repo_root=Path(args.repo_root).resolve(),
        authority_root=Path(args.authority_root).resolve(),
        run_id=args.run_id,
        clip_id=args.clip_id,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("MOTION_RETARGET_ANCESTRY_DIAGNOSTIC", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
