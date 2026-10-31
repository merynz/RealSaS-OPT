#!/usr/bin/env python3
from __future__ import annotations

"""Canonical Full139 sweep entrypoint with final animation replay court.

V3 extends V1 subject execution after all source/rig artifacts exist, then uses V2's
failure-tolerant aggregate.  It upgrades ANIMATION_KINEMATICS from pending only when
the independent canonical replay court passes.  No other family or global authority
is changed.
"""

import json
from pathlib import Path
import sys

from tools.corpus import run_full139_truth_family_sweep_v1 as impl
from tools.corpus.run_full139_truth_family_sweep_v2 import aggregate


_BASE_RUN_SUBJECT = impl.run_subject


def upgrade_animation_replay(subject_id: str, receipt: dict, *, repo_root: Path, output_root: Path, resume: bool) -> dict:
    if str(receipt.get("status") or "").startswith("FAIL_"):
        return receipt
    subject_root = (output_root / "subjects" / subject_id).resolve()
    logs = subject_root / "logs"
    rig = subject_root / "rig_skin"
    anim = subject_root / "animation"
    replay = subject_root / "animation_replay"
    teacher_ref = dict((receipt.get("inputs") or {}).get("teacher_scene") or {})
    teacher_scene = Path(str(teacher_ref.get("path") or "")).resolve()
    if not teacher_scene.is_file() or impl.sha256(teacher_scene) != teacher_ref.get("sha256"):
        raise ValueError("FULL139_ANIMATION_REPLAY_TEACHER_DRIFT:" + subject_id)

    animation_manifest = anim / "GLTF_ANIMATION_KINEMATICS_MANIFEST_V1.json"
    animation_court = anim / "GLTF_ANIMATION_KINEMATICS_COURT_V1.json"
    rig_manifest = rig / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V1.json"
    rig_importer_court = rig / "GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE_COURT_V1.json"
    for code, path in (
        ("ANIMATION_MANIFEST", animation_manifest),
        ("ANIMATION_COURT", animation_court),
        ("RIG_MANIFEST", rig_manifest),
        ("RIG_IMPORTER_COURT", rig_importer_court),
    ):
        if not path.is_file():
            raise FileNotFoundError("FULL139_ANIMATION_REPLAY_" + code + ":" + str(path))

    py = sys.executable
    def tool(rel: str) -> str:
        return str((repo_root / rel).resolve())

    replay_manifest = impl.step(
        "animation_replay_build",
        [
            py, tool("tools/corpus/build_gltf_animation_canonical_replay_v1.py"),
            "--animation-manifest", str(animation_manifest),
            "--animation-court", str(animation_court),
            "--rig-manifest", str(rig_manifest),
            "--rig-importer-court", str(rig_importer_court),
            "--teacher-scene", str(teacher_scene),
            "--output", str(replay / "GLTF_ANIMATION_CANONICAL_REPLAY_V1.json"),
        ],
        replay / "GLTF_ANIMATION_CANONICAL_REPLAY_V1.json",
        repo_root=repo_root, logs=logs, resume=resume,
    )
    replay_court = impl.step(
        "animation_replay_court",
        [
            py, tool("tools/corpus/validate_gltf_animation_canonical_replay_v1.py"),
            "--manifest", str(replay_manifest),
            "--output", str(replay / "GLTF_ANIMATION_CANONICAL_REPLAY_COURT_V1.json"),
        ],
        replay / "GLTF_ANIMATION_CANONICAL_REPLAY_COURT_V1.json",
        repo_root=repo_root, logs=logs, resume=resume,
    )
    court = impl.court_file(replay_court, "ANIMATION_REPLAY")
    outcome = str((court.get("family_candidate_outcomes") or {}).get("ANIMATION_KINEMATICS") or "")
    if outcome not in {"QUALIFIED_CORE_CANDIDATE", "QUALIFIED_MASKED_CANDIDATE"}:
        raise ValueError("FULL139_ANIMATION_REPLAY_OUTCOME:" + outcome)

    receipt = dict(receipt)
    courts = dict(receipt.get("courts") or {})
    courts["animation_replay"] = impl.ref(replay_court)
    receipt["courts"] = courts
    family = dict(receipt.get("canonical_family_candidates") or {})
    family["ANIMATION_KINEMATICS"] = outcome
    receipt["canonical_family_candidates"] = family
    receipt["pending_gates"] = [
        gate for gate in receipt.get("pending_gates") or []
        if gate != "ANIMATION_CANONICAL_REPLAY_COURT"
    ]
    receipt["animation_replay"] = {
        "manifest": impl.ref(replay_manifest),
        "court": impl.ref(replay_court),
        "family_candidate_outcome": outcome,
    }
    receipt_path = subject_root / "SUBJECT_TRUTH_FAMILY_SWEEP_RECEIPT_V1.json"
    impl.write_json(receipt_path, receipt)
    receipt["receipt_sha256"] = impl.sha256(receipt_path)
    return receipt


def run_subject(subject_id, inventory_row, source_cache_row, *, repo_root, output_root, resume):
    receipt = _BASE_RUN_SUBJECT(
        subject_id, inventory_row, source_cache_row,
        repo_root=repo_root, output_root=output_root, resume=resume,
    )
    return upgrade_animation_replay(
        subject_id, receipt, repo_root=repo_root, output_root=output_root, resume=resume,
    )


def main() -> None:
    impl.run_subject = run_subject
    impl.aggregate = aggregate
    impl.main()


if __name__ == "__main__":
    main()
