#!/usr/bin/env python3
"""Exact SHA native .blend control graph observation in Blender 4.5.14.

Only call with --disable-autoexec and --python-exit-code 1. Never execute
embedded .blend scripts or add-ons. The source is NOT automatically an
authoritative native runtime control system (some rigs need custom scripts).
No training or commercial use authority is minted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Blender bundled Python does not always add workspace to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import require, sha
from tools.corpus.blender_corpus4008_control_profile_v1 import source_control_profile

SCHEMA = "RealSaS.ExternalNativeBlendControlObservation.v1"


def run(bpy, source: Path, expected_sha256: str, *,
        source_repo: str, source_commit: str, asset_key: str):
    require(source.is_file() and source.suffix.lower() == ".blend",
            "NATIVE_BLEND_FILE_REQUIRED")
    require(sha(source) == expected_sha256, "EXACT_NATIVE_SOURCE_SHA")
    # Critical: no untrusted native .blend scripts may execute in worker.
    require(not bpy.app.autoexec_fail and not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "AUTOEXEC_MUST_BE_DISABLED")
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            "NATIVE_BLEND_SCRIPTS_ENABLED_AFTER_OPEN")
    profile = source_control_profile(bpy)
    require(profile["training_authorized"] is False,
            "CONTROL_PROFILE_MUST_DENY_TRAINING")
    return {
        "schema": SCHEMA,
        "status": "ORIGINAL_NATIVE_CONTROL_GRAPH_OBSERVATION_ONLY",
        "native_source_sha256": expected_sha256,
        "native_source_repo": source_repo,
        "native_source_commit": source_commit,
        "source_asset_key": asset_key,
        "blender_version": bpy.app.version_string,
        "source_file_script_execution_disabled": True,
        "source_embedded_custom_ui_functionality": "UNKNOWN",
        "source_native_ik_fk_functionality": "UNKNOWN",
        "source_native_attachment_response": "UNKNOWN",
        "commercial_ml_license_review": "PENDING_PER_ASSET",
        "source_independent_behavior_court": "NOT_RUN",
        "profile": profile,
        "all_model_head_truth_admitted": False,
        "training_authorized": False,
    }


def main():
    import bpy
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--asset-key", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run(bpy, a.source.resolve(), a.expected_sha256,
            source_repo=a.source_repo,
            source_commit=a.source_commit,
            asset_key=a.asset_key)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")
    p = r["profile"]
    print(json.dumps({
        "asset_key": r["source_asset_key"],
        "bones": p["imported_pose_bone_count"],
        "imported_ik_constraints": p["imported_ik_constraint_count"],
        "driver_count": p["imported_driver_count"],
        "constraint_types": p["imported_constraint_type_counts"],
        "native_ik_fk_functionality": "UNKNOWN",
        "training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    args = sys.argv
    try:
        sys.argv = [args[0]] + args[args.index("--") + 1:]
        main()
    finally:
        sys.argv = args
