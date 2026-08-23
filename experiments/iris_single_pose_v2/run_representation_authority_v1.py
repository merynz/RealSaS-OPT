from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_SPLIT_FREEZE_SHA256 = "9e766ac61126c9b4787eef24146e36aac40cbeac166d67ba898f8b79133e9d66"
EXPECTED_SELECTION_SHA256 = "af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9"
EXPECTED_PANEL_ASSET_ID_LIST_SHA256 = "366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961"
EXPECTED_PANEL = 256


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def asset_id_list_sha256(asset_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(asset_ids) + "\n").encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(cmd: list[str | Path]) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call([str(x) for x in cmd])


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"{label} SHA mismatch expected={expected} got={got}")


def script_hashes(here: Path) -> dict[str, str]:
    names = [
        "build_representation_seed_v1.py",
        "stage_source.py",
        "prepare_cache.py",
        "audit_staged_cache_v2.py",
        "representation_authority_study_v1.py",
        "compact_representation_handoff_v1.py",
        "geometry.py",
    ]
    return {name: sha256_file(here / name) for name in names}


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command optimizer=0 R0-R3 Representation Authority gate")
    ap.add_argument("--root", default="/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3")
    ap.add_argument("--split-freeze", default=None)
    ap.add_argument("--selection-json", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--input-resolution", type=int, choices=(256, 512, 1024), default=1024)
    ap.add_argument("--geom-samples", type=int, default=4096)
    ap.add_argument("--anchors-per-view", type=int, default=1024)
    ap.add_argument("--max-tracks", type=int, default=4096)
    ap.add_argument("--radius-px", type=int, default=3)
    ap.add_argument("--max-surface-error", type=float, default=0.003)
    ap.add_argument("--mode", choices=("all", "seed", "stage", "cache", "audit", "study"), default="all")
    args = ap.parse_args()

    root = Path(args.root)
    here = Path(__file__).resolve().parent
    split_freeze = Path(args.split_freeze) if args.split_freeze else root / "reports" / "iris_controlled_v1" / "IRIS_CONTROLLED_V1_SPLIT_FREEZE.json"
    selection = Path(args.selection_json) if args.selection_json else root / "metadata" / "CANONICAL_VARIANT_SELECTION.json"
    work = Path(args.work_dir) if args.work_dir else root / "runs" / "IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V1"
    work.mkdir(parents=True, exist_ok=True)

    require_sha(split_freeze, EXPECTED_SPLIT_FREEZE_SHA256, "controlled split freeze")
    require_sha(selection, EXPECTED_SELECTION_SHA256, "canonical selection")

    seed = work / "REPRESENTATION_SEED_V1.json"
    stage = work / f"stage_R{args.input_resolution}"
    stage_manifest = stage / "STAGE_MANIFEST.json"
    cache = work / "cache_v2"
    cache_manifest = cache / "CACHE_MANIFEST.json"
    audit = work / "STAGE_CACHE_AUDIT_V1.json"
    result = work / "REPRESENTATION_AUTHORITY_RESULT_R0_R3_V1.json"
    panel = work / "REPRESENTATION_AUTHORITY_PANEL_V1.json"
    hard_tail = work / "REPRESENTATION_AUTHORITY_HARD_TAIL_V1.jsonl"
    compact = work / "REPRESENTATION_AUTHORITY_COMPACT_HANDOFF_V1.json"
    run_authority = work / "RUN_AUTHORITY_V1.json"

    scripts = script_hashes(here)
    authority = {
        "schema": "RealSaS.IRISSinglePoseV2.RepresentationRunAuthority.v1",
        "optimizer_steps": 0,
        "training_authorized": False,
        "root": str(root),
        "work_dir": str(work),
        "input_resolution": args.input_resolution,
        "truth_authority_resolution": 1024,
        "expected_panel_asset_id_list_sha256": EXPECTED_PANEL_ASSET_ID_LIST_SHA256,
        "settings": {
            "geom_samples": args.geom_samples,
            "anchors_per_view": args.anchors_per_view,
            "max_tracks": args.max_tracks,
            "radius_px": args.radius_px,
            "max_surface_error": args.max_surface_error,
        },
        "source_authorities": {
            "split_freeze": str(split_freeze),
            "split_freeze_sha256": sha256_file(split_freeze),
            "selection": str(selection),
            "selection_sha256": sha256_file(selection),
        },
        "script_sha256": scripts,
        "sealed_splits_opened": False,
        "status": "OPTIMIZER_ZERO_RUN_AUTHORITY_FROZEN",
    }
    if run_authority.exists():
        old = json.load(open(run_authority, encoding="utf-8"))
        fields = (
            "input_resolution", "truth_authority_resolution", "expected_panel_asset_id_list_sha256",
            "settings", "source_authorities", "script_sha256",
        )
        comparable = {k: old.get(k) for k in fields}
        current = {k: authority.get(k) for k in fields}
        if comparable != current:
            raise RuntimeError("existing run authority differs from current source/settings; choose a new --work-dir rather than mutating an opened run")
    else:
        atomic_json(run_authority, authority)

    if args.mode in ("all", "seed"):
        run([
            sys.executable, here / "build_representation_seed_v1.py",
            "--split-freeze", split_freeze,
            "--selection-json", selection,
            "--root", root,
            "--out", seed,
        ])
    if not seed.is_file():
        raise RuntimeError("representation seed absent")
    seed_obj = json.load(open(seed, encoding="utf-8"))
    if seed_obj.get("record_count") != EXPECTED_PANEL or seed_obj.get("sealed_splits_opened") is not False:
        raise RuntimeError("frozen 256/open-only seed contract failed")
    seed_ids = [r["asset_id"] for r in seed_obj["records"]]
    seed_id_digest = asset_id_list_sha256(seed_ids)
    if seed_id_digest != EXPECTED_PANEL_ASSET_ID_LIST_SHA256:
        raise RuntimeError(f"pre-result panel identity drift expected={EXPECTED_PANEL_ASSET_ID_LIST_SHA256} got={seed_id_digest}")

    if args.mode in ("all", "stage"):
        run([
            sys.executable, here / "stage_source.py",
            "--root", root,
            "--seed-manifest", seed,
            "--out", stage,
            "--input-resolution", str(args.input_resolution),
            "--splits", "FIT,TUNE",
        ])
    if not stage_manifest.is_file():
        raise RuntimeError("stage manifest absent")

    if args.mode in ("all", "cache"):
        run([
            sys.executable, here / "prepare_cache.py",
            "--stage-manifest", stage_manifest,
            "--out", cache,
            "--geom-samples", str(args.geom_samples),
            "--anchors-per-view", str(args.anchors_per_view),
            "--max-tracks", str(args.max_tracks),
            "--radius-px", str(args.radius_px),
            "--max-surface-error", str(args.max_surface_error),
        ])
    if not cache_manifest.is_file():
        raise RuntimeError("cache manifest absent")

    if args.mode in ("all", "audit"):
        run([
            sys.executable, here / "audit_staged_cache_v2.py",
            "--stage-manifest", stage_manifest,
            "--cache-manifest", cache_manifest,
            "--seed-manifest", seed,
            "--out", audit,
        ])
    if not audit.is_file():
        raise RuntimeError("stage/cache audit absent")
    audit_obj = json.load(open(audit, encoding="utf-8"))
    if audit_obj.get("status") != "PASS" or audit_obj.get("fatal_asset_count") != 0 or audit_obj.get("asset_count") != EXPECTED_PANEL:
        raise RuntimeError("stage/cache audit did not close PASS on the frozen 256; no asset substitution is allowed")

    if args.mode in ("all", "study"):
        run([
            sys.executable, here / "representation_authority_study_v1.py",
            "--cache-manifest", cache_manifest,
            "--selection-json", selection,
            "--stage-cache-audit", audit,
            "--out", result,
            "--panel-out", panel,
            "--hard-tail-out", hard_tail,
        ])
    if not result.is_file() or not panel.is_file() or not hard_tail.is_file():
        raise RuntimeError("representation result/panel/hard-tail absent")

    result_obj = json.load(open(result, encoding="utf-8"))
    panel_obj = json.load(open(panel, encoding="utf-8"))
    if result_obj.get("status") != "R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED" or result_obj.get("mode") != "CONFIRMATORY":
        raise RuntimeError(f"confirmatory result status invalid: {result_obj.get('status')}")
    if result_obj.get("optimizer_steps") != 0 or panel_obj.get("asset_target") != EXPECTED_PANEL:
        raise RuntimeError("optimizer/panel contract drift")

    panel_ids = panel_obj["selected_asset_ids"]
    if seed_ids != panel_ids:
        raise RuntimeError("study panel differs from the frozen pre-stage seed; result invalid")
    panel_id_digest = asset_id_list_sha256(panel_ids)
    if panel_id_digest != EXPECTED_PANEL_ASSET_ID_LIST_SHA256:
        raise RuntimeError("post-study panel digest differs from the pre-result lock")

    # Always create a small measurement-only handoff after the confirmatory result is frozen.
    run([
        sys.executable, here / "compact_representation_handoff_v1.py",
        "--result", result,
        "--hard-tail", hard_tail,
        "--out", compact,
    ])
    if not compact.is_file():
        raise RuntimeError("compact representation handoff absent")
    compact_obj = json.load(open(compact, encoding="utf-8"))
    if compact_obj.get("status") != "MEASUREMENT_ONLY__CANONICAL_INTERPRETATION_REQUIRED":
        raise RuntimeError("compact handoff decision-discipline drift")
    if compact_obj.get("training_authorized") is not False or compact_obj.get("optimizer_steps") != 0:
        raise RuntimeError("compact handoff optimizer/training authority drift")

    final = {
        **authority,
        "status": "R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED",
        "panel_asset_id_list_sha256": panel_id_digest,
        "artifacts": {
            "seed": str(seed), "seed_sha256": sha256_file(seed),
            "stage_manifest": str(stage_manifest), "stage_manifest_sha256": sha256_file(stage_manifest),
            "cache_manifest": str(cache_manifest), "cache_manifest_sha256": sha256_file(cache_manifest),
            "stage_cache_audit": str(audit), "stage_cache_audit_sha256": sha256_file(audit),
            "panel": str(panel), "panel_sha256": sha256_file(panel),
            "result": str(result), "result_sha256": sha256_file(result),
            "hard_tail": str(hard_tail), "hard_tail_sha256": sha256_file(hard_tail),
            "compact_handoff": str(compact), "compact_handoff_sha256": sha256_file(compact),
        },
        "asset_count": EXPECTED_PANEL,
        "arm_count": result_obj.get("arm_count"),
        "sealed_splits_opened": False,
        "training_authorized": False,
        "next_authority": "Canonical interpretation of R0-R3. Do not create or run an optimizer from this artifact.",
    }
    atomic_json(work / "RUN_COMPLETE_V1.json", final)
    print(json.dumps({
        "status": final["status"],
        "optimizer_steps": 0,
        "asset_count": final["asset_count"],
        "arm_count": final["arm_count"],
        "panel_asset_id_list_sha256": panel_id_digest,
        "result": str(result),
        "compact_handoff": str(compact),
        "run_complete": str(work / "RUN_COMPLETE_V1.json"),
        "sealed_splits_opened": False,
        "training_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
