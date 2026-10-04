"""Colab runner for Knight V9 seam-support locality counterfactual.

Uses the completed Drive mechanics result. Does not rerun repair operators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

COURT_COMMIT = "bc69f8c8fbdb46e79754bba6ea8fe633fab2ca64"
MECHANICS_SOURCE_COMMIT = "06f3453f754db73ed4230ff4f295b7a58ba931ea"
EXPECTED_FINAL_CANDIDATE_SHA256 = "a49170c38167291e629d9c05d91f70193c1547e44a8e0da4e1b649a2f6b36baa"
EXPECTED_FINAL_PARTITION_SHA256 = "b2503c49d3210e48e85f07225bfbcb234350c18674e4a191dac53d26d25be7ed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(event: str, **fields) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    suffix = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[{ts}Z] {event}" + (f" {suffix}" if suffix else ""), flush=True)


def run_streaming(cmd, *, cwd, env, log_path):
    with Path(log_path).open("w", encoding="utf-8", buffering=1) as fh:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"SEAM_SUPPORT_COUNTERFACTUAL_CHILD_FAILED:{rc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--drive-base",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/RealSaS_SUBJECT2_KNIGHT_DEMO_V2_20260924/"
            "KNIGHT_V9_STATIC_MECHANICS_NOTEBOOK_20261004"
        ),
    )
    args = ap.parse_args()
    base = args.drive_base.resolve()

    latest = json.loads((base / "RESULTS/LATEST_RUN.json").read_text(encoding="utf-8"))
    if latest.get("status") != "PASS":
        raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_LATEST_RUN_NOT_PASS")
    result_dir = Path(str(latest["result_dir"]))
    summary_path = result_dir / "RUN_ALL_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("source_commit") != MECHANICS_SOURCE_COMMIT:
        raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_MECHANICS_SOURCE_DRIFT")
    if sha256(summary_path) != str(latest.get("summary_sha256")):
        raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_SUMMARY_SHA_DRIFT")

    candidate = result_dir / "FINAL_CANDIDATE.json"
    partition = result_dir / "FINAL_PARTITION.json"
    if sha256(candidate) != EXPECTED_FINAL_CANDIDATE_SHA256:
        raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_CANDIDATE_SHA_DRIFT")
    if sha256(partition) != EXPECTED_FINAL_PARTITION_SHA256:
        raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_PARTITION_SHA_DRIFT")

    handoff = base / "HANDOFF"
    inputs = {
        "surface": handoff / "REFINED_SURFACE.json",
        "fresh_skeleton": handoff / "FRESH_QUALIFIED_SKELETON.json",
        "teacher_bank": handoff / "V9_TEACHER_BANK.npz",
        "camera_set": handoff / "QUALIFIED_CAMERA_SET.json",
        "mesh_policy": handoff / "MESH_QUALIFICATION_POLICY.json",
    }
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_HANDOFF_MISSING:" + ",".join(missing))

    local = Path("/content/realsas_knight_v9_seam_support_cf")
    repo = local / "source_repo"
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)

    log("SEAM_SUPPORT_COUNTERFACTUAL_SOURCE_CLONE_BEGIN", commit=COURT_COMMIT[:12])
    subprocess.run(
        ["git", "clone", "--quiet", "https://github.com/merynz/RealSaS-OPT.git", str(repo)],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", COURT_COMMIT], check=True)
    got = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if got != COURT_COMMIT:
        raise RuntimeError(f"SEAM_SUPPORT_COUNTERFACTUAL_SOURCE_DRIFT:{got}")
    log("SEAM_SUPPORT_COUNTERFACTUAL_SOURCE_PASS", commit=got[:12])

    req = repo / "requirements/mainline-ci.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", str(req)],
        check=True,
    )

    out_dir = result_dir / "SEAM_SUPPORT_LOCALITY_COUNTERFACTUAL_V1"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "20261004"
    env["CUDA_VISIBLE_DEVICES"] = ""

    script = repo / "tools/audit_knight_v9_seam_support_locality_counterfactual_v1.py"
    cmd = [
        sys.executable, "-u", str(script),
        "--surface-json", str(inputs["surface"]),
        "--partition-json", str(partition),
        "--candidate-json", str(candidate),
        "--fresh-skeleton-json", str(inputs["fresh_skeleton"]),
        "--teacher-bank", str(inputs["teacher_bank"]),
        "--camera-set-json", str(inputs["camera_set"]),
        "--mesh-policy-json", str(inputs["mesh_policy"]),
        "--out-dir", str(out_dir),
    ]
    log("SEAM_SUPPORT_COUNTERFACTUAL_BEGIN", output=out_dir)
    run_streaming(cmd, cwd=repo, env=env, log_path=out_dir / "CONSOLE.log")

    report_path = out_dir / "SEAM_SUPPORT_LOCALITY_COUNTERFACTUAL.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS__AUDIT_ONLY":
        raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_STATUS_DRIFT")
    seal = {
        "schema": "RealSaS.KnightV9SeamSupportLocalityCounterfactualNotebookSeal.v1",
        "status": "PASS",
        "court_commit": COURT_COMMIT,
        "mechanics_source_commit": MECHANICS_SOURCE_COMMIT,
        "final_candidate_sha256": sha256(candidate),
        "final_partition_sha256": sha256(partition),
        "report_sha256": sha256(report_path),
        "best_arm": report["best_arm"],
        "locality_hypothesis_supported": report["locality_hypothesis_supported"],
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    (out_dir / "COUNTERFACTUAL_SEAL.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        "SEAM_SUPPORT_COUNTERFACTUAL_PASS",
        best_arm=seal["best_arm"],
        locality_supported=seal["locality_hypothesis_supported"],
        report_sha=seal["report_sha256"][:12],
    )
    print("\n=== VERIFIED SEAM SUPPORT COUNTERFACTUAL ===", flush=True)
    for arm in report["arms"]:
        print(
            arm["arm"],
            "production=", arm["production"]["unsafe_face_count"],
            "all_face=", arm["all_face"]["unsafe_face_count"],
            "condition=", arm["production"]["condition_unsafe_count"],
            "introduced=", arm["paired_delta"]["production_introduced_count"],
            flush=True,
        )
    print("best_arm:", report["best_arm"], flush=True)
    print("locality_hypothesis_supported:", report["locality_hypothesis_supported"], flush=True)
    print("KNIGHT_V9_SEAM_SUPPORT_LOCALITY_COUNTERFACTUAL_VERIFIED_PASS", flush=True)


if __name__ == "__main__":
    main()
