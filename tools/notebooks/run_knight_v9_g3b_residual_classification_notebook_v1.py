"""Colab/Drive orchestration for Knight V9 residual G3B classification.

Diagnostic-only companion to the completed static-mechanics notebook.
It does NOT rerun repair operators. It hash-verifies the completed Drive result,
checks out the pinned residual-classifier source, reruns the existing mechanical
compatibility proof, and classifies the residual unsafe faces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASSIFIER_SOURCE_COMMIT = "7b0c1d14a9159dcbe97ced9d7a4ad1f32fbd0ac5"
MECHANICS_SOURCE_COMMIT = "06f3453f754db73ed4230ff4f295b7a58ba931ea"
EXPECTED_FINAL_CANDIDATE_SHA256 = "a49170c38167291e629d9c05d91f70193c1547e44a8e0da4e1b649a2f6b36baa"
EXPECTED_FINAL_PARTITION_SHA256 = "b2503c49d3210e48e85f07225bfbcb234350c18674e4a191dac53d26d25be7ed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **fields) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        suffix = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"[{ts}Z] {event}" + (f" {suffix}" if suffix else "")
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def checkout(repo: Path, commit: str, log: RunLog) -> None:
    if repo.exists():
        shutil.rmtree(repo)
    log("CLASSIFIER_SOURCE_CLONE_BEGIN", commit=commit[:12])
    subprocess.run(
        ["git", "clone", "--quiet", "https://github.com/merynz/RealSaS-OPT.git", str(repo)],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", commit], check=True)
    got = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if got != commit:
        raise RuntimeError(f"CLASSIFIER_SOURCE_COMMIT_DRIFT:{got}!={commit}")
    log("CLASSIFIER_SOURCE_CHECKOUT_PASS", commit=got[:12])


def install_runtime(repo: Path, log: RunLog) -> None:
    req = repo / "requirements/mainline-ci.txt"
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", "-r", str(req),
        ],
        check=True,
    )
    script = repo / "tools/audit_v9_g3b_residual_classification_v1.py"
    subprocess.run([sys.executable, "-m", "py_compile", str(script)], check=True)
    log("CLASSIFIER_DEPENDENCY_PASS", script=script.name)


def verify_completed_run(base: Path, log: RunLog) -> tuple[dict, Path, Path]:
    latest_path = base / "RESULTS/LATEST_RUN.json"
    if not latest_path.is_file():
        raise RuntimeError(f"G3B_CLASSIFIER_LATEST_RUN_MISSING:{latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("status") != "PASS":
        raise RuntimeError(f"G3B_CLASSIFIER_LATEST_RUN_NOT_PASS:{latest.get('status')}")
    result_dir = Path(str(latest["result_dir"]))
    summary_path = result_dir / "RUN_ALL_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("G3B_CLASSIFIER_SUMMARY_NOT_PASS")
    if summary.get("source_commit") != MECHANICS_SOURCE_COMMIT:
        raise RuntimeError(
            f"G3B_CLASSIFIER_MECHANICS_SOURCE_DRIFT:{summary.get('source_commit')}"
        )
    if sha256(summary_path) != latest.get("summary_sha256"):
        raise RuntimeError("G3B_CLASSIFIER_SUMMARY_SHA_DRIFT")

    candidate = result_dir / "FINAL_CANDIDATE.json"
    partition = result_dir / "FINAL_PARTITION.json"
    if sha256(candidate) != EXPECTED_FINAL_CANDIDATE_SHA256:
        raise RuntimeError("G3B_CLASSIFIER_FINAL_CANDIDATE_SHA_DRIFT")
    if sha256(partition) != EXPECTED_FINAL_PARTITION_SHA256:
        raise RuntimeError("G3B_CLASSIFIER_FINAL_PARTITION_SHA_DRIFT")
    if summary.get("final_candidate_sha256") != EXPECTED_FINAL_CANDIDATE_SHA256:
        raise RuntimeError("G3B_CLASSIFIER_SUMMARY_CANDIDATE_SHA_DRIFT")
    if summary.get("final_partition_sha256") != EXPECTED_FINAL_PARTITION_SHA256:
        raise RuntimeError("G3B_CLASSIFIER_SUMMARY_PARTITION_SHA_DRIFT")

    log(
        "COMPLETED_MECHANICS_RESULT_VERIFY_PASS",
        run_stamp=summary["run_stamp"],
        candidate=EXPECTED_FINAL_CANDIDATE_SHA256[:12],
        partition=EXPECTED_FINAL_PARTITION_SHA256[:12],
    )
    return summary, candidate, partition


def verify_handoff(base: Path, log: RunLog) -> dict[str, Path]:
    handoff = base / "HANDOFF"
    required = {
        "surface": "REFINED_SURFACE.json",
        "fresh_skeleton": "FRESH_QUALIFIED_SKELETON.json",
        "teacher_bank": "V9_TEACHER_BANK.npz",
        "camera_set": "QUALIFIED_CAMERA_SET.json",
        "mesh_policy": "MESH_QUALIFICATION_POLICY.json",
    }
    out = {}
    for role, name in required.items():
        p = handoff / name
        if not p.is_file():
            raise RuntimeError(f"G3B_CLASSIFIER_HANDOFF_FILE_MISSING:{role}:{p}")
        out[role] = p
    log("G3B_CLASSIFIER_HANDOFF_PASS", roles=len(out))
    return out


def run_streaming(cmd: list[str], *, cwd: Path, env: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as fh:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"G3B_CLASSIFIER_CHILD_FAILED:{rc}")


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
    logs = base / "LOGS"
    logs.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = RunLog(logs / f"G3B_CLASSIFIER_{run_stamp}.log")
    log(
        "G3B_CLASSIFIER_BOOT",
        python=sys.version.split()[0],
        platform=platform.platform(),
        drive_base=base,
        gpu_required=False,
    )

    summary, candidate, partition = verify_completed_run(base, log)
    handoff = verify_handoff(base, log)

    local_root = Path("/content/realsas_knight_v9_g3b_classifier")
    repo = local_root / "source_repo"
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True)

    checkout(repo, CLASSIFIER_SOURCE_COMMIT, log)
    install_runtime(repo, log)

    out_dir = Path(summary["run_stamp"])
    out_dir = base / "RESULTS" / f"RUN_{out_dir.name}" / "G3B_RESIDUAL_CLASSIFICATION_V1"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "20261004"
    env["CUDA_VISIBLE_DEVICES"] = ""

    script = repo / "tools/audit_v9_g3b_residual_classification_v1.py"
    cmd = [
        sys.executable, "-u", str(script),
        "--surface-json", str(handoff["surface"]),
        "--partition-json", str(partition),
        "--candidate-json", str(candidate),
        "--fresh-skeleton-json", str(handoff["fresh_skeleton"]),
        "--teacher-bank", str(handoff["teacher_bank"]),
        "--camera-set-json", str(handoff["camera_set"]),
        "--mesh-policy-json", str(handoff["mesh_policy"]),
        "--out-dir", str(out_dir),
    ]
    log("G3B_CLASSIFIER_BEGIN", output=out_dir)
    run_streaming(
        cmd, cwd=repo, env=env,
        log_path=out_dir / "CONSOLE.log",
    )

    result_path = out_dir / "G3B_RESIDUAL_CLASSIFICATION.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS__DIAGNOSTIC_ONLY":
        raise RuntimeError(f"G3B_CLASSIFIER_STATUS_DRIFT:{result.get('status')}")
    if int(result["summary"]["unsafe_face_count"]) != 48:
        raise RuntimeError(
            f"G3B_CLASSIFIER_EXPECTED_48_DRIFT:{result['summary']['unsafe_face_count']}"
        )

    seal = {
        "schema": "RealSaS.V9G3BResidualClassificationNotebookSeal.v1",
        "status": "PASS",
        "mechanics_run_stamp": summary["run_stamp"],
        "mechanics_source_commit": MECHANICS_SOURCE_COMMIT,
        "classifier_source_commit": CLASSIFIER_SOURCE_COMMIT,
        "final_candidate_sha256": sha256(candidate),
        "final_partition_sha256": sha256(partition),
        "classification_sha256": sha256(result_path),
        "unsafe_face_count": int(result["summary"]["unsafe_face_count"]),
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    (out_dir / "CLASSIFICATION_SEAL.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        "G3B_CLASSIFIER_PASS",
        unsafe=seal["unsafe_face_count"],
        classification_sha=seal["classification_sha256"][:12],
        output=out_dir,
    )
    print("\n=== G3B RESIDUAL CLASSIFICATION SUMMARY ===", flush=True)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print("KNIGHT_V9_G3B_RESIDUAL_CLASSIFICATION_VERIFIED_PASS", flush=True)


if __name__ == "__main__":
    main()
