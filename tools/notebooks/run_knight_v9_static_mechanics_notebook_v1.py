"""Colab/Drive orchestration for the Knight V9 static-mechanics notebook.

Run-All-safe, resume-safe and fail-closed. This file contains notebook
orchestration only; the actual mechanical operator authority remains in
tools/audit_v9_static_mechanical_microstep_v1.py at the exact source commit
sealed by the portable handoff manifest.
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
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


HANDOFF_SCHEMA = "RealSaS.V9StaticMechanicsNotebookHandoff.v1"
STEP_SCHEMA = "RealSaS.V9StaticMechanicsNotebookStepComplete.v1"
RUN_SCHEMA = "RealSaS.V9StaticMechanicsNotebookRunAll.v1"
LATEST_SCHEMA = "RealSaS.V9StaticMechanicsNotebookLatest.v1"

OPS = (
    {
        "name": "00_repartition_flip",
        "operator": "flip",
        "repartition_first": True,
        "repartition_iteration": 2,
    },
    {
        "name": "01_collapse",
        "operator": "collapse",
        "repartition_first": False,
        "repartition_iteration": None,
    },
    {
        "name": "02_vertex_cavity",
        "operator": "vertex_cavity",
        "repartition_first": False,
        "repartition_iteration": None,
    },
    {
        "name": "03_edge_cavity",
        "operator": "edge_cavity",
        "repartition_first": False,
        "repartition_iteration": None,
    },
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return str(value)

    def __call__(self, event: str, **fields) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        suffix = " ".join(f"{k}={self._fmt(v)}" for k, v in fields.items())
        line = f"[{ts}Z] {event}" + (f" {suffix}" if suffix else "")
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def verify_handoff(root: Path) -> dict:
    manifest_path = root / "HANDOFF_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"HANDOFF_MANIFEST_MISSING:{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != HANDOFF_SCHEMA:
        raise RuntimeError(f"HANDOFF_SCHEMA_DRIFT:{manifest.get('schema')}")
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"HANDOFF_STATUS_NOT_PASS:{manifest.get('status')}")
    if manifest.get("product_authority_claimed") is not False:
        raise RuntimeError("HANDOFF_PRODUCT_AUTHORITY_FORBIDDEN")
    for role, meta in dict(manifest.get("files") or {}).items():
        p = root / str(meta["path"])
        if not p.is_file():
            raise FileNotFoundError(f"HANDOFF_FILE_MISSING:{role}:{p}")
        if int(p.stat().st_size) != int(meta["bytes"]):
            raise RuntimeError(f"HANDOFF_SIZE_DRIFT:{role}")
        got = sha256(p)
        if got != str(meta["sha256"]):
            raise RuntimeError(f"HANDOFF_SHA_DRIFT:{role}:{got}!={meta['sha256']}")
    return manifest


def materialize_handoff(handoff: Path, bundle: Path, log: RunLog) -> dict:
    try:
        manifest = verify_handoff(handoff)
        log(
            "HANDOFF_DRIVE_VERIFY_PASS",
            source_commit=str(manifest["source_repo_commit"])[:12],
            files=len(manifest["files"]),
        )
        return manifest
    except Exception as exc:
        log("HANDOFF_DRIVE_MISS_OR_INVALID", error=repr(exc))

    if not bundle.is_file():
        raise RuntimeError(
            "PREFLIGHT_BLOCKER:HANDOFF_BUNDLE_MISSING:"
            f"{bundle}. Upload KNIGHT_V9_STATIC_MECHANICS_HANDOFF.zip to the HANDOFF folder."
        )

    local_stage = Path("/content/realsas_knight_v9_static_mechanics/handoff_stage")
    if local_stage.exists():
        shutil.rmtree(local_stage)
    local_stage.mkdir(parents=True)
    log("HANDOFF_BUNDLE_EXTRACT_BEGIN", bytes=bundle.stat().st_size, sha256=sha256(bundle))
    with zipfile.ZipFile(bundle, "r") as zf:
        zf.extractall(local_stage)

    candidates = [local_stage] + [p for p in local_stage.iterdir() if p.is_dir()]
    source = next((p for p in candidates if (p / "HANDOFF_MANIFEST.json").is_file()), None)
    if source is None:
        raise RuntimeError("PREFLIGHT_BLOCKER:BUNDLE_HANDOFF_MANIFEST_NOT_FOUND")
    manifest = verify_handoff(source)

    incoming = handoff.parent / ".HANDOFF_INCOMING"
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    for p in source.iterdir():
        dst = incoming / p.name
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)
    verify_handoff(incoming)

    # Preserve the bundle itself next to the materialized files for disaster recovery.
    bundle_copy = incoming / bundle.name
    if bundle.resolve() != bundle_copy.resolve():
        shutil.copy2(bundle, bundle_copy)

    if handoff.exists():
        shutil.rmtree(handoff)
    os.replace(incoming, handoff)
    manifest = verify_handoff(handoff)
    log(
        "HANDOFF_MATERIALIZE_PASS",
        source_commit=str(manifest["source_repo_commit"])[:12],
        files=len(manifest["files"]),
    )
    return manifest


def checkout_exact_source(repo: Path, commit: str, log: RunLog) -> None:
    if repo.exists():
        shutil.rmtree(repo)
    log("SOURCE_CLONE_BEGIN", commit=commit[:12])
    subprocess.run(
        ["git", "clone", "--quiet", "https://github.com/merynz/RealSaS-OPT.git", str(repo)],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", commit], check=True)
    got = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if got != commit:
        raise RuntimeError(f"SOURCE_COMMIT_DRIFT:{got}!={commit}")
    log("SOURCE_CHECKOUT_PASS", commit=got[:12])


def install_runtime(repo: Path, log: RunLog) -> None:
    req = repo / "requirements/mainline-ci.txt"
    if not req.is_file():
        raise FileNotFoundError(f"REQUIREMENTS_MISSING:{req}")
    log("DEPENDENCY_INSTALL_BEGIN", requirements=req)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "-r",
            str(req),
        ],
        check=True,
    )
    audit = repo / "tools/audit_v9_static_mechanical_microstep_v1.py"
    subprocess.run([sys.executable, "-m", "py_compile", str(audit)], check=True)
    log("DEPENDENCY_INSTALL_PASS", audit_script=audit.name)


def preflight(
    *,
    manifest: dict,
    handoff: Path,
    results: Path,
    source_commit: str,
    log: RunLog,
) -> dict:
    required = {
        "surface",
        "partition",
        "candidate",
        "fresh_skeleton",
        "teacher_bank",
        "camera_set",
        "mesh_policy",
        "explicit_faces",
    }
    files = dict(manifest.get("files") or {})
    missing = sorted(required - set(files))
    if missing:
        raise RuntimeError(f"PREFLIGHT_BLOCKER:MISSING_HANDOFF_ROLES:{missing}")

    paths = {role: handoff / str(meta["path"]) for role, meta in files.items()}
    surface_doc = json.loads(paths["surface"].read_text(encoding="utf-8"))
    partition_doc = json.loads(paths["partition"].read_text(encoding="utf-8"))
    candidate_doc = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    faces_doc = json.loads(paths["explicit_faces"].read_text(encoding="utf-8"))

    surface_lineage = surface_doc.get("geometry_lineage_hash")
    partition_lineage = partition_doc.get("partition_lineage_hash")
    candidate_lineage = candidate_doc.get("candidate_lineage_hash")

    if surface_lineage != manifest.get("surface_lineage_hash"):
        raise RuntimeError("PREFLIGHT_BLOCKER:SURFACE_LINEAGE_DRIFT")
    if partition_lineage != manifest.get("partition_lineage_hash"):
        raise RuntimeError("PREFLIGHT_BLOCKER:PARTITION_LINEAGE_DRIFT")
    if candidate_lineage != manifest.get("candidate_lineage_hash"):
        raise RuntimeError("PREFLIGHT_BLOCKER:CANDIDATE_LINEAGE_DRIFT")
    if faces_doc.get("surface_lineage_hash") != surface_lineage:
        raise RuntimeError("PREFLIGHT_BLOCKER:FACE_LINEAGE_DRIFT")
    if int(faces_doc.get("face_count") or 0) <= 0:
        raise RuntimeError("PREFLIGHT_BLOCKER:FACE_PROVENANCE_EMPTY")
    if int(faces_doc["face_count"]) != len(faces_doc.get("faces") or ()):
        raise RuntimeError("PREFLIGHT_BLOCKER:FACE_PROVENANCE_COUNT_DRIFT")

    free_gib = shutil.disk_usage("/content").free / (1024 ** 3)
    if free_gib < 8.0:
        raise RuntimeError(f"PREFLIGHT_BLOCKER:LOCAL_DISK_LOW:{free_gib:.2f}GiB")

    preflight_doc = {
        "schema": "RealSaS.V9StaticMechanicsNotebookPreflight.v1",
        "status": "PASS",
        "source_commit": source_commit,
        "surface_lineage_hash": surface_lineage,
        "partition_lineage_hash": partition_lineage,
        "candidate_lineage_hash": candidate_lineage,
        "explicit_face_count": int(faces_doc["face_count"]),
        "local_free_gib": float(free_gib),
        "gpu_required": False,
        "product_authority_claimed": False,
    }
    atomic_write_json(results / "LATEST_PREFLIGHT.json", preflight_doc)
    log(
        "PREFLIGHT_PASS",
        free_gib=round(free_gib, 2),
        faces=faces_doc["face_count"],
        candidate=str(candidate_lineage)[:12],
        partition=str(partition_lineage)[:12],
    )
    return {"paths": paths, "preflight": preflight_doc}


def verified_checkpoint(
    step_dir: Path,
    source_candidate: Path,
    source_partition: Path,
    source_commit: str,
):
    seal_path = step_dir / "STEP_COMPLETE.json"
    if not seal_path.is_file():
        return None
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal.get("status") != "PASS" or seal.get("schema") != STEP_SCHEMA:
            return None
        if seal.get("source_commit") != source_commit:
            return None
        if seal.get("input_candidate_sha256") != sha256(source_candidate):
            return None
        if seal.get("input_partition_sha256") != sha256(source_partition):
            return None
        final_candidate = step_dir / "FINAL_CANDIDATE.json"
        output_partition = step_dir / "INPUT_PARTITION.json"
        report = step_dir / "REPORT.json"
        if not (final_candidate.is_file() and output_partition.is_file() and report.is_file()):
            return None
        if seal.get("final_candidate_sha256") != sha256(final_candidate):
            return None
        if seal.get("output_partition_sha256") != sha256(output_partition):
            return None
        if seal.get("report_sha256") != sha256(report):
            return None
        return seal
    except Exception:
        return None


def publish_checkpoint(local_dir: Path, checkpoints: Path, name: str) -> Path:
    drive_dir = checkpoints / name
    incoming = checkpoints / f".{name}.incoming"
    if incoming.exists():
        shutil.rmtree(incoming)
    shutil.copytree(local_dir, incoming)
    seal = json.loads((incoming / "STEP_COMPLETE.json").read_text(encoding="utf-8"))
    if seal["final_candidate_sha256"] != sha256(incoming / "FINAL_CANDIDATE.json"):
        raise RuntimeError("CHECKPOINT_FINAL_CANDIDATE_SHA_DRIFT")
    if seal["output_partition_sha256"] != sha256(incoming / "INPUT_PARTITION.json"):
        raise RuntimeError("CHECKPOINT_PARTITION_SHA_DRIFT")
    if seal["report_sha256"] != sha256(incoming / "REPORT.json"):
        raise RuntimeError("CHECKPOINT_REPORT_SHA_DRIFT")
    if drive_dir.exists():
        shutil.rmtree(drive_dir)
    os.replace(incoming, drive_dir)
    return drive_dir


def run_streaming(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict,
    log_path: Path,
    master_log: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as lf:
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
            lf.write(line)
            with master_log.open("a", encoding="utf-8") as mf:
                mf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"MICROSTEP_CHILD_FAILED:{rc}:{log_path}")


def run_chain(
    *,
    repo: Path,
    handoff: Path,
    checkpoints: Path,
    results: Path,
    local_steps: Path,
    paths: dict,
    source_commit: str,
    run_stamp: str,
    master_log: Path,
    log: RunLog,
    reset_checkpoints: bool,
) -> tuple[Path, Path, list[dict]]:
    audit_script = repo / "tools/audit_v9_static_mechanical_microstep_v1.py"
    if not audit_script.is_file():
        raise FileNotFoundError(audit_script)

    if reset_checkpoints and checkpoints.exists():
        for spec in OPS:
            p = checkpoints / spec["name"]
            if p.exists():
                shutil.rmtree(p)
        log("CHECKPOINT_RESET_COMPLETE")

    current_candidate = paths["candidate"]
    current_partition = paths["partition"]
    summaries: list[dict] = []

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "20261004"
    env["CUDA_VISIBLE_DEVICES"] = ""

    for spec in OPS:
        name = str(spec["name"])
        drive_step = checkpoints / name
        existing = verified_checkpoint(
            drive_step, current_candidate, current_partition, source_commit
        )
        if existing is not None:
            report = json.loads((drive_step / "REPORT.json").read_text(encoding="utf-8"))
            log(
                "CHECKPOINT_RESUME_PASS",
                step=name,
                accepted=report["decision"]["accepted"],
                static=(
                    f"{report['baseline']['static_violations']}->"
                    f"{report['final']['static_violations']}"
                ),
                g3b=(
                    f"{report['baseline']['g3b_unsafe']}->"
                    f"{report['final']['g3b_unsafe']}"
                ),
            )
            current_candidate = drive_step / "FINAL_CANDIDATE.json"
            current_partition = drive_step / "INPUT_PARTITION.json"
            summaries.append(
                {
                    "step": name,
                    "operator": spec["operator"],
                    "resumed": True,
                    **report["decision"],
                    "baseline_static": report["baseline"]["static_violations"],
                    "final_static": report["final"]["static_violations"],
                    "baseline_g3b": report["baseline"]["g3b_unsafe"],
                    "final_g3b": report["final"]["g3b_unsafe"],
                    "seconds": report["timing_seconds"]["total"],
                }
            )
            continue

        local_step = local_steps / name
        if local_step.exists():
            shutil.rmtree(local_step)
        local_step.mkdir(parents=True)
        child_log = local_step / "CONSOLE.log"

        cmd = [
            sys.executable,
            "-u",
            str(audit_script),
            "--surface-json",
            str(paths["surface"]),
            "--partition-json",
            str(current_partition),
            "--candidate-json",
            str(current_candidate),
            "--fresh-skeleton-json",
            str(paths["fresh_skeleton"]),
            "--teacher-bank",
            str(paths["teacher_bank"]),
            "--camera-set-json",
            str(paths["camera_set"]),
            "--mesh-policy-json",
            str(paths["mesh_policy"]),
            "--explicit-faces-json",
            str(paths["explicit_faces"]),
            "--operator",
            str(spec["operator"]),
            "--cycle-tag",
            name.upper(),
            "--out-dir",
            str(local_step),
        ]
        if bool(spec["repartition_first"]):
            cmd += [
                "--repartition-first",
                "--repartition-iteration",
                str(spec["repartition_iteration"]),
            ]

        log(
            "MICROSTEP_BEGIN",
            step=name,
            operator=spec["operator"],
            repartition=spec["repartition_first"],
            input_candidate_sha=sha256(current_candidate)[:12],
            input_partition_sha=sha256(current_partition)[:12],
        )
        started = time.monotonic()
        try:
            run_streaming(
                cmd,
                cwd=repo,
                env=env,
                log_path=child_log,
                master_log=master_log,
            )
        except Exception as exc:
            fail_dir = results / f"FAILED_{run_stamp}_{name}"
            if fail_dir.exists():
                shutil.rmtree(fail_dir)
            shutil.copytree(local_step, fail_dir)
            log("MICROSTEP_FAIL", step=name, error=repr(exc), partial=fail_dir)
            raise

        report_path = local_step / "REPORT.json"
        final_candidate = local_step / "FINAL_CANDIDATE.json"
        output_partition = local_step / "INPUT_PARTITION.json"
        for p in (report_path, final_candidate, output_partition):
            if not p.is_file():
                raise RuntimeError(f"MICROSTEP_OUTPUT_MISSING:{name}:{p.name}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE__AUDIT_ONLY":
            raise RuntimeError(f"MICROSTEP_STATUS_DRIFT:{name}:{report.get('status')}")
        if report.get("product_authority_minted") is not False:
            raise RuntimeError(f"MICROSTEP_PRODUCT_AUTHORITY_FORBIDDEN:{name}")
        if report.get("portable_handoff_mode") is not True:
            raise RuntimeError(f"MICROSTEP_NOT_PORTABLE_MODE:{name}")

        seal = {
            "schema": STEP_SCHEMA,
            "status": "PASS",
            "step": name,
            "operator": spec["operator"],
            "source_commit": source_commit,
            "input_candidate_sha256": sha256(current_candidate),
            "input_partition_sha256": sha256(current_partition),
            "final_candidate_sha256": sha256(final_candidate),
            "output_partition_sha256": sha256(output_partition),
            "report_sha256": sha256(report_path),
            "accepted": bool(report["decision"]["accepted"]),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "product_authority_claimed": False,
        }
        atomic_write_json(local_step / "STEP_COMPLETE.json", seal)
        drive_step = publish_checkpoint(local_step, checkpoints, name)

        elapsed = time.monotonic() - started
        log(
            "MICROSTEP_PASS",
            step=name,
            accepted=report["decision"]["accepted"],
            static=(
                f"{report['baseline']['static_violations']}->"
                f"{report['final']['static_violations']}"
            ),
            g3b=(
                f"{report['baseline']['g3b_unsafe']}->"
                f"{report['final']['g3b_unsafe']}"
            ),
            all_g3b=(
                f"{report['baseline']['all_face_g3b_unsafe']}->"
                f"{report['final']['all_face_g3b_unsafe']}"
            ),
            child_seconds=round(float(report["timing_seconds"]["total"]), 2),
            wall_seconds=round(elapsed, 2),
        )

        current_candidate = drive_step / "FINAL_CANDIDATE.json"
        current_partition = drive_step / "INPUT_PARTITION.json"
        summaries.append(
            {
                "step": name,
                "operator": spec["operator"],
                "resumed": False,
                **report["decision"],
                "baseline_static": report["baseline"]["static_violations"],
                "final_static": report["final"]["static_violations"],
                "baseline_g3b": report["baseline"]["g3b_unsafe"],
                "final_g3b": report["final"]["g3b_unsafe"],
                "seconds": report["timing_seconds"]["total"],
            }
        )

    log("OPERATOR_CHAIN_COMPLETE", steps=len(summaries))
    return current_candidate, current_partition, summaries


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
    ap.add_argument("--reset-checkpoints", action="store_true")
    a = ap.parse_args()

    base = a.drive_base.resolve()
    handoff = base / "HANDOFF"
    checkpoints = base / "CHECKPOINTS"
    logs = base / "LOGS"
    results = base / "RESULTS"
    bundle = handoff / "KNIGHT_V9_STATIC_MECHANICS_HANDOFF.zip"

    local_root = Path("/content/realsas_knight_v9_static_mechanics")
    repo = local_root / "source_repo"
    local_steps = local_root / "steps"

    for p in (base, handoff, checkpoints, logs, results, local_steps):
        p.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    master_log = logs / f"RUN_{run_stamp}.log"
    log = RunLog(master_log)
    log(
        "NOTEBOOK_RUNNER_BOOT",
        python=sys.version.split()[0],
        platform=platform.platform(),
        drive_base=base,
    )

    manifest = materialize_handoff(handoff, bundle, log)
    source_commit = str(manifest.get("source_repo_commit") or "")
    if not source_commit or source_commit == "UNKNOWN":
        raise RuntimeError("PREFLIGHT_BLOCKER:HANDOFF_SOURCE_COMMIT_UNKNOWN")

    checkout_exact_source(repo, source_commit, log)
    install_runtime(repo, log)
    pf = preflight(
        manifest=manifest,
        handoff=handoff,
        results=results,
        source_commit=source_commit,
        log=log,
    )
    paths = pf["paths"]

    current_candidate, current_partition, summaries = run_chain(
        repo=repo,
        handoff=handoff,
        checkpoints=checkpoints,
        results=results,
        local_steps=local_steps,
        paths=paths,
        source_commit=source_commit,
        run_stamp=run_stamp,
        master_log=master_log,
        log=log,
        reset_checkpoints=bool(a.reset_checkpoints),
    )

    run_result = results / f"RUN_{run_stamp}"
    if run_result.exists():
        shutil.rmtree(run_result)
    run_result.mkdir(parents=True)
    shutil.copy2(current_candidate, run_result / "FINAL_CANDIDATE.json")
    shutil.copy2(current_partition, run_result / "FINAL_PARTITION.json")

    summary = {
        "schema": RUN_SCHEMA,
        "status": "PASS",
        "run_stamp": run_stamp,
        "source_commit": source_commit,
        "handoff_manifest_sha256": sha256(handoff / "HANDOFF_MANIFEST.json"),
        "steps": summaries,
        "final_candidate_sha256": sha256(run_result / "FINAL_CANDIDATE.json"),
        "final_partition_sha256": sha256(run_result / "FINAL_PARTITION.json"),
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    atomic_write_json(run_result / "RUN_ALL_SUMMARY.json", summary)
    latest = {
        "schema": LATEST_SCHEMA,
        "status": "PASS",
        "run_stamp": run_stamp,
        "result_dir": str(run_result),
        "summary_sha256": sha256(run_result / "RUN_ALL_SUMMARY.json"),
    }
    atomic_write_json(results / "LATEST_RUN.json", latest)

    print("\n=== KNIGHT V9 STATIC MECHANICS RUN-ALL SUMMARY ===", flush=True)
    for row in summaries:
        print(
            "STEP_SUMMARY "
            + " ".join(
                [
                    f"step={row['step']}",
                    f"operator={row['operator']}",
                    f"resumed={row['resumed']}",
                    f"accepted={row['accepted']}",
                    f"static={row['baseline_static']}->{row['final_static']}",
                    f"g3b={row['baseline_g3b']}->{row['final_g3b']}",
                    f"seconds={float(row['seconds']):.2f}",
                ]
            ),
            flush=True,
        )

    log(
        "RUN_ALL_PASS",
        final_candidate_sha=summary["final_candidate_sha256"][:12],
        final_partition_sha=summary["final_partition_sha256"][:12],
        result_dir=run_result,
    )
    print("KNIGHT_V9_STATIC_MECHANICS_NOTEBOOK_RUN_ALL_PASS", flush=True)


if __name__ == "__main__":
    main()
