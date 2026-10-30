#!/usr/bin/env python3
"""Rescue the ten Full139 V2-extension S7 face-prefix failures without rerunning 129 PASS subjects.

This is an execution/provenance repair, not a relaxed court.  It binds the exact
129/139 failed receipt, archives each failed V2 peel, rebuilds only the ten failed
subjects with the guarded V3 K-hit producer + unchanged CPU oracle court, resumes
the existing component/affinity/relations runner, and writes an explicit mixed-
provenance 139/139 receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256
from tools.corpus.build_truth139_graphics_layer_peel_v3_rescue import PRODUCER_REVISION
from tools.corpus.validate_truth139_graphics_layer_peel_v3_rescue import PASS_STATUS as LAYER_COURT_PASS

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_PASS = "PASS_TRUTH139_V2_EXTENSION_CHAIN_EXECUTED__DERIVED_TRUTH_ADMISSION_SEPARATE"
BASE_CHAIN_PASS = "PASS_TRUTH139_GRAPHICS_MATERIALIZATION_CHAIN_EXECUTED__DERIVED_TRUTH_ADMISSION_SEPARATE"
FINAL_PASS = "PASS_FULL139_V2_EXTENSION_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
RESCUE_PASS = "PASS_TRUTH139_V2_EXTENSION_S7_RESCUE_10_OF_10__DERIVED_TRUTH_ADMISSION_SEPARATE"
LAYER_MANIFEST_PASS = "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE"

EXPECTED_FAILED_SUBJECTS = (
    "truth139_102df943a583a9034098",
    "truth139_103181009578e7260c12",
    "truth139_17c60860aa99241729d0",
    "truth139_2ed75b9eadf5c08ec4e0",
    "truth139_6a02c059fc3cf1dbc1c5",
    "truth139_8610a7fcff63d4ec28e0",
    "truth139_8833f5d8becc7a8739ea",
    "truth139_8f8d617e4f7cad16b03e",
    "truth139_f3b99fc675b1c99a2ee0",
    "truth139_fd7c6cc1c9a2b9f12f37",
)

WORKER_PATHS = (
    "tools/corpus/build_truth139_graphics_layer_peel_v2.py",
    "tools/corpus/validate_truth139_graphics_layer_peel_v2.py",
    "tools/corpus/build_truth139_graphics_layer_peel_v3_rescue.py",
    "tools/corpus/validate_truth139_graphics_layer_peel_v3_rescue.py",
    "tools/corpus/build_truth139_source_component_masks_v1.py",
    "tools/corpus/validate_truth139_source_component_masks_v1.py",
    "tools/corpus/build_truth139_drawable_affinity_teacher_v1.py",
    "tools/corpus/validate_truth139_drawable_affinity_teacher_v1.py",
    "tools/corpus/build_truth139_relations_teacher_v2.py",
    "tools/corpus/validate_truth139_relations_teacher_v2.py",
    "tools/corpus/run_truth139_v2_extension_subject_v1.py",
    "models/iris/truth/graphics_oracle_v1.py",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path: Path, code: str) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip().lower()


def _run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True,
    )


def _valid_base_subject(root: Path, subject_id: str) -> bool:
    path = root / subject_id / "GRAPHICS_MATERIALIZATION_CHAIN.json"
    try:
        value = _json(path, "RESCUE_BASE_CHAIN")
        return value.get("status") == BASE_CHAIN_PASS and value.get("subject_id") == subject_id
    except Exception:
        return False


def _resolve_base_subject(parent_subjects: Path, rescue_subjects: Path, subject_id: str) -> Path:
    # Prefer rescue authority when both exist; its chain is the corrected parent for Rescue-19 subjects.
    for root in (rescue_subjects, parent_subjects):
        if _valid_base_subject(root, subject_id):
            return root / subject_id
    raise RuntimeError("RESCUE_BASE_SUBJECT_UNBOUND:" + subject_id)


def _chain_complete(path: Path, subject_id: str) -> bool:
    try:
        value = _json(path, "RESCUE_EXTENSION_CHAIN")
        return value.get("status") == CHAIN_PASS and value.get("subject_id") == subject_id
    except Exception:
        return False


def _layer_pair_is_v3_pass(layer_dir: Path, subject_id: str) -> bool:
    try:
        manifest = _json(layer_dir / "LAYER_PEEL_TEACHER_MANIFEST.json", "RESCUE_LAYER_MANIFEST")
        court = _json(layer_dir / "LAYER_PEEL_COURT.json", "RESCUE_LAYER_COURT")
        return (
            manifest.get("status") == LAYER_MANIFEST_PASS
            and manifest.get("subject_id") == subject_id
            and manifest.get("producer_revision") == PRODUCER_REVISION
            and court.get("status") == LAYER_COURT_PASS
            and court.get("subject_id") == subject_id
            and court.get("producer_revision") == PRODUCER_REVISION
            and court.get("layer_peel_teacher_identity_sha256") == manifest.get("layer_peel_teacher_identity_sha256")
        )
    except Exception:
        return False


def _archive_failed_v2_layer(layer_dir: Path) -> None:
    if not layer_dir.exists():
        return
    archive = layer_dir.parent / "layer_peel_v2_failed_s7"
    if archive.exists():
        shutil.rmtree(layer_dir)
    else:
        layer_dir.rename(archive)


def _worker_manifest(code_sha: str) -> dict:
    rows = []
    for rel in WORKER_PATHS:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError("RESCUE_WORKER_FILE_MISSING:" + rel)
        rows.append({"path": rel, "sha256": sha256(path)})
    payload = {
        "schema": "RealSaS.Truth139V2ExtensionS7RescueWorkerCodeManifest.v1",
        "git_source_commit_sha": code_sha,
        "producer_revision": PRODUCER_REVISION,
        "cpu_court_tolerance_relaxed": False,
        "files": rows,
    }
    payload["identity_sha256"] = _digest(payload)
    return payload


def rescue(
    *,
    batch_root: Path,
    parent_subjects: Path,
    rescue_subjects: Path,
    expected_parent_receipt_sha256: str,
    expected_parent_receipt_identity_sha256: str,
    device: str,
) -> dict:
    batch_root = batch_root.resolve()
    subject_root = batch_root / "subjects"
    receipt_path = batch_root / "FULL139_V2_EXTENSION_RECEIPT.json"
    pre_receipt_path = batch_root / "FULL139_V2_EXTENSION_RECEIPT_PRE_RESCUE_129OF139.json"

    if not pre_receipt_path.is_file():
        if not receipt_path.is_file():
            raise FileNotFoundError("RESCUE_PARENT_RECEIPT_MISSING:" + str(receipt_path))
        if sha256(receipt_path) != expected_parent_receipt_sha256:
            raise ValueError("RESCUE_PARENT_RECEIPT_SHA_MISMATCH")
        shutil.copy2(receipt_path, pre_receipt_path)
    if sha256(pre_receipt_path) != expected_parent_receipt_sha256:
        raise ValueError("RESCUE_PARENT_BACKUP_SHA_MISMATCH")

    parent_receipt = _json(pre_receipt_path, "RESCUE_PARENT_RECEIPT")
    if parent_receipt.get("receipt_identity_sha256") != expected_parent_receipt_identity_sha256:
        raise ValueError("RESCUE_PARENT_RECEIPT_IDENTITY_MISMATCH")
    if int(parent_receipt.get("subject_count", -1)) != 139 or int(parent_receipt.get("completed_subject_count", -1)) != 129:
        raise ValueError("RESCUE_PARENT_RECEIPT_COUNTS")
    failed_parent = tuple(sorted((parent_receipt.get("failed_jobs_by_subject") or {}).keys()))
    if failed_parent != tuple(sorted(EXPECTED_FAILED_SUBJECTS)):
        raise ValueError("RESCUE_PARENT_FAILED_SUBJECT_SET_DRIFT")
    subject_ids = tuple(parent_receipt.get("subject_ids") or ())
    if len(subject_ids) != 139 or len(set(subject_ids)) != 139:
        raise ValueError("RESCUE_PARENT_SUBJECT_SET")

    code_sha = _git_sha()
    worker_manifest = _worker_manifest(code_sha)
    worker_path = batch_root / "FULL139_V2_EXTENSION_S7_RESCUE_WORKER_CODE_MANIFEST.json"
    _atomic_json(worker_path, worker_manifest)

    rescued = []
    failures: dict[str, str] = {}
    for index, subject_id in enumerate(EXPECTED_FAILED_SUBJECTS, start=1):
        print(f"\n[S7-RESCUE {index:02d}/10] {subject_id}", flush=True)
        out = subject_root / subject_id
        chain_path = out / "V2_EXTENSION_CHAIN.json"
        if _chain_complete(chain_path, subject_id):
            chain = _json(chain_path, "RESCUE_EXISTING_CHAIN")
            if ((chain.get("bindings") or {}).get("code_commit_sha")) == code_sha:
                print("[REUSE-S7-RESCUE]", subject_id, flush=True)
                rescued.append(subject_id)
                continue

        try:
            base_subject = _resolve_base_subject(parent_subjects, rescue_subjects, subject_id)
            dense_manifest = base_subject / "dense_graphics/DENSE_GRAPHICS_GBUFFER_MANIFEST.json"
            dense_court = base_subject / "dense_graphics/DENSE_GRAPHICS_GEOMETRY_COURT.json"
            layer_dir = out / "layer_peel"
            if not _layer_pair_is_v3_pass(layer_dir, subject_id):
                _archive_failed_v2_layer(layer_dir)
                layer_dir.mkdir(parents=True, exist_ok=True)
                _run([
                    sys.executable,
                    str(REPO_ROOT / "tools/corpus/build_truth139_graphics_layer_peel_v3_rescue.py"),
                    "--dense-graphics-manifest", str(dense_manifest),
                    "--dense-graphics-court", str(dense_court),
                    "--output-dir", str(layer_dir),
                    "--device", device,
                ])
                _run([
                    sys.executable,
                    str(REPO_ROOT / "tools/corpus/validate_truth139_graphics_layer_peel_v3_rescue.py"),
                    "--layer-peel-teacher-manifest", str(layer_dir / "LAYER_PEEL_TEACHER_MANIFEST.json"),
                    "--output", str(layer_dir / "LAYER_PEEL_COURT.json"),
                ])
            if not _layer_pair_is_v3_pass(layer_dir, subject_id):
                raise RuntimeError("RESCUE_V3_LAYER_PAIR_DID_NOT_PASS")

            # Any downstream partial bytes would have been derived from the failed V2 layer.
            for name in ("source_components", "drawable_affinity", "relations"):
                path = out / name
                if path.exists():
                    shutil.rmtree(path)
            if chain_path.exists():
                chain_path.unlink()

            _run([
                sys.executable,
                str(REPO_ROOT / "tools/corpus/run_truth139_v2_extension_subject_v1.py"),
                "--subject-id", subject_id,
                "--base-subject-root", str(base_subject),
                "--output-root", str(out),
                "--device", device,
            ])
            if not _chain_complete(chain_path, subject_id):
                raise RuntimeError("RESCUE_EXTENSION_CHAIN_DID_NOT_CLOSE")
            chain = _json(chain_path, "RESCUE_FINAL_CHAIN")
            if ((chain.get("bindings") or {}).get("code_commit_sha")) != code_sha:
                raise RuntimeError("RESCUE_EXTENSION_CHAIN_CODE_SHA_DRIFT")
            rescued.append(subject_id)
            print("[PASS-S7-RESCUE]", subject_id, flush=True)
        except Exception as exc:
            failures[subject_id] = repr(exc)
            print("[FAIL-S7-RESCUE]", subject_id, repr(exc), flush=True)
            break

    rescue_outputs = {}
    for subject_id in rescued:
        p = subject_root / subject_id / "V2_EXTENSION_CHAIN.json"
        if p.is_file():
            d = _json(p, "RESCUE_OUTPUT_CHAIN")
            rescue_outputs[subject_id] = {
                "chain_path": str(p),
                "chain_sha256": sha256(p),
                "chain_identity_sha256": d.get("chain_identity_sha256"),
            }

    rescue_receipt = {
        "schema": "RealSaS.Truth139V2ExtensionS7RescueReceipt.v1",
        "status": RESCUE_PASS if len(rescued) == 10 and not failures else "TRUTH139_V2_EXTENSION_S7_RESCUE_INCOMPLETE",
        "parent_failed_receipt": {"path": str(pre_receipt_path), "sha256": sha256(pre_receipt_path), "receipt_identity_sha256": parent_receipt.get("receipt_identity_sha256")},
        "rescue_code_sha": code_sha,
        "worker_code_manifest": {"path": str(worker_path), "sha256": sha256(worker_path), "identity_sha256": worker_manifest["identity_sha256"]},
        "producer_revision": PRODUCER_REVISION,
        "cpu_court_tolerance_relaxed": False,
        "requested_subject_ids": list(EXPECTED_FAILED_SUBJECTS),
        "rescued_subject_ids": rescued,
        "failed_jobs_by_subject": failures,
        "subject_output_bindings": rescue_outputs,
        "completed_at_utc": _utc(),
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
    }
    rescue_receipt["receipt_identity_sha256"] = _digest(rescue_receipt)
    rescue_receipt_path = batch_root / "FULL139_V2_EXTENSION_S7_RESCUE_RECEIPT.json"
    _atomic_json(rescue_receipt_path, rescue_receipt)
    if failures or len(rescued) != 10:
        raise RuntimeError("TRUTH139_V2_EXTENSION_S7_RESCUE_INCOMPLETE")

    # Verify all 139 chains and bind their actual per-subject source commit.
    output_bindings = {}
    code_by_subject = {}
    incomplete = []
    for subject_id in subject_ids:
        p = subject_root / subject_id / "V2_EXTENSION_CHAIN.json"
        if not _chain_complete(p, subject_id):
            incomplete.append(subject_id)
            continue
        d = _json(p, "RESCUE_COMPOSITE_CHAIN")
        code = str(((d.get("bindings") or {}).get("code_commit_sha")) or "")
        if len(code) != 40:
            raise ValueError("RESCUE_COMPOSITE_CHAIN_CODE_SHA:" + subject_id)
        code_by_subject[subject_id] = code
        output_bindings[subject_id] = {
            "chain_path": str(p),
            "chain_sha256": sha256(p),
            "chain_identity_sha256": d.get("chain_identity_sha256"),
        }
    if incomplete or len(output_bindings) != 139:
        raise RuntimeError("RESCUE_COMPOSITE_NOT_139:" + ",".join(incomplete))

    parent_pass = [sid for sid in subject_ids if sid not in set(EXPECTED_FAILED_SUBJECTS)]
    composite = dict(parent_receipt)
    composite.pop("receipt_identity_sha256", None)
    composite.update({
        "status": FINAL_PASS,
        "completed_subject_count": 139,
        "completed_subject_ids": list(subject_ids),
        "failed_jobs_by_subject": {},
        "subject_output_bindings": output_bindings,
        "subject_code_commit_sha_by_subject": code_by_subject,
        "mixed_worker_provenance": len(set(code_by_subject.values())) > 1,
        "provenance_mode": "PARENT_129_PLUS_SCOPED_S7_RESCUE_10__EXPLICIT_PER_SUBJECT_CODE_BINDING",
        "parent_pass_subject_ids": parent_pass,
        "rescue_subject_ids": list(EXPECTED_FAILED_SUBJECTS),
        "rescue_overlay": {
            "receipt_path": str(rescue_receipt_path),
            "receipt_sha256": sha256(rescue_receipt_path),
            "receipt_identity_sha256": rescue_receipt["receipt_identity_sha256"],
            "rescue_code_sha": code_sha,
            "producer_revision": PRODUCER_REVISION,
            "cpu_court_tolerance_relaxed": False,
        },
        "completed_at_utc": _utc(),
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
    })
    composite["receipt_identity_sha256"] = _digest(composite)
    _atomic_json(receipt_path, composite)

    progress_path = batch_root / "FULL139_V2_EXTENSION_PROGRESS.json"
    progress = {
        "schema": "RealSaS.Truth139V2ExtensionProgress.v1",
        "batch_id": composite.get("batch_id"),
        "base_authority_kind": (composite.get("base_full139_receipt") or {}).get("authority_kind"),
        "base_receipt_path": (composite.get("base_full139_receipt") or {}).get("path"),
        "base_receipt_sha256": (composite.get("base_full139_receipt") or {}).get("sha256"),
        "base_subject_binding_identity_sha256": composite.get("base_subject_binding_identity_sha256"),
        "subject_count": 139,
        "completed_subject_count": 139,
        "completed_subject_ids": list(subject_ids),
        "failed_subject_count": 0,
        "failed_jobs_by_subject": {},
        "last_subject_id": EXPECTED_FAILED_SUBJECTS[-1],
        "updated_at_utc": _utc(),
        "rescue_receipt_path": str(rescue_receipt_path),
        "training_performed": False,
    }
    _atomic_json(progress_path, progress)

    return {
        "status": composite["status"],
        "completed_subject_count": 139,
        "failed_subject_count": 0,
        "rescue_subject_count": 10,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "receipt_identity_sha256": composite["receipt_identity_sha256"],
        "rescue_receipt_path": str(rescue_receipt_path),
        "rescue_receipt_sha256": sha256(rescue_receipt_path),
        "rescue_code_sha": code_sha,
        "producer_revision": PRODUCER_REVISION,
        "mixed_worker_provenance": composite["mixed_worker_provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--parent-subjects", required=True, type=Path)
    parser.add_argument("--rescue-subjects", required=True, type=Path)
    parser.add_argument("--expected-parent-receipt-sha256", required=True)
    parser.add_argument("--expected-parent-receipt-identity-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = rescue(
        batch_root=args.batch_root,
        parent_subjects=args.parent_subjects,
        rescue_subjects=args.rescue_subjects,
        expected_parent_receipt_sha256=args.expected_parent_receipt_sha256,
        expected_parent_receipt_identity_sha256=args.expected_parent_receipt_identity_sha256,
        device=args.device,
    )
    print("\n[S7-RESCUE-FINAL]")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
