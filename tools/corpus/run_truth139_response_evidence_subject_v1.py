#!/usr/bin/env python3
"""Run one Truth139 Stage47 response-evidence materialization chain.

The runner replays the exact admitted observation-rest import, runs the released
TeacherSceneIR.v2 isolated-response apparatus, derives renderer-independent IRIS
surface response evidence, and closes it with an independent numeric court.
No training, split selection, Stage48 renderer witness, or product authority is
performed here.
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
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.hashing import content_sha256
from tools.corpus.build_truth139_response_evidence_v1 import sha256

CHAIN_SCHEMA = "RealSaS.Truth139IrisResponseEvidenceChain.v1"
CHAIN_PASS = "PASS_TRUTH139_IRIS_RESPONSE_EVIDENCE_CHAIN__DERIVED_TRUTH_ADMISSION_SEPARATE"
COURT_PASS = "PASS_IRIS_RESPONSE_EVIDENCE_INDEPENDENT_NUMERIC_COURT"
CONTRACT = ROOT / "canonical/TRUTH139_IRIS_RESPONSE_EVIDENCE_TEACHER_V1.json"
POLICY = ROOT / "canonical/TEACHER_KINEMATIC_CALIBRATION_POLICY_V2.json"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load(path: Path, code: str) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def ref(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("TRUTH139_RESPONSE_REF_MISSING:" + str(path))
    return {"path": str(path), "sha256": sha256(path)}


def run(command: list[str], *, cwd: Path, log_path: Path | None = None, env: dict | None = None) -> str:
    printable = " ".join(map(str, command))
    print("[RUN]", printable, flush=True)
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(ROOT) + os.pathsep + merged.get("PYTHONPATH", "")
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    process = subprocess.Popen(
        list(map(str, command)),
        cwd=str(cwd),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    text = "".join(lines)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
    if code != 0:
        raise RuntimeError(f"TRUTH139_RESPONSE_COMMAND_FAILED:{code}:{printable}")
    return text


def blender_run(
    blender: Path,
    program: Path,
    arguments: list[object],
    *,
    log_path: Path,
    maximum_seconds: float,
) -> None:
    command = [
        str(blender), "-b", "--factory-startup", "--python-exit-code", "2",
        "--python", str(program), "--", *map(str, arguments),
    ]
    printable = " ".join(command)
    print("[BLENDER]", printable, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f"[RESPONSE-STAGE47] elapsed_s={elapsed:.1f}", flush=True)
                    if elapsed >= maximum_seconds:
                        raise RuntimeError("TRUTH139_RESPONSE_STAGE47_TIME_BUDGET")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if code != 0:
        tail = log_path.read_text(errors="replace")[-16000:]
        raise RuntimeError("TRUTH139_RESPONSE_BLENDER_FAILED\n" + tail)


def verify_packet(subject_id: str, source: Path, packet_root: Path, blender: Path) -> tuple[dict, dict, dict, dict]:
    admission_path = packet_root / "IRIS_OBSERVATION_ADMISSION.json"
    candidate_path = packet_root / "SOURCE_REST_RENDER_CANDIDATE.json"
    snapshot_path = packet_root / "SOURCE_SCENE_SNAPSHOT.json"
    camera_path = packet_root / "camera_bundle.json"
    admission = load(admission_path, "TRUTH139_RESPONSE_ADMISSION")
    candidate = load(candidate_path, "TRUTH139_RESPONSE_CANDIDATE")
    snapshot = load(snapshot_path, "TRUTH139_RESPONSE_SNAPSHOT")
    camera = load(camera_path, "TRUTH139_RESPONSE_CAMERA")

    if admission.get("schema") != "RealSaS.IrisBindObservationAdmission.v1" or admission.get("status") != "PASS_IRIS_BIND_OBSERVATIONS":
        raise ValueError("TRUTH139_RESPONSE_ADMISSION_STATUS")
    if candidate.get("schema") != "RealSaS.Controlled8ViewSourceRestCandidate.v2":
        raise ValueError("TRUTH139_RESPONSE_CANDIDATE_SCHEMA")
    if snapshot.get("schema") != "RealSaS.ImportedSourceSceneSnapshot.v2":
        raise ValueError("TRUTH139_RESPONSE_SNAPSHOT_SCHEMA")
    if camera.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise ValueError("TRUTH139_RESPONSE_CAMERA_SCHEMA")
    source_sha = sha256(source)
    for value in (admission, candidate, snapshot):
        if value.get("source_sha256") != source_sha:
            raise ValueError("TRUTH139_RESPONSE_SOURCE_SHA_DRIFT")
    if admission.get("subject_id") != subject_id or candidate.get("subject_id") != subject_id:
        raise ValueError("TRUTH139_RESPONSE_SUBJECT_DRIFT")
    if admission.get("candidate_sha256") != sha256(candidate_path):
        raise ValueError("TRUTH139_RESPONSE_CANDIDATE_SHA_DRIFT")
    if (candidate.get("source_scene_snapshot") or {}).get("sha256") != sha256(snapshot_path):
        raise ValueError("TRUTH139_RESPONSE_SNAPSHOT_SHA_DRIFT")
    if admission.get("camera_sha256") != sha256(camera_path):
        raise ValueError("TRUTH139_RESPONSE_CAMERA_SHA_DRIFT")
    if len(camera.get("cameras") or ()) != 8:
        raise ValueError("TRUTH139_RESPONSE_CAMERA_COUNT")

    version = subprocess.check_output([str(blender), "--version"], text=True).splitlines()[0]
    if version != "Blender " + str(candidate.get("blender_version")):
        raise ValueError("TRUTH139_RESPONSE_BLENDER_VERSION_DRIFT")
    binary_sha = sha256(blender)
    producer_binary_sha = str(candidate.get("blender_binary_sha256") or "")
    if binary_sha != producer_binary_sha:
        raise ValueError(
            "TRUTH139_RESPONSE_BLENDER_BINARY_DRIFT:"
            + binary_sha + "!=" + producer_binary_sha
        )
    return admission, candidate, snapshot, camera


def lineage_bindings(admission_path: Path, admission: dict, camera_path: Path, camera: dict) -> tuple[str, str]:
    observation_binding = digest({
        "schema": "RealSaS.Truth139ExternalObservationBinding.v1",
        "admission_sha256": sha256(admission_path),
        "source_sha256": admission["source_sha256"],
        "camera_sha256": sha256(camera_path),
        "views": admission["views"],
    })
    normalization_binding = digest({
        "schema": "RealSaS.Truth139ExternalNormalizationBinding.v1",
        "observation_binding": observation_binding,
        "camera_sha256": sha256(camera_path),
        "center_xyz": camera["common_center_xyz"],
        "half_extent": camera["common_half_extent"],
        "normalized_bounds_min": [-1.0, -1.0, -1.0],
        "normalized_bounds_max": [1.0, 1.0, 1.0],
        "coordinate_frame": "REALSAS_OBJECT_FRAME",
        "normalization_policy": "EXACT_MANIFEST_OBJECT_CUBE_V1",
    })
    return observation_binding, normalization_binding


def run_subject(
    *,
    subject_id: str,
    source: Path,
    packet_root: Path,
    output_root: Path,
    blender: Path,
    maximum_seconds: float,
) -> dict:
    source = source.resolve()
    packet_root = packet_root.resolve()
    output_root = output_root.resolve()
    blender = blender.resolve()
    if not source.is_file() or not packet_root.is_dir() or not blender.is_file():
        raise FileNotFoundError("TRUTH139_RESPONSE_REQUIRED_INPUT_MISSING")
    output_root.mkdir(parents=True, exist_ok=True)
    chain_path = output_root / "IRIS_RESPONSE_EVIDENCE_CHAIN.json"
    if chain_path.is_file():
        existing = load(chain_path, "TRUTH139_RESPONSE_EXISTING_CHAIN")
        if existing.get("status") == CHAIN_PASS and existing.get("subject_id") == subject_id:
            print("[REUSE-RESPONSE]", subject_id, flush=True)
            return existing

    admission, candidate, _snapshot, camera = verify_packet(subject_id, source, packet_root, blender)
    admission_path = packet_root / "IRIS_OBSERVATION_ADMISSION.json"
    candidate_path = packet_root / "SOURCE_REST_RENDER_CANDIDATE.json"
    snapshot_path = packet_root / "SOURCE_SCENE_SNAPSHOT.json"
    camera_path = packet_root / "camera_bundle.json"
    observation_binding, normalization_binding = lineage_bindings(admission_path, admission, camera_path, camera)

    stage47_root = output_root / "stage47"
    stage47_root.mkdir(parents=True, exist_ok=True)
    replay = stage47_root / "admitted_observation_import.json"
    if not replay.is_file():
        blender_run(
            blender,
            ROOT / "tools/observation/blender_source_parity_export_v1.py",
            [source, replay],
            log_path=stage47_root / "observation.import.blender.log",
            maximum_seconds=maximum_seconds,
        )
    if sha256(replay) != admission.get("imported_geometry_sha256"):
        raise ValueError("TRUTH139_RESPONSE_ADMITTED_IMPORT_BYTES_DRIFT")

    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    job = {
        "schema": "RealSaS.TeacherSceneCalibrationJob.v2",
        "subject_key": subject_id,
        "source": ref(source),
        "observation_admission": ref(admission_path),
        "source_rest_candidate": ref(candidate_path),
        "source_scene_snapshot": ref(snapshot_path),
        "observation_import_geometry": ref(replay),
        "policy": ref(POLICY),
        "matrix_tolerance": 1.0e-6,
        "observation_bundle_sha256": observation_binding,
        "normalization_sha256": normalization_binding,
        "evaluator_binary_sha256": sha256(blender),
        "producer_source_commit": code_sha,
        "observation_runtime_binding": {
            "status": "PASS_EXACT_ADMITTED_IMPORT_BYTES",
            "observation_producer_binary_sha256": candidate["blender_binary_sha256"],
            "evaluator_binary_sha256": sha256(blender),
            "version": subprocess.check_output([str(blender), "--version"], text=True).splitlines()[0],
            "binary_identity_equal": True,
            "imported_geometry_sha256": sha256(replay),
        },
        "evaluator_runtime": None,
        "training_allowed": False,
    }
    job_hash = content_sha256(job)
    job_path = stage47_root / (job_hash + ".job.json")
    job_path.write_text(json.dumps(job, sort_keys=True, indent=2, allow_nan=False) + "\n")
    teacher_out = stage47_root / job_hash
    scene_path = teacher_out / "TEACHER_SCENE.json"
    if not scene_path.is_file():
        blender_run(
            blender,
            ROOT / "tools/truth_foundry/blender_teacher_scene_v2.py",
            ["--job", job_path, "--output", teacher_out],
            log_path=stage47_root / (job_hash + ".blender.log"),
            maximum_seconds=maximum_seconds,
        )
    scene = load(scene_path, "TRUTH139_RESPONSE_TEACHER_SCENE")
    if (
        scene.get("schema") != "RealSaS.TeacherSceneIR.v2"
        or scene.get("job_sha256") != job_hash
        or scene.get("authority") != "UNQUALIFIED_DERIVED_MEASUREMENTS"
        or scene.get("qualification_minted") is not False
        or scene.get("native_clip_refs") != []
    ):
        raise ValueError("TRUTH139_RESPONSE_TEACHER_SCENE_BINDING")
    binding = dict(scene.get("observation_rest_binding") or {})
    if binding.get("status") != "PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY" or binding.get("imported_geometry_sha256") != sha256(replay):
        raise ValueError("TRUTH139_RESPONSE_TEACHER_REST_BINDING")

    response_root = output_root / "response_evidence"
    manifest_path = response_root / "IRIS_RESPONSE_EVIDENCE_MANIFEST.json"
    if not manifest_path.is_file():
        run([
            sys.executable,
            str(ROOT / "tools/corpus/build_truth139_response_evidence_v1.py"),
            "--subject-id", subject_id,
            "--teacher-scene", scene_path,
            "--output-dir", response_root,
        ], cwd=ROOT, log_path=output_root / "response.producer.log")
    manifest = load(manifest_path, "TRUTH139_RESPONSE_MANIFEST")

    court_path = response_root / "IRIS_RESPONSE_EVIDENCE_COURT.json"
    if not court_path.is_file():
        run([
            sys.executable,
            str(ROOT / "tools/corpus/validate_truth139_response_evidence_v1.py"),
            "--response-manifest", manifest_path,
            "--output", court_path,
        ], cwd=ROOT, log_path=output_root / "response.court.log")
    court = load(court_path, "TRUTH139_RESPONSE_COURT")
    if court.get("status") != COURT_PASS or court.get("subject_id") != subject_id:
        raise ValueError("TRUTH139_RESPONSE_COURT_NOT_PASS")
    if court.get("response_evidence_identity_sha256") != manifest.get("response_evidence_identity_sha256"):
        raise ValueError("TRUTH139_RESPONSE_COURT_MANIFEST_IDENTITY_DRIFT")

    chain = {
        "schema": CHAIN_SCHEMA,
        "status": CHAIN_PASS,
        "subject_id": subject_id,
        "code_commit_sha": code_sha,
        "contract": ref(CONTRACT),
        "policy": ref(POLICY),
        "inputs": {
            "source": ref(source),
            "packet_root": str(packet_root),
            "observation_admission": ref(admission_path),
            "source_rest_candidate": ref(candidate_path),
            "source_scene_snapshot": ref(snapshot_path),
            "camera_bundle": ref(camera_path),
            "admitted_observation_import": ref(replay),
            "blender_binary": ref(blender),
        },
        "bindings": {
            "observation_binding_sha256": observation_binding,
            "normalization_binding_sha256": normalization_binding,
            "stage47_job_sha256": job_hash,
            "teacher_scene_sha256": sha256(scene_path),
            "response_evidence_identity_sha256": manifest["response_evidence_identity_sha256"],
            "response_court_identity_sha256": court["court_identity_sha256"],
        },
        "outputs": {
            "teacher_scene": ref(scene_path),
            "response_manifest": ref(manifest_path),
            "response_arrays": ref(response_root / "IRIS_RESPONSE_EVIDENCE_V1.npz"),
            "response_court": ref(court_path),
        },
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
        "stage48_renderer_witness_performed": False,
    }
    chain["chain_identity_sha256"] = digest(chain)
    temp = chain_path.with_suffix(".json.partial")
    temp.write_text(json.dumps(chain, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(chain_path)
    print(json.dumps({
        "status": chain["status"],
        "subject_id": subject_id,
        "chain_identity_sha256": chain["chain_identity_sha256"],
    }, sort_keys=True), flush=True)
    return chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--maximum-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    run_subject(
        subject_id=args.subject_id,
        source=args.source,
        packet_root=args.packet_root,
        output_root=args.output_root,
        blender=args.blender,
        maximum_seconds=args.maximum_seconds,
    )


if __name__ == "__main__":
    main()
