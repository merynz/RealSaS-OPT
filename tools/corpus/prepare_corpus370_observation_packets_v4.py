#!/usr/bin/env python3
from __future__ import annotations

"""Prepare Corpus370 bind-rest observations under an exact pinned numeric runtime.

V2 retains the V1 source/observation semantics but makes Blender launch provenance
explicit. The patched 4.0.2 executable has a relative PT_INTERP and MUST be launched
through pinned_evaluator_runtime_v1.launch_options(); bare execution is invalid.
Packet identity binds source bytes, producer source, policy, Blender binary and the
full runtime manifest. This preparation still mints no family or training authority.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from tools.corpus.iris_observation_admission_v2 import audit_subject
from tools.observation.pinned_evaluator_runtime_v1 import launch_options

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v4"
REQUEST_SCHEMA = "RealSaS.Corpus370TruthOperatorRequest.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def source_binding() -> dict:
    paths = (
        "tools/observation/blender_controlled_8view_v2.py",
        "tools/observation/blender_controlled_8view_v1.py",
        "tools/observation/source_scene_contract_v2.py",
        "tools/observation/blender_source_parity_export_v1.py",
        "tools/observation/pinned_evaluator_runtime_v1.py",
        "tools/corpus/gltf_bind_scene_v1.py",
        "tools/corpus/gltf_bind_scene_v2.py",
        "tools/corpus/iris_observation_admission_v2.py",
        "tools/corpus/iris_observation_admission_v1.py",
        "tools/corpus/prepare_corpus370_observation_packets_v4.py",
        "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json",
    )
    return {path: sha256(ROOT / path) for path in paths}


def load_runtime(request_path: Path, blender: Path) -> tuple[dict, dict]:
    request_path = request_path.resolve()
    request = json.loads(request_path.read_text())
    require(request.get("schema") == REQUEST_SCHEMA, "CORPUS370_OBSERVATION_RUNTIME_REQUEST_SCHEMA")
    require(request.get("training_allowed") is False, "CORPUS370_OBSERVATION_RUNTIME_REQUEST_TRAINING")
    require(str(request.get("evaluator_executable") or "") == str(blender), "CORPUS370_OBSERVATION_RUNTIME_EXECUTABLE")
    require(str(request.get("expected_evaluator_binary_sha256") or "") == sha256(blender), "CORPUS370_OBSERVATION_RUNTIME_BINARY_SHA")
    runtime = dict(request.get("evaluator_runtime") or {})
    launch_options(blender, runtime)
    return request, runtime


def _packet_complete(packet: Path, subject_id: str, source_sha: str, policy: dict, blender_sha: str) -> bool:
    """Only reuse a complete, hash-bound original eight-view packet.

    Admission was previously checked by header alone. That could reuse a PNG,
    mask, camera or scene snapshot corrupted after the original PASS, especially
    after a Colab/Drive interruption. A stale/corrupt packet must be rebuilt,
    never silently promoted on subject-id or file-presence alone.
    """
    count = int(policy["view_count"])
    required = {
        "SOURCE_REST_RENDER_CANDIDATE.json",
        "SOURCE_SCENE_SNAPSHOT.json",
        "camera_bundle.json",
        "IRIS_OBSERVATION_ADMISSION.json",
        "imported.json",
        *{f"V{i}.png" for i in range(count)},
        *{f"V{i}.mask.bin" for i in range(count)},
    }
    if any(not (packet / name).is_file() for name in required):
        return False
    try:
        admission_path = packet / "IRIS_OBSERVATION_ADMISSION.json"
        candidate_path = packet / "SOURCE_REST_RENDER_CANDIDATE.json"
        camera_path = packet / "camera_bundle.json"
        snapshot_path = packet / "SOURCE_SCENE_SNAPSHOT.json"
        imported_path = packet / "imported.json"
        admission = json.loads(admission_path.read_text())
        candidate = json.loads(candidate_path.read_text())
        camera = json.loads(camera_path.read_text())
        imported = json.loads(imported_path.read_text())
        snapshot = json.loads(snapshot_path.read_text())
        if not (
            admission.get("schema") == "RealSaS.IrisBindObservationAdmission.v1"
            and admission.get("status") == "PASS_IRIS_BIND_OBSERVATIONS"
            and admission.get("subject_id") == subject_id
            and admission.get("source_sha256") == source_sha
            and admission.get("candidate_sha256") == sha256(candidate_path)
            and admission.get("camera_sha256") == sha256(camera_path)
            and admission.get("imported_geometry_sha256") == sha256(imported_path)
            and imported.get("import", {}).get("source_sha256") == source_sha
            and candidate.get("subject_id") == subject_id
            and candidate.get("source_sha256") == source_sha
            and candidate.get("blender_binary_sha256") == blender_sha
            and candidate.get("source_scene_snapshot", {}).get("sha256")
                == sha256(snapshot_path)
            and snapshot.get("source_sha256") == source_sha
            and int(candidate.get("resolution", -1)) == int(policy["resolution"])
            and camera.get("schema") == "RealSaS.CameraProjectionBundle.v1"
            and len(camera.get("cameras") or []) == count
        ):
            return False
        a_views = admission.get("views") or []
        c_views = candidate.get("views") or []
        if len(a_views) != count or len(c_views) != count:
            return False
        if ([v.get("view_index") for v in a_views] != list(range(count))
                or [v.get("view_index") for v in c_views] != list(range(count))
                or [v.get("view_index") for v in camera["cameras"]]
                    != list(range(count))):
            return False
        for i in range(count):
            image_sha = sha256(packet / f"V{i}.png")
            mask_sha = sha256(packet / f"V{i}.mask.bin")
            if not (
                a_views[i].get("raster_sha256") == image_sha
                and a_views[i].get("mask_sha256") == mask_sha
                and c_views[i].get("raster", {}).get("sha256") == image_sha
                and c_views[i].get("foreground_mask", {}).get("sha256") == mask_sha
            ):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def run_blender(blender: Path, runtime: dict, arguments: list[str], log_path: Path) -> None:
    options = launch_options(blender, runtime)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            [str(blender), *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            **options,
        )


def prepare(
    roster_path: Path,
    extracted_source_root: Path,
    output_root: Path,
    blender: Path,
    probe: Path,
    policy_path: Path,
    runtime_request_path: Path,
    subject_id: str | None = None,
) -> dict:
    roster_path = roster_path.resolve()
    extracted_source_root = extracted_source_root.resolve()
    output_root = output_root.resolve()
    blender = blender.resolve()
    probe = probe.resolve()
    policy_path = policy_path.resolve()
    runtime_request_path = runtime_request_path.resolve()
    roster = json.loads(roster_path.read_text())
    policy = json.loads(policy_path.read_text())
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1", "CORPUS370_OBSERVATION_ROSTER_SCHEMA")
    require(int(roster.get("subject_count", -1)) == 370, "CORPUS370_OBSERVATION_ROSTER_COUNT")
    require(policy.get("schema") == "RealSaS.IrisObservationPreparationPolicy.v1", "CORPUS370_OBSERVATION_POLICY_SCHEMA")
    require(roster.get("training_authorized") is False and policy.get("training_authorized") is False, "CORPUS370_OBSERVATION_PREAUTHORIZED")
    require(blender.is_file(), "CORPUS370_OBSERVATION_BLENDER_MISSING")
    require(probe.is_file(), "CORPUS370_OBSERVATION_PROBE_MISSING")
    runtime_request, runtime = load_runtime(runtime_request_path, blender)
    blender_sha = sha256(blender)

    rows = list(roster.get("subjects") or [])
    if subject_id is not None:
        rows = [row for row in rows if row["subject_id"] == subject_id]
        require(len(rows) == 1, "CORPUS370_OBSERVATION_UNKNOWN_SUBJECT")

    binding = {
        "producer_sources": source_binding(),
        "blender_executable": str(blender),
        "blender_binary_sha256": blender_sha,
        "runtime_request": {"path": str(runtime_request_path), "sha256": sha256(runtime_request_path)},
        "runtime_manifest": runtime,
        "runtime_version": runtime_request.get("evaluator_version"),
        "probe_binary_sha256": sha256(probe),
        "policy": {"path": str(policy_path), "sha256": sha256(policy_path)},
    }
    outcomes = []
    output_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        sid = str(row["subject_id"])
        source = (extracted_source_root / str(row["source_relative_path"])).resolve()
        require(source.is_relative_to(extracted_source_root), "CORPUS370_OBSERVATION_SOURCE_ESCAPE")
        require(source.is_file() and sha256(source) == row["source_sha256"], "CORPUS370_OBSERVATION_SOURCE_DRIFT:" + sid)
        identity = hashlib.sha256(
            json.dumps(
                {"binding": binding, "subject_id": sid, "source_sha256": row["source_sha256"]},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        root = output_root / "objects" / identity
        packet = root / "packet" / sid
        if _packet_complete(packet, sid, row["source_sha256"], policy, blender_sha):
            state = "REUSED_EXACT_PACKET"
        else:
            for name in ("imported", "observations", "admission"):
                target = root / name / sid
                if target.exists():
                    target.rename(target.with_name(sid + ".failed." + str(time.time_ns())))
            imported = root / "imported" / sid / "imported.json"
            imported.parent.mkdir(parents=True, exist_ok=True)
            run_blender(
                blender,
                runtime,
                [
                    "--background", "--factory-startup", "--python-exit-code", "2",
                    "--python", str(ROOT / "tools/observation/blender_source_parity_export_v1.py"), "--",
                    str(source), str(imported),
                ],
                root / "logs" / "source_parity.log",
            )
            observation_dir = root / "observations" / sid
            run_blender(
                blender,
                runtime,
                [
                    "--background", "--factory-startup", "--python-exit-code", "2",
                    "--python", str(ROOT / "tools/observation/blender_controlled_8view_v2.py"), "--",
                    "--source", str(source),
                    "--expected-source-sha256", row["source_sha256"],
                    "--subject-id", sid,
                    "--out-dir", str(observation_dir),
                    "--resolution", str(policy["resolution"]),
                    "--minimum-border-margin-px", str(policy["minimum_border_margin_px"]),
                ],
                root / "logs" / "controlled_8view.log",
            )
            synthetic_row = {
                "subject_key": sid,
                "source": {
                    "corpus_relative_path": row["source_relative_path"],
                    "sha256": row["source_sha256"],
                },
            }
            audit_subject(
                synthetic_row,
                extracted_source_root,
                root / "imported",
                root / "observations",
                probe,
                root / "admission",
                policy,
            )
            packet.mkdir(parents=True, exist_ok=True)
            for path in observation_dir.iterdir():
                if path.is_file():
                    shutil.copyfile(path, packet / path.name)
            shutil.copyfile(root / "admission" / sid / "IRIS_OBSERVATION_ADMISSION.json", packet / "IRIS_OBSERVATION_ADMISSION.json")
            # Carry the ORIGINAL admitted bytes: code/hash equality alone cannot
            # recreate JSON whitespace or object order byte-for-byte.
            shutil.copyfile(imported, packet / "imported.json")
            require(_packet_complete(packet, sid, row["source_sha256"], policy, blender_sha), "CORPUS370_OBSERVATION_PACKET_INCOMPLETE:" + sid)
            state = "PREPARED_NEW_PACKET"
        admission = packet / "IRIS_OBSERVATION_ADMISSION.json"
        candidate = packet / "SOURCE_REST_RENDER_CANDIDATE.json"
        camera = packet / "camera_bundle.json"
        snapshot = packet / "SOURCE_SCENE_SNAPSHOT.json"
        outcomes.append(
            {
                "subject_id": sid,
                "source_sha256": row["source_sha256"],
                "packet_identity_sha256": identity,
                "packet_path": str(packet),
                "state": state,
                "bindings": {
                    "observation_admission": {"path": str(admission), "sha256": sha256(admission)},
                    "admitted_import_geometry": {"path": str(packet / "imported.json"),
                                                 "sha256": sha256(packet / "imported.json")},
                    "source_rest_candidate": {"path": str(candidate), "sha256": sha256(candidate)},
                    "camera_bundle": {"path": str(camera), "sha256": sha256(camera)},
                    "source_scene_snapshot": {"path": str(snapshot), "sha256": sha256(snapshot)},
                },
                "training_authorized": False,
            }
        )
        print(json.dumps({"event": "CORPUS370_OBSERVATION_PACKET_V2", "index": index, "total": len(rows), "subject_id": sid, "state": state}), flush=True)

    return {
        "schema": SCHEMA,
        "status": "PREPARED_OBSERVATION_INPUT_BYTES__CURRENT_FAMILY_COURTS_STILL_REQUIRED",
        "subject_count": len(outcomes),
        "full_candidate_count": 370,
        "exact_source_roster": {"path": str(roster_path), "sha256": sha256(roster_path)},
        "binding": binding,
        "subjects": outcomes,
        "authority": {
            "platform_attempts_created": False,
            "family_authority_minted": False,
            "split_minted": False,
            "training_authorized": False,
        },
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json")
    parser.add_argument("--runtime-request", type=Path, default=ROOT / "canonical/CORPUS370_TRUTH_OPERATOR_REQUEST_V1.json")
    parser.add_argument("--subject-id")
    args = parser.parse_args()
    result = prepare(
        args.exact_source_roster,
        args.extracted_source_root,
        args.output_root,
        args.blender,
        args.probe,
        args.policy,
        args.runtime_request,
        args.subject_id,
    )
    print(json.dumps({"schema": result["schema"], "subject_count": result["subject_count"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
