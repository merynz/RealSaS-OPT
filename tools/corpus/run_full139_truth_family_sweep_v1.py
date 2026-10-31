#!/usr/bin/env python3
from __future__ import annotations

"""Resumable Full139 truth-family audit/materialization orchestrator.

Historical V3.2 readiness is used ONLY as an immutable artifact inventory.  Its
split, admission and training_authorized fields have no authority here.

Canonical family precedence:
- SURFACE_GEOMETRY / SIGNED_DISTANCE_OCCUPANCY / ORIENTED_NORMALS come from the
  component-geometry independent court.
- SURFACE_TOPOLOGY comes ONLY from MechanicalSurfaceTopologyCourt.  The local
  owner-preserving geometry partition is retained as sign/addressing evidence and
  never promoted to mechanical topology.
- APPEARANCE_CAA / VISIBILITY_OCCLUSION / RAY_COMPOSITING remain measured-domain
  candidates until a later corpus-level student-observability receipt is sealed.
- ANIMATION_KINEMATICS remains pending after source court until canonical replay
  court is added/passed.

No final split, subject disposition, family seal or training readiness is minted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "RealSaS.Full139TruthFamilySweepReceipt.v1"
STATUS = "PASS_FULL139_TRUTH_FAMILY_SWEEP_MEASURED__FINAL_ADMISSION_PENDING"
EXPECTED_SUBJECTS = 139


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return {"path": str(path), "sha256": sha256(path)}


def load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("FULL139_SWEEP_JSON_OBJECT:" + str(path))
    return value


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    if not path.is_file() or sha256(path) != str(row.get("sha256") or ""):
        raise ValueError(code + ":" + str(path))
    return path


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def run(cmd: list[str], *, repo_root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            cmd, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if process.returncode != 0:
        tail = log_path.read_text(errors="replace")[-12000:]
        raise RuntimeError("COMMAND_FAILED:%d:%s\n%s" % (process.returncode, " ".join(cmd), tail))


def step(
    name: str,
    cmd: list[str],
    expected: Path,
    *,
    repo_root: Path,
    logs: Path,
    resume: bool,
) -> Path:
    expected = expected.resolve()
    if resume and expected.is_file():
        # Existing output is still re-opened by every downstream independent court;
        # resume therefore skips compute but never bypasses later byte/hash checks.
        return expected
    run(cmd, repo_root=repo_root, log_path=logs / f"{name}.log")
    if not expected.is_file():
        raise RuntimeError("STEP_OUTPUT_MISSING:" + name + ":" + str(expected))
    return expected


def court_file(path: Path, code: str) -> dict:
    value = load_json(path)
    if not str(value.get("status") or "").startswith("PASS_"):
        raise ValueError(code + "_NOT_PASS")
    if (value.get("authority") or {}).get("training_authorized") is True:
        raise ValueError(code + "_PREMATURE_TRAINING_AUTHORITY")
    return value


def source_cache_map(manifest: dict) -> dict[str, dict]:
    if manifest.get("subject_count") != EXPECTED_SUBJECTS or manifest.get("source_failures") not in ({}, None):
        raise ValueError("FULL139_SWEEP_SOURCE_CACHE_NOT_COMPLETE")
    rows = list(manifest.get("subjects") or [])
    result = {str(row.get("subject_id")): dict(row) for row in rows}
    if len(result) != EXPECTED_SUBJECTS or any(not row.get("valid") for row in result.values()):
        raise ValueError("FULL139_SWEEP_SOURCE_CACHE_SUBJECT_SET")
    if any(Path(str(row.get("corpus_relative_path") or "")).suffix.lower() != ".gltf" for row in result.values()):
        raise ValueError("FULL139_SWEEP_NON_GLTF_SOURCE_ADAPTER_REQUIRED")
    return result


def historical_inventory(readiness: dict) -> dict[str, dict]:
    rows = dict(readiness.get("subjects") or {})
    if readiness.get("subject_count") != EXPECTED_SUBJECTS or len(rows) != EXPECTED_SUBJECTS:
        raise ValueError("FULL139_SWEEP_HISTORICAL_LEDGER_SUBJECT_SET")
    required = {"geometry", "geometry_court", "rgb", "rgb_court", "stage48", "tessa_teacher", "tessa_court"}
    for subject, row in rows.items():
        if not required.issubset(row):
            raise ValueError("FULL139_SWEEP_HISTORICAL_INVENTORY_ROW:" + subject)
    return rows


def stage48_inputs(subject_id: str, witness_path: Path) -> dict[str, Path]:
    witness = load_json(witness_path)
    if (
        witness.get("schema") != "RealSaS.Truth139Stage48RendererWitness.v1"
        or witness.get("status") != "PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"
        or witness.get("subject_id") != subject_id
    ):
        raise ValueError("FULL139_SWEEP_STAGE48_WITNESS:" + subject_id)
    source = bound((witness.get("parents") or {}).get("source"), "FULL139_SWEEP_STAGE48_SOURCE")
    camera = bound((witness.get("parents") or {}).get("camera"), "FULL139_SWEEP_STAGE48_CAMERA")
    stage47 = bound((witness.get("parents") or {}).get("geometry"), "FULL139_SWEEP_STAGE47_GEOMETRY")
    teacher_scene = bound(witness.get("teacher_scene"), "FULL139_SWEEP_TEACHER_SCENE")
    response_chain = bound(witness.get("response_chain"), "FULL139_SWEEP_RESPONSE_CHAIN")
    packet_root = camera.parent
    expected_packet = {
        "SOURCE_REST_RENDER_CANDIDATE.json",
        "SOURCE_SCENE_SNAPSHOT.json",
        "IRIS_OBSERVATION_ADMISSION.json",
        "REST_SURFACE_GEOMETRY.npz",
        *{f"V{i}.png" for i in range(8)},
        *{f"V{i}.mask.bin" for i in range(8)},
    }
    missing = sorted(name for name in expected_packet if not (packet_root / name).is_file())
    if missing:
        raise FileNotFoundError("FULL139_SWEEP_PACKET_FILES:" + subject_id + ":" + ",".join(missing))
    return {
        "source": source,
        "camera": camera,
        "stage47": stage47,
        "teacher_scene": teacher_scene,
        "response_chain": response_chain,
        "packet_root": packet_root,
        "snapshot": packet_root / "SOURCE_SCENE_SNAPSHOT.json",
        "rest_surface": packet_root / "REST_SURFACE_GEOMETRY.npz",
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run_subject(
    subject_id: str,
    inventory_row: dict,
    source_cache_row: dict,
    *,
    repo_root: Path,
    output_root: Path,
    resume: bool,
) -> dict:
    started = time.time()
    subject_root = (output_root / "subjects" / subject_id).resolve()
    logs = subject_root / "logs"
    stage48_path = bound(inventory_row["stage48"], "FULL139_SWEEP_HISTORICAL_STAGE48_DRIFT")
    inputs = stage48_inputs(subject_id, stage48_path)
    if sha256(inputs["source"]) != str(source_cache_row.get("actual_sha256") or ""):
        raise ValueError("FULL139_SWEEP_SOURCE_CACHE_SHA:" + subject_id)

    py = sys.executable
    def tool(rel: str) -> str:
        return str((repo_root / rel).resolve())

    # 1. Source / observation
    so = subject_root / "source_observation"
    so_manifest = step("source_observation_build", [py, tool("tools/corpus/build_source_observation_authority_v1.py"), "--subject-id", subject_id, "--source", str(inputs["source"]), "--packet-root", str(inputs["packet_root"]), "--output-dir", str(so)], so / "SOURCE_OBSERVATION_AUTHORITY_MANIFEST_V1.json", repo_root=repo_root, logs=logs, resume=resume)
    so_court = step("source_observation_court", [py, tool("tools/corpus/validate_source_observation_authority_v1.py"), "--manifest", str(so_manifest), "--output", str(so / "SOURCE_OBSERVATION_AUTHORITY_COURT_V1.json")], so / "SOURCE_OBSERVATION_AUTHORITY_COURT_V1.json", repo_root=repo_root, logs=logs, resume=resume)

    # 2. Component geometry / SDF / normals. Local geometry topology is NOT final mechanical topology.
    geo = subject_root / "geometry"
    geo_manifest = step("geometry_build", [py, tool("tools/corpus/build_component_geometry_authority_v1.py"), "--subject-id", subject_id, "--geometry-npz", str(inputs["rest_surface"]), "--source-scene-snapshot", str(inputs["snapshot"]), "--camera-bundle", str(inputs["camera"]), "--output-dir", str(geo)], geo / "COMPONENT_GEOMETRY_AUTHORITY_MANIFEST_V1.json", repo_root=repo_root, logs=logs, resume=resume)
    geo_court = step("geometry_court", [py, tool("tools/corpus/validate_component_geometry_authority_v1.py"), "--geometry-manifest", str(geo_manifest), "--output", str(geo / "COMPONENT_GEOMETRY_AUTHORITY_COURT_V1.json")], geo / "COMPONENT_GEOMETRY_AUTHORITY_COURT_V1.json", repo_root=repo_root, logs=logs, resume=resume)

    # 3. Qualified mechanical surface topology from Stage47 posed evidence.
    mech = subject_root / "mechanical_surface"
    mech_manifest = step("mechanical_surface_build", [py, tool("tools/corpus/build_mechanical_surface_topology_manifest_v1.py"), "--subject-id", subject_id, "--geometry-response", str(inputs["stage47"]), "--output-dir", str(mech)], mech / "MECHANICAL_SURFACE_TOPOLOGY_MANIFEST_V1.json", repo_root=repo_root, logs=logs, resume=resume)
    mech_court = step("mechanical_surface_court", [py, tool("tools/corpus/validate_mechanical_surface_topology_authority_v1.py"), "--manifest", str(mech_manifest), "--output", str(mech / "MECHANICAL_SURFACE_TOPOLOGY_COURT_V1.json")], mech / "MECHANICAL_SURFACE_TOPOLOGY_COURT_V1.json", repo_root=repo_root, logs=logs, resume=resume)

    # 4. Raw-source rig/skin plus importer equivalence.
    rig = subject_root / "rig_skin"
    rig_manifest = step("rig_skin_build", [py, tool("tools/corpus/build_gltf_rig_skin_authority_v1.py"), "--source", str(inputs["source"]), "--output-dir", str(rig)], rig / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V1.json", repo_root=repo_root, logs=logs, resume=resume)
    rig_court = step("rig_skin_source_court", [py, tool("tools/corpus/validate_gltf_rig_skin_authority_v1.py"), "--manifest", str(rig_manifest), "--output", str(rig / "GLTF_RIG_SKIN_AUTHORITY_COURT_V1.json")], rig / "GLTF_RIG_SKIN_AUTHORITY_COURT_V1.json", repo_root=repo_root, logs=logs, resume=resume)
    rig_import = step("rig_skin_importer_court", [py, tool("tools/corpus/validate_gltf_imported_rig_skin_equivalence_v1.py"), "--source-manifest", str(rig_manifest), "--source-court", str(rig_court), "--teacher-scene", str(inputs["teacher_scene"]), "--geometry-and-response", str(inputs["stage47"]), "--output", str(rig / "GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE_COURT_V1.json")], rig / "GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE_COURT_V1.json", repo_root=repo_root, logs=logs, resume=resume)

    # 5. Source-native animation. Canonical replay remains a separate gate.
    anim = subject_root / "animation"
    anim_manifest = step("animation_build", [py, tool("tools/corpus/build_gltf_animation_authority_v1.py"), "--source", str(inputs["source"]), "--output-dir", str(anim)], anim / "GLTF_ANIMATION_KINEMATICS_MANIFEST_V1.json", repo_root=repo_root, logs=logs, resume=resume)
    anim_court = step("animation_source_court", [py, tool("tools/corpus/validate_gltf_animation_authority_v1.py"), "--manifest", str(anim_manifest), "--output", str(anim / "GLTF_ANIMATION_KINEMATICS_COURT_V1.json")], anim / "GLTF_ANIMATION_KINEMATICS_COURT_V1.json", repo_root=repo_root, logs=logs, resume=resume)

    # 6. Causal response on mechanical surface topology.
    response = subject_root / "causal_response"
    response_manifest = step("causal_response_build", [py, tool("tools/corpus/build_causal_response_authority_v1.py"), "--subject-id", subject_id, "--mechanical-manifest", str(mech_manifest), "--mechanical-court", str(mech_court), "--teacher-scene", str(inputs["teacher_scene"]), "--output-dir", str(response)], response / "CAUSAL_RESPONSE_AUTHORITY_MANIFEST_V1.json", repo_root=repo_root, logs=logs, resume=resume)
    response_court = step("causal_response_court", [py, tool("tools/corpus/validate_causal_response_authority_v1.py"), "--manifest", str(response_manifest), "--output", str(response / "CAUSAL_RESPONSE_AUTHORITY_COURT_V1.json")], response / "CAUSAL_RESPONSE_AUTHORITY_COURT_V1.json", repo_root=repo_root, logs=logs, resume=resume)

    # 7. Graphics measured domains bound to canonical Stage48 witness.
    graphics = subject_root / "graphics"
    graphics_manifest = step("graphics_build", [py, tool("tools/corpus/build_presentation_family_from_stage48_witness_v2.py"), "--stage48-witness", str(stage48_path), "--output-dir", str(graphics)], graphics / "PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V2.json", repo_root=repo_root, logs=logs, resume=resume)
    graphics_court = step("graphics_court", [py, tool("tools/corpus/validate_presentation_family_from_stage48_witness_v2.py"), "--manifest", str(graphics_manifest), "--output", str(graphics / "PRESENTATION_FAMILY_AUTHORITY_COURT_V2.json")], graphics / "PRESENTATION_FAMILY_AUTHORITY_COURT_V2.json", repo_root=repo_root, logs=logs, resume=resume)

    # 8. TESSA V5 consumer on mechanical topology. CAUSAL_RESPONSE is independently courted above.
    tessa = subject_root / "tessa_v5"
    tessa_manifest = step("tessa_v5_build", [py, tool("tools/training/build_corpus_mechanical_tessa_teacher_v5.py"), "--subject-id", subject_id, "--mechanical-manifest", str(mech_manifest), "--mechanical-court", str(mech_court), "--response-chain", str(inputs["response_chain"]), "--camera-bundle", str(inputs["camera"]), "--output-dir", str(tessa)], tessa / "CORPUS_MECHANICAL_TESSA_TEACHER_MANIFEST_V5.json", repo_root=repo_root, logs=logs, resume=resume)
    tessa_court = step("tessa_v5_court", [py, tool("tools/training/validate_corpus_mechanical_tessa_teacher_v5.py"), "--teacher-manifest", str(tessa_manifest), "--output", str(tessa / "CORPUS_MECHANICAL_TESSA_TEACHER_COURT_V5.json")], tessa / "CORPUS_MECHANICAL_TESSA_TEACHER_COURT_V5.json", repo_root=repo_root, logs=logs, resume=resume)

    courts = {
        "source_observation": court_file(so_court, "SOURCE_OBSERVATION"),
        "geometry": court_file(geo_court, "GEOMETRY"),
        "mechanical_surface": court_file(mech_court, "MECHANICAL_SURFACE"),
        "rig_skin_source": court_file(rig_court, "RIG_SKIN_SOURCE"),
        "rig_skin_importer": court_file(rig_import, "RIG_SKIN_IMPORTER"),
        "animation_source": court_file(anim_court, "ANIMATION_SOURCE"),
        "causal_response": court_file(response_court, "CAUSAL_RESPONSE"),
        "graphics_measurement": court_file(graphics_court, "GRAPHICS_MEASUREMENT"),
        "tessa_v5": court_file(tessa_court, "TESSA_V5"),
    }

    geo_out = dict(courts["geometry"].get("family_candidate_outcomes") or {})
    mech_out = dict(courts["mechanical_surface"].get("family_candidate_outcomes") or {})
    rig_out = dict(courts["rig_skin_importer"].get("family_candidate_outcomes") or {})
    response_out = dict(courts["causal_response"].get("family_candidate_outcomes") or {})
    graphics_out = dict(courts["graphics_measurement"].get("family_candidate_outcomes") or {})
    source_out = dict(courts["source_observation"].get("family_candidate_outcomes") or {})

    # Final-family *candidates* only.  Measured graphics and source-only animation
    # deliberately remain PENDING so no consumer eligibility can be minted yet.
    family = {
        "SOURCE_IMPORT": source_out.get("SOURCE_IMPORT", "PENDING"),
        "CAMERA_OBSERVATION": source_out.get("CAMERA_OBSERVATION", "PENDING"),
        "SURFACE_GEOMETRY": geo_out.get("SURFACE_GEOMETRY", "PENDING"),
        "SURFACE_TOPOLOGY": mech_out.get("SURFACE_TOPOLOGY", "PENDING"),
        "SIGNED_DISTANCE_OCCUPANCY": geo_out.get("SIGNED_DISTANCE_OCCUPANCY", "PENDING"),
        "ORIENTED_NORMALS": geo_out.get("ORIENTED_NORMALS", "PENDING"),
        "APPEARANCE_CAA": "PENDING_STUDENT_OBSERVABILITY_ADMISSION" if graphics_out.get("APPEARANCE_CAA", "").startswith("QUALIFIED_") else graphics_out.get("APPEARANCE_CAA", "PENDING"),
        "VISIBILITY_OCCLUSION": "PENDING_STUDENT_OBSERVABILITY_ADMISSION" if graphics_out.get("VISIBILITY_OCCLUSION", "").startswith("QUALIFIED_") else graphics_out.get("VISIBILITY_OCCLUSION", "PENDING"),
        "RAY_COMPOSITING": "PENDING_STUDENT_OBSERVABILITY_ADMISSION" if graphics_out.get("RAY_COMPOSITING", "").startswith("QUALIFIED_") else graphics_out.get("RAY_COMPOSITING", "PENDING"),
        "CAUSAL_RESPONSE": response_out.get("CAUSAL_RESPONSE", "PENDING"),
        "RIG_HIERARCHY": rig_out.get("RIG_HIERARCHY", "PENDING"),
        "SKIN_WEIGHTS": rig_out.get("SKIN_WEIGHTS", "PENDING"),
        "ANIMATION_KINEMATICS": "PENDING_CANONICAL_REPLAY_COURT",
    }

    receipt = {
        "schema": "RealSaS.SubjectTruthFamilySweepReceipt.v1",
        "status": "PASS_SUBJECT_TRUTH_FAMILY_SWEEP__FINAL_ADMISSION_PENDING",
        "subject_id": subject_id,
        "historical_inventory": {"stage48": dict(inventory_row["stage48"]), "packet_lineage": inventory_row.get("packet_lineage")},
        "source_cache": {k: source_cache_row.get(k) for k in ("corpus_relative_path", "actual_sha256", "byte_size", "valid")},
        "inputs": {name: ref(path) for name, path in inputs.items() if name != "packet_root"},
        "courts": {name: ref(path) for name, path in {
            "source_observation": so_court, "geometry": geo_court, "mechanical_surface": mech_court,
            "rig_skin_source": rig_court, "rig_skin_importer": rig_import, "animation_source": anim_court,
            "causal_response": response_court, "graphics_measurement": graphics_court, "tessa_v5": tessa_court,
        }.items()},
        "local_geometric_topology_evidence": geo_out.get("SURFACE_TOPOLOGY"),
        "canonical_family_candidates": family,
        "pending_gates": [
            "ANIMATION_CANONICAL_REPLAY_COURT",
            "GRAPHICS_STUDENT_OBSERVABILITY_ADMISSION",
            "FINAL_SUBJECT_DISPOSITION",
            "FINAL_LINEAGE_SAFE_SPLIT",
            "TRAINING_READINESS_RECEIPT",
        ],
        "authority": {"final_subject_disposition_minted": False, "split_membership_minted": False, "training_authorized": False},
        "elapsed_seconds": float(time.time() - started),
    }
    receipt_path = subject_root / "SUBJECT_TRUTH_FAMILY_SWEEP_RECEIPT_V1.json"
    write_json(receipt_path, receipt)
    receipt["receipt_sha256"] = sha256(receipt_path)
    return receipt


def aggregate(receipts: dict[str, dict], *, historical_ref: dict, source_cache_ref: dict, code_sha: str) -> dict:
    family_counts: dict[str, dict[str, int]] = {}
    for receipt in receipts.values():
        for family, status in receipt["canonical_family_candidates"].items():
            family_counts.setdefault(family, {})[status] = family_counts.setdefault(family, {}).get(status, 0) + 1
    failures = {subject: row for subject, row in receipts.items() if str(row.get("status") or "").startswith("FAIL_")}
    return {
        "schema": SCHEMA,
        "status": STATUS if not failures and len(receipts) == EXPECTED_SUBJECTS else "INCOMPLETE_FULL139_TRUTH_FAMILY_SWEEP",
        "code_commit_sha": code_sha,
        "subject_count": len(receipts),
        "historical_readiness_inventory": historical_ref,
        "source_cache_inventory": source_cache_ref,
        "family_candidate_counts": family_counts,
        "failures": failures,
        "authority": {
            "historical_split_inherited": False,
            "historical_training_authorization_inherited": False,
            "final_subject_disposition_minted": False,
            "final_split_minted": False,
            "training_authorized": False,
        },
        "pending_global_gates": [
            "ANIMATION_CANONICAL_REPLAY_COURT",
            "GRAPHICS_STUDENT_OBSERVABILITY_ADMISSION_AFTER_FINAL_ADMITTED_CORPUS",
            "FINAL_ADMIT_CORE_MASKED_QUARANTINE_DROP",
            "DUPLICATE_AND_LINEAGE_FAMILY_REBUILD",
            "FINAL_TRAIN_UNSEEN_SPLIT",
            "TRAINING_READINESS_RECEIPT",
        ],
    }


def plan_only(readiness_path: Path, source_cache_path: Path) -> dict:
    readiness = load_json(readiness_path); cache = load_json(source_cache_path)
    inv = historical_inventory(readiness); src = source_cache_map(cache)
    if set(inv) != set(src):
        raise ValueError("FULL139_SWEEP_INVENTORY_SUBJECT_MISMATCH")
    stage48_rows = sum(bool((row.get("stage48") or {}).get("sha256")) for row in inv.values())
    return {
        "schema": "RealSaS.Full139TruthFamilySweepPlan.v1",
        "status": "PASS_FULL139_SWEEP_PLAN__EXECUTION_NOT_PERFORMED",
        "subject_count": len(inv),
        "stage48_inventory_count": stage48_rows,
        "source_format_counts": {".gltf": len(src)},
        "historical_training_authority_ignored": True,
        "historical_split_ignored": True,
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-readiness", type=Path, required=True)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject-id")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    readiness_path = args.historical_readiness.resolve(); source_cache_path = args.source_cache_manifest.resolve(); output_root = args.output_root.resolve()
    if args.plan_only:
        result = plan_only(readiness_path, source_cache_path)
        output_root.mkdir(parents=True, exist_ok=True); write_json(output_root / "FULL139_TRUTH_FAMILY_SWEEP_PLAN_V1.json", result)
        print(json.dumps(result, sort_keys=True)); return

    readiness = load_json(readiness_path); cache = load_json(source_cache_path); inventory = historical_inventory(readiness); sources = source_cache_map(cache)
    if set(inventory) != set(sources): raise ValueError("FULL139_SWEEP_INVENTORY_SUBJECT_MISMATCH")
    subjects = sorted(inventory)
    if args.subject_id:
        if args.subject_id not in inventory: raise ValueError("FULL139_SWEEP_UNKNOWN_SUBJECT")
        subjects = [args.subject_id]
    if args.max_subjects is not None: subjects = subjects[: max(0, args.max_subjects)]

    receipts = {}; output_root.mkdir(parents=True, exist_ok=True)
    for index, subject in enumerate(subjects, 1):
        print(json.dumps({"event":"SUBJECT_START","index":index,"total":len(subjects),"subject_id":subject}), flush=True)
        try:
            receipts[subject] = run_subject(subject, inventory[subject], sources[subject], repo_root=repo_root, output_root=output_root, resume=not args.no_resume)
        except Exception as exc:
            failure = {"schema":"RealSaS.SubjectTruthFamilySweepFailure.v1","status":"FAIL_SUBJECT_TRUTH_FAMILY_SWEEP","subject_id":subject,"error_type":type(exc).__name__,"error":str(exc),"authority":{"training_authorized":False}}
            write_json(output_root / "subjects" / subject / "SUBJECT_TRUTH_FAMILY_SWEEP_FAILURE_V1.json", failure); receipts[subject] = failure
            if not args.continue_on_error: raise
        print(json.dumps({"event":"SUBJECT_END","index":index,"total":len(subjects),"subject_id":subject,"status":receipts[subject]["status"]}), flush=True)

    result = aggregate(receipts, historical_ref=ref(readiness_path), source_cache_ref=ref(source_cache_path), code_sha=git_head(repo_root))
    write_json(output_root / "FULL139_TRUTH_FAMILY_SWEEP_RECEIPT_V1.json", result)
    print(json.dumps({"status":result["status"],"subject_count":result["subject_count"],"family_candidate_counts":result["family_candidate_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
