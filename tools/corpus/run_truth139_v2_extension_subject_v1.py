#!/usr/bin/env python3
"""Run one subject's Truth139 V2 extension pass on an existing Full139 base chain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_SCHEMA = "RealSaS.Truth139V2ExtensionChain.v1"
CHAIN_STATUS = "PASS_TRUTH139_V2_EXTENSION_CHAIN_EXECUTED__DERIVED_TRUTH_ADMISSION_SEPARATE"
BASE_CHAIN_STATUS = "PASS_TRUTH139_GRAPHICS_MATERIALIZATION_CHAIN_EXECUTED__DERIVED_TRUTH_ADMISSION_SEPARATE"
EXPECTED = {
    "layer_manifest": "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE",
    "layer_court": "PASS_LAYER_PEEL_V2_GPU_FACE_SELECTION_MATCHES_COMPLETE_CPU_RAY_STACK",
    "component_manifest": "PASS_ANONYMOUS_SOURCE_COMPONENT_MASK_CANDIDATE__INDEPENDENT_REBINDING_COURT_REQUIRED",
    "component_court": "PASS_SOURCE_COMPONENT_MASKS_REBIND_EXACT_FACE_OWNER_AND_PEEL_TRUTH",
    "affinity_manifest": "PASS_DRAWABLE_AFFINITY_CANDIDATE__INDEPENDENT_REBINDING_COURT_REQUIRED",
    "affinity_court": "PASS_DRAWABLE_AFFINITY_REBINDS_EXACT_COMPONENT_AND_PEEL_TRUTH",
    "relations_manifest": "PASS_RELATIONS_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
    "relations_court": "PASS_RELATIONS_V2_REBINDS_CANONICAL_DENSE_AND_VISIBILITY_PARENT_TRUTH",
}


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


def _git_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, capture_output=True, check=True)
    return proc.stdout.strip().lower()


def _run(command: list[str], expected_output: Path, expected_status: str) -> dict:
    started = time.monotonic()
    proc = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    wall = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError("TRUTH139_V2_EXTENSION_STEP_FAILED\nCOMMAND " + " ".join(command) + "\nSTDOUT\n" + proc.stdout[-12000:] + "\nSTDERR\n" + proc.stderr[-12000:])
    payload = _json(expected_output, "TRUTH139_V2_EXTENSION_STEP_OUTPUT")
    if payload.get("status") != expected_status:
        raise ValueError(f"TRUTH139_V2_EXTENSION_STEP_STATUS:{payload.get('status')}!={expected_status}")
    return {"command": command, "output": str(expected_output), "output_sha256": sha256(expected_output), "status": expected_status, "wall_seconds": wall, "stdout_tail": proc.stdout[-4000:]}


def _pair_complete(manifest: Path, manifest_status: str, court: Path, court_status: str) -> bool:
    try:
        m, c = _json(manifest, "MANIFEST"), _json(court, "COURT")
        return m.get("status") == manifest_status and c.get("status") == court_status and str(m.get("subject_id") or "") == str(c.get("subject_id") or "")
    except Exception:
        return False


def run_subject(*, subject_id: str, base_subject_root: Path, output_root: Path, device: str, response_arrays: Path | None) -> dict:
    base_subject_root = base_subject_root.resolve()
    output_root = output_root.resolve()
    if not subject_id:
        raise ValueError("TRUTH139_V2_EXTENSION_SUBJECT")

    base_chain_path = base_subject_root / "GRAPHICS_MATERIALIZATION_CHAIN.json"
    base_chain = _json(base_chain_path, "TRUTH139_V2_EXTENSION_BASE_CHAIN")
    if base_chain.get("status") != BASE_CHAIN_STATUS or str(base_chain.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_V2_EXTENSION_BASE_CHAIN_STATUS")

    dense_manifest = base_subject_root / "dense_graphics/DENSE_GRAPHICS_GBUFFER_MANIFEST.json"
    dense_court = base_subject_root / "dense_graphics/DENSE_GRAPHICS_GEOMETRY_COURT.json"
    vis_manifest = base_subject_root / "visibility_occlusion/VISIBILITY_OCCLUSION_TEACHER_MANIFEST.json"
    vis_court = base_subject_root / "visibility_occlusion/VISIBILITY_OCCLUSION_COURT.json"
    for p in (dense_manifest, dense_court, vis_manifest, vis_court):
        if not p.is_file():
            raise FileNotFoundError("TRUTH139_V2_EXTENSION_BASE_ARTIFACT:" + str(p))

    output_root.mkdir(parents=True, exist_ok=True)
    layer_dir = output_root / "layer_peel"
    component_dir = output_root / "source_components"
    affinity_dir = output_root / "drawable_affinity"
    relations_dir = output_root / "relations"
    for p in (layer_dir, component_dir, affinity_dir, relations_dir):
        p.mkdir(parents=True, exist_ok=True)

    def py(rel: str, *args: object) -> list[str]:
        return [sys.executable, str((REPO_ROOT / rel).resolve()), *map(str, args)]

    steps: list[dict] = []
    reused: list[str] = []

    layer_manifest = layer_dir / "LAYER_PEEL_TEACHER_MANIFEST.json"
    layer_court = layer_dir / "LAYER_PEEL_COURT.json"
    if not _pair_complete(layer_manifest, EXPECTED["layer_manifest"], layer_court, EXPECTED["layer_court"]):
        steps.append(_run(py("tools/corpus/build_truth139_graphics_layer_peel_v2.py", "--dense-graphics-manifest", dense_manifest, "--dense-graphics-court", dense_court, "--output-dir", layer_dir, "--device", device), layer_manifest, EXPECTED["layer_manifest"]))
        steps.append(_run(py("tools/corpus/validate_truth139_graphics_layer_peel_v2.py", "--layer-peel-teacher-manifest", layer_manifest, "--output", layer_court), layer_court, EXPECTED["layer_court"]))
    else:
        reused.append("K_HIT_GEOMETRIC_PEEL")

    component_manifest = component_dir / "SOURCE_COMPONENT_MASKS_MANIFEST.json"
    component_court = component_dir / "SOURCE_COMPONENT_MASKS_COURT.json"
    if not _pair_complete(component_manifest, EXPECTED["component_manifest"], component_court, EXPECTED["component_court"]):
        steps.append(_run(py("tools/corpus/build_truth139_source_component_masks_v1.py", "--layer-peel-teacher-manifest", layer_manifest, "--output-dir", component_dir), component_manifest, EXPECTED["component_manifest"]))
        steps.append(_run(py("tools/corpus/validate_truth139_source_component_masks_v1.py", "--source-component-masks-manifest", component_manifest, "--output", component_court), component_court, EXPECTED["component_court"]))
    else:
        reused.append("ANONYMOUS_COMPONENT_MASKS")

    affinity_manifest = affinity_dir / "DRAWABLE_AFFINITY_MANIFEST.json"
    affinity_court = affinity_dir / "DRAWABLE_AFFINITY_COURT.json"
    if not _pair_complete(affinity_manifest, EXPECTED["affinity_manifest"], affinity_court, EXPECTED["affinity_court"]):
        steps.append(_run(py("tools/corpus/build_truth139_drawable_affinity_teacher_v1.py", "--component-mask-manifest", component_manifest, "--output-dir", affinity_dir), affinity_manifest, EXPECTED["affinity_manifest"]))
        steps.append(_run(py("tools/corpus/validate_truth139_drawable_affinity_teacher_v1.py", "--drawable-affinity-manifest", affinity_manifest, "--output", affinity_court), affinity_court, EXPECTED["affinity_court"]))
    else:
        reused.append("DRAWABLE_AFFINITY_RELATIONS")

    relations_manifest = relations_dir / "RELATIONS_TEACHER_MANIFEST.json"
    relations_court = relations_dir / "RELATIONS_COURT.json"
    if not _pair_complete(relations_manifest, EXPECTED["relations_manifest"], relations_court, EXPECTED["relations_court"]):
        cmd = py("tools/corpus/build_truth139_relations_teacher_v2.py", "--dense-graphics-manifest", dense_manifest, "--dense-graphics-court", dense_court, "--visibility-manifest", vis_manifest, "--visibility-court", vis_court, "--output-dir", relations_dir)
        if response_arrays is not None:
            cmd += ["--response-arrays", str(response_arrays.resolve())]
        steps.append(_run(cmd, relations_manifest, EXPECTED["relations_manifest"]))
        steps.append(_run(py("tools/corpus/validate_truth139_relations_teacher_v2.py", "--relations-teacher-manifest", relations_manifest, "--output", relations_court), relations_court, EXPECTED["relations_court"]))
    else:
        reused.append("PRIMITIVE_RELATIONS")

    bindings = {"code_commit_sha": _git_sha(), "base_chain": {"path": str(base_chain_path), "sha256": sha256(base_chain_path)}, "dense_graphics_manifest": {"path": str(dense_manifest), "sha256": sha256(dense_manifest)}, "dense_graphics_court": {"path": str(dense_court), "sha256": sha256(dense_court)}, "visibility_manifest": {"path": str(vis_manifest), "sha256": sha256(vis_manifest)}, "visibility_court": {"path": str(vis_court), "sha256": sha256(vis_court)}}
    outputs = {"layer_peel_manifest": {"path": str(layer_manifest), "sha256": sha256(layer_manifest)}, "layer_peel_court": {"path": str(layer_court), "sha256": sha256(layer_court)}, "component_masks_manifest": {"path": str(component_manifest), "sha256": sha256(component_manifest)}, "component_masks_court": {"path": str(component_court), "sha256": sha256(component_court)}, "drawable_affinity_manifest": {"path": str(affinity_manifest), "sha256": sha256(affinity_manifest)}, "drawable_affinity_court": {"path": str(affinity_court), "sha256": sha256(affinity_court)}, "relations_manifest": {"path": str(relations_manifest), "sha256": sha256(relations_manifest)}, "relations_court": {"path": str(relations_court), "sha256": sha256(relations_court)}}
    payload = {"schema": CHAIN_SCHEMA, "date": "2026-10-29", "status": CHAIN_STATUS, "subject_id": subject_id, "bindings": bindings, "executed_steps": steps, "reused_jobs": reused, "outputs": outputs, "derived_truth_execution_completed": True, "independent_courts_passed": True, "derived_truth_admitted": False, "training_performed": False, "split_created": False, "qualification_minted": False}
    payload["chain_identity_sha256"] = _digest(payload)
    chain_path = output_root / "V2_EXTENSION_CHAIN.json"
    staged = chain_path.with_suffix(".json.partial")
    staged.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(chain_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--base-subject-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--response-arrays", type=Path)
    args = parser.parse_args()
    result = run_subject(subject_id=args.subject_id, base_subject_root=args.base_subject_root, output_root=args.output_root, device=args.device, response_arrays=args.response_arrays)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "reused_jobs": result["reused_jobs"], "chain_identity_sha256": result["chain_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
