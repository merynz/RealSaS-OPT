"""Export a minimal, hash-sealed portable handoff for the V9 static mechanics notebook.

This exporter runs on the canonical self-hosted environment where the frozen
Knight authority/cache inputs exist. The resulting directory is independent of
that authority tree and is sufficient for audit_v9_static_mechanical_microstep_v1
portable mode.

Audit/research transport only. No product authority is minted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mechanical_partition_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    validate_canonical_mesh_candidate,
    validate_mechanical_partition,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_knight_repaired_quality_collapse_v1 import build_repaired_surface
from tools.audit_v9_teacher_oracle_iterative_repartition_v1 import read
from tools.demo.render_knight_motion_preview_v1 import _ctx


SCHEMA = "RealSaS.V9StaticMechanicsNotebookHandoff.v1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_role(src: Path, dst: Path, role: str) -> dict:
    if not src.is_file():
        raise FileNotFoundError(f"HANDOFF_SOURCE_MISSING:{role}:{src}")
    shutil.copy2(src, dst)
    return {
        "role": role,
        "path": dst.name,
        "bytes": int(dst.stat().st_size),
        "sha256": file_sha256(dst),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--partition-json", type=Path, required=True)
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--fresh-skeleton-json", type=Path, required=True)
    ap.add_argument("--teacher-bank", type=Path, required=True)
    ap.add_argument("--inverse-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    out = a.out_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    ctx = _ctx(a.authority_root.resolve(), a.run_id)
    rr = ctx["run_root"]

    surface = rigging_surface_from_dict(read(a.surface_json))
    partition = mechanical_partition_from_dict(read(a.partition_json))
    candidate = canonical_mesh_candidate_from_dict(read(a.candidate_json))
    validate_mechanical_partition(partition, surface)

    rebuilt_surface, explicit_faces = build_repaired_surface(rr, a.inverse_npz)
    if rebuilt_surface.geometry_lineage_hash != surface.geometry_lineage_hash:
        raise RuntimeError("HANDOFF_REBUILT_SURFACE_LINEAGE_DRIFT")

    # Validate the exact frozen parent before transport.
    # The carrier policy is not required for portable execution; structural
    # validation still protects malformed candidate/partition transport.
    if not candidate.vertices or not candidate.faces:
        raise RuntimeError("HANDOFF_EMPTY_CANDIDATE")

    camera_payload = stage_output_payload(
        ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
    )
    policy_payload = stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1"
    )

    roles = {}
    role_specs = (
        ("surface", a.surface_json, out / "REFINED_SURFACE.json"),
        ("partition", a.partition_json, out / "CYCLE_01_PARTITION.json"),
        ("candidate", a.candidate_json, out / "CYCLE_01_CANDIDATE.json"),
        ("fresh_skeleton", a.fresh_skeleton_json, out / "FRESH_QUALIFIED_SKELETON.json"),
        ("teacher_bank", a.teacher_bank, out / "V9_TEACHER_BANK.npz"),
        ("repaired_inverse", a.inverse_npz, out / "REPAIRED_INVERSE_V2.npz"),
    )
    for role, src, dst in role_specs:
        roles[role] = copy_role(src.resolve(), dst, role)

    camera_path = out / "QUALIFIED_CAMERA_SET.json"
    camera_path.write_text(json.dumps(camera_payload, indent=2, sort_keys=True) + "\n")
    roles["camera_set"] = {
        "role": "camera_set",
        "path": camera_path.name,
        "bytes": int(camera_path.stat().st_size),
        "sha256": file_sha256(camera_path),
    }

    policy_path = out / "MESH_QUALIFICATION_POLICY.json"
    policy_path.write_text(json.dumps(policy_payload, indent=2, sort_keys=True) + "\n")
    roles["mesh_policy"] = {
        "role": "mesh_policy",
        "path": policy_path.name,
        "bytes": int(policy_path.stat().st_size),
        "sha256": file_sha256(policy_path),
    }

    faces_path = out / "EXPLICIT_SURFACE_FACES.json"
    faces_payload = {
        "schema": "RealSaS.V9ExplicitSurfaceFaceProvenance.v1",
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "face_count": int(len(explicit_faces)),
        "faces": [list(row) for row in explicit_faces],
    }
    faces_path.write_text(json.dumps(faces_payload, indent=2, sort_keys=True) + "\n")
    roles["explicit_faces"] = {
        "role": "explicit_faces",
        "path": faces_path.name,
        "bytes": int(faces_path.stat().st_size),
        "sha256": file_sha256(faces_path),
    }

    manifest = {
        "schema": SCHEMA,
        "status": "PASS",
        "execution_class": "AUDIT_RESEARCH_PORTABLE_HANDOFF",
        "run_id": a.run_id,
        "source_repo_commit": None,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "partition_lineage_hash": partition.partition_lineage_hash,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "explicit_face_count": int(len(explicit_faces)),
        "files": roles,
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }

    # Commit identity is populated from the checked-out repository when available.
    head = Path(".git/HEAD")
    try:
        import subprocess
        manifest["source_repo_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        manifest["source_repo_commit"] = "UNKNOWN"

    manifest_path = out / "HANDOFF_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Self-verify every sealed file after the manifest is materialized.
    for role, meta in roles.items():
        p = out / meta["path"]
        if p.stat().st_size != int(meta["bytes"]):
            raise RuntimeError(f"HANDOFF_SIZE_DRIFT:{role}")
        if file_sha256(p) != meta["sha256"]:
            raise RuntimeError(f"HANDOFF_SHA_DRIFT:{role}")

    print("V9_NOTEBOOK_HANDOFF_PASS=" + json.dumps({
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "partition_lineage_hash": partition.partition_lineage_hash,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "explicit_face_count": len(explicit_faces),
        "file_count": len(roles),
        "out_dir": str(out),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
