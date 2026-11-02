#!/usr/bin/env python3
from __future__ import annotations

"""Full cheap structural audit for the current Truth139 candidate set.

This is deliberately an *evidence collection* pass. It does not repair geometry,
change the historical split, mint sign/SDF truth, admit/drop subjects, or train a
model. The goal is to learn the real authoring/failure-class distribution before
we choose the next qualified geometry/sign contract.

The runner resolves every input through the existing hash-bound Truth139 lineage:
prepared REST_SURFACE_GEOMETRY -> response chain -> TeacherSceneIR -> Stage47
geometry_and_response. Folder naming is never used as authority.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corpus.structural_geometry_audit_v1 import audit_structural_geometry_v1


SCHEMA = "RealSaS.Truth139FullStructuralIntegrityAudit.v1"
ROW_SCHEMA = "RealSaS.Truth139StructuralIntegrityRow.v1"
STATUS = "PASS_TRUTH139_STRUCTURAL_EVIDENCE_COLLECTED__NO_TRAINING_AUTHORITY"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def bound_ref(row: dict, code: str) -> Path:
    path = Path(str((row or {}).get("path") or "")).resolve()
    expected = str((row or {}).get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def plain_ref(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path)}


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    with staged.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    staged.replace(path)


def teacher_geometry_ref(scene: dict) -> dict:
    rows = [
        dict(row)
        for row in scene.get("geometry_payloads") or ()
        if Path(str(row.get("path") or "")).name == "geometry_and_response.npz"
    ]
    if len(rows) != 1:
        raise ValueError("STRUCTURAL_SWEEP_STAGE47_GEOMETRY_CARDINALITY")
    return rows[0]


def packet_geometry(packet_root: Path):
    geometry_path = packet_root / "REST_SURFACE_GEOMETRY.npz"
    if not geometry_path.is_file():
        raise FileNotFoundError("STRUCTURAL_SWEEP_PACKET_GEOMETRY_MISSING:" + str(geometry_path))
    with np.load(geometry_path, allow_pickle=False) as z:
        required = {"rest_vertices_source_world", "faces", "face_owner_privileged"}
        if not required.issubset(z.files):
            raise ValueError("STRUCTURAL_SWEEP_PACKET_GEOMETRY_FIELDS")
        return geometry_path, (
            np.asarray(z["rest_vertices_source_world"], dtype=np.float64),
            np.asarray(z["faces"], dtype=np.int64),
            np.asarray(z["face_owner_privileged"], dtype=np.int32),
        )


def stage47_geometry(scene: dict):
    geometry_path = bound_ref(
        teacher_geometry_ref(scene), "STRUCTURAL_SWEEP_STAGE47_GEOMETRY_DRIFT"
    )
    with np.load(geometry_path, allow_pickle=False) as z:
        required = {
            "rest_vertices_source_world",
            "faces",
            "face_owner_privileged",
            "isolated_position",
        }
        if not required.issubset(z.files):
            raise ValueError("STRUCTURAL_SWEEP_STAGE47_FIELDS")
        return geometry_path, (
            np.asarray(z["rest_vertices_source_world"], dtype=np.float64),
            np.asarray(z["faces"], dtype=np.int64),
            np.asarray(z["face_owner_privileged"], dtype=np.int32),
            np.asarray(z["isolated_position"], dtype=np.float64),
        )


def exact_rest_binding(packet_arrays, stage47_arrays) -> dict:
    pv, pf, po = packet_arrays
    sv, sf, so, posed = stage47_arrays
    layout = bool(
        pv.shape == sv.shape
        and pf.shape == sf.shape
        and po.shape == so.shape
        and posed.ndim == 3
        and posed.shape[1:] == pv.shape
    )
    vertices_equal = bool(layout and np.array_equal(pv, sv))
    faces_equal = bool(layout and np.array_equal(pf, sf))
    owner_equal = bool(layout and np.array_equal(po, so))
    return {
        "layout_match": layout,
        "rest_vertices_exact_match": vertices_equal,
        "faces_exact_match": faces_equal,
        "face_owner_exact_match": owner_equal,
        "stage47_action_count": int(len(posed)) if posed.ndim == 3 else 0,
        "pass": bool(layout and vertices_equal and faces_equal and owner_equal and len(posed) > 0),
    }


def structural_evidence_classes(audit: dict, rest_binding: dict) -> list[str]:
    counts = dict(audit["counts"])
    seam = dict(audit["cross_owner_boundary_evidence"])
    classes = []
    if counts["within_owner_exact_duplicate_collapse_count"] > 0:
        classes.append("WITHIN_OWNER_RENDER_SEAM_EVIDENCE")
    if counts["owner_count"] > 1:
        classes.append("MULTI_OWNER_AUTHORED_ASSEMBLY_EVIDENCE")
    if counts["cross_owner_exact_position_group_count"] > 0:
        classes.append("CROSS_OWNER_EXACT_COINCIDENCE_EVIDENCE")
    if counts["cross_owner_matched_boundary_edge_count"] > 0:
        if seam["trajectory_incompatible_count"] == 0 and seam["trajectory_unknown_count"] == 0:
            classes.append("CROSS_OWNER_MECHANICALLY_COHERENT_SEAM_EVIDENCE")
        else:
            classes.append("CROSS_OWNER_BOUNDARY_SEAM_AMBIGUOUS")
    if counts["boundary_edge_count"] > 0:
        classes.append("OPEN_BOUNDARY_EVIDENCE")
    if counts["nonmanifold_edge_count"] > 0:
        classes.append("NONMANIFOLD_EDGE_EVIDENCE")
    if counts["orientation_conflict_edge_count"] > 0:
        classes.append("ORIENTATION_CONFLICT_EVIDENCE")
    if (
        counts["raw_zero_area_face_count"] > 0
        or counts["quotient_collapsed_face_count"] > 0
        or counts["quotient_zero_area_face_count"] > 0
    ):
        classes.append("DEGENERATE_FACE_EVIDENCE")
    if audit["diagnostics"]["point_only_contact_detected"]:
        classes.append("POINT_ONLY_CONTACT_EVIDENCE")
    if not rest_binding["pass"]:
        classes.append("STAGE47_REST_BINDING_MISMATCH")
    return classes


def compact_row(subject_id: str, audit: dict, rest_binding: dict, refs: dict) -> dict:
    closed_components = sum(
        1
        for row in audit["edge_components"]
        if row["owner_preserving_closed_sign_candidate"]
    )
    total_components = len(audit["edge_components"])
    evidence = structural_evidence_classes(audit, rest_binding)
    return {
        "schema": ROW_SCHEMA,
        "subject_id": subject_id,
        "candidate_disposition": "PENDING_AUTHORITY_AUDIT",
        "training_authorized": False,
        "source_mutated": False,
        "lineage": refs,
        "stage47_rest_binding": rest_binding,
        "counts": audit["counts"],
        "closed_sign_candidate_component_count": int(closed_components),
        "surface_component_count": int(total_components),
        "closed_sign_candidate_fraction": (
            float(closed_components / total_components) if total_components else 0.0
        ),
        "structural_evidence_classes": evidence,
        "audit_identity_sha256": audit["audit_identity_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-readiness", type=Path, required=True)
    parser.add_argument("--response-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    prior = load_json(args.prior_readiness, "STRUCTURAL_SWEEP_PRIOR_READINESS")
    response = load_json(args.response_receipt, "STRUCTURAL_SWEEP_RESPONSE_RECEIPT")
    if int(prior.get("subject_count", -1)) != 139:
        raise ValueError("STRUCTURAL_SWEEP_PRIOR_NOT_139")
    if int(response.get("completed_subject_count", -1)) != 139:
        raise ValueError("STRUCTURAL_SWEEP_RESPONSE_NOT_139")

    subjects = dict(prior.get("subjects") or {})
    response_bindings = dict(response.get("subject_output_bindings") or {})
    if len(subjects) != 139 or set(subjects) != set(response_bindings):
        raise ValueError("STRUCTURAL_SWEEP_SUBJECT_SET_DRIFT")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: dict[str, str] = {}

    for index, subject_id in enumerate(sorted(subjects), 1):
        print(f"[STRUCTURAL-AUDIT {index:03d}/139] {subject_id}", flush=True)
        try:
            binding = dict(response_bindings[subject_id])
            chain_path = Path(str(binding.get("chain_path") or "")).resolve()
            expected_chain = str(binding.get("chain_sha256") or "")
            if not chain_path.is_file() or sha256(chain_path) != expected_chain:
                raise ValueError("STRUCTURAL_SWEEP_RESPONSE_CHAIN_DRIFT:" + subject_id)
            chain = load_json(chain_path, "STRUCTURAL_SWEEP_RESPONSE_CHAIN")

            packet_root = Path(
                str((binding.get("prepared_packet_lineage") or {}).get("packet_root") or "")
            ).resolve()
            packet_path, packet_arrays = packet_geometry(packet_root)
            teacher_scene_path = bound_ref(
                dict((chain.get("outputs") or {}).get("teacher_scene") or {}),
                "STRUCTURAL_SWEEP_TEACHER_SCENE_DRIFT:" + subject_id,
            )
            scene = load_json(teacher_scene_path, "STRUCTURAL_SWEEP_TEACHER_SCENE")
            if scene.get("schema") != "RealSaS.TeacherSceneIR.v2" or scene.get("subject_key") != subject_id:
                raise ValueError("STRUCTURAL_SWEEP_TEACHER_SCENE_BINDING:" + subject_id)
            stage47_path, stage47_arrays = stage47_geometry(scene)
            rest_binding = exact_rest_binding(packet_arrays, stage47_arrays)
            if not rest_binding["pass"]:
                raise ValueError("STRUCTURAL_SWEEP_STAGE47_REST_BINDING:" + subject_id)

            vertices, faces, owner = packet_arrays
            posed = stage47_arrays[3]
            audit = audit_structural_geometry_v1(
                vertices,
                faces,
                owner,
                posed_vertices=posed,
                subject_id=subject_id,
            )
            refs = {
                "response_chain": plain_ref(chain_path),
                "packet_geometry": plain_ref(packet_path),
                "teacher_scene": plain_ref(teacher_scene_path),
                "stage47_geometry_and_response": plain_ref(stage47_path),
            }
            row = compact_row(subject_id, audit, rest_binding, refs)
            detail = {
                "schema": ROW_SCHEMA,
                "subject_id": subject_id,
                "row": row,
                "structural_audit": audit,
            }
            detail["detail_identity_sha256"] = digest(detail)
            subject_out = output_root / "subjects" / subject_id / "STRUCTURAL_GEOMETRY_AUDIT_V1.json"
            atomic_json(subject_out, detail)
            row["detail"] = plain_ref(subject_out)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "subject_id": subject_id,
                        "components": row["surface_component_count"],
                        "closed_candidates": row["closed_sign_candidate_component_count"],
                        "boundary_edges": row["counts"]["boundary_edge_count"],
                        "nonmanifold_edges": row["counts"]["nonmanifold_edge_count"],
                        "cross_owner_seam_edges": row["counts"]["cross_owner_matched_boundary_edge_count"],
                        "evidence": row["structural_evidence_classes"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            failures[subject_id] = repr(exc)
            print(f"[STRUCTURAL-AUDIT-FAIL] {subject_id} {exc!r}", flush=True)
            if not args.continue_on_failure:
                break
        atomic_json(
            output_root / "STRUCTURAL_AUDIT_PROGRESS.json",
            {
                "schema": "RealSaS.Truth139StructuralIntegrityProgress.v1",
                "completed_subject_count": len(rows),
                "failed_subject_count": len(failures),
                "failures": failures,
                "training_authorized": False,
                "updated_at_utc": utc(),
            },
        )

    evidence_histogram = Counter()
    for row in rows:
        evidence_histogram.update(row["structural_evidence_classes"])
    closed_all = sum(
        1
        for row in rows
        if row["surface_component_count"] > 0
        and row["closed_sign_candidate_component_count"] == row["surface_component_count"]
    )
    some_closed = sum(1 for row in rows if row["closed_sign_candidate_component_count"] > 0)
    cross_owner_coherent = sum(
        1
        for row in rows
        if "CROSS_OWNER_MECHANICALLY_COHERENT_SEAM_EVIDENCE" in row["structural_evidence_classes"]
    )

    table_path = output_root / "TRUTH139_STRUCTURAL_AUDIT_TABLE_V1.jsonl"
    atomic_jsonl(table_path, rows)
    summary = {
        "schema": SCHEMA,
        "status": STATUS if len(rows) == 139 and not failures else "STRUCTURAL_AUDIT_INCOMPLETE",
        "subject_count_expected": 139,
        "subject_count_completed": len(rows),
        "failed_subject_count": len(failures),
        "failures": failures,
        "authority": {
            "training_authorized": False,
            "sign_authority_minted": False,
            "final_disposition_minted": False,
            "historical_split_changed": False,
            "purpose": "STRUCTURAL_EVIDENCE_ONLY",
        },
        "corpus_counts": {
            "all_surface_components_closed_candidate_subject_count": int(closed_all),
            "at_least_one_closed_candidate_component_subject_count": int(some_closed),
            "cross_owner_mechanically_coherent_seam_subject_count": int(cross_owner_coherent),
            "total_surface_component_count": int(sum(row["surface_component_count"] for row in rows)),
            "total_closed_sign_candidate_component_count": int(
                sum(row["closed_sign_candidate_component_count"] for row in rows)
            ),
            "total_boundary_edge_count": int(sum(row["counts"]["boundary_edge_count"] for row in rows)),
            "total_nonmanifold_edge_count": int(sum(row["counts"]["nonmanifold_edge_count"] for row in rows)),
            "total_cross_owner_matched_boundary_edge_count": int(
                sum(row["counts"]["cross_owner_matched_boundary_edge_count"] for row in rows)
            ),
        },
        "structural_evidence_histogram": dict(sorted(evidence_histogram.items())),
        "table": plain_ref(table_path),
        "prior_readiness": plain_ref(args.prior_readiness),
        "response_receipt": plain_ref(args.response_receipt),
        "completed_at_utc": utc(),
    }
    summary["audit_identity_sha256"] = digest(summary)
    atomic_json(output_root / "TRUTH139_FULL_STRUCTURAL_AUDIT_V1.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)

    if failures or len(rows) != 139:
        raise RuntimeError(
            f"TRUTH139_STRUCTURAL_AUDIT_INCOMPLETE:{len(rows)}/139 failures={len(failures)}"
        )


if __name__ == "__main__":
    main()
