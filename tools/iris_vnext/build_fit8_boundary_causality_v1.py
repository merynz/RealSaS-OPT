from __future__ import annotations

"""Build source-defensible Fit8 Boundary Causality V1 evidence.

This producer intentionally separates structural source-mesh causes from
view-conditioned projected causes. It never infers drawable ownership,
amodal support, hidden continuation, or semantic body-part classes.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.boundary_causality_v1 import (
    BoundaryCause,
    BoundaryCausalityTeacherV1,
    BoundaryEvidenceV1,
    EvidenceAuthority,
)
from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.build_fit8_vnext_shards_v1 import SUBJECT_SOURCE_IDS, _sha

SCHEMA = "RealSaS.IRISVNextFit8BoundaryCausalityCorpus.v1"
MANIFEST_SCHEMA = "RealSaS.IRISVNextFit8BoundaryCausalityManifest.v1"
VIEW_SCHEMA = "RealSaS.IRISVNextFit8BoundaryCausalityViewEvidence.v1"


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--stage01-root", type=Path, required=True)
    p.add_argument("--relation-truth-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _source_provenance() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repo / "models/iris/vnext/boundary_causality_v1.py",
        repo / "tools/iris_vnext/build_fit8_relation_truth_v2.py",
        repo / "tools/iris_vnext/blender_authored_teacher_export_v3.py",
    )
    return {str(path.relative_to(repo)): _sha(path) for path in paths}


def _edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        if face.shape != (3,):
            raise ValueError("VNEXT_BOUNDARY_CAUSALITY_FACE_SHAPE_INVALID")
        a, b, c = map(int, face)
        if min(a, b, c) < 0 or len({a, b, c}) != 3:
            raise ValueError("VNEXT_BOUNDARY_CAUSALITY_FACE_INVALID")
        for x, y in ((a, b), (b, c), (c, a)):
            edge = (x, y) if x < y else (y, x)
            result[edge].append(face_index)
    return result


def _static_rows(
    faces: np.ndarray,
    face_owner: np.ndarray,
    face_material: np.ndarray,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    faces = np.asarray(faces, dtype=np.int64)
    owners = np.asarray(face_owner, dtype=np.int64).reshape(-1)
    materials = np.asarray(face_material, dtype=np.int64).reshape(-1)
    if len(owners) != len(faces) or len(materials) != len(faces):
        raise ValueError("VNEXT_BOUNDARY_CAUSALITY_FACE_METADATA_SHAPE_DRIFT")

    teacher = BoundaryCausalityTeacherV1()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for edge, incident in sorted(_edge_incidence(faces).items()):
        face_ids = tuple(sorted(set(map(int, incident))))
        evidence: list[BoundaryEvidenceV1] = []
        if len(face_ids) == 1:
            evidence.append(
                BoundaryEvidenceV1(
                    BoundaryCause.GEOMETRIC_TERMINATION,
                    EvidenceAuthority.DETERMINISTIC_DERIVED,
                    "SOURCE_TRIANGLE_EDGE_INCIDENCE_COUNT_EQ_1",
                )
            )
        material_ids = tuple(sorted({int(materials[i]) for i in face_ids}))
        if len(material_ids) > 1:
            evidence.append(
                BoundaryEvidenceV1(
                    BoundaryCause.MATERIAL_BOUNDARY,
                    EvidenceAuthority.DETERMINISTIC_DERIVED,
                    "AUTHORED_FACE_MATERIAL_INDEX_CHANGES_ACROSS_SHARED_SOURCE_EDGE",
                )
            )
        owner_ids = tuple(sorted({int(owners[i]) for i in face_ids}))
        if len(owner_ids) > 1:
            evidence.append(
                BoundaryEvidenceV1(
                    BoundaryCause.SOURCE_PRIMITIVE_COMPONENT_SEAM,
                    EvidenceAuthority.DETERMINISTIC_DERIVED,
                    "ANONYMOUS_SOURCE_MESH_OWNER_CHANGES_ACROSS_SHARED_INDEX_EDGE",
                )
            )
        qualified = teacher.qualify(evidence)
        for cause in qualified.positive_causes:
            counts[cause.value] += 1
        if qualified.abstained:
            counts["ABSTAINED"] += 1
        rows.append(
            {
                "edge_vertex_ids": list(edge),
                "incident_face_ids": list(face_ids),
                "owner_ids": list(owner_ids),
                "material_ids": list(material_ids),
                "positive_causes": [cause.value for cause in qualified.positive_causes],
                "abstained": qualified.abstained,
                "evidence": [
                    {
                        "cause": item.cause.value,
                        "authority": item.authority.value,
                        "provenance": item.provenance,
                    }
                    for item in qualified.evidence
                ],
            }
        )
    return rows, counts


def _relation_filename(display: str) -> str:
    stem = display.lower().replace("-", "").replace(" ", "")
    return f"{stem}.relation_truth_v2.pt"


def _view_sidecar(
    relation_payload: dict[str, Any],
) -> tuple[dict[str, Any], Counter[str]]:
    views_out = []
    counts: Counter[str] = Counter()
    for view in relation_payload.get("views") or ():
        truth = view.get("view_truth") or {}
        valid = view.get("view_validity") or {}
        alpha = torch.as_tensor(
            truth["projected_alpha_boundary_probability"], dtype=torch.float32
        ).reshape(-1)
        alpha_valid = torch.as_tensor(
            valid["projected_alpha_boundary_probability"], dtype=torch.bool
        ).reshape(-1)
        occ = torch.as_tensor(
            truth["occlusion_probability"], dtype=torch.float32
        ).reshape(-1)
        occ_valid = torch.as_tensor(
            valid["occlusion_probability"], dtype=torch.bool
        ).reshape(-1)
        if not (len(alpha) == len(alpha_valid) == len(occ) == len(occ_valid)):
            raise ValueError("VNEXT_BOUNDARY_CAUSALITY_VIEW_LENGTH_DRIFT")
        alpha_positive = alpha_valid & (alpha >= 0.5)
        occ_positive = occ_valid & (occ >= 0.5)
        counts[BoundaryCause.ALPHA_BOUNDARY.value] += int(alpha_positive.sum().item())
        counts[BoundaryCause.OCCLUSION_VISIBILITY_BOUNDARY.value] += int(
            occ_positive.sum().item()
        )
        views_out.append(
            {
                "view_index": int(view["view_index"]),
                "point_a": torch.as_tensor(view["point_a"]).detach().cpu(),
                "point_b": torch.as_tensor(view["point_b"]).detach().cpu(),
                "alpha_boundary_positive": alpha_positive.detach().cpu(),
                "occlusion_visibility_boundary_positive": occ_positive.detach().cpu(),
                "alpha_boundary_authority": EvidenceAuthority.DETERMINISTIC_DERIVED.value,
                "occlusion_visibility_boundary_authority": EvidenceAuthority.DETERMINISTIC_DERIVED.value,
                "provenance": {
                    "alpha_boundary": "RELATION_TRUTH_V2_CURRENT_VIEW_SOURCE_RGBA_PROJECTED_ALPHA_BOUNDARY",
                    "occlusion_visibility_boundary": "RELATION_TRUTH_V2_EXACT_SAME_PIXEL_TWO_HIT_RAY_PAIR",
                },
            }
        )
    return {
        "schema": VIEW_SCHEMA,
        "subject_id": relation_payload.get("subject_id"),
        "source_subject_id": relation_payload.get("source_subject_id"),
        "teacher_inputs_in_inference_forward": False,
        "views": views_out,
    }, counts


def main() -> int:
    args = _args()
    teacher_dir = args.teacher_dir.resolve()
    stage01 = args.stage01_root.resolve()
    relation_dir = args.relation_truth_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bootstrap_path = stage01 / "BOOTSTRAP_MANIFEST.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_rows = {row["subject_id"]: row for row in bootstrap["subjects"]}
    relation_manifest_path = relation_dir / "RELATION_TRUTH_V2_MANIFEST.json"
    relation_manifest = json.loads(
        relation_manifest_path.read_text(encoding="utf-8")
    )
    if relation_manifest.get("schema") != "RealSaS.IRISVNextFit8RelationTruthManifest.v2":
        raise ValueError("VNEXT_BOUNDARY_CAUSALITY_RELATION_MANIFEST_SCHEMA_INVALID")

    manifest_subjects = []
    total_causes: Counter[str] = Counter()
    for display in SUBJECT_IDS:
        source_id = SUBJECT_SOURCE_IDS[display]
        boot = bootstrap_rows[source_id]
        teacher_path = teacher_dir / f"{source_id}.teacher.npz"
        teacher_receipt_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
        relation_path = relation_dir / _relation_filename(display)
        for path in (teacher_path, teacher_receipt_path, relation_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"VNEXT_BOUNDARY_CAUSALITY_INPUT_MISSING:{display}:{path.name}"
                )

        teacher_receipt = json.loads(
            teacher_receipt_path.read_text(encoding="utf-8")
        )
        if teacher_receipt["source"]["sha256"] != boot["raw_source"]["sha256"]:
            raise ValueError(
                "VNEXT_BOUNDARY_CAUSALITY_TEACHER_SOURCE_HASH_DRIFT:" + display
            )
        with np.load(teacher_path, allow_pickle=False) as data:
            static_rows, static_counts = _static_rows(
                np.asarray(data["faces"], dtype=np.int64),
                np.asarray(data["face_owner"], dtype=np.int64),
                np.asarray(data["face_material"], dtype=np.int64),
            )

        static_path = output / (
            display.lower().replace("-", "").replace(" ", "")
            + ".boundary_causality_v1.jsonl"
        )
        with static_path.open("w", encoding="utf-8") as handle:
            for row in static_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

        relation_payload = torch.load(
            relation_path, map_location="cpu", weights_only=True
        )
        if relation_payload.get("subject_id") != display:
            raise ValueError(
                "VNEXT_BOUNDARY_CAUSALITY_RELATION_SUBJECT_DRIFT:" + display
            )
        view_payload, view_counts = _view_sidecar(relation_payload)
        view_path = output / (
            display.lower().replace("-", "").replace(" ", "")
            + ".boundary_causality_view_v1.pt"
        )
        torch.save(view_payload, view_path)

        subject_counts = static_counts + view_counts
        total_causes.update(subject_counts)
        manifest_subjects.append(
            {
                "subject_id": display,
                "source_subject_id": source_id,
                "source_asset_sha256": boot["raw_source"]["sha256"],
                "teacher_export_sha256": _sha(teacher_path),
                "teacher_receipt_sha256": _sha(teacher_receipt_path),
                "relation_truth_v2_sha256": _sha(relation_path),
                "static_file": static_path.name,
                "static_file_sha256": _sha(static_path),
                "view_file": view_path.name,
                "view_file_sha256": _sha(view_path),
                "static_edge_rows": len(static_rows),
                "cause_counts": dict(sorted(subject_counts.items())),
            }
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "corpus_schema": SCHEMA,
        "builder": "tools/iris_vnext/build_fit8_boundary_causality_v1.py",
        "source_provenance": _source_provenance(),
        "stage01_bootstrap_sha256": _sha(bootstrap_path),
        "relation_truth_v2_manifest_sha256": _sha(relation_manifest_path),
        "relation_truth_v2_manifest_value_sha256": relation_manifest.get(
            "manifest_value_sha256"
        ),
        "subject_order": list(SUBJECT_IDS),
        "subjects": manifest_subjects,
        "cause_counts": dict(sorted(total_causes.items())),
        "authority_policy": {
            "structural_source_mesh": EvidenceAuthority.DETERMINISTIC_DERIVED.value,
            "view_projected": EvidenceAuthority.DETERMINISTIC_DERIVED.value,
            "unknown": "ABSTAIN",
        },
        "forbidden_inferences": [
            "DRAWABLE_OWNER_FROM_GENERIC_FIT8",
            "AMODAL_HIDDEN_SUPPORT_FROM_GENERIC_FIT8",
            "INTENTIONAL_2D_PRESENTATION_OVERRIDE_FROM_GENERIC_FIT8",
            "GLOBAL_NAMED_SEMANTIC_TAXONOMY_FROM_GENERIC_FIT8",
        ],
        "teacher_inputs_in_inference_forward": False,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
    }
    manifest["manifest_value_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = output / "BOUNDARY_CAUSALITY_V1_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "VNEXT_FIT8_BOUNDARY_CAUSALITY_V1_PASS",
        "subjects=8",
        f"manifest={manifest['manifest_value_sha256']}",
        "scientific_fit_started=false",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
