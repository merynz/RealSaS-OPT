from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.corpus.build_corpus370_rig_skin_evidence_v1 import build


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path, schema: str | None = None) -> dict:
    row = {"path": str(path.resolve()), "sha256": sha256(path)}
    if schema:
        row["schema"] = schema
    return row


def roster(tmp_path: Path):
    rows = []
    sources = []
    for index in range(370):
        source = tmp_path / f"source_{index}.{'fbx' if index == 1 else 'gltf'}"
        source.write_bytes(f"source-{index}".encode())
        sources.append(source)
        rows.append(
            {
                "subject_id": f"subject_{index:03d}",
                "origin": "ESTABLISHED_BASE" if index < 139 else "EXPANSION",
                "source_id": "fixture",
                "source_format": source.suffix,
                "source_relative_path": source.name,
                "source_sha256": sha256(source),
            }
        )
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({"schema": "RealSaS.Corpus370ExactSourceRoster.v1", "subject_count": 370, "subjects": rows, "training_authorized": False}))
    return path, rows, sources


def test_unifies_gltf_core_and_fbx_row_masked_skin_as_pass(tmp_path: Path):
    roster_path, rows, sources = roster(tmp_path)

    gltf = tmp_path / "gltf_court.json"
    gltf.write_text(json.dumps({
        "schema": "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1",
        "status": "PASS_GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE__FINAL_ADMISSION_SEPARATE",
        "subject_id": rows[0]["subject_id"],
        "inputs": {"raw_source": ref(sources[0])},
        "family_candidate_outcomes": {
            "RIG_HIERARCHY": "QUALIFIED_CORE_CANDIDATE",
            "SKIN_WEIGHTS": "QUALIFIED_CORE_CANDIDATE",
        },
    }))

    fbx = tmp_path / "fbx_court.json"
    fbx.write_text(json.dumps({
        "schema": "RealSaS.FbxImportedRigSkinAuthorityCourt.v1",
        "status": "PASS_FBX_IMPORTED_RIG_SKIN_DECODE_EQUIVALENCE__FINAL_ADMISSION_SEPARATE",
        "subject_id": rows[1]["subject_id"],
        "source": ref(sources[1]),
        "authority_scope": "PINNED_BLENDER_FBX_DECODED_SOURCE_SEMANTICS__NOT_RAW_FBX_PARSER_PROOF",
        "family_candidate_outcomes": {
            "RIG_HIERARCHY": "QUALIFIED_CORE_CANDIDATE",
            "SKIN_WEIGHTS": "QUALIFIED_MASKED_CANDIDATE",
        },
        "skin_quality": {
            "simplex_valid_weight_coverage_fraction": 0.75,
            "authored_weight_coverage_fraction": 0.8,
        },
    }))

    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "schema": "RealSaS.Corpus370RigSkinCourtIndex.v1",
        "subject_count": 2,
        "subjects": [
            {
                "subject_id": rows[0]["subject_id"],
                "source_sha256": rows[0]["source_sha256"],
                "lane": "GLTF2_RAW_PLUS_IMPORT_EQUIVALENCE",
                "court": ref(gltf, "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1"),
            },
            {
                "subject_id": rows[1]["subject_id"],
                "source_sha256": rows[1]["source_sha256"],
                "lane": "FBX_PINNED_IMPORT_REPLAY_EQUIVALENCE",
                "court": ref(fbx, "RealSaS.FbxImportedRigSkinAuthorityCourt.v1"),
            },
        ],
        "training_authorized": False,
    }))

    result = build(roster_path, index)
    by_id = {row["subject_id"]: row for row in result["subjects"]}
    gltf_skin = by_id[rows[0]["subject_id"]]["families"]["SKIN_WEIGHTS"]
    fbx_skin = by_id[rows[1]["subject_id"]]["families"]["SKIN_WEIGHTS"]
    assert gltf_skin["status"] == "PASS"
    assert gltf_skin["metadata"]["row_validity_mask_required"] is False
    assert fbx_skin["status"] == "PASS"
    assert fbx_skin["metadata"]["row_validity_mask_required"] is True
    assert fbx_skin["metadata"]["valid_row_coverage_fraction"] == 0.75
    assert result["training_authorized"] is False
