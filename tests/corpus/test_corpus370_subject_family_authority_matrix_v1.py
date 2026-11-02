import hashlib
import json
from pathlib import Path

from tools.corpus.build_corpus370_subject_family_authority_matrix_v1 import build
from tools.corpus.derive_corpus370_model_rosters_v1 import derive


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _candidate_manifest():
    return {
        "subject_count": 370,
        "subjects": [
            {
                "subject_id": f"corpus_{index:03d}",
                "origin": "ESTABLISHED_BASE" if index < 139 else "EXPANSION",
                "source_id": "fixture",
                "source_format": ".gltf",
                "source_relative_path": f"assets/{index}.gltf",
                "source_sha256": _sha(f"source-{index}"),
            }
            for index in range(370)
        ],
    }


def _claim(status: str, tag: str, reason_codes=None):
    court_status = "FAIL" if status == "FAIL" else "PASS"
    return {
        "status": status,
        "reason_codes": list(reason_codes or []),
        "evidence_refs": [
            {
                "schema": "FixtureEvidence.v1",
                "sha256": _sha("evidence-" + tag),
                "path": f"evidence/{tag}.json",
            }
        ],
        "court": {
            "schema": "FixtureCourt.v1",
            "sha256": _sha("court-" + tag),
            "path": f"courts/{tag}.json",
            "status": court_status,
        },
    }


def test_missing_evidence_is_unknown_and_never_eligible(tmp_path: Path):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(_candidate_manifest()))
    matrix_path = tmp_path / "matrix.json"
    matrix = build(
        Path("canonical/CORPUS370_SUBJECT_FAMILY_AUTHORITY_POLICY_V1.json"),
        candidates,
        [],
    )
    matrix_path.write_text(json.dumps(matrix))
    assert matrix["rollup"]["unknown_cell_count"] == 370 * 14
    assert all(values["blocked"] == 370 for values in matrix["rollup"]["models"].values())
    rosters = derive(matrix_path)
    assert all(row["eligible_count"] == 0 for row in rosters["models"].values())
    assert rosters["training_authorized"] is False


def test_iris_can_use_masked_global_sdf_without_zero_target(tmp_path: Path):
    candidate_doc = _candidate_manifest()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(candidate_doc))
    first = candidate_doc["subjects"][0]
    evidence = {
        "schema": "RealSaS.Corpus370FamilyEvidenceIndex.v1",
        "subjects": [
            {
                "subject_id": first["subject_id"],
                "source_sha256": first["source_sha256"],
                "families": {
                    "SOURCE_IMPORT": _claim("PASS", "source"),
                    "CAMERA_OBSERVATION": _claim("PASS", "camera"),
                    "SURFACE_GEOMETRY_UDF": _claim("PASS", "udf"),
                    "GLOBAL_SOLID_SDF": _claim("MASKED", "sdf", ["NO_QUALIFIED_EXPLICIT_SOLID"]),
                },
            }
        ],
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    matrix = build(
        Path("canonical/CORPUS370_SUBJECT_FAMILY_AUTHORITY_POLICY_V1.json"),
        candidates,
        [evidence_path],
    )
    row = next(x for x in matrix["subjects"] if x["subject_id"] == first["subject_id"])
    assert row["model_eligibility"]["IRIS"]["corpus_eligible"] is True
    assert row["families"]["GLOBAL_SOLID_SDF"]["status"] == "MASKED"
    assert row["model_eligibility"]["TESSA"]["corpus_eligible"] is False
    assert matrix["training_authorized"] is False


def test_conflicting_current_claims_quarantine_cell(tmp_path: Path):
    candidate_doc = _candidate_manifest()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(candidate_doc))
    first = candidate_doc["subjects"][0]
    paths = []
    for status in ("PASS", "FAIL"):
        doc = {
            "schema": "RealSaS.Corpus370FamilyEvidenceIndex.v1",
            "subjects": [
                {
                    "subject_id": first["subject_id"],
                    "source_sha256": first["source_sha256"],
                    "families": {
                        "SOURCE_IMPORT": _claim(
                            status,
                            status.lower(),
                            [] if status == "PASS" else ["FIXTURE_FAILURE"],
                        )
                    },
                }
            ],
        }
        path = tmp_path / f"{status}.json"
        path.write_text(json.dumps(doc))
        paths.append(path)
    matrix = build(
        Path("canonical/CORPUS370_SUBJECT_FAMILY_AUTHORITY_POLICY_V1.json"),
        candidates,
        paths,
    )
    row = next(x for x in matrix["subjects"] if x["subject_id"] == first["subject_id"])
    assert row["families"]["SOURCE_IMPORT"]["status"] == "QUARANTINED"
    assert "CONFLICTING_CURRENT_AUTHORITY_CLAIMS" in row["families"]["SOURCE_IMPORT"]["reason_codes"]
