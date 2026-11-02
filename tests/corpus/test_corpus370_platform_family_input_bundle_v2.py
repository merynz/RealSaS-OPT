from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from tools.corpus.prepare_corpus370_platform_truth_bundle_v2 import prepare


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def test_full_candidate_universe_survives_partial_observation_preparation(tmp_path: Path):
    extracted = tmp_path / "source"
    extracted.mkdir()
    provenance_root = tmp_path / "provenance"
    failures_root = tmp_path / "failures"
    packet = tmp_path / "packet-pass"
    packet.mkdir()

    roster_rows = []
    source_rows = []
    provenance_rows = []
    failures = []

    license_file = tmp_path / "LICENSE.txt"
    license_file.write_text("CC0 test bytes\n")
    source_page = tmp_path / "SOURCE_PAGE.snapshot"
    source_page.write_text("source page bytes\n")

    for index in range(370):
        sid = f"subject_{index:03d}"
        rel = f"assets/{sid}.glb"
        source = extracted / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"glb-{sid}".encode())
        source_sha = _sha(source)
        roster_rows.append(
            {
                "subject_id": sid,
                "source_relative_path": rel,
                "source_sha256": source_sha,
                "source_format": ".glb",
                "source_id": "synthetic-pack",
                "origin": "synthetic",
            }
        )
        source_rows.append(
            {
                "subject_id": sid,
                "source_relative_path": rel,
                "source_sha256": source_sha,
            }
        )
        provenance_rows.append(
            {
                "subject_id": sid,
                "source_sha256": source_sha,
                "source_pack": "synthetic-pack",
                "display_name": sid,
                "source_url_or_locator": "urn:test:" + sid,
                "license_name": "CC0-1.0",
                "license_ref": "https://creativecommons.org/publicdomain/zero/1.0/",
                "license_bytes": {"path": str(license_file), "sha256": _sha(license_file)},
                "source_page_snapshot": {"path": str(source_page), "sha256": _sha(source_page)},
            }
        )
        if index:
            failure = _write_json(
                failures_root / f"{sid}.json",
                {
                    "subject_id": sid,
                    "source_sha256": source_sha,
                    "error_type": "ValueError",
                    "reason": "ADMISSION_MASK_DRIFT:test",
                    "training_authorized": False,
                },
            )
            failures.append(
                {
                    "subject_id": sid,
                    "source_sha256": source_sha,
                    "source_format": ".glb",
                    "source_id": "synthetic-pack",
                    "error_type": "ValueError",
                    "reason": "ADMISSION_MASK_DRIFT:test",
                    "receipt": {"path": str(failure), "sha256": _sha(failure)},
                }
            )

    pass_sid = "subject_000"
    pass_sha = roster_rows[0]["source_sha256"]
    required = [
        "SOURCE_REST_RENDER_CANDIDATE.json",
        "SOURCE_SCENE_SNAPSHOT.json",
        "camera_bundle.json",
        "IRIS_OBSERVATION_ADMISSION.json",
        *[f"V{i}.png" for i in range(8)],
        *[f"V{i}.mask.bin" for i in range(8)],
    ]
    for name in required:
        path = packet / name
        if name == "IRIS_OBSERVATION_ADMISSION.json":
            _write_json(
                path,
                {
                    "schema": "RealSaS.IrisBindObservationAdmission.v1",
                    "status": "PASS_IRIS_BIND_OBSERVATIONS",
                    "subject_id": pass_sid,
                    "source_sha256": pass_sha,
                },
            )
        elif name.endswith(".json"):
            _write_json(path, {"schema": "test"})
        else:
            path.write_bytes((name + " bytes").encode())

    roster = _write_json(
        tmp_path / "roster.json",
        {
            "schema": "RealSaS.Corpus370ExactSourceRoster.v1",
            "subject_count": 370,
            "subjects": roster_rows,
            "training_authorized": False,
        },
    )
    source_manifest = _write_json(
        tmp_path / "source_manifest.json",
        {
            "schema": "RealSaS.ExpandedCorpusSourceInputBundle.v1",
            "subject_count": 370,
            "subjects": source_rows,
            "files": [],
            "training_authorized": False,
        },
    )
    provenance = _write_json(
        tmp_path / "provenance.json",
        {
            "schema": "RealSaS.Corpus370SourceProvenanceIndex.v1",
            "subject_count": 370,
            "subjects": provenance_rows,
            "training_authorized": False,
        },
    )
    observations = _write_json(
        tmp_path / "observations.json",
        {
            "schema": "RealSaS.Corpus370PreparedObservationPackets.v3",
            "subject_count": 370,
            "prepared_subject_count": 1,
            "failure_count": 369,
            "subjects": [
                {
                    "subject_id": pass_sid,
                    "source_sha256": pass_sha,
                    "packet_path": str(packet),
                    "packet_schema": "test.packet",
                    "packet_identity": "test-identity",
                }
            ],
            "failures": failures,
            "training_authorized": False,
        },
    )

    output = tmp_path / "bundle.zip"
    manifest, receipt, archive = prepare(
        roster,
        source_manifest,
        extracted,
        observations,
        provenance,
        output,
    )

    assert manifest["candidate_subject_count"] == 370
    assert manifest["observation_qualified_count"] == 1
    assert manifest["observation_failed_count"] == 369
    assert len(manifest["subjects"]) == 370
    by_id = {row["subject_id"]: row for row in manifest["subjects"]}
    assert by_id[pass_sid]["observation_preparation"]["status"] == "PASS"
    assert by_id["subject_001"]["observation_preparation"]["status"] == "FAIL"
    assert by_id["subject_001"]["raw_source"]["sha256"] == roster_rows[1]["source_sha256"]
    assert manifest["training_authorized"] is False
    assert archive["sha256"] == _sha(output)

    with zipfile.ZipFile(output) as zipped:
        assert "BOOTSTRAP_MANIFEST.json" in zipped.namelist()
        assert "observations/subject_001/PREPARATION_FAILURE.json" in zipped.namelist()
        assert "observations/subject_000/IRIS_OBSERVATION_ADMISSION.json" in zipped.namelist()
        assert len(receipt["files"]) + 1 == len(zipped.namelist())
