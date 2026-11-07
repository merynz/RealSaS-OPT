"""Private-batch fixture tests: exactly 3×2 pixels, exact SHA, no owner labels."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from tools.corpus.court_corpus370_source_locus_visibility_v1 import qualify
from tools.corpus.qualify_corpus370_visibility_oracle_batch_v1 import run


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture():
    return runpy.run_path(str(Path(__file__).with_name(
        "test_corpus370_source_locus_visibility_v1.py")))["fixture"]()


def _build(tmp_path):
    source_dir = tmp_path/"source"
    visibility_dir = tmp_path/"visibility"
    result_dir = tmp_path/"result"
    ids = ["synthetic_0", "synthetic_1", "synthetic_2"]
    summary = {
        "schema": "RealSaS.Corpus370V5SourcePaletteInterventionPilot.v1",
        "training_authorized": False,
        "graphics_family_authority_minted": False,
        "failure_count": 0,
        "exact_subject_ids": ids,
        "subjects": {sid: {"source_sha256": "b"*64} for sid in ids},
    }
    checksum = _write(source_dir/"CORPUS370_V5_PALETTE_INTERVENTION_PILOT_V1.json",
                      summary)
    entries = {}
    for sid in ids:
        source = _fixture()
        source["subject_id"] = sid
        source["parents"]["subject_id"] = sid
        row = copy.deepcopy(source["rows"][0])
        row["view_index"] = 4
        row["generic_counterfactual"]["view_index"] = 4
        row["generic_counterfactual_v2"]["view_index"] = 4
        source["rows"].append(row)
        source["qualified_intervention_rows"] = 2
        src_path = source_dir/"subjects"/sid/"SOURCE_PALETTE_INTERVENTION_V1.json"
        src_sha = _write(src_path, source)
        visibility = qualify(source)
        visibility["parents"] = {
            "source_manifest_sha256": src_sha,
            "source_pilot_summary_sha256": checksum,
        }
        vis_path = visibility_dir/"subjects"/sid/"SOURCE_HIT_VISIBILITY_CANDIDATE_COURT_V1.json"
        entries[sid] = {"receipt_sha256": _write(vis_path, visibility)}
    batch = {
        "schema": "RealSaS.Corpus370BoundedSourceHitVisibilityBatchCourt.v1",
        "parent_summary_sha256": checksum,
        "training_authorized": False,
        "graphics_family_authority_minted": False,
        "subjects": entries,
    }
    _write(visibility_dir/"CORPUS370_HOSTED_SOURCE_HIT_VISIBILITY_BATCH_V1.json",batch)
    return source_dir,visibility_dir,result_dir,checksum


def test_three_two_view_oracle_consumer_has_hidden_edit_consequence(tmp_path):
    src,vis,out,checksum = _build(tmp_path)
    report = run(src,vis,out,checksum)
    assert report["subject_count"] == 3
    assert report["evaluated_pixel_count"] == 6
    assert report["recoverable_occluded_edit_sources"] == 6
    assert report["abstain_pixel_count"] == 0
    assert report["qualified_same_drawable_positive_pairs"] == 0
    assert report["qualified_different_drawable_negative_pairs"] == 0
    assert report["teacher_oracle_only"] is True
    assert report["training_authorized"] is False
    assert all((out/"subjects"/sid/"SOURCE_HIT_ORACLE_CONSUMER_V1.json").is_file()
               for sid in report["subjects"])


def test_forged_batch_parent_sha_fails_closed(tmp_path):
    src,vis,out,checksum = _build(tmp_path)
    with pytest.raises(ValueError, match="SOURCE_SUMMARY_HASH_DRIFT"):
        run(src,vis,out,"0"*64)


def test_stale_visibility_receipt_fails_closed(tmp_path):
    src,vis,out,checksum = _build(tmp_path)
    path = vis/"subjects"/"synthetic_0"/"SOURCE_HIT_VISIBILITY_CANDIDATE_COURT_V1.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError,match="VISIBILITY_RECEIPT_HASH_DRIFT"):
        run(src,vis,out,checksum)


def test_forged_graphics_authority_fails_closed(tmp_path):
    src,vis,out,checksum = _build(tmp_path)
    path = vis/"CORPUS370_HOSTED_SOURCE_HIT_VISIBILITY_BATCH_V1.json"
    data = json.loads(path.read_text())
    data["graphics_family_authority_minted"] = True
    _write(path,data)
    with pytest.raises(ValueError,match="VISIBILITY_SCOPE"):
        run(src,vis,out,checksum)
