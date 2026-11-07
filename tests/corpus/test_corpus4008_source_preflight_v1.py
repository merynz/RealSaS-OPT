"""4,008 original record census tests. All source meshes in tests are synthetic."""
import hashlib
import json

import pytest

from tools.corpus.build_corpus4008_source_preflight_v1 import build_census


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path):
    root = tmp_path / "sources"
    source_dir = root / "quaternius_cc0"
    source_dir.mkdir(parents=True)
    payload = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    token = _sha(payload)
    (source_dir / (token + "_sample.obj")).write_bytes(payload)
    master, selected = [], {}
    for i in range(4008):
        qa = i >= 3993
        provider = "khronos_gltf_qa" if qa else "quaternius_cc0"
        source_token = token if i == 0 else _sha(("synthetic"+str(i)).encode())
        cid = provider + ":" + source_token
        asset = "asset_" + str(i).zfill(4)
        row = {
            "canonical_asset_id":asset,"candidate_id":cid,"source_registry_id":provider,
            "decision":"ADMIT_GEOMETRY_ONLY","split":"EXTERNAL_HOLDOUT" if i==1 else (
                "QA_ONLY" if qa else "FIT"),
        }
        master.append(row)
        if not qa:selected[asset] = row
    path = tmp_path / "master.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in master)+"\n")
    selection=tmp_path / "selected.json"
    selection.write_text(json.dumps(selected))
    return root,path,selection


def test_full_census_4008_with_one_measured_source(tmp_path):
    src,master,selected = _fixture(tmp_path)
    result = build_census(master_path=master,selection_path=selected,
                          sources_root=src,output_dir=tmp_path/"out")
    assert result["universe_count"] == 4008
    assert result["census_entries"] == 4008
    assert result["raw_source_probe_attempt_count"] == 1
    assert result["statuses"]["OBJ_SOURCE_DOMAINS_OBSERVED__NO_CHARACTER_OR_RENDER_COURT"] == 1
    assert result["statuses"]["SOURCE_RAW_NOT_FOUND"] == 4007
    assert result["old_holdout_count"] == 1
    assert result["qa_only_count"] == 15
    assert result["source_camera_qualified_count"] == 0
    assert result["training_authorized"] is False
    evidence = json.loads((tmp_path/"out"/"source_receipts"/"asset_0000.json").read_text())
    assert evidence["source_bytes_verified"] is True
    assert evidence["character_vs_prop"] == "UNKNOWN"
    assert evidence["eight_view_camera_courted"] is False


def test_unexpected_duplicate_source_candidates_are_not_silently_selected(tmp_path):
    src,master,selected = _fixture(tmp_path)
    p=next((src/"quaternius_cc0").glob("*.obj"))
    (src/"quaternius_cc0"/(p.stem+"_variant.obj")).write_bytes(p.read_bytes())
    result=build_census(master_path=master,selection_path=selected,
                        sources_root=src,output_dir=tmp_path/"out")
    assert result["statuses"]["AMBIGUOUS_SOURCE_VARIANTS__REVIEW"] == 1
    assert result["raw_source_probe_attempt_count"] == 0


def test_deleted_original_does_not_mint_training(tmp_path):
    src,master,selected = _fixture(tmp_path)
    next((src/"quaternius_cc0").glob("*.obj")).unlink()
    result=build_census(master_path=master,selection_path=selected,
                        sources_root=src,output_dir=tmp_path/"out")
    assert result["statuses"]["SOURCE_RAW_NOT_FOUND"]==4008
    assert result["training_authorized"] is False
