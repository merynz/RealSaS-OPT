"""Original V3 PASS cannot be reused unless all eight sources/import bytes match."""
import json
from pathlib import Path
import shutil

import pytest

from tools.corpus.audit_corpus370_historical_observation_reuse_v1 import inspect
from test_corpus370_platform_family_input_bundle_v3 import fixture, sha


def legacy_packet(tmp):
    _, _, _, _, _, seed = fixture(tmp)
    subject = "subject_000"
    path = tmp / "objects" / ("0"*64) / "packet" / subject
    path.mkdir(parents=True,exist_ok=True)
    for f in seed.iterdir():
        if f.name != "imported.json":
            shutil.copyfile(f,path/f.name)
    imported = tmp / "objects" / ("0"*64) / "imported" / subject / "imported.json"
    imported.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(seed/"imported.json",imported)
    admission_path = path/"IRIS_OBSERVATION_ADMISSION.json"
    admission = json.loads(admission_path.read_text())
    policy = json.loads((Path(__file__).resolve().parents[2]/
                         "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json").read_text())
    admission["policy"] = policy
    admission_path.write_text(json.dumps(admission))
    return {"subject_id":subject,"source_sha256":admission["source_sha256"],
            "packet_path":str(path)},path,imported,policy


def test_legacy_pass_reusable_only_as_separately_typed_byte_proof(tmp_path):
    row, packet, imported, policy = legacy_packet(tmp_path)
    result = inspect(row,policy)
    assert result["status"]=="EXACT_HISTORICAL_V3_PASS_BYTES_REUSABLE"
    assert result["original_import_sha256"]==sha(imported)
    assert result["eight_view_raster_and_mask_hashes_verified"] is True


@pytest.mark.parametrize("target",["imported","mask","raster","candidate","camera"])
def test_legacy_pass_parent_byte_tampering_fails_closed(tmp_path,target):
    row,packet,imported,policy = legacy_packet(tmp_path)
    files = {
        "imported":imported,
        "mask":packet/"V7.mask.bin",
        "raster":packet/"V0.png",
        "candidate":packet/"SOURCE_REST_RENDER_CANDIDATE.json",
        "camera":packet/"camera_bundle.json",
    }
    path = files[target]
    path.write_bytes(path.read_bytes()+b"tamper")
    result=inspect(row,policy)
    assert result["status"]=="NOT_PROVEN"
    assert result["eight_view_raster_and_mask_hashes_verified"] is False


def test_legacy_policy_is_not_promoted_to_current_without_proof(tmp_path):
    row,packet,imported,policy = legacy_packet(tmp_path)
    admission=packet/"IRIS_OBSERVATION_ADMISSION.json"
    content=json.loads(admission.read_text())
    content["policy"]["maximum_mask_gap_px"]=3.0
    admission.write_text(json.dumps(content))
    result=inspect(row,policy)
    assert result["status"]=="NOT_PROVEN"
    assert "REUSE_OLD_POLICY_DIFFERENT" in result["reason"]
