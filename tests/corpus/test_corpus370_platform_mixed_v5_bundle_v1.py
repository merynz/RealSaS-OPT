"""Real-shaped V5 composite transport retains 294 old PASS producer identities.

The parent pin is substituted only in this isolated synthetic fixture; production
always requires the independently recorded original 294/76 V3 parent SHA.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import zipfile

import pytest
from tools.corpus import prepare_corpus370_platform_truth_bundle_v3 as pack
from test_corpus370_platform_family_input_bundle_v3 import fixture, sha, write


def mixed(root, monkeypatch):
    roster, sources, extracted, index_path, provenance, seed = fixture(root)
    old = json.loads(index_path.read_text())
    roster_rows = {r["subject_id"]:r for r in json.loads(roster.read_text())["subjects"]}
    base_admission = json.loads((seed/"IRIS_OBSERVATION_ADMISSION.json").read_text())
    base_candidate = json.loads((seed/"SOURCE_REST_RENDER_CANDIDATE.json").read_text())
    base_import = json.loads((seed/"imported.json").read_text())
    policy = json.loads((Path(__file__).resolve().parents[2]/
                         "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json").read_text())
    legacy=[]
    for i in range(1,295):
        sid=f"subject_{i:03d}"
        subject_sha=roster_rows[sid]["source_sha256"]
        container=root/"objects"/f"{i:064x}"
        packet=container/"packet"/sid
        packet.mkdir(parents=True,exist_ok=True)
        for file in seed.iterdir():
            if file.name not in ("imported.json", "SOURCE_REST_RENDER_CANDIDATE.json",
                                  "SOURCE_SCENE_SNAPSHOT.json",
                                  "IRIS_OBSERVATION_ADMISSION.json"):
                shutil.copyfile(file,packet/file.name)
        imported=write(container/"imported"/sid/"imported.json",
                       {**base_import,
                        "import":{**base_import["import"],"source_sha256":subject_sha}})
        snapshot=write(packet/"SOURCE_SCENE_SNAPSHOT.json",{"source_sha256":subject_sha})
        candidate=write(packet/"SOURCE_REST_RENDER_CANDIDATE.json",{
            **base_candidate,
            "subject_id":sid,"source_sha256":subject_sha,
            "source_scene_snapshot":{"sha256":sha(snapshot)},
        })
        admission=write(packet/"IRIS_OBSERVATION_ADMISSION.json",{
            **base_admission,
            "subject_id":sid,"source_sha256":subject_sha,
            "imported_geometry_sha256":sha(imported),
            "candidate_sha256":sha(candidate),
            "policy":policy,
        })
        legacy.append({
            "subject_id":sid,"source_sha256":subject_sha,
            "packet_path":str(packet),
            "evidence_producer":"HISTORICAL_V3_PASS_BYTES_RECHECKED",
            "admitted_import_geometry":{"path":str(imported),"sha256":sha(imported)},
            "legacy_admission_sha256":sha(admission),
            "training_authorized":False,
        })
    new=dict(old["subjects"][0])
    new["evidence_producer"]="NEW_V4_REPAIR_FROM_ORIGINAL_FAIL"
    original_fail_receipt=write(root/"parent_failure_subject_000.json",
                               {"subject_id":"subject_000","reason":"PARENT_FAIL"})
    new["parent_failure_receipt"]={"path":str(original_fail_receipt),
                                   "sha256":sha(original_fail_receipt)}
    parent=write(root/"historical_v3_index.json",{
        "schema":"RealSaS.Corpus370PreparedObservationPackets.v3",
        "subject_count":370,"prepared_subject_count":294,"failure_count":76,
        "parent":{"sha256":"129b173b5d789cf4dc020f883ee8b16db8156380dd0cae939319ad3c279c139a"},
        "training_authorized":False,
    })
    # This opt-in patch is only a synthetic test of mixed transport branches.
    monkeypatch.setattr(pack,"PINNED_HISTORICAL_V3_SHA256",sha(parent))
    kept_fail=[r for r in old["failures"] if int(r["subject_id"].split("_")[1])>=295]
    index=write(root/"v5.json",{
        "schema":"RealSaS.Corpus370PreparedObservationPackets.v5",
        "subject_count":370,"prepared_subject_count":295,"failure_count":75,
        "historical_v3_pass_preserved_count":294,"new_v4_repaired_count":1,
        "parent":{"schema":"RealSaS.Corpus370PreparedObservationPackets.v3",
                  "path":str(parent),"sha256":sha(parent),
                  "prepared_subject_count":294,"failure_count":76},
        "binding":old["binding"],
        "subjects":[new,*legacy],"failures":kept_fail,
        "authority":{"old_pass_status_upgraded_to_v4":False},
        "training_authorized":False,
    })
    return roster,sources,extracted,index,provenance,legacy,seed


def test_mixed_294_legacy_plus_one_new_v4_transport_is_exact(tmp_path,monkeypatch):
    roster,source,extracted,index,provenance,legacy,seed=mixed(tmp_path,monkeypatch)
    manifest,receipt,archive=pack.prepare(roster,source,extracted,index,provenance,
                                          tmp_path/"mixed.zip")
    assert manifest["candidate_subject_count"]==370
    assert manifest["observation_qualified_count"]==295
    assert manifest["observation_failed_count"]==75
    rows={r["subject_id"]:r for r in manifest["subjects"]}
    assert rows["subject_000"]["observation_preparation"]["new_v4_admission_claimed"] is True
    assert rows["subject_001"]["observation_preparation"]["historical_v3_pass_reattested"] is True
    legacy_import=Path(legacy[0]["admitted_import_geometry"]["path"])
    with zipfile.ZipFile(archive["path"]) as zipped:
        assert zipped.read("observations/subject_001/imported.json")==legacy_import.read_bytes()
        assert zipped.read("observations/subject_000/imported.json")==(seed/"imported.json").read_bytes()
        assert zipped.read("observations/subject_299/PREPARATION_FAILURE.json")
    assert manifest["training_authorized"] is False


def test_mixed_v5_cannot_promote_legacy_with_tampered_source_mask(tmp_path,monkeypatch):
    roster,source,extracted,index,provenance,legacy,seed=mixed(tmp_path,monkeypatch)
    path=Path(legacy[0]["packet_path"])/"V6.mask.bin"
    path.write_bytes(path.read_bytes()+b"tamper")
    with pytest.raises(ValueError,match="HISTORICAL_PASS_BYTES_NOT_PROVEN"):
        pack.prepare(roster,source,extracted,index,provenance,tmp_path/"reject.zip")


def test_mixed_v5_parent_signature_is_mandatory(tmp_path,monkeypatch):
    roster,source,extracted,index,provenance,legacy,seed=mixed(tmp_path,monkeypatch)
    parent=Path(json.loads(index.read_text())["parent"]["path"])
    parent.write_bytes(parent.read_bytes()+b"tamper")
    with pytest.raises(ValueError,match="V5_PARENT_BINDING"):
        pack.prepare(roster,source,extracted,index,provenance,tmp_path/"reject.zip")
