import json

import pytest

from tools.corpus.classify_corpus370_observation_failures_v1 import classify, SCHEMA


def fixture(root):
    reason = "ADMISSION_MASK_DRIFT:s0:V4:4.0"
    rows = []
    for i in range(370):
        sid = f"s{i}"
        rows.append({"subject_id":sid,"display_name":"Source-"+sid,
                     "pack_id":"test-pack","raw_source":{"sha256":"a"*64},
                     "observation_preparation":(
                         {"status":"FAIL","reason":reason,"source_format":".gltf",
                          "failure_receipt":{"path":str(root/"fail"),"sha256":"b"*64}}
                         if i==0 else {"status":"PASS"})})
    data={"schema":"RealSaS.Corpus370PlatformFamilyInputBundle.v2",
          "candidate_subject_count":370,"subject_count":370,
          "observation_failed_count":1,"observation_qualified_count":369,
          "subjects":rows,"training_authorized":False}
    policy={"schema":"RealSaS.IrisObservationPreparationPolicy.v1","view_count":8,
            "maximum_mask_gap_px":2.0,
            "maximum_relative_vertex_error":0.00001,"training_authorized":False}
    root.mkdir(parents=True,exist_ok=True)
    (root/"input.json").write_text(json.dumps(data))
    (root/"policy.json").write_text(json.dumps(policy))
    return root/"input.json",root/"policy.json"


def test_strata_preserves_all_370_identities_and_unchanged_failures(tmp_path):
    mp, pp=fixture(tmp_path)
    result=classify(mp,pp)
    assert result["schema"]==SCHEMA
    assert result["candidate_count"]==370
    assert result["mask_failures"]==1 and result["bind_vertex_failures"]==0
    assert result["subjects"][0]["historical_admission_status"]=="FAIL"
    assert result["subjects"][0]["training_authorized"] is False
    assert result["by_lane_and_pack"]["MASK"]=={"test-pack":1}


@pytest.mark.parametrize("altered",["below_threshold","wrong_subject","extra_failure","dupe_id","nan"])
def test_strata_rejects_mistyped_or_fabricated_promotion(tmp_path,altered):
    mp, pp=fixture(tmp_path)
    doc=json.loads(mp.read_text())
    if altered=="below_threshold":
        doc["subjects"][0]["observation_preparation"]["reason"]="ADMISSION_MASK_DRIFT:s0:V4:1.0"
    elif altered=="wrong_subject":
        doc["subjects"][0]["observation_preparation"]["reason"]="ADMISSION_MASK_DRIFT:s1:V4:4.0"
    elif altered=="extra_failure":
        doc["subjects"][1]["observation_preparation"]={"status":"FAIL","reason":"ADMISSION_MASK_DRIFT:s1:V0:12.0","source_format":".gltf","failure_receipt":{}}
    elif altered=="dupe_id":
        doc["subjects"][1]["subject_id"]="s0"
    else:
        doc["subjects"][0]["observation_preparation"]["reason"]="ADMISSION_MASK_DRIFT:s0:V4:nan"
    mp.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="CORPUS370_OBSERVATION_STRATA_"):
        classify(mp,pp)
