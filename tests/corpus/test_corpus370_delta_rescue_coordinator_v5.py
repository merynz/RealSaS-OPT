"""V5 delta coordinator preserves the 370 partition under repairs and failure.

Real source/import/view byte correctness is independently verified in
test_corpus370_historical_observation_reuse_v1.py and by the live SHA audit.
These synthetic tests isolate coordinator transitions and typed failure records.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest
from tools.corpus import rescue_corpus370_observation_delta_v5 as worker
from test_corpus370_platform_family_input_bundle_v3 import fixture,write,sha


def inputs(root):
    roster,sm,extracted,obs,prov,packet=fixture(root)
    source_rows=json.loads(roster.read_text())["subjects"]
    successes,failures=[],[]
    for i,row in enumerate(source_rows):
        sid=row["subject_id"]
        if i<294:
            original=root/"legacy"/"objects"/f"{i:064x}"/"imported"/sid/"imported.json"
            write(original,{"import":{"source_sha256":row["source_sha256"]}})
            path=original.parent.parent/"packet"/sid
            path.mkdir(parents=True,exist_ok=True)
            successes.append({"subject_id":sid,"source_sha256":row["source_sha256"],
                              "packet_path":str(path)})
        else:
            receipt=write(root/"old_failures"/f"{sid}.json",{"subject_id":sid})
            failures.append({"subject_id":sid,"source_sha256":row["source_sha256"],
                             "reason":"ORIGINAL_FAILURE_PRESERVED",
                             "source_format":".glb",
                             "receipt":{"path":str(receipt),"sha256":sha(receipt)}})
    parent=write(root/"old-parent-v3.json",{
        "schema":"RealSaS.Corpus370PreparedObservationPackets.v3",
        "training_authorized":False,
        "subject_count":370,"prepared_subject_count":294,
        "failure_count":76,
        "parent":{"sha256":"129b173b5d789cf4dc020f883ee8b16db8156380dd0cae939319ad3c279c139a"},
        "subjects":successes,"failures":failures,
    })
    policy=Path(__file__).resolve().parents[2]/"canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json"
    blender=root/"blender";blender.write_bytes(b"pinned-binary")
    probe=root/"ufbx";probe.write_bytes(b"pinned-source-probe")
    runtime=root/"runtime.json";runtime.write_text("{}")
    return parent,roster,extracted,policy,blender,probe,runtime


def fake_audit(row,policy):
    return {"status":"EXACT_HISTORICAL_V3_PASS_BYTES_REUSABLE",
            "original_import_sha256":sha(
                Path(row["packet_path"]).parent.parent/"imported"/row["subject_id"]/"imported.json"
            ),
            "legacy_admission_sha256":"b"*64}


def run(root,monkeypatch,selected):
    parent,roster,extracted,policy,blender,probe,runtime=inputs(root)
    monkeypatch.setattr(worker,"PINNED_HISTORICAL_V3_SHA256",sha(parent))
    monkeypatch.setattr(worker,"inspect",fake_audit)
    monkeypatch.setattr(worker,"_packet_complete",lambda *a,**k:True)
    def prepare_v4(roster,extracted,output,blender,probe,policy,runtime,sid):
        if sid=="subject_295":
            raise ValueError("SYNTHETIC_SOURCE_RASTER_FAIL")
        return {
            "schema":"RealSaS.Corpus370PreparedObservationPackets.v4",
            "subjects":[{"subject_id":sid,"source_sha256":
                         json.loads(Path(roster).read_text())["subjects"][int(sid.split("_")[1])]["source_sha256"],
                         "packet_path":str(root/"new"/sid)}],
            "binding":{"blender_binary_sha256":sha(blender),
                       "producer_sources":{"source":"fake-source-sha"}},
        }
    monkeypatch.setattr(worker,"prepare_v4",prepare_v4)
    result=worker.rescue(
        parent_index_path=parent,expected_parent_sha256=sha(parent),
        roster_path=roster,expected_roster_sha256=sha(roster),
        extracted_source_root=extracted,output_root=root/"repair",
        blender=blender,probe=probe,policy_path=policy,
        runtime_request_path=runtime,selected_subject_ids=selected)
    return result


def test_delta_preserves_294_legacy_two_original_failures_outcome_partition(tmp_path,monkeypatch):
    result=run(tmp_path,monkeypatch,["subject_294","subject_295"])
    assert result["historical_v3_pass_preserved_count"]==294
    assert result["new_v4_repaired_count"]==1
    assert result["new_v4_repair_failure_count"]==1
    assert result["unselected_original_failure_count"]==74
    assert result["prepared_subject_count"]==295
    assert result["failure_count"]==75
    assert result["subject_count"]==370
    assert not result["authority"]["old_pass_status_upgraded_to_v4"]
    assert result["training_authorized"] is False
    repaired=next(x for x in result["subjects"] if x["subject_id"]=="subject_294")
    assert repaired["evidence_producer"]=="NEW_V4_REPAIR_FROM_ORIGINAL_FAIL"
    failure=next(x for x in result["failures"] if x["subject_id"]=="subject_295")
    assert failure["parent_failure"]["reason"]=="ORIGINAL_FAILURE_PRESERVED"
    assert failure["original_parent_failure_unchanged"] is True
    assert (tmp_path/"repair"/"CORPUS370_PREPARED_OBSERVATION_PACKETS_V5.json").is_file()


def test_old_pass_cannot_be_resubmitted_as_v4_repair(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="REPAIR_SCOPE_NOT_PARENT_FAILURE"):
        run(tmp_path,monkeypatch,["subject_000"])


def test_duplicate_repair_subject_id_is_not_silently_collapsed(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="REPAIR_SCOPE_NOT_PARENT_FAILURE"):
        run(tmp_path,monkeypatch,["subject_294","subject_294"])
