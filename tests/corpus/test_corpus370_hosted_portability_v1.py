from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.corpus import plan_corpus370_hosted_portability_v1 as planner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixtures(tmp_path,monkeypatch):
    root=tmp_path/"home"/"monster"/"realsas_authority"
    root.mkdir(parents=True)
    source_root=root/"source_bundle"/"extracted"/"sourceid"
    source_root.mkdir(parents=True)
    other=root/"prepared"
    other.mkdir()
    v5_path=root/"V5.json"
    roster_path=root/"ROSTER.json"
    rest_path=root/"REST.json"
    camera=root/"camera.json"
    camera.write_text('{"camera":"fixture"}')
    rows=[]
    originals=[]
    rest_rows={}
    sources=[]
    for i in range(370):
        sid="subject_%03d"%i
        original=source_root/(sid+".gltf")
        original.write_text("original-art-"+sid)
        sources.append({"subject_id":sid,"source_sha256":sha(original),
                        "source_relative_path":original.name})
        if i>=352:
            continue
        rows.append({"subject_id":sid,"source_sha256":sha(original)})
        folder=other/sid
        folder.mkdir()
        arrays=folder/"rest.npz"
        arrays.write_bytes(("source-rest-"+sid).encode())
        manifest=folder/"SOURCE_REST_GEOMETRY_MANIFEST_V1.json"
        manifest.write_text(json.dumps({
            "subject_id":sid,"source_sha256":sha(original),
            "output":{"path":str(arrays),"sha256":sha(arrays)},
            "authority":{"training_authorized":False},
        }))
        rest_rows[sid]={
            "status":"PASS_SOURCE_REST_INDEPENDENT_COURT",
            "input":{"source_sha256":sha(original)},
            "manifest":{"path":str(manifest),"sha256":sha(manifest)},
        }
        originals.append(original)
    v5_path.write_text(json.dumps({
        "subject_count":370,"prepared_subject_count":352,"failure_count":18,
        "subjects":rows,
        "failures":[{"subject_id":x["subject_id"],"source_sha256":x["source_sha256"]}
                    for x in sources[352:]],
        "training_authorized":False,
    }))
    roster_path.write_text(json.dumps({
        "subject_count":370,"subjects":sources,"training_authorized":False
    }))
    rest_path.write_text(json.dumps({
        "summary":{"independent_source_rest_pass":352},
        "subjects":rest_rows,"training_authorized":False
    }))
    monkeypatch.setattr(planner,"REQUIRED_BASE",root)
    monkeypatch.setattr(planner,"V5_SHA",sha(v5_path))
    monkeypatch.setattr(planner,"ROSTER_SHA",sha(roster_path))
    monkeypatch.setattr(planner,"REST_PIN",sha(rest_path))
    monkeypatch.setattr(planner,"row_input",lambda row:{
        "camera":{"path":str(camera),"sha256":sha(camera)}
    })
    return v5_path,roster_path,rest_path,source_root


def test_full_subject_partition_and_texture_dependencies(tmp_path,monkeypatch):
    v5,roster,rest,source_root=fixtures(tmp_path,monkeypatch)
    texture=source_root/"texture.png"
    texture.write_bytes(b"external texture")
    result=planner.capture(v5,roster,rest,source_root)
    assert result["status"].endswith("NO_HOSTED_BYTE_TRANSPORT_YET")
    assert result["training_authorized"] is False
    assert result["source_cohort"]==370
    assert result["observation_qualified"]==352
    assert result["observation_failed"]==18
    assert result["summary"]["source_archive_dependency_count"]>=371
    assert any(r["path"]==str(texture) and r["sha256"]==sha(texture)
               for r in result["files"])
    assert all(r["path"].startswith(str(planner.REQUIRED_BASE))
               for r in result["files"])


def test_manifests_do_not_silently_accept_modified_source(tmp_path,monkeypatch):
    v5,roster,rest,source_root=fixtures(tmp_path,monkeypatch)
    (source_root/"subject_000.gltf").write_bytes(b"modified")
    with pytest.raises(ValueError,match="HOSTED_INVENTORY_PARENT_HASH_DRIFT"):
        planner.capture(v5,roster,rest,source_root)
