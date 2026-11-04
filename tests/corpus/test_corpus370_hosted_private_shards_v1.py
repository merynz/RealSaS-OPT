from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.corpus import hosted_v5_private_shards_v1 as shards


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data(tmp_path,monkeypatch):
    original_root=tmp_path/"old"/"home"/"monster"/"realsas_authority"
    original_root.mkdir(parents=True)
    monkeypatch.setattr(shards,"ALLOWED_ROOT",original_root)
    source=original_root/"corpus"/"source"
    source.mkdir(parents=True)
    a=source/"subject_001.gltf"
    b=source/"texture.bin"
    a.write_bytes(b"original gltf asset bytes")
    b.write_bytes(b"external texture bytes")
    third=original_root/"same"/"subject_ref.gltf"
    third.parent.mkdir(parents=True)
    third.write_bytes(a.read_bytes())
    files=[{"path":str(f),"size_bytes":f.stat().st_size,"sha256":sha(f)}
           for f in (a,b,third)]
    doc={
        "schema":shards.INV_SCHEMA,
        "status":"SHA_BOUND_INPUT_MANIFEST__NO_HOSTED_BYTE_TRANSPORT_YET",
        "source_cohort":370,"observation_qualified":352,"observation_failed":18,
        "training_authorized":False,
        "files":files,
        "summary":{"unique_file_count":len(files)},
    }
    inventory=tmp_path/"private_original_inventory.json"
    inventory.write_text(json.dumps(doc))
    return original_root,inventory,files


def test_roundtrip_reconstitutes_exact_private_original_bytes(tmp_path,monkeypatch):
    root,inventory,files=data(tmp_path,monkeypatch)
    out=tmp_path/"private_shards"
    receipt=shards.export(inventory,out,max_shard_bytes=64*1024**2)
    assert receipt["subject_count"]==370
    assert receipt["unique_object_count"]==2  # identical content deduplicates
    assert receipt["training_authorized"] is False
    assert receipt["shard_count"]==1
    assert (out/"PUBLIC_PATH_FREE_RECEIPT.json").is_file()
    assert "source" not in (out/"PUBLIC_PATH_FREE_RECEIPT.json").read_text()
    restored=tmp_path/"restored"/"home"/"monster"/"realsas_authority"
    restored.mkdir(parents=True)
    status=shards.hydrate(out/"PRIVATE_V5_SHARD_MANIFEST.json",out,
                          tmp_path/"cache",
                          expected_manifest_sha=receipt["manifest_sha256"],
                          target_root=restored)
    assert status["status"]=="EXACT_ABSOLUTE_V5_SOURCE_BYTES_REHYDRATED"
    for row in files:
        relative=Path(row["path"]).relative_to(root)
        assert sha(restored/relative)==row["sha256"]


def test_source_mutation_is_not_silently_exported(tmp_path,monkeypatch):
    _,inventory,files=data(tmp_path,monkeypatch)
    Path(files[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError,match="HOSTED_SHARD_SOURCE_FILE_DRIFT"):
        shards.export(inventory,tmp_path/"out",max_shard_bytes=64*1024**2)


def test_archive_tampering_is_detected_prior_to_extract(tmp_path,monkeypatch):
    root,inventory,_=data(tmp_path,monkeypatch)
    out=tmp_path/"out"
    receipt=shards.export(inventory,out,max_shard_bytes=64*1024**2)
    archive=out/"private-objects-0001.zip"
    archive.write_bytes(archive.read_bytes()+b"tamper")
    target=tmp_path/"target"
    target.mkdir()
    with pytest.raises(ValueError,match="HOSTED_SHARD_DOWNLOAD_ARCHIVE_SHA_DRIFT"):
        shards.hydrate(out/"PRIVATE_V5_SHARD_MANIFEST.json",out,
                       tmp_path/"cache",expected_manifest_sha=receipt["manifest_sha256"],
                       target_root=target)


def test_reject_private_inventory_that_promotes_training(tmp_path,monkeypatch):
    _,inventory,_=data(tmp_path,monkeypatch)
    d=json.loads(inventory.read_text());d["training_authorized"]=True
    inventory.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="HOSTED_SHARD_EXACT_INPUT_CENSUS_REQUIRED"):
        shards.export(inventory,tmp_path/"out",max_shard_bytes=64*1024**2)


def test_path_escape_refused_before_export(tmp_path,monkeypatch):
    _,inventory,_=data(tmp_path,monkeypatch)
    d=json.loads(inventory.read_text())
    d["files"][0]["path"]=str(tmp_path/"outside.gltf")
    inventory.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="HOSTED_SHARD_UNSAFE_ABSOLUTE_SOURCE_PATH"):
        shards.export(inventory,tmp_path/"out",max_shard_bytes=64*1024**2)


def test_manifest_integrity_required(tmp_path,monkeypatch):
    _,inventory,_=data(tmp_path,monkeypatch)
    out=tmp_path/"out"
    receipt=shards.export(inventory,out,max_shard_bytes=64*1024**2)
    manifest=out/"PRIVATE_V5_SHARD_MANIFEST.json"
    content=json.loads(manifest.read_text());content["training_authorized"]=True
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError,match="HOSTED_SHARD_PINNED_MANIFEST_SHA_MISMATCH"):
        shards.hydrate(manifest,out,tmp_path/"cache",
                       expected_manifest_sha=receipt["manifest_sha256"],
                       target_root=tmp_path/"restored")
