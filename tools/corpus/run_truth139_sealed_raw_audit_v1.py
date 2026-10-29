#!/usr/bin/env python3
"""Run Truth139 raw capability audit only against the exact October-4 corpus seal.

The lower-level capability scanner intentionally knows geometry/rig/material semantics.  This
wrapper owns corpus identity: exact manifest/ledger/seal bytes, every primary source ledger hash,
and every external glTF buffer/image dependency.  A capability result cannot become PASS unless
both layers agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

from tools.corpus.audit_truth139_gltf_capabilities_v1 import audit as capability_audit


SCHEMA = "RealSaS.Truth139SealedRawAudit.v1"
EXPECTED_ROOT_SHA256 = "2f15bf84694417cee3862c39a89e04bfa131f51331d6da5880739d40b7eab5ab"
EXPECTED_MANIFEST_SHA256 = "3799b170d4138dbc0447d2d3456cc9ad5ba7e185fddace6d6ce2de0d0a2adc00"
EXPECTED_LEDGER_SHA256 = "1524edeb99e2383e2d7516bf9680275289976d57efa3fac3a4b01f78426979b4"
EXPECTED_SEAL_JSON_SHA256 = "0626ce4aacb6540d1ec634484a8db45ff26a09b43c95f6ef13fd73acfd655d2d"
EXPECTED_GLTF_GLBS = 395


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def value_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def parse_ledger(path: Path) -> dict[str,str]:
    rows={}
    for line_number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            digest,relative=line.split("  ",1)
        except ValueError as exc:
            raise ValueError(f"TRUTH139_LEDGER_LINE_INVALID:{line_number}") from exc
        relative=relative.replace("\\","/")
        if len(digest)!=64 or relative in rows:
            raise ValueError(f"TRUTH139_LEDGER_IDENTITY_INVALID:{line_number}")
        rows[relative]=digest.lower()
    return rows


def verify_seal(raw_root: Path) -> tuple[dict,dict[str,str],dict]:
    manifest_path=raw_root/"manifest/MANIFEST.json"
    ledger_path=raw_root/"manifest/FILE_HASHES.sha256"
    seal_path=raw_root/"manifest/CORPUS_SEAL.json"
    material_path=raw_root/"manifest/CORPUS_SEAL_MATERIAL.txt"
    required=(manifest_path,ledger_path,seal_path,material_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("TRUTH139_CORPUS_SEAL_FILES_MISSING")
    if sha256(manifest_path)!=EXPECTED_MANIFEST_SHA256:
        raise ValueError("TRUTH139_CORPUS_MANIFEST_SHA_DRIFT")
    if sha256(ledger_path)!=EXPECTED_LEDGER_SHA256:
        raise ValueError("TRUTH139_CORPUS_LEDGER_SHA_DRIFT")
    if sha256(seal_path)!=EXPECTED_SEAL_JSON_SHA256:
        raise ValueError("TRUTH139_CORPUS_SEAL_JSON_SHA_DRIFT")
    if sha256(material_path)!=EXPECTED_ROOT_SHA256:
        raise ValueError("TRUTH139_CORPUS_ROOT_SEAL_DRIFT")
    manifest=json.loads(manifest_path.read_text())
    seal=json.loads(seal_path.read_text())
    if manifest.get("schema")!="realsas.truth_corpus.v1":
        raise ValueError("TRUTH139_CORPUS_MANIFEST_SCHEMA")
    if manifest.get("corpus_id")!="realsas-truth-corpus-v1-kaykit-quaternius":
        raise ValueError("TRUTH139_CORPUS_ID_DRIFT")
    if (manifest.get("policy") or {}).get("raw_files_immutable") is not True:
        raise ValueError("TRUTH139_CORPUS_IMMUTABILITY_POLICY_DRIFT")
    ledger=parse_ledger(ledger_path)
    gltf=[name for name in ledger if name.lower().endswith((".gltf",".glb")) and name.startswith("raw/quaternius/")]
    if len(gltf)!=EXPECTED_GLTF_GLBS:
        raise ValueError(f"TRUTH139_LEDGER_GLTF_CARDINALITY:{len(gltf)}")
    return manifest,ledger,seal


def _external_dependencies(source: Path, raw_root: Path, ledger: dict[str,str]) -> list[dict]:
    suffix=source.suffix.lower()
    if suffix==".gltf":
        doc=json.loads(source.read_text(encoding="utf-8"))
    elif suffix==".glb":
        from tools.observation.source_scene_contract_v2 import _json_and_chunks
        doc,_=_json_and_chunks(source)
    else:
        return []
    found={}
    for kind,rows in (("BUFFER",doc.get("buffers",[])),("IMAGE",doc.get("images",[]))):
        for row in rows:
            uri=str(row.get("uri") or "")
            if not uri or uri.startswith("data:"):
                continue
            path=(source.parent/unquote(uri)).resolve()
            if not path.is_relative_to(raw_root) or not path.is_file():
                raise ValueError("TRUTH139_EXTERNAL_DEPENDENCY_MISSING_OR_ESCAPE:"+str(source))
            relative=path.relative_to(raw_root).as_posix()
            expected=ledger.get(relative)
            actual=sha256(path)
            if expected is None or actual!=expected:
                raise ValueError("TRUTH139_EXTERNAL_DEPENDENCY_LEDGER_DRIFT:"+relative)
            found[relative]={"path":relative,"sha256":actual,"size_bytes":path.stat().st_size,"kind":kind}
    return [found[key] for key in sorted(found)]


def run(raw_root: Path, fit8_selection: Path) -> dict:
    raw_root=raw_root.resolve()
    manifest,ledger,seal=verify_seal(raw_root)
    base=capability_audit(raw_root,fit8_selection=fit8_selection,expected_corpus_root_sha256=EXPECTED_ROOT_SHA256)
    if base.get("schema")!="RealSaS.Truth139RawCapabilityAudit.v1":
        raise ValueError("TRUTH139_CAPABILITY_AUDIT_SCHEMA")

    pack={str(row["id"]):row for row in manifest.get("results") or () if row.get("provider")=="quaternius"}
    enriched=[]
    for row in base.get("subjects") or ():
        value=dict(row)
        source=dict(value["source"])
        relative=str(source["relative_path"])
        expected=ledger.get(relative)
        if expected is None or expected!=source["sha256"]:
            raise ValueError("TRUTH139_PRIMARY_SOURCE_LEDGER_DRIFT:"+relative)
        path=(raw_root/relative).resolve()
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError("TRUTH139_PRIMARY_SOURCE_BYTES_DRIFT:"+relative)
        source["dependencies"]=_external_dependencies(path,raw_root,ledger)
        source["dependency_count"]=len(source["dependencies"])
        pack_row=pack.get(str(source.get("pack_id") or ""))
        if pack_row is None:
            raise ValueError("TRUTH139_SOURCE_PACK_NOT_IN_SEALED_MANIFEST:"+relative)
        source["official_url"]=str(pack_row.get("official_url") or "")
        source["drive_folder_id"]=str(pack_row.get("drive_folder_id") or "")
        source["source_page_snapshot"]={
            "path":str(pack_row.get("source_snapshot") or ""),
            "sha256":str(pack_row.get("source_snapshot_sha256") or ""),
        }
        snapshot=source["source_page_snapshot"]
        snapshot_path=(raw_root/snapshot["path"]).resolve()
        if (not snapshot["path"] or ledger.get(snapshot["path"])!=snapshot["sha256"]
                or not snapshot_path.is_file() or sha256(snapshot_path)!=snapshot["sha256"]):
            raise ValueError("TRUTH139_SOURCE_PAGE_SNAPSHOT_DRIFT:"+relative)
        value["source"]=source
        enriched.append(value)

    status=("PASS_SEALED_RAW_CAPABILITY_AUDIT__PLATFORM_ADMISSION_PENDING"
            if base.get("status")=="PASS_RAW_CAPABILITY_AUDIT__PLATFORM_ADMISSION_PENDING"
            and base.get("historical_cardinality_reproduced") is True else
            "FAIL_OR_INCOMPLETE_SEALED_RAW_AUDIT__DO_NOT_BUILD_APPARATUS")
    payload={
        "schema":SCHEMA,
        "date":"2026-10-29",
        "status":status,
        "corpus":{
            "root_path":str(raw_root),
            "root_sha256":EXPECTED_ROOT_SHA256,
            "manifest_sha256":EXPECTED_MANIFEST_SHA256,
            "ledger_sha256":EXPECTED_LEDGER_SHA256,
            "seal_json_sha256":EXPECTED_SEAL_JSON_SHA256,
            "ledger_entry_count":len(ledger),
            "quaternius_gltf_glb_ledger_count":EXPECTED_GLTF_GLBS,
            "seal_verified":True,
        },
        "base_capability_audit_identity_sha256":base["audit_identity_sha256"],
        "counts":dict(base["counts"]),
        "subjects":enriched,
        "duplicate_groups":list(base.get("duplicate_groups") or ()),
        "file_failures":list(base.get("file_failures") or ()),
        "split_authorized":False,
        "training_authorized":False,
        "next_gate":"BUILD_TRUTH139_APPARATUS_MANIFEST_AND_COMMON_OBSERVATIONS",
    }
    payload["audit_identity_sha256"]=value_hash(payload)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root",required=True)
    parser.add_argument("--fit8-selection",default="canonical/FIT8_DIVERSITY_SELECTION_V1.json")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=run(Path(args.raw_root),Path(args.fit8_selection).resolve())
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    staged=output.with_suffix(output.suffix+".partial")
    staged.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    staged.replace(output)
    print(json.dumps({"output":str(output),"status":result["status"],"counts":result["counts"],"audit_identity_sha256":result["audit_identity_sha256"]},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
