#!/usr/bin/env python3
"""Package admitted Truth139 observation packets into the generic Platform Stage01 bundle format.

This is transport only.  It binds exact October-4 corpus provenance, primary/dependency bytes and
already-qualified observation packet bytes into a deterministic receipt.  It does not create a
Platform Subject, Attempt, truth PASS, split, or training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


APPARATUS_SCHEMA="RealSaS.Truth139ApparatusManifest.v1"
PACKETS_SCHEMA="RealSaS.PreparedIrisInputPackets.v1"
MANIFEST_SCHEMA="RealSaS.Truth139PlatformStage01InputBundle.v1"
RECEIPT_SCHEMA="RealSaS.Truth139PlatformStage01InputBundleReceipt.v1"
EXPECTED_MANIFEST_SHA="3799b170d4138dbc0447d2d3456cc9ad5ba7e185fddace6d6ce2de0d0a2adc00"
EXPECTED_LEDGER_SHA="1524edeb99e2383e2d7516bf9680275289976d57efa3fac3a4b01f78426979b4"
EXPECTED_ROOT_SHA="2f15bf84694417cee3862c39a89e04bfa131f51331d6da5880739d40b7eab5ab"
EXPECTED_SEAL_JSON_SHA="0626ce4aacb6540d1ec634484a8db45ff26a09b43c95b984a8db45ff26a09b43c95b"  # overwritten below
# Keep the exact published seal JSON hash separate from the root material hash.
EXPECTED_SEAL_JSON_SHA="0626ce4aacb6540d1ec634484a8db45ff26a09b43c95f6ef13fd73acfd655d2d"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def _ledger(path: Path) -> dict[str,str]:
    out={}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest,relative=line.split("  ",1)
        out[relative.replace("\\","/")]=digest
    return out


def _verify_corpus(raw_root: Path) -> tuple[dict,dict[str,str]]:
    manifest_path=raw_root/"manifest/MANIFEST.json"
    ledger_path=raw_root/"manifest/FILE_HASHES.sha256"
    seal=raw_root/"manifest/CORPUS_SEAL.json"
    material=raw_root/"manifest/CORPUS_SEAL_MATERIAL.txt"
    if sha256(manifest_path)!=EXPECTED_MANIFEST_SHA or sha256(ledger_path)!=EXPECTED_LEDGER_SHA:
        raise ValueError("TRUTH139_STAGE01_CORPUS_MANIFEST_OR_LEDGER_DRIFT")
    if sha256(seal)!=EXPECTED_SEAL_JSON_SHA or sha256(material)!=EXPECTED_ROOT_SHA:
        raise ValueError("TRUTH139_STAGE01_CORPUS_SEAL_DRIFT")
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!="realsas.truth_corpus.v1":
        raise ValueError("TRUTH139_STAGE01_CORPUS_MANIFEST_SCHEMA")
    return manifest,_ledger(ledger_path)


def _ledger_file(raw_root: Path, ledger: dict[str,str], relative: str) -> dict:
    relative=str(relative).replace("\\","/")
    path=(raw_root/relative).resolve()
    if not path.is_relative_to(raw_root) or not path.is_file():
        raise ValueError("TRUTH139_STAGE01_CORPUS_FILE_MISSING:"+relative)
    actual=sha256(path)
    if ledger.get(relative)!=actual:
        raise ValueError("TRUTH139_STAGE01_CORPUS_FILE_LEDGER_DRIFT:"+relative)
    return {"path":relative,"sha256":actual,"size_bytes":path.stat().st_size}


def _license_ref(raw_root: Path, ledger: dict[str,str], pack_id: str) -> dict:
    relative=f"raw/quaternius/{pack_id}/License.txt"
    if relative in ledger and (raw_root/relative).is_file():
        return {**_ledger_file(raw_root,ledger,relative),"scope":"PACK_LICENSE_BYTES"}
    fallback="provenance/licenses/CC0-1.0-legalcode.txt"
    return {**_ledger_file(raw_root,ledger,fallback),"scope":"GLOBAL_CC0_LEGALCODE_FALLBACK"}


def _packet_index(index: dict) -> dict[str,dict]:
    rows={str(row["subject_id"]):dict(row) for row in index.get("subjects") or ()}
    if len(rows)!=len(index.get("subjects") or ()):
        raise ValueError("TRUTH139_STAGE01_PACKET_SUBJECT_DUPLICATE")
    return rows


def build(apparatus: dict, packets: dict, raw_root: Path, output: Path) -> dict:
    if apparatus.get("schema")!=APPARATUS_SCHEMA or apparatus.get("training_authorized") is not False:
        raise ValueError("TRUTH139_STAGE01_APPARATUS_SCHEMA_OR_AUTHORITY")
    if apparatus.get("status")!="READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED":
        raise ValueError("TRUTH139_STAGE01_APPARATUS_STATUS")
    if packets.get("schema")!=PACKETS_SCHEMA or packets.get("training_authorized") is not False:
        raise ValueError("TRUTH139_STAGE01_PACKETS_SCHEMA_OR_AUTHORITY")
    if (packets.get("roster") or {}).get("identity_sha256")!=apparatus.get("apparatus_identity_sha256"):
        raise ValueError("TRUTH139_STAGE01_PACKET_ROSTER_BINDING_DRIFT")
    raw_root=raw_root.resolve()
    corpus_manifest,ledger=_verify_corpus(raw_root)
    packet_rows=_packet_index(packets)
    selected=list(apparatus.get("selected") or ())
    if int(apparatus.get("subject_count",-1))!=len(selected) or set(packet_rows)!={row["subject_key"] for row in selected}:
        raise ValueError("TRUTH139_STAGE01_SUBJECT_ROSTER_DRIFT")

    pack_rows={str(row["id"]):row for row in corpus_manifest.get("results") or ()}
    members: dict[str,Path|bytes]={}
    for relative in ("manifest/CORPUS_SEAL.json","manifest/CORPUS_SEAL_MATERIAL.txt","manifest/FILE_HASHES.sha256","manifest/MANIFEST.json"):
        members[relative]=raw_root/relative

    subjects=[]
    for row in selected:
        key=str(row["subject_key"]);source=dict(row["source"]);pack_id=str(row["pack_id"])
        primary=_ledger_file(raw_root,ledger,source["corpus_relative_path"])
        members[primary["path"]]=raw_root/primary["path"]
        dependencies=[]
        for dep in source.get("dependencies") or ():
            ref=_ledger_file(raw_root,ledger,dep["path"])
            if ref["sha256"]!=dep["sha256"] or ref["size_bytes"]!=int(dep["size_bytes"]):
                raise ValueError("TRUTH139_STAGE01_DEPENDENCY_AUDIT_DRIFT:"+key)
            ref["kind"]=str(dep["kind"])
            dependencies.append(ref);members[ref["path"]]=raw_root/ref["path"]
        license_ref=_license_ref(raw_root,ledger,pack_id);members[license_ref["path"]]=raw_root/license_ref["path"]
        pack=pack_rows.get(pack_id)
        if pack is None:
            raise ValueError("TRUTH139_STAGE01_PACK_NOT_IN_MANIFEST:"+pack_id)
        page=_ledger_file(raw_root,ledger,str(pack["source_snapshot"]));members[page["path"]]=raw_root/page["path"]
        if page["sha256"]!=str(pack["source_snapshot_sha256"]):
            raise ValueError("TRUTH139_STAGE01_SOURCE_PAGE_MANIFEST_DRIFT:"+pack_id)

        packet=packet_rows[key]
        packet_root=Path(str(packet["packet_path"])).resolve()
        if not packet_root.is_dir():
            raise ValueError("TRUTH139_STAGE01_PACKET_PATH_MISSING:"+key)
        observation_names=["IRIS_OBSERVATION_ADMISSION.json","SOURCE_REST_RENDER_CANDIDATE.json","SOURCE_SCENE_SNAPSHOT.json","camera_bundle.json"]
        observation_names += [name for view in range(8) for name in (f"V{view}.mask.bin",f"V{view}.png")]
        evidence={}
        for name in observation_names:
            src=packet_root/name
            if not src.is_file():
                raise ValueError("TRUTH139_STAGE01_OBSERVATION_FILE_MISSING:"+key+":"+name)
            relative=f"observations/{key}/{name}"
            members[relative]=src
            if name in {"IRIS_OBSERVATION_ADMISSION.json","SOURCE_REST_RENDER_CANDIDATE.json","SOURCE_SCENE_SNAPSHOT.json","camera_bundle.json"}:
                evidence[name]={"path":relative,"sha256":sha256(src),"size_bytes":src.stat().st_size}
        if evidence["IRIS_OBSERVATION_ADMISSION.json"]["sha256"]!=packet["proof_sha256"]:
            raise ValueError("TRUTH139_STAGE01_OBSERVATION_PROOF_DRIFT:"+key)
        subjects.append({
            "subject_id":key,"display_name":str(row["display_name"]),"pack_id":pack_id,
            "raw_source":{**primary,"origin":"SEALED_CORPUS_OCTOBER4"},
            "raw_dependencies":dependencies,
            "license_bytes":license_ref,
            "source_page_snapshot":page,
            "observation_file_count":len(observation_names),
            "source_scene_evidence":{
                "camera_bundle":evidence["camera_bundle.json"],
                "iris_observation_admission":evidence["IRIS_OBSERVATION_ADMISSION.json"],
                "source_rest_candidate":evidence["SOURCE_REST_RENDER_CANDIDATE.json"],
                "source_scene_snapshot":evidence["SOURCE_SCENE_SNAPSHOT.json"],
            },
        })

    manifest={
        "schema":MANIFEST_SCHEMA,"date":"2026-10-29",
        "scope":"EXACT_SOURCE_PROVENANCE_DEPENDENCY_CLOSURE_AND_ADMITTED_8VIEW_OBSERVATIONS__NO_PASS_MINTED",
        "corpus_root_sha256":EXPECTED_ROOT_SHA,"apparatus_identity_sha256":apparatus["apparatus_identity_sha256"],
        "prepared_packets_roster_identity_sha256":packets["roster"]["identity_sha256"],
        "subject_count":len(subjects),"subjects":subjects,
        "truth_pass_claimed":False,"product_pass_claimed":False,"training_authorized":False,
    }
    manifest_raw=json_bytes(manifest);members["BOOTSTRAP_MANIFEST.json"]=manifest_raw

    file_rows=[]
    for name in sorted(members):
        value=members[name]
        if isinstance(value,bytes):
            raw=value
        else:
            raw=value.read_bytes()
        file_rows.append({"path":name,"sha256":sha_bytes(raw),"size_bytes":len(raw)})
    receipt={"schema":RECEIPT_SCHEMA,"file_count":len(file_rows),"manifest_sha256":sha_bytes(manifest_raw),"files":file_rows}
    receipt_raw=json_bytes(receipt)

    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp_name=tempfile.mkstemp(dir=output.parent,prefix=output.name+".",suffix=".partial")
    os.close(fd);temp=Path(temp_name)
    try:
        with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(members):
                value=members[name]
                raw=value if isinstance(value,bytes) else value.read_bytes()
                info=zipfile.ZipInfo(name,date_time=(2026,10,29,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o644<<16
                archive.writestr(info,raw)
            info=zipfile.ZipInfo("BUNDLE_RECEIPT.json",date_time=(2026,10,29,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,receipt_raw)
        os.replace(temp,output)
    finally:
        temp.unlink(missing_ok=True)
    return {"schema":"RealSaS.Truth139PlatformStage01BundleBuildReceipt.v1","output":str(output),
            "bundle_sha256":sha256(output),"size_bytes":output.stat().st_size,"subject_count":len(subjects),
            "manifest_sha256":receipt["manifest_sha256"],"receipt_sha256":sha_bytes(receipt_raw),
            "qualification_minted":False,"training_authorized":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apparatus",required=True);parser.add_argument("--prepared-packets",required=True)
    parser.add_argument("--raw-root",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--build-receipt")
    args=parser.parse_args()
    apparatus=json.loads(Path(args.apparatus).resolve().read_text());packets=json.loads(Path(args.prepared_packets).resolve().read_text())
    result=build(apparatus,packets,Path(args.raw_root),Path(args.output))
    if args.build_receipt:
        path=Path(args.build_receipt).resolve();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json_bytes(result))
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
