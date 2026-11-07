#!/usr/bin/env python3
"""SHA-verified cross-corpus source identity *overlay*; original rows immutable.

Requires actual original bytes rehashed by the upstream identity court for
EVERY grouped edge. No train/eval resplit and no head truth minted.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from collections import Counter
from pathlib import Path

SCHEMA="RealSaS.VerifiedCrossCorpusSourceIdentityClusterOverlay.v2"

def req(ok,tag):
    if not ok:raise ValueError("SOURCE_CLUSTER_"+tag)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def readcsv(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def writecsv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    return sha(path)

def run(pair_file,pair_summary_file,global_roster,quarantine_file,out,
        *,expected_pairs=53,expected_routes_per_source=37,
        expected_roster_count=4406,
        expected_roster_sha="558289d87a29bda29ff73b617c84a4fbddfdc12f7a88aa72f88e8c3a7fcfb885"):
    summary=json.loads(pair_summary_file.read_text())
    req(summary["schema"]=="RealSaS.CrossCorpusSourceIdentityCourt.v1"
        and summary["training_authorized"] is False
        and summary.get("fully_byte_sha_verified_overlap_count",
                        summary.get("exact_full_sha_byte_verified_pairs"))==expected_pairs
        and summary.get("prefix_overlap_requires_full_source_rehash_count",
                        summary.get("high_confidence_prefix_pending_full_rehash"))==0,
        "REQUIRED_FULL_BYTE_SHA_COURT")
    expected_sha=summary.get("pair_file_sha256",summary.get("pairs_sha256"))
    req(sha(pair_file)==expected_sha,"FROZEN_PAIR_HASH")
    pairs=readcsv(pair_file)
    req(sha(global_roster)==expected_roster_sha,"FROZEN_GLOBAL_SOURCE_ROSTER_SHA")
    sources=readcsv(global_roster)
    quarantined=readcsv(quarantine_file)
    req(len(pairs)==expected_pairs and len(sources)==expected_roster_count,
        "INPUT_CENSUS")
    lookup={x["global_record_key"]:x for x in sources}
    req(len(lookup)==len(sources),"DUPLICATE_SOURCE")
    qcounts=Counter(r["global_record_key"] for r in quarantined)
    req(len(qcounts)==expected_pairs*2 and
        all(n==expected_routes_per_source for n in qcounts.values()),
        "QUARANTINE_ROUTE_CENSUS")
    req(all(r["training_authorized"]=="False" and
            r["loss_mask_valid"]=="False" for r in quarantined),
        "PROMOTED_QUARANTINE_TARGET")
    groups=[];members=[];seen=set();split_counts=Counter()
    for pair in pairs:
        digest=pair.get("core_source_sha256_historical",
                        pair.get("core_source_sha256_historical",""))
        actual=pair.get("real_source_sha256",
                        pair.get("byte_readback_sha256",""))
        req(re.fullmatch("[a-f0-9]{64}",digest or "") and digest==actual,
            "UNVERIFIED_SOURCE_BYTES")
        req(pair["identity_evidence"] in (
            "FULL_SHA256_BYTE_VERIFIED","FULL_64_SHA_ACTUAL_BYTES_VERIFIED"),
            "ONLY_FULL_SHA_ALLOWED")
        a="MASTER3993:"+pair["master_asset_id"]
        b="CORE370:"+pair["core_subject_id"]
        req(a not in seen and b not in seen and a in lookup and b in lookup,
            "DUPLICATE_OR_MISSING_SOURCE")
        req(lookup[a]["split_state"] in ("FIT","TUNE","DEV","CAL","EXTERNAL_HOLDOUT"),
            "MASTER_SPLIT_NOT_FROZEN")
        split=lookup[a]["split_state"]
        declared=pair.get("master_original_split",pair.get("master_split",""))
        req(split==declared,"SPLIT_MISMATCH")
        seen.update((a,b));split_counts[split]+=1
        req(qcounts[a]==expected_routes_per_source and
            qcounts[b]==expected_routes_per_source,"MISSING_QUARANTINE_MEMBER")
        policy=("EXTERNAL_HOLDOUT_HARD_BLOCK" if split=="EXTERNAL_HOLDOUT"
                else "PROTECTED_EVALUATION_NO_TRAIN_PENDING_AUTHORITY"
                if split in ("DEV","TUNE","CAL")
                else "FIT_PENDING_CROSS_CORPUS_SPLIT_AND_RIGHTS")
        group="source_sha256:"+digest
        groups.append({
            "source_cluster_id":group,"source_sha256_actual_bytes":digest,
            "master_asset_id":pair["master_asset_id"],
            "core_subject_id":pair["core_subject_id"],
            "historical_master_split":split,
            "source_group_policy":policy,
            "true_training_leakage_observed":False,
            "physical_source_identity_courted":True,
            "character_or_prop_courted":False,
            "model_training_authorized":False,
            "source_record_mutated":False})
        for key in (a,b):
            members.append({
                "source_cluster_id":group,"original_global_record_key":key,
                "namespace":lookup[key]["namespace"],
                "historical_split_preserved":lookup[key]["split_state"],
                "target_routes_quarantined":expected_routes_per_source,
                "source_group_policy":policy,
                "training_authorized":False,"original_record_preserved":True})
    req(len({x["source_cluster_id"] for x in groups})==expected_pairs
        and len(members)==2*expected_pairs,"CLUSTER_IDENTITY_COLLISION")
    out.mkdir(parents=True,exist_ok=True)
    a=out/"VERIFIED_CROSS_CORPUS_SOURCE_IDENTITY_CLUSTERS_V2.csv"
    b=out/"VERIFIED_CROSS_CORPUS_SOURCE_IDENTITY_MEMBERS_V2.csv"
    ah=writecsv(a,groups);bh=writecsv(b,members)
    result={
        "schema":SCHEMA,"verified_source_sha256_clusters":len(groups),
        "cross_corpus_source_members":len(members),
        "blocked_head_target_routes":len(quarantined),
        "protected_identity_clusters":sum(v for k,v in split_counts.items()
            if k!="FIT"),
        "split_counts":dict(sorted(split_counts.items())),
        "provisional_known_canonical_record_components":
            4378-len(groups) if expected_roster_count==4406 else None,
        "unique_usable_characters_certified":False,
        "cross_format_semantic_dedup_complete":False,
        "source_rights_courted_by_this_work":False,
        "training_authorized":False,
        "source_mutated":False,
        "input_pair_sha256":sha(pair_file),
        "input_roster_sha256":sha(global_roster),
        "input_quarantine_sha256":sha(quarantine_file),
        "cluster_sha256":ah,"members_sha256":bh}
    (out/"VERIFIED_CROSS_CORPUS_SOURCE_IDENTITY_SUMMARY_V2.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("pairs","pair-summary","roster","quarantine","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(run(a.pairs,a.pair_summary,a.roster,a.quarantine,a.output),
                     sort_keys=True))
