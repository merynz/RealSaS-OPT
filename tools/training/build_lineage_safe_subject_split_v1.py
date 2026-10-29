#!/usr/bin/env python3
"""Build a deterministic leakage-component train/unseen split from the final admitted truth ledger.

The input cardinality is derived from corpus admission; 139 is a historical raw census, not a split
constant.  Components that touch development/regression lineage may remain useful TRAIN data but
are never eligible for the sealed UNSEEN court.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


SCHEMA = "RealSaS.TruthCorpusLineageLedger.v1"
OUTPUT_SCHEMA = "RealSaS.LineageSafeSubjectSplit.v1"
REQUIRED_LINEAGE_FIELDS = (
    "source_sha256",
    "skeleton_hierarchy_signature",
    "topology_lineage_signature",
    "material_lineage_signature",
)
OPTIONAL_LINEAGE_FIELDS = ("source_identity",)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tokens(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError("TRUTH139_LINEAGE_VALUE_TYPE")
    out = tuple(sorted({str(item).strip() for item in values if str(item).strip()}))
    if not out:
        raise ValueError("TRUTH139_LINEAGE_VALUE_EMPTY")
    return out


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count)); self.rank = [0] * count
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1


def _ledger_identity(ledger: dict) -> str:
    value=dict(ledger); value.pop("ledger_identity_sha256",None)
    return _digest(value)


def _admitted_rows(ledger: dict, expected_subject_count: int | None) -> list[dict]:
    if ledger.get("schema") != SCHEMA:
        raise ValueError("TRUTH139_LINEAGE_LEDGER_SCHEMA")
    if ledger.get("status") != "PASS_TRUTH139_FULL_FAMILY_ADMISSION__SPLIT_AUTHORIZED":
        raise ValueError("TRUTH139_LINEAGE_LEDGER_NOT_SPLIT_AUTHORIZED")
    if ledger.get("split_authorized") is not True or ledger.get("training_authorized") is not False:
        raise ValueError("TRUTH139_LINEAGE_LEDGER_AUTHORITY_DRIFT")
    identity=str(ledger.get("ledger_identity_sha256") or "")
    if len(identity)!=64 or identity!=_ledger_identity(ledger):
        raise ValueError("TRUTH139_LINEAGE_LEDGER_IDENTITY_DRIFT")
    rows = [dict(row) for row in ledger.get("subjects") or () if row.get("admitted_for_truth139") is True]
    if not rows:
        raise ValueError("TRUTH139_LINEAGE_LEDGER_EMPTY")
    ids = [str(row.get("subject_id") or "") for row in rows]
    if any(not subject for subject in ids) or len(set(ids)) != len(ids):
        raise ValueError("TRUTH139_SUBJECT_ID_INVALID_OR_DUPLICATE")
    declared=int(ledger.get("admitted_subject_count",-1))
    if declared!=len(rows):
        raise ValueError("TRUTH139_LEDGER_ADMITTED_COUNT_DRIFT")
    if expected_subject_count is not None and len(rows) != int(expected_subject_count):
        raise ValueError("TRUTH139_EXPECTED_SUBJECT_COUNT_MISMATCH")

    for row in rows:
        lineage = row.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("TRUTH139_LINEAGE_BLOCK_REQUIRED:" + str(row["subject_id"]))
        if row.get("training_authorized") is not False:
            raise ValueError("TRUTH139_SUBJECT_PREMATURE_TRAINING_AUTHORITY:"+str(row["subject_id"]))
        if row.get("unseen_eligible") not in {True,False}:
            raise ValueError("TRUTH139_SUBJECT_UNSEEN_ELIGIBILITY_MISSING:"+str(row["subject_id"]))
        for field in REQUIRED_LINEAGE_FIELDS:
            if field not in lineage:
                raise ValueError("TRUTH139_REQUIRED_LINEAGE_FIELD_MISSING:" + str(row["subject_id"]) + ":" + field)
            _tokens(lineage[field])
        for field in OPTIONAL_LINEAGE_FIELDS:
            if field in lineage and lineage[field] not in (None, "", []): _tokens(lineage[field])
    return rows


def _components(rows: list[dict]) -> list[dict]:
    uf = _UnionFind(len(rows)); owner: dict[tuple[str, str], int] = {}; row_tokens=[]
    for index, row in enumerate(rows):
        lineage=row["lineage"]; tokens=[]
        for field in REQUIRED_LINEAGE_FIELDS + OPTIONAL_LINEAGE_FIELDS:
            if field not in lineage or lineage[field] in (None, "", []): continue
            for token in _tokens(lineage[field]):
                key=(field,token); tokens.append(key); prior=owner.get(key)
                if prior is None: owner[key]=index
                else: uf.union(index,prior)
        row_tokens.append(tokens)
    grouped={}
    for index in range(len(rows)): grouped.setdefault(uf.find(index),[]).append(index)
    components=[]
    for members in grouped.values():
        subject_ids=sorted(str(rows[index]["subject_id"]) for index in members)
        evidence=sorted({f"{field}:{token}" for index in members for field,token in row_tokens[index]})
        unseen_eligible=all(rows[index].get("unseen_eligible") is True for index in members)
        reasons=sorted({str(reason) for index in members for reason in (rows[index].get("unseen_ineligibility_reasons") or ())})
        component_id="lc_"+hashlib.sha256(_canonical_bytes({"subjects":subject_ids,"evidence":evidence})).hexdigest()[:20]
        components.append({"component_id":component_id,"subject_ids":subject_ids,"subject_count":len(subject_ids),
                           "leakage_evidence":evidence,"unseen_eligible":unseen_eligible,
                           "unseen_ineligibility_reasons":reasons})
    return components


def _select_unseen_components(components: list[dict], *, target_unseen_fraction: float, seed: int) -> set[str]:
    if not 0.0 < float(target_unseen_fraction) < 1.0: raise ValueError("TRUTH139_UNSEEN_FRACTION_INVALID")
    total=sum(int(row["subject_count"]) for row in components)
    if total < 2: raise ValueError("TRUTH139_SPLIT_REQUIRES_MULTIPLE_SUBJECTS")
    eligible=[row for row in components if row["unseen_eligible"] is True]
    if not eligible: raise ValueError("TRUTH139_NO_UNSEEN_ELIGIBLE_COMPONENTS")
    target=float(total)*float(target_unseen_fraction)
    ordered=sorted(eligible,key=lambda row:hashlib.sha256(f"{int(seed)}|{row['component_id']}".encode()).hexdigest())
    reachable: dict[int, tuple[int, ...]]={0:()}
    for index,component in enumerate(ordered):
        size=int(component["subject_count"]); additions={}
        for count,chosen in list(reachable.items()):
            new_count=count+size; candidate=chosen+(index,)
            if new_count not in reachable and new_count not in additions: additions[new_count]=candidate
        reachable.update(additions)
    candidates=[count for count in reachable if 0<count<total]
    if not candidates: raise ValueError("TRUTH139_COMPONENT_GRAPH_UNSPLITTABLE")
    best_count=min(candidates,key=lambda count:(abs(count-target),count))
    return {ordered[index]["component_id"] for index in reachable[best_count]}


def build_split(ledger: dict, *, target_train_fraction: float = 0.82, seed: int = 261029139,
                expected_subject_count: int | None = None) -> dict:
    if not 0.0 < float(target_train_fraction) < 1.0: raise ValueError("TRUTH139_TRAIN_FRACTION_INVALID")
    rows=_admitted_rows(ledger,expected_subject_count); components=_components(rows)
    unseen_ids=_select_unseen_components(components,target_unseen_fraction=1.0-float(target_train_fraction),seed=int(seed))
    assignment={row["component_id"]:("UNSEEN" if row["component_id"] in unseen_ids else "TRAIN") for row in components}
    if any(assignment[row["component_id"]]=="UNSEEN" and row["unseen_eligible"] is not True for row in components):
        raise RuntimeError("TRUTH139_DEVELOPMENT_CONNECTED_COMPONENT_IN_UNSEEN")
    train_subjects=sorted(subject for component in components if assignment[component["component_id"]]=="TRAIN" for subject in component["subject_ids"])
    unseen_subjects=sorted(subject for component in components if assignment[component["component_id"]]=="UNSEEN" for subject in component["subject_ids"])
    if not train_subjects or not unseen_subjects: raise ValueError("TRUTH139_SPLIT_EMPTY_SIDE")
    if set(train_subjects)&set(unseen_subjects): raise RuntimeError("TRUTH139_SPLIT_SUBJECT_OVERLAP")
    component_rows=[{**row,"split":assignment[row["component_id"]]} for row in sorted(components,key=lambda item:item["component_id"])]
    payload={"schema":OUTPUT_SCHEMA,"source_ledger_sha256":ledger["ledger_identity_sha256"],
             "expected_subject_count":expected_subject_count,"admitted_subject_count":len(rows),"component_count":len(components),
             "unseen_eligible_component_count":sum(row["unseen_eligible"] is True for row in components),
             "target_train_fraction":float(target_train_fraction),"actual_train_fraction":len(train_subjects)/len(rows),"seed":int(seed),
             "split_unit":"LEAKAGE_CONNECTED_COMPONENT","required_lineage_fields":list(REQUIRED_LINEAGE_FIELDS),
             "train_subjects":train_subjects,"unseen_subjects":unseen_subjects,
             "train_subject_count":len(train_subjects),"unseen_subject_count":len(unseen_subjects),"components":component_rows,
             "development_connected_components_forced_out_of_unseen":True,"pack_level_split_used":False,
             "unseen_bytes_opened_for_fit":False,"checkpoint_selection_may_read_unseen":False,"optimizer_may_read_unseen":False,
             "training_authorized":False}
    payload["split_identity_sha256"]=_digest(payload)
    return payload


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); staged=path.with_suffix(path.suffix+".partial")
    staged.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n"); staged.replace(path)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--ledger",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--target-train-fraction",type=float,default=0.82);parser.add_argument("--seed",type=int,default=261029139)
    parser.add_argument("--expected-subject-count",type=int)
    args=parser.parse_args();ledger_path=Path(args.ledger).resolve()
    if not ledger_path.is_file(): raise FileNotFoundError(ledger_path)
    split=build_split(json.loads(ledger_path.read_text()),target_train_fraction=args.target_train_fraction,
                      seed=args.seed,expected_subject_count=args.expected_subject_count)
    output=Path(args.output).resolve();_atomic_json(output,split)
    print(json.dumps({"output":str(output),"split_identity_sha256":split["split_identity_sha256"],
                      "train_subject_count":split["train_subject_count"],"unseen_subject_count":split["unseen_subject_count"],
                      "component_count":split["component_count"]},sort_keys=True),flush=True)


if __name__=="__main__": main()
