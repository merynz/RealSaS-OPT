from __future__ import annotations

"""Conservative RSS.vNext source-supervision coverage inventory.

Inputs MUST come from the independently verified licensed-source foundry.
This audit NEVER mints training admission, ownership, geometry or rig truth:
it only counts already issued per-source teacher receipt digests, exposes
coverage gaps, and rejects train/validation/unseen source-group leakage.

Nothing here converts diffusion views into original observations, an open
mesh into signed geometry, components into rig truth, or missing labels
into negative examples.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse
import json
import os
import tempfile

SCHEMA="RealSaS.RSSvNext.SourceSupervisionCoverageInventory.v1"
SPLITS=("TRAIN","VAL","UNSEEN")
TEACHER_FAMILIES=frozenset({
    "SOURCE_RENDER_RGB_ALPHA",
    "SOURCE_RENDER_SEMANTIC_LAYER",
    "SOURCE_SDF_SIGNED",
    "SOURCE_FLEXICUBES_SURFACE",
    "SOURCE_TESSA_TOPOLOGY",
    "SOURCE_RIG_JOINTS",
    "SOURCE_MIRA_DENSE_SKIN",
    "SOURCE_MIRA_AREA_SKIN",
    "SOURCE_POSED_LBS",
    "SOURCE_GRAPHICS_FIRST_HIT",
    "SOURCE_DRAWABLE",
    "SOURCE_ATTACHMENT_FACE",
})
# AND of required source families. No generated diffusion/teacher-free
# prediction counts as a ground truth supervision family.
STAGES={
    "SLRM_FOUNDATION": ("SOURCE_RENDER_RGB_ALPHA",),
    "IRIS_SEMANTIC_LAYER": ("SOURCE_RENDER_RGB_ALPHA","SOURCE_RENDER_SEMANTIC_LAYER"),
    "IRIS_SIGNED_SDF": ("SOURCE_SDF_SIGNED",),
    "IRIS_FLEXICUBES_SURFACE": ("SOURCE_FLEXICUBES_SURFACE",),
    "TESSA_TOPOLOGY": ("SOURCE_TESSA_TOPOLOGY",),
    "MIRA_SOURCE_CODEC": ("SOURCE_RIG_JOINTS","SOURCE_MIRA_DENSE_SKIN"),
    "MIRA_AREA_REFINEMENT": ("SOURCE_RIG_JOINTS","SOURCE_MIRA_DENSE_SKIN","SOURCE_MIRA_AREA_SKIN"),
    "AXIS_PRODUCT_MATCHED_RIG": ("SOURCE_RIG_JOINTS","SOURCE_TESSA_TOPOLOGY"),
    "PRODUCT_POSED_CONSEQUENCE": (
        "SOURCE_RIG_JOINTS","SOURCE_MIRA_DENSE_SKIN",
        "SOURCE_TESSA_TOPOLOGY","SOURCE_POSED_LBS",
    ),
    "GRAPHICS_SOURCE_RGB": ("SOURCE_GRAPHICS_FIRST_HIT",),
    "DRAWABLE_SUPERVISION": ("SOURCE_DRAWABLE",),
    "ATTACHMENT_FACE_SUPERVISION": (
        "SOURCE_ATTACHMENT_FACE","SOURCE_RIG_JOINTS","SOURCE_TESSA_TOPOLOGY",
    ),
}


def _digest(value):
    return type(value) is str and len(value)==64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def _license_allowed(row):
    """Never infer rights from a filename; NoAI always excludes the asset."""
    info=row.get("license")
    if type(info) is not dict:
        raise ValueError("RSS_VNEXT_COVERAGE_LICENSE_REQUIRED")
    ident=info.get("spdx_id")
    noai=info.get("explicit_noai")
    proof=info.get("license_evidence_sha256")
    attribution=info.get("attribution_sha256")
    if (type(ident) is not str or type(noai) is not bool
        or not _digest(proof)
        or attribution is not None and not _digest(attribution)):
        raise ValueError("RSS_VNEXT_COVERAGE_LICENSE_RECEIPT_INVALID")
    allowed=(ident in ("CC0-1.0","MIT")
             or ident.startswith("CC-BY-"))
    if ident.startswith("CC-BY-") and not _digest(attribution):
        return False,"MISSING_CC_BY_ATTRIBUTION_RECEIPT"
    if noai:
        return False,"EXPLICIT_NOAI"
    if not allowed:
        return False,"LICENSE_NOT_IN_ALLOWED_SET"
    return True,"LICENSE_DECLARED_ALLOWED_NOT_LEGAL_PROOF"


def _validate_row(row):
    if type(row) is not dict:
        raise ValueError("RSS_VNEXT_COVERAGE_ROW_INVALID")
    if (
        row.get("schema")!=SCHEMA
        or not _digest(row.get("asset_sha256"))
        or not _digest(row.get("source_group_sha256"))
        or row.get("split") not in SPLITS
    ):
        raise ValueError("RSS_VNEXT_COVERAGE_ROW_IDENTITY_INVALID")
    observed=row.get("original_source_rgba_sha256")
    if (type(observed) is not list or len(observed)!=2
        or any(not _digest(x) for x in observed)
        or observed[0]==observed[1]):
        raise ValueError("RSS_VNEXT_COVERAGE_TWO_DISTINCT_ORIGINAL_IMAGES_REQUIRED")
    teachers=row.get("qualified_teacher_receipts")
    if (type(teachers) is not dict
        or any(k not in TEACHER_FAMILIES or not _digest(v)
               for k,v in teachers.items())):
        raise ValueError("RSS_VNEXT_COVERAGE_TEACHER_RECEIPT_DOMAIN")
    return _license_allowed(row)


def audit_rss_vnext_coverage_v1(rows, *, train_asset_target=10000):
    """Validate split + rights scope, then count claimed receipts by split.

    This is a coverage *inventory*; receipt validity/lineage must be
    independently checked against the trusted source-registry before a
    trainer can use the asset. Count != trainer admission.
    """
    if (type(train_asset_target) is not int or train_asset_target<1
        or type(rows) not in (list,tuple)):
        raise ValueError("RSS_VNEXT_COVERAGE_INPUT_INVALID")
    seen_assets=set()
    seen_groups={}
    accepted={split:[] for split in SPLITS}
    rejected=Counter()
    total=0
    for row in rows:
        total+=1
        allowed,reason=_validate_row(row)
        asset=row["asset_sha256"]
        group=row["source_group_sha256"]
        split=row["split"]
        if asset in seen_assets:
            raise ValueError("RSS_VNEXT_COVERAGE_DUPLICATE_ASSET_SHA")
        seen_assets.add(asset)
        previous=seen_groups.setdefault(group,split)
        if previous!=split:
            raise ValueError("RSS_VNEXT_COVERAGE_SOURCE_GROUP_SPLIT_LEAKAGE")
        if allowed:
            accepted[split].append(row)
        else:
            rejected[reason]+=1
    families={
        split:{family:sum(
            family in row["qualified_teacher_receipts"]
            for row in accepted[split]
        ) for family in sorted(TEACHER_FAMILIES)}
        for split in SPLITS
    }
    stages={
        split:{name:sum(
            all(f in row["qualified_teacher_receipts"] for f in required)
            for row in accepted[split]
        ) for name,required in STAGES.items()}
        for split in SPLITS
    }
    # Hash the exact reported evidence and membership, independently of
    # input ordering, so later inventory comparisons are stable.
    serialized=json.dumps(
        sorted(rows,key=lambda row:row["asset_sha256"]),
        sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode("utf8")
    counts={s:len(accepted[s]) for s in SPLITS}
    return {
        "schema":SCHEMA,
        "authority":"INSPECTION_ONLY_NOT_LICENSE_OR_TEACHER_ADMISSION",
        "input_records":total,
        "inventory_sha256":sha256(serialized).hexdigest(),
        "accepted_license_counts_by_split":counts,
        "rejected_license_counts_by_reason":dict(sorted(rejected.items())),
        "declared_teacher_receipt_counts_by_split":families,
        "declared_stage_candidate_counts_by_split":stages,
        "train_asset_target":train_asset_target,
        "train_asset_target_met":counts["TRAIN"]>=train_asset_target,
        "unseen_split_nonempty":counts["UNSEEN"]>0,
        "source_group_split_disjoint":True,
        "training_authorized":False,
        "unseen_product_quality_proven":False,
    }


def _read_jsonl(path):
    rows=[]
    with Path(path).open("rt",encoding="utf8") as stream:
        for lineno,line in enumerate(stream,1):
            if not line.strip():
                continue
            try:
                record=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"RSS_VNEXT_COVERAGE_BAD_JSONL_LINE:{lineno}"
                ) from exc
            rows.append(record)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--qualified-source-jsonl",required=True,type=Path)
    parser.add_argument("--output-json",required=True,type=Path)
    parser.add_argument("--train-asset-target",type=int,default=10000)
    args=parser.parse_args()
    output=audit_rss_vnext_coverage_v1(
        _read_jsonl(args.qualified_source_jsonl),
        train_asset_target=args.train_asset_target,
    )
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wt",encoding="utf8",dir=args.output_json.parent,
        prefix="rss-source-audit-",suffix=".json",delete=False,
    ) as stream:
        temp=Path(stream.name)
        json.dump(output,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temp,args.output_json)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({
        "schema":SCHEMA,
        "accepted_by_split":output["accepted_license_counts_by_split"],
        "train_asset_target_met":output["train_asset_target_met"],
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
