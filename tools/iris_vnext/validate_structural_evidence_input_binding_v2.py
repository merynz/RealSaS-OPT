from __future__ import annotations

"""Fail-closed static and payload-semantic validator for structural evidence V2.

A hash match is necessary but not sufficient: admission opens every NPZ and
verifies CSR order, tie abstention, explicit global visibility validity,
positive-only occlusion support, pair/stack consistency and orientation symmetry.
This tool never trains or authorizes a scientific court.
"""

import argparse, hashlib, json, re
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA = "RealSaS.StructuralEvidenceInputBinding.v2"
OUTPUT_SCHEMA = "RealSaS.IRISVNextFit8AllHitRelationEvidenceManifest.v2"
OUTPUT_STATUS = "MATERIALIZED__TIE_SAFE_EVIDENCE_ONLY__NO_TRAINING_AUTHORIZATION"
SUBJECTS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$"); GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()


def _req(ok: bool, code: str) -> None:
    if not ok: raise ValueError(code)


def _sha_ok(v: Any) -> bool: return isinstance(v,str) and bool(SHA256_RE.fullmatch(v))


def _subject_order(rows: Any, code: str) -> None:
    _req(isinstance(rows,list) and len(rows)==8, code+"_COUNT")
    _req(tuple(str(x.get("subject_id")) for x in rows)==SUBJECTS, code+"_ORDER")


def validate_binding(v: dict[str,Any]) -> dict[str,Any]:
    _req(v.get("schema")==SCHEMA, "STRUCTURAL_EVIDENCE_V2_BINDING_SCHEMA_DRIFT")
    _req(v.get("experiment_id")=="STRUCTURAL-EVIDENCE-INPUT-AUTHORITY-V2", "STRUCTURAL_EVIDENCE_V2_EXPERIMENT_DRIFT")
    for k in ("scientific_execution_authorized","optimizer_or_training_authorized","notebook_execution_authorized"):
        _req(v.get(k) is False, "STRUCTURAL_EVIDENCE_V2_AUTHORIZATION_DRIFT:"+k)
    status=v.get("status")
    _req(status in {"SOURCE_FREEZE_PENDING__NO_EXECUTION_AUTHORITY","READY_FOR_ZERO_OPTIMIZER_MATERIALIZATION__NOT_SCIENTIFIC_PASS"}, "STRUCTURAL_EVIDENCE_V2_STATUS_INVALID")
    impl=v.get("implementation_source_sha")
    if status.startswith("READY_"): _req(isinstance(impl,str) and bool(GIT_SHA_RE.fullmatch(impl)), "STRUCTURAL_EVIDENCE_V2_SOURCE_SHA_MISSING")
    else: _req(impl is None, "STRUCTURAL_EVIDENCE_V2_PENDING_SHA_MUST_BE_NULL")
    p=v.get("producer",{})
    _req(p.get("path")=="tools/iris_vnext/build_fit8_relation_all_hit_evidence_v2.py", "STRUCTURAL_EVIDENCE_V2_PRODUCER_DRIFT")
    _req(p.get("candidate_rays_per_view")==8192 and p.get("grid_size")==32, "STRUCTURAL_EVIDENCE_V2_POLICY_DRIFT")
    _req(p.get("output_manifest")=="ALL_HIT_RELATION_EVIDENCE_V2_MANIFEST.json", "STRUCTURAL_EVIDENCE_V2_MANIFEST_NAME_DRIFT")
    inputs=v.get("source_inputs",{}); stage=inputs.get("stage01",{}); legacy=inputs.get("legacy_fit8",{}); teachers=inputs.get("authored_teachers",{}); prepared=inputs.get("prepared_fit8_receipt",{})
    _req(stage.get("file")=="FIT8_PLATFORM_STAGE01_INPUT_BUNDLE_20261013.zip" and _sha_ok(stage.get("sha256")) and _sha_ok(stage.get("bootstrap_sha256")), "STRUCTURAL_EVIDENCE_V2_STAGE01_DRIFT")
    _req(legacy.get("binding_path")=="canonical/VNEXT_FIT8_INPUT_BINDING_V1.json" and _sha_ok(legacy.get("binding_sha256")) and _sha_ok(legacy.get("cohort_manifest_sha256")), "STRUCTURAL_EVIDENCE_V2_LEGACY_DRIFT")
    _subject_order(legacy.get("subjects"), "STRUCTURAL_EVIDENCE_V2_LEGACY_SUBJECT")
    for r in legacy["subjects"]: _req(_sha_ok(r.get("sha256")) and str(r.get("shard_file","")).endswith(".pt"), "STRUCTURAL_EVIDENCE_V2_LEGACY_ROW_INVALID")
    _req(bool(teachers.get("folder_id")) and _sha_ok(teachers.get("batch_receipt_sha256")), "STRUCTURAL_EVIDENCE_V2_TEACHER_DRIFT")
    _subject_order(teachers.get("subjects"), "STRUCTURAL_EVIDENCE_V2_TEACHER_SUBJECT")
    for r in teachers["subjects"]: _req(_sha_ok(r.get("teacher_sha256")) and _sha_ok(r.get("receipt_sha256")) and int(r.get("sampled_frame_count",0))>0, "STRUCTURAL_EVIDENCE_V2_TEACHER_ROW_INVALID")
    _req(bool(prepared.get("folder_id")) and _sha_ok(prepared.get("preparation_receipt_sha256")) and _sha_ok(prepared.get("cohort_manifest_sha256")), "STRUCTURAL_EVIDENCE_V2_PREPARED_DRIFT")
    law=v.get("authority_law",{})
    _req(law.get("front")=="PAIRWISE_STRICT_DEPTH_ORDER__EQUAL_DEPTH_WITHIN_EPS_ABSTAIN", "STRUCTURAL_EVIDENCE_V2_FRONT_LAW_DRIFT")
    _req(law.get("global_visibility")=="UNIQUE_RANK_ZERO_ONLY__TIED_MINIMA_ABSTAIN__DEEPER_HITS_VALID_INVISIBLE", "STRUCTURAL_EVIDENCE_V2_VISIBILITY_LAW_DRIFT")
    _req(law.get("incomplete_stack_visibility_policy")=="ABSTAIN", "STRUCTURAL_EVIDENCE_V2_INCOMPLETE_LAW_DRIFT")
    _req(law.get("occlusion")=="POSITIVE_SAME_RAY_OVERLAP_SUPPORT_ONLY__NO_FABRICATED_NEGATIVES", "STRUCTURAL_EVIDENCE_V2_OCCLUSION_LAW_DRIFT")
    _req(law.get("pair_point_coordinate_domain")=="EXACT_LEGACY_IRIS_NORMALIZED_SOURCE_WORLD", "STRUCTURAL_EVIDENCE_V2_COORD_DOMAIN_DRIFT")
    _req(law.get("tessa_coordinate_conversion")=="MULTIPLY_IRIS_XYZ_BY_0.5_BEFORE_TESSA_MEMORY", "STRUCTURAL_EVIDENCE_V2_TESSA_COORD_DRIFT")
    for k in ("material_id_as_owner_truth","named_semantic_taxonomy_as_structural_authority","teacher_topology_or_rig_in_inference_forward"): _req(law.get(k) is False, "STRUCTURAL_EVIDENCE_V2_SHORTCUT:"+k)
    old=v.get("supersedes_v1",{})
    _req(old.get("scientific_input_authorized") is False and old.get("equal_depth_base_pairs")==25 and old.get("equal_depth_oriented_rows")==50 and old.get("subject")=="Bat", "STRUCTURAL_EVIDENCE_V2_V1_AUDIT_DRIFT")
    out=v.get("required_output",{})
    _req(out.get("manifest_schema")==OUTPUT_SCHEMA and out.get("manifest_status")==OUTPUT_STATUS and tuple(out.get("subject_order",()))==SUBJECTS and out.get("subject_count")==8, "STRUCTURAL_EVIDENCE_V2_OUTPUT_CONTRACT_DRIFT")
    _req(out.get("per_subject_file_suffix")==".all_hit_relation_evidence_v2.npz", "STRUCTURAL_EVIDENCE_V2_OUTPUT_SUFFIX_DRIFT")
    _req(out.get("required_manifest_flags")=={"training_authorized":False,"scientific_fit_started":False,"teacher_inputs_in_inference_forward":False}, "STRUCTURAL_EVIDENCE_V2_OUTPUT_FLAGS_DRIFT")
    for k in ("materialization_is_scientific_pass","scientific_court_authorized","relation_training_authorized","tessa_training_authorized"):
        _req(v.get("post_materialization_gate",{}).get(k) is False, "STRUCTURAL_EVIDENCE_V2_POST_AUTHORIZATION:"+k)
    return v


def _validate_view(data: Any, prefix: str, eps: float, subject: str, view: int) -> dict[str,int]:
    def a(name,dtype=None):
        key=prefix+name; _req(key in data, f"V2_ARRAY_MISSING:{subject}:v{view}:{name}"); return np.asarray(data[key],dtype=dtype)
    offsets=a("stack_offsets",np.int64); faces=a("stack_face",np.int64); depth=a("stack_depth",np.float64)
    sv=a("stack_visibility_probability",np.float64); svv=a("stack_visibility_valid",bool); tied=a("stack_tied_minimum",bool)
    _req(offsets.ndim==1 and len(offsets)>=2 and offsets[0]==0 and np.all(np.diff(offsets)>=0), f"V2_CSR_OFFSETS:{subject}:v{view}")
    _req(int(offsets[-1])==len(faces)==len(depth)==len(sv)==len(svv)==len(tied), f"V2_CSR_CARDINALITY:{subject}:v{view}")
    _req(np.isfinite(depth).all() and np.all(depth>0), f"V2_DEPTH_INVALID:{subject}:v{view}")
    _req(np.all((sv==0)|(sv==1)) and np.all(sv[~svv]==0), f"V2_VIS_VALUE_INVALID:{subject}:v{view}")
    for ray in range(len(offsets)-1):
        s,e=int(offsets[ray]),int(offsets[ray+1]); d=depth[s:e]
        if not len(d): continue
        _req(np.all(np.diff(d)>=0), f"V2_DEPTH_ORDER:{subject}:v{view}:{ray}")
        minmask=np.abs(d-d[0])<=eps
        if int(minmask.sum())==1:
            _req(bool(svv[s]) and sv[s]==1 and not tied[s], f"V2_UNIQUE_MIN_VIS:{subject}:v{view}:{ray}")
        else:
            _req(np.all(~svv[s:e][minmask]) and np.all(sv[s:e][minmask]==0) and np.all(tied[s:e][minmask]), f"V2_TIED_MIN_ABSTAIN:{subject}:v{view}:{ray}")
        if np.any(~minmask):
            _req(np.all(svv[s:e][~minmask]) and np.all(sv[s:e][~minmask]==0) and np.all(~tied[s:e][~minmask]), f"V2_DEEP_VIS:{subject}:v{view}:{ray}")
    names=("face_a","face_b","rank_a","rank_b","ray_index","depth_a","depth_b","front_probability","front_valid","visibility_probability_a","visibility_probability_b","visibility_valid_a","visibility_valid_b","occlusion_positive_support","base_pair_id")
    arr={n:a(n) for n in names}; n=len(arr["front_probability"])
    _req(n%2==0 and all(len(x)==n for x in arr.values()), f"V2_PAIR_CARDINALITY:{subject}:v{view}")
    fp=np.asarray(arr["front_probability"],float); fv=np.asarray(arr["front_valid"],bool); da=np.asarray(arr["depth_a"],float); db=np.asarray(arr["depth_b"],float)
    istie=np.abs(da-db)<=eps
    _req(np.array_equal(fv,~istie), f"V2_FRONT_VALIDITY:{subject}:v{view}")
    _req(np.all(fp[~fv]==0) and np.array_equal(fp[fv],(da[fv]<db[fv]).astype(float)), f"V2_FRONT_VALUE:{subject}:v{view}")
    _req(np.all(np.asarray(arr["occlusion_positive_support"],float)==1.0), f"V2_OCCLUSION_SUPPORT:{subject}:v{view}")
    ray=np.asarray(arr["ray_index"],int); ra=np.asarray(arr["rank_a"],int); rb=np.asarray(arr["rank_b"],int)
    flat_a=offsets[ray]+ra; flat_b=offsets[ray]+rb
    _req(np.array_equal(np.asarray(arr["visibility_probability_a"],float),sv[flat_a]) and np.array_equal(np.asarray(arr["visibility_probability_b"],float),sv[flat_b]), f"V2_PAIR_VIS_VALUE:{subject}:v{view}")
    _req(np.array_equal(np.asarray(arr["visibility_valid_a"],bool),svv[flat_a]) and np.array_equal(np.asarray(arr["visibility_valid_b"],bool),svv[flat_b]), f"V2_PAIR_VIS_VALID:{subject}:v{view}")
    if n:
        even=np.arange(0,n,2); odd=even+1
        _req(np.array_equal(arr["base_pair_id"][even],arr["base_pair_id"][odd]), f"V2_PAIR_ID_SYMMETRY:{subject}:v{view}")
        _req(np.array_equal(arr["face_a"][even],arr["face_b"][odd]) and np.array_equal(arr["face_b"][even],arr["face_a"][odd]), f"V2_PAIR_FACE_SYMMETRY:{subject}:v{view}")
        _req(np.array_equal(fv[even],fv[odd]) and np.array_equal(fp[even][fv[even]],1.0-fp[odd][fv[odd]]), f"V2_PAIR_FRONT_SYMMETRY:{subject}:v{view}")
    return {"rows":n,"front_abstain":int((~fv).sum()),"visibility_abstain_slots":int((~np.asarray(arr["visibility_valid_a"],bool)).sum()+(~np.asarray(arr["visibility_valid_b"],bool)).sum())}


def validate_materialized(binding: dict[str,Any], output_dir: Path) -> dict[str,Any]:
    validate_binding(binding); _req(binding.get("status")=="READY_FOR_ZERO_OPTIMIZER_MATERIALIZATION__NOT_SCIENTIFIC_PASS", "STRUCTURAL_EVIDENCE_V2_BINDING_NOT_FROZEN")
    mp=output_dir/binding["producer"]["output_manifest"]; _req(mp.is_file(),"STRUCTURAL_EVIDENCE_V2_MANIFEST_MISSING")
    m=json.loads(mp.read_text()); out=binding["required_output"]
    _req(m.get("schema")==out["manifest_schema"] and m.get("status")==out["manifest_status"],"STRUCTURAL_EVIDENCE_V2_MANIFEST_CONTRACT_DRIFT")
    for k,e in out["required_manifest_flags"].items(): _req(m.get(k) is e,"STRUCTURAL_EVIDENCE_V2_MANIFEST_FLAG:"+k)
    _req(tuple(m.get("subject_order",()))==SUBJECTS,"STRUCTURAL_EVIDENCE_V2_SUBJECT_ORDER")
    _req(m.get("authority",{}).get("front")=="PAIRWISE_STRICT_DEPTH_ORDER__EQUAL_DEPTH_WITHIN_EPS_ABSTAIN","STRUCTURAL_EVIDENCE_V2_MANIFEST_FRONT")
    _req(m.get("authority",{}).get("visibility")=="UNIQUE_MINIMUM_VISIBLE__TIED_MINIMA_ABSTAIN__DEEPER_VALID_INVISIBLE","STRUCTURAL_EVIDENCE_V2_MANIFEST_VIS")
    rows=m.get("subjects"); _subject_order(rows,"STRUCTURAL_EVIDENCE_V2_OUTPUT_SUBJECT")
    totals={"rows":0,"front_abstain":0,"visibility_abstain_slots":0}
    for row in rows:
        subject=row["subject_id"]; f=output_dir/row["file"]; _req(f.is_file() and _sha_ok(row.get("sha256")) and _sha(f)==row["sha256"],"STRUCTURAL_EVIDENCE_V2_HASH:"+subject)
        eps=float(row.get("depth_tie_epsilon",0)); _req(np.isfinite(eps) and eps>0,"STRUCTURAL_EVIDENCE_V2_EPS:"+subject)
        with np.load(f,allow_pickle=False) as data:
            _req("subject_depth_tie_epsilon" in data and float(data["subject_depth_tie_epsilon"][0])==eps,"STRUCTURAL_EVIDENCE_V2_EPS_PAYLOAD:"+subject)
            for view in range(8):
                stats=_validate_view(data,f"v{view}_",eps,subject,view)
                for k in totals: totals[k]+=stats[k]
                declared=row["views"][view]
                _req(int(declared["oriented_pair_rows"])==stats["rows"] and int(declared["front_abstain_rows"])==stats["front_abstain"],f"STRUCTURAL_EVIDENCE_V2_STATS:{subject}:v{view}")
    return {"schema":"RealSaS.StructuralEvidenceMaterializationAdmission.v2","status":"ADMITTED_TIE_SAFE_HASHED_EVIDENCE_ONLY__NO_SCIENTIFIC_PASS","subject_count":8,"manifest_sha256":_sha(mp),"implementation_source_sha":binding["implementation_source_sha"],"payload_totals":totals,"training_authorized":False,"scientific_court_authorized":False}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--binding",type=Path,required=True); p.add_argument("--materialized-dir",type=Path); a=p.parse_args()
    b=json.loads(a.binding.read_text()); validate_binding(b)
    if a.materialized_dir is None: print("STRUCTURAL_EVIDENCE_INPUT_BINDING_V2_PASS",b["status"],flush=True)
    else: print("STRUCTURAL_EVIDENCE_MATERIALIZATION_V2_ADMISSION_PASS",json.dumps(validate_materialized(b,a.materialized_dir),sort_keys=True),flush=True)

if __name__=="__main__": main()
