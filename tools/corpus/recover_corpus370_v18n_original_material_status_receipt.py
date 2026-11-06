#!/usr/bin/env python3
"""Lossless status-only migration of frozen V18n original-source render manifest.

This does not change or repair original Cycles pixels, geometry, appearance,
source alpha, or scientific court threshold. It permits an independent court
to read *exactly the same immutable original NPZ* after a producer copied an
obsolete literal V18l status. Writes original/canonical SHA lineage receipt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

RENDER_SCHEMA="RealSaS.Corpus370OriginalArtistPolicyMaterialSampleReplay.v18n"
WRONG="V18L_ORIGINAL_CYCLES_COMMON_SOURCE_SAMPLE_RENDERED__INDEPENDENT_COURT_REQUIRED"
CORRECT="V18N_ORIGINAL_MATERIAL_CYCLES_SOURCE_REPLAYED__INDEPENDENT_COURT_REQUIRED"
RECEIPT_SCHEMA="RealSaS.Corpus370OriginalMaterialStatusOnlyRecoveryReceipt.v18n"
RECEIPT_STATUS="ORIGINAL_SOURCE_RENDER_BYTES_IMMUTABLE__STATUS_LITERAL_ONLY_CORRECTED"


def require(value,msg):
    if not value:raise ValueError(msg)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def recover(raw_manifest,raw_npz,expected_raw_sha):
    require(isinstance(expected_raw_sha,str)
        and len(expected_raw_sha)==64 and sha(raw_manifest)==expected_raw_sha,
        "V18N_RECOVERY_SOURCE_ORIGINAL_MANIFEST_SHA_MISMATCH")
    m=json.loads(Path(raw_manifest).read_text())
    require(m.get("schema")==RENDER_SCHEMA
        and m.get("status")==WRONG
        and m.get("new_source_actual_edited_rgba_npz_sha256")==sha(raw_npz)
        and m.get("source_edit_arms")==
           ["SOURCE_NATIVE","SOUP_FULL","SOUP_ISOLATE","SOUP_REMOVE"]
        and m.get("source_material_policy")==
           "ORIGINAL_SEALED_SOURCE_MATERIAL_GRAPH_AND_UV_REPLAY"
        and m.get("source_appearance_material_policy")==
           "SEALED_SOURCE_APPEARANCE_POLICY_REPLAY"
        and m.get("heldout_source_camera_views")==[0,2,4,6]
        and m.get("original_source_cycles_sample_indices")==[0,4,8,12,16,20,24,28]
        and m.get("source_original_v18n_parent_v18m_run")=="42135271057-1"
        and m.get("source_original_v18n_parent_v18l_run")=="42131579613-1"
        and m.get("full_original_caa_draw_order_qualified") is False
        and m.get("source_vs_original_v14_RGBA_reference_source_parity_verified") is False
        and all(m.get(k) is False for k in
            ("graphics_owner_authorized","iris_training_authorized",
             "tessa_training_authorized","materialization_authorized",
             "training_authorized")),
        "V18N_RECOVERY_SOURCE_RECEIPT_NOT_STATUS_ONLY")
    canonical=dict(m)
    canonical["status"]=CORRECT
    require(set(canonical)==set(m)
        and all(canonical[k]==m[k] for k in m if k!="status"),
        "V18N_RECOVERY_UNDECLARED_RENDER_MANIFEST_CHANGE")
    return canonical


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-manifest",type=Path,required=True)
    p.add_argument("--source-rgba",type=Path,required=True)
    p.add_argument("--expected-original-source-manifest-sha",required=True)
    p.add_argument("--canonical-manifest",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    canonical=recover(a.source_manifest,a.source_rgba,
                      a.expected_original_source_manifest_sha)
    a.canonical_manifest.parent.mkdir(parents=True,exist_ok=True)
    a.canonical_manifest.write_text(json.dumps(
        canonical,indent=2,sort_keys=True,allow_nan=False)+"\n")
    receipt={
       "schema":RECEIPT_SCHEMA,"status":RECEIPT_STATUS,
       "original_render_manifest_sha256":sha(a.source_manifest),
       "canonical_render_manifest_sha256":sha(a.canonical_manifest),
       "original_cycles_source_rgba_sha256":sha(a.source_rgba),
       "only_field_changed":"status",
       "original_incorrect_literal":WRONG,
       "canonical_literal":CORRECT,
       "source_material_policy":canonical["source_material_policy"],
       "source_3d_geometry_unchanged":True,
       "original_caa_rgba_values_unchanged":True,
       "original_source_sha256":canonical["original_source_sha256"],
       "source_artist_ownership_admitted":False,
       "source_material_fidelity_admitted":False,
       "graphics_owner_authorized":False,
       "iris_training_authorized":False,
       "tessa_training_authorized":False,
       "materialization_authorized":False,
       "training_authorized":False,
    }
    a.receipt.write_text(json.dumps(
        receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":RECEIPT_STATUS,
        "original_source_manifest_sha":receipt["original_render_manifest_sha256"],
        "canonical_source_manifest_sha":receipt["canonical_render_manifest_sha256"],
        "source_rgba_immutable_sha":receipt["original_cycles_source_rgba_sha256"],
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
