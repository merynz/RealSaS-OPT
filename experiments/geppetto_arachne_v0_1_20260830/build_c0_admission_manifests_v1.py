from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GEPPETTO_C0_MAX_CONTROLS = 160

EXPECTED_AUDIT_STATUS = "PASS_MEASUREMENT_COMPLETE__POLICY_FREEZE_NEXT"
EXPECTED_SELECTION_SHA256 = "af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9"
EXPECTED_AUDITED_ASSET_SET_SHA256 = "908609d8044a96d812bd0e2f0ce928c470c1fca86e47c26ba03be0a2fb372e99"
EXPECTED_GEPPETTO_SCOPE = 2914
EXPECTED_ARACHNE_SCOPE = 2540
EXPECTED_GEPPETTO_C0 = 2897
EXPECTED_GEPPETTO_OVERFLOW = 17
EXPECTED_ARACHNE_C0 = 2527
EXPECTED_ARACHNE_EXCLUDED = 13


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json_sha256(obj) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return sha256_bytes(b)


def build(audit: dict) -> dict:
    if audit.get("status") != EXPECTED_AUDIT_STATUS:
        raise RuntimeError(f"BAD_AUDIT_STATUS:{audit.get('status')}")
    if audit.get("scope_complete") is not True:
        raise RuntimeError("AUDIT_SCOPE_INCOMPLETE")
    if audit.get("split") != "FIT":
        raise RuntimeError(f"BAD_SPLIT:{audit.get('split')}")
    if audit.get("selection_file_sha256") != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("SELECTION_AUTHORITY_DRIFT")
    if audit.get("audited_asset_set_sha256") != EXPECTED_AUDITED_ASSET_SET_SHA256:
        raise RuntimeError("AUDITED_ASSET_SET_DRIFT")
    if audit.get("training_authorized_by_this_audit") is not False:
        raise RuntimeError("AUDIT_AUTHORITY_SEMANTICS_DRIFT")

    rows = audit.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_GEPPETTO_SCOPE:
        raise RuntimeError(f"BAD_ROW_COUNT:{None if not isinstance(rows, list) else len(rows)}")
    if any(r.get("status") != "PASS" for r in rows):
        raise RuntimeError("NONPASS_ROW_IN_COMPLETE_AUDIT")

    g_ok, g_over, a_ok, a_ex = [], [], [], []
    seen = set()
    for r in rows:
        aid = r["canonical_asset_id"]
        if aid in seen:
            raise RuntimeError(f"DUPLICATE_ASSET:{aid}")
        seen.add(aid)
        k = int(r["rig"]["deform_control_count"])
        source = r["source_registry_id"]
        base = {
            "canonical_asset_id": aid,
            "source_registry_id": source,
            "deform_control_count": k,
        }
        if k > GEPPETTO_C0_MAX_CONTROLS:
            g_over.append({
                **base,
                "reason": f"STRUCTURAL_OVERFLOW_{k}_GT_{GEPPETTO_C0_MAX_CONTROLS}",
            })
        else:
            g_ok.append(base)

        if bool(r.get("arachne_capable")):
            skin = r.get("skin")
            if not isinstance(skin, dict):
                raise RuntimeError(f"ARACHNE_SKIN_AUDIT_MISSING:{aid}")
            mass = float(skin["nondeform_mass_total"])
            if k > GEPPETTO_C0_MAX_CONTROLS:
                a_ex.append({
                    **base,
                    "reason": f"STRUCTURAL_OVERFLOW_{k}_GT_{GEPPETTO_C0_MAX_CONTROLS}",
                })
            elif mass > 0.0:
                a_ex.append({
                    **base,
                    "nondeform_skin_mass_total": mass,
                    "reason": "NONDEFORM_SKIN_MASS_REQUIRES_UNFROZEN_SEMANTIC_TRANSPORT",
                })
            else:
                a_ok.append(base)

    for x in (g_ok, g_over, a_ok, a_ex):
        x.sort(key=lambda z: z["canonical_asset_id"])

    if len(g_ok) != EXPECTED_GEPPETTO_C0 or len(g_over) != EXPECTED_GEPPETTO_OVERFLOW:
        raise RuntimeError(f"GEPPETTO_COUNT_DRIFT:{len(g_ok)}:{len(g_over)}")
    arachne_scope = len(a_ok) + len(a_ex)
    if arachne_scope != EXPECTED_ARACHNE_SCOPE:
        raise RuntimeError(f"ARACHNE_SCOPE_DRIFT:{arachne_scope}")
    if len(a_ok) != EXPECTED_ARACHNE_C0 or len(a_ex) != EXPECTED_ARACHNE_EXCLUDED:
        raise RuntimeError(f"ARACHNE_COUNT_DRIFT:{len(a_ok)}:{len(a_ex)}")

    out = {
        "schema": "RealSaS.GeppettoArachne.C0AdmissionManifest.v1",
        "status": "PASS__C0_MEMBERSHIP_FROZEN",
        "authority": {
            "audit_status": audit["status"],
            "selection_file_sha256": audit["selection_file_sha256"],
            "audited_asset_set_sha256": audit["audited_asset_set_sha256"],
            "geppetto_c0_max_controls": GEPPETTO_C0_MAX_CONTROLS,
            "overflow_policy": "ABSTAIN_NO_TRUNCATION",
            "arachne_nondeform_skin_policy": "EXCLUDE_FROM_C0__NO_TRANSPORT__NO_RENORMALIZATION",
        },
        "counts": {
            "geppetto_fit_scope": len(rows),
            "geppetto_c0": len(g_ok),
            "geppetto_overflow": len(g_over),
            "arachne_fit_scope": arachne_scope,
            "arachne_c0": len(a_ok),
            "arachne_excluded": len(a_ex),
        },
        "geppetto_c0_assets": g_ok,
        "geppetto_overflow_assets": g_over,
        "arachne_c0_assets": a_ok,
        "arachne_excluded_assets": a_ex,
        "training_authorized": False,
    }
    out["membership_sha256"] = canonical_json_sha256({
        "geppetto_c0_assets": [x["canonical_asset_id"] for x in g_ok],
        "geppetto_overflow_assets": [x["canonical_asset_id"] for x in g_over],
        "arachne_c0_assets": [x["canonical_asset_id"] for x in a_ok],
        "arachne_excluded_assets": [x["canonical_asset_id"] for x in a_ex],
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-result", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    raw = args.audit_result.read_bytes()
    audit = json.loads(raw)
    out = build(audit)
    out["audit_result_file_sha256"] = sha256_bytes(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": out["status"],
        "counts": out["counts"],
        "membership_sha256": out["membership_sha256"],
        "audit_result_file_sha256": out["audit_result_file_sha256"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
