#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from skeleton_projection_corpus_audit_v1 import (
    aggregate_projection_audit_v1,
    audit_projection_asset_v1,
)

DEFAULT_ROOT = "/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3"
DEFAULT_SPLIT = "FIT"
REQUIRED_RIG_KEYS = ("bone_heads", "bone_tails", "parents", "deform_mask")
OPTIONAL_KEYS = ("skin",)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_selected_v1(root: Path) -> tuple[list[dict[str, Any]], Path]:
    p = root / "metadata" / "CANONICAL_VARIANT_SELECTION.json"
    raw = read_json(p)
    if isinstance(raw, dict) and isinstance(raw.get("selected"), list):
        selected = raw["selected"]
    elif isinstance(raw, dict):
        selected = list(raw.values())
    else:
        selected = raw
    if not isinstance(selected, list) or not all(isinstance(x, dict) for x in selected):
        raise RuntimeError("unexpected CANONICAL_VARIANT_SELECTION schema")
    return selected, p


def load_primary_arrays_v1(path: Path, *, require_skin: bool) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        missing = [k for k in REQUIRED_RIG_KEYS if k not in z.files]
        if require_skin and "skin" not in z.files:
            missing.append("skin")
        if missing:
            raise ValueError(f"primary_geometry missing required arrays: {sorted(set(missing))}; keys={keys}")
        arrays = {k: np.asarray(z[k]) for k in REQUIRED_RIG_KEYS}
        if "skin" in z.files:
            arrays["skin"] = np.asarray(z["skin"])
    return arrays, keys


def render_markdown_v1(result: dict[str, Any]) -> str:
    a = result["aggregate"]
    s = a.get("skin", {})
    controls = a["deform_control_count"]
    roots = a["projected_root_count"]
    hops = a["helper_skip_hops_global"]
    lines = [
        "# Geppetto/Arachne V0.1 — FIT Skeleton Projection Corpus Audit V1",
        "",
        f"Status: **`{result['status']}`**",
        f"Scope complete: **{result['scope_complete']}**",
        f"Master split: **`{result['split']}`**",
        "",
        "## Census",
        "",
        f"- Geppetto-capable FIT assets audited: **{a['asset_count']}**",
        f"- hard contract PASS: **{a['pass_count']}**",
        f"- hard contract FAIL: **{a['fail_count']}**",
        f"- Arachne-capable skin assets audited: **{a['arachne_skin_asset_count']}**",
        "",
        "## Skeleton projection",
        "",
        f"- deform controls min / p50 / p95 / p99 / max: **{controls['min']} / {controls['p50']} / {controls['p95']} / {controls['p99']} / {controls['max']}**",
        f"- projected roots min / p50 / p95 / max: **{roots['min']} / {roots['p50']} / {roots['p95']} / {roots['max']}**",
        f"- multi-root assets: **{a['multi_root_asset_count']}**",
        f"- assets with helper-chain skip: **{a['assets_with_any_helper_skip']}**",
        f"- helper skip hops p50 / p95 / p99 / max: **{hops['p50']} / {hops['p95']} / {hops['p99']} / {hops['max']}**",
        f"- maximum helper-chain skip hops: **{a['max_helper_skip_hops']}**",
        f"- assets with exact zero-length deform bone: **{a['asset_with_exact_zero_length_deform_bone_count']}**",
        f"- exact zero-length deform bones total: **{a['exact_zero_length_deform_bone_total']}**",
        f"- assets with near-zero deform bone (`1e-8 * max(bbox_diag,1)` diagnostic): **{a['asset_with_near_zero_length_deform_bone_count_rel1e8_bbox']}**",
        "",
        "## Arachne skin transport evidence",
        "",
    ]
    if s:
        lines += [
            f"- corpus non-deform skin mass fraction: **{s['corpus_nondeform_mass_fraction']:.10g}**",
            f"- assets with non-deform skin mass: **{s['assets_with_nondeform_mass']}**",
            f"- candidate nearest-deform-ancestor untransportable fraction of non-deform mass: **{s['candidate_nearest_ancestor_untransportable_fraction_of_nondeform_mass']:.10g}**",
            f"- assets with untransportable non-deform skin mass: **{s['assets_with_untransportable_nondeform_mass']}**",
            "- **No skin mass was transported or repaired by this audit.**",
        ]
    else:
        lines.append("- No Arachne-capable skin rows were present in the audited scope.")
    lines += [
        "",
        "## Decision boundary",
        "",
        "This audit measures the preregistered FIT data-plane facts only. It does **not** choose `max_controls`, rewrite `SkeletonTeacherProjectionV1`, choose Arachne helper-column transport, train a model, open DEV/VALIDATION/SEALED sets, or authorize product qualification.",
        "",
        "The next decision is architecture-policy freeze from these measurements; model outputs must remain unseen until that freeze is committed.",
        "",
        "## Provenance",
        "",
        f"- selection file SHA-256: `{result['selection_file_sha256']}`",
        f"- audited asset-set SHA-256: `{result['audited_asset_set_sha256']}`",
        f"- result schema: `{result['schema']}`",
        "",
    ]
    if a["failure_examples"]:
        lines += ["## Hard failure examples", "", "```json", json.dumps(a["failure_examples"], indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--max-assets", type=int, default=0, help="0 = full selected scope; positive values are smoke-only")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    selected, selection_path = load_selected_v1(root)

    target: list[dict[str, Any]] = []
    for row in selected:
        caps = row.get("capabilities") or {}
        if str(row.get("split")) != args.split:
            continue
        if not bool(caps.get("geppetto")):
            continue
        target.append(row)
    target.sort(key=lambda x: str(x["canonical_asset_id"]))

    ids = [str(x["canonical_asset_id"]) for x in target]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate canonical_asset_id in selected Geppetto FIT scope")
    full_count = len(target)
    if args.max_assets > 0:
        target = target[: args.max_assets]
    scope_complete = len(target) == full_count

    rows = []
    for i, sel in enumerate(target, 1):
        aid = str(sel["canonical_asset_id"])
        caps = sel.get("capabilities") or {}
        p = root / "master" / "assets" / aid / "primary_geometry.npz"
        if not p.is_file():
            row = {
                "canonical_asset_id": aid,
                "source_registry_id": sel.get("source_registry_id"),
                "arachne_capable": bool(caps.get("arachne")),
                "status": "FAIL",
                "errors": [f"FileNotFoundError: {p}"],
            }
        else:
            try:
                arrays, inventory = load_primary_arrays_v1(p, require_skin=bool(caps.get("arachne")))
                row = audit_projection_asset_v1(
                    canonical_asset_id=aid,
                    source_registry_id=sel.get("source_registry_id"),
                    arrays=arrays,
                    arachne_capable=bool(caps.get("arachne")),
                )
                row["primary_geometry_array_inventory"] = inventory
            except Exception as e:
                row = {
                    "canonical_asset_id": aid,
                    "source_registry_id": sel.get("source_registry_id"),
                    "arachne_capable": bool(caps.get("arachne")),
                    "status": "FAIL",
                    "errors": [f"{type(e).__name__}: {e}"],
                }
        rows.append(row)
        if i % 100 == 0 or i == len(target):
            print(f"[skeleton-projection-audit] {i}/{len(target)}", flush=True)

    aggregate = aggregate_projection_audit_v1(rows)
    audited_ids = [r["canonical_asset_id"] for r in rows]
    if aggregate["fail_count"] > 0:
        status = "FAIL_HARD_DATA_CONTRACT"
    elif not scope_complete:
        status = "SMOKE_ONLY__NOT_FREEZEABLE"
    else:
        status = "PASS_MEASUREMENT_COMPLETE__POLICY_FREEZE_NEXT"

    result = {
        "schema": "RealSaS.GeppettoArachne.SkeletonProjectionCorpusAudit.v1",
        "status": status,
        "root": str(root),
        "split": args.split,
        "scope_complete": scope_complete,
        "full_target_asset_count": full_count,
        "audited_asset_count": len(rows),
        "selection_file": str(selection_path),
        "selection_file_sha256": sha256_file(selection_path),
        "audited_asset_set_sha256": canonical_json_sha256(audited_ids),
        "aggregate": aggregate,
        "rows": rows,
        "authority": {
            "teacher_projection": "SkeletonTeacherProjectionV1",
            "control_location": "SOURCE_BONE_HEAD",
            "helper_parent_policy_under_audit": "NEAREST_DEFORM_ANCESTOR",
            "skin_transport_applied": False,
            "silent_skin_transpose_or_reindex": False,
            "consumer_model_outputs_read": False,
            "closed_sets_opened": False,
            "corpus_mutation": False,
        },
        "next_gate": "ARCHITECTURE_POLICY_FREEZE_FROM_FIT_AUDIT",
        "training_authorized_by_this_audit": False,
    }

    outdir = Path(args.out_dir) if args.out_dir else root / "reports" / "geppetto_arachne_v0_1"
    outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / "SKELETON_PROJECTION_CORPUS_AUDIT_RESULT_V1.json"
    mp = outdir / "SKELETON_PROJECTION_CORPUS_AUDIT_REPORT_V1.md"
    jp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mp.write_text(render_markdown_v1(result), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "out_dir": str(outdir),
        "audited": len(rows),
        "hard_failures": aggregate["fail_count"],
        "scope_complete": scope_complete,
        "deform_control_max": aggregate["deform_control_count"]["max"],
        "multi_root_assets": aggregate["multi_root_asset_count"],
        "zero_length_assets": aggregate["asset_with_exact_zero_length_deform_bone_count"],
        "corpus_nondeform_skin_mass_fraction": (aggregate.get("skin") or {}).get("corpus_nondeform_mass_fraction"),
    }, indent=2))
    if aggregate["fail_count"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
