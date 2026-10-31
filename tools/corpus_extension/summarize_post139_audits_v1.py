import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "realsas.post139_full_scan_summary.v1"
EXT_ORDER = [".glb", ".gltf", ".fbx", ".blend"]


def load_audits(root):
    rows = []
    for p in sorted(Path(root).glob("*.json")):
        if p.name == "MANIFEST.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "capabilities" not in d or "source" not in d:
            continue
        rel = d["source"].get("relative_path", d["source"].get("name", ""))
        parts = Path(rel).parts
        provider = parts[0] if len(parts) > 0 else "unknown"
        pack = parts[1] if len(parts) > 1 else "unknown"
        d["_audit_file"] = p.name
        d["_relative_path"] = rel
        d["_provider"] = provider
        d["_pack"] = pack
        rows.append(d)
    return rows


def chosen_ext(rows):
    exts = {Path(r["_relative_path"]).suffix.lower() for r in rows}
    for ext in EXT_ORDER:
        if ext in exts:
            return ext
    return None


def mechanical_key(r):
    c = r["capabilities"]
    g, rig, skin = c["geometry"], c["rig"], c["skin"]
    if not (g.get("pass") and rig.get("pass") and skin.get("pass")):
        return None
    fields = [
        g.get("geometry_signature"),
        g.get("topology_signature"),
        rig.get("skeleton_hierarchy_signature"),
        rig.get("rest_bind_signature"),
        skin.get("skin_signature"),
    ]
    if any(x is None for x in fields):
        return None
    return hashlib.sha256("|".join(fields).encode()).hexdigest()


def capability_bits(r):
    c = r["capabilities"]
    return {k: bool(c[k].get("pass")) for k in ["geometry", "appearance", "rig", "skin", "animation"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_audits(args.audits)
    by_pack = defaultdict(list)
    for r in rows:
        by_pack[(r["_provider"], r["_pack"])].append(r)

    selected = []
    pack_summaries = []
    for (provider, pack), items in sorted(by_pack.items()):
        ext = chosen_ext(items)
        chosen = [r for r in items if Path(r["_relative_path"]).suffix.lower() == ext]
        full = [r for r in chosen if all(capability_bits(r).values())]
        partial = [r for r in chosen if r["capabilities"]["geometry"].get("pass") and r["capabilities"]["rig"].get("pass") and r["capabilities"]["skin"].get("pass") and not r["capabilities"]["animation"].get("pass")]
        static = [r for r in chosen if r["capabilities"]["geometry"].get("pass") and not r["capabilities"]["rig"].get("pass")]
        errors = [r for r in chosen if not r["capabilities"]["geometry"].get("pass")]

        mech_groups = defaultdict(list)
        no_key = []
        for r in full + partial:
            key = mechanical_key(r)
            if key:
                mech_groups[key].append(r)
            else:
                no_key.append(r)
        representatives = [v[0] for _, v in sorted(mech_groups.items())] + no_key
        selected.extend(representatives)

        pack_summaries.append({
            "provider": provider,
            "pack": pack,
            "preferred_extension": ext,
            "source_files_preferred_format": len(chosen),
            "full_actor_files": len(full),
            "partial_actor_files": len(partial),
            "static_or_prop_files": len(static),
            "invalid_geometry_files": len(errors),
            "distinct_mechanical_subject_candidates": len(representatives),
            "within_pack_mechanical_duplicate_groups": sum(1 for v in mech_groups.values() if len(v) > 1),
        })

    cross = defaultdict(list)
    for r in selected:
        key = mechanical_key(r)
        if key:
            cross[key].append(r)
    cross_dups = {k: v for k, v in cross.items() if len(v) > 1}

    roster = []
    for r in selected:
        bits = capability_bits(r)
        cls = "FULL_TRUTH_ACTOR" if all(bits.values()) else "PARTIAL_TRUTH_ACTOR"
        roster.append({
            "candidate_id": f"{r['_provider']}:{r['_pack']}:{Path(r['_relative_path']).stem}",
            "provider": r["_provider"],
            "pack": r["_pack"],
            "source_path": r["_relative_path"],
            "source_sha256": r["source"].get("sha256"),
            "source_bytes": r["source"].get("bytes"),
            "subject_class_candidate": cls,
            "capabilities": bits,
            "mechanical_signature": mechanical_key(r),
            "geometry_signature": r["capabilities"]["geometry"].get("geometry_signature"),
            "topology_signature": r["capabilities"]["geometry"].get("topology_signature"),
            "skeleton_hierarchy_signature": r["capabilities"]["rig"].get("skeleton_hierarchy_signature"),
            "rest_bind_signature": r["capabilities"]["rig"].get("rest_bind_signature"),
            "skin_signature": r["capabilities"]["skin"].get("skin_signature"),
            "appearance_signature": r["capabilities"]["appearance"].get("appearance_signature"),
            "animation_signature": r["capabilities"]["animation"].get("animation_signature"),
            "audit_file": r["_audit_file"],
            "final_admit": False,
            "remaining_gate": "Base139 overlap + existing Truth139 canonical courts",
        })

    with (out / "ROSTER_CANDIDATES.jsonl").open("w", encoding="utf-8") as f:
        for r in roster:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dup_json = []
    for key, vals in cross_dups.items():
        dup_json.append({
            "mechanical_signature": key,
            "members": [f"{r['_provider']}:{r['_pack']}:{r['_relative_path']}" for r in vals],
        })
    (out / "CROSS_PACK_DUPLICATE_GROUPS.json").write_text(json.dumps(dup_json, indent=2), encoding="utf-8")

    summary = {
        "schema_version": SCHEMA,
        "raw_audit_file_count": len(rows),
        "pack_count": len(pack_summaries),
        "pack_summaries": pack_summaries,
        "selected_subject_candidate_count_before_base139_dedup": len(roster),
        "selected_full_truth_actor_candidate_count": sum(1 for r in roster if r["subject_class_candidate"] == "FULL_TRUTH_ACTOR"),
        "selected_partial_truth_actor_candidate_count": sum(1 for r in roster if r["subject_class_candidate"] == "PARTIAL_TRUTH_ACTOR"),
        "cross_pack_exact_mechanical_duplicate_group_count": len(cross_dups),
        "authority": {
            "base139": "IMMUTABLE",
            "final_admission": "existing Truth139 courts -> realsas.source_admission.v1",
            "same_rig_alone_is_not_duplicate": True,
            "appearance_only_variants_with_identical_mechanics_collapse": True,
            "base139_overlap_still_required": True,
        },
    }
    (out / "FULL_SCAN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["raw_audit_file_count", "pack_count", "selected_subject_candidate_count_before_base139_dedup", "selected_full_truth_actor_candidate_count", "selected_partial_truth_actor_candidate_count", "cross_pack_exact_mechanical_duplicate_group_count"]}, indent=2))


if __name__ == "__main__":
    main()
