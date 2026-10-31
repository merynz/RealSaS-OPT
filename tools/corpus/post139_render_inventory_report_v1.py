"""Render the V2 exact corpus inventory into a durable Markdown ledger."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inventory-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.inventory_dir)
    report = read(root / "POST139_FINAL_INVENTORY_REPORT_V2.json")
    publisher = read(root / "POST139_EXACT_PUBLISHER_INVENTORY_V2.json")
    subjects = read(root / "POST139_SUBJECT_STRUCTURE_INVENTORY_V2.json")
    admission = read(root / "POST139_GLOBAL_ADMISSION_V2.json")
    lineage = read(root / "POST139_LINEAGE_COMPONENTS_V2.json")
    roster = read(root / "POST139_FROZEN_ROSTER_V2.json")
    plan = read(root / "POST139_MATERIALIZATION_PLAN_V2.json")

    subject_by_source = defaultdict(list)
    for row in subjects["subjects"]:
        subject_by_source[row["source_id"]].append(row)
    admission_by_id = {row["subject_id"]: row for row in admission["subjects"]}

    lines = [
        "# RealSaS Post139 Exact Corpus Inventory V2",
        "",
        "> This ledger separates publisher-visible files, acquired canonical byte closure, immutable Base139 training rows, mechanical structural identities, and new Post139 admissions.",
        "",
        "## Freeze status",
        "",
        f"- Status: **{report['status']}**",
        f"- Base139 immutable training rows: **{report['base139_training_rows']}**",
        f"- Base139 structural identities: **{report['base139_structural_identities']}**",
        f"- Post139 primary materialization subjects: **{report['post139_primary_materialization_subjects']}**",
        f"- Post139 auxiliary materialization subjects: **{report['post139_auxiliary_materialization_subjects']}**",
        f"- Expanded full-training rows: **{report['expanded_full_training_rows']}**",
        f"- Publisher files inventoried: **{publisher['publisher_file_count']}**",
        f"- Acquired closure files: **{publisher['acquired_file_count']}**",
        f"- Lineage components: **{report['lineage_component_count']}** ({report['cross_pack_lineage_component_count']} cross-pack)",
        f"- Blockers: **{report['blocker_count']}**; warnings: **{report['warning_count']}**",
        "",
        "## Source / pack inventory",
        "",
        "| Family | Pack | Role | Publisher files | Acquired files | Audited rows | Full static | Full animated | Structural identities | Admission summary | Closure ledger |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    pub_by_id = {row["source_id"]: row for row in publisher["sources"]}
    summary_by_id = {row["source_id"]: row for row in report["source_summaries"]}
    for source_id, pub in pub_by_id.items():
        summary = summary_by_id.get(source_id, {})
        rows = subject_by_source.get(source_id, [])
        statuses = Counter(admission_by_id.get(row["subject_id"], {}).get("status", "UNCLASSIFIED") for row in rows)
        status_text = ", ".join(f"{k}:{v}" for k, v in sorted(statuses.items())) or pub.get("status", "—")
        lines.append(
            "| " + " | ".join([
                esc(pub.get("source_family")), esc(pub.get("pack")), esc(pub.get("role")),
                esc(pub.get("publisher_file_count")), esc(pub.get("acquired_file_count")),
                esc(summary.get("audited_rows")), esc(summary.get("full_static_rows")),
                esc(summary.get("full_animated_rows")), esc(summary.get("structural_identity_count")),
                esc(status_text), esc(pub.get("source_closure_ledger_sha256", "")[:16] if pub.get("source_closure_ledger_sha256") else "—"),
            ]) + " |"
        )

    lines += ["", "## Structural duplicates and appearance variants", ""]
    variants = admission.get("appearance_variant_groups", [])
    if variants:
        lines += ["| Canonical | Members | Appearance identities | Structural hash |", "|---|---|---:|---|"]
        for group in variants:
            lines.append(
                f"| {esc(group['canonical_subject_id'])} | {esc(', '.join(group['member_subject_ids']))} | {group['appearance_identity_count']} | `{group['structural_subject'][:16]}…` |"
            )
    else:
        lines.append("No multi-appearance structural groups detected.")

    duplicates = [row for row in admission["subjects"] if row["status"] in {"DUPLICATE_OF_BASE139_STRUCTURE", "DUPLICATE_COLLAPSE", "SEMANTIC_ALIAS_COLLAPSE"}]
    lines += ["", "### Collapsed rows", ""]
    if duplicates:
        lines += ["| Subject | Source pack | Decision | Canonical target |", "|---|---|---|---|"]
        for row in duplicates:
            lines.append(f"| {esc(row['subject_id'])} | {esc(row['pack'])} | {esc(row['status'])} | {esc(row['canonical_subject_id'])} |")
    else:
        lines.append("No collapsed rows.")

    lines += ["", "## Exact subject structures", ""]
    lines += ["| Subject | Pack | Truth | V/E/P | Joints | Skinned verts | Clips | Structural | Rig hierarchy | Appearance | Admission |", "|---|---|---|---:|---:|---:|---:|---|---|---|---|"]
    for row in sorted(subjects["subjects"], key=lambda r: (r["source_family"], r["pack"], r["subject_id"])):
        m = row.get("metrics", {})
        truth = "FULL+ANIM" if row.get("full_static_truth_ready") and row.get("animation_truth") else ("FULL-STATIC" if row.get("full_static_truth_ready") else "PARTIAL")
        f = row.get("fingerprints", {})
        lines.append(
            "| " + " | ".join([
                esc(row["subject_id"]), esc(row["pack"]), truth,
                f"{m.get('vertex_count','—')}/{m.get('edge_count','—')}/{m.get('polygon_count','—')}",
                esc(m.get("joint_count")), esc(m.get("skinned_vertex_count")), esc(m.get("animation_count")),
                f"`{f.get('structural_subject','')[:12]}…`" if f.get("structural_subject") else "—",
                f"`{f.get('rig_hierarchy','')[:12]}…`" if f.get("rig_hierarchy") else "—",
                f"`{f.get('appearance','')[:12]}…`" if f.get("appearance") else "—",
                esc(admission_by_id.get(row["subject_id"], {}).get("status")),
            ]) + " |"
        )

    lines += ["", "## Leakage-safe lineage components", ""]
    cross = [row for row in lineage["components"] if row.get("cross_pack")]
    lines.append(f"Cross-pack components: **{len(cross)}**")
    lines.append("")
    if cross:
        lines += ["| Component | Members | Packs |", "|---|---:|---|"]
        for row in cross:
            lines.append(f"| {row['component_id']} | {row['member_count']} | {esc(', '.join(row['packs']))} |")

    lines += ["", "## Materialization contract", ""]
    lines += [
        f"- Plan status: **{plan['status']}**",
        f"- Primary subjects: **{plan['primary_subject_count']}**",
        f"- Auxiliary subjects: **{plan['auxiliary_subject_count']}**",
        "- Every primary row is bound to model SHA-256, publisher-tree digest, source-closure ledger SHA-256, structural identity, appearance identity, and lineage component.",
        "- The split unit is the lineage component, not the asset file or pack name.",
    ]

    if report["blockers"]:
        lines += ["", "## Blockers", "", "```json", json.dumps(report["blockers"], indent=2, sort_keys=True), "```"]
    if report["warnings"]:
        lines += ["", "## Warnings", "", "```json", json.dumps(report["warnings"], indent=2, sort_keys=True), "```"]

    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("POST139_EXACT_INVENTORY_REPORT_PASS", output, len(subjects["subjects"]))


if __name__ == "__main__":
    main()
