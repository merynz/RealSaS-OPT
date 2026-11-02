#!/usr/bin/env python3
from __future__ import annotations

"""Render a human-review Markdown report from a sealed/open Corpus370 matrix.

Read-only formatter. It does not resolve authority, change eligibility, derive a
split, or authorize training. The JSON matrix remains machine authority.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

MATRIX_SCHEMA = "RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1"
MODELS = ("IRIS", "TESSA", "MIRA", "AXIS")
IMPORTANT_FAMILIES = (
    "SOURCE_IMPORT",
    "CAMERA_OBSERVATION",
    "SURFACE_GEOMETRY_UDF",
    "GLOBAL_SOLID_SDF",
    "MECHANICAL_SURFACE_TOPOLOGY",
    "RIG_HIERARCHY",
    "SKIN_WEIGHTS",
    "ANIMATION_KINEMATICS",
    "TESSA_TEACHER",
)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _escape(value) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _coverage(cell: dict) -> str:
    values = []
    for metadata in cell.get("claim_metadata") or []:
        if not isinstance(metadata, dict):
            continue
        if "valid_row_coverage_fraction" in metadata:
            values.append(float(metadata["valid_row_coverage_fraction"]))
    if not values:
        return ""
    if max(values) - min(values) > 1e-12:
        return ",".join(f"{x:.3f}" for x in values)
    return f"{values[0]:.3f}"


def _cell_text(cell: dict) -> str:
    status = str(cell.get("status") or "UNKNOWN")
    coverage = _coverage(cell)
    return status + (f" ({coverage})" if coverage else "")


def _blocker_rollup(rows: list[dict], model: str) -> Counter:
    counter = Counter()
    for row in rows:
        model_row = (row.get("model_eligibility") or {}).get(model) or {}
        if model_row.get("corpus_eligible"):
            continue
        blockers = list(model_row.get("blocking_families") or ())
        if not blockers:
            counter["UNSPECIFIED_BLOCKER"] += 1
        else:
            counter.update(blockers)
    return counter


def render(matrix: dict) -> str:
    require(matrix.get("schema") == MATRIX_SCHEMA, "CORPUS370_REPORT_MATRIX_SCHEMA")
    rows = list(matrix.get("subjects") or [])
    require(len(rows) == int(matrix.get("subject_count", -1)), "CORPUS370_REPORT_SUBJECT_COUNT")

    out = [
        "# Corpus370 Subject × Truth-Family Authority Matrix",
        "",
        "> Human-readable view only. The JSON matrix is the machine authority. This report does not mint family authority, split membership, or training readiness.",
        "",
        "## Closure state",
        "",
        f"- Matrix status: `{_escape(matrix.get('status'))}`",
        f"- Subjects: **{len(rows)}**",
        f"- Unknown cells: **{int((matrix.get('rollup') or {}).get('unknown_cell_count', 0))}**",
        f"- Quarantined cells: **{int((matrix.get('rollup') or {}).get('quarantined_cell_count', 0))}**",
        f"- Training authorized: **{bool(matrix.get('training_authorized'))}**",
        "",
        "## Truth-family rollup",
        "",
        "| Family | PASS | MASKED | FAIL | QUARANTINED | UNKNOWN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    family_rollup = (matrix.get("rollup") or {}).get("families") or {}
    for family in sorted(family_rollup):
        counts = family_rollup[family]
        out.append(
            f"| `{_escape(family)}` | {counts.get('PASS', 0)} | {counts.get('MASKED', 0)} | {counts.get('FAIL', 0)} | {counts.get('QUARANTINED', 0)} | {counts.get('UNKNOWN', 0)} |"
        )

    out.extend(["", "## Model corpus eligibility", "", "| Model | Eligible | Blocked | Main blocking families |", "|---|---:|---:|---|"])
    model_rollup = (matrix.get("rollup") or {}).get("models") or {}
    for model in MODELS:
        counts = model_rollup.get(model) or {}
        blockers = _blocker_rollup(rows, model)
        summary = ", ".join(f"`{name}`={count}" for name, count in blockers.most_common()) or "—"
        out.append(f"| **{model}** | {counts.get('eligible', 0)} | {counts.get('blocked', 0)} | {summary} |")

    sdf = Counter((row.get("families") or {}).get("GLOBAL_SOLID_SDF", {}).get("status", "UNKNOWN") for row in rows)
    out.extend(
        [
            "",
            "## Global solid-SDF scope",
            "",
            f"- PASS explicit-solid/global-SDF subjects: **{sdf.get('PASS', 0)}**",
            f"- Whole-family MASKED subjects: **{sdf.get('MASKED', 0)}**",
            f"- FAIL: **{sdf.get('FAIL', 0)}**; QUARANTINED: **{sdf.get('QUARANTINED', 0)}**; UNKNOWN: **{sdf.get('UNKNOWN', 0)}**",
            "- `MASKED` means no global-SDF loss for that subject. It is not numeric zero.",
            "",
        ]
    )

    for model in MODELS:
        eligible = [row for row in rows if ((row.get("model_eligibility") or {}).get(model) or {}).get("corpus_eligible")]
        out.extend([f"## {model} eligible subjects ({len(eligible)})", ""])
        if not eligible:
            out.extend(["_No currently eligible subjects._", ""])
            continue
        out.extend(
            [
                "| Subject | Format | Source | Required-family state | Row-mask notes |",
                "|---|---|---|---|---|",
            ]
        )
        for row in eligible:
            model_row = (row.get("model_eligibility") or {}).get(model) or {}
            required = list(model_row.get("required_pass") or ())
            families = row.get("families") or {}
            # Keep FAMILY=STATE literal in the report so Sir can scan/filter the
            # table directly and downstream text checks do not depend on Markdown
            # code-span punctuation.
            states = "; ".join(f"{family}={_cell_text(families.get(family) or {})}" for family in required)
            mask_notes = []
            for family in required:
                cell = families.get(family) or {}
                for metadata in cell.get("claim_metadata") or []:
                    if not isinstance(metadata, dict):
                        continue
                    if metadata.get("row_validity_mask_required"):
                        coverage = metadata.get("valid_row_coverage_fraction")
                        mask_notes.append(f"{family}: validity-mask" + (f" {float(coverage):.3f}" if coverage is not None else ""))
                    scope = metadata.get("authority_scope")
                    if scope and family in {"RIG_HIERARCHY", "SKIN_WEIGHTS"}:
                        mask_notes.append(f"{family}: {scope}")
            out.append(
                f"| `{_escape(row.get('subject_id'))}` | `{_escape(row.get('source_format'))}` | `{_escape(row.get('source_id'))}` | {_escape(states)} | {_escape('; '.join(dict.fromkeys(mask_notes)) or '—')} |"
            )
        out.append("")

    out.extend(
        [
            "## Per-subject review table",
            "",
            "Coverage in parentheses is a courted row-validity fraction for a PASS family, not subject-level admission probability.",
            "",
            "| Subject | Format | IRIS | TESSA | MIRA | AXIS | " + " | ".join(IMPORTANT_FAMILIES) + " |",
            "|---|---|---:|---:|---:|---:|" + "---|" * len(IMPORTANT_FAMILIES),
        ]
    )
    for row in rows:
        models = row.get("model_eligibility") or {}
        flags = ["YES" if (models.get(model) or {}).get("corpus_eligible") else "NO" for model in MODELS]
        families = row.get("families") or {}
        cells = [_cell_text(families.get(family) or {}) for family in IMPORTANT_FAMILIES]
        out.append(
            "| `" + _escape(row.get("subject_id")) + "` | `" + _escape(row.get("source_format")) + "` | " + " | ".join(flags + cells) + " |"
        )

    out.extend(["", "## Remaining closure", ""])
    for item in matrix.get("next_required_authority") or []:
        out.append(f"- `{_escape(item)}`")
    out.extend(["", "Training remains closed until the separate readiness receipt passes.", ""])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    text = render(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "REPORT_WRITTEN__NO_AUTHORITY_CHANGE", "subject_count": matrix.get("subject_count"), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
