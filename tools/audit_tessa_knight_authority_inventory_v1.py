from __future__ import annotations

"""Read-only inventory of sealed Knight authority artifacts relevant to TESSA Stage35.

This tool never writes below --authority-root. It exists to distinguish reusable typed
artifacts from historical/research reports before attempting any current Stage35 replay.
An optional --out writes the inventory to a caller-owned path outside the authority root
so CI can upload it as diagnostic evidence without mutating sealed authority.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TARGET_SCHEMA_TOKENS = (
    "TESSA",
    "CanonicalMeshCandidate",
    "StaticCanonicalMeshQualification",
    "QualifiedSkeleton",
    "QualifiedSkin",
    "DeformationCapabilityEnvelope",
    "Axis",
    "QualifiedMesh",
)
PATH_TOKENS = (
    "tessa",
    "canonical_mesh_candidate",
    "static_canonical",
    "static_mesh",
    "qualified_skeleton",
    "qualified_skin",
    "deformation_envelope",
    "axis",
    "qualified_mesh",
    "stage18",
    "stage19",
    "stage28",
    "stage32",
    "stage34",
    "stage35",
)
LINEAGE_KEYS = (
    "candidate_lineage_hash",
    "candidate_mesh_binding_hash",
    "bridge_evidence_hash",
    "static_mesh_qualification_hash",
    "proposal_geometry_hash",
    "material_support_field_hash",
    "partition_binding_hash",
    "binding_hash",
    "geometry_lineage_hash",
    "surface_binding_hash",
    "skeleton_lineage_hash",
    "skeleton_binding_hash",
    "skin_lineage_hash",
    "mesh_lineage_hash",
    "envelope_lineage_hash",
    "axis_contract_hash",
    "topology_sequence_hash",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            return None, "JSON_TOO_LARGE_FOR_INVENTORY_PARSE"
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # diagnostic inventory must not hide malformed evidence
        return None, f"{type(exc).__name__}:{exc}"


def _schema(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("schema_version") or payload.get("schema") or "")


def _lineage(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    for key in LINEAGE_KEYS:
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)) and value not in ("", None):
            result[key] = value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in LINEAGE_KEYS:
            value = metadata.get(key)
            if isinstance(value, (str, int, float, bool)) and value not in ("", None):
                result[f"metadata.{key}"] = value
    return result


def _status(payload: Any):
    if not isinstance(payload, dict):
        return None
    value = payload.get("status")
    return value if isinstance(value, (str, bool, int, float)) else None


def _row(path: Path, *, root: Path, payload: Any, parse_note: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema": _schema(payload),
        "status": _status(payload),
        "lineage": _lineage(payload),
    }
    if parse_note:
        row["parse_note"] = parse_note
    return row


def build_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    runs_root = root / "runs"
    if not runs_root.is_dir():
        raise RuntimeError(f"AUTHORITY_RUNS_ROOT_MISSING::{runs_root}")

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    knight_runs = [path for path in run_dirs if "KNIGHT" in path.name.upper()]
    result: dict[str, Any] = {
        "schema": "RealSaS.TESSAKnightAuthorityInventory.v1",
        "status": "PASS_READ_ONLY_INVENTORY",
        "authority_root": str(root),
        "run_count": len(run_dirs),
        "knight_run_count": len(knight_runs),
        "read_only": True,
        "knight_runs": [],
        "external_knight_research_artifacts": [],
    }

    for run in knight_runs:
        artifacts = []
        for path in sorted(run.rglob("*.json")):
            payload, parse_note = _load_json(path)
            schema = _schema(payload)
            relative = path.relative_to(root).as_posix().lower()
            if not (
                any(token in relative for token in PATH_TOKENS)
                or any(token.lower() in schema.lower() for token in TARGET_SCHEMA_TOKENS)
            ):
                continue
            artifacts.append(_row(path, root=root, payload=payload, parse_note=parse_note))
        result["knight_runs"].append(
            {"run_id": run.name, "artifact_count": len(artifacts), "artifacts": artifacts}
        )

    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        low = relative.lower()
        if low.startswith("runs/"):
            continue
        if "knight" not in low or not any(
            token in low for token in ("tessa", "mira", "axis", "stage35", "g3")
        ):
            continue
        payload, parse_note = _load_json(path)
        result["external_knight_research_artifacts"].append(
            _row(path, root=root, payload=payload, parse_note=parse_note)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_inventory(args.authority_root)

    if args.out is not None:
        out = args.out.resolve()
        authority_root = args.authority_root.resolve()
        if out == authority_root or authority_root in out.parents:
            raise RuntimeError("INVENTORY_OUTPUT_MUST_NOT_MUTATE_AUTHORITY_ROOT")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "TESSA_KNIGHT_AUTHORITY_INVENTORY_FILE="
            + json.dumps({"path": str(out), "sha256": _sha256(out)}, sort_keys=True),
            flush=True,
        )

    print("TESSA_KNIGHT_AUTHORITY_INVENTORY=" + json.dumps(report, sort_keys=True), flush=True)
    print(
        "TESSA_KNIGHT_AUTHORITY_INVENTORY_SUMMARY="
        + json.dumps(
            {
                "status": report["status"],
                "run_count": report["run_count"],
                "knight_run_count": report["knight_run_count"],
                "matched_run_artifact_count": sum(
                    row["artifact_count"] for row in report["knight_runs"]
                ),
                "external_research_artifact_count": len(
                    report["external_knight_research_artifacts"]
                ),
                "read_only": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
