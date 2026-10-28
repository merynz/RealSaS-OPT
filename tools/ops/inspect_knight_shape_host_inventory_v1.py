from __future__ import annotations

"""Read-only authority-host inventory for genuine learned ShapeProposal evidence.

This tool does not import, reinterpret, replay, infer, qualify, promote or register
anything. It searches historical authority runs for exact ShapeProposal v1/v2 NPZ
payloads whose own metadata declares LEARNED_SHAPE_PROPOSAL_ONLY, validates them
with the current decoders, and reports subject/lineage witnesses separately.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


SCHEMA = "RealSaS.KnightShapeHostInventory.v1"
AUTHORITY = "LEARNED_SHAPE_PROPOSAL_ONLY"
V1_SCHEMA = "RealSaS.ShapeProposal.v1"
V2_SCHEMA = "RealSaS.ShapeProposal.v2"
MAX_CANDIDATE_BYTES = 2 * 1024 * 1024 * 1024
MAX_JSON_WITNESS_BYTES = 32 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_from_npz(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            if "metadata_json.npy" not in names or "token_ids.npy" not in names:
                return None
            raw = archive.read("metadata_json.npy")
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        metadata = json.loads(str(np.asarray(value).reshape(-1)[0]))
        return metadata if isinstance(metadata, dict) else None
    except Exception:
        return None


def _run_root(runs_root: Path, path: Path) -> Path:
    rel = path.relative_to(runs_root)
    if not rel.parts:
        return runs_root
    return runs_root / rel.parts[0]


def _json_witnesses(run_root: Path, content_sha256: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in run_root.rglob("*.json"):
        try:
            if not path.is_file() or path.stat().st_size > MAX_JSON_WITNESS_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lower = text.lower()
        knight_mentioned = "knight" in lower
        hash_referenced = content_sha256 in lower
        if not knight_mentioned and not hash_referenced:
            continue
        subject_fields: list[dict[str, str]] = []
        try:
            payload = json.loads(text)
        except Exception:
            payload = None

        def walk(value: Any, prefix: str = "$") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    child_prefix = f"{prefix}.{key}"
                    key_lower = str(key).lower()
                    if key_lower in {"subject", "subject_id", "subject_name", "subject_key"}:
                        subject_fields.append({"path": child_prefix, "value": str(child)})
                    walk(child, child_prefix)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{prefix}[{index}]")

        if payload is not None:
            walk(payload)
        out.append(
            {
                "path": str(path),
                "knight_mentioned": knight_mentioned,
                "content_sha256_referenced": hash_referenced,
                "subject_fields": subject_fields[:32],
            }
        )
    return out[:128]


def _knight_binding(metadata: dict[str, Any], witnesses: list[dict[str, Any]]) -> str:
    metadata_text = json.dumps(metadata, sort_keys=True).lower()
    if "knight" in metadata_text:
        return "PAYLOAD_METADATA_MENTIONS_KNIGHT"
    for witness in witnesses:
        if witness["content_sha256_referenced"] and witness["knight_mentioned"]:
            return "EXACT_HASH_AND_KNIGHT_JSON_WITNESS"
    for witness in witnesses:
        fields = witness.get("subject_fields") or []
        if any(str(item.get("value", "")).strip().lower() == "knight" for item in fields):
            return "RUN_JSON_SUBJECT_FIELD_KNIGHT__NO_EXACT_HASH_BINDING"
    if any(witness["knight_mentioned"] for witness in witnesses):
        return "RUN_JSON_MENTIONS_KNIGHT__NO_EXACT_HASH_BINDING"
    return "NO_KNIGHT_BINDING_WITNESS"


def inspect(*, authority_root: Path, expected_main_sha: str) -> dict[str, Any]:
    runs_root = authority_root / "runs"
    if not runs_root.is_dir():
        raise RuntimeError("KNIGHT_SHAPE_HOST_RUNS_ROOT_MISSING")

    from compiler.realsas_compiler_services.platform_worker.shape_capability_v1 import (
        decode_shape_proposal_metadata_v1,
    )
    from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import (
        decode_shape_proposal_v2,
    )

    scanned_files = 0
    zip_like_files = 0
    metadata_npz_files = 0
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for root, _dirs, files in os.walk(runs_root):
        for name in files:
            path = Path(root) / name
            scanned_files += 1
            try:
                size = path.stat().st_size
                if size <= 0 or size > MAX_CANDIDATE_BYTES:
                    continue
                with path.open("rb") as handle:
                    if handle.read(4) != b"PK\x03\x04":
                        continue
            except Exception:
                continue
            zip_like_files += 1
            metadata = _metadata_from_npz(path)
            if metadata is None:
                continue
            metadata_npz_files += 1
            schema = str(metadata.get("schema") or "")
            authority = str(metadata.get("authority") or "")
            if schema not in {V1_SCHEMA, V2_SCHEMA}:
                continue
            if authority != AUTHORITY:
                rejected.append(
                    {
                        "path": str(path),
                        "schema": schema,
                        "authority": authority,
                        "reason": "SHAPE_AUTHORITY_NOT_LEARNED_SHAPE_PROPOSAL_ONLY",
                    }
                )
                continue

            raw = path.read_bytes()
            try:
                if schema == V1_SCHEMA:
                    validated_metadata = decode_shape_proposal_metadata_v1(raw)
                    decoded_summary = {
                        "token_count": int(validated_metadata.get("token_count") or 0),
                        "completed": bool(validated_metadata.get("completed")),
                        "stop_reason": str(validated_metadata.get("stop_reason") or ""),
                    }
                else:
                    decoded = decode_shape_proposal_v2(raw)
                    validated_metadata = dict(decoded.metadata)
                    decoded_summary = {
                        "token_count": int(decoded.token_ids.size),
                        "decoded_vertex_count": int(decoded.decoded_vertices_tessa.shape[0]),
                        "decoded_face_count": int(decoded.decoded_faces.shape[0]),
                        "decoded_topology_sha256": str(decoded.metadata.get("decoded_topology_sha256") or ""),
                    }
            except Exception as exc:
                rejected.append(
                    {
                        "path": str(path),
                        "schema": schema,
                        "authority": authority,
                        "reason": "CURRENT_DECODER_REJECTED",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                continue

            content_sha = hashlib.sha256(raw).hexdigest()
            run_root = _run_root(runs_root, path)
            witnesses = _json_witnesses(run_root, content_sha)
            candidates.append(
                {
                    "path": str(path),
                    "run_name": run_root.name,
                    "size_bytes": size,
                    "content_sha256": content_sha,
                    "schema": schema,
                    "authority": authority,
                    "models": validated_metadata.get("models") or {},
                    "decoded_summary": decoded_summary,
                    "knight_binding": _knight_binding(validated_metadata, witnesses),
                    "lineage_witnesses": witnesses,
                }
            )

    candidates.sort(key=lambda item: (item["schema"], item["run_name"], item["path"]))
    exact_knight = [
        item
        for item in candidates
        if item["knight_binding"] in {
            "PAYLOAD_METADATA_MENTIONS_KNIGHT",
            "EXACT_HASH_AND_KNIGHT_JSON_WITNESS",
        }
    ]
    if exact_knight:
        status = "EXACT_LEARNED_SHAPE_WITH_KNIGHT_BINDING_FOUND"
        next_action = "REVIEW_EXACT_LINEAGE_AND_ADMIT_OR_REPLAY_WITHOUT_MODEL_INFERENCE"
    elif candidates:
        status = "LEARNED_SHAPE_CANDIDATES_FOUND__KNIGHT_BINDING_NOT_PROVEN"
        next_action = "RESOLVE_CANDIDATE_SUBJECT_AND_SOURCE_LINEAGE_BEFORE_ANY_REUSE"
    else:
        status = "NO_EXACT_LEARNED_SHAPE_FOUND_ON_AUTHORITY_HOST"
        next_action = "CURRENT_SHAPE_PRODUCER_REQUIRES_REAL_MODEL_RELEASE_OR_SEPARATE_NONPRODUCT_CONSUMER_FIXTURE"

    return {
        "schema": SCHEMA,
        "source_code_sha": expected_main_sha,
        "authority_root": str(authority_root),
        "status": status,
        "decision": {"next_action": next_action},
        "scan_counts": {
            "files": scanned_files,
            "zip_like_files": zip_like_files,
            "npz_with_metadata_and_tokens": metadata_npz_files,
            "valid_learned_shape_candidates": len(candidates),
            "exact_knight_bound_candidates": len(exact_knight),
            "rejected_shape_like_candidates": len(rejected),
        },
        "candidates": candidates,
        "rejected": rejected[:128],
        "host_scan_is_execution_authority": False,
        "legacy_payload_reinterpretation_performed": False,
        "artifact_import_performed": False,
        "deterministic_shape_replay_performed": False,
        "model_inference_executed": False,
        "compiler_capability_executed": False,
        "qualification_minted": False,
        "model_promotion_performed": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    expected = str(args.expected_main_sha).strip().lower()
    if len(expected) != 40:
        raise RuntimeError("KNIGHT_SHAPE_HOST_EXPECTED_MAIN_SHA_INVALID")
    receipt = inspect(
        authority_root=Path(args.authority_root).expanduser().resolve(),
        expected_main_sha=expected,
    )
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
