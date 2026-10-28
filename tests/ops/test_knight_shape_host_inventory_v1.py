from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.ops.inspect_knight_shape_host_inventory_v1 import inspect


def _shape_v1(path: Path, *, authority: str) -> str:
    metadata = {
        "schema": "RealSaS.ShapeProposal.v1",
        "authority": authority,
        "canonical_mesh_authority": False,
        "mesh_authority": False,
        "completed": True,
        "stop_reason": "EOS",
        "models": {
            "IRIS": {"model_release_id": "iris-historical", "checkpoint_content_sha256": "1" * 64},
            "TESSA": {"model_release_id": "tessa-historical", "checkpoint_content_sha256": "2" * 64},
        },
    }
    np.savez_compressed(
        path,
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True, separators=(",", ":"))]),
        token_ids=np.asarray([1, 2, 3], dtype=np.int64),
    )
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def test_inventory_accepts_only_exact_learned_shape_and_keeps_authority_negative(tmp_path: Path) -> None:
    authority_root = tmp_path / "authority"
    run_root = authority_root / "runs" / "KNIGHT_HISTORICAL_SHAPE"
    run_root.mkdir(parents=True)

    accepted_path = run_root / "shape_v1.npz"
    accepted_sha = _shape_v1(accepted_path, authority="LEARNED_SHAPE_PROPOSAL_ONLY")
    _shape_v1(run_root / "oracle_disguised_as_shape.npz", authority="ORACLE_TRUTH_ONLY")
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "Test.Manifest.v1",
                "subject_id": "knight",
                "shape_content_sha256": accepted_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    receipt = inspect(authority_root=authority_root, expected_main_sha="a" * 40)

    assert receipt["status"] == "EXACT_LEARNED_SHAPE_WITH_KNIGHT_BINDING_FOUND"
    assert receipt["scan_counts"]["valid_learned_shape_candidates"] == 1
    assert receipt["scan_counts"]["exact_knight_bound_candidates"] == 1
    assert receipt["scan_counts"]["rejected_shape_like_candidates"] == 1
    candidate = receipt["candidates"][0]
    assert candidate["content_sha256"] == accepted_sha
    assert candidate["knight_binding"] == "EXACT_HASH_AND_KNIGHT_JSON_WITNESS"
    assert candidate["models"]["IRIS"]["model_release_id"] == "iris-historical"
    assert candidate["models"]["TESSA"]["model_release_id"] == "tessa-historical"

    for key in (
        "host_scan_is_execution_authority",
        "legacy_payload_reinterpretation_performed",
        "artifact_import_performed",
        "deterministic_shape_replay_performed",
        "model_inference_executed",
        "compiler_capability_executed",
        "qualification_minted",
        "model_promotion_performed",
        "product_revision_minted",
        "genericity_claimed",
    ):
        assert receipt[key] is False
