from __future__ import annotations

"""Seal dense BODY topology/W derivation and bind it to the final motion authority.

V1 proved the exact dense-zero-surface -> current GSA -> fresh FIT2 W derivation.
V2 additionally proves that every persisted dynamic face witness was qualified against
the same historical phase-quality motion_state_hash that the final product consumes.
No training, model query, topology generation, weight repair or threshold relaxation
occurs in this stage. This stage does not claim PRODUCT_PASS.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import experiments.mage_demo_fit1_v5_p1.seal_fit2_dense_zero_surface_body_binding_v1 as v1

SCHEMA = "RealSaS.MageFIT2.DenseZeroSurfaceBodyBindingDerivation.v2.motion_bound"
DERIVATION_FILE = "FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION_V2_MANIFEST.json"
DERIVATION_STATUS = "PASS__DENSE_TO_GSA_EXACT_COMPACTION_ROWS_BOUND_TO_FRESH_FIT2_W_AND_FINAL_MOTION"
EXPECTED_MOTION_AUTHORITY = "HISTORICAL_PHASE_QUALITY_MOTION_STATE"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _artifact_sha_map(manifest: dict) -> dict[str, str]:
    rows: dict[str, str] = {}
    for row in manifest.get("artifacts") or ():
        name = str(row.get("path") or "")
        digest = str(row.get("sha256") or "")
        if not name or not digest or name in rows:
            raise RuntimeError("DENSE_DERIVATION_V2_ARTIFACT_TABLE_INVALID")
        rows[name] = digest
    return rows


def run(args) -> dict:
    # Re-run the evidence-only V1 seal so the base derivation is freshly validated
    # against the exact files that V2 is about to bind to the motion witnesses.
    base = v1.run(args)["derivation"]

    body_dir = Path(args.dense_body_dir)
    body_path = body_dir / "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"
    body_sha = _sha(body_path)
    body = json.loads(body_path.read_text(encoding="utf-8"))

    motion_hash = str(body.get("motion_state_hash") or "")
    if not motion_hash:
        raise RuntimeError("DENSE_DERIVATION_V2_MOTION_STATE_HASH_MISSING")
    if body.get("dynamic_motion_authority") != EXPECTED_MOTION_AUTHORITY:
        raise RuntimeError("DENSE_DERIVATION_V2_MOTION_AUTHORITY_DRIFT")
    if body.get("historical_motion_engine_recovered") is not True:
        raise RuntimeError("DENSE_DERIVATION_V2_HISTORICAL_MOTION_FLAG_MISSING")

    policy = dict(body.get("policy") or {})
    edge_limit = float(policy.get("edge_stretch_limit", -1.0))
    area_limit = float(policy.get("area_change_limit", -1.0))
    flip_limit = int(policy.get("triangle_flip_limit", -1))
    if edge_limit <= 0.0 or area_limit <= 0.0 or flip_limit != 0:
        raise RuntimeError("DENSE_DERIVATION_V2_DYNAMIC_POLICY_INVALID")

    artifacts = _artifact_sha_map(body)
    views = {int(row["view"]): row for row in body.get("views") or ()}
    if set(views) != set(range(8)):
        raise RuntimeError("DENSE_DERIVATION_V2_REQUIRES_8_VIEWS")

    witness_rows = []
    for view in range(8):
        row = views[view]
        files = dict(row.get("files") or {})
        dynamic_name = str(files.get("dynamic") or "")
        if not dynamic_name:
            raise RuntimeError(f"DENSE_DERIVATION_V2_DYNAMIC_FILE_MISSING_V{view}")
        dynamic_path = body_dir / dynamic_name
        dynamic_sha = _sha(dynamic_path)
        if artifacts.get(dynamic_name) != dynamic_sha:
            raise RuntimeError(f"DENSE_DERIVATION_V2_DYNAMIC_FILE_SHA_DRIFT_V{view}")
        witness = json.loads(dynamic_path.read_text(encoding="utf-8"))
        if int(witness.get("view", -1)) != view:
            raise RuntimeError(f"DENSE_DERIVATION_V2_DYNAMIC_VIEW_DRIFT_V{view}")
        if str(witness.get("motion_state_hash") or "") != motion_hash:
            raise RuntimeError(f"DENSE_DERIVATION_V2_DYNAMIC_MOTION_HASH_DRIFT_V{view}")
        if witness.get("motion_authority") != EXPECTED_MOTION_AUTHORITY:
            raise RuntimeError(f"DENSE_DERIVATION_V2_DYNAMIC_MOTION_AUTHORITY_DRIFT_V{view}")
        if witness.get("historical_motion_engine_recovered") is not True:
            raise RuntimeError(f"DENSE_DERIVATION_V2_DYNAMIC_HISTORICAL_FLAG_MISSING_V{view}")
        if float(witness.get("selected_max_edge_stretch_ratio", float("inf"))) > edge_limit + 1e-12:
            raise RuntimeError(f"DENSE_DERIVATION_V2_EDGE_GATE_DRIFT_V{view}")
        if float(witness.get("selected_max_area_change_ratio", float("inf"))) > area_limit + 1e-12:
            raise RuntimeError(f"DENSE_DERIVATION_V2_AREA_GATE_DRIFT_V{view}")
        if int(witness.get("selected_triangle_flip_count", -1)) != 0:
            raise RuntimeError(f"DENSE_DERIVATION_V2_FLIP_GATE_DRIFT_V{view}")
        if str(witness.get("selected_face_set_sha256") or "") != str(row.get("selected_face_set_sha256") or ""):
            raise RuntimeError(f"DENSE_DERIVATION_V2_SELECTED_FACE_SET_DRIFT_V{view}")
        witness_rows.append({
            "view": view,
            "dynamic_witness_file": dynamic_name,
            "dynamic_witness_file_sha256": dynamic_sha,
            "motion_state_hash": motion_hash,
            "selected_face_set_sha256": str(row["selected_face_set_sha256"]),
            "selected_max_edge_stretch_ratio": float(witness["selected_max_edge_stretch_ratio"]),
            "selected_max_area_change_ratio": float(witness["selected_max_area_change_ratio"]),
            "selected_triangle_flip_count": int(witness["selected_triangle_flip_count"]),
        })

    derivation = {
        **base,
        "schema": SCHEMA,
        "status": DERIVATION_STATUS,
        "source_dense_body_manifest_sha256": body_sha,
        "motion_state_hash": motion_hash,
        "dynamic_motion_authority": EXPECTED_MOTION_AUTHORITY,
        "historical_motion_engine_recovered": True,
        "dynamic_witnesses": witness_rows,
        "topology_W_and_motion_share_one_persisted_authority_chain": True,
        "product_pass_claimed": False,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / DERIVATION_FILE
    _write_json(path, derivation)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION_FINAL_MOTION",
        "derivation_manifest": path.name,
        "derivation_manifest_sha256": _sha(path),
        "source_dense_body_manifest_sha256": body_sha,
        "motion_state_hash": motion_hash,
        "dynamic_motion_authority": EXPECTED_MOTION_AUTHORITY,
        "surface_lineage_hash": derivation["surface_lineage_hash"],
        "skeleton_lineage_hash": derivation["skeleton_lineage_hash"],
        "skin_lineage_hash": derivation["skin_lineage_hash"],
        "product_pass_claimed": False,
    }
    seal_path = output / "FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION_V2_SEAL.json"
    _write_json(seal_path, seal)
    print("FIT2_DENSE_BODY_BINDING_DERIVATION_V2=" + json.dumps(seal, sort_keys=True), flush=True)
    return {"derivation": derivation, "seal": seal}


def parse_args():
    return v1.parse_args()


if __name__ == "__main__":
    run(parse_args())
