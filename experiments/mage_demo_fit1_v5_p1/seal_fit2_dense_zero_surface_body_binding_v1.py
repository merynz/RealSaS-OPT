from __future__ import annotations

"""Seal the exact dense-zero-surface -> current GSA -> fresh FIT2 W derivation.

This is evidence-only. It performs no training, model query, topology generation,
weight synthesis, or product proof. The seal exists so final external-render support
can bind to a persisted derivation *file SHA* instead of a generic content hash.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from compiler.realsas_compiler_core.mesh.dense_zero_surface_bridge import DENSE_ZERO_SURFACE_TOPOLOGY_METHOD
from compiler.realsas_compiler_core.mesh.mwb2_skin import SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_p1q_current_authority_v1 as legacy_p1q
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base

SCHEMA = "RealSaS.MageFIT2.DenseZeroSurfaceBodyBindingDerivation.v1"
BODY_STATUS = "PASS__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY_V0_V7"
DERIVATION_STATUS = "PASS__DENSE_TO_GSA_EXACT_COMPACTION_ROWS_BOUND_TO_FRESH_FIT2_W"
EXPECTED_ZERO_SURFACE_SHA256 = "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"


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
    rows = {}
    for row in manifest.get("artifacts") or ():
        path = str(row.get("path") or "")
        digest = str(row.get("sha256") or "")
        if not path or not digest or path in rows:
            raise RuntimeError("DENSE_DERIVATION_ARTIFACT_TABLE_INVALID")
        rows[path] = digest
    return rows


def run(args) -> dict:
    body_dir = Path(args.dense_body_dir)
    body_manifest_path = body_dir / "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"
    body_manifest_sha = _sha(body_manifest_path)
    body = json.loads(body_manifest_path.read_text(encoding="utf-8"))
    if body.get("status") != BODY_STATUS:
        raise RuntimeError("DENSE_DERIVATION_BODY_STATUS_INVALID")
    if body.get("topology_method") != DENSE_ZERO_SURFACE_TOPOLOGY_METHOD:
        raise RuntimeError("DENSE_DERIVATION_TOPOLOGY_METHOD_DRIFT")
    if body.get("zero_surface_sha256") != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("DENSE_DERIVATION_ZERO_SURFACE_SHA_DRIFT")
    if body.get("surface_lineage_hash") != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("DENSE_DERIVATION_SURFACE_LINEAGE_DRIFT")
    if body.get("skeleton_lineage_hash") != fit2io.EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("DENSE_DERIVATION_SKELETON_LINEAGE_DRIFT")
    if body.get("skin_lineage_hash") != fit2io.EXPECTED_SKIN_LINEAGE:
        raise RuntimeError("DENSE_DERIVATION_SKIN_LINEAGE_DRIFT")
    for flag in ("P1_or_P1Q_topology_used", "source_or_teacher_mesh_topology_used", "historical_weight_transfer_used", "model_query_used", "training_executed", "threshold_relaxation_used", "product_pass_claimed"):
        if body.get(flag) is not False:
            raise RuntimeError(f"DENSE_DERIVATION_FORBIDDEN_OR_MISSING_FALSE_FLAG:{flag}")

    replay = dict(body.get("compaction_replay") or {})
    if replay.get("method") != DENSE_ZERO_SURFACE_TOPOLOGY_METHOD:
        raise RuntimeError("DENSE_DERIVATION_REPLAY_METHOD_DRIFT")
    if replay.get("source_zero_surface_sha256") != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("DENSE_DERIVATION_REPLAY_ZERO_SURFACE_DRIFT")
    if replay.get("surface_lineage_hash") != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("DENSE_DERIVATION_REPLAY_SURFACE_DRIFT")
    if int(replay.get("compact_node_count", -1)) != fit2io.EXPECTED_SURFACE_NODES:
        raise RuntimeError("DENSE_DERIVATION_REPLAY_CARDINALITY_DRIFT")
    replay_hash = str(replay.get("replay_hash") or "")
    inverse_sha = str(replay.get("inverse_sha256") or "")
    if not replay_hash or not inverse_sha:
        raise RuntimeError("DENSE_DERIVATION_REPLAY_EVIDENCE_MISSING")

    surface, skeleton, skin, _mechanical = fit2io.build_exact_mechanical(
        Path(args.fit2_surface), Path(args.skeleton), Path(args.fit2_skin)
    )
    artifact_sha = _artifact_sha_map(body)
    views = {int(row["view"]): row for row in body.get("views") or ()}
    if set(views) != set(range(8)):
        raise RuntimeError("DENSE_DERIVATION_REQUIRES_8_VIEWS")

    rows = []
    for view in range(8):
        row = views[view]
        files = dict(row.get("files") or {})
        mesh_name, skin_name = str(files.get("mesh") or ""), str(files.get("skin") or "")
        if not mesh_name or not skin_name:
            raise RuntimeError(f"DENSE_DERIVATION_BODY_FILES_MISSING_V{view}")
        mesh_path, skin_path = body_dir / mesh_name, body_dir / skin_name
        if artifact_sha.get(mesh_name) != _sha(mesh_path):
            raise RuntimeError(f"DENSE_DERIVATION_MESH_FILE_SHA_DRIFT_V{view}")
        if artifact_sha.get(skin_name) != _sha(skin_path):
            raise RuntimeError(f"DENSE_DERIVATION_SKIN_FILE_SHA_DRIFT_V{view}")
        mesh = v5base._load_mesh(mesh_path, surface, str(row["mesh_lineage_hash"]))
        mesh_skin = legacy_p1q._load_mesh_skin(skin_path, mesh.mesh_lineage_hash)
        if mesh_skin.transfer_method != SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD:
            raise RuntimeError(f"DENSE_DERIVATION_TRANSFER_METHOD_DRIFT_V{view}")
        if mesh_skin.surface_binding_hash != surface.geometry_lineage_hash:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_SKIN_SURFACE_DRIFT_V{view}")
        if mesh_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_SKIN_SKELETON_DRIFT_V{view}")
        if mesh_skin.skin_binding_hash != skin.skin_lineage_hash:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_SKIN_SKIN_DRIFT_V{view}")
        md = dict(mesh.metadata or {})
        if md.get("dense_zero_surface_bridge") is not True:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_BRIDGE_FLAG_MISSING_V{view}")
        if md.get("original_dense_zero_surface_faces_only") is not True or md.get("new_adjacency_created") is not False:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_TOPOLOGY_AUTHORITY_DRIFT_V{view}")
        if str(md.get("compaction_replay_hash") or "") != replay_hash or str(md.get("compaction_inverse_sha256") or "") != inverse_sha:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_COMPACTION_WITNESS_DRIFT_V{view}")
        if mesh_skin.metadata.get("surface_support_convex_transfer") is not True or mesh_skin.metadata.get("source_skin_rows_consumed") is not True:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_SKIN_REPLAY_EVIDENCE_MISSING_V{view}")
        if mesh_skin.metadata.get("historical_weight_transfer_used") is not False or mesh_skin.metadata.get("direct_model_query") is not False:
            raise RuntimeError(f"DENSE_DERIVATION_MESH_SKIN_AUTHORITY_FIREWALL_V{view}")
        if len(mesh_skin.rows) != len(mesh.vertices):
            raise RuntimeError(f"DENSE_DERIVATION_ROW_CARDINALITY_DRIFT_V{view}")
        identity_support = 0
        for vertex, skin_row in zip(sorted(mesh.vertices, key=lambda v: v.canonical_mesh_vertex_id), sorted(mesh_skin.rows, key=lambda r: r.canonical_mesh_vertex_id)):
            if vertex.canonical_mesh_vertex_id != skin_row.canonical_mesh_vertex_id:
                raise RuntimeError(f"DENSE_DERIVATION_VERTEX_ROW_ORDER_DRIFT_V{view}")
            coeffs = tuple(vertex.support_binding.coefficients)
            if vertex.support_binding.mode != "IDENTITY_SURFACE_NODE" or len(coeffs) != 1 or abs(float(coeffs[0][1]) - 1.0) > 1e-12:
                raise RuntimeError(f"DENSE_DERIVATION_VERTEX_NOT_EXACT_COMPACTION_ADDRESS_V{view}")
            if tuple(skin_row.source_support_coefficients) != coeffs:
                raise RuntimeError(f"DENSE_DERIVATION_SKIN_SUPPORT_REPLAY_DRIFT_V{view}")
            identity_support += 1
        rows.append({
            "view": view,
            "mesh_file": mesh_name,
            "mesh_file_sha256": _sha(mesh_path),
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "mesh_skin_file": skin_name,
            "mesh_skin_file_sha256": _sha(skin_path),
            "mesh_skin_lineage_hash": mesh_skin.mesh_skin_lineage_hash,
            "mesh_vertex_count": len(mesh.vertices),
            "mesh_face_count": len(mesh.faces),
            "identity_compaction_address_row_count": identity_support,
            "transfer_method": mesh_skin.transfer_method,
            "selected_dense_face_set_sha256": str(row["selected_face_set_sha256"]),
        })

    derivation = {
        "schema": SCHEMA,
        "status": DERIVATION_STATUS,
        "source_dense_body_manifest": body_manifest_path.name,
        "source_dense_body_manifest_sha256": body_manifest_sha,
        "source_zero_surface_sha256": EXPECTED_ZERO_SURFACE_SHA256,
        "topology_method": DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
        "compaction_replay_hash": replay_hash,
        "compaction_inverse_sha256": inverse_sha,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "transfer_method": SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD,
        "views": rows,
        "topology_and_W_share_exact_compaction_address": True,
        "original_dense_zero_surface_faces_only": True,
        "new_adjacency_created": False,
        "P1_or_P1Q_topology_used": False,
        "source_or_teacher_mesh_topology_used": False,
        "historical_weight_transfer_used": False,
        "direct_model_query_used": False,
        "training_executed": False,
        "product_pass_claimed": False,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    path = output / "FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION_MANIFEST.json"
    _write_json(path, derivation)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION",
        "derivation_manifest": path.name,
        "derivation_manifest_sha256": _sha(path),
        "source_dense_body_manifest_sha256": body_manifest_sha,
        "compaction_replay_hash": replay_hash,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "product_pass_claimed": False,
    }
    seal_path = output / "FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION_SEAL.json"
    _write_json(seal_path, seal)
    print("FIT2_DENSE_BODY_BINDING_DERIVATION=" + json.dumps(seal, sort_keys=True), flush=True)
    return {"derivation": derivation, "seal": seal}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dense-body-dir", required=True)
    p.add_argument("--fit2-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
