from __future__ import annotations

"""Bind the fresh corrected-lineage FIT2 skin to the exact current P1 meshes.

This is the causal replacement for the historical FIT1 V5 direct-query seam.  Mesh
vertices already carry qualified SurfaceSupportBinding against the corrected GSA;
therefore mesh skin is derived only as the frozen convex transfer

    w(v,j) = sum_s a(v,s) * w(s,j)

from the Compiler-qualified FIT2 QualifiedSkinIR.  No model query, nearest-neighbor
transfer, semantic repair, optimizer step, or historical 950-node authority is used.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.mesh.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.v4 import build_mechanical_state

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageFIT2.P1CurrentSkinBinding.v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path}")
    return value


def run(args) -> dict:
    p1_dir = Path(args.p1_dir)
    skeleton_path = Path(args.skeleton)
    skin_path = Path(args.fit2_skin)
    handoff_path = Path(args.fit2_handoff)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1_manifest_path = p1_dir / "P1_B2_G10_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    if _sha(p1_manifest_path) != str(args.expected_p1_manifest):
        raise RuntimeError("FIT2_BIND_P1_MANIFEST_SHA_DRIFT")
    p1_manifest = _load_json(p1_manifest_path)
    if p1_manifest.get("status") != "PASS__CURRENT_CANONICAL_GSA_P1_B2_G10_MATERIALIZED__HISTORICAL_HASH_PROJECTION_EXACT":
        raise RuntimeError("FIT2_BIND_P1_STATUS_INVALID")
    if p1_manifest.get("current_canonical_gsa_lineage_hash") != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("FIT2_BIND_P1_SURFACE_LINEAGE_DRIFT")

    # Rebuild the exact corrected 8171/23656 GSA through the already-qualified current path.
    surface, tensor, replay = ceiling_v2._preflight_surface(args)
    if not replay.get("gsa_lineage_exact_match"):
        raise RuntimeError("FIT2_BIND_CURRENT_GSA_REPLAY_FAIL")
    if surface.geometry_lineage_hash != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("FIT2_BIND_CURRENT_GSA_LINEAGE_DRIFT")
    if int(tensor.node_count) != fit2io.EXPECTED_SURFACE_NODES or int(tensor.edge_count) != fit2io.EXPECTED_SURFACE_RELATIONS:
        raise RuntimeError("FIT2_BIND_CURRENT_GSA_CARDINALITY_DRIFT")
    if tensor.tensorization_hash != fit2io.EXPECTED_TENSORIZATION_HASH:
        raise RuntimeError("FIT2_BIND_CURRENT_GSA_TENSORIZATION_DRIFT")

    skeleton = fit2io.load_skeleton(skeleton_path)
    skin = fit2io.load_skin(skin_path)
    mechanical = build_mechanical_state(surface, skeleton, skin)

    handoff = _load_json(handoff_path)
    handoff_checks = {
        "status": handoff.get("status") == "AUTHORIZED__FIT2_ARACHNE_COMPILER_QUALIFIED",
        "compiler_qualified": handoff.get("compiler_qualified") is True,
        "science_gate": handoff.get("science_gate") is True,
        "terminal_stable": handoff.get("terminal_stable") is True,
        "surface_lineage": handoff.get("surface_lineage") == fit2io.EXPECTED_SURFACE_LINEAGE,
        "skeleton_lineage": handoff.get("skeleton_lineage") == fit2io.EXPECTED_SKELETON_LINEAGE,
        "skin_lineage": handoff.get("skin_lineage") == fit2io.EXPECTED_SKIN_LINEAGE,
        "model_sha": handoff.get("model_sha256") == fit2io.EXPECTED_MODEL_SHA256,
        "qualified_skin_sha": handoff.get("qualified_skin_sha256") == fit2io.EXPECTED_SKIN_SHA256,
        "historical_checkpoint_loaded": handoff.get("historical_checkpoint_loaded") is False,
        "product_pass_not_claimed": handoff.get("product_pass_claimed") is False,
    }
    if not all(handoff_checks.values()):
        raise RuntimeError("FIT2_BIND_ARACHNE_HANDOFF_INVALID:" + json.dumps(handoff_checks, sort_keys=True))

    surface_path = output_dir / "CURRENT_FIT2_RIGGING_SURFACE_IR.json"
    _write_json(surface_path, surface.to_dict())

    p1_rows = {int(row["view"]): row for row in p1_manifest.get("views", ())}
    if set(p1_rows) != set(range(8)):
        raise RuntimeError("FIT2_BIND_P1_VIEW_SET_INVALID")

    rows = []
    artifacts = [{"path": surface_path.name, "sha256": _sha(surface_path), "bytes": surface_path.stat().st_size}]
    for view in range(8):
        p1row = p1_rows[view]
        mesh = v5base._load_mesh(
            p1_dir / f"P1_B2_G10_CURRENT_V{view}_QUALIFIED_MESH_IR.json",
            surface,
            str(p1row["current_mesh_lineage_hash"]),
        )
        mesh_skin = bind_mwb2_mesh_skin(surface, skeleton, skin, mesh)
        if mesh_skin.skin_binding_hash != fit2io.EXPECTED_SKIN_LINEAGE:
            raise RuntimeError(f"FIT2_BIND_SKIN_LINEAGE_DRIFT_V{view}")
        if mesh_skin.surface_binding_hash != fit2io.EXPECTED_SURFACE_LINEAGE:
            raise RuntimeError(f"FIT2_BIND_SURFACE_LINEAGE_DRIFT_V{view}")
        if mesh_skin.skeleton_binding_hash != fit2io.EXPECTED_SKELETON_LINEAGE:
            raise RuntimeError(f"FIT2_BIND_SKELETON_LINEAGE_DRIFT_V{view}")
        if mesh_skin.mesh_binding_hash != mesh.mesh_lineage_hash:
            raise RuntimeError(f"FIT2_BIND_MESH_LINEAGE_DRIFT_V{view}")
        if len(mesh_skin.rows) != len(mesh.vertices):
            raise RuntimeError(f"FIT2_BIND_ROW_COUNT_DRIFT_V{view}")

        name = f"P1_B2_G10_V{view}_FIT2_QUALIFIED_MESH_SKIN_IR.json"
        path = output_dir / name
        _write_json(path, mesh_skin.to_dict())
        artifacts.append({"path": name, "sha256": _sha(path), "bytes": path.stat().st_size})
        rows.append({
            "view": view,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "mesh_skin_lineage_hash": mesh_skin.mesh_skin_lineage_hash,
            "row_count": len(mesh_skin.rows),
            "transfer_method": mesh_skin.transfer_method,
            "qualification_report": dict(mesh_skin.qualification_report),
            "file": name,
        })
        print("FIT2_P1_BIND_V" + str(view) + "=" + json.dumps({
            "mesh": mesh.mesh_lineage_hash,
            "mesh_skin": mesh_skin.mesh_skin_lineage_hash,
            "rows": len(mesh_skin.rows),
            "corr": mesh_skin.qualification_report.get("total_correction_l1"),
        }, sort_keys=True), flush=True)

    manifest = {
        "schema": SCHEMA,
        "status": "PASS__FRESH_FIT2_SKIN_BOUND_TO_CURRENT_P1_V0_V7",
        "source_p1_manifest_sha256": _sha(p1_manifest_path),
        "fit2_handoff_sha256": _sha(handoff_path),
        "current_surface_lineage_hash": surface.geometry_lineage_hash,
        "current_surface_file": surface_path.name,
        "current_surface_file_sha256": _sha(surface_path),
        "current_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "current_skeleton_file_sha256": _sha(skeleton_path),
        "current_skin_lineage_hash": skin.skin_lineage_hash,
        "current_skin_file_sha256": _sha(skin_path),
        "arachne_model_sha256": fit2io.EXPECTED_MODEL_SHA256,
        "binding_method": "SURFACE_SUPPORT_CONVEX_TRANSFER_V1",
        "historical_fit1_skin_used": False,
        "historical_checkpoint_loaded": False,
        "model_query_on_mesh_vertices": False,
        "semantic_skin_synthesis": False,
        "product_pass_claimed": False,
        "views": rows,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "FIT2_P1_SKIN_BINDING_MANIFEST.json"
    _write_json(manifest_path, manifest)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__FRESH_FIT2_SKIN_CURRENT_P1_BINDING",
        "manifest": manifest_path.name,
        "manifest_sha256": _sha(manifest_path),
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "product_pass_claimed": False,
    }
    seal_path = output_dir / "FIT2_P1_SKIN_BINDING_SEAL.json"
    _write_json(seal_path, seal)
    print("FIT2_P1_SKIN_BINDING_SEAL=" + json.dumps(seal, sort_keys=True), flush=True)
    return seal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--p1-dir", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--fit2-handoff", required=True)
    p.add_argument("--expected-p1-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
