from __future__ import annotations

"""Bind sealed FIT1 Arachne V5 directly to current-authority P1_B2_G10 meshes.

This is a narrow product runner layered on the sealed V5 implementation. It requires
an exact current-authority P1 materialization manifest whose eight views each prove
the historical lineage-tag projection. Historical P1 hashes remain provenance only;
the current mesh hashes are the binding authority. No source skin rows are transferred,
and no optimizer/backward/update path is permitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.mesh.direct_model_skin import qualify_direct_model_mesh_skin
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as base
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageDemo.FIT1V5DirectP1CurrentAuthorityBinding.v1"


def _load_current_materialization(mesh_dir: Path, surface):
    path = mesh_dir / "P1_B2_G10_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    if not path.is_file():
        raise RuntimeError("P1_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST_MISSING")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS__CURRENT_CANONICAL_GSA_P1_B2_G10_MATERIALIZED__HISTORICAL_HASH_PROJECTION_EXACT":
        raise RuntimeError("P1_CURRENT_AUTHORITY_MATERIALIZATION_STATUS_INVALID")
    if value.get("current_canonical_gsa_lineage_hash") != surface.geometry_lineage_hash:
        raise RuntimeError("P1_CURRENT_AUTHORITY_SURFACE_LINEAGE_DRIFT")
    if bool(value.get("historical_lineage_relabelled_as_current", True)):
        raise RuntimeError("P1_CURRENT_AUTHORITY_HISTORICAL_RELABEL_FORBIDDEN")
    if bool(value.get("scientific_mechanical_surface_relabelled", True)):
        raise RuntimeError("P1_CURRENT_AUTHORITY_SCIENTIFIC_RELABEL_FORBIDDEN")
    rows = {int(row["view"]): row for row in value.get("views", ())}
    if set(rows) != set(range(8)):
        raise RuntimeError("P1_CURRENT_AUTHORITY_REQUIRES_EXACT_8_VIEWS")
    if not all(bool(rows[v].get("historical_projection_exact", False)) for v in range(8)):
        raise RuntimeError("P1_CURRENT_AUTHORITY_HISTORICAL_PROJECTION_NOT_EXACT")
    return path, value, rows


def run(args) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    if int(args.chunk) <= 0:
        raise ValueError("chunk must be positive")

    surface, tensor, replay = ceiling_v2._preflight_surface(args)
    if not bool(replay.get("gsa_lineage_exact_match")):
        raise RuntimeError("P1_DIRECT_BINDING_REQUIRES_EXACT_FIT2_GSA_LINEAGE")

    skeleton = base._load_skeleton(Path(args.skeleton))
    source_skin = base._load_source_skin(Path(args.source_skin))
    authority = base._load_v5_authority(
        Path(args.field_tokens), Path(args.conditioning_witness), Path(args.decoder_delta), device
    )
    if set(authority["joint_ids"]) != {str(j.canonical_joint_id) for j in skeleton.joints}:
        raise RuntimeError("V5_FIELD_TOKEN_SKELETON_JOINT_SET_DRIFT")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh_dir = Path(args.materialized_mesh_dir)
    materialization_path, _materialization, materialized_views = _load_current_materialization(mesh_dir, surface)

    rows, artifacts = [], []
    for view in range(8):
        mrow = materialized_views[view]
        mesh_path = mesh_dir / f"P1_B2_G10_CURRENT_V{view}_QUALIFIED_MESH_IR.json"
        mesh = base._load_mesh(mesh_path, surface, str(mrow["current_mesh_lineage_hash"]))
        if int(mesh.view_index) != view:
            raise RuntimeError(f"P1_VIEW_INDEX_DRIFT::{mesh.view_index}::{view}")

        geometry7, weights = base._predict_mesh_weights(mesh, surface, tensor, authority, device, int(args.chunk))
        weight_map = {
            str(vertex.canonical_mesh_vertex_id): tuple(float(x) for x in weights[i])
            for i, vertex in enumerate(mesh.vertices)
        }
        bound = qualify_direct_model_mesh_skin(
            surface,
            skeleton,
            source_skin,
            mesh,
            joint_ids=authority["joint_ids"],
            predicted_weights_by_vertex=weight_map,
            model_provenance={
                "architecture_id": "RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5",
                "field_tokens_file_sha256": base.EXPECTED_FIELD_TOKENS_FILE_SHA256,
                "field_token_payload_sha256": authority["field_token_payload_sha256"],
                "conditioning_witness_sha256": base.EXPECTED_CONDITIONING_WITNESS_SHA256,
                "decoder_delta_sha256": base.EXPECTED_DECODER_DELTA_SHA256,
                "backbone_checkpoint_sha256": base.EXPECTED_BACKBONE_SHA256,
                "optimizer_constructed": False,
                "backward_executed": False,
                "parameter_update_performed": False,
            },
        )

        weight_name = f"P1_B2_G10_V{view}_V5_DIRECT_WEIGHTS.npz"
        binding_name = f"P1_B2_G10_V{view}_QUALIFIED_MESH_SKIN_IR.json"
        weight_path, binding_path = out / weight_name, out / binding_name
        np.savez_compressed(
            weight_path,
            canonical_mesh_vertex_id=np.asarray([str(v.canonical_mesh_vertex_id) for v in mesh.vertices]),
            joint_ids=np.asarray(authority["joint_ids"]),
            weights=weights,
            geometry7=geometry7,
            mesh_lineage_hash=np.asarray([mesh.mesh_lineage_hash]),
            mesh_skin_lineage_hash=np.asarray([bound.mesh_skin_lineage_hash]),
        )
        base._write_json(binding_path, bound.to_dict())
        artifacts.extend([
            {"path": weight_name, "sha256": base._sha(weight_path), "bytes": weight_path.stat().st_size},
            {"path": binding_name, "sha256": base._sha(binding_path), "bytes": binding_path.stat().st_size},
        ])
        report = bound.qualification_report
        rows.append({
            "view": view,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "historical_projected_mesh_lineage_hash": str(mrow["historical_projected_mesh_lineage_hash"]),
            "historical_projection_exact": bool(mrow["historical_projection_exact"]),
            "mesh_skin_lineage_hash": bound.mesh_skin_lineage_hash,
            "vertex_count": len(mesh.vertices),
            "joint_count": len(authority["joint_ids"]),
            "direct_model_query": True,
            "source_skin_rows_consumed": False,
            "historical_barycentric_weight_transfer_used": False,
            "max_simplex_residual_before": float(report["max_simplex_residual_before"]),
            "total_correction_l1": float(report["total_correction_l1"]),
        })

    manifest = {
        "schema": SCHEMA,
        "status": "PASS__V5_DIRECT_QUERY_AND_COMPILER_CURRENT_AUTHORITY_P1_BINDING_V0_V7",
        "authority": "CURRENT_CANONICAL_GSA_P1_B2_G10_MATERIALIZATION_PLUS_SEALED_FIT1_V5",
        "p1_current_authority_materialization_manifest": materialization_path.name,
        "p1_current_authority_materialization_manifest_sha256": base._sha(materialization_path),
        "historical_p1_hashes_are_provenance_only": True,
        "current_p1_mesh_hashes_are_product_binding_authority": True,
        "fit2_arachne_training_executed": False,
        "a100_required": False,
        "optimizer_constructed": False,
        "backward_executed": False,
        "parameter_update_performed": False,
        "historical_barycentric_weight_transfer_used": False,
        "source_skin_rows_consumed": False,
        "field_tokens_file_sha256": base.EXPECTED_FIELD_TOKENS_FILE_SHA256,
        "conditioning_witness_sha256": base.EXPECTED_CONDITIONING_WITNESS_SHA256,
        "decoder_delta_sha256": base.EXPECTED_DECODER_DELTA_SHA256,
        "fit1_skeleton_file_sha256": base.EXPECTED_SKELETON_FILE_SHA256,
        "fit1_source_skin_file_sha256": base.EXPECTED_SOURCE_SKIN_FILE_SHA256,
        "fit1_skeleton_lineage_hash": base.EXPECTED_SKELETON_LINEAGE,
        "fit1_source_skin_lineage_hash": base.EXPECTED_SOURCE_SKIN_LINEAGE,
        "views": rows,
        "artifacts": artifacts,
        "product_pass_claimed": False,
        "unseen_generalization_claimed": False,
    }
    manifest_path = out / "V5_DIRECT_P1_BINDING_MANIFEST.json"
    base._write_json(manifest_path, manifest)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__V5_DIRECT_CURRENT_AUTHORITY_P1_BINDING",
        "manifest": manifest_path.name,
        "manifest_sha256": base._sha(manifest_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "product_pass_claimed": False,
    }
    base._write_json(out / "V5_DIRECT_P1_BINDING_SEAL.json", seal)
    print("V5_DIRECT_P1_BINDING=" + json.dumps(seal, sort_keys=True), flush=True)
    return {"manifest": manifest, "seal": seal}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--materialized-mesh-dir", required=True)
    p.add_argument("--field-tokens", required=True)
    p.add_argument("--conditioning-witness", required=True)
    p.add_argument("--decoder-delta", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--source-skin", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=2048)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
