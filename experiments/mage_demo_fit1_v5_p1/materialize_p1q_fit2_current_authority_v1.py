from __future__ import annotations

"""Materialize P1Q from current P1 geometry and fresh FIT2 surface-domain skin.

The only P1 -> P1Q mutation admitted here is the already-frozen exact face-subset
repair.  Vertices, SurfaceSupportBinding, and FIT2-derived mesh-skin rows remain
unchanged.  Mechanical authority is corrected FIT2 S/G/W throughout.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.appearance import build_observed_appearance_binding
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.face_subset_repair import qualify_exact_policy_face_subset_repair
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from compiler.realsas_compiler_core.product_external_render import (
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.v4 import build_mechanical_state

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_p1q_current_authority_v1 as legacy_p1q
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base
import experiments.mage_demo_fit1_v5_p1.materialize_p1_b2_g10_v1 as p1base
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageFIT2.P1QCurrentAuthorityFaceSubset.v1"
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n", encoding="utf-8")


def run(args) -> dict:
    p1_dir = Path(args.p1_dir)
    fit2_binding_dir = Path(args.fit2_binding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1_manifest_path = p1_dir / "P1_B2_G10_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    bind_manifest_path = fit2_binding_dir / "FIT2_P1_SKIN_BINDING_MANIFEST.json"
    if _sha(p1_manifest_path) != args.expected_p1_manifest:
        raise RuntimeError("FIT2_P1Q_P1_MANIFEST_SHA_DRIFT")
    if _sha(bind_manifest_path) != args.expected_fit2_binding_manifest:
        raise RuntimeError("FIT2_P1Q_BINDING_MANIFEST_SHA_DRIFT")

    bind_manifest = json.loads(bind_manifest_path.read_text(encoding="utf-8"))
    if bind_manifest.get("status") != "PASS__FRESH_FIT2_SKIN_BOUND_TO_CURRENT_P1_V0_V7":
        raise RuntimeError("FIT2_P1Q_BINDING_STATUS_INVALID")
    if bind_manifest.get("current_surface_lineage_hash") != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("FIT2_P1Q_BINDING_SURFACE_DRIFT")
    if bind_manifest.get("current_skeleton_lineage_hash") != fit2io.EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("FIT2_P1Q_BINDING_SKELETON_DRIFT")
    if bind_manifest.get("current_skin_lineage_hash") != fit2io.EXPECTED_SKIN_LINEAGE:
        raise RuntimeError("FIT2_P1Q_BINDING_SKIN_DRIFT")
    if bind_manifest.get("historical_fit1_skin_used") is not False:
        raise RuntimeError("FIT2_P1Q_HISTORICAL_SKIN_FORBIDDEN")

    current_surface, tensor, replay = ceiling_v2._preflight_surface(args)
    if not replay.get("gsa_lineage_exact_match") or current_surface.geometry_lineage_hash != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("FIT2_P1Q_CURRENT_GSA_DRIFT")
    if tensor.tensorization_hash != fit2io.EXPECTED_TENSORIZATION_HASH:
        raise RuntimeError("FIT2_P1Q_TENSORIZATION_DRIFT")

    skeleton = fit2io.load_skeleton(Path(args.skeleton))
    skin = fit2io.load_skin(Path(args.fit2_skin))
    mechanical = build_mechanical_state(current_surface, skeleton, skin)

    observations = tuple(map(Path, args.observations))
    cameras = tuple(map(Path, args.cameras))
    _baselines, domains = p1base._baseline_inputs(current_surface, observations, cameras)
    observation_hash = {view: ceiling_v1.OBSERVATION_SHA256[view] for view in range(8)}

    p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
    p1_rows = {int(row["view"]): row for row in p1_manifest["views"]}
    bind_rows = {int(row["view"]): row for row in bind_manifest["views"]}
    if set(p1_rows) != set(range(8)) or set(bind_rows) != set(range(8)):
        raise RuntimeError("FIT2_P1Q_REQUIRES_8_VIEWS")

    rows, artifacts, repaired_objects = [], [], {}
    for view in range(8):
        mesh = v5base._load_mesh(
            p1_dir / f"P1_B2_G10_CURRENT_V{view}_QUALIFIED_MESH_IR.json",
            current_surface,
            str(p1_rows[view]["current_mesh_lineage_hash"]),
        )
        skin_path = fit2_binding_dir / str(bind_rows[view]["file"])
        mesh_skin = legacy_p1q._load_mesh_skin(skin_path, mesh.mesh_lineage_hash)
        if mesh_skin.surface_binding_hash != fit2io.EXPECTED_SURFACE_LINEAGE:
            raise RuntimeError(f"FIT2_P1Q_MESH_SKIN_SURFACE_DRIFT_V{view}")
        if mesh_skin.skeleton_binding_hash != fit2io.EXPECTED_SKELETON_LINEAGE:
            raise RuntimeError(f"FIT2_P1Q_MESH_SKIN_SKELETON_DRIFT_V{view}")
        if mesh_skin.skin_binding_hash != fit2io.EXPECTED_SKIN_LINEAGE:
            raise RuntimeError(f"FIT2_P1Q_MESH_SKIN_SKIN_DRIFT_V{view}")

        atlas = content_sha256({
            "observation_sha256": observation_hash[view],
            "view": view,
            "authority": "EXACT_SOURCE_OBSERVATION",
        })
        appearance = build_observed_appearance_binding(
            surface=current_surface,
            mesh=mesh,
            target_view_index=view,
            camera_binding_hash=mesh.camera_binding_hash,
            observation_hash_by_view=observation_hash,
            atlas_payload_hash=atlas,
        )
        source_component = build_external_renderable_component(
            component_id="FULL_SUBJECT_UNDERLAY",
            view_index=view,
            mesh=mesh,
            mesh_skin=mesh_skin,
            mechanical=mechanical,
            appearance=appearance,
            setup_order=0,
            coverage_classification=mesh.support_coverage_classification,
            materialization_manifest_sha256=_sha(p1_manifest_path),
            direct_binding_manifest_sha256=_sha(bind_manifest_path),
            metadata={
                "p1q_source": True,
                "fit2_current_skin": True,
                "semantic_component_partition_claimed": False,
            },
        )

        remove = legacy_p1q._sliver_indices(mesh)
        coverage = legacy_p1q._coverage_for(mesh, domains[view], remove)
        coverage_witness = content_sha256({
            "schema": "RealSaS.FIT2P1QCoverageWitness.v1",
            "view": view,
            "source_mesh": mesh.mesh_lineage_hash,
            "removed": remove,
            "observation_mask_sha256": domains[view].mask_sha256,
            "coverage": coverage,
        })
        ownership = content_sha256({
            "schema": "RealSaS.FIT2P1QObservationOwnershipWitness.v2",
            "surface_lineage_hash": current_surface.geometry_lineage_hash,
            "observation_sha256": observation_hash[view],
            "observation_mask_sha256": domains[view].mask_sha256,
            "view": view,
            "ownership": "QUALIFIED_SURFACE_PLUS_EXACT_OBSERVATION_DOMAIN",
            "teacher_truth_used": False,
            "source_component_truth_used": False,
            "semantic_component_partition_claimed": False,
        })

        if remove:
            result = qualify_exact_policy_face_subset_repair(
                source_component,
                mechanical,
                remove_face_indices=remove,
                repaired_coverage_report=coverage,
                coverage_measurement_sha256=coverage_witness,
                source_truth_ownership_sha256=ownership,
                metadata={
                    "vertices_mutated": False,
                    "edges_mutated": False,
                    "weights_mutated": False,
                    "retriangulated": False,
                    "new_pixels_generated": False,
                    "policy_thresholds_changed": False,
                    "scientific_mechanical_surface_relabelled": False,
                    "fit2_skin_lineage_preserved": fit2io.EXPECTED_SKIN_LINEAGE,
                },
            )
            repaired_mesh, repaired_skin, repaired_appearance = result.mesh, result.mesh_skin, result.appearance
            mode = "EXACT_POLICY_FACE_SUBSET_REPAIR"
            repair_payload = result.qualification.to_dict()
        else:
            quality = evaluate_mesh_quality(
                coverage=coverage,
                raster_report=mesh_raster_quality_report(mesh, surface=None, view_index=view),
                policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
            )
            if not quality["passed"]:
                raise RuntimeError(f"FIT2_P1Q_IDENTITY_VIEW_NOT_FULL_POLICY_PASS_V{view}:{quality['failure_invariants']}")
            repaired_mesh, repaired_skin, repaired_appearance = mesh, mesh_skin, appearance
            mode = "IDENTITY_ALREADY_FULL_POLICY_PASS"
            repair_payload = {
                "schema": "RealSaS.FIT2P1QIdentityQualification.v1",
                "view_index": view,
                "status": "PASS",
                "source_mesh_lineage_hash": mesh.mesh_lineage_hash,
                "coverage_measurement_sha256": coverage_witness,
                "surface_observation_ownership_sha256": ownership,
                "quality": quality,
            }

        if repaired_skin.skin_binding_hash != fit2io.EXPECTED_SKIN_LINEAGE:
            raise RuntimeError(f"FIT2_P1Q_REPAIRED_SKIN_LINEAGE_DRIFT_V{view}")
        final_quality = evaluate_mesh_quality(
            coverage=coverage,
            raster_report=mesh_raster_quality_report(repaired_mesh, surface=None, view_index=view),
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        if not final_quality["passed"]:
            raise RuntimeError(f"FIT2_P1Q_FINAL_POLICY_FAIL_V{view}:{final_quality['failure_invariants']}")

        names = {
            "mesh": f"P1Q_FIT2_CURRENT_V{view}_QUALIFIED_MESH_IR.json",
            "skin": f"P1Q_FIT2_CURRENT_V{view}_QUALIFIED_MESH_SKIN_IR.json",
            "appearance": f"P1Q_FIT2_CURRENT_V{view}_QUALIFIED_APPEARANCE_IR.json",
            "repair": f"P1Q_FIT2_CURRENT_V{view}_FACE_SUBSET_QUALIFICATION.json",
        }
        payloads = {
            "mesh": repaired_mesh.to_dict(),
            "skin": repaired_skin.to_dict(),
            "appearance": repaired_appearance.to_dict(),
            "repair": repair_payload,
        }
        for key, name in names.items():
            path = output_dir / name
            _write_json(path, payloads[key])
            artifacts.append({"path": name, "sha256": _sha(path), "bytes": path.stat().st_size})

        row = {
            "view": view,
            "mode": mode,
            "removed_face_indices": list(remove),
            "removed_face_count": len(remove),
            "source_mesh_lineage_hash": mesh.mesh_lineage_hash,
            "mesh_lineage_hash": repaired_mesh.mesh_lineage_hash,
            "source_mesh_skin_lineage_hash": mesh_skin.mesh_skin_lineage_hash,
            "mesh_skin_lineage_hash": repaired_skin.mesh_skin_lineage_hash,
            "skin_binding_hash": repaired_skin.skin_binding_hash,
            "appearance_lineage_hash": repaired_appearance.appearance_lineage_hash,
            "coverage": coverage,
            "coverage_measurement_sha256": coverage_witness,
            "surface_observation_ownership_sha256": ownership,
            "final_mesh_quality": final_quality,
            "files": names,
        }
        rows.append(row)
        repaired_objects[view] = (repaired_mesh, repaired_skin, repaired_appearance)
        print("FIT2_P1Q_V" + str(view) + "=" + json.dumps({
            "removed": len(remove),
            "indices": list(remove),
            "recall": coverage["source_alpha_recall"],
            "precision": coverage["precision_inside_alpha"],
            "skin": repaired_skin.mesh_skin_lineage_hash,
        }, sort_keys=True), flush=True)

    materialization = {
        "schema": SCHEMA,
        "status": "PASS__FIT2_P1Q_CURRENT_AUTHORITY_V0_V7_FROZEN_FACE_POLICY",
        "source_p1_manifest_sha256": _sha(p1_manifest_path),
        "source_fit2_binding_manifest_sha256": _sha(bind_manifest_path),
        "current_gsa_lineage_hash": current_surface.geometry_lineage_hash,
        "current_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "current_skin_lineage_hash": skin.skin_lineage_hash,
        "teacher_truth_used": False,
        "source_component_truth_used": False,
        "historical_fit1_skin_used": False,
        "weight_rows_mutated": False,
        "faces_only_policy": True,
        "views": rows,
        "artifacts": artifacts,
        "product_pass_claimed": False,
    }
    materialization_path = output_dir / "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    _write_json(materialization_path, materialization)
    materialization_sha = _sha(materialization_path)

    derivation = {
        "schema": SCHEMA + ".BindingDerivation.v1",
        "status": "PASS__FIT2_ROWS_PRESERVED_THROUGH_P1Q_FACE_SUBSET",
        "source_fit2_binding_manifest_sha256": _sha(bind_manifest_path),
        "p1q_materialization_manifest_sha256": materialization_sha,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "rows": [{
            "view": row["view"],
            "source_mesh_skin_lineage_hash": row["source_mesh_skin_lineage_hash"],
            "p1q_mesh_skin_lineage_hash": row["mesh_skin_lineage_hash"],
            "weight_rows_preserved_exactly": True,
            "faces_only_changed": row["mode"] == "EXACT_POLICY_FACE_SUBSET_REPAIR",
        } for row in rows],
        "historical_weight_transfer_used": False,
        "product_pass_claimed": False,
    }
    derivation_path = output_dir / "P1Q_FIT2_CURRENT_BINDING_DERIVATION_MANIFEST.json"
    _write_json(derivation_path, derivation)
    derivation_sha = _sha(derivation_path)

    directions = []
    for view in range(8):
        repaired_mesh, repaired_skin, repaired_appearance = repaired_objects[view]
        component = build_external_renderable_component(
            component_id="FULL_SUBJECT_UNDERLAY",
            view_index=view,
            mesh=repaired_mesh,
            mesh_skin=repaired_skin,
            mechanical=mechanical,
            appearance=repaired_appearance,
            setup_order=0,
            coverage_classification=repaired_mesh.support_coverage_classification,
            materialization_manifest_sha256=materialization_sha,
            direct_binding_manifest_sha256=derivation_sha,
            metadata={"p1q_fit2_current": True, "semantic_component_partition_claimed": False},
        )
        directions.append(build_external_directional_renderable(
            view_index=view,
            camera_binding_hash=repaired_mesh.camera_binding_hash,
            components=(component,),
            mechanical=mechanical,
            metadata={"p1q_fit2_current": True},
        ))
    render_set = build_external_directional_renderable_set(tuple(directions), mechanical, metadata={"p1q_fit2_current": True})
    render_name = "P1Q_FIT2_CURRENT_DIRECTIONAL_RENDERABLE_SET.json"
    render_path = output_dir / render_name
    _write_json(render_path, render_set.to_dict())
    artifacts.append({"path": render_name, "sha256": _sha(render_path), "bytes": render_path.stat().st_size})

    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__FIT2_P1Q_CURRENT_AUTHORITY_V0_V7",
        "manifest": materialization_path.name,
        "manifest_sha256": materialization_sha,
        "binding_derivation_sha256": derivation_sha,
        "directional_visual_state_hash": render_set.directional_visual_state_hash,
        "surface_lineage_hash": current_surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "artifact_count": len(artifacts),
        "product_pass_claimed": False,
    }
    seal_path = output_dir / "P1Q_FIT2_CURRENT_AUTHORITY_SEAL.json"
    _write_json(seal_path, seal)
    print("FIT2_P1Q_SEAL=" + json.dumps(seal, sort_keys=True), flush=True)
    return seal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--p1-dir", required=True)
    p.add_argument("--fit2-binding-dir", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--expected-p1-manifest", required=True)
    p.add_argument("--expected-fit2-binding-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
