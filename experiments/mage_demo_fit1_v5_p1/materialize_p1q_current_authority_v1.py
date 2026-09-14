from __future__ import annotations

"""Materialize current-authority P1Q by exact frozen-policy face-subset repair.

This bounded product runner consumes only:
- current-authority P1 meshes,
- direct sealed FIT1 V5 mesh skin on those exact vertices,
- current corrected GSA only as external render/raster support,
- sealed FIT1 scientific S/G/W as mechanical authority,
- exact source observation alpha for reraster measurement.

It never retriangulates, repairs weights, generates pixels, relabels the scientific
surface, or forces historical P1Q hashes/counts. Identity views remain identity only
when the frozen product mesh policy already passes.
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
    _triangle_metrics,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from compiler.realsas_compiler_core.product_external_render import (
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.types import QualifiedMeshSkinIR, QualifiedMeshSkinRow
from compiler.realsas_compiler_core.v4 import build_mechanical_state

import experiments.mage_demo_fit1_v5_p1.materialize_p1_b2_g10_v1 as p1base
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2
import experiments.single_family_e2e_v1.run_mage_fit1_real_static_v1 as fit1


SCHEMA = "RealSaS.MageDemo.P1QCurrentAuthorityFaceSubset.v2"
SOURCE_TRUTH_EXPECTED = "a23565b0904e9683d11083984a87405f4e0a1069984ca11b690ad431f35f5e86"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n", encoding="utf-8")


def _load_mesh_skin(path: Path, expected_mesh: str) -> QualifiedMeshSkinIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(
        QualifiedMeshSkinRow(
            canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
            influences=tuple((str(jid), float(weight)) for jid, weight in row["influences"]),
            source_support_coefficients=tuple((str(sid), float(weight)) for sid, weight in row["source_support_coefficients"]),
            simplex_residual_before=float(row["simplex_residual_before"]),
            correction_l1=float(row["correction_l1"]),
        )
        for row in data["rows"]
    )
    value = QualifiedMeshSkinIR(
        rows=rows,
        surface_binding_hash=str(data["surface_binding_hash"]),
        skeleton_binding_hash=str(data["skeleton_binding_hash"]),
        skin_binding_hash=str(data["skin_binding_hash"]),
        mesh_binding_hash=str(data["mesh_binding_hash"]),
        transfer_method=str(data["transfer_method"]),
        qualification_report=dict(data.get("qualification_report") or {}),
        mesh_skin_lineage_hash=str(data["mesh_skin_lineage_hash"]),
        schema_version=str(data.get("schema_version") or "RealSaS.QualifiedMeshSkinIR.v1"),
        metadata=dict(data.get("metadata") or {}),
    )
    if value.mesh_binding_hash != expected_mesh:
        raise RuntimeError("V5_MESH_BINDING_DRIFT")
    return value


def _sliver_indices(mesh) -> tuple[int, ...]:
    raster_xy = {vertex.canonical_mesh_vertex_id: tuple(map(float, vertex.metadata["raster_xy"])) for vertex in mesh.vertices}
    bad = []
    for face_index, face in enumerate(mesh.faces):
        area, angle, aspect = _triangle_metrics(raster_xy[face[0]], raster_xy[face[1]], raster_xy[face[2]])
        if area <= 1e-12:
            raise RuntimeError(f"DEGENERATE_FACE_UNSUPPORTED:{face_index}")
        if (
            angle < FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.min_raster_triangle_angle_deg
            or aspect > FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.max_raster_triangle_aspect_ratio
        ):
            bad.append(face_index)
    return tuple(bad)


def _coverage_for(mesh, domain, remove=()):
    removed = set(map(int, remove))
    raster_xy = {vertex.canonical_mesh_vertex_id: tuple(map(float, vertex.metadata["raster_xy"])) for vertex in mesh.vertices}
    triangles = [
        (raster_xy[face[0]], raster_xy[face[1]], raster_xy[face[2]])
        for face_index, face in enumerate(mesh.faces)
        if face_index not in removed
    ]
    return domain.coverage(triangles)


def run(args) -> dict:
    p1_dir = Path(args.p1_dir)
    v5_dir = Path(args.v5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1_manifest_path = p1_dir / "P1_B2_G10_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    v5_manifest_path = v5_dir / "V5_DIRECT_P1_BINDING_MANIFEST.json"
    if _sha(p1_manifest_path) != args.expected_p1_manifest:
        raise RuntimeError("P1_MANIFEST_SHA_DRIFT")
    if _sha(v5_manifest_path) != args.expected_v5_manifest:
        raise RuntimeError("V5_MANIFEST_SHA_DRIFT")

    source_truth = Path(args.source_truth)
    if _sha(source_truth) != SOURCE_TRUTH_EXPECTED:
        raise RuntimeError("SOURCE_TRUTH_SHA_DRIFT")

    current_surface, _tensor, replay = ceiling_v2._preflight_surface(args)
    if not replay["gsa_lineage_exact_match"]:
        raise RuntimeError("CURRENT_GSA_DRIFT")
    _baselines, domains = p1base._baseline_inputs(
        current_surface,
        tuple(map(Path, args.observations)),
        tuple(map(Path, args.cameras)),
    )

    fit_surface = fit1._load_surface(Path(args.fit1_surface))
    skeleton = fit1._load_skeleton(Path(args.skeleton))
    fit_skin = fit1._load_skin(Path(args.fit1_skin))
    mechanical = build_mechanical_state(fit_surface, skeleton, fit_skin)
    observation_hash = {view: ceiling_v1.OBSERVATION_SHA256[view] for view in range(8)}

    p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
    p1_rows = {int(row["view"]): row for row in p1_manifest["views"]}
    rows, artifacts, repaired_objects = [], [], {}

    for view in range(8):
        mesh = v5base._load_mesh(
            p1_dir / f"P1_B2_G10_CURRENT_V{view}_QUALIFIED_MESH_IR.json",
            current_surface,
            str(p1_rows[view]["current_mesh_lineage_hash"]),
        )
        mesh_skin = _load_mesh_skin(
            v5_dir / f"P1_B2_G10_V{view}_QUALIFIED_MESH_SKIN_IR.json",
            mesh.mesh_lineage_hash,
        )
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
            direct_binding_manifest_sha256=_sha(v5_manifest_path),
            metadata={"p1q_source": True, "semantic_component_partition_claimed": False},
        )

        remove = _sliver_indices(mesh)
        coverage = _coverage_for(mesh, domains[view], remove)
        coverage_witness = content_sha256({
            "schema": "RealSaS.P1QCoverageWitness.v1",
            "view": view,
            "source_mesh": mesh.mesh_lineage_hash,
            "removed": remove,
            "observation_mask_sha256": domains[view].mask_sha256,
            "coverage": coverage,
        })
        ownership = content_sha256({
            "schema": "RealSaS.P1QSourceTruthOwnershipWitness.v1",
            "source_truth_sha256": SOURCE_TRUTH_EXPECTED,
            "view": view,
            "ownership": "FULL_SUBJECT_OBSERVATION_SUBSTRATE",
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
                raise RuntimeError(f"IDENTITY_VIEW_NOT_FULL_POLICY_PASS_V{view}:{quality['failure_invariants']}")
            repaired_mesh, repaired_skin, repaired_appearance = mesh, mesh_skin, appearance
            mode = "IDENTITY_ALREADY_FULL_POLICY_PASS"
            repair_payload = {
                "schema": "RealSaS.P1QIdentityQualification.v1",
                "view_index": view,
                "status": "PASS",
                "source_mesh_lineage_hash": mesh.mesh_lineage_hash,
                "coverage_measurement_sha256": coverage_witness,
                "source_truth_ownership_sha256": ownership,
                "quality": quality,
            }

        final_quality = evaluate_mesh_quality(
            coverage=coverage,
            raster_report=mesh_raster_quality_report(repaired_mesh, surface=None, view_index=view),
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        if not final_quality["passed"]:
            raise RuntimeError(f"P1Q_FINAL_POLICY_FAIL_V{view}:{final_quality['failure_invariants']}")

        names = {
            "mesh": f"P1Q_CURRENT_V{view}_QUALIFIED_MESH_IR.json",
            "skin": f"P1Q_CURRENT_V{view}_QUALIFIED_MESH_SKIN_IR.json",
            "appearance": f"P1Q_CURRENT_V{view}_QUALIFIED_APPEARANCE_IR.json",
            "repair": f"P1Q_CURRENT_V{view}_FACE_SUBSET_QUALIFICATION.json",
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
            "appearance_lineage_hash": repaired_appearance.appearance_lineage_hash,
            "coverage": coverage,
            "coverage_measurement_sha256": coverage_witness,
            "source_truth_ownership_sha256": ownership,
            "final_mesh_quality": final_quality,
            "files": names,
        }
        rows.append(row)
        repaired_objects[view] = (repaired_mesh, repaired_skin, repaired_appearance)
        print("P1Q_V" + str(view) + "=" + json.dumps({
            "removed": len(remove),
            "indices": list(remove),
            "recall": coverage["source_alpha_recall"],
            "precision": coverage["precision_inside_alpha"],
            "min_angle": final_quality["min_raster_triangle_angle_deg"],
            "max_aspect": final_quality["max_raster_triangle_aspect_ratio"],
        }, sort_keys=True), flush=True)

    materialization = {
        "schema": SCHEMA,
        "status": "PASS__P1Q_CURRENT_AUTHORITY_V0_V7_FROZEN_POLICY",
        "source_p1_manifest_sha256": _sha(p1_manifest_path),
        "source_v5_direct_manifest_sha256": _sha(v5_manifest_path),
        "current_gsa_lineage_hash": current_surface.geometry_lineage_hash,
        "scientific_fit1_surface_lineage_hash": fit_surface.geometry_lineage_hash,
        "source_truth_sha256": SOURCE_TRUTH_EXPECTED,
        "views": rows,
        "artifacts": artifacts,
        "product_pass_claimed": False,
    }
    materialization_path = output_dir / "P1Q_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    _write_json(materialization_path, materialization)
    materialization_sha = _sha(materialization_path)

    derivation = {
        "schema": SCHEMA + ".DirectBindingDerivation.v1",
        "status": "PASS__UNCHANGED_V5_ROWS_REBOUND_ONLY_TO_P1Q_FACE_SUBSET",
        "source_v5_direct_manifest_sha256": _sha(v5_manifest_path),
        "p1q_materialization_manifest_sha256": materialization_sha,
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
    derivation_path = output_dir / "P1Q_CURRENT_DIRECT_BINDING_DERIVATION_MANIFEST.json"
    _write_json(derivation_path, derivation)
    derivation_sha = _sha(derivation_path)

    directions, binding_rows = [], []
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
            metadata={"p1q_current": True, "semantic_component_partition_claimed": False},
        )
        direction = build_external_directional_renderable(
            view_index=view,
            camera_binding_hash=repaired_mesh.camera_binding_hash,
            components=(component,),
            mechanical=mechanical,
            metadata={"p1q_current": True},
        )
        directions.append(direction)
        binding_rows.append({
            "view": view,
            "component_state_hash": component.component_state_hash,
            "direction_state_hash": direction.direction_state_hash,
            "external_render_support_qualification_hash": component.metadata["external_render_support_qualification"]["qualification_hash"],
        })

    directional_set = build_external_directional_renderable_set(
        tuple(directions),
        mechanical,
        metadata={"p1q_current": True},
    )
    external = {
        "schema": SCHEMA + ".ExternalRenderBinding.v1",
        "status": "PASS__P1Q_EXTERNAL_RENDER_SUPPORT_V0_V7",
        "p1q_materialization_manifest_sha256": materialization_sha,
        "p1q_direct_binding_derivation_sha256": derivation_sha,
        "directional_visual_state_hash": directional_set.directional_visual_state_hash,
        "views": binding_rows,
        "product_pass_claimed": False,
    }
    external_path = output_dir / "P1Q_CURRENT_EXTERNAL_RENDER_BINDING_MANIFEST.json"
    _write_json(external_path, external)
    external_sha = _sha(external_path)

    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__P1Q_CURRENT_AUTHORITY_V0_V7",
        "materialization_manifest": materialization_path.name,
        "materialization_manifest_sha256": materialization_sha,
        "direct_binding_derivation_manifest": derivation_path.name,
        "direct_binding_derivation_sha256": derivation_sha,
        "external_render_binding_manifest": external_path.name,
        "external_render_binding_manifest_sha256": external_sha,
        "directional_visual_state_hash": directional_set.directional_visual_state_hash,
        "artifact_count": len(artifacts) + 3,
        "product_pass_claimed": False,
    }
    seal_path = output_dir / "P1Q_CURRENT_AUTHORITY_SEAL.json"
    _write_json(seal_path, seal)
    print("P1Q_SEAL=" + json.dumps(seal, sort_keys=True), flush=True)
    return seal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-surface", required=True)
    parser.add_argument("--cameras", nargs=8, required=True)
    parser.add_argument("--observations", nargs=8, required=True)
    parser.add_argument("--p1-dir", required=True)
    parser.add_argument("--v5-dir", required=True)
    parser.add_argument("--fit1-surface", required=True)
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--fit1-skin", required=True)
    parser.add_argument("--source-truth", required=True)
    parser.add_argument("--expected-p1-manifest", required=True)
    parser.add_argument("--expected-v5-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
