from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import argparse
import json
import shutil

from compiler.realsas_compiler_core.appearance import build_observed_appearance_binding
from compiler.realsas_compiler_core.bundle_routes import route_for
from compiler.realsas_compiler_core.directional_binding import qualify_directional_joint_view_binding
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion import build_deterministic_preset_motion
from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh
from compiler.realsas_compiler_core.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from compiler.realsas_compiler_core.v4 import (
    assemble_product_v3,
    build_directional_renderable,
    build_directional_renderable_set,
    build_mechanical_state,
    build_renderable_component,
    make_single_family_e2e_capability_contract,
)
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2


EXPECTED_SURFACE_LINEAGE = "67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb"
EXPECTED_SKELETON_LINEAGE = "738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306"
EXPECTED_SKIN_LINEAGE = "ef28f75e0306dbbabc32e75b837412ede39b180248502f6e504994310d91edaf"
EXPECTED_VIEWS = tuple(range(8))


def _load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_typed(root: Path, value) -> str:
    route = route_for(value)
    path = root / route.section / route.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path.relative_to(root).as_posix()


def _load_surface(path: Path) -> RiggingSurfaceIR:
    raw = _load_json(path)
    nodes = tuple(
        SurfaceNode(
            surface_id=str(row["surface_id"]),
            P=tuple(map(float, row["P"])),
            support_views=tuple(map(int, row.get("support_views") or ())),
            provenance_refs=tuple(map(str, row.get("provenance_refs") or ())),
            source_observation_ids=tuple(map(str, row.get("source_observation_ids") or ())),
            raster_bindings=tuple(
                (int(view), tuple(map(float, xy)))
                for view, xy in row.get("raster_bindings") or ()
            ),
            persistence_group_id=row.get("persistence_group_id"),
            derived_normal=None if row.get("derived_normal") is None else tuple(map(float, row["derived_normal"])),
            validity_flags=tuple(map(str, row.get("validity_flags") or ())),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in raw.get("surface_nodes") or ()
    )
    relations = tuple(
        SurfaceRelation(
            relation_id=str(row["relation_id"]),
            a_surface_id=str(row["a_surface_id"]),
            b_surface_id=str(row["b_surface_id"]),
            relation_kind=str(row["relation_kind"]),
            score=float(row.get("score", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in raw.get("local_relations") or ()
    )
    value = RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash=str(raw["geometry_lineage_hash"]),
        builder_id=str(raw.get("builder_id") or "RealSaS.GeometricSubstrateAssembler.current"),
        schema_version=str(raw.get("schema_version") or "RealSaS.RiggingSurfaceIR.v1"),
        metadata=dict(raw.get("metadata") or {}),
    )
    if value.geometry_lineage_hash != EXPECTED_SURFACE_LINEAGE:
        raise ValueError("MAGE_FIT1_SURFACE_LINEAGE_DRIFT")
    if len(value.surface_nodes) != 950 or len(value.local_relations) != 2813:
        raise ValueError("MAGE_FIT1_SURFACE_CARDINALITY_DRIFT")
    if value.metadata.get("raster_coordinate_system") != "PIXEL_CENTER_XY" or int(value.metadata.get("resolution", 0)) != 1024:
        raise ValueError("MAGE_FIT1_RASTER_CONTRACT_DRIFT")
    return value


def _load_skeleton(path: Path) -> QualifiedSkeletonIRV2:
    raw = _load_json(path)
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=str(row["canonical_joint_id"]),
            position=tuple(map(float, row["position"])),
            parent_canonical_id=None if row.get("parent_canonical_id") is None else str(row["parent_canonical_id"]),
            support_surface_ids=tuple(map(str, row.get("support_surface_ids") or ())),
            source_proposal_id=str(row.get("source_proposal_id") or ""),
        )
        for row in raw.get("joints") or ()
    )
    value = QualifiedSkeletonIRV2(
        joints=joints,
        deform_root_ids=tuple(map(str, raw.get("deform_root_ids") or ())),
        assembly_root_binding=dict(raw.get("assembly_root_binding") or {}),
        qualification_report=dict(raw.get("qualification_report") or {}),
        skeleton_lineage_hash=str(raw["skeleton_lineage_hash"]),
        schema_version=str(raw.get("schema_version") or "RealSaS.QualifiedSkeletonIR.v2"),
    )
    if value.skeleton_lineage_hash != EXPECTED_SKELETON_LINEAGE or len(value.joints) != 22:
        raise ValueError("MAGE_FIT1_SKELETON_AUTHORITY_DRIFT")
    return value


def _load_skin(path: Path) -> QualifiedSkinIR:
    raw = _load_json(path)
    rows = tuple(
        QualifiedSkinRow(
            surface_id=str(row["surface_id"]),
            influences=tuple((str(jid), float(weight)) for jid, weight in row.get("influences") or ()),
            simplex_residual_before=float(row.get("simplex_residual_before", 0.0)),
            correction_l1=float(row.get("correction_l1", 0.0)),
        )
        for row in raw.get("rows") or ()
    )
    value = QualifiedSkinIR(
        rows=rows,
        surface_binding_hash=str(raw["surface_binding_hash"]),
        skeleton_binding_hash=str(raw["skeleton_binding_hash"]),
        qualification_report=dict(raw.get("qualification_report") or {}),
        skin_lineage_hash=str(raw["skin_lineage_hash"]),
        schema_version=str(raw.get("schema_version") or "RealSaS.QualifiedSkinIR.v1"),
    )
    if value.skin_lineage_hash != EXPECTED_SKIN_LINEAGE or len(value.rows) != 950:
        raise ValueError("MAGE_FIT1_SKIN_AUTHORITY_DRIFT")
    if value.surface_binding_hash != EXPECTED_SURFACE_LINEAGE or value.skeleton_binding_hash != EXPECTED_SKELETON_LINEAGE:
        raise ValueError("MAGE_FIT1_SKIN_BINDING_DRIFT")
    return value


def _observation_authority(input_root: Path):
    obs_root = input_root / "AUDIT_AUTHORITIES" / "OBSERVATIONS"
    manifest = _load_json(obs_root / "SOURCE_OBSERVATION_MANIFEST.json")
    views = sorted(manifest.get("views") or (), key=lambda row: int(row["view_index"]))
    if tuple(int(row["view_index"]) for row in views) != EXPECTED_VIEWS:
        raise ValueError("MAGE_FIT1_SOURCE_OBSERVATION_SET_NOT_EXACT_8")
    return obs_root, views


def _copy_textures(obs_root: Path, views, output_root: Path):
    rows = []
    for row in views:
        view = int(row["view_index"])
        src = obs_root / str(row["image_name"])
        expected = str(row["image_sha256"])
        if not src.is_file() or _sha256_file(src) != expected:
            raise ValueError(f"MAGE_FIT1_SOURCE_IMAGE_AUTHORITY_DRIFT:V{view}")
        rel = f"native_texture_source/textures/view_{view}.png"
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if _sha256_file(dst) != expected:
            raise ValueError(f"MAGE_FIT1_TEXTURE_COPY_DRIFT:V{view}")
        rows.append(
            {
                "view_index": view,
                "image_relpath": rel,
                "image_sha256": expected,
                "camera_sha256": str(row["camera_sha256"]),
                "atlas_payload_hash": content_sha256({"image_sha256": expected, "image_relpath": rel}),
            }
        )
    return tuple(rows)


def _build_real_visuals(mechanical, texture_rows):
    texture_by_view = {int(row["view_index"]): row for row in texture_rows}
    observation_hash_by_view = {
        view: content_sha256(
            {
                "asset_id": "MAGE_FIT1_EXACT_8VIEW",
                "view_index": view,
                "image_relpath": texture_by_view[view]["image_relpath"],
                "image_sha256": texture_by_view[view]["image_sha256"],
                "camera_sha256": texture_by_view[view]["camera_sha256"],
                "surface_lineage_hash": mechanical.surface.geometry_lineage_hash,
            }
        )
        for view in EXPECTED_VIEWS
    }
    directions = []
    stats = []
    for view in EXPECTED_VIEWS:
        tex = texture_by_view[view]
        camera_hash = tex["camera_sha256"]
        candidate = build_mwb2_candidate(mechanical.surface, view_index=view, camera_binding_hash=camera_hash)
        mesh = qualify_mwb2_mesh(mechanical.surface, candidate)
        mesh_skin = bind_mwb2_mesh_skin(mechanical.surface, mechanical.skeleton, mechanical.skin, mesh)
        appearance = build_observed_appearance_binding(
            surface=mechanical.surface,
            mesh=mesh,
            target_view_index=view,
            camera_binding_hash=camera_hash,
            observation_hash_by_view=observation_hash_by_view,
            atlas_payload_hash=tex["atlas_payload_hash"],
        )
        component = build_renderable_component(
            component_id="BODY",
            view_index=view,
            mesh=mesh,
            mesh_skin=mesh_skin,
            appearance=appearance,
            setup_order=0,
            coverage_classification=mesh.support_coverage_classification,
            metadata={
                "real_mage_fit1": True,
                "full_silhouette_substrate": False,
                "visual_completion_used": False,
                "unknown_regions_remain_empty": True,
                "source_mesh_used": False,
            },
        )
        directions.append(
            build_directional_renderable(
                view_index=view,
                camera_binding_hash=camera_hash,
                components=(component,),
                metadata={
                    "real_mage_fit1": True,
                    "camera_authority": "EXACT_SOURCE_CAMERA_SHA256",
                    "visual_completion_used": False,
                },
            )
        )
        rr = dict(candidate.residual_report or {})
        stats.append(
            {
                "view_index": view,
                "candidate_vertex_count": int(rr.get("vertex_count", len(mesh.vertices))),
                "qualified_vertex_count": len(mesh.vertices),
                "qualified_face_count": len(mesh.faces),
                "qualified_edge_count": len(mesh.edges),
                "relation_clique_face_count": int(rr.get("relation_clique_face_count", 0)),
                "overlap_rejected_face_count": int(rr.get("overlap_rejected_face_count", 0)),
                "coverage_classification": mesh.support_coverage_classification,
                "visual_completion_used": False,
            }
        )
    visuals = build_directional_renderable_set(
        tuple(directions),
        metadata={
            "real_mage_fit1": True,
            "source_surface_lineage_hash": mechanical.surface.geometry_lineage_hash,
            "visual_completion_used": False,
            "unknown_regions_remain_empty": True,
        },
    )
    return visuals, tuple(stats)


def _export_static_bundle(root: Path, *, product, directional_binding, texture_rows, source_authority: dict, view_stats):
    files = []
    for value in (
        product,
        product.mechanical_state,
        product.directional_renderables,
        directional_binding,
        product.capability_contract,
        product.motion_state,
    ):
        files.append(_write_typed(root, value))
    for direction in product.directional_renderables.directions:
        files.append(_write_typed(root, direction))
        for component in direction.components:
            files.append(_write_typed(root, component))
            files.append(_write_typed(root, component.appearance))
    manifest = {
        "schema": "RealSaS.StaticInspectionBundle.v1",
        "status": "STATIC_INSPECTION_READY__PRODUCT_PROOF_UNAVAILABLE",
        "source_product_state_hash": product.product_state_hash,
        "direction_count": 8,
        "component_count": 8,
        "directional_binding_file": "renderables/directional_joint_view_binding_set_ir.json",
        "directional_binding_set_hash": directional_binding.binding_set_hash,
        "product_proof_file": None,
        "product_proof_status": "UNAVAILABLE",
        "product_pass_claimed": False,
        "runtime_preview_available": False,
        "motion_state_is_unproven_candidate": True,
        "visual_completion_used": False,
        "unknown_regions_remain_empty": True,
        "representation_class": product.representation_class,
        "source_authority": source_authority,
        "texture_rows": list(texture_rows),
        "view_stats": list(view_stats),
        "files": sorted(set(files + [row["image_relpath"] for row in texture_rows])),
    }
    (root / "static_inspection_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_mage_fit1_real_static_v1(input_root, output_root):
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = input_root / "EXACT_PRODUCT_ARTIFACTS"
    report_path = artifact_root / "MAGE_V5_EXACT_SKIN_REEMIT_V3_REPORT.json"
    report = _load_json(report_path)
    if report.get("status") != "PASS__EXACT_FP32_V5_PRODUCT_SKIN_REEMITTED":
        raise ValueError("MAGE_FIT1_EXACT_V5_REEMIT_REPORT_NOT_PASS")

    surface = _load_surface(artifact_root / "MAGE_FIT1_EXACT_RIGGING_SURFACE_IR.json")
    skeleton = _load_skeleton(artifact_root / "MAGE_FIT1_EXACT_QUALIFIED_SKELETON_IR.json")
    skin = _load_skin(artifact_root / "MAGE_V5_EXACT_QUALIFIED_SKIN_IR.json")
    mechanical = build_mechanical_state(surface, skeleton, skin)

    obs_root, views = _observation_authority(input_root)
    texture_rows = _copy_textures(obs_root, views, output_root)
    visuals, view_stats = _build_real_visuals(mechanical, texture_rows)

    motion = build_deterministic_preset_motion(
        mechanical,
        clip_id="mage_fit1_mechanical_probe_v1",
        duration_sec=1.0,
        amplitude_deg=4.0,
    )
    policy_hash = content_sha256(
        {
            "profile": "MAGE_FIT1_REAL_STATIC_INSPECTION_V1",
            "product_pass_claimed": False,
            "visual_completion_used": False,
        }
    )
    capability = make_single_family_e2e_capability_contract(
        base_lbs_hash=content_sha256({"implementation": "verified_lbs_v1"}),
        mesh_hash=content_sha256({"mesh_lineages": tuple(d.components[0].mesh.mesh_lineage_hash for d in visuals.directions)}),
        skinning_hash=skin.skin_lineage_hash,
        visual_hash=visuals.directional_visual_state_hash,
        preset_motion_hash=motion.motion_state_hash,
        runtime_hash=content_sha256({"runtime": "CURRENT_V4_PROOF_GATED_RUNTIME"}),
        policy_hash=policy_hash,
    )
    product = assemble_product_v3(
        mechanical,
        visuals,
        capability,
        motion,
        editable_metadata={
            "real_mage_fit1": True,
            "rigging_core_status": "CLOSED",
            "product_status": "UNQUALIFIED",
            "static_inspection_only": True,
            "source_exact_v5_reemit_report_sha256": _sha256_file(report_path),
            "visual_completion_used": False,
            "unknown_regions_remain_empty": True,
        },
        runtime_policy={
            "backend": "CURRENT_V4_PROOF_GATED_RUNTIME",
            "static_inspection_without_proof": True,
            "runtime_preview_requires_pass_proof": True,
        },
    )
    directional_binding = qualify_directional_joint_view_binding(product)

    source_authority = {
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "exact_v5_reemit_report_sha256": _sha256_file(report_path),
        "teacher_predictor_input_used": False,
        "source_mesh_used": False,
    }
    manifest = _export_static_bundle(
        output_root,
        product=product,
        directional_binding=directional_binding,
        texture_rows=texture_rows,
        source_authority=source_authority,
        view_stats=view_stats,
    )
    return {
        "status": "PASS_MAGE_FIT1_REAL_STATIC_CHAIN_V1",
        "output_root": str(output_root),
        "product_state_hash": product.product_state_hash,
        "mechanical_state_hash": product.mechanical_state_hash,
        "directional_visual_state_hash": product.directional_visual_state_hash,
        "directional_binding_hash": directional_binding.binding_set_hash,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "view_stats": view_stats,
        "product_proof_status": "UNAVAILABLE",
        "product_pass_claimed": False,
        "runtime_preview_available": False,
        "visual_completion_used": False,
        "unknown_regions_remain_empty": True,
        "manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = run_mage_fit1_real_static_v1(args.input_root, args.output_root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
