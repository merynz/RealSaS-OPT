from __future__ import annotations

"""Assemble the bounded Mage current product state from sealed upstream artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.appearance_atlas import rebind_local_observed_appearance_to_atlas_panel
from compiler.realsas_compiler_core.component_attachment import (
    QualifiedComponentAssemblyIR,
    QualifiedComponentAttachmentIR,
    bind_component_assembly_to_directional_renderable_set,
    validate_component_assembly,
)
from compiler.realsas_compiler_core.continuity_underlay import (
    build_continuity_underlay_set,
    qualify_continuity_underlay,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.rigid_attachment_skin import qualify_rigid_attachment_mesh_skin
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_core.motion_rotation_only import build_mage_rotation_only_qualified_motion
from compiler.realsas_compiler_core.product_external_render import (
    assemble_product_v3_with_external_render_support,
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedMeshVertex,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import (
    build_capability_contract,
    build_mechanical_state,
    validate_appearance_binding,
)
from compiler.realsas_compiler_core.v4_types import AppearanceBindingIR, AppearanceCornerBinding, CapabilityRequirement

import experiments.mage_demo_fit1_v5_p1.materialize_p1q_current_authority_v1 as p1q
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2
import experiments.single_family_e2e_v1.run_mage_fit1_real_static_v1 as fit1

SCHEMA = "RealSaS.MageCurrentProductState.v1"
RIGID_IDS = ("BOOK_FOREGROUND", "STAFF_FOREGROUND", "HAT_FOREGROUND", "CAPE_FOREGROUND")
SETUP_ORDER = {"BODY_UNDERLAY": 0, "CAPE_FOREGROUND": 1, "BOOK_FOREGROUND": 2, "STAFF_FOREGROUND": 3, "HAT_FOREGROUND": 4}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_appearance(path: Path) -> AppearanceBindingIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    corners = tuple(AppearanceCornerBinding(
        int(row["face_index"]), int(row["corner_index"]),
        tuple(map(float, row["material_uv"])), int(row["donor_view_index"]),
        tuple(map(float, row["donor_raster_xy"])), str(row["source_observation_hash"]),
        str(row["authority_class"]), str(row.get("completion_id", "")), float(row.get("confidence", 1.0)),
    ) for row in data["corner_bindings"])
    value = AppearanceBindingIR(
        int(data["target_view_index"]), str(data["mesh_binding_hash"]), str(data["camera_binding_hash"]),
        corners, str(data["appearance_lineage_hash"]), str(data.get("atlas_payload_hash", "")),
        str(data.get("schema_version", "RealSaS.AppearanceBindingIR.v1")), dict(data.get("metadata") or {}),
    )
    validate_appearance_binding(value)
    return value


def _load_external_mesh(path: Path, expected_hash: str) -> QualifiedEditableMeshIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    vertices = tuple(QualifiedMeshVertex(
        str(row["canonical_mesh_vertex_id"]), tuple(map(float, row["P"])),
        SurfaceSupportBinding(
            str(row["support_binding"]["mode"]),
            tuple((str(sid), float(weight)) for sid, weight in row["support_binding"]["coefficients"]),
            dict(row["support_binding"].get("metadata") or {}),
            str(row["support_binding"].get("schema_version", "RealSaS.SurfaceSupportBinding.v1")),
        ),
        str(row.get("source_candidate_vertex_id", "")), dict(row.get("metadata") or {}),
    ) for row in data["vertices"])
    mesh = QualifiedEditableMeshIR(
        vertices,
        tuple(tuple(map(str, face)) for face in data["faces"]),
        tuple(tuple(map(str, edge)) for edge in data["edges"]),
        str(data["surface_binding_hash"]), int(data["view_index"]), str(data["camera_binding_hash"]),
        dict(data.get("qualification_report") or {}), str(data["mesh_lineage_hash"]),
        tuple(data.get("boundary_constraints") or ()), str(data.get("support_coverage_classification", "")),
        str(data.get("schema_version", "RealSaS.QualifiedEditableMeshIR.v1")), dict(data.get("metadata") or {}),
    )
    if mesh.mesh_lineage_hash != expected_hash or mesh.mesh_lineage_hash != mesh_lineage_hash(mesh):
        raise RuntimeError("PRODUCT_STATE_EXTERNAL_MESH_LINEAGE_DRIFT")
    return mesh


def _load_assembly(path: Path) -> QualifiedComponentAssemblyIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    components = tuple(QualifiedComponentAttachmentIR(
        component_id=str(row["component_id"]),
        source_provenance_refs=tuple(map(str, row["source_provenance_refs"])),
        product_view_membership=tuple(map(int, row["product_view_membership"])),
        directional_render_membership=tuple((int(v), str(cid)) for v, cid in row["directional_render_membership"]),
        mechanical_class=str(row["mechanical_class"]),
        canonical_parent_joint_id=str(row["canonical_parent_joint_id"]), socket_id=str(row["socket_id"]),
        bind_state_authority_hash=str(row["bind_state_authority_hash"]),
        geometry_membership_refs=tuple(map(str, row["geometry_membership_refs"])),
        geometry_lineage_hash=str(row["geometry_lineage_hash"]), skin_deformer_lineage_hash=str(row["skin_deformer_lineage_hash"]),
        detachability_class=str(row["detachability_class"]), visible_required=bool(row["visible_required"]),
        qualification_report=dict(row["qualification_report"]), component_lineage_hash=str(row["component_lineage_hash"]),
        schema_version=str(row.get("schema_version", "RealSaS.QualifiedComponentAttachmentIR.v1")), metadata=dict(row.get("metadata") or {}),
    ) for row in data["components"])
    return QualifiedComponentAssemblyIR(
        components, tuple(map(str, data["required_visible_component_ids"])), dict(data["qualification_report"]),
        str(data["component_assembly_hash"]), str(data.get("schema_version", "RealSaS.QualifiedComponentAssemblyIR.v1")),
        dict(data.get("metadata") or {}),
    )


def build_state(args, *, persist: bool = True):
    p1q_dir, fg_dir, assembly_dir = Path(args.p1q_dir), Path(args.foreground_dir), Path(args.assembly_dir)
    output_dir = Path(args.output_dir)
    if persist:
        output_dir.mkdir(parents=True, exist_ok=True)

    p1q_manifest_path = p1q_dir / "P1Q_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    p1q_derivation_path = p1q_dir / "P1Q_CURRENT_DIRECT_BINDING_DERIVATION_MANIFEST.json"
    fg_manifest_path = fg_dir / "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json"
    assembly_manifest_path = assembly_dir / "COMPONENT_ASSEMBLY_MANIFEST.json"
    assembly_path = assembly_dir / "QUALIFIED_COMPONENT_ASSEMBLY.json"
    hashes = {
        "p1q_manifest": _sha(p1q_manifest_path),
        "p1q_derivation": _sha(p1q_derivation_path),
        "foreground_manifest": _sha(fg_manifest_path),
        "assembly_manifest": _sha(assembly_manifest_path),
    }
    for key, arg_name in (
        ("p1q_manifest", "expected_p1q_manifest"),
        ("foreground_manifest", "expected_foreground_manifest"),
        ("assembly_manifest", "expected_assembly_manifest"),
    ):
        expected = str(getattr(args, arg_name, "") or "")
        if expected and hashes[key] != expected:
            raise RuntimeError(f"PRODUCT_STATE_{key.upper()}_SHA_DRIFT")

    current_surface, _tensor, replay = ceiling_v2._preflight_surface(args)
    if not replay["gsa_lineage_exact_match"]:
        raise RuntimeError("PRODUCT_STATE_CURRENT_GSA_DRIFT")
    fit_surface = fit1._load_surface(Path(args.fit1_surface))
    skeleton = fit1._load_skeleton(Path(args.skeleton))
    fit_skin = fit1._load_skin(Path(args.fit1_skin))
    mechanical = build_mechanical_state(fit_surface, skeleton, fit_skin)

    assembly = _load_assembly(assembly_path)
    validate_component_assembly(assembly, skeleton)
    assembly_by_id = {component.component_id: component for component in assembly.components}
    if set(("BODY_UNDERLAY", *RIGID_IDS)) - set(assembly_by_id):
        raise RuntimeError("PRODUCT_STATE_ASSEMBLY_COMPONENT_SET_INCOMPLETE")

    p1q_manifest = json.loads(p1q_manifest_path.read_text(encoding="utf-8"))
    fg_manifest = json.loads(fg_manifest_path.read_text(encoding="utf-8"))
    p1q_rows = {int(row["view"]): row for row in p1q_manifest["views"]}
    fg_rows = {int(row["view"]): row for row in fg_manifest["views"]}
    if set(p1q_rows) != set(range(8)) or set(fg_rows) != set(range(8)):
        raise RuntimeError("PRODUCT_STATE_REQUIRES_8_VIEWS")

    directions = []
    underlay_components = []
    persisted = []
    for view in range(8):
        prow, frow = p1q_rows[view], fg_rows[view]
        mesh = v5base._load_mesh(p1q_dir / prow["files"]["mesh"], current_surface, str(prow["mesh_lineage_hash"]))
        mesh_skin = p1q._load_mesh_skin(p1q_dir / prow["files"]["skin"], mesh.mesh_lineage_hash)
        source_appearance = _load_appearance(p1q_dir / prow["files"]["appearance"])
        rebound_appearance = rebind_local_observed_appearance_to_atlas_panel(
            source_appearance,
            source_width=int(frow["width"]), source_height=int(frow["height"]),
            atlas_width=int(frow["atlas_width"]), atlas_height=int(frow["atlas_height"]),
            panel_x_offset=0, atlas_payload_hash=str(frow["atlas_payload_hash"]),
        )
        underlay = build_external_renderable_component(
            component_id="BODY_UNDERLAY", view_index=view, mesh=mesh, mesh_skin=mesh_skin,
            mechanical=mechanical, appearance=rebound_appearance, setup_order=SETUP_ORDER["BODY_UNDERLAY"],
            coverage_classification=mesh.support_coverage_classification,
            materialization_manifest_sha256=hashes["p1q_manifest"], direct_binding_manifest_sha256=hashes["p1q_derivation"],
            metadata={
                "product_role": "CONTINUITY_DEFORMABLE_UNDERLAY",
                "source_p1q_appearance_lineage_hash": source_appearance.appearance_lineage_hash,
                "runtime_atlas_rebind_only": True,
                "component_assembly_hash": assembly.component_assembly_hash,
            },
        )
        components = [underlay]
        underlay_components.append(underlay)
        fg_by_id = {str(row["runtime_component_id"]): row for row in frow["components"]}
        for runtime_id in RIGID_IDS:
            if runtime_id not in fg_by_id:
                raise RuntimeError(f"PRODUCT_STATE_FOREGROUND_COMPONENT_MISSING_V{view}:{runtime_id}")
            row = fg_by_id[runtime_id]
            sprite_mesh = _load_external_mesh(fg_dir / row["mesh_file"], str(row["mesh_lineage_hash"]))
            sprite_appearance = _load_appearance(fg_dir / row["appearance_file"])
            attachment = assembly_by_id[runtime_id]
            carry_skin = qualify_rigid_attachment_mesh_skin(
                sprite_mesh, mechanical,
                parent_joint_id=attachment.canonical_parent_joint_id,
                component_assembly_hash=assembly.component_assembly_hash,
                component_lineage_hash=attachment.component_lineage_hash,
                bind_state_authority_hash=attachment.bind_state_authority_hash,
            )
            component = build_external_renderable_component(
                component_id=runtime_id, view_index=view, mesh=sprite_mesh, mesh_skin=carry_skin,
                mechanical=mechanical, appearance=sprite_appearance, setup_order=SETUP_ORDER[runtime_id],
                coverage_classification="RIGID_FOREGROUND_SPRITE",
                materialization_manifest_sha256=hashes["foreground_manifest"],
                direct_binding_manifest_sha256=hashes["assembly_manifest"],
                metadata={
                    "product_role": "RIGID_FOREGROUND_DRAWABLE",
                    "component_assembly_hash": assembly.component_assembly_hash,
                    "component_lineage_hash": attachment.component_lineage_hash,
                    "bind_state_authority_hash": attachment.bind_state_authority_hash,
                    "canonical_parent_joint_id": attachment.canonical_parent_joint_id,
                    "runtime_carry_skin_lineage_hash": carry_skin.mesh_skin_lineage_hash,
                },
            )
            components.append(component)
            if persist:
                skin_name = f"V{view}_{runtime_id}_RUNTIME_CARRY_SKIN.json"
                _write_json(output_dir / skin_name, carry_skin.to_dict())
                persisted.append({"path": skin_name, "sha256": _sha(output_dir / skin_name)})
        direction = build_external_directional_renderable(
            view_index=view, camera_binding_hash=mesh.camera_binding_hash,
            components=tuple(components), mechanical=mechanical,
            metadata={"runtime_atlas_payload_hash": frow["atlas_payload_hash"], "component_count": len(components)},
        )
        directions.append(direction)

    render_set = build_external_directional_renderable_set(tuple(directions), mechanical, metadata={
        "runtime_texture_layout": "ONE_ATLAS_PER_VIEW_PANEL0_UNDERLAY_PANEL1_4_RIGID_FOREGROUND",
        "foreground_manifest_sha256": hashes["foreground_manifest"],
    })
    render_set = bind_component_assembly_to_directional_renderable_set(render_set, assembly, skeleton)

    underlays = tuple(qualify_continuity_underlay(
        view_index=view, substrate_component=underlay_components[view], mechanical=mechanical,
        component_assembly=assembly, partition_authority_sha256=str(fg_manifest["owner_manifest_sha256"]),
        metadata={
            "runtime_atlas_rebind": True,
            "source_appearance_lineage_hash": underlay_components[view].metadata["source_p1q_appearance_lineage_hash"],
            "new_pixels_generated": False,
            "topology_mutated": False,
            "weights_mutated": False,
            "scientific_mechanical_surface_relabelled": False,
        },
    ) for view in range(8))
    underlay_set = build_continuity_underlay_set(underlays, component_assembly=assembly, metadata={
        "partition_authority_sha256": fg_manifest["owner_manifest_sha256"],
        "runtime_atlas_rebind": True,
    })

    base_motion = build_mage_rotation_only_qualified_motion(mechanical)
    motion = compile_motion_quality(base_motion, mechanical)
    policy_hash = content_sha256({
        "schema": "RealSaS.MageBoundedDemoCapabilityPolicy.v2",
        "fit2_training_deferred": True,
        "unseen_generalization_claimed": False,
        "rotation_only_motion_scope": True,
        "historical_motion_quality_recovery": True,
        "motion_quality_scope": (
            "PHASES", "NAMED_TIMING_CURVES", "CUBIC_HERMITE_ROTATION",
            "MINIMUM_JERK_TIMING_BLEND", "EXACT_LOOP_REPAIR",
        ),
        "contact_lock_claimed": False,
        "xpbd_secondary_claimed": False,
        "corrective_deformation_claimed": False,
        "runtime_requires_qualification_owned_bakes": True,
    })
    runtime_impl_hash = content_sha256({
        "runtime": "RealSaS.NativeRuntimeV2+C++17",
        "draw_order": "QUALIFICATION_OWNED_PER_FRAME_PER_VIEW",
        "solver_replay": False,
    })
    requirements = (
        CapabilityRequirement("MECHANICAL_STRUCTURE", "REQUIRED", mechanical.mechanical_state_hash, policy_hash, ("MECHANICAL_STRUCTURE",)),
        CapabilityRequirement("MESH_QUALITY", "REQUIRED", render_set.directional_visual_state_hash, policy_hash, ("MESH_QUALITY",)),
        CapabilityRequirement("VISUAL_8_DIRECTION", "REQUIRED", render_set.directional_visual_state_hash, policy_hash, ("DIRECTIONAL_VISUAL",)),
        CapabilityRequirement("PRESET_MOTION", "REQUIRED", motion.motion_state_hash, policy_hash, ("MOTION",)),
        CapabilityRequirement("RUNTIME_BACKEND", "REQUIRED", runtime_impl_hash, policy_hash, ("RUNTIME_CONSUMPTION",)),
    )
    capability = build_capability_contract("MAGE_FIT1_BOUNDED_DEMO_V2_HISTORICAL_MOTION_QUALITY", requirements, metadata={
        "generalization_claim": False,
        "fit2_training_executed": False,
        "component_assembly_hash": assembly.component_assembly_hash,
        "continuity_underlay_set_hash": underlay_set.qualification_hash,
        "historical_motion_engine_recovered": True,
        "contact_lock_claimed": False,
        "xpbd_secondary_claimed": False,
        "corrective_deformation_claimed": False,
    })
    product = assemble_product_v3_with_external_render_support(
        mechanical, render_set, capability, motion,
        editable_metadata={
            "bounded_demo": True,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "historical_motion_engine_recovered": True,
        },
        runtime_policy={
            "runtime_texture_layout": "ONE_ATLAS_PER_VIEW",
            "qualification_owned_motion_bake_required": True,
            "export_solver_replay_forbidden": True,
        },
    )

    if persist:
        _write_json(output_dir / "FINAL_DIRECTIONAL_RENDERABLE_SET.json", render_set.to_dict())
        _write_json(output_dir / "CONTINUITY_UNDERLAY_SET.json", underlay_set.to_dict())
        _write_json(output_dir / "ROTATION_ONLY_BASE_MOTION_STATE.json", base_motion.to_dict())
        _write_json(output_dir / "HISTORICAL_QUALITY_MOTION_STATE.json", motion.to_dict())
        _write_json(output_dir / "CAPABILITY_CONTRACT.json", capability.to_dict())
        _write_json(output_dir / "CANONICAL_PUPPET_GRAPH_V3.json", product.to_dict())
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_PRODUCT_STATE_MATERIALIZED",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "base_motion_state_hash": base_motion.motion_state_hash,
            "historical_motion_engine_recovered": True,
            "historical_motion_recovery_scope": motion.metadata.get("historical_recovery_scope"),
            "contact_lock_claimed": False,
            "xpbd_secondary_claimed": False,
            "corrective_deformation_claimed": False,
            "capability_contract_hash": capability.capability_contract_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "source_hashes": hashes,
            "runtime_atlas_views": [{
                "view": int(row["view"]), "atlas_file": row["atlas_file"], "atlas_relpath": row["atlas_relpath"],
                "atlas_sha256": row["atlas_sha256"], "atlas_payload_hash": row["atlas_payload_hash"],
                "width": row["atlas_width"], "height": row["atlas_height"],
            } for row in fg_manifest["views"]],
            "persisted_runtime_carry_skins": persisted,
            "product_pass_claimed": False,
        }
        manifest_path = output_dir / "PRODUCT_STATE_MANIFEST.json"
        _write_json(manifest_path, manifest)
        print("PRODUCT_STATE=" + json.dumps({"product_state_hash": product.product_state_hash, "manifest_sha256": _sha(manifest_path)}, sort_keys=True), flush=True)
    return {
        "product": product,
        "mechanical": mechanical,
        "render_set": render_set,
        "motion": motion,
        "base_motion": base_motion,
        "capability": capability,
        "assembly": assembly,
        "underlay_set": underlay_set,
        "foreground_manifest": fg_manifest,
        "source_hashes": hashes,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--p1q-dir", required=True)
    p.add_argument("--foreground-dir", required=True)
    p.add_argument("--assembly-dir", required=True)
    p.add_argument("--fit1-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit1-skin", required=True)
    p.add_argument("--expected-p1q-manifest", default="")
    p.add_argument("--expected-foreground-manifest", default="")
    p.add_argument("--expected-assembly-manifest", default="")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    build_state(parse_args(), persist=True)
