from __future__ import annotations

"""Assemble the bounded Mage product state on corrected FIT2 S/G/W.

This module deliberately mirrors the proven product/rest-render path while replacing
the stale FIT1 mechanical authority with the fresh corrected-lineage FIT2 surface,
Geppetto skeleton, Arachne skin, and FIT2-derived P1Q mesh-skin artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.appearance_atlas import rebind_local_observed_appearance_to_atlas_panel
from compiler.realsas_compiler_core.component_attachment import (
    bind_component_assembly_to_directional_renderable_set,
    validate_component_assembly,
)
from compiler.realsas_compiler_core.continuity_underlay import (
    build_continuity_underlay_set,
    qualify_continuity_underlay,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.rigid_attachment_skin import qualify_rigid_attachment_mesh_skin
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_core.motion_rotation_only import build_mage_rotation_only_qualified_motion
from compiler.realsas_compiler_core.product_external_render import (
    assemble_product_v3_with_external_render_support,
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.v4 import build_capability_contract
from compiler.realsas_compiler_core.v4_types import CapabilityRequirement

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_p1q_current_authority_v1 as legacy_p1q
import experiments.mage_demo_fit1_v5_p1.materialize_product_state_v1 as legacy_state
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base


SCHEMA = "RealSaS.MageFIT2.CurrentProductState.v1"
PROFILE = "MAGE_FIT2_BOUNDED_DEMO_V1"
RIGID_IDS = legacy_state.RIGID_IDS
SETUP_ORDER = legacy_state.SETUP_ORDER


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_state(args, *, persist: bool = True):
    p1q_dir = Path(args.p1q_dir)
    fg_dir = Path(args.foreground_dir)
    assembly_dir = Path(args.assembly_dir)
    output_dir = Path(args.output_dir)
    if persist:
        output_dir.mkdir(parents=True, exist_ok=True)

    p1q_manifest_path = p1q_dir / "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    p1q_derivation_path = p1q_dir / "P1Q_FIT2_CURRENT_BINDING_DERIVATION_MANIFEST.json"
    fg_manifest_path = fg_dir / "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json"
    assembly_manifest_path = assembly_dir / "FIT2_COMPONENT_ASSEMBLY_MANIFEST.json"
    assembly_path = assembly_dir / "FIT2_QUALIFIED_COMPONENT_ASSEMBLY.json"
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
            raise RuntimeError(f"FIT2_PRODUCT_STATE_{key.upper()}_SHA_DRIFT")

    surface, skeleton, skin, mechanical = fit2io.build_exact_mechanical(
        Path(args.fit2_surface), Path(args.skeleton), Path(args.fit2_skin)
    )

    assembly = legacy_state._load_assembly(assembly_path)
    validate_component_assembly(assembly, skeleton)
    assembly_by_id = {component.component_id: component for component in assembly.components}
    if set(("BODY_UNDERLAY", *RIGID_IDS)) - set(assembly_by_id):
        raise RuntimeError("FIT2_PRODUCT_STATE_ASSEMBLY_COMPONENT_SET_INCOMPLETE")

    p1q_manifest = json.loads(p1q_manifest_path.read_text(encoding="utf-8"))
    fg_manifest = json.loads(fg_manifest_path.read_text(encoding="utf-8"))
    if p1q_manifest.get("status") != "PASS__FIT2_P1Q_CURRENT_AUTHORITY_V0_V7_FROZEN_FACE_POLICY":
        raise RuntimeError("FIT2_PRODUCT_STATE_P1Q_STATUS_INVALID")
    if p1q_manifest.get("current_skin_lineage_hash") != skin.skin_lineage_hash:
        raise RuntimeError("FIT2_PRODUCT_STATE_SKIN_LINEAGE_DRIFT")
    assembly_manifest = json.loads(assembly_manifest_path.read_text(encoding="utf-8"))
    if assembly_manifest.get("status") != "PASS__FIT2_TYPED_COMPONENT_ASSEMBLY_BODY_PLUS_4_RIGID_FOREGROUND":
        raise RuntimeError("FIT2_PRODUCT_STATE_ASSEMBLY_STATUS_INVALID")
    if assembly_manifest.get("skin_lineage_hash") != skin.skin_lineage_hash:
        raise RuntimeError("FIT2_PRODUCT_STATE_ASSEMBLY_SKIN_DRIFT")

    p1q_rows = {int(row["view"]): row for row in p1q_manifest["views"]}
    fg_rows = {int(row["view"]): row for row in fg_manifest["views"]}
    if set(p1q_rows) != set(range(8)) or set(fg_rows) != set(range(8)):
        raise RuntimeError("FIT2_PRODUCT_STATE_REQUIRES_8_VIEWS")

    directions = []
    underlay_components = []
    persisted = []
    for view in range(8):
        prow, frow = p1q_rows[view], fg_rows[view]
        mesh = v5base._load_mesh(p1q_dir / prow["files"]["mesh"], surface, str(prow["mesh_lineage_hash"]))
        mesh_skin = legacy_p1q._load_mesh_skin(p1q_dir / prow["files"]["skin"], mesh.mesh_lineage_hash)
        if mesh_skin.skin_binding_hash != skin.skin_lineage_hash:
            raise RuntimeError(f"FIT2_PRODUCT_STATE_BODY_SKIN_DRIFT_V{view}")
        if mesh_skin.surface_binding_hash != surface.geometry_lineage_hash:
            raise RuntimeError(f"FIT2_PRODUCT_STATE_BODY_SURFACE_DRIFT_V{view}")
        if mesh_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
            raise RuntimeError(f"FIT2_PRODUCT_STATE_BODY_SKELETON_DRIFT_V{view}")

        source_appearance = legacy_state._load_appearance(p1q_dir / prow["files"]["appearance"])
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
                "fit2_skin_lineage_hash": skin.skin_lineage_hash,
            },
        )
        components = [underlay]
        underlay_components.append(underlay)
        fg_by_id = {str(row["runtime_component_id"]): row for row in frow["components"]}
        for runtime_id in RIGID_IDS:
            if runtime_id not in fg_by_id:
                raise RuntimeError(f"FIT2_PRODUCT_STATE_FOREGROUND_COMPONENT_MISSING_V{view}:{runtime_id}")
            row = fg_by_id[runtime_id]
            sprite_mesh = legacy_state._load_external_mesh(fg_dir / row["mesh_file"], str(row["mesh_lineage_hash"]))
            sprite_appearance = legacy_state._load_appearance(fg_dir / row["appearance_file"])
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
                    "fit2_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
                },
            )
            components.append(component)
            if persist:
                skin_name = f"V{view}_{runtime_id}_FIT2_RUNTIME_CARRY_SKIN.json"
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
        "fit2_skin_lineage_hash": skin.skin_lineage_hash,
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
            "fit2_skin_lineage_hash": skin.skin_lineage_hash,
        },
    ) for view in range(8))
    underlay_set = build_continuity_underlay_set(underlays, component_assembly=assembly, metadata={
        "partition_authority_sha256": fg_manifest["owner_manifest_sha256"],
        "runtime_atlas_rebind": True,
        "fit2_skin_lineage_hash": skin.skin_lineage_hash,
    })

    base_motion = build_mage_rotation_only_qualified_motion(mechanical)
    motion = compile_motion_quality(base_motion, mechanical)
    policy_hash = content_sha256({
        "schema": "RealSaS.MageFIT2BoundedDemoCapabilityPolicy.v1",
        "unseen_generalization_claimed": False,
        "rotation_only_motion_scope": True,
        "fresh_fit2_mechanics": True,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
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
    capability = build_capability_contract(PROFILE, requirements, metadata={
        "generalization_claim": False,
        "fresh_fit2_mechanics": True,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "component_assembly_hash": assembly.component_assembly_hash,
        "continuity_underlay_set_hash": underlay_set.qualification_hash,
    })
    product = assemble_product_v3_with_external_render_support(
        mechanical, render_set, capability, motion,
        editable_metadata={
            "bounded_demo": True,
            "fresh_fit2_mechanics": True,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
        },
        runtime_policy={
            "runtime_texture_layout": "ONE_ATLAS_PER_VIEW",
            "qualification_owned_motion_bake_required": True,
            "export_solver_replay_forbidden": True,
            "mechanical_continuity_underlay_required": True,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "contact_lock_unqualified_fail_closed": True,
            "secondary_motion_unqualified_fail_closed": True,
            "corrective_deformation_unqualified_fail_closed": True,
        },
    )

    state = {
        "surface": surface,
        "skeleton": skeleton,
        "skin": skin,
        "mechanical": mechanical,
        "assembly": assembly,
        "render_set": render_set,
        "underlay_set": underlay_set,
        "capability": capability,
        "motion": motion,
        "product": product,
        "source_hashes": hashes,
        "persisted": persisted,
    }
    if persist:
        _write_json(output_dir / "FIT2_DIRECTIONAL_RENDERABLE_SET.json", render_set.to_dict())
        _write_json(output_dir / "FIT2_CONTINUITY_UNDERLAY_SET.json", underlay_set.to_dict())
        _write_json(output_dir / "FIT2_BASE_MOTION_STATE.json", motion.to_dict())
        _write_json(output_dir / "FIT2_CAPABILITY_CONTRACT.json", capability.to_dict())
        _write_json(output_dir / "FIT2_CANONICAL_PUPPET_GRAPH_V3.json", product.to_dict())
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_PRODUCT_STATE_FRESH_FIT2_MECHANICS_BOUND",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "historical_fit1_skin_used": False,
            "product_pass_claimed": False,
        }
        manifest_path = output_dir / "FIT2_PRODUCT_STATE_MANIFEST.json"
        _write_json(manifest_path, manifest)
        print("FIT2_PRODUCT_STATE=" + json.dumps({
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "manifest_sha256": _sha(manifest_path),
        }, sort_keys=True), flush=True)
    return state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--p1q-dir", required=True)
    p.add_argument("--foreground-dir", required=True)
    p.add_argument("--assembly-dir", required=True)
    p.add_argument("--fit2-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--expected-p1q-manifest", default="")
    p.add_argument("--expected-foreground-manifest", default="")
    p.add_argument("--expected-assembly-manifest", default="")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    build_state(parse_args(), persist=True)
