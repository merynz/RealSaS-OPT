from __future__ import annotations

"""Finalize Mage product state with continuity-underlay bound into visual identity."""

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.continuity_underlay import bind_continuity_underlay_set_to_directional_renderables
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_external_render import assemble_product_v3_with_external_render_support
from compiler.realsas_compiler_core.v4 import build_capability_contract
from compiler.realsas_compiler_core.v4_types import CapabilityRequirement

import experiments.mage_demo_fit1_v5_p1.materialize_product_state_v1 as v1

SCHEMA = "RealSaS.MageCurrentProductState.v2.continuity_bound"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_final_state(args, *, persist: bool = True):
    state = v1.build_state(args, persist=persist)
    mechanical = state["mechanical"]
    assembly = state["assembly"]
    underlay_set = state["underlay_set"]
    motion = state["motion"]
    render_set = bind_continuity_underlay_set_to_directional_renderables(
        state["render_set"], underlay_set, assembly, mechanical
    )

    policy_hash = content_sha256({
        "schema": "RealSaS.MageBoundedDemoCapabilityPolicy.v2",
        "fit2_training_deferred": True,
        "unseen_generalization_claimed": False,
        "rotation_only_motion_scope": True,
        "runtime_requires_qualification_owned_bakes": True,
        "continuity_underlay_bound_to_visual_state": True,
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
    capability = build_capability_contract("MAGE_FIT1_BOUNDED_DEMO_V2", requirements, metadata={
        "generalization_claim": False,
        "fit2_training_executed": False,
        "component_assembly_hash": assembly.component_assembly_hash,
        "continuity_underlay_set_hash": underlay_set.qualification_hash,
        "continuity_bound_directional_visual_state_hash": render_set.directional_visual_state_hash,
    })
    product = assemble_product_v3_with_external_render_support(
        mechanical, render_set, capability, motion,
        editable_metadata={
            "bounded_demo": True,
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
        },
    )
    state.update({"product": product, "render_set": render_set, "capability": capability})

    if persist:
        out = Path(args.output_dir)
        _write_json(out / "FINAL_DIRECTIONAL_RENDERABLE_SET.json", render_set.to_dict())
        _write_json(out / "CAPABILITY_CONTRACT.json", capability.to_dict())
        _write_json(out / "CANONICAL_PUPPET_GRAPH_V3.json", product.to_dict())
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_PRODUCT_STATE_CONTINUITY_BOUND",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "capability_contract_hash": capability.capability_contract_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "foreground_manifest_sha256": state["source_hashes"]["foreground_manifest"],
            "runtime_atlas_views": [{
                "view": int(row["view"]), "atlas_file": row["atlas_file"], "atlas_relpath": row["atlas_relpath"],
                "atlas_sha256": row["atlas_sha256"], "atlas_payload_hash": row["atlas_payload_hash"],
                "width": row["atlas_width"], "height": row["atlas_height"],
            } for row in state["foreground_manifest"]["views"]],
            "product_pass_claimed": False,
        }
        path = out / "PRODUCT_STATE_MANIFEST.json"
        _write_json(path, manifest)
        print("FINAL_PRODUCT_STATE=" + json.dumps({"product_state_hash": product.product_state_hash, "manifest_sha256": _sha(path)}, sort_keys=True), flush=True)
    return state


def parse_args():
    return v1.parse_args()


if __name__ == "__main__":
    build_final_state(parse_args(), persist=True)
