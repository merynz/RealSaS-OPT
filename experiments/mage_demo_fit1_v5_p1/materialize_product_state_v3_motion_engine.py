from __future__ import annotations

"""Finalize the bounded Mage with the recovered historical motion engine.

This is deliberately a post-puppet finalizer. Mesh/rig/skin/component/appearance
and continuity authority come from the already qualified current product state.
The recovered historical engine owns only animation authoring/polish and remains
subordinate to the current directional evaluator + proof + runtime export chain.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.continuity_underlay import bind_continuity_underlay_set_to_directional_renderables
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion_locomotion import build_mage_historical_phase_motion
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_core.product_external_render import assemble_product_v3_with_external_render_support
from compiler.realsas_compiler_core.v4 import build_capability_contract
from compiler.realsas_compiler_core.v4_types import CapabilityRequirement

import experiments.mage_demo_fit1_v5_p1.materialize_product_state_v1 as v1

SCHEMA = "RealSaS.MageCurrentProductState.v3.historical_motion_engine"
PROFILE = "MAGE_FIT1_BOUNDED_DEMO_V3_HISTORICAL_MOTION_ENGINE"


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
    # V1 remains the canonical materializer for qualified puppet/render artifacts.
    # Its motion output is ignored below; recovery never reclaims mesh/rig/skin authority.
    state = v1.build_state(args, persist=persist)
    mechanical = state["mechanical"]
    assembly = state["assembly"]
    underlay_set = state["underlay_set"]
    render_set = bind_continuity_underlay_set_to_directional_renderables(
        state["render_set"], underlay_set, assembly, mechanical
    )

    phase_motion = build_mage_historical_phase_motion(mechanical)
    motion = compile_motion_quality(phase_motion, mechanical)

    policy_hash = content_sha256({
        "schema": "RealSaS.MageBoundedDemoHistoricalMotionEnginePolicy.v1",
        "fit2_training_deferred": True,
        "unseen_generalization_claimed": False,
        "rotation_only_directional_evaluator_scope": True,
        "historical_motion_engine_recovered": True,
        "phase_authored_locomotion": True,
        "motion_quality_scope": (
            "PHASE_AWARE_LOCOMOTION",
            "NAMED_TIMING_CURVES",
            "CUBIC_HERMITE_ROTATION",
            "MINIMUM_JERK_TIMING_BLEND",
            "CYCLIC_LOOP_REPAIR",
        ),
        "contact_schedule_present": True,
        "contact_lock_claimed": False,
        "xpbd_secondary_subsystem_available_not_product_bound": True,
        "xpbd_secondary_claimed": False,
        "corrective_subsystem_available_not_product_bound": True,
        "corrective_deformation_claimed": False,
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
    capability = build_capability_contract(PROFILE, requirements, metadata={
        "generalization_claim": False,
        "fit2_training_executed": False,
        "component_assembly_hash": assembly.component_assembly_hash,
        "continuity_underlay_set_hash": underlay_set.qualification_hash,
        "continuity_bound_directional_visual_state_hash": render_set.directional_visual_state_hash,
        "historical_motion_engine_recovered": True,
        "phase_motion_state_hash": phase_motion.motion_state_hash,
        "quality_motion_state_hash": motion.motion_state_hash,
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
            "phase_motion_state_hash": phase_motion.motion_state_hash,
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
    state.update({
        "product": product,
        "render_set": render_set,
        "capability": capability,
        "phase_motion": phase_motion,
        "motion": motion,
    })

    if persist:
        out = Path(args.output_dir)
        _write_json(out / "FINAL_DIRECTIONAL_RENDERABLE_SET.json", render_set.to_dict())
        _write_json(out / "HISTORICAL_PHASE_MOTION_STATE.json", phase_motion.to_dict())
        _write_json(out / "HISTORICAL_QUALITY_MOTION_STATE.json", motion.to_dict())
        _write_json(out / "CAPABILITY_CONTRACT.json", capability.to_dict())
        _write_json(out / "CANONICAL_PUPPET_GRAPH_V3.json", product.to_dict())
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_PRODUCT_STATE_HISTORICAL_MOTION_ENGINE_BOUND",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "phase_motion_state_hash": phase_motion.motion_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "capability_contract_hash": capability.capability_contract_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "historical_motion_engine_recovered": True,
            "recovered_product_bound_scope": [
                "PHASE_AWARE_LOCOMOTION",
                "NAMED_TIMING_CURVES",
                "CUBIC_HERMITE_ROTATION",
                "MINIMUM_JERK_TIMING_BLEND",
                "CYCLIC_LOOP_REPAIR",
            ],
            "available_not_product_bound_scope": ["CONTACT_LOCK", "CORRECTIVE_DEFORMATION", "XPBD_SECONDARY"],
            "contact_lock_claimed": False,
            "corrective_deformation_claimed": False,
            "xpbd_secondary_claimed": False,
            "foreground_manifest_sha256": state["source_hashes"]["foreground_manifest"],
            "product_pass_claimed": False,
        }
        path = out / "PRODUCT_STATE_MOTION_ENGINE_MANIFEST.json"
        _write_json(path, manifest)
        print("MOTION_ENGINE_PRODUCT_STATE=" + json.dumps({
            "product_state_hash": product.product_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "manifest_sha256": _sha(path),
        }, sort_keys=True), flush=True)
    return state


def parse_args():
    return v1.parse_args()


if __name__ == "__main__":
    build_final_state(parse_args(), persist=True)
