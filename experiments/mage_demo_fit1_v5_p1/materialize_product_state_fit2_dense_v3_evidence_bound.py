from __future__ import annotations

"""Final dense FIT2 product state with file-backed BODY binding evidence.

V2 is used only as an in-memory constructor for already-qualified mesh/skin/art and
rigid carry objects. This module then replaces every BODY external-render support
qualification with a persisted dense-binding-derivation manifest SHA, rebuilds the
continuity authority, capability contract and product identity, and persists only the
requalified state. No V2 product is persisted or promoted by this module.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from compiler.realsas_compiler_core.continuity_underlay import (
    bind_continuity_underlay_set_to_directional_renderables,
    build_continuity_underlay_set,
    qualify_continuity_underlay,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_external_render import (
    assemble_product_v3_with_external_render_support,
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.v4 import build_capability_contract
from compiler.realsas_compiler_core.v4_types import CapabilityRequirement

import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_dense_v2 as dense_v2
import experiments.mage_demo_fit1_v5_p1.seal_fit2_dense_zero_surface_body_binding_v1 as dense_derivation

SCHEMA = "RealSaS.MageFIT2.DenseProductState.v3.evidence_bound"
PROFILE = "MAGE_FIT2_DENSE_BOUNDED_DEMO_V3_EVIDENCE_BOUND"
DERIVATION_FILE = "FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION_MANIFEST.json"
DENSE_BODY_MANIFEST_FILE = "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_derivation(args, state):
    dense_dir = Path(args.dense_body_dir)
    derivation_dir = Path(args.dense_derivation_dir)
    body_path = dense_dir / DENSE_BODY_MANIFEST_FILE
    path = derivation_dir / DERIVATION_FILE
    body_sha, derivation_sha = _sha(body_path), _sha(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != dense_derivation.DERIVATION_STATUS:
        raise RuntimeError("FIT2_DENSE_V3_DERIVATION_STATUS_INVALID")
    if value.get("source_dense_body_manifest_sha256") != body_sha:
        raise RuntimeError("FIT2_DENSE_V3_DERIVATION_BODY_MANIFEST_DRIFT")
    if value.get("surface_lineage_hash") != state["surface"].geometry_lineage_hash:
        raise RuntimeError("FIT2_DENSE_V3_DERIVATION_SURFACE_DRIFT")
    if value.get("skeleton_lineage_hash") != state["skeleton"].skeleton_lineage_hash:
        raise RuntimeError("FIT2_DENSE_V3_DERIVATION_SKELETON_DRIFT")
    if value.get("skin_lineage_hash") != state["skin"].skin_lineage_hash:
        raise RuntimeError("FIT2_DENSE_V3_DERIVATION_SKIN_DRIFT")
    body = json.loads(body_path.read_text(encoding="utf-8"))
    replay = dict(body.get("compaction_replay") or {})
    if value.get("compaction_replay_hash") != replay.get("replay_hash"):
        raise RuntimeError("FIT2_DENSE_V3_DERIVATION_REPLAY_HASH_DRIFT")
    if value.get("compaction_inverse_sha256") != replay.get("inverse_sha256"):
        raise RuntimeError("FIT2_DENSE_V3_DERIVATION_INVERSE_HASH_DRIFT")
    for flag in ("topology_and_W_share_exact_compaction_address", "original_dense_zero_surface_faces_only"):
        if value.get(flag) is not True:
            raise RuntimeError(f"FIT2_DENSE_V3_DERIVATION_REQUIRED_TRUE_FLAG:{flag}")
    for flag in ("new_adjacency_created", "P1_or_P1Q_topology_used", "source_or_teacher_mesh_topology_used", "historical_weight_transfer_used", "direct_model_query_used", "training_executed", "product_pass_claimed"):
        if value.get(flag) is not False:
            raise RuntimeError(f"FIT2_DENSE_V3_DERIVATION_REQUIRED_FALSE_FLAG:{flag}")
    return value, body_sha, derivation_sha


def _requalify_render_set(state, *, body_manifest_sha: str, derivation_sha: str):
    mechanical, assembly = state["mechanical"], state["assembly"]
    clean_directions, clean_underlays = [], []
    for direction in sorted(state["render_set"].directions, key=lambda d: int(d.view_index)):
        view = int(direction.view_index)
        components = []
        for component in sorted(direction.components, key=lambda c: (int(c.setup_order), str(c.component_id))):
            if component.component_id != "BODY_UNDERLAY":
                components.append(component)
                continue
            body = build_external_renderable_component(
                component_id="BODY_UNDERLAY",
                view_index=view,
                mesh=component.mesh,
                mesh_skin=component.mesh_skin,
                mechanical=mechanical,
                appearance=component.appearance,
                setup_order=int(component.setup_order),
                coverage_classification=str(component.coverage_classification),
                materialization_manifest_sha256=body_manifest_sha,
                direct_binding_manifest_sha256=derivation_sha,
                metadata={
                    "product_role": "CONTINUITY_DEFORMABLE_UNDERLAY",
                    "dense_zero_surface_topology": True,
                    "fit2_skin_lineage_hash": state["skin"].skin_lineage_hash,
                    "component_assembly_hash": assembly.component_assembly_hash,
                    "P1_or_P1Q_topology_used": False,
                    "binding_derivation_file_backed": True,
                    "dense_binding_derivation_manifest_sha256": derivation_sha,
                },
            )
            components.append(body)
            clean_underlays.append(body)
        clean_directions.append(build_external_directional_renderable(
            view_index=view,
            camera_binding_hash=str(direction.camera_binding_hash),
            components=tuple(components),
            mechanical=mechanical,
            metadata={
                "dense_zero_surface_body": True,
                "binding_derivation_file_backed": True,
                "dense_binding_derivation_manifest_sha256": derivation_sha,
            },
        ))
    if len(clean_underlays) != 8:
        raise RuntimeError("FIT2_DENSE_V3_REQUALIFIED_BODY_VIEW_SET_INCOMPLETE")
    render_set = build_external_directional_renderable_set(
        tuple(clean_directions), mechanical,
        metadata={
            "dense_zero_surface_body": True,
            "binding_derivation_file_backed": True,
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "fit2_skin_lineage_hash": state["skin"].skin_lineage_hash,
        },
    )
    fg_owner = json.loads((Path(state["source_paths"]["foreground_manifest"])).read_text(encoding="utf-8"))["owner_manifest_sha256"] if "source_paths" in state else None
    if not fg_owner:
        # V2 exposes the foreground manifest SHA but not path; the caller supplies the same file path in args.
        raise RuntimeError("FIT2_DENSE_V3_INTERNAL_FOREGROUND_OWNER_PATH_REQUIRED")
    underlay_rows = tuple(qualify_continuity_underlay(
        view_index=view,
        substrate_component=clean_underlays[view],
        mechanical=mechanical,
        component_assembly=assembly,
        partition_authority_sha256=str(fg_owner),
        metadata={
            "dense_zero_surface_topology": True,
            "binding_derivation_file_backed": True,
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "new_pixels_generated": False,
            "topology_mutated": False,
            "weights_mutated": False,
            "scientific_mechanical_surface_relabelled": False,
            "fit2_skin_lineage_hash": state["skin"].skin_lineage_hash,
        },
    ) for view in range(8))
    underlay_set = build_continuity_underlay_set(
        underlay_rows, component_assembly=assembly,
        metadata={
            "partition_authority_sha256": str(fg_owner),
            "dense_zero_surface_topology": True,
            "binding_derivation_file_backed": True,
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "fit2_skin_lineage_hash": state["skin"].skin_lineage_hash,
        },
    )
    render_set = bind_continuity_underlay_set_to_directional_renderables(
        render_set, underlay_set, assembly, mechanical
    )
    return render_set, underlay_set


def build_final_state(args, *, persist: bool = True):
    # V2 is never persisted here; it is only an object constructor for exact sealed inputs.
    state = dense_v2.build_final_state(args, persist=False)
    state["source_paths"] = {"foreground_manifest": str(Path(args.foreground_dir) / "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json")}
    derivation, body_manifest_sha, derivation_sha = _load_derivation(args, state)
    render_set, underlay_set = _requalify_render_set(
        state, body_manifest_sha=body_manifest_sha, derivation_sha=derivation_sha
    )

    mechanical, surface, skeleton, skin = state["mechanical"], state["surface"], state["skeleton"], state["skin"]
    assembly, phase_motion, motion = state["assembly"], state["phase_motion"], state["motion"]
    policy_hash = content_sha256({
        "schema": "RealSaS.MageFIT2DenseEvidenceBoundHistoricalMotionEnginePolicy.v1",
        "unseen_generalization_claimed": False,
        "fresh_fit2_mechanics": True,
        "dense_zero_surface_topology": True,
        "binding_derivation_file_backed": True,
        "dense_binding_derivation_manifest_sha256": derivation_sha,
        "rotation_only_directional_evaluator_scope": True,
        "historical_motion_engine_recovered": True,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
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
    capability = build_capability_contract(PROFILE, requirements, metadata={
        "generalization_claim": False,
        "fresh_fit2_mechanics": True,
        "dense_zero_surface_topology": True,
        "binding_derivation_file_backed": True,
        "dense_binding_derivation_manifest_sha256": derivation_sha,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "component_assembly_hash": assembly.component_assembly_hash,
        "continuity_underlay_set_hash": underlay_set.qualification_hash,
        "historical_motion_engine_recovered": True,
        "phase_motion_state_hash": phase_motion.motion_state_hash,
        "quality_motion_state_hash": motion.motion_state_hash,
    })
    product = assemble_product_v3_with_external_render_support(
        mechanical, render_set, capability, motion,
        editable_metadata={
            "bounded_demo": True,
            "fresh_fit2_mechanics": True,
            "dense_zero_surface_topology": True,
            "binding_derivation_file_backed": True,
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
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
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "contact_lock_unqualified_fail_closed": True,
            "secondary_motion_unqualified_fail_closed": True,
            "corrective_deformation_unqualified_fail_closed": True,
        },
    )
    state.update({"render_set": render_set, "underlay_set": underlay_set, "capability": capability, "product": product, "dense_derivation": derivation, "dense_derivation_sha256": derivation_sha})

    if persist:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "FIT2_DENSE_V3_DIRECTIONAL_RENDERABLE_SET.json", render_set.to_dict())
        _write_json(out / "FIT2_DENSE_V3_CONTINUITY_UNDERLAY_SET.json", underlay_set.to_dict())
        _write_json(out / "FIT2_DENSE_V3_HISTORICAL_PHASE_MOTION_STATE.json", phase_motion.to_dict())
        _write_json(out / "FIT2_DENSE_V3_HISTORICAL_QUALITY_MOTION_STATE.json", motion.to_dict())
        _write_json(out / "FIT2_DENSE_V3_MOTION_CAPABILITY_CONTRACT.json", capability.to_dict())
        _write_json(out / "FIT2_DENSE_V3_CANONICAL_PUPPET_GRAPH_V3_MOTION.json", product.to_dict())
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_FIT2_DENSE_PRODUCT_STATE_EVIDENCE_BOUND",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "phase_motion_state_hash": phase_motion.motion_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "capability_contract_hash": capability.capability_contract_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "dense_body_manifest_sha256": body_manifest_sha,
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "dense_binding_derivation_status": derivation["status"],
            "historical_fit1_skin_used": False,
            "P1_or_P1Q_topology_used": False,
            "dense_zero_surface_topology": True,
            "historical_motion_engine_recovered": True,
            "product_pass_claimed": False,
        }
        path = out / "FIT2_DENSE_V3_PRODUCT_STATE_MOTION_ENGINE_MANIFEST.json"
        _write_json(path, manifest)
        print("FIT2_DENSE_V3_PRODUCT_STATE=" + json.dumps({
            "product_state_hash": product.product_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "manifest_sha256": _sha(path),
        }, sort_keys=True), flush=True)
    return state


def parse_args():
    p = dense_v2.parse_args()
    p.add_argument("--dense-derivation-dir", required=True)
    return p


if __name__ == "__main__":
    build_final_state(parse_args(), persist=True)
