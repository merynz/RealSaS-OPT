from __future__ import annotations

"""Final dense FIT2 product state bound to file-backed topology/W/final-motion evidence.

The dense BODY manifest is accepted only when its persisted dynamic witnesses were
qualified against the exact motion_state_hash consumed by this product. V2 is used
only as an in-memory constructor for sealed mesh/skin/art/rigid objects. No P1/P1Q
mechanical topology, historical skin, training, model query or threshold relaxation
is permitted. This stage does not claim PRODUCT_PASS.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_external_render import assemble_product_v3_with_external_render_support
from compiler.realsas_compiler_core.v4 import build_capability_contract
from compiler.realsas_compiler_core.v4_types import CapabilityRequirement

import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_dense_v2 as dense_v2
import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_dense_v3_evidence_bound as dense_v3
import experiments.mage_demo_fit1_v5_p1.seal_fit2_dense_zero_surface_body_binding_v2_motion_bound as derivation_v2

SCHEMA = "RealSaS.MageFIT2.DenseProductState.v4.motion_bound"
PROFILE = "MAGE_FIT2_DENSE_BOUNDED_DEMO_V4_MOTION_BOUND"
DERIVATION_FILE = derivation_v2.DERIVATION_FILE
BODY_MANIFEST_FILE = "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"
FOREGROUND_MANIFEST_FILE = "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _log(stage: str, started: float, **fields) -> None:
    payload = {"stage": stage, "elapsed_s": round(monotonic() - started, 3), **fields}
    print("FIT2_DENSE_V4_PROGRESS=" + json.dumps(payload, sort_keys=True), flush=True)


def _build_v2_constructor_state(args, started: float):
    """Use V2 only for the exact sealed object graph that V4 actually consumes.

    The previous V4 path let V2 assemble and validate an intermediate canonical
    product even though V4 immediately discarded it and assembled the final product
    again after motion-bound requalification. On dense BODY that repeated the whole
    directional graph validation/hashing for no product-visible benefit.
    """
    original_assemble = dense_v2.assemble_product_v3_with_external_render_support

    def _constructor_only_product(*_args, **_kwargs):
        return None

    dense_v2.assemble_product_v3_with_external_render_support = _constructor_only_product
    try:
        _log("V2_CONSTRUCTOR_BEGIN", started)
        state = dense_v2.build_final_state(args, persist=False)
    finally:
        dense_v2.assemble_product_v3_with_external_render_support = original_assemble
    if state.get("render_set") is None or state.get("underlay_set") is None:
        raise RuntimeError("FIT2_DENSE_V4_CONSTRUCTOR_STATE_INCOMPLETE")
    _log("V2_CONSTRUCTOR_DONE", started)
    return state


def _load_motion_bound_derivation(args, state):
    dense_dir = Path(args.dense_body_dir)
    derivation_dir = Path(args.dense_derivation_dir)
    body_path = dense_dir / BODY_MANIFEST_FILE
    derivation_path = derivation_dir / DERIVATION_FILE
    body_sha = _sha(body_path)
    derivation_sha = _sha(derivation_path)
    body = json.loads(body_path.read_text(encoding="utf-8"))
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))

    if derivation.get("status") != derivation_v2.DERIVATION_STATUS:
        raise RuntimeError("FIT2_DENSE_V4_DERIVATION_STATUS_INVALID")
    if derivation.get("source_dense_body_manifest_sha256") != body_sha:
        raise RuntimeError("FIT2_DENSE_V4_DERIVATION_BODY_MANIFEST_DRIFT")
    if derivation.get("surface_lineage_hash") != state["surface"].geometry_lineage_hash:
        raise RuntimeError("FIT2_DENSE_V4_DERIVATION_SURFACE_DRIFT")
    if derivation.get("skeleton_lineage_hash") != state["skeleton"].skeleton_lineage_hash:
        raise RuntimeError("FIT2_DENSE_V4_DERIVATION_SKELETON_DRIFT")
    if derivation.get("skin_lineage_hash") != state["skin"].skin_lineage_hash:
        raise RuntimeError("FIT2_DENSE_V4_DERIVATION_SKIN_DRIFT")

    product_motion_hash = str(state["motion"].motion_state_hash)
    if str(body.get("motion_state_hash") or "") != product_motion_hash:
        raise RuntimeError("FIT2_DENSE_V4_BODY_FINAL_MOTION_HASH_DRIFT")
    if str(derivation.get("motion_state_hash") or "") != product_motion_hash:
        raise RuntimeError("FIT2_DENSE_V4_DERIVATION_FINAL_MOTION_HASH_DRIFT")
    if body.get("dynamic_motion_authority") != derivation_v2.EXPECTED_MOTION_AUTHORITY:
        raise RuntimeError("FIT2_DENSE_V4_BODY_MOTION_AUTHORITY_DRIFT")
    if derivation.get("dynamic_motion_authority") != derivation_v2.EXPECTED_MOTION_AUTHORITY:
        raise RuntimeError("FIT2_DENSE_V4_DERIVATION_MOTION_AUTHORITY_DRIFT")
    if body.get("historical_motion_engine_recovered") is not True or derivation.get("historical_motion_engine_recovered") is not True:
        raise RuntimeError("FIT2_DENSE_V4_HISTORICAL_MOTION_FLAG_MISSING")
    if derivation.get("topology_W_and_motion_share_one_persisted_authority_chain") is not True:
        raise RuntimeError("FIT2_DENSE_V4_MOTION_BOUND_CHAIN_FLAG_MISSING")

    for flag in (
        "new_adjacency_created", "P1_or_P1Q_topology_used",
        "source_or_teacher_mesh_topology_used", "historical_weight_transfer_used",
        "direct_model_query_used", "training_executed", "product_pass_claimed",
    ):
        if derivation.get(flag) is not False:
            raise RuntimeError(f"FIT2_DENSE_V4_DERIVATION_FIREWALL:{flag}")

    witness_rows = derivation.get("dynamic_witnesses") or ()
    if len(witness_rows) != 8 or {int(r["view"]) for r in witness_rows} != set(range(8)):
        raise RuntimeError("FIT2_DENSE_V4_DYNAMIC_WITNESS_SET_INCOMPLETE")
    if any(str(r.get("motion_state_hash") or "") != product_motion_hash for r in witness_rows):
        raise RuntimeError("FIT2_DENSE_V4_DYNAMIC_WITNESS_MOTION_HASH_DRIFT")

    return body, derivation, body_sha, derivation_sha


def build_final_state(args, *, persist: bool = True):
    started = monotonic()
    _log("BEGIN", started)

    # V2 constructs exact sealed inputs only. Its intermediate product assembly is
    # deliberately skipped because V4 will replace render identity and assemble the
    # one final product below.
    state = _build_v2_constructor_state(args, started)

    _log("MOTION_BOUND_DERIVATION_BEGIN", started)
    body, derivation, body_sha, derivation_sha = _load_motion_bound_derivation(args, state)
    _log(
        "MOTION_BOUND_DERIVATION_DONE",
        started,
        body_manifest_sha256=body_sha,
        derivation_manifest_sha256=derivation_sha,
    )

    foreground_manifest_path = Path(args.foreground_dir) / FOREGROUND_MANIFEST_FILE
    foreground_manifest = json.loads(foreground_manifest_path.read_text(encoding="utf-8"))
    foreground_owner_sha = str(foreground_manifest.get("owner_manifest_sha256") or "")
    if not foreground_owner_sha:
        raise RuntimeError("FIT2_DENSE_V4_FOREGROUND_OWNER_AUTHORITY_MISSING")

    # Requalify BODY support so external-render identity binds the persisted V2
    # topology/W/final-motion derivation file SHA, not a generic replay content hash.
    _log("REQUALIFY_RENDER_SET_BEGIN", started)
    render_set, underlay_set = dense_v3._requalify_render_set(
        state,
        body_manifest_sha=body_sha,
        derivation_sha=derivation_sha,
        foreground_owner_sha=foreground_owner_sha,
    )
    _log(
        "REQUALIFY_RENDER_SET_DONE",
        started,
        directional_visual_state_hash=render_set.directional_visual_state_hash,
        continuity_underlay_set_hash=underlay_set.qualification_hash,
    )

    mechanical = state["mechanical"]
    surface = state["surface"]
    skeleton = state["skeleton"]
    skin = state["skin"]
    assembly = state["assembly"]
    phase_motion = state["phase_motion"]
    motion = state["motion"]

    motion_hash = str(motion.motion_state_hash)
    _log("FINAL_CAPABILITY_BEGIN", started)
    policy_hash = content_sha256({
        "schema": "RealSaS.MageFIT2DenseMotionBoundHistoricalMotionEnginePolicy.v1",
        "unseen_generalization_claimed": False,
        "fresh_fit2_mechanics": True,
        "dense_zero_surface_topology": True,
        "binding_derivation_file_backed": True,
        "dense_binding_derivation_manifest_sha256": derivation_sha,
        "dynamic_face_admission_motion_state_hash": motion_hash,
        "final_product_motion_state_hash": motion_hash,
        "dynamic_motion_authority": derivation_v2.EXPECTED_MOTION_AUTHORITY,
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
        CapabilityRequirement("PRESET_MOTION", "REQUIRED", motion_hash, policy_hash, ("MOTION",)),
        CapabilityRequirement("RUNTIME_BACKEND", "REQUIRED", runtime_impl_hash, policy_hash, ("RUNTIME_CONSUMPTION",)),
    )
    capability = build_capability_contract(PROFILE, requirements, metadata={
        "generalization_claim": False,
        "fresh_fit2_mechanics": True,
        "dense_zero_surface_topology": True,
        "binding_derivation_file_backed": True,
        "dense_binding_derivation_manifest_sha256": derivation_sha,
        "dynamic_face_admission_motion_state_hash": motion_hash,
        "final_product_motion_state_hash": motion_hash,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "component_assembly_hash": assembly.component_assembly_hash,
        "continuity_underlay_set_hash": underlay_set.qualification_hash,
        "historical_motion_engine_recovered": True,
        "phase_motion_state_hash": phase_motion.motion_state_hash,
        "quality_motion_state_hash": motion_hash,
    })
    _log("FINAL_CAPABILITY_DONE", started, capability_contract_hash=capability.capability_contract_hash)

    _log("FINAL_PRODUCT_ASSEMBLY_BEGIN", started)
    product = assemble_product_v3_with_external_render_support(
        mechanical, render_set, capability, motion,
        editable_metadata={
            "bounded_demo": True,
            "fresh_fit2_mechanics": True,
            "dense_zero_surface_topology": True,
            "binding_derivation_file_backed": True,
            "dense_binding_derivation_manifest_sha256": derivation_sha,
            "dynamic_face_admission_motion_state_hash": motion_hash,
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
            "dynamic_face_admission_motion_state_hash": motion_hash,
            "contact_lock_unqualified_fail_closed": True,
            "secondary_motion_unqualified_fail_closed": True,
            "corrective_deformation_unqualified_fail_closed": True,
        },
    )
    _log("FINAL_PRODUCT_ASSEMBLY_DONE", started, product_state_hash=product.product_state_hash)

    state.update({
        "render_set": render_set,
        "underlay_set": underlay_set,
        "capability": capability,
        "product": product,
        "dense_derivation_v2": derivation,
        "dense_derivation_v2_sha256": derivation_sha,
    })

    if persist:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _log("PERSIST_RENDER_SET_BEGIN", started)
        _write_json(out / "FIT2_DENSE_V4_DIRECTIONAL_RENDERABLE_SET.json", render_set.to_dict())
        _log("PERSIST_RENDER_SET_DONE", started)
        _write_json(out / "FIT2_DENSE_V4_CONTINUITY_UNDERLAY_SET.json", underlay_set.to_dict())
        _write_json(out / "FIT2_DENSE_V4_HISTORICAL_PHASE_MOTION_STATE.json", phase_motion.to_dict())
        _write_json(out / "FIT2_DENSE_V4_HISTORICAL_QUALITY_MOTION_STATE.json", motion.to_dict())
        _write_json(out / "FIT2_DENSE_V4_MOTION_CAPABILITY_CONTRACT.json", capability.to_dict())
        _log("PERSIST_PRODUCT_GRAPH_BEGIN", started)
        _write_json(out / "FIT2_DENSE_V4_CANONICAL_PUPPET_GRAPH_V3_MOTION.json", product.to_dict())
        _log("PERSIST_PRODUCT_GRAPH_DONE", started)
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_FIT2_DENSE_PRODUCT_STATE_FINAL_MOTION_BOUND",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "phase_motion_state_hash": phase_motion.motion_state_hash,
            "motion_state_hash": motion_hash,
            "dynamic_face_admission_motion_state_hash": motion_hash,
            "capability_contract_hash": capability.capability_contract_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "dense_body_manifest_sha256": body_sha,
            "dense_binding_derivation_v2_manifest_sha256": derivation_sha,
            "dense_binding_derivation_v2_status": derivation["status"],
            "dynamic_motion_authority": derivation_v2.EXPECTED_MOTION_AUTHORITY,
            "historical_fit1_skin_used": False,
            "P1_or_P1Q_topology_used": False,
            "dense_zero_surface_topology": True,
            "historical_motion_engine_recovered": True,
            "product_pass_claimed": False,
        }
        path = out / "FIT2_DENSE_V4_PRODUCT_STATE_MOTION_ENGINE_MANIFEST.json"
        _write_json(path, manifest)
        _log("DONE", started, manifest_sha256=_sha(path))
        print("FIT2_DENSE_V4_PRODUCT_STATE=" + json.dumps({
            "product_state_hash": product.product_state_hash,
            "motion_state_hash": motion_hash,
            "dense_binding_derivation_v2_manifest_sha256": derivation_sha,
            "manifest_sha256": _sha(path),
        }, sort_keys=True), flush=True)
    return state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dense-body-dir", required=True)
    p.add_argument("--dense-derivation-dir", required=True)
    p.add_argument("--foreground-dir", required=True)
    p.add_argument("--assembly-dir", required=True)
    p.add_argument("--fit2-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--expected-dense-body-manifest", default="")
    p.add_argument("--expected-foreground-manifest", default="")
    p.add_argument("--expected-assembly-manifest", default="")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    build_final_state(parse_args(), persist=True)
