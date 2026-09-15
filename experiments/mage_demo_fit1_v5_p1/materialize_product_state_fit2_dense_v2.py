from __future__ import annotations

"""Build the bounded Mage product on dense zero-surface BODY + fresh FIT2 S/G/W.

The historical motion engine owns animation semantics only. Dense zero-surface BODY
owns deformable topology; current GSA/Geppetto/Arachne own mechanics; exact owner
sprites own rigid foreground. This stage still does not claim PRODUCT_PASS.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from compiler.realsas_compiler_core.component_attachment import (
    bind_component_assembly_to_directional_renderable_set,
    validate_component_assembly,
)
from compiler.realsas_compiler_core.continuity_underlay import (
    bind_continuity_underlay_set_to_directional_renderables,
    build_continuity_underlay_set,
    qualify_continuity_underlay,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.rigid_attachment_skin import qualify_rigid_attachment_mesh_skin
from compiler.realsas_compiler_core.motion_locomotion import build_mage_historical_phase_motion
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
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

SCHEMA = "RealSaS.MageFIT2.DenseProductState.v2.historical_motion_engine"
PROFILE = "MAGE_FIT2_DENSE_BOUNDED_DEMO_V2_HISTORICAL_MOTION_ENGINE"
DENSE_STATUS = "PASS__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY_V0_V7"
ASSEMBLY_STATUS = "PASS__FIT2_DENSE_TYPED_COMPONENT_ASSEMBLY_BODY_PLUS_4_RIGID_FOREGROUND"
FOREGROUND_STATUS = "PASS__EXACT_OBSERVATION_PLUS_TYPED_RIGID_FOREGROUND_ATLAS_V0_V7"
RIGID_IDS = legacy_state.RIGID_IDS
SETUP_ORDER = legacy_state.SETUP_ORDER


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_final_state(args, *, persist: bool = True):
    dense_dir, fg_dir, assembly_dir, out = Path(args.dense_body_dir), Path(args.foreground_dir), Path(args.assembly_dir), Path(args.output_dir)
    if persist: out.mkdir(parents=True, exist_ok=True)
    dense_manifest_path = dense_dir / "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"
    fg_manifest_path = fg_dir / "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json"
    assembly_manifest_path = assembly_dir / "FIT2_DENSE_COMPONENT_ASSEMBLY_MANIFEST.json"
    assembly_path = assembly_dir / "FIT2_DENSE_QUALIFIED_COMPONENT_ASSEMBLY.json"
    hashes = {"dense_body_manifest":_sha(dense_manifest_path),"foreground_manifest":_sha(fg_manifest_path),"assembly_manifest":_sha(assembly_manifest_path)}
    for key,arg_name in (("dense_body_manifest","expected_dense_body_manifest"),("foreground_manifest","expected_foreground_manifest"),("assembly_manifest","expected_assembly_manifest")):
        expected = str(getattr(args,arg_name,"") or "")
        if expected and hashes[key] != expected: raise RuntimeError(f"FIT2_DENSE_PRODUCT_{key.upper()}_SHA_DRIFT")

    surface,skeleton,skin,mechanical = fit2io.build_exact_mechanical(Path(args.fit2_surface),Path(args.skeleton),Path(args.fit2_skin))
    dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8")); fg_manifest = json.loads(fg_manifest_path.read_text(encoding="utf-8")); assembly_manifest = json.loads(assembly_manifest_path.read_text(encoding="utf-8"))
    if dense_manifest.get("status") != DENSE_STATUS: raise RuntimeError("FIT2_DENSE_PRODUCT_BODY_STATUS_INVALID")
    if dense_manifest.get("surface_lineage_hash") != surface.geometry_lineage_hash or dense_manifest.get("skeleton_lineage_hash") != skeleton.skeleton_lineage_hash or dense_manifest.get("skin_lineage_hash") != skin.skin_lineage_hash:
        raise RuntimeError("FIT2_DENSE_PRODUCT_BODY_AUTHORITY_DRIFT")
    if dense_manifest.get("P1_or_P1Q_topology_used") is not False or dense_manifest.get("historical_weight_transfer_used") is not False: raise RuntimeError("FIT2_DENSE_PRODUCT_BODY_FIREWALL")
    if fg_manifest.get("status") != FOREGROUND_STATUS: raise RuntimeError("FIT2_DENSE_PRODUCT_FOREGROUND_STATUS_INVALID")
    if assembly_manifest.get("status") != ASSEMBLY_STATUS or assembly_manifest.get("skin_lineage_hash") != skin.skin_lineage_hash: raise RuntimeError("FIT2_DENSE_PRODUCT_ASSEMBLY_STATUS_INVALID")

    assembly = legacy_state._load_assembly(assembly_path); validate_component_assembly(assembly,skeleton)
    assembly_by_id = {c.component_id:c for c in assembly.components}
    if set(("BODY_UNDERLAY",*RIGID_IDS)) - set(assembly_by_id): raise RuntimeError("FIT2_DENSE_PRODUCT_ASSEMBLY_COMPONENT_SET_INCOMPLETE")
    dense_rows = {int(r["view"]):r for r in dense_manifest.get("views") or ()}; fg_rows = {int(r["view"]):r for r in fg_manifest.get("views") or ()}
    if set(dense_rows) != set(range(8)) or set(fg_rows) != set(range(8)): raise RuntimeError("FIT2_DENSE_PRODUCT_REQUIRES_8_VIEWS")

    directions, underlays, persisted = [], [], []
    for view in range(8):
        drow,frow = dense_rows[view],fg_rows[view]
        mesh = v5base._load_mesh(dense_dir/drow["files"]["mesh"],surface,str(drow["mesh_lineage_hash"]))
        mesh_skin = legacy_p1q._load_mesh_skin(dense_dir/drow["files"]["skin"],mesh.mesh_lineage_hash)
        if mesh_skin.skin_binding_hash != skin.skin_lineage_hash or mesh_skin.surface_binding_hash != surface.geometry_lineage_hash or mesh_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
            raise RuntimeError(f"FIT2_DENSE_PRODUCT_BODY_BINDING_DRIFT_V{view}")
        appearance = legacy_state._load_appearance(dense_dir/drow["files"]["appearance"])
        if appearance.mesh_binding_hash != mesh.mesh_lineage_hash or int(appearance.target_view_index) != view: raise RuntimeError(f"FIT2_DENSE_PRODUCT_BODY_APPEARANCE_DRIFT_V{view}")
        underlay = build_external_renderable_component(
            component_id="BODY_UNDERLAY",view_index=view,mesh=mesh,mesh_skin=mesh_skin,mechanical=mechanical,appearance=appearance,setup_order=SETUP_ORDER["BODY_UNDERLAY"],coverage_classification=mesh.support_coverage_classification,
            materialization_manifest_sha256=hashes["dense_body_manifest"],direct_binding_manifest_sha256=str(drow["topology_qualification"]["compaction_replay_hash"]),
            metadata={"product_role":"CONTINUITY_DEFORMABLE_UNDERLAY","dense_zero_surface_topology":True,"fit2_skin_lineage_hash":skin.skin_lineage_hash,"component_assembly_hash":assembly.component_assembly_hash,"P1_or_P1Q_topology_used":False},
        )
        components=[underlay]; underlays.append(underlay)
        fg_by_id={str(r["runtime_component_id"]):r for r in frow["components"]}
        for runtime_id in RIGID_IDS:
            if runtime_id not in fg_by_id: raise RuntimeError(f"FIT2_DENSE_PRODUCT_FOREGROUND_COMPONENT_MISSING_V{view}:{runtime_id}")
            row=fg_by_id[runtime_id]; sprite_mesh=legacy_state._load_external_mesh(fg_dir/row["mesh_file"],str(row["mesh_lineage_hash"])); sprite_appearance=legacy_state._load_appearance(fg_dir/row["appearance_file"]); attachment=assembly_by_id[runtime_id]
            carry_skin=qualify_rigid_attachment_mesh_skin(sprite_mesh,mechanical,parent_joint_id=attachment.canonical_parent_joint_id,component_assembly_hash=assembly.component_assembly_hash,component_lineage_hash=attachment.component_lineage_hash,bind_state_authority_hash=attachment.bind_state_authority_hash)
            component=build_external_renderable_component(
                component_id=runtime_id,view_index=view,mesh=sprite_mesh,mesh_skin=carry_skin,mechanical=mechanical,appearance=sprite_appearance,setup_order=SETUP_ORDER[runtime_id],coverage_classification="RIGID_FOREGROUND_SPRITE",
                materialization_manifest_sha256=hashes["foreground_manifest"],direct_binding_manifest_sha256=hashes["assembly_manifest"],
                metadata={"product_role":"RIGID_FOREGROUND_DRAWABLE","component_assembly_hash":assembly.component_assembly_hash,"component_lineage_hash":attachment.component_lineage_hash,"bind_state_authority_hash":attachment.bind_state_authority_hash,"canonical_parent_joint_id":attachment.canonical_parent_joint_id,"runtime_carry_skin_lineage_hash":carry_skin.mesh_skin_lineage_hash,"fit2_skeleton_lineage_hash":skeleton.skeleton_lineage_hash},
            ); components.append(component)
            if persist:
                name=f"V{view}_{runtime_id}_FIT2_DENSE_RUNTIME_CARRY_SKIN.json"; _write_json(out/name,carry_skin.to_dict()); persisted.append({"path":name,"sha256":_sha(out/name)})
        directions.append(build_external_directional_renderable(view_index=view,camera_binding_hash=mesh.camera_binding_hash,components=tuple(components),mechanical=mechanical,metadata={"runtime_atlas_payload_hash":frow["atlas_payload_hash"],"component_count":len(components),"dense_zero_surface_body":True}))

    render_set=build_external_directional_renderable_set(tuple(directions),mechanical,metadata={"runtime_texture_layout":"ONE_ATLAS_PER_VIEW_PANEL0_UNDERLAY_PANEL1_4_RIGID_FOREGROUND","foreground_manifest_sha256":hashes["foreground_manifest"],"dense_body_manifest_sha256":hashes["dense_body_manifest"],"fit2_skin_lineage_hash":skin.skin_lineage_hash})
    render_set=bind_component_assembly_to_directional_renderable_set(render_set,assembly,skeleton)
    underlay_rows=tuple(qualify_continuity_underlay(view_index=view,substrate_component=underlays[view],mechanical=mechanical,component_assembly=assembly,partition_authority_sha256=str(fg_manifest["owner_manifest_sha256"]),metadata={"dense_zero_surface_topology":True,"new_pixels_generated":False,"topology_mutated":False,"weights_mutated":False,"scientific_mechanical_surface_relabelled":False,"fit2_skin_lineage_hash":skin.skin_lineage_hash}) for view in range(8))
    underlay_set=build_continuity_underlay_set(underlay_rows,component_assembly=assembly,metadata={"partition_authority_sha256":fg_manifest["owner_manifest_sha256"],"dense_zero_surface_topology":True,"fit2_skin_lineage_hash":skin.skin_lineage_hash})
    render_set=bind_continuity_underlay_set_to_directional_renderables(render_set,underlay_set,assembly,mechanical)

    phase_motion=build_mage_historical_phase_motion(mechanical); motion=compile_motion_quality(phase_motion,mechanical)
    policy_hash=content_sha256({"schema":"RealSaS.MageFIT2DenseHistoricalMotionEnginePolicy.v1","unseen_generalization_claimed":False,"fresh_fit2_mechanics":True,"dense_zero_surface_topology":True,"rotation_only_directional_evaluator_scope":True,"historical_motion_engine_recovered":True,"phase_authored_locomotion":True,"motion_quality_scope":("PHASE_AWARE_LOCOMOTION","NAMED_TIMING_CURVES","CUBIC_HERMITE_ROTATION","MINIMUM_JERK_TIMING_BLEND","CYCLIC_LOOP_REPAIR"),"surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,"contact_lock_claimed":False,"xpbd_secondary_claimed":False,"corrective_deformation_claimed":False,"runtime_requires_qualification_owned_bakes":True})
    runtime_impl_hash=content_sha256({"runtime":"RealSaS.NativeRuntimeV2+C++17","draw_order":"QUALIFICATION_OWNED_PER_FRAME_PER_VIEW","solver_replay":False})
    requirements=(
        CapabilityRequirement("MECHANICAL_STRUCTURE","REQUIRED",mechanical.mechanical_state_hash,policy_hash,("MECHANICAL_STRUCTURE",)),
        CapabilityRequirement("MESH_QUALITY","REQUIRED",render_set.directional_visual_state_hash,policy_hash,("MESH_QUALITY",)),
        CapabilityRequirement("VISUAL_8_DIRECTION","REQUIRED",render_set.directional_visual_state_hash,policy_hash,("DIRECTIONAL_VISUAL",)),
        CapabilityRequirement("PRESET_MOTION","REQUIRED",motion.motion_state_hash,policy_hash,("MOTION",)),
        CapabilityRequirement("RUNTIME_BACKEND","REQUIRED",runtime_impl_hash,policy_hash,("RUNTIME_CONSUMPTION",)),
    )
    capability=build_capability_contract(PROFILE,requirements,metadata={"generalization_claim":False,"fresh_fit2_mechanics":True,"dense_zero_surface_topology":True,"surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,"component_assembly_hash":assembly.component_assembly_hash,"continuity_underlay_set_hash":underlay_set.qualification_hash,"historical_motion_engine_recovered":True,"phase_motion_state_hash":phase_motion.motion_state_hash,"quality_motion_state_hash":motion.motion_state_hash,"contact_lock_claimed":False,"xpbd_secondary_claimed":False,"corrective_deformation_claimed":False})
    product=assemble_product_v3_with_external_render_support(mechanical,render_set,capability,motion,editable_metadata={"bounded_demo":True,"fresh_fit2_mechanics":True,"dense_zero_surface_topology":True,"surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,"component_assembly_hash":assembly.component_assembly_hash,"continuity_underlay_set_hash":underlay_set.qualification_hash,"historical_motion_engine_recovered":True,"phase_motion_state_hash":phase_motion.motion_state_hash},runtime_policy={"runtime_texture_layout":"ONE_ATLAS_PER_VIEW","qualification_owned_motion_bake_required":True,"export_solver_replay_forbidden":True,"mechanical_continuity_underlay_required":True,"continuity_underlay_set_hash":underlay_set.qualification_hash,"component_assembly_hash":assembly.component_assembly_hash,"contact_lock_unqualified_fail_closed":True,"secondary_motion_unqualified_fail_closed":True,"corrective_deformation_unqualified_fail_closed":True})

    state={"surface":surface,"skeleton":skeleton,"skin":skin,"mechanical":mechanical,"assembly":assembly,"render_set":render_set,"underlay_set":underlay_set,"phase_motion":phase_motion,"motion":motion,"capability":capability,"product":product,"source_hashes":hashes,"persisted":persisted}
    if persist:
        _write_json(out/"FIT2_DENSE_FINAL_DIRECTIONAL_RENDERABLE_SET.json",render_set.to_dict()); _write_json(out/"FIT2_DENSE_CONTINUITY_UNDERLAY_SET.json",underlay_set.to_dict()); _write_json(out/"FIT2_DENSE_HISTORICAL_PHASE_MOTION_STATE.json",phase_motion.to_dict()); _write_json(out/"FIT2_DENSE_HISTORICAL_QUALITY_MOTION_STATE.json",motion.to_dict()); _write_json(out/"FIT2_DENSE_MOTION_CAPABILITY_CONTRACT.json",capability.to_dict()); _write_json(out/"FIT2_DENSE_CANONICAL_PUPPET_GRAPH_V3_MOTION.json",product.to_dict())
        manifest={"schema":SCHEMA,"status":"PASS__CURRENT_MAGE_FIT2_DENSE_PRODUCT_STATE_HISTORICAL_MOTION_ENGINE_BOUND","product_state_hash":product.product_state_hash,"mechanical_state_hash":mechanical.mechanical_state_hash,"directional_visual_state_hash":render_set.directional_visual_state_hash,"phase_motion_state_hash":phase_motion.motion_state_hash,"motion_state_hash":motion.motion_state_hash,"capability_contract_hash":capability.capability_contract_hash,"component_assembly_hash":assembly.component_assembly_hash,"continuity_underlay_set_hash":underlay_set.qualification_hash,"surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,"dense_body_manifest_sha256":hashes["dense_body_manifest"],"foreground_manifest_sha256":hashes["foreground_manifest"],"assembly_manifest_sha256":hashes["assembly_manifest"],"historical_fit1_skin_used":False,"P1_or_P1Q_topology_used":False,"dense_zero_surface_topology":True,"historical_motion_engine_recovered":True,"recovered_product_bound_scope":["PHASE_AWARE_LOCOMOTION","NAMED_TIMING_CURVES","CUBIC_HERMITE_ROTATION","MINIMUM_JERK_TIMING_BLEND","CYCLIC_LOOP_REPAIR"],"available_not_product_bound_scope":["CONTACT_LOCK","CORRECTIVE_DEFORMATION","XPBD_SECONDARY"],"contact_lock_claimed":False,"corrective_deformation_claimed":False,"xpbd_secondary_claimed":False,"product_pass_claimed":False}
        path=out/"FIT2_DENSE_PRODUCT_STATE_MOTION_ENGINE_MANIFEST.json"; _write_json(path,manifest); print("FIT2_DENSE_MOTION_ENGINE_PRODUCT_STATE="+json.dumps({"product_state_hash":product.product_state_hash,"motion_state_hash":motion.motion_state_hash,"skin_lineage_hash":skin.skin_lineage_hash,"manifest_sha256":_sha(path)},sort_keys=True),flush=True)
    return state


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--dense-body-dir",required=True); p.add_argument("--foreground-dir",required=True); p.add_argument("--assembly-dir",required=True); p.add_argument("--fit2-surface",required=True); p.add_argument("--skeleton",required=True); p.add_argument("--fit2-skin",required=True); p.add_argument("--expected-dense-body-manifest",default=""); p.add_argument("--expected-foreground-manifest",default=""); p.add_argument("--expected-assembly-manifest",default=""); p.add_argument("--output-dir",required=True); return p.parse_args()


if __name__ == "__main__":
    build_final_state(parse_args(),persist=True)
