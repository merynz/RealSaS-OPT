from __future__ import annotations

"""Current Mage product state from qualified S/G/W mechanical component partition.

This path intentionally does not consume source component truth, source owner rasters,
or the historical rigid-foreground atlas.  P1Q is used only as an already-qualified
render/deformation carrier; Compiler mechanics partition its supported faces before
product assembly.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.component_attachment import (
    ComponentAttachmentEvidenceIR,
    bind_component_assembly_to_directional_renderable_set,
    qualify_component_assembly,
    qualify_component_attachment,
)
from compiler.realsas_compiler_core.continuity_underlay import (
    bind_continuity_underlay_set_to_directional_renderables,
    build_continuity_underlay_set,
    qualify_continuity_underlay,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_component_partition import (
    derive_mechanical_component_partition,
)
from compiler.realsas_compiler_core.mesh.component_partition import (
    project_mesh_to_mechanical_components,
)
from compiler.realsas_compiler_core.motion_locomotion import (
    build_mage_historical_phase_motion,
)
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


SCHEMA = "RealSaS.MageFIT2.CurrentProductState.v4.mechanical_component_partition"
PROFILE = "MAGE_FIT2_BOUNDED_DEMO_V4_MECHANICAL_COMPONENT_PARTITION"
BODY_COMPONENT_ID = "BODY_UNDERLAY"
SETUP_ORDER = {
    "BODY_UNDERLAY": 0,
    "RIGID_CHEST": 1,
    "RIGID_HAND_LEFT": 2,
    "RIGID_HAND_RIGHT": 3,
    "RIGID_HEAD": 4,
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_p1q_state(args, surface, skeleton, skin):
    p1q_dir = Path(args.p1q_dir)
    manifest_path = p1q_dir / "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"FIT2_V4_P1Q_MANIFEST_MISSING:{manifest_path}")
    manifest_sha = _sha(manifest_path)
    if str(getattr(args, "expected_p1q_manifest", "") or ""):
        if manifest_sha != str(args.expected_p1q_manifest):
            raise RuntimeError("FIT2_V4_P1Q_MANIFEST_SHA_DRIFT")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS__FIT2_P1Q_CURRENT_AUTHORITY_V0_V7_FROZEN_FACE_POLICY":
        raise RuntimeError("FIT2_V4_P1Q_STATUS_INVALID")
    if manifest.get("teacher_truth_used") is not False:
        raise RuntimeError("FIT2_V4_P1Q_TEACHER_TRUTH_FORBIDDEN")
    if manifest.get("source_component_truth_used") is not False:
        raise RuntimeError("FIT2_V4_P1Q_SOURCE_COMPONENT_TRUTH_FORBIDDEN")
    if manifest.get("current_gsa_lineage_hash") != surface.geometry_lineage_hash:
        raise RuntimeError("FIT2_V4_P1Q_SURFACE_LINEAGE_DRIFT")
    if manifest.get("current_skeleton_lineage_hash") != skeleton.skeleton_lineage_hash:
        raise RuntimeError("FIT2_V4_P1Q_SKELETON_LINEAGE_DRIFT")
    if manifest.get("current_skin_lineage_hash") != skin.skin_lineage_hash:
        raise RuntimeError("FIT2_V4_P1Q_SKIN_LINEAGE_DRIFT")

    rows = {int(row["view"]): row for row in manifest.get("views") or ()}
    if set(rows) != set(range(8)):
        raise RuntimeError("FIT2_V4_P1Q_REQUIRES_8_VIEWS")
    out = {}
    for view in range(8):
        row = rows[view]
        mesh = v5base._load_mesh(
            p1q_dir / str(row["files"]["mesh"]),
            surface,
            str(row["mesh_lineage_hash"]),
        )
        mesh_skin = legacy_p1q._load_mesh_skin(
            p1q_dir / str(row["files"]["skin"]),
            mesh.mesh_lineage_hash,
        )
        appearance = legacy_state._load_appearance(
            p1q_dir / str(row["files"]["appearance"])
        )
        if mesh_skin.surface_binding_hash != surface.geometry_lineage_hash:
            raise RuntimeError(f"FIT2_V4_P1Q_MESH_SKIN_SURFACE_DRIFT_V{view}")
        if mesh_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
            raise RuntimeError(f"FIT2_V4_P1Q_MESH_SKIN_SKELETON_DRIFT_V{view}")
        if mesh_skin.skin_binding_hash != skin.skin_lineage_hash:
            raise RuntimeError(f"FIT2_V4_P1Q_MESH_SKIN_SKIN_DRIFT_V{view}")
        if appearance.mesh_binding_hash != mesh.mesh_lineage_hash:
            raise RuntimeError(f"FIT2_V4_P1Q_APPEARANCE_MESH_DRIFT_V{view}")
        out[view] = (mesh, mesh_skin, appearance)
    return manifest_path, manifest_sha, manifest, out


def _active_joint_count(surface_ids, skin) -> int:
    wanted = set(map(str, surface_ids))
    joints = set()
    for row in skin.rows:
        if str(row.surface_id) not in wanted:
            continue
        joints.update(str(jid) for jid, weight in row.influences if float(weight) > 1.0e-8)
    return len(joints)


def _verify_rigid_projected_skin(projected, *, parent_joint_id: str) -> None:
    parent = str(parent_joint_id)
    if not parent:
        raise RuntimeError("FIT2_V4_RIGID_PARENT_REQUIRED")
    for row in projected.mesh_skin.rows:
        influences = {str(jid): float(weight) for jid, weight in row.influences}
        owner = float(influences.get(parent, 0.0))
        other = float(sum(weight for jid, weight in influences.items() if jid != parent))
        if owner < 0.999 - 1.0e-12 or other > 0.001 + 1.0e-12:
            raise RuntimeError(
                f"FIT2_V4_RIGID_PROJECTED_SKIN_NOT_ONE_HOT:{projected.component_id}:"
                f"{row.canonical_mesh_vertex_id}:owner={owner}:other={other}"
            )


def _build_assembly(partition, projections_by_view, skeleton, skin, p1q_manifest_sha):
    qualified = []
    for component_id in sorted(partition.component_surface_ids):
        mechanical_class = str(partition.component_mechanical_classes[component_id])
        parent_joint = str(partition.component_parent_joint_ids[component_id])
        projected_rows = [
            next(
                row for row in projections_by_view[view].components
                if row.component_id == component_id
            )
            for view in range(8)
        ]
        if mechanical_class == "RIGID_SKINNED_COMPONENT":
            for row in projected_rows:
                _verify_rigid_projected_skin(row, parent_joint_id=parent_joint)
        mesh_hashes = tuple(row.mesh.mesh_lineage_hash for row in projected_rows)
        mesh_skin_hashes = tuple(row.mesh_skin.mesh_skin_lineage_hash for row in projected_rows)
        geometry_hash = content_sha256({
            "schema": SCHEMA + ".ComponentGeometry.v1",
            "component_partition_hash": partition.partition_hash,
            "component_id": component_id,
            "mesh_lineage_hashes": mesh_hashes,
        })
        deformer_hash = content_sha256({
            "schema": SCHEMA + ".ComponentMeshSkin.v1",
            "component_partition_hash": partition.partition_hash,
            "component_id": component_id,
            "mesh_skin_lineage_hashes": mesh_skin_hashes,
            "skin_lineage_hash": skin.skin_lineage_hash,
        })
        evidence_hash = content_sha256({
            "schema": SCHEMA + ".ComponentEvidence.v1",
            "component_partition_hash": partition.partition_hash,
            "component_id": component_id,
            "mechanical_class": mechanical_class,
            "parent_joint_id": parent_joint,
            "geometry_lineage_hash": geometry_hash,
            "skin_deformer_lineage_hash": deformer_hash,
        })
        is_rigid = mechanical_class == "RIGID_SKINNED_COMPONENT"
        body_joint_count = (
            _active_joint_count(partition.component_surface_ids[component_id], skin)
            if mechanical_class == "DEFORMABLE_COMPONENT"
            else 0
        )
        if mechanical_class == "DEFORMABLE_COMPONENT" and body_joint_count < 2:
            raise RuntimeError("FIT2_V4_BODY_MULTI_JOINT_SUPPORT_NOT_PROVEN")
        evidence = ComponentAttachmentEvidenceIR(
            component_id=component_id,
            source_provenance_refs=(
                str(partition.partition_hash),
                str(p1q_manifest_sha),
                str(skin.skin_lineage_hash),
            ),
            mechanical_class=mechanical_class,
            directional_render_membership=tuple(
                (view, component_id) for view in range(8)
            ),
            geometry_membership_refs=tuple(
                f"MECHANICAL_PARTITION:V{view}:{mesh_hashes[view]}"
                for view in range(8)
            ),
            geometry_lineage_hash=geometry_hash,
            skin_deformer_lineage_hash=deformer_hash,
            canonical_parent_joint_id=parent_joint if is_rigid else "",
            socket_id="",
            bind_state_authority_hash="",
            detachability_class="FIXED_COMPONENT",
            visible_required=True,
            one_hot_carry_verified=is_rigid,
            one_hot_carry_joint_id=parent_joint if is_rigid else "",
            visual_only_qualification_hash="",
            exclusion_reason="",
            qualification_evidence_hash=evidence_hash,
            metadata={
                "component_partition_hash": partition.partition_hash,
                "mechanical_role_component": True,
                "teacher_truth_used": False,
                "source_component_name_used": False,
                "source_rig_identity_used": False,
                "body_active_joint_count": body_joint_count,
            },
        )
        report = {
            "passed": True,
            "source_authority_verified": True,
            "classification_inferred_from_filename": False,
            "mechanical_partition_hash": partition.partition_hash,
            "teacher_truth_used": False,
        }
        if mechanical_class == "DEFORMABLE_COMPONENT":
            report["deformable_multi_joint_support"] = True
            report["active_canonical_joint_count"] = body_joint_count
        qualified.append(
            qualify_component_attachment(
                evidence,
                skeleton,
                qualification_report=report,
            )
        )

    required = tuple(sorted(partition.component_surface_ids))
    return qualify_component_assembly(
        tuple(qualified),
        skeleton,
        required_visible_component_ids=required,
        metadata={
            "component_partition_hash": partition.partition_hash,
            "surface_lineage_hash": partition.surface_binding_hash,
            "skeleton_lineage_hash": partition.skeleton_binding_hash,
            "skin_lineage_hash": partition.skin_binding_hash,
            "p1q_manifest_sha256": p1q_manifest_sha,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
            "mechanical_role_components_only": True,
        },
    )


def build_final_state(args, *, persist: bool = True):
    output_dir = Path(args.output_dir)
    if persist:
        output_dir.mkdir(parents=True, exist_ok=True)

    surface, skeleton, skin, mechanical = fit2io.build_exact_mechanical(
        Path(args.fit2_surface),
        Path(args.skeleton),
        Path(args.fit2_skin),
    )
    partition = derive_mechanical_component_partition(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    p1q_manifest_path, p1q_manifest_sha, p1q_manifest, p1q = _load_p1q_state(
        args, surface, skeleton, skin
    )

    projections_by_view = {}
    projection_rows = []
    for view in range(8):
        mesh, mesh_skin, appearance = p1q[view]
        projection = project_mesh_to_mechanical_components(
            source_mesh=mesh,
            source_mesh_skin=mesh_skin,
            source_appearance=appearance,
            partition=partition,
            mechanical=mechanical,
        )
        projections_by_view[view] = projection
        projection_rows.append({
            "view": view,
            "source_mesh_lineage_hash": projection.source_mesh_lineage_hash,
            "retained_pure_face_count": projection.qualification_report["retained_pure_face_count"],
            "cross_component_face_count": projection.qualification_report["cross_component_face_count"],
            "ambiguous_vertex_count": projection.qualification_report["ambiguous_vertex_count"],
            "component_face_counts": {
                row.component_id: len(row.mesh.faces)
                for row in projection.components
            },
        })

    assembly = _build_assembly(
        partition,
        projections_by_view,
        skeleton,
        skin,
        p1q_manifest_sha,
    )

    directions = []
    underlay_components = []
    for view in range(8):
        projected_by_id = {
            row.component_id: row
            for row in projections_by_view[view].components
        }
        components = []
        for component_id in sorted(
            projected_by_id,
            key=lambda cid: (SETUP_ORDER.get(cid, 100), cid),
        ):
            if component_id not in SETUP_ORDER:
                raise RuntimeError(f"FIT2_V4_COMPONENT_SETUP_ORDER_MISSING:{component_id}")
            row = projected_by_id[component_id]
            component = build_external_renderable_component(
                component_id=component_id,
                view_index=view,
                mesh=row.mesh,
                mesh_skin=row.mesh_skin,
                mechanical=mechanical,
                appearance=row.appearance,
                setup_order=SETUP_ORDER[component_id],
                coverage_classification=row.mesh.support_coverage_classification,
                materialization_manifest_sha256=p1q_manifest_sha,
                direct_binding_manifest_sha256=partition.partition_hash,
                metadata={
                    "product_role": (
                        "CONTINUITY_DEFORMABLE_UNDERLAY"
                        if component_id == BODY_COMPONENT_ID
                        else "MECHANICAL_RIGID_FOREGROUND"
                    ),
                    "component_partition_hash": partition.partition_hash,
                    "source_full_subject_mesh_lineage_hash": row.mesh.metadata.get(
                        "source_mesh_lineage_hash", ""
                    ),
                    "teacher_truth_used": False,
                    "source_component_truth_used": False,
                    "new_pixels_generated": False,
                },
            )
            components.append(component)
            if component_id == BODY_COMPONENT_ID:
                underlay_components.append(component)
        if len(components) != len(partition.component_surface_ids):
            raise RuntimeError(f"FIT2_V4_COMPONENT_SET_INCOMPLETE_V{view}")
        directions.append(
            build_external_directional_renderable(
                view_index=view,
                camera_binding_hash=components[0].mesh.camera_binding_hash,
                components=tuple(components),
                mechanical=mechanical,
                metadata={
                    "component_partition_hash": partition.partition_hash,
                    "runtime_texture_layout": "ONE_EXACT_SOURCE_OBSERVATION_PER_VIEW",
                    "teacher_truth_used": False,
                },
            )
        )
    if len(underlay_components) != 8:
        raise RuntimeError("FIT2_V4_BODY_UNDERLAY_REQUIRES_8_VIEWS")

    render_set = build_external_directional_renderable_set(
        tuple(directions),
        mechanical,
        metadata={
            "runtime_texture_layout": "ONE_EXACT_SOURCE_OBSERVATION_PER_VIEW",
            "component_partition_hash": partition.partition_hash,
            "p1q_manifest_sha256": p1q_manifest_sha,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
        },
    )
    render_set = bind_component_assembly_to_directional_renderable_set(
        render_set,
        assembly,
        skeleton,
    )

    underlays = tuple(
        qualify_continuity_underlay(
            view_index=view,
            substrate_component=underlay_components[view],
            mechanical=mechanical,
            component_assembly=assembly,
            partition_authority_sha256=partition.partition_hash,
            metadata={
                "component_partition_hash": partition.partition_hash,
                "new_pixels_generated": False,
                "topology_mutated": False,
                "weights_mutated": False,
                "scientific_mechanical_surface_relabelled": False,
                "teacher_truth_used": False,
            },
        )
        for view in range(8)
    )
    underlay_set = build_continuity_underlay_set(
        underlays,
        component_assembly=assembly,
        metadata={
            "partition_authority_sha256": partition.partition_hash,
            "component_partition_hash": partition.partition_hash,
            "teacher_truth_used": False,
        },
    )
    render_set = bind_continuity_underlay_set_to_directional_renderables(
        render_set,
        underlay_set,
        assembly,
        mechanical,
    )

    phase_motion = build_mage_historical_phase_motion(mechanical)
    motion = compile_motion_quality(phase_motion, mechanical)
    policy_hash = content_sha256({
        "schema": "RealSaS.MageFIT2BoundedDemoMechanicalPartitionPolicy.v1",
        "unseen_generalization_claimed": False,
        "fresh_fit2_mechanics": True,
        "component_partition_hash": partition.partition_hash,
        "teacher_truth_used": False,
        "historical_motion_engine_recovered": True,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
    })
    runtime_impl_hash = content_sha256({
        "runtime": "RealSaS.NativeRuntimeV4+C++17",
        "draw_order": "QUALIFICATION_OWNED_PER_FRAME_PER_VIEW",
        "solver_replay": False,
    })
    requirements = (
        CapabilityRequirement(
            "MECHANICAL_STRUCTURE", "REQUIRED",
            mechanical.mechanical_state_hash, policy_hash, ("MECHANICAL_STRUCTURE",)
        ),
        CapabilityRequirement(
            "MESH_QUALITY", "REQUIRED",
            render_set.directional_visual_state_hash, policy_hash, ("MESH_QUALITY",)
        ),
        CapabilityRequirement(
            "VISUAL_8_DIRECTION", "REQUIRED",
            render_set.directional_visual_state_hash, policy_hash, ("DIRECTIONAL_VISUAL",)
        ),
        CapabilityRequirement(
            "PRESET_MOTION", "REQUIRED",
            motion.motion_state_hash, policy_hash, ("MOTION",)
        ),
        CapabilityRequirement(
            "RUNTIME_BACKEND", "REQUIRED",
            runtime_impl_hash, policy_hash, ("RUNTIME_CONSUMPTION",)
        ),
    )
    capability = build_capability_contract(
        PROFILE,
        requirements,
        metadata={
            "generalization_claim": False,
            "fresh_fit2_mechanics": True,
            "component_partition_hash": partition.partition_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "historical_motion_engine_recovered": True,
            "teacher_truth_used": False,
        },
    )
    product = assemble_product_v3_with_external_render_support(
        mechanical,
        render_set,
        capability,
        motion,
        editable_metadata={
            "bounded_demo": True,
            "fresh_fit2_mechanics": True,
            "component_partition_hash": partition.partition_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
        },
        runtime_policy={
            "runtime_texture_layout": "ONE_EXACT_SOURCE_OBSERVATION_PER_VIEW",
            "qualification_owned_motion_bake_required": True,
            "export_solver_replay_forbidden": True,
            "mechanical_continuity_underlay_required": True,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "component_partition_hash": partition.partition_hash,
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
        "partition": partition,
        "assembly": assembly,
        "render_set": render_set,
        "underlay_set": underlay_set,
        "phase_motion": phase_motion,
        "motion": motion,
        "capability": capability,
        "product": product,
        "p1q_manifest": p1q_manifest,
        "p1q_manifest_sha256": p1q_manifest_sha,
        "p1q_manifest_path": str(p1q_manifest_path),
        "projection_rows": tuple(projection_rows),
        "source_hashes": {
            "p1q_manifest": p1q_manifest_sha,
            "component_partition": partition.partition_hash,
        },
    }

    if persist:
        _write_json(output_dir / "FIT2_MECHANICAL_COMPONENT_PARTITION.json", partition.to_dict())
        _write_json(output_dir / "FIT2_MECHANICAL_COMPONENT_ASSEMBLY.json", assembly.to_dict())
        _write_json(output_dir / "FIT2_COMPONENT_PARTITION_PROJECTION.json", {
            "schema": SCHEMA + ".Projection.v1",
            "component_partition_hash": partition.partition_hash,
            "views": projection_rows,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
        })
        _write_json(output_dir / "FIT2_DIRECTIONAL_RENDERABLE_SET_V4.json", render_set.to_dict())
        _write_json(output_dir / "FIT2_CONTINUITY_UNDERLAY_SET_V4.json", underlay_set.to_dict())
        _write_json(output_dir / "FIT2_HISTORICAL_PHASE_MOTION_STATE_V4.json", phase_motion.to_dict())
        _write_json(output_dir / "FIT2_HISTORICAL_QUALITY_MOTION_STATE_V4.json", motion.to_dict())
        _write_json(output_dir / "FIT2_CANONICAL_PUPPET_GRAPH_V3_COMPONENT_PARTITION.json", product.to_dict())
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_FIT2_MECHANICAL_COMPONENT_PARTITION_BOUND",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "component_partition_hash": partition.partition_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "continuity_underlay_set_hash": underlay_set.qualification_hash,
            "p1q_manifest_sha256": p1q_manifest_sha,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
            "foreground_owner_raster_used": False,
            "foreground_atlas_used": False,
            "product_pass_claimed": False,
        }
        manifest_path = output_dir / "FIT2_PRODUCT_STATE_V4_MANIFEST.json"
        _write_json(manifest_path, manifest)
        print("FIT2_V4_PRODUCT_STATE=" + json.dumps({
            "product_state_hash": product.product_state_hash,
            "component_partition_hash": partition.partition_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "manifest_sha256": _sha(manifest_path),
        }, sort_keys=True), flush=True)
    return state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--p1q-dir", required=True)
    p.add_argument("--fit2-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--expected-p1q-manifest", default="")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    build_final_state(parse_args(), persist=True)
