from __future__ import annotations

"""Current Mage product state from compiler-owned component-first materialization.

The product path is now partition-first by construction:

    qualified Surface/Skeleton/Skin
        -> Compiler mechanical component partition
        -> component-local source-backed CDT
        -> exact current-skin binding + observation appearance
        -> typed component assembly
        -> Runtime-v4

Historical P1/P1Q full-subject meshes, source-component truth, source-owner rasters,
and teacher topology are not product inputs on this path.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.component_attachment import (
    ComponentAttachmentEvidenceIR,
    bind_component_assembly_to_directional_renderable_set,
    qualify_component_assembly,
    qualify_component_attachment,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_component_partition import (
    derive_mechanical_component_partition,
)
from compiler.realsas_compiler_core.mesh.component_materialization import (
    materialize_mechanical_component_view,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
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


SCHEMA = "RealSaS.MageFIT2.CurrentProductState.v5.component_first"
PROFILE = "MAGE_FIT2_BOUNDED_DEMO_V5_COMPONENT_FIRST"
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


def _observation_contexts(args, surface):
    camera_paths = tuple(Path(path).resolve() for path in args.cameras)
    observation_paths = tuple(Path(path).resolve() for path in args.observations)
    if len(camera_paths) != 8 or len(observation_paths) != 8:
        raise RuntimeError("FIT2_V5_COMPONENT_FIRST_REQUIRES_EXACT_8_CAMERAS_AND_OBSERVATIONS")
    if any(not path.is_file() for path in camera_paths):
        raise RuntimeError("FIT2_V5_COMPONENT_FIRST_CAMERA_MISSING")
    if any(not path.is_file() for path in observation_paths):
        raise RuntimeError("FIT2_V5_COMPONENT_FIRST_OBSERVATION_MISSING")

    expected_resolution = int(dict(surface.metadata or {}).get("resolution", 0))
    if expected_resolution <= 0:
        raise RuntimeError("FIT2_V5_COMPONENT_FIRST_SURFACE_RESOLUTION_MISSING")

    camera_hash_by_view = {view: _sha(path) for view, path in enumerate(camera_paths)}
    observation_hash_by_view = {
        view: _sha(path) for view, path in enumerate(observation_paths)
    }
    domains = {}
    atlas_hash_by_view = {}
    for view, path in enumerate(observation_paths):
        with Image.open(path) as image:
            source = image.convert("RGBA")
            if source.size != (expected_resolution, expected_resolution):
                raise RuntimeError(
                    f"FIT2_V5_COMPONENT_FIRST_OBSERVATION_RESOLUTION_DRIFT:"
                    f"V{view}:{source.size}:{expected_resolution}"
                )
            alpha = np.asarray(source, dtype=np.uint8)[..., 3] >= 8
        domains[view] = ObservationRasterDomain.from_rows(
            alpha.tolist(),
            view_index=view,
            source_alpha_sha256=observation_hash_by_view[view],
        )
        atlas_hash_by_view[view] = content_sha256(
            {
                "observation_sha256": observation_hash_by_view[view],
                "view": view,
                "authority": "EXACT_SOURCE_OBSERVATION",
            }
        )
    return (
        camera_paths,
        observation_paths,
        camera_hash_by_view,
        observation_hash_by_view,
        domains,
        atlas_hash_by_view,
    )


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


def _build_assembly(partition, projections_by_view, skeleton, skin, materialization_authority_hash):
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
                str(materialization_authority_hash),
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
            "component_materialization_authority_hash": materialization_authority_hash,
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
    (
        _camera_paths,
        _observation_paths,
        camera_hash_by_view,
        observation_hash_by_view,
        domains,
        atlas_hash_by_view,
    ) = _observation_contexts(args, surface)

    projections_by_view = {}
    projection_rows = []
    for view in range(8):
        projection = materialize_mechanical_component_view(
            surface=surface,
            skeleton=skeleton,
            skin=skin,
            mechanical=mechanical,
            partition=partition,
            view_index=view,
            camera_binding_hash=camera_hash_by_view[view],
            observation_domain=domains[view],
            observation_hash_by_view=observation_hash_by_view,
            atlas_payload_hash=atlas_hash_by_view[view],
            require_product_mesh_quality=True,
        )
        projections_by_view[view] = projection
        projection_rows.append(
            {
                "view": view,
                "materialization_hash": projection.materialization_hash,
                "union_coverage": dict(projection.union_coverage_report),
                "component_face_counts": {
                    row.component_id: len(row.mesh.faces)
                    for row in projection.components
                },
                "component_vertex_counts": {
                    row.component_id: len(row.mesh.vertices)
                    for row in projection.components
                },
                "component_mesh_lineage_hashes": {
                    row.component_id: row.mesh.mesh_lineage_hash
                    for row in projection.components
                },
                "cross_component_faces_generated": bool(
                    projection.qualification_report.get(
                        "cross_component_faces_generated", False
                    )
                ),
                "compiler_boundary_seam": dict(
                    projection.qualification_report.get(
                        "compiler_boundary_seam", {}
                    )
                ),
                "historical_full_subject_mesh_used": False,
            }
        )

    materialization_authority_hash = content_sha256(
        {
            "schema": SCHEMA + ".ComponentMaterializationAuthority.v1",
            "component_partition_hash": partition.partition_hash,
            "view_materialization_hashes": tuple(
                projections_by_view[view].materialization_hash
                for view in range(8)
            ),
            "camera_hashes": tuple(camera_hash_by_view[view] for view in range(8)),
            "observation_hashes": tuple(
                observation_hash_by_view[view] for view in range(8)
            ),
        }
    )

    assembly = _build_assembly(
        partition,
        projections_by_view,
        skeleton,
        skin,
        materialization_authority_hash,
    )

    directions = []
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
                raise RuntimeError(
                    f"FIT2_V5_COMPONENT_SETUP_ORDER_MISSING:{component_id}"
                )
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
                materialization_manifest_sha256=materialization_authority_hash,
                direct_binding_manifest_sha256=partition.partition_hash,
                metadata={
                    "product_role": (
                        "MECHANICAL_DEFORMABLE_COMPONENT"
                        if component_id == BODY_COMPONENT_ID
                        else "MECHANICAL_RIGID_COMPONENT"
                    ),
                    "component_partition_hash": partition.partition_hash,
                    "component_materialization_hash": projections_by_view[
                        view
                    ].materialization_hash,
                    "full_silhouette_substrate": False,
                    "component_local_mesh": True,
                    "historical_full_subject_mesh_used": False,
                    "teacher_truth_used": False,
                    "source_component_truth_used": False,
                    "source_owner_raster_used": False,
                    "new_pixels_generated": False,
                },
            )
            components.append(component)
        if len(components) != len(partition.component_surface_ids):
            raise RuntimeError(f"FIT2_V5_COMPONENT_SET_INCOMPLETE_V{view}")
        directions.append(
            build_external_directional_renderable(
                view_index=view,
                camera_binding_hash=camera_hash_by_view[view],
                components=tuple(components),
                mechanical=mechanical,
                metadata={
                    "component_partition_hash": partition.partition_hash,
                    "component_materialization_hash": projections_by_view[
                        view
                    ].materialization_hash,
                    "runtime_texture_layout": "ONE_EXACT_SOURCE_OBSERVATION_PER_VIEW",
                    "component_local_materialization_qualified": True,
                    "teacher_truth_used": False,
                },
            )
        )

    render_set = build_external_directional_renderable_set(
        tuple(directions),
        mechanical,
        metadata={
            "runtime_texture_layout": "ONE_EXACT_SOURCE_OBSERVATION_PER_VIEW",
            "component_partition_hash": partition.partition_hash,
            "component_materialization_authority_hash": materialization_authority_hash,
            "component_local_materialization_qualified": True,
            "historical_full_subject_mesh_used": False,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
        },
    )
    render_set = bind_component_assembly_to_directional_renderable_set(
        render_set,
        assembly,
        skeleton,
    )

    phase_motion = build_mage_historical_phase_motion(mechanical)
    motion = compile_motion_quality(phase_motion, mechanical)
    policy_hash = content_sha256(
        {
            "schema": "RealSaS.MageFIT2BoundedDemoComponentFirstPolicy.v1",
            "unseen_generalization_claimed": False,
            "fresh_fit2_mechanics": True,
            "component_partition_hash": partition.partition_hash,
            "component_materialization_authority_hash": materialization_authority_hash,
            "teacher_truth_used": False,
            "historical_motion_engine_recovered": True,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
        }
    )
    runtime_impl_hash = content_sha256(
        {
            "runtime": "RealSaS.NativeRuntimeV4+C++17",
            "draw_order": "QUALIFICATION_OWNED_PER_FRAME_PER_VIEW",
            "solver_replay": False,
        }
    )
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
            "component_materialization_authority_hash": materialization_authority_hash,
            "component_local_materialization_qualified": True,
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
            "component_materialization_authority_hash": materialization_authority_hash,
            "component_local_materialization_qualified": True,
            "historical_full_subject_mesh_used": False,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
        },
        runtime_policy={
            "runtime_texture_layout": "ONE_EXACT_SOURCE_OBSERVATION_PER_VIEW",
            "qualification_owned_motion_bake_required": True,
            "export_solver_replay_forbidden": True,
            "component_local_materialization_required": True,
            "component_materialization_authority_hash": materialization_authority_hash,
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
        "component_materializations": projections_by_view,
        "component_materialization_authority_hash": materialization_authority_hash,
        "phase_motion": phase_motion,
        "motion": motion,
        "capability": capability,
        "product": product,
        "projection_rows": tuple(projection_rows),
        "source_hashes": {
            "component_materialization": materialization_authority_hash,
            "component_partition": partition.partition_hash,
            "observations": tuple(observation_hash_by_view[view] for view in range(8)),
            "cameras": tuple(camera_hash_by_view[view] for view in range(8)),
        },
    }

    if persist:
        _write_json(output_dir / "FIT2_MECHANICAL_COMPONENT_PARTITION.json", partition.to_dict())
        _write_json(output_dir / "FIT2_MECHANICAL_COMPONENT_ASSEMBLY.json", assembly.to_dict())
        _write_json(
            output_dir / "FIT2_COMPONENT_LOCAL_MATERIALIZATION.json",
            {
                "schema": SCHEMA + ".ComponentLocalMaterialization.v1",
                "component_partition_hash": partition.partition_hash,
                "component_materialization_authority_hash": materialization_authority_hash,
                "views": projection_rows,
                "teacher_truth_used": False,
                "source_component_truth_used": False,
                "historical_full_subject_mesh_used": False,
            },
        )
        _write_json(output_dir / "FIT2_DIRECTIONAL_RENDERABLE_SET_V5.json", render_set.to_dict())
        _write_json(output_dir / "FIT2_HISTORICAL_PHASE_MOTION_STATE_V5.json", phase_motion.to_dict())
        _write_json(output_dir / "FIT2_HISTORICAL_QUALITY_MOTION_STATE_V5.json", motion.to_dict())
        _write_json(
            output_dir / "FIT2_CANONICAL_PUPPET_GRAPH_V3_COMPONENT_FIRST.json",
            product.to_dict(),
        )
        manifest = {
            "schema": SCHEMA,
            "status": "PASS__CURRENT_MAGE_FIT2_COMPONENT_FIRST_MATERIALIZATION_BOUND",
            "product_state_hash": product.product_state_hash,
            "mechanical_state_hash": mechanical.mechanical_state_hash,
            "directional_visual_state_hash": render_set.directional_visual_state_hash,
            "motion_state_hash": motion.motion_state_hash,
            "component_partition_hash": partition.partition_hash,
            "component_assembly_hash": assembly.component_assembly_hash,
            "component_materialization_authority_hash": materialization_authority_hash,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "teacher_truth_used": False,
            "source_component_truth_used": False,
            "source_owner_raster_used": False,
            "historical_full_subject_mesh_used": False,
            "foreground_atlas_used": False,
            "product_pass_claimed": False,
        }
        manifest_path = output_dir / "FIT2_PRODUCT_STATE_V5_MANIFEST.json"
        _write_json(manifest_path, manifest)
        print(
            "FIT2_V5_PRODUCT_STATE="
            + json.dumps(
                {
                    "product_state_hash": product.product_state_hash,
                    "component_partition_hash": partition.partition_hash,
                    "component_assembly_hash": assembly.component_assembly_hash,
                    "component_materialization_authority_hash": materialization_authority_hash,
                    "manifest_sha256": _sha(manifest_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs=8, required=True)
    parser.add_argument("--observations", nargs=8, required=True)
    parser.add_argument("--fit2-surface", required=True)
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--fit2-skin", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_final_state(parse_args())
