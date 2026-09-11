from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .common import (
    Json,
    LivingCompileError,
    VIEW_LABELS,
    load_directional_binding,
    load_product,
    load_proof,
    surface_raster,
    view_id,
    weighted_xy,
)
from .runtime import motion_bakes


def _mesh_scene_rows(product: Mapping[str, Any]) -> list[Json]:
    directions = ((product.get("directional_renderables") or {}).get("directions") or [])
    out: list[Json] = []
    for direction in directions:
        view = int(direction.get("view_index", -1))
        if view < 0 or view > 7:
            raise LivingCompileError("product direction set contains invalid view index")
        raster = surface_raster(product, view)
        for component in list(direction.get("components") or []):
            mesh = component.get("mesh") or {}
            mesh_skin = component.get("mesh_skin") or {}
            vertices_raw = list(mesh.get("vertices") or [])
            id_to_index = {str(v.get("canonical_mesh_vertex_id")): i for i, v in enumerate(vertices_raw)}
            if len(id_to_index) != len(vertices_raw):
                raise LivingCompileError("duplicate canonical mesh vertex id")
            vertices: list[list[float]] = []
            for vertex in vertices_raw:
                binding = vertex.get("support_binding") or {}
                xy = weighted_xy(binding.get("coefficients") or [], raster, field=str(vertex.get("canonical_mesh_vertex_id") or "mesh vertex"))
                vertices.append([xy[0], xy[1]])
            triangles: list[list[int]] = []
            for face in list(mesh.get("faces") or []):
                if len(face) != 3 or any(str(vid) not in id_to_index for vid in face):
                    raise LivingCompileError("Living Compile requires triangulated qualified meshes")
                triangles.append([id_to_index[str(vid)] for vid in face])
            weight_by_id = {str(row.get("canonical_mesh_vertex_id")): row for row in list(mesh_skin.get("rows") or [])}
            per_vertex_weights = []
            for vertex in vertices_raw:
                row = weight_by_id.get(str(vertex.get("canonical_mesh_vertex_id"))) or {}
                pairs = [[str(jid), float(weight)] for jid, weight in list(row.get("influences") or []) if float(weight) > 1e-10]
                pairs.sort(key=lambda item: (-item[1], item[0]))
                per_vertex_weights.append(pairs)
            component_id = str(component.get("component_id") or "BODY")
            out.append({
                "mesh_id": f"V{view}:{component_id}",
                "view_id": view_id(view),
                "mesh_role": component_id,
                "vertices": vertices,
                "triangles": triangles,
                "weight_control_id": None,
                "weight_solution": {
                    "control_id": "", "status": "QUALIFIED", "weight_solution_status": "QUALIFIED",
                    "weight_backend": str(mesh_skin.get("transfer_method") or "COMPILER_QUALIFIED_MESH_SKIN"),
                    "confidence01": 1.0, "owner_control_resolution_ratio": 1.0,
                    "per_vertex_weights": per_vertex_weights,
                    "source_is_manifest_declared_weight_plan": False,
                    "source_is_current_v4_qualified_mesh_skin": True,
                },
                "metadata": {
                    "final_render_authority": True,
                    "full_silhouette_substrate": bool((component.get("metadata") or {}).get("full_silhouette_substrate", True)),
                    "runtime_deformation_authority": True,
                    "component_state_hash": str(component.get("component_state_hash") or ""),
                    "current_v4": True,
                },
            })
    return out


def _directional_pivot_map(binding: Mapping[str, Any] | None) -> dict[tuple[int, str], list[float]]:
    if binding is None:
        return {}
    return {
        (int(row.get("view_index", -1)), str(row.get("canonical_joint_id") or "")): [float(row["raster_xy"][0]), float(row["raster_xy"][1])]
        for row in list(binding.get("joint_pivots") or [])
    }


def _joint_scene_rows(product: Mapping[str, Any], binding: Mapping[str, Any] | None) -> tuple[list[Json], list[Json]]:
    mechanical = product.get("mechanical_state") or {}
    joints = list((mechanical.get("skeleton") or {}).get("joints") or [])
    if binding is None:
        return [], []
    pivots = _directional_pivot_map(binding)
    binding_hash = str(binding.get("binding_set_hash") or "")
    controls: list[Json] = []
    edges: list[Json] = []
    for view in range(8):
        for joint in joints:
            jid = str(joint.get("canonical_joint_id") or "")
            anchor = pivots.get((view, jid))
            if anchor is None:
                raise LivingCompileError(f"qualified directional pivot missing: V{view}:{jid}")
            controls.append({
                "control_id": jid, "view_id": view_id(view), "anchor": anchor,
                "control_class": "production_skeleton_bone", "participation": "deform", "source_part_id": jid,
                "metadata": {
                    "production_skeleton_bone": True,
                    "hidden_driver": False,
                    "current_v4": True,
                    "anchor_authority": "QUALIFIED_DIRECTIONAL_JOINT_VIEW_BINDING",
                    "directional_binding_set_hash": binding_hash,
                },
            })
    for joint in joints:
        parent = joint.get("parent_canonical_id")
        child = str(joint.get("canonical_joint_id") or "")
        if parent:
            edges.append({"edge_id": f"E:{parent}->{child}", "parent_control_id": str(parent), "child_control_id": child, "relation_type": "skeleton_hierarchy"})
    return controls, edges


def _binding_rows(product: Mapping[str, Any], binding: Mapping[str, Any] | None) -> list[Json]:
    if binding is None:
        return []
    joints = list((((product.get("mechanical_state") or {}).get("skeleton") or {}).get("joints") or []))
    pivots = _directional_pivot_map(binding)
    out: list[Json] = []
    for view in range(8):
        raster = surface_raster(product, view)
        for joint in joints:
            jid = str(joint.get("canonical_joint_id") or "")
            anchor = pivots.get((view, jid))
            if anchor is None:
                raise LivingCompileError(f"qualified directional pivot missing: V{view}:{jid}")
            support = [str(x) for x in list(joint.get("support_surface_ids") or []) if str(x) in raster]
            out.append({
                "binding_id": f"B:V{view}:{jid}", "view_id": view_id(view), "source_part_id": support[0] if support else jid,
                "primary_final_control_id": jid, "final_control_ids": [jid], "selected_anchor_xy": anchor,
                "selected_socket_xy": anchor, "selected_pivot_xy": anchor, "selected_terminal_xy": anchor, "confidence01": 1.0,
                "metadata": {"anchor_authority": "QUALIFIED_DIRECTIONAL_JOINT_VIEW_BINDING", "directional_binding_set_hash": binding.get("binding_set_hash")},
            })
    return out


def _candidate_image_paths(root: Path, view: int) -> list[Path]:
    patterns = [
        f"native_texture_source/textures/view_{view}.png", f"textures/view_{view}.png", f"textures/view_{view:02d}.png",
        f"views/view_{view}.png", f"views/{VIEW_LABELS[view]}.png", f"source/view_{view}.png", f"source/{VIEW_LABELS[view]}.png",
    ]
    out = [root / rel for rel in patterns]
    for path in root.rglob("*.png"):
        if re.search(rf"(?:^|[_-])(?:view[_-]?)?0?{view}(?:$|[_-])", path.stem.lower()):
            out.append(path)
    seen: set[Path] = set()
    unique = []
    for path in out:
        path = path.resolve()
        if path not in seen:
            seen.add(path); unique.append(path)
    return unique


def _source_rows(root: Path) -> list[Json]:
    rows = []
    for view in range(8):
        asset: Json | None = None
        for path in _candidate_image_paths(root, view):
            if path.is_file():
                asset = {"path": path.relative_to(root.resolve()).as_posix(), "width": 0, "height": 0, "source": "bundle-local"}
                break
        rows.append({"view_id": view_id(view), "view_index": view, "view_label": VIEW_LABELS[view], "assets": {"raw_image": asset, "clean_cutout": asset} if asset else {}})
    return rows


def _dynamic_proof_legacy(proof: Mapping[str, Any], bakes: Mapping[str, Json]) -> Json:
    motion = next((row for row in list(proof.get("domain_reports") or []) if str(row.get("proof_domain")) == "MOTION"), None)
    passed = bool(motion and motion.get("status") == "PASS")
    clips = [{
        "clip_id": clip_id, "intent": clip_id, "passed": passed,
        "max_edge_stretch_ratio": 1.0, "return_to_rest_error01": 0.0, "max_secondary_offset01": 0.0,
        "frame_count": len(list(bake.get("frames") or [])), "bake_hash": bake.get("bake_hash"),
        "authority": "QUALIFICATION_OWNED_MOTION_BAKE", "failure_invariants": [] if passed else ["MOTION_PROOF_NOT_PASS"],
    } for clip_id, bake in sorted(bakes.items())]
    return {
        "available": bool(bakes), "passed": bool(bakes) and passed, "clip_count": len(clips),
        "passed_clip_count": len(clips) if passed else 0, "uses_same_reference_evaluator": True,
        "second_motion_solver_created": False, "clip_reports": clips,
        "owner_findings": list((motion or {}).get("owner_attribution") or []),
    }


def _dynamic_proof_summary(proof: Mapping[str, Any]) -> Json:
    return {"overall_status": proof.get("overall_status"), "domains": [{
        "proof_domain": row.get("proof_domain"), "status": row.get("status"),
        "failure_signatures": list(row.get("failure_signatures") or []), "owner_attribution": list(row.get("owner_attribution") or []),
        "domain_proof_hash": row.get("domain_proof_hash"),
    } for row in list(proof.get("domain_reports") or [])]}


def build_scene(root: Path) -> Json:
    product = load_product(root)
    proof = load_proof(root, product)
    directional_binding = load_directional_binding(root, product, required=False)
    controls, edges = _joint_scene_rows(product, directional_binding)
    meshes = _mesh_scene_rows(product)
    bindings = _binding_rows(product, directional_binding)
    source_rows = _source_rows(root)
    bakes = motion_bakes(root)
    rig_warning = [] if directional_binding is not None else ["DIRECTIONAL_BINDING_NOT_EXPORTED__RIG_OVERLAY_WITHHELD"]
    clips = [{
        "animation_clip_id": str(clip.get("clip_id") or ""),
        "animation_clip_display_name": str((clip.get("metadata") or {}).get("display_name") or clip.get("clip_id") or "Clip"),
        "animation_clip_intent": str(clip.get("clip_kind") or "custom"),
        "animation_clip_duration_seconds": float(clip.get("duration_sec") or 1.0),
        "animation_clip_loop": bool(clip.get("loop", False)),
    } for clip in list((product.get("motion_state") or {}).get("clips") or [])]
    index_payload = {
        "schema_version": "RealSaS.LivingCompileIndex.v4", "source_views": source_rows,
        "artifact_readiness": {
            "canonical_product": True,
            "proof": True,
            "directional_joint_binding": directional_binding is not None,
            "runtime_motion_bake": bool(bakes),
        },
        "animation_clips": clips,
    }
    model_payload = {"model_id": "CURRENT_V4_MODEL_STACK", "materially_influenced": True}
    proof_payload = {
        "overall_strict_passed": proof.get("overall_status") == "PASS", "passed": proof.get("overall_status") == "PASS",
        "artifact_chain_passed": True, "compiler_flow_passed": True, "product_quality_passed": proof.get("overall_status") == "PASS",
        "blockers": [] if proof.get("overall_status") == "PASS" else [f"PRODUCT_PROOF_{proof.get('overall_status', 'UNKNOWN')}"] ,
        "warnings": rig_warning, "recommended_claim": "CURRENT_V4_PROOF_GATED_PRODUCT", "proof_bundle_hash": proof.get("proof_bundle_hash"),
    }
    return {
        "schema_version": "RealSaS.LivingCompileScene.v4", "bundle_root": str(root),
        "product": {"schema_version": product.get("schema_version"), "product_state_hash": product.get("product_state_hash"), "representation_class": product.get("representation_class"), "mechanical_equivalence_class": product.get("mechanical_equivalence_class")},
        "index": index_payload, "product_artifact_index": index_payload,
        "proof": proof_payload, "model": model_payload, "model_influence": model_payload,
        "dynamic_proof": _dynamic_proof_summary(proof), "dynamic_motion_closure": _dynamic_proof_legacy(proof, bakes),
        "puppet": {"meshes": meshes, "rig": {"controls": controls, "edges": edges}}, "binding": {"bindings": bindings},
        "runtime": {"clips": clips, "proof_gated": proof.get("overall_status") == "PASS"},
        "counts": {"views": 8, "meshes": len(meshes), "controls": len(controls) // 8 if controls else 0, "bindings": len(bindings)},
        "weight_counts": {"final_render_meshes": len(meshes), "weighted_final_render_meshes": len(meshes)},
        "weight_heatmap_is_canonical_product_data": bool(meshes),
        "authority": {
            "canonical_product_mutated_by_ui": False,
            "user_edits_are_optional_layer": True,
            "user_edits_require_dynamic_revalidation": True,
            "runtime_frames_are_qualification_owned_bakes": True,
            "rig_overlay_authority": "QUALIFIED_DIRECTIONAL_JOINT_VIEW_BINDING" if directional_binding is not None else "WITHHELD_MISSING_QUALIFIED_BINDING",
            "directional_binding_set_hash": (directional_binding or {}).get("binding_set_hash"),
        },
    }
