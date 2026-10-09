"""Connected pose intervention; Stage45 requires independent visual relations.

S37 and the sealed mechanics are reused verbatim. The legacy field is retained
as a separately verified control; it cannot certify the connected pose itself.
The reduced source-only graph currently lacks qualified contact/coverage/order
inputs. Their absence is reported as FAIL even if replay and raster parity pass.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_connected_motion_v1 import OPERATOR_ID, POLICY
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import (
    ConnectedPresentationPalette, compile_connected_palette, apply_connected_palette,
    apply_connected_attachment_slots, prove_connected_palette,
)
from compiler.realsas_compiler_core.visual_presentation_contract_v1 import source_cut_pairs, contact_residuals, relational_verdict
from compiler.realsas_compiler_core.visual_motion_safety_v1 import compile_motion_repair
from compiler.realsas_compiler_core.visual_attachment_motion_v1 import slot_rigid_transform_2d
from compiler.realsas_compiler_services.orchestrator.adapters import presentation_research_v3_impl as control

base = control.base
S37, S42, S43, S44, S45 = control.S37, control.S42, control.S43, control.S44, control.S45
TOPOLOGY_SCHEMA, PACKAGE_SCHEMA, PLAYBACK_SCHEMA = control.TOPOLOGY_SCHEMA, control.PACKAGE_SCHEMA, control.PLAYBACK_SCHEMA
compile_source_domains_stage = base.compile_source_domains_stage
package_stage = base.package_stage
playback_stage = base.playback_stage
_topology, _projection, _dynamic = control._topology, control._projection, control._dynamic
SAFETY_OPERATOR_ID, SAFETY_POLICY = control.SAFETY_OPERATOR_ID, control.SAFETY_POLICY


def _field(field, palette, rest, blend, owners, attachments):
    return apply_connected_attachment_slots(apply_connected_palette(field,
        rest_source_xy=rest, coefficients=blend, palette=palette),
        rest_source_xy=rest, vertex_attachment_owner=owners, attachments=attachments, palette=palette)


def compile_projection_stage(ctx):
    result = control.compile_projection_stage(ctx)
    path = Path(next(o["path"] for o in result["outputs"]
                     if o.get("schema") == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"))
    projection = base.source_owned_visual_runtime_projection_from_dict(json.loads(path.read_text()))
    topology = _topology(ctx)
    witness = base._load_npz(topology["witness"])
    arrays = base._load_npz({"path": projection.projection_npz_path, "sha256": projection.projection_npz_sha256})
    summary = {}
    for view in projection.views:
        vi, camera = int(view.view_index), control._source_camera(view)
        binding = base.domain_binding_from_arrays(arrays, vi)
        rest, owners = arrays[f"view_{vi}_rest_positions"], arrays[f"view_{vi}_vertex_attachment_owner"]
        blend = arrays[f"view_{vi}_motion_blend_coefficients"]
        arrays[f"view_{vi}_legacy_motion_safety_repair_domain_ids"] = arrays[f"view_{vi}_motion_safety_repair_domain_ids"].copy()
        fields = {}
        for clip in projection.clips:
            p = clip.array_prefix
            for suffix in ("positions", "motion_safety_baseline_positions"):
                arrays[f"{p}_view_{vi}_legacy_{suffix}"] = arrays[f"{p}_view_{vi}_{suffix}"].copy()
            palettes = [compile_connected_palette(axis_positions_source=witness["axis_positions_source"],
                axis_parents=witness["axis_parents"], skin_matrices_source=m, camera=camera)
                for m in witness[f"{p}_skin_matrices_source"]]
            arrays[f"{p}_view_{vi}_palette_rotations"] = np.asarray([x.rotations for x in palettes])
            arrays[f"{p}_view_{vi}_palette_translations"] = np.asarray([x.translations for x in palettes])
            fields[p] = np.asarray([_field(np.c_[rest, arrays[f"{p}_view_{vi}_depths"][fi]],
                palette, rest, blend, owners, topology["attachments"]) for fi, palette in enumerate(palettes)])
            arrays[f"{p}_view_{vi}_motion_safety_baseline_positions"] = fields[p][:, :, :2].copy()
        repair = compile_motion_repair(rest_positions=rest, visual_faces=arrays[f"view_{vi}_faces"],
            domain_id=binding["domain_id"], vertex_attachment_owner=owners, clip_fields=fields)
        arrays[f"view_{vi}_motion_safety_repair_domain_ids"] = repair["repair_domain_ids"]
        for clip in projection.clips:
            arrays[f"{clip.array_prefix}_view_{vi}_positions"] = repair["repaired_fields"][clip.array_prefix][:, :, :2]
        summary[f"V{vi}"] = {"repair_domain_ids": repair["repair_domain_ids"].tolist(),
            "selection": repair["selection_diagnostics"],
            "maximum_baseline_projection_residual_px": repair["maximum_baseline_projection_residual_px"]}
    digest = base._save_npz(Path(projection.projection_npz_path), **arrays)
    metadata = dict(projection.metadata, operator_policy=POLICY, motion_safety_repair_summary=summary,
        relational_qualification_scope="SOURCE_ONLY__CONTACT_COVERAGE_AND_SEMANTIC_ORDER_UNQUALIFIED")
    projection = replace(projection, projection_npz_sha256=digest, metadata=metadata,
        visual_deformation_operator_id=OPERATOR_ID, visual_deformation_policy_hash=content_sha256(POLICY), projection_hash="")
    projection = replace(projection, projection_hash=base.source_owned_visual_runtime_projection_hash(projection))
    base.write_ir(path, projection, authority_class="SCOPED_RESEARCH_PRESENTATION_PROJECTION")
    for output in result["outputs"]:
        if output.get("schema") == "application/x-npz": output["sha256"] = digest
        if output.get("schema") == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1": output["sha256"] = base.sha256_file(path)
    result["diagnostics"].update(projection_hash=projection.projection_hash, connected_pose_operator_id=OPERATOR_ID)
    return result


def _repair_proof(projection, arrays, topology, witness, mesh, dynamic):
    if projection.visual_deformation_operator_id not in (OPERATOR_ID, control.OPERATOR_ID):
        raise QualificationError("SCOPED_PRESENTATION_RELATIONAL_OPERATOR_UNSUPPORTED")
    is_connected = projection.visual_deformation_operator_id == OPERATOR_ID
    expected_policy = POLICY if is_connected else control.POLICY
    if projection.visual_deformation_policy_hash != content_sha256(expected_policy):
        raise QualificationError("SCOPED_PRESENTATION_CONNECTED_POLICY_DRIFT")
    legacy = dict(arrays)
    if is_connected:
        for view in projection.views:
            vi = int(view.view_index)
            legacy[f"view_{vi}_motion_safety_repair_domain_ids"] = arrays[f"view_{vi}_legacy_motion_safety_repair_domain_ids"]
            for clip in projection.clips:
                for suffix in ("positions", "motion_safety_baseline_positions"):
                    legacy[f"{clip.array_prefix}_view_{vi}_{suffix}"] = arrays[f"{clip.array_prefix}_view_{vi}_legacy_{suffix}"]
    legacy_projection = replace(projection, visual_deformation_operator_id=control.OPERATOR_ID,
        visual_deformation_policy_hash=content_sha256(control.POLICY))
    proof = control._repair_proof(legacy_projection, legacy, topology, witness, mesh, dynamic)
    relation_bad = area_bad = condition_bad = 0
    max_relation = max_cut = max_loop = max_prop = max_setup = 0.0
    rows = []
    for view in projection.views:
        vi, camera = int(view.view_index), control._source_camera(view)
        binding = base.domain_binding_from_arrays(arrays, vi)
        rest, owners = arrays[f"view_{vi}_rest_positions"], arrays[f"view_{vi}_vertex_attachment_owner"]
        blend = arrays[f"view_{vi}_motion_blend_coefficients"]
        pairs = source_cut_pairs(rest, binding["domain_id"], owners)
        identity = compile_connected_palette(axis_positions_source=witness["axis_positions_source"],
            axis_parents=witness["axis_parents"], skin_matrices_source=np.tile(np.eye(4), (len(witness["axis_parents"]), 1, 1)), camera=camera)
        setup = _field(np.c_[rest, arrays[f"view_{vi}_attachment_rest_depths"]], identity,
            rest, blend, owners, topology["attachments"])
        max_setup = max(max_setup, float(np.max(np.abs(setup[:, :2] - rest), initial=0)))
        fields = {}
        for clip in projection.clips:
            p = clip.array_prefix
            frame_fields = []
            for fi, m in enumerate(witness[f"{p}_skin_matrices_source"]):
                if is_connected:
                    palette = ConnectedPresentationPalette(arrays[f"{p}_view_{vi}_palette_rotations"][fi],
                        arrays[f"{p}_view_{vi}_palette_translations"][fi])
                else:
                    transforms = [slot_rigid_transform_2d(
                        slot_rest_xyz=x, skin_matrix_source=matrix, camera=camera)
                        for x, matrix in zip(witness["axis_positions_source"], m)]
                    palette = ConnectedPresentationPalette(np.asarray([x[0] for x in transforms]), np.asarray([x[1] for x in transforms]))
                metrics = prove_connected_palette(palette, axis_positions_source=witness["axis_positions_source"],
                    axis_parents=witness["axis_parents"], skin_matrices_source=m, camera=camera)
                relation_bad += not metrics["connected_palette_relations_passed"]
                max_relation = max(max_relation, metrics["maximum_parent_child_relation_residual_px"])
                if is_connected:
                    field = _field(np.c_[rest, arrays[f"{p}_view_{vi}_depths"][fi]], palette,
                        rest, blend, owners, topology["attachments"])
                    if not np.array_equal(field[:, :2], arrays[f"{p}_view_{vi}_motion_safety_baseline_positions"][fi]):
                        raise QualificationError("SCOPED_PRESENTATION_CONNECTED_PALETTE_CONSUMPTION_DRIFT")
                    selected = owners > 0
                    max_prop = max(max_prop, float(np.max(np.abs(field[selected, :2] - arrays[f"{p}_view_{vi}_positions"][fi, selected]), initial=0)))
                    frame_fields.append(field)
                actual = arrays[f"{p}_view_{vi}_positions"][fi]
                cut = contact_residuals(actual, pairs)
                max_cut = max(max_cut, cut["maximum_contact_residual_px"])
                condition = base.presentation_condition_metrics(rest, actual, arrays[f"view_{vi}_faces"])
                area_bad += condition["area_collapse_count"]
                condition_bad += condition["condition_failure_count"]
                rows.append({"clip_id": clip.clip_id, "view_id": view.view_id, "frame_index": fi,
                    "palette_relations": metrics, "source_cut_candidates": cut,
                    "source_cut_contacts_qualified": False})
            if is_connected: fields[p] = np.asarray(frame_fields)
            if clip.loop:
                max_loop = max(max_loop, float(np.max(np.linalg.norm(arrays[f"{p}_view_{vi}_positions"][0] -
                    arrays[f"{p}_view_{vi}_positions"][-1], axis=1), initial=0)))
        if is_connected:
            repair = compile_motion_repair(rest_positions=rest, visual_faces=arrays[f"view_{vi}_faces"],
                domain_id=binding["domain_id"], vertex_attachment_owner=owners, clip_fields=fields)
            if not np.array_equal(repair["repair_domain_ids"], arrays[f"view_{vi}_motion_safety_repair_domain_ids"]):
                raise QualificationError("SCOPED_PRESENTATION_CONNECTED_REPAIR_SELECTION_DRIFT")
            for clip in projection.clips:
                if not np.array_equal(repair["repaired_fields"][clip.array_prefix][:, :, :2], arrays[f"{clip.array_prefix}_view_{vi}_positions"]):
                    raise QualificationError("SCOPED_PRESENTATION_CONNECTED_REPAIR_FIELD_DRIFT")
    proof.update(connected_palette_relations_passed=relation_bad == 0,
        connected_palette_relation_failed_frame_views=relation_bad,
        maximum_parent_child_relation_residual_source_px=max_relation,
        maximum_unqualified_source_cut_residual_source_px=max_cut,
        maximum_loop_endpoint_displacement_source_px=max_loop,
        maximum_setup_identity_residual_source_px=max_setup,
        maximum_attachment_xy_owner_residual=max_prop if is_connected else proof["maximum_attachment_xy_owner_residual"],
        motion_safety_repair_passed=area_bad == condition_bad == 0,
        area_condition_passed=area_bad == condition_bad == 0,
        area_collapse_count=area_bad, condition_failure_count=condition_bad,
        attachment_slot_motion_passed=proof["attachment_slot_motion_passed"] and max_prop <= 1e-7,
        qualified_contact_relations_passed=None, qualified_dynamic_coverage_passed=None,
        semantic_occlusion_passed=None, setup_identity_passed=max_setup <= 1e-7,
        frame0_palette_relations_passed=relation_bad == 0,
        frame0_relations_passed=None, temporal_relations_passed=None,
        missing_qualification_inputs=["CANONICAL_OR_AUTHORED_VISUAL_CONTACT_SUPPORT",
            "ADMITTED_APPEARANCE_AND_DYNAMIC_EXPOSURE", "SEMANTIC_OVERLAP_INTERVALS_AND_TRANSITIONS"],
        appearance_consumer_scope="SOURCE_RASTER_DIRECT_ONLY__NO_CAA_COMPLETION_CONSUMED",
        relational_frames=rows)
    proof.update(relational_verdict(proof))
    return proof


def prove_presentation_stage(ctx):
    return _prove_presentation_stage(ctx)


def _prove_presentation_stage(ctx):
    cfg, root = base._guard(ctx)
    topology, projection = _topology(ctx), _projection(ctx)
    package = base.stage_output_payload(ctx, S43, PACKAGE_SCHEMA)
    playback = base.stage_output_payload(ctx, S44, PLAYBACK_SCHEMA)
    witness = base._load_npz(topology["witness"])
    mesh = base._numeric_mesh(witness)
    mesh.mesh_lineage_hash = topology["mechanical_mesh_binding_hash"]
    dynamic = _dynamic(topology, witness)
    arrays = base._load_npz({"path": projection.projection_npz_path, "sha256": projection.projection_npz_sha256})
    if (projection.metadata.get("target_attachments") != topology["attachments"]
            or projection.metadata.get("body_motion_preset") != topology["motion_preset"]
            or projection.metadata.get("attachment_frame0") != topology["attachment_frame0"]
            or projection.metadata.get("operator_policy") != (POLICY if projection.visual_deformation_operator_id == OPERATOR_ID else control.POLICY)
            or projection.metadata.get("presentation_post_operator_id") != SAFETY_OPERATOR_ID
            or projection.metadata.get("presentation_post_operator_policy") != SAFETY_POLICY
            or projection.metadata.get("presentation_post_operator_policy_hash") != content_sha256(SAFETY_POLICY)):
        raise QualificationError("SCOPED_PRESENTATION_SAFETY_CONTRACT_DRIFT")
    proof = _repair_proof(projection, arrays, topology, witness, mesh, dynamic)
    player, player_sha = base._native_player(ctx)
    if (playback["native_player_sha256"] != player_sha or playback["projection_hash"] != projection.projection_hash
            or playback["archive_sha256"] != package["archive"]["sha256"]):
        raise QualificationError("SCOPED_PRESENTATION_PLAYBACK_BINDING_DRIFT")
    archive = base.load_file_ref(package["archive"], json_required=False)
    parity_bad = empty = ties = overflow = flipped = edge_bad = 0
    rendered, attachment = [], {a["attachment_id"]: {"visible_pixel_count": 0, "area_collapse_count": 0,
        "condition_failure_count": 0} for a in topology["attachments"]["attachments"]}
    for clip in projection.clips:
        for fi in range(int(clip.frame_count)):
            native = base._run_native_many(player=player, package=archive,
                requests=[{"clip_id": clip.clip_id, "view_id": v.view_id, "frame_index": fi}
                          for v in projection.views], root=root / "frames", max_workers=2)
            for view, (rgba, provenance, owner, stdout) in zip(projection.views, native):
                ref = base._source_owned_visual_reference_frame(projection, arrays, clip=clip, view=view, frame_index=fi)
                mismatch = (rgba.read_bytes() != ref.straight_rgba_u8.tobytes()
                    or provenance.read_bytes() != ref.provenance_code.tobytes()
                    or owner.read_bytes() != ref.owner_face_index.astype("<i4").tobytes())
                parity_bad += int(mismatch)
                empty += int(not np.any(ref.straight_rgba_u8[:, :, 3]))
                ties += int(ref.unresolved_depth_tie_count)
                overflow += int(ref.fragment_overflow_count)
                vi = int(view.view_index)
                faces = arrays[f"view_{vi}_faces"]
                metrics = base._visual_mesh_motion_metrics(arrays[f"view_{vi}_rest_positions"],
                    arrays[f"{clip.array_prefix}_view_{vi}_positions"][fi], faces)
                flipped += int(metrics["flipped_triangle_count"])
                edge_bad += int(metrics["edge_gt_4_count"])
                face_owners = arrays[f"view_{vi}_face_attachment_owner"]
                visible = ref.owner_face_index[ref.owner_face_index >= 0]
                for a in topology["attachments"]["attachments"]:
                    aid, index = a["attachment_id"], a["attachment_index"]
                    selected = face_owners == index
                    attachment[aid]["visible_pixel_count"] += int(np.count_nonzero(face_owners[visible] == index))
                    if selected.any():
                        m = base.presentation_condition_metrics(arrays[f"view_{vi}_rest_positions"],
                            arrays[f"{clip.array_prefix}_view_{vi}_positions"][fi], faces[selected])
                        attachment[aid]["area_collapse_count"] += int(m["area_collapse_count"])
                        attachment[aid]["condition_failure_count"] += int(m["condition_failure_count"])
                rendered.append({"clip_id": clip.clip_id, "view_id": view.view_id, "frame_index": fi,
                                 "rgba": {"path": str(rgba), "sha256": base.sha256_file(rgba)},
                                 "native_reference_parity": not mismatch})
    attachment_pass = proof["attachment_ownership_passed"] and all(v["area_collapse_count"] == v["condition_failure_count"] == 0
                          for v in attachment.values())
    passed = (all(proof[k] for k in ("domain_coherence_passed", "frame_view_matrix_complete",
        "area_condition_passed", "attachment_slot_motion_passed", "canonical_pose_palette_passed",
        "motion_safety_repair_passed", "attachment_ownership_passed", "relational_presentation_passed")) and attachment_pass
        and parity_bad == empty == ties == overflow == flipped == edge_bad == 0)
    data = {"schema": "RealSaS.ScopedPresentationProof.v4", "status": "PASS_DEMO_ONLY" if passed else "FAIL",
        **proof, "canonical_depth_ownership_passed": bool(proof["semantic_occlusion_passed"]) and ties == overflow == 0,
        "native_reference_byte_parity_passed": parity_bad == 0, "native_reference_mismatch_frame_views": parity_bad,
        "empty_frame_views": empty, "unresolved_depth_ties": ties, "fragment_overflow": overflow,
        "flipped_triangles": flipped, "edge_gt_4_count": edge_bad, "rendered_frames": rendered,
        "projection_hash": projection.projection_hash, "native_player_sha256": player_sha,
        "attachment_presentation_passed": attachment_pass,
        "attachment_summary": attachment, "target_attachments": topology["attachments"],
        "body_motion_preset": topology["motion_preset"], "attachment_frame0": topology["attachment_frame0"],
        "product_authority": False, "mechanics_reopened": False}
    output = base.write_json(root / "presentation_proof.json", data,
        authority_class="SCOPED_RESEARCH_PRESENTATION_PROOF", schema=data["schema"])
    frame_outputs = [{**r["rgba"], "schema": "application/x-rgba8",
                      "authority_class": "SCOPED_NATIVE_PRESENTATION_FRAME"} for r in rendered]
    return {"status": data["status"], "outputs": [output, *frame_outputs],
            "diagnostics": {k: v for k, v in data.items() if k not in ("rendered_frames", "frames")},
            "blockers": [] if passed else ["SCOPED_PRESENTATION_FAIL_CLOSED", *proof["relational_presentation_blockers"]]}
