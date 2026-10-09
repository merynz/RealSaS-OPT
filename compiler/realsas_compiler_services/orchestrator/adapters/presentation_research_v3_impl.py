"""Scoped presentation V3: shared rigid attachment XY/depth with fail-closed ownership."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_motion_safety_v1 import (
    OPERATOR_ID as SAFETY_OPERATOR_ID, POLICY as SAFETY_POLICY, compile_motion_repair,
)
from compiler.realsas_compiler_services.orchestrator.adapters import presentation_research_v1 as base
from compiler.realsas_compiler_services.orchestrator.adapters import presentation_research_v2_impl as safety
from compiler.realsas_compiler_core.visual_attachment_depth_v1 import (
    OPERATOR_ID, POLICY, DEPTH_OPERATOR_ID, evaluate_slot_owned_depth, attachment_owner_frame_metrics,
)

compile_source_domains_stage = base.compile_source_domains_stage
package_stage = base.package_stage
playback_stage = base.playback_stage
S37, S42, S43, S44, S45 = base.S37, base.S42, base.S43, base.S44, base.S45
TOPOLOGY_SCHEMA, PACKAGE_SCHEMA, PLAYBACK_SCHEMA = base.TOPOLOGY_SCHEMA, base.PACKAGE_SCHEMA, base.PLAYBACK_SCHEMA


def _projection(ctx):
    return base.source_owned_visual_runtime_projection_from_dict(
        base.stage_output_payload(ctx, S42, "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"))


def _topology(ctx):
    return base.stage_output_payload(ctx, S37, TOPOLOGY_SCHEMA)


def _view_owners(binding, witness):
    return base.visual_vertex_attachment_owners(
        binding, witness["presentation_attachment_vertex_owner"])


def compile_projection_stage(ctx):
    result = safety.compile_projection_stage(ctx)
    projection_path = Path(next(o["path"] for o in result["outputs"]
                                if o.get("schema") == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"))
    projection = base.source_owned_visual_runtime_projection_from_dict(json.loads(projection_path.read_text()))
    topology = _topology(ctx)
    witness = base._load_npz(topology["witness"])
    arrays = base._load_npz({"path": projection.projection_npz_path, "sha256": projection.projection_npz_sha256})
    for view in projection.views:
        vi = int(view.view_index)
        binding = base.domain_binding_from_arrays(arrays, vi)
        camera = _source_camera(view)
        owners = _view_owners(binding, witness)
        rest_depth = base.evaluate_domain_binding(binding, visual_faces=arrays[f"view_{vi}_faces"],
            posed_mechanical_positions_xyz=witness["vertices"], camera=camera)[:, 2]
        arrays[f"view_{vi}_attachment_rest_depths"] = rest_depth
        arrays[f"view_{vi}_vertex_attachment_owner"] = owners
        for clip in projection.clips:
            p = clip.array_prefix
            fields = np.dstack((arrays[f"{p}_view_{vi}_positions"], arrays[f"{p}_view_{vi}_depths"]))
            fields = np.asarray([evaluate_slot_owned_depth(field, rest_depths=rest_depth,
                vertex_attachment_owner=owners, attachments=topology["attachments"],
                axis_positions_source=witness["axis_positions_source"],
                skin_matrices_source=witness[f"{p}_skin_matrices_source"][fi], camera=camera)
                for fi, field in enumerate(fields)])
            arrays[f"{p}_view_{vi}_depths"] = fields[:, :, 2]
    array_sha = base._save_npz(Path(projection.projection_npz_path), **arrays)
    metadata = dict(projection.metadata, operator_policy=POLICY,
                    attachment_depth_operator_id=DEPTH_OPERATOR_ID)
    projection = replace(projection, projection_npz_sha256=array_sha, metadata=metadata,
                         visual_deformation_operator_id=OPERATOR_ID,
                         visual_deformation_policy_hash=content_sha256(POLICY), projection_hash="")
    projection = replace(projection, projection_hash=base.source_owned_visual_runtime_projection_hash(projection))
    base.write_ir(projection_path, projection, authority_class="SCOPED_RESEARCH_PRESENTATION_PROJECTION")
    for output in result["outputs"]:
        if output.get("schema") == "application/x-npz": output["sha256"] = array_sha
        if output.get("schema") == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1":
            output["sha256"] = base.sha256_file(projection_path)
    result["diagnostics"].update(projection_hash=projection.projection_hash,
                                  attachment_depth_operator_id=DEPTH_OPERATOR_ID)
    return result


def _source_camera(view):
    value = dict(view.camera, resolution=int(view.source_width))
    return base.qualify_camera_v3(value, view_id=view.view_id, view_index=view.view_index)


def _dynamic(topology, witness):
    value = SimpleNamespace(dynamic_motion_hash=topology["dynamic_motion_binding_hash"], clips=[])
    for clip in topology["clips"]:
        p = clip["array_prefix"]
        value.clips.append(SimpleNamespace(clip_id=clip["clip_id"], frames=[
            SimpleNamespace(time_seconds=float(t), posed_vertex_xyz=[(f"v{i}", x) for i, x in enumerate(xyz)])
            for t, xyz in zip(witness[f"{p}_times"], witness[f"{p}_canonical_xyz"])]))
    return value


def _repair_proof(projection, arrays, topology, witness, mesh, dynamic):
    if (projection.visual_deformation_operator_id != OPERATOR_ID
            or projection.visual_deformation_policy_hash != content_sha256(POLICY)
            or projection.metadata.get("attachment_depth_operator_id") != DEPTH_OPERATOR_ID):
        raise QualificationError("SCOPED_PRESENTATION_ATTACHMENT_DEPTH_CONTRACT_DRIFT")
    baseline = dict(arrays)
    ownership_passed, xy_residual, depth_residual = True, 0.0, 0.0
    for view in projection.views:
        vi = int(view.view_index)
        source = base._load_npz(next(r["mesh"] for r in topology["views"] if r["view_index"] == vi))
        for key, source_key in (("rest_positions", "positions"), ("faces", "faces"), ("uv", "uv")):
            if not np.array_equal(arrays[f"view_{vi}_{key}"], source[source_key]):
                raise QualificationError("SCOPED_PRESENTATION_SOURCE_VISUAL_MESH_DRIFT")
        camera = _source_camera(view)
        binding = base.domain_binding_from_arrays(arrays, vi)
        rebound = base.build_domain_binding(points_source_xy=source["positions"], visual_faces=source["faces"],
            vertex_region_id=source["vertex_region_id"], seed_region_labels=source["seed_region_labels"],
            owner_face_index=source["owner_face_index"], mechanical_positions_xyz=witness["vertices"],
            mechanical_faces=witness["faces"], camera=camera)
        if any(not np.array_equal(rebound[k], binding[k]) for k in rebound):
            raise QualificationError("SCOPED_PRESENTATION_CANONICAL_ANCHOR_REBIND_DRIFT")
        owners = _view_owners(rebound, witness)
        face_owners = base.visual_face_attachment_owners(rebound, source["faces"],
                                                       witness["presentation_attachment_vertex_owner"])
        if (not np.array_equal(owners, arrays.get(f"view_{vi}_vertex_attachment_owner"))
                or not np.array_equal(face_owners, arrays[f"view_{vi}_face_attachment_owner"])):
            raise QualificationError("SCOPED_PRESENTATION_ATTACHMENT_OWNERSHIP_DRIFT")
        rest_depth = base.evaluate_domain_binding(rebound, visual_faces=source["faces"],
            posed_mechanical_positions_xyz=witness["vertices"], camera=camera)[:, 2]
        if not np.array_equal(rest_depth, arrays.get(f"view_{vi}_attachment_rest_depths")):
            raise QualificationError("SCOPED_PRESENTATION_ATTACHMENT_REST_DEPTH_DRIFT")
        for clip in projection.clips:
            p = clip.array_prefix
            canonical_depth = []
            for fi, xyz in enumerate(witness[f"{p}_canonical_xyz"]):
                harmonic = base.evaluate_domain_binding(rebound, visual_faces=source["faces"],
                    posed_mechanical_positions_xyz=xyz, camera=camera)[:, 2]
                canonical_depth.append(harmonic)
                actual = np.c_[arrays[f"{p}_view_{vi}_positions"][fi], arrays[f"{p}_view_{vi}_depths"][fi]]
                if not np.array_equal(actual[owners == 0, 2], harmonic[owners == 0]):
                    raise QualificationError("SCOPED_PRESENTATION_BODY_DEPTH_DRIFT")
                metrics = attachment_owner_frame_metrics(actual, rest_source_xy=source["positions"],
                    rest_depths=rest_depth, vertex_attachment_owner=owners, attachments=topology["attachments"],
                    axis_positions_source=witness["axis_positions_source"],
                    skin_matrices_source=witness[f"{p}_skin_matrices_source"][fi], camera=camera)
                ownership_passed &= metrics["attachment_ownership_passed"]
                xy_residual = max(xy_residual, metrics["maximum_attachment_xy_owner_residual"])
                depth_residual = max(depth_residual, metrics["maximum_attachment_depth_owner_residual"])
            baseline[f"{p}_view_{vi}_depths"] = np.asarray(canonical_depth)
    # Independently replay the original XY palette and witness-wide body repair.
    # Restoring M depth here certifies the old operator, while the separate owner
    # proof above certifies the new actual depth. Neither uses actual depth as truth.
    legacy = replace(projection, visual_deformation_operator_id=base.OPERATOR_ID,
                     visual_deformation_policy_hash=content_sha256(base.POLICY))
    proof = safety._repair_proof(legacy, baseline, topology, witness, mesh, dynamic)
    proof.update(attachment_ownership_passed=bool(ownership_passed),
                 attachment_depth_operator_id=DEPTH_OPERATOR_ID,
                 maximum_attachment_xy_owner_residual=xy_residual,
                 maximum_attachment_depth_owner_residual=depth_residual)
    return proof


def prove_presentation_stage(ctx):
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
            or projection.metadata.get("operator_policy") != POLICY
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
        "motion_safety_repair_passed", "attachment_ownership_passed")) and attachment_pass
        and parity_bad == empty == ties == overflow == flipped == edge_bad == 0)
    data = {"schema": "RealSaS.ScopedPresentationProof.v3", "status": "PASS_DEMO_ONLY" if passed else "FAIL",
        **proof, "canonical_depth_ownership_passed": ties == overflow == 0,
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
            "blockers": [] if passed else ["SCOPED_PRESENTATION_FAIL_CLOSED"]}
