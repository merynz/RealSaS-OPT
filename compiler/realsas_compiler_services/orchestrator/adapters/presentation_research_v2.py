"""Scoped presentation V2: bounded body-domain SE(2) safety repair over V1 motion."""
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
    """Compile V1 field first, then freeze one witness-wide body safety mask per view."""
    result = base.compile_projection_stage(ctx)
    projection_path = Path(next(o["path"] for o in result["outputs"]
                                if o.get("schema") == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"))
    projection = base.source_owned_visual_runtime_projection_from_dict(json.loads(projection_path.read_text()))
    topology = _topology(ctx)
    witness = base._load_npz(topology["witness"])
    arrays = base._load_npz({"path": projection.projection_npz_path,
                             "sha256": projection.projection_npz_sha256})
    summary = {}
    for view in projection.views:
        vi = int(view.view_index)
        binding = base.domain_binding_from_arrays(arrays, vi)
        owners = _view_owners(binding, witness)
        clip_fields = {}
        for clip in projection.clips:
            p = clip.array_prefix
            xyz = np.dstack((arrays[f"{p}_view_{vi}_positions"], arrays[f"{p}_view_{vi}_depths"]))
            arrays[f"{p}_view_{vi}_motion_safety_baseline_positions"] = xyz[:, :, :2].copy()
            clip_fields[p] = xyz
        repair = compile_motion_repair(
            rest_positions=arrays[f"view_{vi}_rest_positions"],
            visual_faces=arrays[f"view_{vi}_faces"], domain_id=binding["domain_id"],
            vertex_attachment_owner=owners, clip_fields=clip_fields)
        arrays[f"view_{vi}_motion_safety_repair_domain_ids"] = repair["repair_domain_ids"]
        for clip in projection.clips:
            p = clip.array_prefix
            arrays[f"{p}_view_{vi}_positions"] = repair["repaired_fields"][p][:, :, :2]
        summary[f"V{vi}"] = {
            "repair_domain_ids": repair["repair_domain_ids"].tolist(),
            "repair_domain_count": int(len(repair["repair_domain_ids"])),
            "maximum_baseline_projection_residual_px": float(repair["maximum_baseline_projection_residual_px"]),
            "selection": repair["selection_diagnostics"],
        }
    array_sha = base._save_npz(Path(projection.projection_npz_path), **arrays)
    metadata = dict(projection.metadata or {})
    metadata.update({
        "presentation_post_operator_id": SAFETY_OPERATOR_ID,
        "presentation_post_operator_policy": SAFETY_POLICY,
        "presentation_post_operator_policy_hash": content_sha256(SAFETY_POLICY),
        "motion_safety_repair_summary": summary,
    })
    projection = replace(projection, projection_npz_sha256=array_sha, metadata=metadata, projection_hash="")
    projection = replace(projection, projection_hash=base.source_owned_visual_runtime_projection_hash(projection))
    base.write_ir(projection_path, projection, authority_class="SCOPED_RESEARCH_PRESENTATION_PROJECTION")
    for output in result["outputs"]:
        if output.get("schema") == "application/x-npz": output["sha256"] = array_sha
        if output.get("schema") == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1":
            output["sha256"] = base.sha256_file(projection_path)
    result["diagnostics"].update({"projection_hash": projection.projection_hash,
                                  "presentation_post_operator_id": SAFETY_OPERATOR_ID,
                                  "motion_safety_repair_summary": summary})
    return result


def _dynamic(topology, witness):
    value = SimpleNamespace(dynamic_motion_hash=topology["dynamic_motion_binding_hash"], clips=[])
    for clip in topology["clips"]:
        p = clip["array_prefix"]
        value.clips.append(SimpleNamespace(clip_id=clip["clip_id"], frames=[
            SimpleNamespace(time_seconds=float(t), posed_vertex_xyz=[(f"v{i}", x) for i, x in enumerate(xyz)])
            for t, xyz in zip(witness[f"{p}_times"], witness[f"{p}_canonical_xyz"])]))
    return value


def _repair_proof(projection, arrays, topology, witness, mesh, dynamic):
    baseline = dict(arrays)
    repair_summary, area_bad, condition_bad = {}, 0, 0
    for view in projection.views:
        vi = int(view.view_index)
        binding = base.domain_binding_from_arrays(arrays, vi)
        owners = _view_owners(binding, witness)
        clip_fields = {}
        for clip in projection.clips:
            p = clip.array_prefix
            key = f"{p}_view_{vi}_motion_safety_baseline_positions"
            if key not in arrays:
                raise QualificationError("SCOPED_PRESENTATION_SAFETY_BASELINE_MISSING")
            baseline[f"{p}_view_{vi}_positions"] = arrays[key]
            clip_fields[p] = np.dstack((arrays[key], arrays[f"{p}_view_{vi}_depths"]))
        repair = compile_motion_repair(
            rest_positions=arrays[f"view_{vi}_rest_positions"], visual_faces=arrays[f"view_{vi}_faces"],
            domain_id=binding["domain_id"], vertex_attachment_owner=owners, clip_fields=clip_fields)
        stored_ids = arrays.get(f"view_{vi}_motion_safety_repair_domain_ids")
        if stored_ids is None or not np.array_equal(stored_ids, repair["repair_domain_ids"]):
            raise QualificationError("SCOPED_PRESENTATION_SAFETY_SELECTION_DRIFT")
        for clip in projection.clips:
            p = clip.array_prefix
            expected = repair["repaired_fields"][p][:, :, :2]
            actual = arrays[f"{p}_view_{vi}_positions"]
            if not np.array_equal(actual, expected):
                raise QualificationError("SCOPED_PRESENTATION_SAFETY_FIELD_DRIFT")
            for frame in actual:
                m = base.presentation_condition_metrics(arrays[f"view_{vi}_rest_positions"], frame,
                                                        arrays[f"view_{vi}_faces"])
                area_bad += int(m["area_collapse_count"])
                condition_bad += int(m["condition_failure_count"])
        repair_summary[f"V{vi}"] = {"repair_domain_ids": repair["repair_domain_ids"].tolist(),
                                      "repair_domain_count": int(len(repair["repair_domain_ids"]))}
    base_proof = base.prove_visual_domain_matrix(projection, baseline, mesh=mesh, dynamic=dynamic,
                                                 attachment_witness=witness)
    base_proof.update({
        "motion_safety_repair_passed": area_bad == 0 and condition_bad == 0,
        "motion_safety_repair_operator_id": SAFETY_OPERATOR_ID,
        "motion_safety_repair_policy_hash": content_sha256(SAFETY_POLICY),
        "motion_safety_repair_summary": repair_summary,
        "area_condition_passed": area_bad == 0 and condition_bad == 0,
        "area_collapse_count": area_bad, "condition_failure_count": condition_bad,
    })
    return base_proof


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
            or projection.metadata.get("presentation_post_operator_id") != SAFETY_OPERATOR_ID
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
    attachment_pass = all(v["area_collapse_count"] == v["condition_failure_count"] == 0
                          for v in attachment.values())
    passed = (all(proof[k] for k in ("domain_coherence_passed", "frame_view_matrix_complete",
        "area_condition_passed", "attachment_slot_motion_passed", "canonical_pose_palette_passed",
        "motion_safety_repair_passed")) and attachment_pass
        and parity_bad == empty == ties == overflow == flipped == edge_bad == 0)
    data = {"schema": "RealSaS.ScopedPresentationProof.v2", "status": "PASS_DEMO_ONLY" if passed else "FAIL",
        **proof, "canonical_depth_ownership_passed": ties == overflow == 0,
        "native_reference_byte_parity_passed": parity_bad == 0, "native_reference_mismatch_frame_views": parity_bad,
        "empty_frame_views": empty, "unresolved_depth_ties": ties, "fragment_overflow": overflow,
        "flipped_triangles": flipped, "edge_gt_4_count": edge_bad, "rendered_frames": rendered,
        "projection_hash": projection.projection_hash, "native_player_sha256": player_sha,
        "attachment_ownership_passed": True, "attachment_presentation_passed": attachment_pass,
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
