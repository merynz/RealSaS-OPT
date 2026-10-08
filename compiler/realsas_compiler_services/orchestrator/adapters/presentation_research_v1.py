"""Go-owned, downstream-only FIT presentation execution over sealed evidence.

This reduced RESEARCH graph imports exact M/G/W and replays its existing motion
witness. It never admits Stage35 product mechanics or constructs a ProductRevision.
Its outputs retain PASS_DEMO_ONLY throughout the Registry dependency chain.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.artifact_codec_v2 import qualified_camera_set_from_dict
from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    partition_source_mask_by_safe_face_adjacency_v1, build_visual_mesh_from_region_labels_v1,
    visual_mesh_semantic_hash,
)
from compiler.realsas_compiler_core.visual_domain_v2 import (
    build_domain_binding, evaluate_domain_binding,
)
from compiler.realsas_compiler_core.visual_attachment_motion_v1 import (
    ATTACHMENT_OPERATOR_ID, evaluate_attachment_motion,
)
from compiler.realsas_compiler_core.visual_motion_blend_v1 import (
    OPERATOR_ID, POLICY, build_motion_blend_coefficients, evaluate_motion_blend,
)
from compiler.realsas_compiler_core.runtime_visual_authority_v1 import (
    SourceOwnedVisualRuntimeProjectionV1IR, SourceOwnedVisualRuntimeViewV1IR,
    SourceOwnedVisualRuntimeClipV1IR, source_owned_visual_runtime_projection_hash,
    source_owned_visual_runtime_projection_from_dict,
)
from compiler.realsas_compiler_core.runtime_package_v2 import build_source_owned_visual_rss_v2_entries, write_rss_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload, write_json, write_ir, sha256_file, load_file_ref,
)
from compiler.realsas_compiler_services.orchestrator.adapters.runtime_v2 import (
    _save_npz, _numeric_mesh, _native_player, _run_native_many,
    _source_owned_visual_reference_frame, prove_visual_domain_matrix,
    _visual_mesh_motion_metrics,
)
from tools.sealed_motion_witness_replay import replay_witness, read_ref
from compiler.realsas_compiler_core.presentation_attachment_v1 import (
    qualify_target_attachments, qualify_body_motion_preset,
    visual_face_attachment_owners, visual_vertex_attachment_owners, attachment_frame0_report,
)
from compiler.realsas_compiler_core.visual_domain_v2 import domain_binding_from_arrays, presentation_condition_metrics

S37 = "37_QUALIFIED_PRESENTATION_STRUCTURE"
S42 = "42_RUNTIME_PROJECTION_AND_CAA_BINDING"
S43 = "43_RSS_MATERIALIZE_COMPACT"
S44 = "44_NATIVE_PACKAGE_OPEN_PLAYBACK"
S45 = "45_DYNAMIC_VISUAL_INTEGRITY_PROOF"
TOPOLOGY_SCHEMA = "RealSaS.ScopedPresentationDomains.v1"
PACKAGE_SCHEMA = "RealSaS.ScopedPresentationPackage.v1"
PLAYBACK_SCHEMA = "RealSaS.ScopedPresentationPlayback.v1"


def _guard(ctx):
    if (ctx.get("platform_execution_mode") != "RESEARCH"
            or ctx["ledger"].get("execution_class") != "DEMO_WITNESS"
            or ctx["stage"]["policy"].get("product_pass_authority") is not False):
        raise QualificationError("SCOPED_PRESENTATION_REQUIRES_GO_RESEARCH_DEMO_SCOPE")
    cfg = dict(ctx["run_manifest"].get("presentation_research") or {})
    if not cfg:
        raise QualificationError("SCOPED_PRESENTATION_PINNED_INPUTS_REQUIRED")
    return cfg, ctx["run_root"] / "artifacts" / ctx["stage"]["id"]


def _load_npz(ref):
    with np.load(load_file_ref(ref, json_required=False), allow_pickle=False) as data:
        return {k: np.asarray(data[k]).copy() for k in data.files}


def compile_source_domains_stage(ctx):
    cfg, root = _guard(ctx)
    promotion = json.loads((ctx["repo_root"] / "canonical/KNIGHT_AXIS541_MIRA55_CARRIER_NATIVE_PROMOTION_20261007.json").read_text())
    arrays, clips, replay = replay_witness(cfg, promotion)
    attachment_owners, attachments = qualify_target_attachments(cfg["attachments"],
        vertices=arrays["vertices"], faces=arrays["faces"],
        carrier_basis_sha256=replay["carrier_basis_sha256"], skeleton_sha256=cfg["axis"]["sha256"],
        canonical_to_raw=read_ref(cfg["alignment"])["canonical_to_raw_teacher_index_fit_only"])
    preset = qualify_body_motion_preset(cfg["motion_preset"],
        clip_hashes=[r["sha256"] for r in cfg["motion"]], target_attachments=attachments)
    frame0 = attachment_frame0_report(attachments, arrays, clips)
    arrays["presentation_attachment_vertex_owner"] = attachment_owners
    witness_path = root / "sealed_witness_replay.npz"
    witness_sha = _save_npz(witness_path, **arrays)
    mesh = _numeric_mesh(arrays)
    camera_set = qualified_camera_set_from_dict(read_ref(cfg["camera_set"]))
    handoff = read_ref(cfg["source_handoff"])
    if cfg["camera_set"]["sha256"] != handoff["files"]["qualified_camera_set.json"]["sha256"]:
        raise QualificationError("SCOPED_PRESENTATION_SOURCE_CAMERA_BINDING_DRIFT")
    if sorted(int(v["view_index"]) for v in cfg["views"]) != list(range(8)):
        raise QualificationError("SCOPED_PRESENTATION_SOURCE_MATRIX_INCOMPLETE")
    source_by_view = {int(v["view_index"]): v["source"] for v in cfg["views"]}
    views = []
    file_outputs = [{"path": str(witness_path), "sha256": witness_sha, "schema": "application/x-npz", "authority_class": "SEALED_FIT_MOTION_WITNESS_REPLAY"}]
    for camera in camera_set.cameras:
        vi = int(camera.view_index)
        ref = source_by_view[vi]
        if ref["sha256"] != handoff["files"][f"observations/V{vi}.png"]["sha256"]:
            raise QualificationError("SCOPED_PRESENTATION_SOURCE_ART_BINDING_DRIFT")
        texture = np.asarray(Image.open(read_ref(ref, raw=True)).convert("RGBA"))
        height, width = texture.shape[:2]
        if height != width or width != camera.resolution:
            raise QualificationError("SCOPED_PRESENTATION_SOURCE_DIMENSION_DRIFT")
        mask = texture[:, :, 3] > 0
        visibility = rasterize_visible_owner(mesh, camera, width=width, height=height, max_layers=4)
        labels, seeds, regions = partition_source_mask_by_safe_face_adjacency_v1(
            mask, visibility.owner_face_index, arrays["faces"], arrays["unsafe_faces_diagnostic_union"],
            minimum_seed_pixels=1,
        )
        value = build_visual_mesh_from_region_labels_v1(mask, labels, target_edge_px=16)
        path = root / f"V{vi}.npz"
        digest = _save_npz(path, positions=value.mesh.positions, faces=value.mesh.faces,
            uv=value.mesh.uv, vertex_region_id=value.vertex_region_id,
            face_region_id=value.face_region_id, seed_region_labels=seeds,
            owner_face_index=visibility.owner_face_index)
        file_outputs.append({"path": str(path), "sha256": digest, "schema": "application/x-npz", "authority_class": "SCOPED_SOURCE_VISUAL_DOMAIN_ARRAYS"})
        views.append({"view_index": vi, "camera": asdict(camera), "texture": ref,
            "width": width, "height": height, "mesh": {"path": str(path), "sha256": digest},
            "visual_mesh_hash": visual_mesh_semantic_hash(value.mesh),
            "region_count": len(regions), "source_foreground_pixel_count": int(mask.sum())})
    data = {"schema": TOPOLOGY_SCHEMA, "views": views, "clips": clips,
            "witness": {"path": str(witness_path), "sha256": witness_sha},
            "mechanical_mesh_binding_hash": replay["carrier_basis_sha256"],
            "dynamic_motion_binding_hash": content_sha256({"inputs": {k: cfg[k] for k in
                ("weights", "axis", "alignment", "frame_qualification", "frame_metrics", "motion")}, "replay": replay}),
            "mechanical_state_binding_hash": content_sha256({k: cfg[k]["sha256"] for k in ("weights", "axis", "alignment")}),
            "camera_set_binding_hash": camera_set.camera_set_hash,
            "source_handoff_binding_hash": cfg["source_handoff"]["sha256"],
            "replay": replay, "attachments": attachments, "motion_preset": preset,
            "attachment_frame0": frame0, "product_authority": False}
    data["topology_hash"] = content_sha256(data)
    output = write_json(root / "source_domains.json", data,
                        authority_class="SCOPED_RESEARCH_PRESENTATION_DOMAINS", schema=TOPOLOGY_SCHEMA)
    return {"status": "PASS_DEMO_ONLY", "outputs": [output, *file_outputs], "diagnostics": replay}


def compile_projection_stage(ctx):
    cfg, root = _guard(ctx)
    topology = stage_output_payload(ctx, S37, TOPOLOGY_SCHEMA)
    witness = _load_npz(topology["witness"])
    arrays, views = {}, []
    from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
    for row in topology["views"]:
        vi = row["view_index"]
        source = _load_npz(row["mesh"])
        camera = qualify_camera_v3(row["camera"], view_id=f"V{vi}", view_index=vi)
        binding = build_domain_binding(points_source_xy=source["positions"], visual_faces=source["faces"],
            vertex_region_id=source["vertex_region_id"], seed_region_labels=source["seed_region_labels"],
            owner_face_index=source["owner_face_index"], mechanical_positions_xyz=witness["vertices"],
            mechanical_faces=witness["faces"], camera=camera)
        for name in ("positions", "faces", "uv"):
            arrays[f"view_{vi}_" + ("rest_positions" if name == "positions" else name)] = source[name]
        for name, value in binding.items():
            arrays[f"view_{vi}_domain_{name}"] = value
        arrays[f"view_{vi}_face_attachment_owner"] = visual_face_attachment_owners(
            binding, source["faces"], witness["presentation_attachment_vertex_owner"])
        vertex_owners = visual_vertex_attachment_owners(binding, witness["presentation_attachment_vertex_owner"])
        blend = build_motion_blend_coefficients(binding, visual_faces=source["faces"],
                                                mechanical_weights=witness["canonical_motion_weights"])
        arrays[f"view_{vi}_motion_blend_coefficients"] = blend
        for clip in topology["clips"]:
            prefix = clip["array_prefix"]
            arrays[f"{prefix}_times"] = witness[f"{prefix}_times"]
            fields = np.asarray([evaluate_attachment_motion(
                evaluate_motion_blend(evaluate_domain_binding(binding, visual_faces=source["faces"],
                    posed_mechanical_positions_xyz=xyz, camera=camera), rest_source_xy=source["positions"],
                    coefficients=blend, axis_positions_source=witness["axis_positions_source"],
                    skin_matrices_source=witness[f"{prefix}_skin_matrices_source"][fi], camera=camera),
                rest_source_xy=source["positions"], vertex_attachment_owner=vertex_owners,
                attachments=topology["attachments"], axis_positions_source=witness["axis_positions_source"],
                skin_matrices_source=witness[f"{prefix}_skin_matrices_source"][fi], camera=camera)
                for fi, xyz in enumerate(witness[f"{prefix}_canonical_xyz"])])
            arrays[f"{prefix}_view_{vi}_positions"] = fields[:, :, :2]
            arrays[f"{prefix}_view_{vi}_depths"] = fields[:, :, 2]
        views.append(SourceOwnedVisualRuntimeViewV1IR(view_index=vi, view_id=f"V{vi}",
            camera=dict(row["camera"], resolution=int(cfg.get("render_resolution", 256))),
            source_width=row["width"], source_height=row["height"],
            texture_path=row["texture"]["path"], texture_sha256=row["texture"]["sha256"],
            visual_mesh_npz_path=row["mesh"]["path"], visual_mesh_npz_sha256=row["mesh"]["sha256"],
            visual_mesh_hash=row["visual_mesh_hash"], visual_vertex_count=len(source["positions"]),
            visual_face_count=len(source["faces"]), metadata={"qualification_scope": "RESEARCH_FIT_ONLY"}))
    array_path = root / "projection_arrays.npz"
    array_sha = _save_npz(array_path, **arrays)
    value = SourceOwnedVisualRuntimeProjectionV1IR(
        complete_puppet_binding_hash=topology["mechanical_state_binding_hash"],
        mechanical_state_binding_hash=topology["mechanical_state_binding_hash"],
        mechanical_mesh_binding_hash=topology["mechanical_mesh_binding_hash"],
        qualified_visual_presentation_binding_hash=topology["topology_hash"],
        dynamic_motion_binding_hash=topology["dynamic_motion_binding_hash"],
        appearance_asset_binding_hash=content_sha256([v.texture_sha256 for v in views]),
        appearance_qualification_binding_hash=topology["source_handoff_binding_hash"],
        camera_set_binding_hash=topology["camera_set_binding_hash"],
        visual_deformation_operator_id=OPERATOR_ID, visual_deformation_policy_hash=content_sha256(POLICY),
        projection_npz_path=str(array_path), projection_npz_sha256=array_sha, views=tuple(views),
        clips=tuple(SourceOwnedVisualRuntimeClipV1IR(**{k: c[k] for k in (
            "clip_id", "duration_seconds", "loop", "frame_count", "array_prefix")}) for c in topology["clips"]),
        projection_hash="", metadata={"product_authority": False, "qualification_scope": "RESEARCH_FIT_ONLY",
            "mechanical_mesh_render_authority": False, "runtime_generation": False, "operator_policy": POLICY,
            "target_attachments": topology["attachments"], "body_motion_preset": topology["motion_preset"],
            "attachment_frame0": topology["attachment_frame0"],
            "attachment_motion_operator_id": ATTACHMENT_OPERATOR_ID})
    value = replace(value, projection_hash=source_owned_visual_runtime_projection_hash(value))
    return {"status": "PASS_DEMO_ONLY", "outputs": [write_ir(root / "projection.json", value,
            authority_class="SCOPED_RESEARCH_PRESENTATION_PROJECTION"),
            {"path": str(array_path), "sha256": array_sha, "schema": "application/x-npz", "authority_class": "SCOPED_PROJECTION_ARRAYS"}],
            "diagnostics": {"projection_hash": value.projection_hash, "mechanical_inference_executed": False}}


def package_stage(ctx):
    cfg, root = _guard(ctx)
    projection = source_owned_visual_runtime_projection_from_dict(stage_output_payload(
        ctx, S42, "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "research_presentation.rss"
    result = write_rss_v2(path, build_source_owned_visual_rss_v2_entries(projection))
    data = {"schema": PACKAGE_SCHEMA, "projection_hash": projection.projection_hash,
            "archive": {"path": str(path), "sha256": result["archive_sha256"]}, "product_authority": False}
    return {"status": "PASS_DEMO_ONLY", "outputs": [write_json(root / "package.json", data,
            authority_class="SCOPED_RESEARCH_PRESENTATION_PACKAGE", schema=PACKAGE_SCHEMA),
            {"path": str(path), "sha256": result["archive_sha256"], "schema": "application/x-realsas-rss-v2", "authority_class": "SCOPED_NATIVE_PACKAGE"}], "diagnostics": result}


def playback_stage(ctx):
    cfg, root = _guard(ctx)
    projection = source_owned_visual_runtime_projection_from_dict(stage_output_payload(ctx, S42,
        "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"))
    package = stage_output_payload(ctx, S43, PACKAGE_SCHEMA)
    player, player_sha = _native_player(ctx)
    arrays = _load_npz({"path": projection.projection_npz_path, "sha256": projection.projection_npz_sha256})
    archive = load_file_ref(package["archive"], json_required=False)
    probes = []
    for clip in projection.clips:
        requests = [{"clip_id": clip.clip_id, "view_id": view.view_id,
                     "frame_index": int(clip.frame_count)//2} for view in projection.views]
        native = _run_native_many(player=player, package=archive, requests=requests, root=root,
                                  max_workers=2)
        for view, row in zip(projection.views, native):
            rgba, provenance, owner, stdout = row
            ref = _source_owned_visual_reference_frame(projection, arrays, clip=clip, view=view,
                                                      frame_index=int(clip.frame_count)//2)
            if (ref.unresolved_depth_tie_count or ref.fragment_overflow_count
                    or rgba.read_bytes() != ref.straight_rgba_u8.tobytes()
                    or provenance.read_bytes() != ref.provenance_code.tobytes()
                    or owner.read_bytes() != ref.owner_face_index.astype("<i4").tobytes()):
                raise QualificationError("SCOPED_PRESENTATION_NATIVE_REFERENCE_PARITY_FAILED")
            probes.append({"clip_id": clip.clip_id, "view_id": view.view_id, "frame_index": int(clip.frame_count)//2,
                           "rgba_sha256": sha256_file(rgba)})
    data = {"schema": PLAYBACK_SCHEMA, "projection_hash": projection.projection_hash,
            "archive_sha256": package["archive"]["sha256"], "native_player_sha256": player_sha,
            "native_reference_byte_parity_passed": True, "probes": probes, "product_authority": False}
    return {"status": "PASS_DEMO_ONLY", "outputs": [write_json(root / "playback.json", data,
            authority_class="SCOPED_RESEARCH_NATIVE_PLAYBACK", schema=PLAYBACK_SCHEMA)], "diagnostics": data}


def prove_presentation_stage(ctx):
    cfg, root = _guard(ctx)
    topology = stage_output_payload(ctx, S37, TOPOLOGY_SCHEMA)
    projection = source_owned_visual_runtime_projection_from_dict(stage_output_payload(ctx, S42,
        "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"))
    package = stage_output_payload(ctx, S43, PACKAGE_SCHEMA)
    playback = stage_output_payload(ctx, S44, PLAYBACK_SCHEMA)
    witness = _load_npz(topology["witness"])
    mesh = _numeric_mesh(witness)
    mesh.mesh_lineage_hash = topology["mechanical_mesh_binding_hash"]
    dynamic = SimpleNamespace(dynamic_motion_hash=topology["dynamic_motion_binding_hash"], clips=[])
    for clip in topology["clips"]:
        prefix = clip["array_prefix"]
        dynamic.clips.append(SimpleNamespace(clip_id=clip["clip_id"], frames=[
            SimpleNamespace(time_seconds=float(t), posed_vertex_xyz=[(f"v{i}", p) for i, p in enumerate(xyz)])
            for t, xyz in zip(witness[f"{prefix}_times"], witness[f"{prefix}_canonical_xyz"])]))
    arrays = _load_npz({"path": projection.projection_npz_path, "sha256": projection.projection_npz_sha256})
    if projection.metadata.get("target_attachments") != topology["attachments"]:
        raise QualificationError("SCOPED_PRESENTATION_ATTACHMENT_CONTRACT_DRIFT")
    proof = prove_visual_domain_matrix(projection, arrays, mesh=mesh, dynamic=dynamic,
                                      attachment_witness=witness)
    attachment_matrix = []
    for view in projection.views:
        vi = view.view_index
        source = _load_npz(next(r["mesh"] for r in topology["views"] if r["view_index"] == vi))
        rebuilt = build_domain_binding(points_source_xy=source["positions"], visual_faces=source["faces"],
            vertex_region_id=source["vertex_region_id"], seed_region_labels=source["seed_region_labels"],
            owner_face_index=source["owner_face_index"], mechanical_positions_xyz=witness["vertices"],
            mechanical_faces=witness["faces"], camera=qualify_camera_v3(next(r["camera"] for r in topology["views"] if r["view_index"] == vi),
                               view_id=view.view_id, view_index=vi))
        binding = domain_binding_from_arrays(arrays, vi)
        if any(not np.array_equal(rebuilt[k], binding[k]) for k in rebuilt):
            raise QualificationError("SCOPED_PRESENTATION_CANONICAL_ANCHOR_REBIND_DRIFT")
        owners = visual_face_attachment_owners(binding, arrays[f"view_{vi}_faces"],
                                               witness["presentation_attachment_vertex_owner"])
        if not np.array_equal(owners, arrays[f"view_{vi}_face_attachment_owner"]):
            raise QualificationError("SCOPED_PRESENTATION_ATTACHMENT_OWNERSHIP_DRIFT")
    player, player_sha = _native_player(ctx)
    if (playback["native_player_sha256"] != player_sha or playback["projection_hash"] != projection.projection_hash
            or playback["archive_sha256"] != package["archive"]["sha256"]):
        raise QualificationError("SCOPED_PRESENTATION_PLAYBACK_BINDING_DRIFT")
    archive = load_file_ref(package["archive"], json_required=False)
    parity_bad = empty = ties = overflow = flipped = edge_bad = 0
    matrix = []
    for clip in projection.clips:
        for fi in range(clip.frame_count):
            native = _run_native_many(player=player, package=archive,
                requests=[{"clip_id": clip.clip_id, "view_id": v.view_id, "frame_index": fi} for v in projection.views],
                root=root / "frames", max_workers=2)
            for view, (rgba, provenance, owner, stdout) in zip(projection.views, native):
                ref = _source_owned_visual_reference_frame(projection, arrays, clip=clip, view=view, frame_index=fi)
                mismatch = (rgba.read_bytes() != ref.straight_rgba_u8.tobytes()
                    or provenance.read_bytes() != ref.provenance_code.tobytes()
                    or owner.read_bytes() != ref.owner_face_index.astype("<i4").tobytes())
                parity_bad += int(mismatch)
                empty += int(not np.any(ref.straight_rgba_u8[:, :, 3]))
                ties += ref.unresolved_depth_tie_count
                overflow += ref.fragment_overflow_count
                vi = view.view_index
                metrics = _visual_mesh_motion_metrics(arrays[f"view_{vi}_rest_positions"],
                    arrays[f"{clip.array_prefix}_view_{vi}_positions"][fi], arrays[f"view_{vi}_faces"])
                flipped += metrics["flipped_triangle_count"]
                edge_bad += metrics["edge_gt_4_count"]
                face_owners = arrays[f"view_{vi}_face_attachment_owner"]
                for attachment in topology["attachments"]["attachments"]:
                    selected = face_owners == attachment["attachment_index"]
                    visible = ref.owner_face_index[ref.owner_face_index >= 0]
                    attachment_matrix.append({"attachment_id": attachment["attachment_id"],
                        "clip_id": clip.clip_id, "view_id": view.view_id, "frame_index": fi,
                        "visual_face_count": int(selected.sum()),
                        "visible_pixel_count": int(np.count_nonzero(face_owners[visible] == attachment["attachment_index"])),
                        "condition": presentation_condition_metrics(arrays[f"view_{vi}_rest_positions"],
                            arrays[f"{clip.array_prefix}_view_{vi}_positions"][fi],
                            arrays[f"view_{vi}_faces"][selected]) if selected.any() else None})
                matrix.append({"clip_id": clip.clip_id, "view_id": view.view_id, "frame_index": fi,
                    "rgba": {"path": str(rgba), "sha256": sha256_file(rgba)}, "native_reference_parity": not mismatch})
    passed = (all(proof[k] for k in ("domain_coherence_passed", "frame_view_matrix_complete", "area_condition_passed",
                                    "attachment_slot_motion_passed", "canonical_pose_palette_passed"))
              and parity_bad == empty == ties == overflow == flipped == edge_bad == 0)
    data = {"schema": "RealSaS.ScopedPresentationProof.v1", "status": "PASS_DEMO_ONLY" if passed else "FAIL",
        **proof, "canonical_depth_ownership_passed": ties == overflow == 0,
        "native_reference_byte_parity_passed": parity_bad == 0, "native_reference_mismatch_frame_views": parity_bad,
        "empty_frame_views": empty, "unresolved_depth_ties": ties, "fragment_overflow": overflow,
        "flipped_triangles": flipped, "edge_gt_4_count": edge_bad, "rendered_frames": matrix,
        "projection_hash": projection.projection_hash, "native_player_sha256": player_sha,
        "attachment_ownership_passed": True, "attachment_matrix": attachment_matrix,
        "attachment_presentation_passed": proof["attachment_slot_motion_passed"]
            and proof["domain_coherence_passed"] and parity_bad == ties == overflow == 0
            and all(row["condition"] is None or (row["condition"]["area_collapse_count"] == 0
                and row["condition"]["condition_failure_count"] == 0) for row in attachment_matrix),
        "target_attachments": topology["attachments"], "body_motion_preset": topology["motion_preset"],
        "attachment_frame0": topology["attachment_frame0"],
        "product_authority": False, "mechanics_reopened": False}
    output = write_json(root / "presentation_proof.json", data,
        authority_class="SCOPED_RESEARCH_PRESENTATION_PROOF", schema=data["schema"])
    frame_outputs = [{**r["rgba"], "schema": "application/x-rgba8", "authority_class": "SCOPED_NATIVE_PRESENTATION_FRAME"} for r in matrix]
    return {"status": data["status"], "outputs": [output, *frame_outputs], "diagnostics": {k: v for k,v in data.items() if k not in ("rendered_frames", "attachment_matrix", "frames")},
            "blockers": [] if passed else ["SCOPED_PRESENTATION_FAIL_CLOSED"]}
