from __future__ import annotations

from dataclasses import replace
"""V2 Stage37-38 presentation and complete puppet sealing."""

from PIL import Image
import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE,
    complete_appearance_asset_from_dict,
    complete_appearance_qualification_from_dict,
)
from compiler.realsas_compiler_core.canonical_puppet_state_v1 import (
    build_canonical_puppet_state,
)
from compiler.realsas_compiler_core.output_presentation_v1 import (
    output_direction_set_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index,
    load_face_uv,
    load_provenance_atlas,
)
from compiler.realsas_compiler_core.presentation_partition_v2 import (
    PresentationPartitionEvidenceV2IR,
    build_presentation_partition_evidence,
    presentation_partition_evidence_from_dict,
    presentation_partition_evidence_hash,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    component_carrier_policy_from_dict,
    deformation_envelope_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.product_state_v2 import (
    build_caa_bound_presentation_graph,
    build_complete_puppet_state_v2,
    build_presentation_structure_v2,
    presentation_structure_v2_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    resolved_path,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    build_visual_mesh_from_region_labels_v1,
    partition_source_mask_by_safe_face_adjacency_v1,
    visual_mesh_semantic_hash,
    visual_mesh_set_from_dict,
)
from compiler.realsas_compiler_core.visual_presentation_v1 import (
    QualifiedVisualPresentationSetIR,
    QualifiedVisualPresentationViewIR,
    qualified_visual_presentation_set_from_dict,
    qualified_visual_presentation_set_hash,
    qualified_visual_presentation_view_hash,
)
from compiler.realsas_compiler_core.visibility_v2 import (
    rasterize_visible_owner,
)
from compiler.realsas_compiler_core.types import QualificationError


def _require_caa_final_mesh_candidate_binding(mesh, asset) -> str:
    source_candidate_hash = str(
        mesh.metadata.get("source_candidate_lineage_hash") or ""
    )
    if not source_candidate_hash:
        raise QualificationError(
            "PRESENTATION_V2_MESH_SOURCE_CANDIDATE_BINDING_MISSING"
        )
    if str(asset.candidate_mesh_binding_hash) != source_candidate_hash:
        raise QualificationError(
            "PRESENTATION_V2_CAA_MESH_CANDIDATE_BINDING_DRIFT"
        )
    return source_candidate_hash



def _qualified_visual_presentation_stage37(
    ctx: dict,
    *,
    mesh,
    asset,
    appearance,
):
    """Qualify source-owned presentation topology downstream of Stage35.

    Stage18 owns source visual substrate evidence. Stage35 owns mechanical
    qualification. Stage37 combines those already-sealed authorities into the
    final per-view presentation topology. Dynamic binding/deformation remains a
    Stage42/45 concern.
    """
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    source_visual = visual_mesh_set_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.VisualMeshSetIR.v1",
        )
    )
    compatibility = stage_output_payload(
        ctx,
        "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "RealSaS.SkinTopologyCompatibilityReport.v1",
    )
    if compatibility.get("passed") is not True:
        raise QualificationError(
            "PRESENTATION_V2_STAGE35_SKIN_TOPOLOGY_NOT_PASS"
        )
    compatibility_hash = str(compatibility.get("report_hash") or "")
    if len(compatibility_hash) != 64:
        raise QualificationError(
            "PRESENTATION_V2_STAGE35_COMPATIBILITY_HASH_INVALID"
        )
    source_candidate_hash = str(
        dict(mesh.metadata or {}).get("source_candidate_lineage_hash") or ""
    )
    if (
        source_candidate_hash
        != str(compatibility.get("candidate_lineage_hash") or "")
    ):
        raise QualificationError(
            "PRESENTATION_V2_STAGE35_CANDIDATE_BINDING_DRIFT"
        )
    if (
        str(dict(asset.metadata or {}).get("visual_mesh_set_binding_hash") or "")
        != source_visual.set_hash
    ):
        raise QualificationError(
            "PRESENTATION_V2_SOURCE_VISUAL_ASSET_BINDING_DRIFT"
        )
    if (
        source_visual.observation_set_binding_hash
        != observation.observation_set_hash
    ):
        raise QualificationError(
            "PRESENTATION_V2_SOURCE_VISUAL_OBSERVATION_DRIFT"
        )
    if appearance.asset_binding_hash != asset.asset_hash:
        raise QualificationError(
            "PRESENTATION_V2_VISUAL_APPEARANCE_BINDING_DRIFT"
        )

    observations = {
        int(row.view_index): row for row in observation.views
    }
    camera_by_view = {
        int(row.view_index): row for row in cameras.cameras
    }
    source_by_view = {
        int(row.view_index): row for row in source_visual.views
    }
    if (
        set(observations) != set(range(8))
        or set(camera_by_view) != set(range(8))
        or set(source_by_view) != set(range(8))
    ):
        raise QualificationError(
            "PRESENTATION_V2_VISUAL_REQUIRES_EXACT_V0_V7"
        )

    mask_rows = tuple(
        dict(row)
        for row in (
            dict(ctx["run_manifest"].get("observation") or {}).get(
                "source_foreground_masks"
            )
            or ()
        )
    )
    if (
        len(mask_rows) != 8
        or {int(row["view_index"]) for row in mask_rows} != set(range(8))
    ):
        raise QualificationError(
            "PRESENTATION_V2_VISUAL_FOREGROUND_MATRIX_INCOMPLETE"
        )
    mask_ref_by_view = {
        int(row["view_index"]): dict(row.get("mask") or {})
        for row in mask_rows
    }

    vertex_ids = [
        str(vertex.canonical_mesh_vertex_id) for vertex in mesh.vertices
    ]
    vertex_index = {
        vertex_id: index for index, vertex_id in enumerate(vertex_ids)
    }
    if len(vertex_index) != len(vertex_ids):
        raise QualificationError(
            "PRESENTATION_V2_VISUAL_DUPLICATE_MESH_VERTEX_ID"
        )
    mechanical_faces = np.asarray(
        [
            [vertex_index[str(vertex_id)] for vertex_id in face]
            for face in mesh.faces
        ],
        dtype=np.int64,
    )
    if (
        mechanical_faces.ndim != 2
        or mechanical_faces.shape[1] != 3
        or len(mechanical_faces) == 0
    ):
        raise QualificationError(
            "PRESENTATION_V2_VISUAL_MECHANICAL_FACES_INVALID"
        )

    unsafe = tuple(
        map(int, compatibility.get("unsafe_face_indices") or ())
    )
    if any(index < 0 or index >= len(mechanical_faces) for index in unsafe):
        raise QualificationError(
            "PRESENTATION_V2_VISUAL_UNSAFE_FACE_INDEX_DRIFT"
        )

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    visual_root = root / "qualified_visual_presentation"
    visual_root.mkdir(parents=True, exist_ok=True)
    view_rows = []
    region_total = 0
    vertex_total = 0
    face_total = 0
    for view_index in range(8):
        authority = observations[view_index]
        source_row = source_by_view[view_index]
        mask_path = load_file_ref(
            mask_ref_by_view[view_index],
            json_required=False,
        )
        mask_sha = sha256_file(mask_path)
        if (
            mask_sha != str(authority.foreground_mask_sha256)
            or mask_sha != str(source_row.source_foreground_mask_sha256)
        ):
            raise QualificationError(
                "PRESENTATION_V2_VISUAL_FOREGROUND_AUTHORITY_DRIFT"
            )
        raw = mask_path.read_bytes()
        width = int(authority.width)
        height = int(authority.height)
        if (
            len(raw) != width * height
            or any(value not in (0, 1) for value in raw)
        ):
            raise QualificationError(
                "PRESENTATION_V2_VISUAL_FOREGROUND_MASK_INVALID"
            )
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(
            height, width
        ).astype(bool)

        visibility = rasterize_visible_owner(
            mesh,
            camera_by_view[view_index],
            width=width,
            height=height,
            max_layers=4,
        )
        region_labels, seed_region_labels, region_rows = (
            partition_source_mask_by_safe_face_adjacency_v1(
                mask,
                visibility.owner_face_index,
                mechanical_faces,
                unsafe,
                minimum_seed_pixels=1,
            )
        )
        target_edge_px = int(
            dict(source_row.metadata or {}).get("target_edge_px") or 16
        )
        if target_edge_px < 1:
            raise QualificationError(
                "PRESENTATION_V2_VISUAL_TARGET_EDGE_INVALID"
            )
        region_value = build_visual_mesh_from_region_labels_v1(
            mask,
            region_labels,
            target_edge_px=target_edge_px,
        )
        visual_mesh = region_value.mesh
        region_ids = set(map(int, region_value.face_region_id.tolist()))
        if len(region_ids) != len(region_rows):
            raise QualificationError(
                "PRESENTATION_V2_VISUAL_REGION_COUNT_DRIFT"
            )
        mesh_path = visual_root / f"V{view_index}.npz"
        np.savez_compressed(
            mesh_path,
            positions=np.asarray(
                visual_mesh.positions, dtype=np.float64
            ),
            faces=np.asarray(visual_mesh.faces, dtype=np.uint32),
            uv=np.asarray(visual_mesh.uv, dtype=np.float64),
            vertex_region_id=np.asarray(
                region_value.vertex_region_id, dtype=np.int32
            ),
            face_region_id=np.asarray(
                region_value.face_region_id, dtype=np.int32
            ),
            region_labels=np.asarray(region_labels, dtype=np.int32),
            seed_region_labels=np.asarray(
                seed_region_labels, dtype=np.int32
            ),
        )
        mesh_sha = sha256_file(mesh_path)
        row = QualifiedVisualPresentationViewIR(
            view_index=view_index,
            direction_id=f"V{view_index}",
            width=width,
            height=height,
            vertex_count=int(len(visual_mesh.positions)),
            face_count=int(len(visual_mesh.faces)),
            region_count=int(len(region_rows)),
            mesh_npz_path=str(mesh_path.resolve()),
            mesh_npz_sha256=mesh_sha,
            source_visual_mesh_hash=str(source_row.mesh_hash),
            source_raster_sha256=str(source_row.source_raster_sha256),
            source_foreground_mask_sha256=str(
                source_row.source_foreground_mask_sha256
            ),
            visual_mesh_hash=visual_mesh_semantic_hash(visual_mesh),
            view_hash="",
            metadata={
                "ownership": "STAGE37_QUALIFIED_SOURCE_OWNED_PRESENTATION",
                "qualification_scope": (
                    "REST_SOURCE_TOPOLOGY_X_STAGE35_SAFE_ADJACENCY_V1"
                ),
                "dynamic_deformation_qualified": False,
                "dynamic_deformation_owner": "STAGE42_AND_STAGE45",
                "mechanical_mesh_render_authority": False,
                "source_foreground_pixel_count": int(
                    np.count_nonzero(mask)
                ),
                "safe_seed_pixel_count": int(
                    np.count_nonzero(seed_region_labels >= 0)
                ),
                "visibility_layer_overflow_pixel_count": int(
                    np.count_nonzero(visibility.layer_overflow & mask)
                ),
                "target_edge_px": target_edge_px,
            },
        )
        row = replace(
            row,
            view_hash=qualified_visual_presentation_view_hash(row),
        )
        view_rows.append(row)
        region_total += int(row.region_count)
        vertex_total += int(row.vertex_count)
        face_total += int(row.face_count)

    value = QualifiedVisualPresentationSetIR(
        source_visual_mesh_set_binding_hash=str(source_visual.set_hash),
        observation_set_binding_hash=str(observation.observation_set_hash),
        output_direction_set_binding_hash=str(
            source_visual.output_direction_set_binding_hash
        ),
        mechanical_mesh_binding_hash=str(mesh.mesh_lineage_hash),
        skin_topology_compatibility_report_hash=compatibility_hash,
        appearance_asset_binding_hash=str(asset.asset_hash),
        appearance_qualification_binding_hash=str(
            appearance.qualification_hash
        ),
        views=tuple(view_rows),
        set_hash="",
        metadata={
            "authority": "STAGE37_QUALIFIED_VISUAL_PRESENTATION_TOPOLOGY",
            "source_owned_visual_mesh_mode": True,
            "mechanical_mesh_render_authority": False,
            "dynamic_deformation_qualified": False,
            "dynamic_deformation_owner": "STAGE42_AND_STAGE45",
            "region_partition_contract": (
                "SOURCE_RASTER_4N_X_STAGE35_SAFE_SHARED_EDGE_V1"
            ),
            "minimum_seed_pixels": 1,
            "subject_specific_code_used": False,
        },
    )
    value = replace(
        value,
        set_hash=qualified_visual_presentation_set_hash(value),
    )
    return value, {
        "qualified_visual_presentation_set_hash": value.set_hash,
        "source_visual_mesh_set_hash": source_visual.set_hash,
        "visual_region_count": region_total,
        "visual_vertex_count": vertex_total,
        "visual_face_count": face_total,
        "skin_topology_compatibility_report_hash": compatibility_hash,
    }


def qualify_presentation_structure_stage(ctx: dict) -> dict:
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    mesh = qualified_mesh_from_dict(
        stage_output_payload(
            ctx,
            "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
            "RealSaS.QualifiedMeshIR.v1",
        )
    )
    mesh_skin = qualified_mesh_skin_from_dict(
        stage_output_payload(
            ctx,
            "36_QUALIFIED_MESH_SKIN_TRANSFER",
            "RealSaS.QualifiedMeshSkinIR.v1",
        )
    )
    partition = mechanical_partition_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.MechanicalPartitionIR.v1",
        )
    )
    carrier = component_carrier_policy_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.ComponentCarrierPolicyIR.v1",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    appearance = complete_appearance_qualification_from_dict(
        stage_output_payload(
            ctx,
            "24_COMPLETE_APPEARANCE_QUALIFIED",
            "RealSaS.CompleteAppearanceQualificationIR.v2",
        )
    )
    if appearance.asset_binding_hash != asset.asset_hash:
        raise QualificationError("PRESENTATION_V2_APPEARANCE_BINDING_DRIFT")
    _require_caa_final_mesh_candidate_binding(mesh, asset)
    if str(appearance.qualification_report.get("status") or "") != "PASS_COMPLETE_APPEARANCE":
        raise QualificationError("PRESENTATION_V2_APPEARANCE_NOT_QUALIFIED")

    cfg = dict(ctx["run_manifest"].get("presentation") or {})
    unknown = set(cfg) - {
        "mode",
        "policy_document",
    }
    if unknown:
        return {
            "status": "BLOCKED",
            "blockers": ["PRESENTATION_V2_CONFIG_UNSUPPORTED"],
            "diagnostics": {"unsupported_keys": sorted(unknown)},
        }
    if str(cfg.get("mode") or "AUTO_ROLE_FREE_V2") != "AUTO_ROLE_FREE_V2":
        return {
            "status": "BLOCKED",
            "blockers": ["PRESENTATION_V2_MODE_UNSUPPORTED"],
            "diagnostics": {"mode": cfg.get("mode")},
        }

    policy_ref = dict(cfg.get("policy_document") or {})
    if not policy_ref:
        return {
            "status": "BLOCKED",
            "blockers": ["PRESENTATION_V2_FROZEN_POLICY_DOCUMENT_REQUIRED"],
            "diagnostics": {},
        }
    policy = load_file_ref(
        policy_ref,
        expected_schema="RealSaS.PresentationPartitionPolicy.v1",
    )
    if not str(policy.get("status") or "").startswith("FROZEN_SUBJECT_FREE"):
        raise QualificationError("PRESENTATION_V2_POLICY_NOT_SUBJECT_FREE_FROZEN")
    forbidden = set(map(str, policy.get("forbidden_inputs") or ()))
    if not {"KNIGHT_RESULT", "SUBJECT_ID", "CATEGORY_LABEL"}.issubset(forbidden):
        raise QualificationError("PRESENTATION_V2_POLICY_FORBIDDEN_INPUTS_INCOMPLETE")
    mechanical_policy = dict(policy.get("mechanical_binding_policy") or {})
    required_mechanical = {
        "min_rigid_owner_weight",
        "max_rigid_other_mass",
        "rigidity_noop_required",
        "rigidity_noop_relative_edge_tolerance",
        "rigidity_noop_probe_rotation_degrees",
        "legacy_weight_thresholds_final_authority",
    }
    if not required_mechanical.issubset(mechanical_policy):
        raise QualificationError("PRESENTATION_V2_MECHANICAL_POLICY_INCOMPLETE")
    if bool(mechanical_policy["rigidity_noop_required"]) is not True:
        raise QualificationError("PRESENTATION_V2_RIGIDITY_NOOP_REQUIRED")
    if bool(mechanical_policy["legacy_weight_thresholds_final_authority"]):
        raise QualificationError(
            "PRESENTATION_V2_LEGACY_WEIGHT_THRESHOLD_AUTHORITY_FORBIDDEN"
        )

    if dict(asset.metadata or {}).get("source_owned_visual_mesh_mode") is True:
        visual_hash = str(
            dict(asset.metadata or {}).get("visual_mesh_set_binding_hash") or ""
        )
        if len(visual_hash) != 64:
            raise QualificationError(
                "PRESENTATION_V2_VISUAL_MESH_SET_BINDING_MISSING"
            )
        qualified_visual, visual_diagnostics = (
            _qualified_visual_presentation_stage37(
                ctx,
                mesh=mesh,
                asset=asset,
                appearance=appearance,
            )
        )
        evidence = PresentationPartitionEvidenceV2IR(
            mesh_binding_hash=str(mesh.mesh_lineage_hash),
            appearance_asset_binding_hash=str(asset.asset_hash),
            appearance_qualification_binding_hash=str(
                appearance.qualification_hash
            ),
            policy_hash=content_sha256(policy),
            evaluated_shared_edge_count=0,
            source_supported_edge_count=0,
            cut_face_pairs=(),
            boundary_measurements=(),
            evidence_hash="",
            metadata={
                "role_free": True,
                "categorical_recognition_used": False,
                "conceptual_object_identity_claimed": False,
                "evidence_supported_visual_partition": True,
                "appearance_boundary_does_not_mint_appearance": True,
                "source_owned_visual_mesh_mode": True,
                "visual_mesh_set_binding_hash": qualified_visual.set_hash,
                "source_visual_mesh_set_binding_hash": visual_hash,
                "qualified_visual_presentation_set_binding_hash": (
                    qualified_visual.set_hash
                ),
                "mechanical_face_appearance_partition_not_applicable": True,
                "mechanical_mesh_render_authority": False,
            },
        )
        evidence = replace(
            evidence,
            evidence_hash=presentation_partition_evidence_hash(evidence),
        )
        structure = build_presentation_structure_v2(
            skeleton=skeleton,
            mesh=mesh,
            mesh_skin=mesh_skin,
            partition=partition,
            carrier_policy=carrier,
            min_rigid_owner_weight=float(
                mechanical_policy["min_rigid_owner_weight"]
            ),
            max_rigid_other_mass=float(
                mechanical_policy["max_rigid_other_mass"]
            ),
            rigidity_noop_relative_edge_tolerance=float(
                mechanical_policy[
                    "rigidity_noop_relative_edge_tolerance"
                ]
            ),
            rigidity_noop_probe_rotation_degrees=float(
                mechanical_policy[
                    "rigidity_noop_probe_rotation_degrees"
                ]
            ),
            presentation_cut_face_pairs=(),
            presentation_partition_evidence_hash=evidence.evidence_hash,
        )
        structure = replace(
            structure,
            metadata={
                **dict(structure.metadata or {}),
                "source_owned_visual_mesh_mode": True,
                "visual_mesh_set_binding_hash": qualified_visual.set_hash,
                "source_visual_mesh_set_binding_hash": visual_hash,
                "qualified_visual_presentation_set_binding_hash": (
                    qualified_visual.set_hash
                ),
                "mechanical_mesh_render_authority": False,
                "visual_partition_authority": (
                    "STAGE37_QUALIFIED_SOURCE_OWNED_VISUAL_PRESENTATION"
                ),
            },
        )
        # Metadata participates in the structure hash.
        from compiler.realsas_compiler_core.product_state_v2 import (
            presentation_structure_v2_hash,
        )
        structure = replace(
            structure,
            structure_hash=presentation_structure_v2_hash(structure),
        )
        root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
        return {
            "status": "PASS",
            "outputs": [
                write_ir(
                    root / "presentation_partition_evidence_v2.json",
                    evidence,
                    authority_class=(
                        "QUALIFIED_PRESENTATION_PARTITION_EVIDENCE_V2"
                    ),
                ),
                write_ir(
                    root / "qualified_visual_presentation_set_v1.json",
                    qualified_visual,
                    authority_class=(
                        "QUALIFIED_SOURCE_OWNED_VISUAL_PRESENTATION"
                    ),
                ),
                write_ir(
                    root / "qualified_presentation_structure_v2.json",
                    structure,
                    authority_class=(
                        "QUALIFIED_PRESENTATION_STRUCTURE_V2"
                    ),
                ),
            ],
            "diagnostics": {
                "structure_hash": structure.structure_hash,
                "slot_count": len(structure.slots),
                "attachment_count": len(structure.attachments),
                "presentation_partition_evidence_hash": (
                    evidence.evidence_hash
                ),
                "evaluated_shared_edge_count": 0,
                "source_supported_edge_count": 0,
                "appearance_boundary_cut_count": 0,
                "categorical_recognition_used": False,
                "conceptual_object_identity_claimed": False,
                "appearance_authority_minted": False,
                "source_view_identity_preserved": True,
                "source_view_identity_is_render_authority": False,
                "source_owned_visual_mesh_mode": True,
                "visual_mesh_set_binding_hash": qualified_visual.set_hash,
                "source_visual_mesh_set_binding_hash": visual_hash,
                "qualified_visual_presentation_set_binding_hash": (
                    qualified_visual.set_hash
                ),
                "mechanical_mesh_render_authority": False,
                **visual_diagnostics,
            },
        }

    uv_path = resolved_path(asset.uv_npz_path)
    provenance_path = resolved_path(asset.provenance_npz_path)
    if not uv_path.is_file() or sha256_file(uv_path) != asset.uv_npz_sha256:
        raise QualificationError("PRESENTATION_V2_CAA_UV_BYTES_DRIFT")
    if (
        not provenance_path.is_file()
        or sha256_file(provenance_path) != asset.provenance_npz_sha256
    ):
        raise QualificationError("PRESENTATION_V2_CAA_PROVENANCE_BYTES_DRIFT")
    face_uv = load_face_uv(asset)
    face_page_index = load_face_page_index(asset)
    provenance = load_provenance_atlas(asset)
    with np.load(provenance_path, allow_pickle=False) as lineage_data:
        if "source_view" not in lineage_data.files:
            raise QualificationError(
                "PRESENTATION_V2_CAA_SOURCE_VIEW_LINEAGE_MISSING"
            )
        source_view = np.asarray(
            lineage_data["source_view"],
            dtype=np.int16,
        )
    if source_view.shape != provenance.shape:
        raise QualificationError(
            "PRESENTATION_V2_CAA_SOURCE_VIEW_LINEAGE_SHAPE_DRIFT"
        )
    padding = np.iinfo(np.int16).min
    direct_or_other = (
        (provenance == CAA_PROVENANCE["DIRECT_SOURCE"])
        | (provenance == CAA_PROVENANCE["OTHER_VIEW_SOURCE"])
    )
    harmonic = (
        provenance == CAA_PROVENANCE["COMPILED_LOCAL_HARMONIC"]
    )
    unsupported = (
        provenance == CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"]
    )
    canonical_global = (
        provenance == CAA_PROVENANCE["CANONICAL_GLOBAL_COMPLETION"]
    )
    padding_mask = provenance == 255
    valid_provenance = (
        direct_or_other
        | harmonic
        | unsupported
        | canonical_global
        | padding_mask
    )
    if not np.all(valid_provenance):
        raise QualificationError(
            "PRESENTATION_V2_CAA_PROVENANCE_VALUE_INVALID"
        )
    if np.any(
        direct_or_other
        & ~((source_view >= 0) & (source_view < 8))
    ):
        raise QualificationError(
            "PRESENTATION_V2_CAA_SOURCE_VIEW_IDENTITY_DRIFT"
        )
    if np.any(harmonic & (source_view != -2)):
        raise QualificationError(
            "PRESENTATION_V2_CAA_HARMONIC_LINEAGE_DRIFT"
        )
    if np.any(canonical_global & (source_view != -3)):
        raise QualificationError(
            "PRESENTATION_V2_CAA_CANONICAL_GLOBAL_LINEAGE_DRIFT"
        )
    if np.any(unsupported & (source_view != -4)):
        raise QualificationError(
            "PRESENTATION_V2_CAA_UNSUPPORTED_LINEAGE_DRIFT"
        )
    if np.any(padding_mask & (source_view != padding)):
        raise QualificationError(
            "PRESENTATION_V2_CAA_SOURCE_VIEW_PADDING_DRIFT"
        )
    textures = {}
    for row in asset.textures:
        metadata = dict(row.metadata or {})
        page_rows = tuple(metadata.get("pages") or ())
        if not page_rows:
            page_rows = ({
                "page_index": 0,
                "path": row.transport_png_path,
                "sha256": row.transport_png_sha256,
            },)
        ordered = tuple(
            sorted((dict(item) for item in page_rows), key=lambda item: int(item["page_index"]))
        )
        if tuple(int(item["page_index"]) for item in ordered) != tuple(range(len(ordered))):
            raise QualificationError("PRESENTATION_V2_CAA_TEXTURE_PAGE_INDEX_DRIFT")
        pages = []
        for item in ordered:
            path = resolved_path(str(item["path"]))
            if not path.is_file() or sha256_file(path) != str(item["sha256"]):
                raise QualificationError("PRESENTATION_V2_CAA_TEXTURE_BYTES_DRIFT")
            pages.append(np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8))
        if bool(metadata.get("paged_atlas")):
            if len({page.shape for page in pages}) != 1:
                raise QualificationError("PRESENTATION_V2_CAA_TEXTURE_PAGE_SHAPE_DRIFT")
            textures[int(row.direction_index)] = np.stack(pages, axis=0)
        else:
            if len(pages) != 1:
                raise QualificationError("PRESENTATION_V2_CAA_LEGACY_MULTIPAGE_DRIFT")
            textures[int(row.direction_index)] = pages[0]

    evidence = build_presentation_partition_evidence(
        mesh=mesh,
        appearance_asset_hash=asset.asset_hash,
        appearance_qualification_hash=appearance.qualification_hash,
        face_uv=face_uv,
        textures_by_direction=textures,
        provenance_by_direction=provenance,
        policy=policy,
        face_page_index=face_page_index,
    )
    structure = build_presentation_structure_v2(
        skeleton=skeleton,
        mesh=mesh,
        mesh_skin=mesh_skin,
        partition=partition,
        carrier_policy=carrier,
        min_rigid_owner_weight=float(mechanical_policy["min_rigid_owner_weight"]),
        max_rigid_other_mass=float(mechanical_policy["max_rigid_other_mass"]),
        rigidity_noop_relative_edge_tolerance=float(
            mechanical_policy["rigidity_noop_relative_edge_tolerance"]
        ),
        rigidity_noop_probe_rotation_degrees=float(
            mechanical_policy["rigidity_noop_probe_rotation_degrees"]
        ),
        presentation_cut_face_pairs=evidence.cut_face_pairs,
        presentation_partition_evidence_hash=evidence.evidence_hash,
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "presentation_partition_evidence_v2.json",
                evidence,
                authority_class="QUALIFIED_PRESENTATION_PARTITION_EVIDENCE_V2",
            ),
            write_ir(
                root / "qualified_presentation_structure_v2.json",
                structure,
                authority_class="QUALIFIED_PRESENTATION_STRUCTURE_V2",
            ),
        ],
        "diagnostics": {
            "structure_hash": structure.structure_hash,
            "slot_count": len(structure.slots),
            "attachment_count": len(structure.attachments),
            "presentation_partition_evidence_hash": evidence.evidence_hash,
            "evaluated_shared_edge_count": evidence.evaluated_shared_edge_count,
            "source_supported_edge_count": evidence.source_supported_edge_count,
            "appearance_boundary_cut_count": len(evidence.cut_face_pairs),
            "categorical_recognition_used": False,
            "conceptual_object_identity_claimed": False,
            "appearance_authority_minted": False,
            "source_view_identity_preserved": True,
            "source_view_identity_is_render_authority": False,
        },
    }


def seal_complete_puppet_stage(ctx: dict) -> dict:
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedSkinIR.v1"
        )
    )
    partition = mechanical_partition_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.MechanicalPartitionIR.v1",
        )
    )
    carrier = component_carrier_policy_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.ComponentCarrierPolicyIR.v1",
        )
    )
    envelope = deformation_envelope_from_dict(
        stage_output_payload(
            ctx,
            "34_DEFORMATION_CAPABILITY_ENVELOPE",
            "RealSaS.DeformationCapabilityEnvelopeIR.v1",
        )
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )
    mesh = qualified_mesh_from_dict(
        stage_output_payload(
            ctx,
            "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
            "RealSaS.QualifiedMeshIR.v1",
        )
    )
    mesh_skin = qualified_mesh_skin_from_dict(
        stage_output_payload(
            ctx,
            "36_QUALIFIED_MESH_SKIN_TRANSFER",
            "RealSaS.QualifiedMeshSkinIR.v1",
        )
    )
    structure = presentation_structure_v2_from_dict(
        stage_output_payload(
            ctx,
            "37_QUALIFIED_PRESENTATION_STRUCTURE",
            "RealSaS.QualifiedPresentationStructureIR.v2",
        )
    )
    partition_evidence = presentation_partition_evidence_from_dict(
        stage_output_payload(
            ctx,
            "37_QUALIFIED_PRESENTATION_STRUCTURE",
            "RealSaS.PresentationPartitionEvidenceIR.v2",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    appearance = complete_appearance_qualification_from_dict(
        stage_output_payload(
            ctx,
            "24_COMPLETE_APPEARANCE_QUALIFIED",
            "RealSaS.CompleteAppearanceQualificationIR.v2",
        )
    )
    directions = output_direction_set_from_dict(
        stage_output_payload(
            ctx,
            "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )
    if appearance.asset_binding_hash != asset.asset_hash:
        raise ValueError("COMPLETE_PUPPET_CAA_QUALIFICATION_ASSET_DRIFT")
    _require_caa_final_mesh_candidate_binding(mesh, asset)
    if partition_evidence.mesh_binding_hash != mesh.mesh_lineage_hash:
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_MESH_DRIFT")
    if partition_evidence.appearance_asset_binding_hash != asset.asset_hash:
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_ASSET_DRIFT")
    if (
        partition_evidence.appearance_qualification_binding_hash
        != appearance.qualification_hash
    ):
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_QUALIFICATION_DRIFT")
    if (
        str(structure.metadata.get("presentation_partition_evidence_hash") or "")
        != partition_evidence.evidence_hash
    ):
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_EVIDENCE_DRIFT")
    if (
        str(appearance.qualification_report.get("status") or "")
        != "PASS_COMPLETE_APPEARANCE"
    ):
        raise ValueError("COMPLETE_PUPPET_CAA_QUALIFICATION_NOT_PASS")
    if (
        float(appearance.total_defined_fraction) != 1.0
        or not bool(appearance.qualification_report.get("totality_passed", False))
    ):
        raise ValueError("COMPLETE_PUPPET_CAA_TOTALITY_NOT_PROVEN")

    source_owned_visual_mode = bool(
        dict(asset.metadata or {}).get("source_owned_visual_mesh_mode")
    )
    visual_mesh_set_hash = ""
    if source_owned_visual_mode:
        visual_set = visual_mesh_set_from_dict(
            stage_output_payload(
                ctx,
                "18_CANONICAL_MESH_ADDRESSING_BUILD",
                "RealSaS.VisualMeshSetIR.v1",
            )
        )
        visual_mesh_set_hash = str(visual_set.set_hash)
        for label, value in (
            (
                "ASSET",
                dict(asset.metadata or {}).get(
                    "visual_mesh_set_binding_hash"
                ),
            ),
            (
                "STRUCTURE",
                dict(structure.metadata or {}).get(
                    "visual_mesh_set_binding_hash"
                ),
            ),
            (
                "PARTITION_EVIDENCE",
                dict(partition_evidence.metadata or {}).get(
                    "visual_mesh_set_binding_hash"
                ),
            ),
        ):
            if str(value or "") != visual_mesh_set_hash:
                raise ValueError(
                    "COMPLETE_PUPPET_VISUAL_MESH_BINDING_DRIFT:"
                    + label
                )
        if (
            dict(asset.metadata or {}).get(
                "mechanical_mesh_render_authority"
            )
            is not False
            or dict(structure.metadata or {}).get(
                "mechanical_mesh_render_authority"
            )
            is not False
        ):
            raise ValueError(
                "COMPLETE_PUPPET_MECHANICAL_RENDER_AUTHORITY_DRIFT"
            )

    mechanical = build_canonical_puppet_state(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        mesh=mesh,
        mesh_skin=mesh_skin,
        metadata={
            "v2_role": "MECHANICAL_SUBSTATE_OF_COMPLETE_PUPPET",
            "appearance_bound_externally": True,
        },
    )
    graph = build_caa_bound_presentation_graph(
        structure=structure,
        product_state=mechanical,
        skeleton=skeleton,
        mesh=mesh,
        partition=partition,
        carrier_policy=carrier,
        directions=directions,
        appearance_asset_hash=asset.asset_hash,
        appearance_qualification_hash=appearance.qualification_hash,
        visual_mesh_set_hash=visual_mesh_set_hash,
    )
    complete = build_complete_puppet_state_v2(
        mechanical_state=mechanical,
        structure=structure,
        presentation_graph=graph,
        skeleton=skeleton,
        mesh=mesh,
        mesh_skin=mesh_skin,
        appearance_asset_hash=asset.asset_hash,
        appearance_qualification_hash=appearance.qualification_hash,
        output_direction_set_hash=directions.direction_set_hash,
        visual_mesh_set_hash=visual_mesh_set_hash,
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "canonical_mechanical_puppet_state.json",
                mechanical,
                authority_class="CANONICAL_MECHANICAL_PUPPET_STATE",
            ),
            write_ir(
                root / "qualified_presentation_graph.json",
                graph,
                authority_class="CAA_BOUND_QUALIFIED_PRESENTATION_GRAPH",
            ),
            write_ir(
                root / "complete_puppet_state_v2.json",
                complete,
                authority_class="COMPLETE_PUPPET_STATE_V2",
            ),
        ],
        "diagnostics": {
            "mechanical_state_hash": mechanical.product_state_hash,
            "presentation_lineage_hash": graph.presentation_lineage_hash,
            "complete_puppet_hash": complete.complete_puppet_hash,
            "complete_appearance_qualification_hash": appearance.qualification_hash,
            "geometry_mechanics_appearance_coequal": True,
            "source_owned_visual_mesh_mode": source_owned_visual_mode,
            "visual_mesh_set_binding_hash": visual_mesh_set_hash,
            "mechanical_mesh_render_authority": (
                False if source_owned_visual_mode else True
            ),
            "dynamic_visual_composition_authority_claimed": False,
        },
    }
