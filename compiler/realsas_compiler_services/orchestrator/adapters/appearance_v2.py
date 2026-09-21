from __future__ import annotations

"""RealSaS V2 Complete Appearance Authority stages 20-25."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    AppearanceTextureIR,
    CAACompileArtifactIR,
    CAACompileSealIR,
    CompleteAppearanceAssetIR,
    CompleteAppearanceQualificationIR,
    CAARestViewProofIR,
    CAARestRenderProofIR,
    CAA_PROVENANCE,
    caa_rest_render_proof_hash,
    build_caa_preregistration,
    caa_compile_artifact_from_dict,
    caa_compile_hash,
    caa_compile_seal_from_dict,
    caa_compile_seal_hash,
    caa_preregistration_from_dict,
    complete_appearance_asset_from_dict,
    complete_appearance_asset_hash,
    complete_appearance_qualification_hash,
)
from compiler.realsas_compiler_core.appearance_bake_v2 import (
    bake_direction_atlas,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    compile_deterministic_caa,
    resolve_projected_tile_resolution,
)
from compiler.realsas_compiler_core.appearance_color_v2 import (
    source_sample_roundtrip_pm_error,
)
from compiler.realsas_compiler_core.appearance_quality_v2 import (
    cross_view_source_compatibility_metrics,
    provenance_boundary_metrics,
    rgba_l1_premultiplied,
    source_feature_preservation_metrics,
    structured_holdout_metrics,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_uv,
    load_provenance_atlas,
    render_caa_reference,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import coverage_metrics
from compiler.realsas_compiler_core.output_presentation_v1 import (
    output_direction_set_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    appearance_domain_from_dict,
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    resolved_path,
    sha256_file,
    stage_output_payload,
    write_ir,
)


_POLICY_SCHEMA = "RealSaS.CAAQualificationPolicy.v1"


def _load_policy_document(ctx: dict) -> dict:
    cfg = dict(ctx["run_manifest"].get("appearance") or {})
    ref = dict(cfg.get("policy_document") or {})
    if not ref:
        raise QualificationError("CAA_FROZEN_POLICY_DOCUMENT_REQUIRED")
    policy = load_file_ref(ref, expected_schema=_POLICY_SCHEMA)
    if not str(policy.get("status") or "").startswith("FROZEN_SUBJECT_FREE"):
        raise QualificationError("CAA_POLICY_NOT_SUBJECT_FREE_FROZEN")
    for key in (
        "compile_policy",
        "source_lock_policy",
        "completion_quality_policy",
    ):
        if not isinstance(policy.get(key), dict):
            raise QualificationError(f"CAA_POLICY_SECTION_MISSING:{key}")
    return policy


def _load_source_inputs(ctx: dict, observation):
    cfg = dict(ctx["run_manifest"].get("observation") or {})
    raster_rows = tuple(cfg.get("source_rasters") or ())
    mask_rows = tuple(cfg.get("source_foreground_masks") or ())
    if (
        len(raster_rows) != 8
        or len(mask_rows) != 8
        or {int(row["view_index"]) for row in raster_rows} != set(range(8))
        or {int(row["view_index"]) for row in mask_rows} != set(range(8))
    ):
        raise QualificationError("CAA_SOURCE_OBSERVATION_MATRIX_INCOMPLETE")
    obs = {int(row.view_index): row for row in observation.views}
    rgba = {}
    masks = {}
    for row in raster_rows:
        view = int(row["view_index"])
        path = load_file_ref(dict(row.get("image") or {}), json_required=False)
        if sha256_file(path) != obs[view].source_raster_sha256:
            raise QualificationError("CAA_SOURCE_RASTER_AUTHORITY_DRIFT")
        image = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
        if image.shape != (int(obs[view].height), int(obs[view].width), 4):
            raise QualificationError("CAA_SOURCE_RASTER_DIMENSION_DRIFT")
        rgba[view] = image
    for row in mask_rows:
        view = int(row["view_index"])
        path = load_file_ref(dict(row.get("mask") or {}), json_required=False)
        if sha256_file(path) != obs[view].foreground_mask_sha256:
            raise QualificationError("CAA_SOURCE_MASK_AUTHORITY_DRIFT")
        raw = path.read_bytes()
        expected = int(obs[view].width) * int(obs[view].height)
        if len(raw) != expected or any(value not in (0, 1) for value in raw):
            raise QualificationError("CAA_SOURCE_MASK_INVALID")
        masks[view] = np.frombuffer(raw, dtype=np.uint8).reshape(
            int(obs[view].height), int(obs[view].width)
        ).astype(bool)
    return rgba, masks


def _save_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def _load_compile_arrays(artifact: CAACompileArtifactIR) -> dict:
    path = resolved_path(artifact.compile_npz_path)
    if not path.is_file() or sha256_file(path) != artifact.compile_npz_sha256:
        raise QualificationError("CAA_COMPILE_NPZ_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "barycentric",
            "sample_positions",
            "sample_face_index",
            "sample_component_index",
            "direct_valid",
            "direct_rgba",
            "direct_pm_linear",
            "source_xy",
            "rgba",
            "provenance",
            "source_view",
        }
        if not required.issubset(set(data.files)):
            raise QualificationError("CAA_COMPILE_NPZ_ARRAYS_MISSING")
        return {name: np.asarray(data[name]).copy() for name in required}


def preregister_caa_backend_stage(ctx: dict) -> dict:
    cfg = dict(ctx["run_manifest"].get("appearance") or {})
    backend = str(cfg.get("backend") or "")
    if not backend:
        return {
            "status": "BLOCKED",
            "blockers": ["CAA_BACKEND_MUST_BE_EXPLICIT"],
            "diagnostics": {},
        }
    policy = _load_policy_document(ctx)

    contract_path = (
        Path(ctx["repo_root"])
        / "canonical"
        / "COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json"
    )
    if not contract_path.is_file():
        raise QualificationError("CAA_CANONICAL_CONTRACT_MISSING")
    contract_sha = sha256_file(contract_path)

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
    directions = output_direction_set_from_dict(
        stage_output_payload(
            ctx,
            "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    addressing = surface_addressing_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.SurfaceAddressingIR.v1",
        )
    )
    domain = appearance_domain_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.AppearanceDomainIR.v1",
        )
    )
    static_mesh = static_mesh_qualification_from_dict(
        stage_output_payload(
            ctx,
            "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        )
    )

    observation_camera_hashes = tuple(
        row.camera_binding_hash
        for row in sorted(observation.views, key=lambda value: value.view_index)
    )
    if observation_camera_hashes != tuple(cameras.camera_binding_hashes):
        raise QualificationError("CAA_RASTER_CORRESPONDENCE_PREREQUISITE_FAILED")

    compile_policy = dict(policy["compile_policy"])
    if str(compile_policy.get("tile_resolution_mode") or "") != "PROJECTED_SOURCE_DENSITY_V1":
        raise QualificationError("CAA_TILE_RESOLUTION_MODE_UNSUPPORTED")
    _source_rgba, source_masks = _load_source_inputs(ctx, observation)
    tile_evidence = resolve_projected_tile_resolution(
        candidate=candidate,
        cameras=cameras.cameras,
        foreground_mask_by_view=source_masks,
        candidate_resolutions=tuple(
            int(value)
            for value in compile_policy.get("tile_resolution_candidates") or ()
        ),
        max_source_pixels_per_atlas_texel=float(
            compile_policy["max_source_pixels_per_atlas_texel"]
        ),
        bleed_px=int(compile_policy["bleed_px"]),
        max_atlas_resolution=int(compile_policy["max_atlas_resolution"]),
    )
    compile_policy["tile_resolution"] = int(
        tile_evidence["selected_tile_resolution"]
    )
    compile_policy["max_supported_face_count"] = int(
        tile_evidence["max_supported_face_count"]
    )
    compile_policy["tile_resolution_evidence"] = tile_evidence

    prereg = build_caa_preregistration(
        backend_id=backend,
        contract_sha256=contract_sha,
        observation_set_hash=observation.observation_set_hash,
        camera_set_hash=cameras.camera_set_hash,
        output_direction_set_hash=directions.direction_set_hash,
        candidate_mesh_hash=candidate.candidate_lineage_hash,
        surface_addressing_hash=addressing.addressing_hash,
        appearance_domain_hash=domain.domain_hash,
        static_mesh_qualification_hash=static_mesh.qualification_hash,
        compile_policy=compile_policy,
        source_lock_policy=dict(policy["source_lock_policy"]),
        completion_quality_policy=dict(policy["completion_quality_policy"]),
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "caa_backend_preregistration.json",
                prereg,
                authority_class="CAA_BACKEND_PREREGISTRATION",
            )
        ],
        "diagnostics": {
            "backend_id": backend,
            "shipping_eligible": prereg.shipping_eligible,
            "contract_sha256": contract_sha,
            "preregistration_hash": prereg.preregistration_hash,
            "policy_document_sha256": str(
                dict(cfg.get("policy_document") or {}).get("sha256") or ""
            ),
            "raster_correspondence_prerequisite": "PASS",
            "tile_resolution_evidence": tile_evidence,
        },
    }


def compile_caa_stage(ctx: dict) -> dict:
    prereg = caa_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "20_CAA_BACKEND_PREREGISTERED",
            "RealSaS.CAACompilePreregistrationIR.v2",
        )
    )
    if prereg.backend_id != "DETERMINISTIC_V1":
        return {
            "status": "BLOCKED",
            "blockers": [f"CAA_BACKEND_EXECUTOR_NOT_BOUND:{prereg.backend_id}"],
            "diagnostics": {
                "note": "LEARNED_V2 and IM2SURFTEX_RESEARCH_ONLY require separate exact-hash execution receipts."
            },
        }

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    rgba, masks = _load_source_inputs(ctx, observation)

    compile_policy = dict(prereg.compile_policy)
    result = compile_deterministic_caa(
        candidate=candidate,
        cameras=cameras.cameras,
        source_rgba_by_view=rgba,
        foreground_mask_by_view=masks,
        tile_resolution=int(compile_policy["tile_resolution"]),
        source_lock_policy=prereg.source_lock_policy,
        completion_quality_policy=prereg.completion_quality_policy,
    )
    component_ids = tuple(result["component_ids"])
    component_index = {component_id: index for index, component_id in enumerate(component_ids)}
    sample_component_index = np.asarray(
        [component_index[value] for value in result["sample_component"]],
        dtype=np.int32,
    )

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    npz_path = root / "caa_compile.npz"
    npz_sha = _save_npz(
        npz_path,
        barycentric=result["barycentric"],
        sample_positions=result["sample_positions"],
        sample_face_index=result["sample_face_index"],
        sample_component_index=sample_component_index,
        direct_valid=result["direct_valid"],
        direct_rgba=result["direct_rgba"],
        direct_pm_linear=result["direct_pm_linear"],
        source_xy=result["source_xy"],
        rgba=result["rgba"],
        provenance=result["provenance"],
        source_view=result["source_view"],
    )
    counts = result["counts"]
    total = int(result["rgba"].shape[0] * result["rgba"].shape[1])
    artifact = CAACompileArtifactIR(
        backend_id=prereg.backend_id,
        preregistration_binding_hash=prereg.preregistration_hash,
        candidate_mesh_binding_hash=prereg.candidate_mesh_binding_hash,
        surface_addressing_binding_hash=prereg.surface_addressing_binding_hash,
        appearance_domain_binding_hash=prereg.appearance_domain_binding_hash,
        output_direction_set_binding_hash=prereg.output_direction_set_binding_hash,
        compile_npz_path=str(npz_path),
        compile_npz_sha256=npz_sha,
        face_count=int(result["face_count"]),
        direction_count=8,
        tile_resolution=int(compile_policy["tile_resolution"]),
        sample_count_per_face=int(result["sample_count_per_face"]),
        total_sample_count=total,
        direct_source_sample_count=int(counts["DIRECT_SOURCE"]),
        other_view_source_sample_count=int(counts["OTHER_VIEW_SOURCE"]),
        compiled_local_harmonic_sample_count=int(
            counts["COMPILED_LOCAL_HARMONIC"]
        ),
        compile_hash="",
        metadata={
            "component_ids": component_ids,
            "total_appearance_defined": True,
            "runtime_generation_used": False,
            "geometry_mutated": False,
            "visibility_authority": "RealSaS.VisibilityContract.v2",
            "completion_mode": "BOUNDED_CANONICAL_SURFACE_HARMONIC",
            "completion_rows": list(result["completion_rows"]),
            "global_surface_fill_used": False,
        },
    )
    artifact = replace(artifact, compile_hash=caa_compile_hash(artifact))
    root.mkdir(parents=True, exist_ok=True)
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "caa_compile_artifact.json",
                artifact,
                authority_class="CAA_COMPILE_ARTIFACT",
            ),
            {
                "path": str(npz_path),
                "sha256": npz_sha,
                "authority_class": "CAA_COMPILE_ARRAYS",
                "schema": "RealSaS.CAACompileArrays.v2",
            },
        ],
        "diagnostics": {
            "compile_hash": artifact.compile_hash,
            "total_sample_count": total,
            **{f"provenance_{key.lower()}": int(value) for key, value in counts.items()},
        },
    }


def seal_caa_compile_stage(ctx: dict) -> dict:
    prereg = caa_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "20_CAA_BACKEND_PREREGISTERED",
            "RealSaS.CAACompilePreregistrationIR.v2",
        )
    )
    artifact = caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,
            "21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    arrays = _load_compile_arrays(artifact)
    if artifact.preregistration_binding_hash != prereg.preregistration_hash:
        raise QualificationError("CAA_COMPILE_PREREG_BINDING_DRIFT")
    if np.any(arrays["provenance"] == 255):
        raise QualificationError("CAA_COMPILE_NOT_TOTAL")
    direct = arrays["direct_valid"]
    direct_exact = np.all(
        arrays["rgba"][direct] == arrays["direct_rgba"][direct],
        axis=-1,
    )
    if len(direct_exact) and not np.all(direct_exact):
        raise QualificationError("CAA_DIRECT_SOURCE_MUTATED_DURING_COMPILE")

    seal = CAACompileSealIR(
        compile_binding_hash=artifact.compile_hash,
        preregistration_binding_hash=prereg.preregistration_hash,
        compile_npz_sha256=artifact.compile_npz_sha256,
        qualification_report={
            "status": "PASS_CAA_COMPILE_SEAL",
            "total_appearance_defined": True,
            "direct_source_immutable": True,
            "runtime_generation_required": False,
            "geometry_mutation_used": False,
        },
        seal_hash="",
        metadata={
            "backend_id": artifact.backend_id,
            "shipping_eligible": prereg.shipping_eligible,
        },
    )
    seal = replace(seal, seal_hash=caa_compile_seal_hash(seal))
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "caa_compile_seal.json",
                seal,
                authority_class="CAA_COMPILE_SEAL",
            )
        ],
        "diagnostics": {
            "seal_hash": seal.seal_hash,
            "direct_source_immutable": True,
            "totality": True,
        },
    }


def bake_complete_appearance_stage(ctx: dict) -> dict:
    prereg = caa_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "20_CAA_BACKEND_PREREGISTERED",
            "RealSaS.CAACompilePreregistrationIR.v2",
        )
    )
    artifact = caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,
            "21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    seal = caa_compile_seal_from_dict(
        stage_output_payload(
            ctx,
            "22_CAA_COMPILE_SEALED",
            "RealSaS.CAACompileSealIR.v2",
        )
    )
    if seal.compile_binding_hash != artifact.compile_hash:
        raise QualificationError("CAA_BAKE_COMPILE_SEAL_DRIFT")
    arrays = _load_compile_arrays(artifact)
    bleed = int(prereg.compile_policy["bleed_px"])
    tile_resolution = int(prereg.compile_policy["tile_resolution"])

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    root.mkdir(parents=True, exist_ok=True)
    from compiler.realsas_compiler_core.appearance_compile_v2 import face_atlas_layout
    max_supported_face_count = int(prereg.compile_policy["max_supported_face_count"])
    if artifact.face_count > max_supported_face_count:
        return {
            "status": "FAIL",
            "blockers": ["CAA_FACE_COUNT_EXCEEDS_FROZEN_ATLAS_CAPACITY"],
            "diagnostics": {
                "face_count": artifact.face_count,
                "max_supported_face_count": max_supported_face_count,
            },
        }
    projected_layout = face_atlas_layout(
        artifact.face_count,
        tile_resolution=tile_resolution,
        bleed_px=bleed,
    )
    max_atlas_resolution = int(prereg.compile_policy["max_atlas_resolution"])
    if (
        int(projected_layout["width"]) > max_atlas_resolution
        or int(projected_layout["height"]) > max_atlas_resolution
    ):
        return {
            "status": "FAIL",
            "blockers": ["CAA_ATLAS_EXCEEDS_FROZEN_PRODUCT_RESOLUTION_CAP"],
            "diagnostics": {
                "width": int(projected_layout["width"]),
                "height": int(projected_layout["height"]),
                "max_atlas_resolution": max_atlas_resolution,
                "face_count": artifact.face_count,
                "tile_resolution": tile_resolution,
                "bleed_px": bleed,
            },
        }
    texture_rows = []
    output_rows = []
    provenance_atlases = []
    reference_uv = None
    reference_layout = None
    for direction in range(8):
        atlas, provenance_atlas, uv, layout = bake_direction_atlas(
            face_sample_rgba=arrays["rgba"][direction],
            face_sample_provenance=arrays["provenance"][direction],
            face_count=artifact.face_count,
            tile_resolution=tile_resolution,
            bleed_px=bleed,
        )
        if reference_uv is None:
            reference_uv = uv
            reference_layout = layout
        elif not np.array_equal(reference_uv, uv) or reference_layout != layout:
            raise QualificationError("CAA_BAKE_DIRECTION_LAYOUT_DRIFT")
        path = root / f"V{direction}_appearance.png"
        Image.fromarray(atlas, mode="RGBA").save(
            path,
            format="PNG",
            optimize=False,
            compress_level=6,
        )
        digest = sha256_file(path)
        texture_rows.append(
            AppearanceTextureIR(
                direction_index=direction,
                direction_id=f"V{direction}",
                transport_png_path=str(path),
                transport_png_sha256=digest,
                width=int(atlas.shape[1]),
                height=int(atlas.shape[0]),
                metadata={
                    "transport_alpha": "STRAIGHT",
                    "runtime_filtering": "PREMULTIPLIED",
                    "atlas_bleed_px": bleed,
                },
            )
        )
        output_rows.append(
            {
                "path": str(path),
                "sha256": digest,
                "authority_class": "CAA_TRANSPORT_TEXTURE",
                "schema": f"RealSaS.CAATransportTexture.V{direction}.v2",
            }
        )
        provenance_atlases.append(provenance_atlas)

    uv_path = root / "surface_uv.npz"
    uv_sha = _save_npz(uv_path, face_uv=np.asarray(reference_uv, dtype=np.float32))
    provenance_path = root / "provenance_atlas.npz"
    provenance_sha = _save_npz(
        provenance_path,
        provenance=np.stack(provenance_atlases, axis=0).astype(np.uint8),
    )

    asset = CompleteAppearanceAssetIR(
        compile_seal_binding_hash=seal.seal_hash,
        candidate_mesh_binding_hash=artifact.candidate_mesh_binding_hash,
        surface_addressing_binding_hash=artifact.surface_addressing_binding_hash,
        appearance_domain_binding_hash=artifact.appearance_domain_binding_hash,
        output_direction_set_binding_hash=artifact.output_direction_set_binding_hash,
        textures=tuple(texture_rows),
        uv_npz_path=str(uv_path),
        uv_npz_sha256=uv_sha,
        provenance_npz_path=str(provenance_path),
        provenance_npz_sha256=provenance_sha,
        atlas_layout=dict(reference_layout),
        asset_hash="",
        metadata={
            "total_appearance_asset": True,
            "unique_face_barycentric_atlas": True,
            "internal_alpha": "PREMULTIPLIED",
            "transport_png_alpha": "STRAIGHT",
            "unpremultiply_export_boundary_count": 1,
            "runtime_generation_forbidden": True,
        },
    )
    asset = replace(asset, asset_hash=complete_appearance_asset_hash(asset))
    output_rows.extend(
        [
            {
                "path": str(uv_path),
                "sha256": uv_sha,
                "authority_class": "CAA_SURFACE_UV",
                "schema": "RealSaS.CAASurfaceUV.v2",
            },
            {
                "path": str(provenance_path),
                "sha256": provenance_sha,
                "authority_class": "CAA_PROVENANCE_ATLAS",
                "schema": "RealSaS.CAAProvenanceAtlas.v2",
            },
            write_ir(
                root / "complete_appearance_asset.json",
                asset,
                authority_class="COMPLETE_APPEARANCE_ASSET",
            ),
        ]
    )
    return {
        "status": "PASS",
        "outputs": output_rows,
        "diagnostics": {
            "asset_hash": asset.asset_hash,
            "atlas_width": int(reference_layout["width"]),
            "atlas_height": int(reference_layout["height"]),
            "bleed_px": bleed,
            "direction_count": 8,
        },
    }


def qualify_complete_appearance_stage(ctx: dict) -> dict:
    prereg = caa_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "20_CAA_BACKEND_PREREGISTERED",
            "RealSaS.CAACompilePreregistrationIR.v2",
        )
    )
    artifact = caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,
            "21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    if asset.candidate_mesh_binding_hash != artifact.candidate_mesh_binding_hash:
        raise QualificationError("CAA_QUALIFICATION_MESH_BINDING_DRIFT")
    for texture in asset.textures:
        path = resolved_path(texture.transport_png_path)
        if not path.is_file() or sha256_file(path) != texture.transport_png_sha256:
            raise QualificationError("CAA_QUALIFICATION_TEXTURE_BYTES_DRIFT")

    arrays = _load_compile_arrays(artifact)
    direct = arrays["direct_valid"]
    direct_count = int(np.count_nonzero(direct))
    if direct_count <= 0:
        raise QualificationError("CAA_QUALIFICATION_REQUIRES_DIRECT_SOURCE_SAMPLES")
    direct_exact = np.all(
        arrays["rgba"][direct] == arrays["direct_rgba"][direct],
        axis=-1,
    )
    source_exact_fraction = float(np.mean(direct_exact))
    total_fraction = float(np.mean(arrays["provenance"] != 255))

    policy = dict(prereg.completion_quality_policy)
    required = (
        "min_source_lock_exact_fraction",
        "min_total_defined_fraction",
        "holdout_band_fraction",
        "min_structured_holdout_samples",
        "min_structured_holdout_samples_per_view",
        "max_structured_holdout_mean_rgba_l1",
        "max_structured_holdout_p95_rgba_l1",
        "max_provenance_boundary_mean_rgba_l1",
        "max_provenance_boundary_p95_rgba_l1",
        "max_provenance_boundary_mean_gradient_jump",
        "max_provenance_boundary_p95_gradient_jump",
        "max_source_sample_pm_roundtrip_abs_error",
        "cross_view_color_conflict_cut_rgba_l1",
        "cross_view_alpha_conflict_cut",
        "cross_view_min_shared_direct_samples_per_pair",
        "cross_view_min_component_samples_for_gate",
        "cross_view_max_pair_p95_rgba_l1",
        "cross_view_max_pair_color_conflict_fraction",
        "cross_view_max_pair_p95_alpha_abs",
        "cross_view_max_pair_alpha_conflict_fraction",
        "cross_view_max_component_color_conflict_fraction",
        "cross_view_max_component_alpha_conflict_fraction",
    )
    if any(key not in policy for key in required):
        raise QualificationError("CAA_QUALITY_POLICY_INCOMPLETE")

    holdout = structured_holdout_metrics(
        direct_valid=arrays["direct_valid"],
        direct_rgba=arrays["direct_rgba"],
        source_xy=arrays["source_xy"],
        sample_positions=arrays["sample_positions"],
        sample_component_index=arrays["sample_component_index"],
        sample_face_index=arrays["sample_face_index"],
        face_count=artifact.face_count,
        tile_resolution=artifact.tile_resolution,
        band_fraction=float(policy["holdout_band_fraction"]),
        max_region_samples=int(policy["max_local_harmonic_region_samples"]),
        max_graph_hops=int(policy["max_local_harmonic_graph_hops"]),
    )
    seam = provenance_boundary_metrics(
        rgba=arrays["rgba"],
        provenance=arrays["provenance"],
        source_view=arrays["source_view"],
        sample_positions=arrays["sample_positions"],
        sample_face_index=arrays["sample_face_index"],
        face_count=artifact.face_count,
        tile_resolution=artifact.tile_resolution,
    )
    cross_view = cross_view_source_compatibility_metrics(
        direct_valid=arrays["direct_valid"],
        direct_rgba=arrays["direct_rgba"],
        sample_component_index=arrays["sample_component_index"],
        color_conflict_cut_rgba_l1=float(
            policy["cross_view_color_conflict_cut_rgba_l1"]
        ),
        alpha_conflict_cut=float(policy["cross_view_alpha_conflict_cut"]),
    )

    direct_pm_roundtrip = source_sample_roundtrip_pm_error(
        arrays["direct_rgba"][direct],
        arrays["direct_pm_linear"][direct],
    )
    max_source_pm_roundtrip_error = (
        0.0
        if len(direct_pm_roundtrip) == 0
        else float(np.max(direct_pm_roundtrip))
    )
    source_pm_roundtrip_passed = (
        max_source_pm_roundtrip_error
        <= float(policy["max_source_sample_pm_roundtrip_abs_error"])
    )

    cross_view_pair_passed = (
        len(cross_view["per_pair"]) == 8
        and all(
            int(row["shared_direct_sample_count"])
            >= int(policy["cross_view_min_shared_direct_samples_per_pair"])
            and float(row["p95_premultiplied_rgba_l1"])
            <= float(policy["cross_view_max_pair_p95_rgba_l1"])
            and float(row["color_conflict_fraction"])
            <= float(policy["cross_view_max_pair_color_conflict_fraction"])
            and float(row["p95_alpha_abs"])
            <= float(policy["cross_view_max_pair_p95_alpha_abs"])
            and float(row["alpha_conflict_fraction"])
            <= float(policy["cross_view_max_pair_alpha_conflict_fraction"])
            for row in cross_view["per_pair"]
        )
    )
    component_min = int(policy["cross_view_min_component_samples_for_gate"])
    qualified_component_rows = [
        row
        for row in cross_view["per_pair_component"]
        if int(row["shared_direct_sample_count"]) >= component_min
    ]
    cross_view_component_passed = all(
        float(row["color_conflict_fraction"])
        <= float(policy["cross_view_max_component_color_conflict_fraction"])
        and float(row["alpha_conflict_fraction"])
        <= float(policy["cross_view_max_component_alpha_conflict_fraction"])
        for row in qualified_component_rows
    )
    cross_view_passed = cross_view_pair_passed and cross_view_component_passed

    holdout_per_view_passed = (
        len(holdout["per_view"]) == 8
        and all(
            int(row.get("holdout_sample_count", 0))
            >= int(policy["min_structured_holdout_samples_per_view"])
            and float(row.get("mean_rgba_l1", float("inf")))
            <= float(policy["max_structured_holdout_mean_rgba_l1"])
            and float(row.get("p95_rgba_l1", float("inf")))
            <= float(policy["max_structured_holdout_p95_rgba_l1"])
            for row in holdout["per_view"]
        )
    )
    seam_per_view_passed = (
        len(seam["per_view"]) == 8
        and all(
            int(row.get("boundary_pair_count", 0)) == 0
            or (
                float(row.get("mean_rgba_l1", float("inf")))
                <= float(policy["max_provenance_boundary_mean_rgba_l1"])
                and float(row.get("p95_rgba_l1", float("inf")))
                <= float(policy["max_provenance_boundary_p95_rgba_l1"])
                and float(row.get("mean_gradient_jump", float("inf")))
                <= float(policy["max_provenance_boundary_mean_gradient_jump"])
                and float(row.get("p95_gradient_jump", float("inf")))
                <= float(policy["max_provenance_boundary_p95_gradient_jump"])
            )
            for row in seam["per_view"]
        )
    )

    passed = (
        source_exact_fraction
        >= float(policy["min_source_lock_exact_fraction"])
        and total_fraction >= float(policy["min_total_defined_fraction"])
        and int(holdout["sample_count"])
        >= int(policy["min_structured_holdout_samples"])
        and holdout_per_view_passed
        and float(holdout["mean_rgba_l1"])
        <= float(policy["max_structured_holdout_mean_rgba_l1"])
        and float(holdout["p95_rgba_l1"])
        <= float(policy["max_structured_holdout_p95_rgba_l1"])
        and float(seam["mean_rgba_l1"])
        <= float(policy["max_provenance_boundary_mean_rgba_l1"])
        and float(seam["p95_rgba_l1"])
        <= float(policy["max_provenance_boundary_p95_rgba_l1"])
        and float(seam["mean_gradient_jump"])
        <= float(policy["max_provenance_boundary_mean_gradient_jump"])
        and float(seam["p95_gradient_jump"])
        <= float(policy["max_provenance_boundary_p95_gradient_jump"])
        and seam_per_view_passed
        and source_pm_roundtrip_passed
        and cross_view_passed
    )

    value = CompleteAppearanceQualificationIR(
        asset_binding_hash=asset.asset_hash,
        preregistration_binding_hash=prereg.preregistration_hash,
        source_lock_exact_fraction=source_exact_fraction,
        total_defined_fraction=total_fraction,
        structured_holdout_sample_count=int(holdout["sample_count"]),
        structured_holdout_mean_rgba_l1=float(holdout["mean_rgba_l1"]),
        structured_holdout_p95_rgba_l1=float(holdout["p95_rgba_l1"]),
        provenance_boundary_pair_count=int(seam["boundary_pair_count"]),
        provenance_boundary_mean_rgba_l1=float(seam["mean_rgba_l1"]),
        provenance_boundary_p95_rgba_l1=float(seam["p95_rgba_l1"]),
        provenance_boundary_gradient_pair_count=int(seam["gradient_pair_count"]),
        provenance_boundary_mean_gradient_jump=float(seam["mean_gradient_jump"]),
        provenance_boundary_p95_gradient_jump=float(seam["p95_gradient_jump"]),
        qualification_report={
            "status": "PASS_COMPLETE_APPEARANCE" if passed else "FAIL_COMPLETE_APPEARANCE",
            "source_lock_passed": source_exact_fraction
            >= float(policy["min_source_lock_exact_fraction"]),
            "totality_passed": total_fraction
            >= float(policy["min_total_defined_fraction"]),
            "structured_holdout_passed": (
                int(holdout["sample_count"])
                >= int(policy["min_structured_holdout_samples"])
                and holdout_per_view_passed
                and float(holdout["mean_rgba_l1"])
                <= float(policy["max_structured_holdout_mean_rgba_l1"])
                and float(holdout["p95_rgba_l1"])
                <= float(policy["max_structured_holdout_p95_rgba_l1"])
            ),
            "provenance_seam_passed": (
                float(seam["mean_rgba_l1"])
                <= float(policy["max_provenance_boundary_mean_rgba_l1"])
                and float(seam["p95_rgba_l1"])
                <= float(policy["max_provenance_boundary_p95_rgba_l1"])
                and float(seam["mean_gradient_jump"])
                <= float(policy["max_provenance_boundary_mean_gradient_jump"])
                and float(seam["p95_gradient_jump"])
                <= float(policy["max_provenance_boundary_p95_gradient_jump"])
                and seam_per_view_passed
            ),
            "holdout_every_view_passed": holdout_per_view_passed,
            "seam_every_view_passed": seam_per_view_passed,
            "cross_view_source_compatibility_measured": True,
            "cross_view_source_compatibility_shipping_gate_frozen": True,
            "cross_view_source_compatibility_passed": cross_view_passed,
            "cross_view_pair_passed": cross_view_pair_passed,
            "cross_view_component_passed": cross_view_component_passed,
            "source_pm_roundtrip_max_abs_error": max_source_pm_roundtrip_error,
            "source_pm_roundtrip_passed": source_pm_roundtrip_passed,
            "appearance_is_coequal_product_authority": True,
        },
        qualification_hash="",
        metadata={
            "holdout": holdout,
            "seam": seam,
            "cross_view_source_compatibility": cross_view,
            "source_pm_roundtrip_max_abs_error": max_source_pm_roundtrip_error,
            "policy": policy,
            "totality_does_not_claim_geometry_or_visibility_correctness": True,
        },
    )
    value = replace(
        value,
        qualification_hash=complete_appearance_qualification_hash(value),
    )
    if not passed:
        return {
            "status": "FAIL",
            "blockers": ["COMPLETE_APPEARANCE_QUALITY_GATE_FAILED"],
            "diagnostics": value.to_dict(),
        }
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "complete_appearance_qualification.json",
                value,
                authority_class="QUALIFIED_COMPLETE_APPEARANCE",
            )
        ],
        "diagnostics": {
            "qualification_hash": value.qualification_hash,
            "source_lock_exact_fraction": source_exact_fraction,
            "total_defined_fraction": total_fraction,
            "holdout_p95_rgba_l1": holdout["p95_rgba_l1"],
            "seam_p95_rgba_l1": seam["p95_rgba_l1"],
            "seam_p95_gradient_jump": seam["p95_gradient_jump"],
            "cross_view_shared_direct_sample_count": cross_view[
                "shared_direct_sample_count"
            ],
            "cross_view_p95_premultiplied_rgba_l1": cross_view[
                "p95_premultiplied_rgba_l1"
            ],
            "cross_view_p95_alpha_abs": cross_view["p95_alpha_abs"],
        },
    }


def prove_caa_reference_rest_stage(ctx: dict) -> dict:
    prereg = caa_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "20_CAA_BACKEND_PREREGISTERED",
            "RealSaS.CAACompilePreregistrationIR.v2",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    qualification = stage_output_payload(
        ctx,
        "24_COMPLETE_APPEARANCE_QUALIFIED",
        "RealSaS.CompleteAppearanceQualificationIR.v2",
    )
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    static_mesh = static_mesh_qualification_from_dict(
        stage_output_payload(
            ctx,
            "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    if str(qualification.get("asset_binding_hash")) != asset.asset_hash:
        raise QualificationError("CAA_REST_PROOF_QUALIFICATION_ASSET_DRIFT")
    if static_mesh.candidate_mesh_binding_hash != asset.candidate_mesh_binding_hash:
        raise QualificationError("CAA_REST_PROOF_STATIC_MESH_BINDING_DRIFT")

    source_rgba, source_masks = _load_source_inputs(ctx, observation)
    face_uv = load_face_uv(asset)
    provenance_all = load_provenance_atlas(asset)
    by_camera = {int(camera.view_index): camera for camera in cameras.cameras}
    by_texture = {int(row.direction_index): row for row in asset.textures}

    policy = dict(prereg.completion_quality_policy)
    required = (
        "rest_min_source_lock_fraction_of_source_foreground",
        "rest_max_source_locked_mean_rgba_l1",
        "rest_max_source_locked_p95_rgba_l1",
        "rest_max_source_foreground_mean_rgba_l1",
        "rest_max_source_foreground_p95_rgba_l1",
        "rest_min_source_alpha_recall",
        "rest_min_source_alpha_precision",
        "rest_max_largest_coherent_alpha_hole_fraction",
        "rest_max_alpha_interior_uncovered_fraction",
        "rest_max_exact_depth_ambiguous_fraction",
        "rest_max_visibility_layer_overflow_pixel_count",
        "rest_feature_high_error_cut_rgba_l1",
        "rest_feature_edge_gradient_cut",
        "rest_max_feature_high_error_fraction",
        "rest_max_largest_connected_high_error_fraction",
        "rest_max_feature_p999_rgba_l1",
        "rest_min_feature_edge_recall_1px",
        "rest_min_feature_edge_precision_1px",
    )
    if any(key not in policy for key in required):
        raise QualificationError("CAA_REST_PROOF_POLICY_INCOMPLETE")

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    outputs = []
    all_pass = True
    for direction in range(8):
        texture_row = by_texture[direction]
        texture_path = resolved_path(texture_row.transport_png_path)
        texture = np.asarray(Image.open(texture_path).convert("RGBA"), dtype=np.uint8)
        render = render_caa_reference(
            mesh=candidate,
            camera=by_camera[direction],
            face_uv=face_uv,
            texture_rgba_u8=texture,
            provenance_atlas=provenance_all[direction],
        )
        image_path = root / f"V{direction}_reference_rest.png"
        Image.fromarray(render.straight_rgba_u8, mode="RGBA").save(
            image_path, format="PNG", optimize=False, compress_level=6
        )
        image_sha = sha256_file(image_path)
        outputs.append(
            {
                "path": str(image_path),
                "sha256": image_sha,
                "authority_class": "CAA_REFERENCE_REST_RENDER",
                "schema": f"RealSaS.CAAReferenceRestRender.V{direction}.v2",
            }
        )

        visible = render.geometry_visible
        final_alpha = render.final_alpha
        direct = visible & (
            render.provenance_code == int(CAA_PROVENANCE["DIRECT_SOURCE"])
        )
        direct_count = int(np.count_nonzero(direct))
        exact_count = 0
        mean_error = 0.0
        p95_error = 0.0
        if direct_count:
            predicted = render.straight_rgba_u8[direct]
            truth = source_rgba[direction][direct]
            exact = np.all(predicted == truth, axis=1)
            exact_count = int(np.count_nonzero(exact))
            error = rgba_l1_premultiplied(predicted, truth)
            mean_error = float(np.mean(error))
            p95_error = float(np.quantile(error, 0.95))
        exact_fraction = (
            1.0 if direct_count == 0 else float(exact_count) / float(direct_count)
        )

        visible_count = int(np.count_nonzero(visible))
        exact_depth_ambiguous_count = int(
            np.count_nonzero(render.exact_depth_ambiguity)
        )
        layer_overflow_count = int(np.count_nonzero(render.layer_overflow))
        exact_depth_ambiguous_fraction = (
            0.0
            if visible_count <= 0
            else float(exact_depth_ambiguous_count) / float(visible_count)
        )
        final_alpha_count = int(np.count_nonzero(final_alpha))
        hole = visible & ~final_alpha
        hole_count = int(np.count_nonzero(hole))
        hole_fraction = (
            0.0 if visible_count == 0 else float(hole_count) / float(visible_count)
        )
        foreground_pixels = source_masks[direction]
        foreground_error = rgba_l1_premultiplied(
            render.straight_rgba_u8[foreground_pixels],
            source_rgba[direction][foreground_pixels],
        )
        foreground_mean_error = (
            0.0 if len(foreground_error) == 0 else float(np.mean(foreground_error))
        )
        foreground_p95_error = (
            0.0
            if len(foreground_error) == 0
            else float(np.quantile(foreground_error, 0.95))
        )
        feature_metrics = source_feature_preservation_metrics(
            predicted_rgba=render.straight_rgba_u8,
            source_rgba=source_rgba[direction],
            source_foreground=source_masks[direction],
            high_error_cut_rgba_l1=float(
                policy["rest_feature_high_error_cut_rgba_l1"]
            ),
            edge_gradient_cut=float(policy["rest_feature_edge_gradient_cut"]),
        )
        source_alpha_bytes = bytes(source_masks[direction].astype(np.uint8).reshape(-1))
        final_alpha_bytes = bytes(final_alpha.astype(np.uint8).reshape(-1))
        alpha_metrics = coverage_metrics(
            source_alpha_bytes,
            final_alpha_bytes,
            width=int(observation.views[direction].width),
            height=int(observation.views[direction].height),
        )
        source_foreground_count = int(np.count_nonzero(source_masks[direction]))
        source_lock_fraction = (
            0.0
            if source_foreground_count <= 0
            else float(direct_count) / float(source_foreground_count)
        )
        view_pass = (
            source_lock_fraction
            >= float(policy["rest_min_source_lock_fraction_of_source_foreground"])
            and mean_error <= float(policy["rest_max_source_locked_mean_rgba_l1"])
            and p95_error <= float(policy["rest_max_source_locked_p95_rgba_l1"])
            and foreground_mean_error
            <= float(policy["rest_max_source_foreground_mean_rgba_l1"])
            and foreground_p95_error
            <= float(policy["rest_max_source_foreground_p95_rgba_l1"])
            and float(alpha_metrics["recall"])
            >= float(policy["rest_min_source_alpha_recall"])
            and float(alpha_metrics["precision"])
            >= float(policy["rest_min_source_alpha_precision"])
            and float(alpha_metrics["largest_coherent_hole_fraction"])
            <= float(policy["rest_max_largest_coherent_alpha_hole_fraction"])
            and float(alpha_metrics["interior_uncovered_fraction"])
            <= float(policy["rest_max_alpha_interior_uncovered_fraction"])
            and exact_depth_ambiguous_fraction
            <= float(policy["rest_max_exact_depth_ambiguous_fraction"])
            and layer_overflow_count
            <= int(policy["rest_max_visibility_layer_overflow_pixel_count"])
            and float(feature_metrics["high_error_fraction"])
            <= float(policy["rest_max_feature_high_error_fraction"])
            and float(feature_metrics["largest_connected_high_error_fraction"])
            <= float(policy["rest_max_largest_connected_high_error_fraction"])
            and float(feature_metrics["p999_rgba_l1"])
            <= float(policy["rest_max_feature_p999_rgba_l1"])
            and float(feature_metrics["edge_recall_1px"])
            >= float(policy["rest_min_feature_edge_recall_1px"])
            and float(feature_metrics["edge_precision_1px"])
            >= float(policy["rest_min_feature_edge_precision_1px"])
        )
        all_pass = all_pass and view_pass
        rows.append(
            CAARestViewProofIR(
                direction_index=direction,
                rendered_rgba_sha256=image_sha,
                rendered_alpha_pixel_count=final_alpha_count,
                source_locked_pixel_count=direct_count,
                source_locked_fraction_of_source_foreground=source_lock_fraction,
                source_locked_exact_pixel_count=exact_count,
                source_locked_exact_fraction=exact_fraction,
                source_locked_mean_rgba_l1=mean_error,
                source_locked_p95_rgba_l1=p95_error,
                source_foreground_mean_rgba_l1=foreground_mean_error,
                source_foreground_p95_rgba_l1=foreground_p95_error,
                source_feature_p999_rgba_l1=float(feature_metrics["p999_rgba_l1"]),
                source_feature_high_error_fraction=float(
                    feature_metrics["high_error_fraction"]
                ),
                largest_connected_feature_high_error_fraction=float(
                    feature_metrics["largest_connected_high_error_fraction"]
                ),
                source_feature_edge_recall_1px=float(
                    feature_metrics["edge_recall_1px"]
                ),
                source_feature_edge_precision_1px=float(
                    feature_metrics["edge_precision_1px"]
                ),
                geometry_visible_pixel_count=visible_count,
                final_alpha_pixel_count=final_alpha_count,
                geometry_visible_final_alpha_hole_count=hole_count,
                geometry_visible_final_alpha_hole_fraction=hole_fraction,
                source_alpha_recall=float(alpha_metrics["recall"]),
                source_alpha_precision=float(alpha_metrics["precision"]),
                largest_coherent_alpha_hole_fraction=float(alpha_metrics["largest_coherent_hole_fraction"]),
                alpha_interior_uncovered_fraction=float(alpha_metrics["interior_uncovered_fraction"]),
                metadata={
                    "status": "PASS" if view_pass else "FAIL",
                    "source_evidence_available": True,
                    "visibility_and_appearance_masks_separate": True,
                    "geometry_visible_alpha_zero_is_diagnostic_not_undefinedness": True,
                    "exact_depth_ambiguous_pixel_count": exact_depth_ambiguous_count,
                    "exact_depth_ambiguous_fraction": exact_depth_ambiguous_fraction,
                    "exact_depth_ambiguity_passed": (
                        exact_depth_ambiguous_fraction
                        <= float(policy["rest_max_exact_depth_ambiguous_fraction"])
                    ),
                    "visibility_layer_overflow_pixel_count": layer_overflow_count,
                    "visibility_layer_overflow_passed": (
                        layer_overflow_count
                        <= int(policy["rest_max_visibility_layer_overflow_pixel_count"])
                    ),
                    "source_feature_preservation": dict(feature_metrics),
                    "source_feature_preservation_passed": (
                        float(feature_metrics["high_error_fraction"])
                        <= float(policy["rest_max_feature_high_error_fraction"])
                        and float(feature_metrics["largest_connected_high_error_fraction"])
                        <= float(policy["rest_max_largest_connected_high_error_fraction"])
                        and float(feature_metrics["p999_rgba_l1"])
                        <= float(policy["rest_max_feature_p999_rgba_l1"])
                        and float(feature_metrics["edge_recall_1px"])
                        >= float(policy["rest_min_feature_edge_recall_1px"])
                        and float(feature_metrics["edge_precision_1px"])
                        >= float(policy["rest_min_feature_edge_precision_1px"])
                    ),
                },
            )
        )

        diagnostic_path = root / f"V{direction}_diagnostics.npz"
        diagnostic_sha = _save_npz(
            diagnostic_path,
            geometry_visible=visible.astype(np.uint8),
            final_alpha=final_alpha.astype(np.uint8),
            provenance=render.provenance_code.astype(np.uint8),
            owner_face_index=render.owner_face_index.astype(np.int32),
        )
        outputs.append(
            {
                "path": str(diagnostic_path),
                "sha256": diagnostic_sha,
                "authority_class": "CAA_REST_DIAGNOSTIC_MASKS",
                "schema": f"RealSaS.CAARestDiagnosticMasks.V{direction}.v2",
            }
        )

    proof = CAARestRenderProofIR(
        asset_binding_hash=asset.asset_hash,
        static_mesh_qualification_binding_hash=static_mesh.qualification_hash,
        camera_set_binding_hash=cameras.camera_set_hash,
        views=tuple(rows),
        qualification_report={
            "status": "PASS_CAA_REFERENCE_REST" if all_pass else "FAIL_CAA_REFERENCE_REST",
            "every_direction_passed": bool(all_pass),
            "visibility_authority": "RealSaS.VisibilityContract.v2",
            "appearance_authority": "RealSaS.CompleteAppearanceAssetIR.v2",
            "geometry_visibility_appearance_attribution_separated": True,
        },
        proof_hash="",
        metadata={
            "policy": policy,
            "source_observation_set_hash": observation.observation_set_hash,
            "coequal_appearance_product_gate": True,
        },
    )
    proof = replace(proof, proof_hash=caa_rest_render_proof_hash(proof))
    if not all_pass:
        return {
            "status": "FAIL",
            "blockers": ["CAA_REFERENCE_REST_RENDER_PROOF_FAILED"],
            "diagnostics": proof.to_dict(),
        }
    outputs.append(
        write_ir(
            root / "caa_reference_rest_proof.json",
            proof,
            authority_class="CAA_REFERENCE_REST_PROOF",
        )
    )
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "proof_hash": proof.proof_hash,
            "every_direction_passed": True,
            "maximum_alpha_hole_fraction": max(
                row.geometry_visible_final_alpha_hole_fraction for row in rows
            ),
            "maximum_source_locked_p95_rgba_l1": max(
                row.source_locked_p95_rgba_l1 for row in rows
            ),
            "maximum_source_foreground_p95_rgba_l1": max(
                row.source_foreground_p95_rgba_l1 for row in rows
            ),
            "minimum_source_alpha_recall": min(row.source_alpha_recall for row in rows),
            "minimum_source_alpha_precision": min(row.source_alpha_precision for row in rows),
            "maximum_exact_depth_ambiguous_fraction": max(
                float(row.metadata.get("exact_depth_ambiguous_fraction", 0.0))
                for row in rows
            ),
        },
    }
