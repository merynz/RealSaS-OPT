from __future__ import annotations

"""V2 Stage46: exact product closure and editable authoring export."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import zipfile

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
    complete_appearance_qualification_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v1 import (
    qualified_dynamic_motion_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_motion_v2_from_dict,
    qualified_presentation_graph_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.product_state_v2 import (
    complete_puppet_state_v2_from_dict,
    presentation_structure_v2_from_dict,
)
from compiler.realsas_compiler_core.runtime_authority_v2 import (
    ProductClosureV2IR,
    dynamic_visual_integrity_from_dict,
    native_playback_from_dict,
    product_closure_hash,
    runtime_package_seal_from_dict,
    runtime_projection_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    resolved_path,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_core.types import QualificationError


def _canonical_json_bytes(payload: dict) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _zip_add_bytes(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
) -> None:
    info = zipfile.ZipInfo(str(name))
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def _zip_add_file(
    archive: zipfile.ZipFile,
    name: str,
    path: Path,
    expected_sha256: str | None = None,
) -> dict:
    path = resolved_path(str(path))
    if not path.is_file():
        raise QualificationError(f"V2_AUTHORING_FILE_MISSING:{path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise QualificationError(f"V2_AUTHORING_FILE_SHA_DRIFT:{path}")
    payload = path.read_bytes()
    _zip_add_bytes(archive, name, payload)
    return {
        "name": str(name),
        "sha256": digest,
        "bytes": len(payload),
    }


def _stage_json(
    ctx: dict,
    stage_id: str,
    schema: str,
) -> tuple[dict, Path]:
    payload = stage_output_payload(ctx, stage_id, schema)
    row = next(row for row in ctx["ledger"]["stages"] if row["id"] == stage_id)
    matches = [out for out in row.get("outputs", ()) if out.get("schema") == schema]
    if len(matches) != 1:
        raise QualificationError(
            f"V2_AUTHORING_STAGE_SCHEMA_CARDINALITY:{stage_id}:{schema}"
        )
    path = resolved_path(matches[0]["path"])
    return payload, path


def _build_editable_bundle(ctx: dict, root: Path) -> tuple[Path, str, dict]:
    complete_payload, complete_path = _stage_json(
        ctx,
        "38_CANONICAL_PUPPET_SEALED",
        "RealSaS.CompletePuppetStateIR.v2",
    )
    mechanical_payload, mechanical_path = _stage_json(
        ctx,
        "38_CANONICAL_PUPPET_SEALED",
        "RealSaS.CanonicalPuppetStateIR.v1",
    )
    graph_payload, graph_path = _stage_json(
        ctx,
        "38_CANONICAL_PUPPET_SEALED",
        "RealSaS.QualifiedPresentationGraphIR.v2",
    )
    structure_payload, structure_path = _stage_json(
        ctx,
        "37_QUALIFIED_PRESENTATION_STRUCTURE",
        "RealSaS.QualifiedPresentationStructureIR.v2",
    )
    skeleton_payload, skeleton_path = _stage_json(
        ctx,
        "28_SKELETON_QUALIFIED",
        "RealSaS.QualifiedSkeletonIR.v1",
    )
    mesh_payload, mesh_path = _stage_json(
        ctx,
        "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "RealSaS.QualifiedMeshIR.v1",
    )
    mesh_skin_payload, mesh_skin_path = _stage_json(
        ctx,
        "36_QUALIFIED_MESH_SKIN_TRANSFER",
        "RealSaS.QualifiedMeshSkinIR.v1",
    )
    appearance_asset_payload, appearance_asset_path = _stage_json(
        ctx,
        "23_COMPLETE_APPEARANCE_ASSET_BAKED",
        "RealSaS.CompleteAppearanceAssetIR.v2",
    )
    appearance_qualification_payload, appearance_qualification_path = _stage_json(
        ctx,
        "24_COMPLETE_APPEARANCE_QUALIFIED",
        "RealSaS.CompleteAppearanceQualificationIR.v2",
    )
    motion_constraints_payload, motion_constraints_path = _stage_json(
        ctx,
        "40_MOTION_COMPILE_RUN",
        "RealSaS.MotionCompileConstraintSetIR.v2",
    )
    motion_payload, motion_path = _stage_json(
        ctx,
        "40_MOTION_COMPILE_RUN",
        "RealSaS.QualifiedMotionIR.v2",
    )

    complete = complete_puppet_state_v2_from_dict(complete_payload)
    skeleton = qualified_skeleton_from_dict(skeleton_payload)
    mesh = qualified_mesh_from_dict(mesh_payload)
    mesh_skin = qualified_mesh_skin_from_dict(mesh_skin_payload)
    structure = presentation_structure_v2_from_dict(structure_payload)
    graph = qualified_presentation_graph_from_dict(graph_payload)
    appearance_asset = complete_appearance_asset_from_dict(appearance_asset_payload)
    appearance_qualification = complete_appearance_qualification_from_dict(
        appearance_qualification_payload
    )
    motion = qualified_motion_v2_from_dict(motion_payload)

    exact_checks = (
        (
            complete.skeleton_binding_hash,
            skeleton.skeleton_lineage_hash,
            "SKELETON",
        ),
        (
            complete.mesh_binding_hash,
            mesh.mesh_lineage_hash,
            "MESH",
        ),
        (
            complete.mesh_skin_binding_hash,
            mesh_skin.mesh_skin_lineage_hash,
            "MESH_SKIN",
        ),
        (
            complete.presentation_structure_binding_hash,
            structure.structure_hash,
            "PRESENTATION_STRUCTURE",
        ),
        (
            complete.presentation_graph_binding_hash,
            graph.presentation_lineage_hash,
            "PRESENTATION_GRAPH",
        ),
        (
            complete.complete_appearance_asset_binding_hash,
            appearance_asset.asset_hash,
            "CAA_ASSET",
        ),
        (
            complete.complete_appearance_qualification_binding_hash,
            appearance_qualification.qualification_hash,
            "CAA_QUALIFICATION",
        ),
        (
            motion.product_state_binding_hash,
            complete.mechanical_state_binding_hash,
            "MOTION_MECHANICAL_STATE",
        ),
    )
    for actual, expected, label in exact_checks:
        if actual != expected:
            raise QualificationError(f"V2_AUTHORING_BINDING_DRIFT:{label}")

    archive_path = root / "editable_puppet_v2.rsedit"
    root.mkdir(parents=True, exist_ok=True)
    file_rows = []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, path in (
            ("authority/complete_puppet_state_v2.json", complete_path),
            ("authority/mechanical_puppet_state.json", mechanical_path),
            ("authority/qualified_skeleton.json", skeleton_path),
            ("authority/qualified_mesh.json", mesh_path),
            ("authority/qualified_mesh_skin.json", mesh_skin_path),
            ("authority/presentation_structure_v2.json", structure_path),
            ("authority/presentation_graph.json", graph_path),
            ("appearance/complete_appearance_asset.json", appearance_asset_path),
            (
                "appearance/complete_appearance_qualification.json",
                appearance_qualification_path,
            ),
            ("motion/motion_compile_constraints.json", motion_constraints_path),
            ("motion/qualified_motion.json", motion_path),
        ):
            file_rows.append(_zip_add_file(archive, name, path))

        file_rows.append(
            _zip_add_file(
                archive,
                "appearance/surface_uv.npz",
                Path(appearance_asset.uv_npz_path),
                appearance_asset.uv_npz_sha256,
            )
        )
        file_rows.append(
            _zip_add_file(
                archive,
                "appearance/provenance_atlas.npz",
                Path(appearance_asset.provenance_npz_path),
                appearance_asset.provenance_npz_sha256,
            )
        )
        for texture in sorted(
            appearance_asset.textures,
            key=lambda row: row.direction_index,
        ):
            file_rows.append(
                _zip_add_file(
                    archive,
                    f"appearance/{texture.direction_id}_appearance.png",
                    Path(texture.transport_png_path),
                    texture.transport_png_sha256,
                )
            )

        bundle_manifest = {
            "schema": "RealSaS.EditablePuppetBundleManifest.v2",
            "complete_puppet_hash": complete.complete_puppet_hash,
            "mechanical_state_hash": complete.mechanical_state_binding_hash,
            "skeleton_hash": skeleton.skeleton_lineage_hash,
            "mesh_hash": mesh.mesh_lineage_hash,
            "mesh_skin_hash": mesh_skin.mesh_skin_lineage_hash,
            "presentation_structure_hash": structure.structure_hash,
            "presentation_graph_hash": graph.presentation_lineage_hash,
            "appearance_asset_hash": appearance_asset.asset_hash,
            "appearance_qualification_hash": appearance_qualification.qualification_hash,
            "motion_hash": motion.motion_lineage_hash,
            "appearance_authority": "COMPLETE_APPEARANCE_AUTHORITY_V2",
            "runtime_generation_required": False,
            "editable": True,
            "files": tuple(sorted(file_rows, key=lambda row: row["name"])),
        }
        bundle_manifest["manifest_sha256"] = hashlib.sha256(
            _canonical_json_bytes(bundle_manifest)
        ).hexdigest()
        _zip_add_bytes(
            archive,
            "manifest.json",
            _canonical_json_bytes(bundle_manifest),
        )

    archive_sha = sha256_file(archive_path)
    return archive_path, archive_sha, bundle_manifest


def seal_product_closure_stage(ctx: dict) -> dict:
    complete = complete_puppet_state_v2_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.CompletePuppetStateIR.v2",
        )
    )
    dynamic = qualified_dynamic_motion_from_dict(
        stage_output_payload(
            ctx,
            "41_MOTION_DYNAMIC_PROOF",
            "RealSaS.QualifiedDynamicMotionIR.v1",
        )
    )
    projection = runtime_projection_from_dict(
        stage_output_payload(
            ctx,
            "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
            "RealSaS.RuntimeProjectionIR.v2",
        )
    )
    package = runtime_package_seal_from_dict(
        stage_output_payload(
            ctx,
            "43_RSS_MATERIALIZE_COMPACT",
            "RealSaS.RuntimePackageSealIR.v2",
        )
    )
    native = native_playback_from_dict(
        stage_output_payload(
            ctx,
            "44_NATIVE_PACKAGE_OPEN_PLAYBACK",
            "RealSaS.NativePlaybackIR.v2",
        )
    )
    visual = dynamic_visual_integrity_from_dict(
        stage_output_payload(
            ctx,
            "45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
            "RealSaS.DynamicVisualIntegrityIR.v2",
        )
    )

    exact_checks = (
        (
            dynamic.product_state_binding_hash,
            complete.mechanical_state_binding_hash,
            "DYNAMIC_MECHANICAL_STATE",
        ),
        (
            projection.complete_puppet_binding_hash,
            complete.complete_puppet_hash,
            "PROJECTION_COMPLETE_PUPPET",
        ),
        (
            projection.dynamic_motion_binding_hash,
            dynamic.dynamic_motion_hash,
            "PROJECTION_DYNAMIC_MOTION",
        ),
        (
            package.projection_binding_hash,
            projection.projection_hash,
            "PACKAGE_PROJECTION",
        ),
        (
            native.package_binding_hash,
            package.package_hash,
            "NATIVE_PACKAGE",
        ),
        (
            native.projection_binding_hash,
            projection.projection_hash,
            "NATIVE_PROJECTION",
        ),
        (
            visual.package_binding_hash,
            package.package_hash,
            "VISUAL_PACKAGE",
        ),
        (
            visual.projection_binding_hash,
            projection.projection_hash,
            "VISUAL_PROJECTION",
        ),
        (
            visual.native_playback_binding_hash,
            native.playback_hash,
            "VISUAL_NATIVE",
        ),
    )
    for actual, expected, label in exact_checks:
        if actual != expected:
            raise QualificationError(f"V2_PRODUCT_CLOSURE_BINDING_DRIFT:{label}")
    if (
        visual.qualification_report.get("status")
        != "PASS_DYNAMIC_VISUAL_INTEGRITY"
    ):
        raise QualificationError("V2_PRODUCT_CLOSURE_VISUAL_NOT_PASS")

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    archive_path, archive_sha, authoring_manifest = _build_editable_bundle(
        ctx, root
    )
    value = ProductClosureV2IR(
        complete_puppet_binding_hash=complete.complete_puppet_hash,
        dynamic_motion_binding_hash=dynamic.dynamic_motion_hash,
        runtime_projection_binding_hash=projection.projection_hash,
        runtime_package_binding_hash=package.package_hash,
        native_playback_binding_hash=native.playback_hash,
        dynamic_visual_integrity_binding_hash=visual.visual_integrity_hash,
        editable_authoring_archive_path=str(archive_path),
        editable_authoring_archive_sha256=archive_sha,
        qualification_report={
            "status": "PASS_PRODUCT_V2",
            "product_pass": True,
            "geometry_authority_passed": True,
            "mechanics_authority_passed": True,
            "appearance_authority_passed": True,
            "native_visual_integrity_passed": True,
            "editable_authoring_export_passed": True,
            "runtime_generation_required": False,
            "legacy_qualified_appearance_set_used": False,
        },
        product_closure_hash="",
        metadata={
            "authoring_manifest_sha256": authoring_manifest["manifest_sha256"],
            "geometry_mechanics_appearance_coequal": True,
            "product_stage": 46,
        },
    )
    value = replace(value, product_closure_hash=product_closure_hash(value))
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "product_closure_v2.json",
                value,
                authority_class="PRODUCT_PASS_CLOSURE_V2",
            ),
            {
                "path": str(archive_path),
                "sha256": archive_sha,
                "authority_class": "EDITABLE_PUPPET_AUTHORING_ARCHIVE_V2",
                "schema": "application/x-realsas-editable-v2",
            },
        ],
        "diagnostics": {
            "product_closure_hash": value.product_closure_hash,
            "product_pass": True,
            "editable_authoring_archive_sha256": archive_sha,
            "legacy_qualified_appearance_set_used": False,
            "geometry_mechanics_appearance_coequal": True,
        },
    }
