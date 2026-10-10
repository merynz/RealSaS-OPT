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
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    qualified_dynamic_motion_v2_from_dict,
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
from compiler.realsas_compiler_core.presentation_partition_v2 import (
    presentation_partition_evidence_from_dict,
)
from compiler.realsas_compiler_core.visual_presentation_v1 import (
    qualified_visual_presentation_set_from_dict,
)
from compiler.realsas_compiler_core.runtime_visual_authority_v1 import (
    source_owned_visual_dynamic_integrity_from_dict,
    source_owned_visual_runtime_projection_from_dict,
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
    write_json,
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
    partition_evidence_payload, partition_evidence_path = _stage_json(
        ctx,
        "37_QUALIFIED_PRESENTATION_STRUCTURE",
        "RealSaS.PresentationPartitionEvidenceIR.v2",
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
    source_owned_visual = bool(
        dict(appearance_asset_payload.get("metadata") or {}).get(
            "source_owned_visual_mesh_mode"
        )
    )
    if source_owned_visual:
        (
            visual_presentation_payload,
            visual_presentation_path,
        ) = _stage_json(
            ctx,
            "37_QUALIFIED_PRESENTATION_STRUCTURE",
            "RealSaS.QualifiedVisualPresentationSetIR.v1",
        )
    else:
        visual_presentation_payload = None
        visual_presentation_path = None
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
    partition_evidence = presentation_partition_evidence_from_dict(
        partition_evidence_payload
    )
    graph = qualified_presentation_graph_from_dict(graph_payload)
    appearance_asset = complete_appearance_asset_from_dict(appearance_asset_payload)
    appearance_qualification = complete_appearance_qualification_from_dict(
        appearance_qualification_payload
    )
    visual_presentation = (
        qualified_visual_presentation_set_from_dict(
            visual_presentation_payload
        )
        if source_owned_visual
        else None
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
            str(structure.metadata.get("presentation_partition_evidence_hash") or ""),
            partition_evidence.evidence_hash,
            "PRESENTATION_PARTITION_EVIDENCE",
        ),
        (
            partition_evidence.mesh_binding_hash,
            mesh.mesh_lineage_hash,
            "PRESENTATION_PARTITION_MESH",
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
    if source_owned_visual:
        visual_checks = (
            (
                visual_presentation.mechanical_mesh_binding_hash,
                mesh.mesh_lineage_hash,
                "VISUAL_PRESENTATION_MESH",
            ),
            (
                visual_presentation.appearance_asset_binding_hash,
                appearance_asset.asset_hash,
                "VISUAL_PRESENTATION_APPEARANCE",
            ),
            (
                visual_presentation.appearance_qualification_binding_hash,
                appearance_qualification.qualification_hash,
                "VISUAL_PRESENTATION_APPEARANCE_QUALIFICATION",
            ),
            (
                str(
                    dict(complete.metadata or {}).get(
                        "visual_mesh_set_binding_hash"
                    )
                    or ""
                ),
                visual_presentation.set_hash,
                "VISUAL_PRESENTATION_COMPLETE_PUPPET",
            ),
        )
        for actual, expected, label in visual_checks:
            if actual != expected:
                raise QualificationError(
                    f"V2_AUTHORING_BINDING_DRIFT:{label}"
                )

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
            (
                "authority/presentation_partition_evidence_v2.json",
                partition_evidence_path,
            ),
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

        if source_owned_visual:
            file_rows.append(
                _zip_add_file(
                    archive,
                    "authority/qualified_visual_presentation_v1.json",
                    visual_presentation_path,
                )
            )
            for view in sorted(
                visual_presentation.views,
                key=lambda row: int(row.view_index),
            ):
                file_rows.append(
                    _zip_add_file(
                        archive,
                        (
                            "visual/"
                            f"{view.direction_id}_presentation_mesh_v1.npz"
                        ),
                        Path(view.mesh_npz_path),
                        view.mesh_npz_sha256,
                    )
                )

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
            metadata = dict(texture.metadata or {})
            raw_pages = tuple(metadata.get("pages") or ())
            if raw_pages:
                page_rows = tuple(
                    sorted(
                        (dict(row) for row in raw_pages),
                        key=lambda row: int(row["page_index"]),
                    )
                )
                if tuple(int(row["page_index"]) for row in page_rows) != tuple(
                    range(len(page_rows))
                ):
                    raise QualificationError(
                        "V2_AUTHORING_CAA_PAGE_INDEX_SEQUENCE_DRIFT"
                    )
                if int(metadata.get("page_count", len(page_rows))) != len(page_rows):
                    raise QualificationError(
                        "V2_AUTHORING_CAA_PAGE_COUNT_DRIFT"
                    )
                for row in page_rows:
                    page_index = int(row["page_index"])
                    file_rows.append(
                        _zip_add_file(
                            archive,
                            (
                                f"appearance/{texture.direction_id}_appearance_"
                                f"p{page_index}.png"
                            ),
                            Path(str(row["path"])),
                            str(row["sha256"]),
                        )
                    )
            else:
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
            "presentation_partition_evidence_hash": partition_evidence.evidence_hash,
            "presentation_graph_hash": graph.presentation_lineage_hash,
            "appearance_asset_hash": appearance_asset.asset_hash,
            "appearance_qualification_hash": appearance_qualification.qualification_hash,
            "motion_hash": motion.motion_lineage_hash,
            "appearance_authority": "COMPLETE_APPEARANCE_AUTHORITY_V2",
            "appearance_page_transport_complete": True,
            "presentation_geometry_mode": (
                "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
                if source_owned_visual
                else "MECHANICAL_CANONICAL_DEPTH_V2"
            ),
            "qualified_visual_presentation_hash": (
                visual_presentation.set_hash
                if source_owned_visual
                else None
            ),
            "runtime_generation_required": False,
            "editable": True,
            "qualification_scope": "SEALED_EXPORTED_STATE_ONLY",
            "reseal_after_edit_required": True,
            "edited_state_inherits_product_pass": False,
            "runtime_export_after_edit_forbidden_until_reseal": True,
            "edit_invalidation_semantics": (
                "ANY_AUTHORITY_EDIT_INVALIDATES_CHANGED_AUTHORITY_AND_ALL_"
                "TRANSITIVE_DOWNSTREAM_PROOFS"
            ),
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
    if str(ctx["ledger"].get("execution_class") or "") == "DEMO_WITNESS":
        root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
        archive_path, archive_sha, authoring_manifest = _build_editable_bundle(ctx, root)
        payload = {
            "schema": "RealSaS.DemoClosureV2.v1",
            "status": "PASS_DEMO_ONLY",
            "run_id": str(ctx["ledger"].get("run_id") or ""),
            "subject_id": str(ctx["ledger"].get("subject_id") or ""),
            "product_pass": False,
            "product_authority_claimed": False,
            "stage13_scientific_pass": False,
            "editable_authoring_archive_sha256": archive_sha,
            "authoring_manifest_sha256": authoring_manifest["manifest_sha256"],
            "note": "Demo closure only; canonical Stage13 scientific qualification did not pass.",
        }
        return {
            "status": "PASS_DEMO_ONLY",
            "outputs": [
                write_json(
                    root / "demo_closure_v2.json",
                    payload,
                    authority_class="DEMO_ONLY_CLOSURE_V2",
                    schema="RealSaS.DemoClosureV2.v1",
                ),
                {
                    "path": str(archive_path),
                    "sha256": archive_sha,
                    "authority_class": "DEMO_ONLY_EDITABLE_PUPPET_ARCHIVE_V2",
                    "schema": "application/x-realsas-editable-v2",
                },
            ],
            "diagnostics": {
                "product_pass": False,
                "product_authority_claimed": False,
                "demo_closure": True,
                "editable_authoring_archive_sha256": archive_sha,
            },
        }
    complete = complete_puppet_state_v2_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.CompletePuppetStateIR.v2",
        )
    )
    dynamic = qualified_dynamic_motion_v2_from_dict(
        stage_output_payload(
            ctx,
            "41_MOTION_DYNAMIC_PROOF",
            "RealSaS.QualifiedDynamicMotionIR.v2",
        )
    )
    appearance_asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    source_owned_visual = bool(
        dict(appearance_asset.metadata or {}).get(
            "source_owned_visual_mesh_mode"
        )
    )
    if source_owned_visual:
        projection = source_owned_visual_runtime_projection_from_dict(
            stage_output_payload(
                ctx,
                "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1",
            )
        )
    else:
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
    if source_owned_visual:
        visual = source_owned_visual_dynamic_integrity_from_dict(
            stage_output_payload(
                ctx,
                "45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
                "RealSaS.SourceOwnedVisualDynamicIntegrityIR.v1",
            )
        )
        visual_integrity_hash = visual.integrity_hash
    else:
        visual = dynamic_visual_integrity_from_dict(
            stage_output_payload(
                ctx,
                "45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
                "RealSaS.DynamicVisualIntegrityIR.v2",
            )
        )
        visual_integrity_hash = visual.visual_integrity_hash
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

    exact_checks = (
        (
            dynamic.mechanical_state_binding_hash,
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
    if source_owned_visual:
        report = dict(visual.qualification_report or {})
        if projection.metadata.get("visual_material_contract"):
            for key in ("material_provenance_passed", "compiled_appearance_exposure_passed"):
                if report.get(key) is not True:
                    raise QualificationError("V2_PRODUCT_CLOSURE_VISUAL_MATERIAL_GATE_NOT_PASS:" + key)
            for key in ("hidden_layer_material_qualified", "semantic_contact_and_order_qualified"):
                if report.get(key) is not True:
                    raise QualificationError("V2_PRODUCT_CLOSURE_FULL_VISUAL_ACCEPTANCE_REQUIRED:" + key)
        if (
            report.get("status")
            != "PASS_SOURCE_OWNED_VISUAL_DYNAMIC_INTEGRITY"
        ):
            raise QualificationError(
                "V2_PRODUCT_CLOSURE_SOURCE_VISUAL_NOT_PASS"
            )
        for key, label in (
            (
                "native_reference_byte_parity_passed",
                "NATIVE_REFERENCE_PARITY",
            ),
            (
                "direct_source_provenance_passed",
                "DIRECT_SOURCE_PROVENANCE",
            ),
            ("all_frame_views_nonempty", "NONEMPTY_FRAME_VIEW"),
            ("visual_orientation_passed", "VISUAL_ORIENTATION"),
            (
                "catastrophic_edge_stretch_passed",
                "CATASTROPHIC_EDGE_STRETCH",
            ),
        ):
            if not bool(report.get(key, False)):
                raise QualificationError(
                    "V2_PRODUCT_CLOSURE_SOURCE_VISUAL_"
                    + label
                    + "_NOT_PASS"
                )
        dynamic_conditioning_mode = (
            "SOURCE_OWNED_VISUAL_2D_STRUCTURAL_INTEGRITY_V1"
        )
    else:
        if (
            visual.qualification_report.get("status")
            != "PASS_DYNAMIC_VISUAL_INTEGRITY"
        ):
            raise QualificationError(
                "V2_PRODUCT_CLOSURE_VISUAL_NOT_PASS"
            )
        if not bool(
            visual.qualification_report.get(
                "dynamic_appearance_conditioning_passed", False
            )
        ):
            raise QualificationError(
                "V2_PRODUCT_CLOSURE_DYNAMIC_APPEARANCE_CONDITIONING_NOT_PASS"
            )
        if visual.dynamic_conditioning_sample_count <= 0:
            raise QualificationError(
                "V2_PRODUCT_CLOSURE_DYNAMIC_APPEARANCE_EVIDENCE_EMPTY"
            )
        if not bool(
            visual.qualification_report.get(
                "interior_shared_edge_continuity_passed", False
            )
        ):
            raise QualificationError(
                "V2_PRODUCT_CLOSURE_SHARED_EDGE_CONTINUITY_NOT_PASS"
            )
        if bool(
            visual.qualification_report.get(
                "cross_component_background_gap_is_crack_authority", True
            )
        ):
            raise QualificationError(
                "V2_PRODUCT_CLOSURE_CROSS_COMPONENT_CRACK_AUTHORITY_FORBIDDEN"
            )
        dynamic_conditioning_mode = (
            "RIGID_INVARIANT_CAA_UV_TO_POSED_SURFACE_METRIC"
        )
    if (
        str(structure.metadata.get("presentation_partition_evidence_hash") or "")
        != partition_evidence.evidence_hash
    ):
        raise QualificationError(
            "V2_PRODUCT_CLOSURE_PRESENTATION_PARTITION_EVIDENCE_DRIFT"
        )
    if not bool(
        partition_evidence.metadata.get(
            "evidence_supported_visual_partition", False
        )
    ):
        raise QualificationError(
            "V2_PRODUCT_CLOSURE_PRESENTATION_PARTITION_NOT_QUALIFIED"
        )

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
        dynamic_visual_integrity_binding_hash=visual_integrity_hash,
        editable_authoring_archive_path=str(archive_path),
        editable_authoring_archive_sha256=archive_sha,
        qualification_report={
            "status": "PASS_PRODUCT_V2",
            "product_pass": True,
            "geometry_authority_passed": True,
            "mechanics_authority_passed": True,
            "appearance_authority_passed": True,
            "presentation_partition_authority_passed": True,
            "dynamic_appearance_conditioning_passed": True,
            "dynamic_appearance_conditioning_mode": (
                dynamic_conditioning_mode
            ),
            "source_owned_visual_dynamic_integrity_passed": (
                source_owned_visual
            ),
            "interior_shared_edge_continuity_passed": True,
            "cross_component_crack_authority_claimed": False,
            "native_visual_integrity_passed": True,
            "editable_authoring_export_passed": True,
            "runtime_generation_required": False,
            "legacy_qualified_appearance_set_used": False,
        },
        product_closure_hash="",
        metadata={
            "authoring_manifest_sha256": authoring_manifest["manifest_sha256"],
            "presentation_partition_evidence_hash": partition_evidence.evidence_hash,
            "dynamic_visual_integrity_hash": visual_integrity_hash,
            "presentation_geometry_mode": (
                "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
                if source_owned_visual
                else "MECHANICAL_CANONICAL_DEPTH_V2"
            ),
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
