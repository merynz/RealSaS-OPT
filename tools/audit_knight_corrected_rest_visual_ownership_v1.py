from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index,
    load_face_uv,
    load_provenance_atlas,
    render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import (
    _ctx,
    _load_texture_pages,
)


def _face_indices(candidate) -> np.ndarray:
    vertex_index = {
        str(vertex.candidate_vertex_id): index
        for index, vertex in enumerate(candidate.vertices)
    }
    return np.asarray(
        [
            [vertex_index[str(vertex_id)] for vertex_id in face]
            for face in candidate.faces
        ],
        dtype=np.int64,
    )


def _vertex_support(candidate):
    identity = np.zeros((len(candidate.vertices),), dtype=bool)
    support_ids: list[tuple[str, ...]] = []
    for index, vertex in enumerate(candidate.vertices):
        coeffs = tuple(vertex.support_binding.coefficients)
        identity[index] = (
            len(coeffs) == 1
            and abs(float(coeffs[0][1]) - 1.0) <= 1e-12
        )
        support_ids.append(tuple(sorted(str(sid) for sid, _weight in coeffs)))
    return identity, support_ids


def _face_classes(candidate):
    faces = _face_indices(candidate)
    identity_vertex, support_ids = _vertex_support(candidate)
    generated_vertex_count = np.count_nonzero(~identity_vertex[faces], axis=1)
    support_surface_count = np.asarray(
        [
            len(
                {
                    sid
                    for vertex_index in face
                    for sid in support_ids[int(vertex_index)]
                }
            )
            for face in faces
        ],
        dtype=np.int32,
    )
    return faces, generated_vertex_count, support_surface_count


def _face_pixel_counts(render, pixel_mask: np.ndarray, face_count: int):
    mask = np.asarray(pixel_mask, dtype=bool)
    contribution = np.asarray(render.contributing_layer_mask, dtype=bool)
    owners = np.asarray(render.layer_owner_face_index, dtype=np.int64)
    if contribution.shape != owners.shape:
        raise RuntimeError("CONTRIBUTION_OWNER_SHAPE_DRIFT")
    if contribution.shape[:2] != mask.shape:
        raise RuntimeError("CONTRIBUTION_MASK_SHAPE_DRIFT")

    flat_pixel = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    selected = mask[:, :, None] & contribution & (owners >= 0)
    selected_faces = owners[selected]
    occurrence_counts = np.bincount(
        selected_faces, minlength=face_count
    ).astype(np.int64)

    pixel_values = np.broadcast_to(
        flat_pixel[:, :, None], owners.shape
    )[selected]
    pair_base = np.int64(face_count + 1)
    pair_code = pixel_values * pair_base + selected_faces
    if len(pair_code):
        unique_pairs = np.unique(pair_code)
        pair_faces = unique_pairs % pair_base
        pixel_counts = np.bincount(
            pair_faces.astype(np.int64), minlength=face_count
        ).astype(np.int64)
    else:
        pixel_counts = np.zeros((face_count,), dtype=np.int64)
    return pixel_counts, occurrence_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ctx = _ctx(args.authority_root.expanduser().resolve(), args.run_id)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(
        sorted(camera_set.cameras, key=lambda camera: int(camera.view_index))
    )
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    _source_rgba, source_masks = _load_source_inputs(ctx, observation)
    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {
        int(row.direction_index): row for row in appearance.textures
    }
    rest = np.asarray(
        [vertex.P for vertex in candidate.vertices], dtype=np.float64
    )
    faces, generated_vertex_count, support_surface_count = _face_classes(candidate)
    face_generated = generated_vertex_count > 0
    face_count = len(candidate.faces)

    report = {
        "schema": "RealSaS.KnightCorrectedRestVisualOwnershipAudit.v1",
        "status": "MEASURED_AUDIT_ONLY__NO_PRODUCT_AUTHORITY",
        "run_id": args.run_id,
        "candidate_lineage_hash": str(candidate.candidate_lineage_hash),
        "appearance_asset_hash": str(appearance.asset_hash),
        "candidate_vertex_count": int(len(candidate.vertices)),
        "candidate_face_count": int(face_count),
        "generated_vertex_count": int(
            np.count_nonzero(
                [
                    not (
                        len(tuple(vertex.support_binding.coefficients)) == 1
                        and abs(float(tuple(vertex.support_binding.coefficients)[0][1]) - 1.0)
                        <= 1e-12
                    )
                    for vertex in candidate.vertices
                ]
            )
        ),
        "faces_with_generated_vertex_count": int(np.count_nonzero(face_generated)),
        "claim_boundary": [
            "This audit measures rest-pose visual ownership only.",
            "It does not mutate geometry, skin, appearance, motion, or product authority.",
            "Extra alpha means rendered CAA alpha outside the exact qualified source foreground mask for the same view.",
            "Contributing-face attribution uses renderer alpha-contributing layers rather than geometry-only front ownership.",
        ],
        "views": [],
    }

    for view in (0, 2):
        texture = _load_texture_pages(texture_by_view[view])
        render = render_caa_reference(
            mesh=candidate,
            camera=cameras[view],
            face_uv=face_uv,
            texture_rgba_u8=texture,
            provenance_atlas=provenance[view],
            face_page_index=face_page,
            positions=rest,
        )
        source_mask = np.asarray(source_masks[view], dtype=bool)
        final_alpha = np.asarray(render.final_alpha, dtype=bool)
        geometry_visible = np.asarray(render.geometry_visible, dtype=bool)
        if final_alpha.shape != source_mask.shape:
            raise RuntimeError(
                f"SOURCE_RENDER_SHAPE_DRIFT:V{view}:{source_mask.shape}:{final_alpha.shape}"
            )

        overlap = final_alpha & source_mask
        extra = final_alpha & ~source_mask
        missing = source_mask & ~final_alpha
        geometry_extra = geometry_visible & ~source_mask
        source_count = int(np.count_nonzero(source_mask))
        final_count = int(np.count_nonzero(final_alpha))
        overlap_count = int(np.count_nonzero(overlap))
        extra_count = int(np.count_nonzero(extra))
        missing_count = int(np.count_nonzero(missing))

        face_pixel_count, face_occurrence_count = _face_pixel_counts(
            render, extra, face_count
        )

        contributing = np.asarray(render.contributing_layer_mask, dtype=bool)
        layer_owner = np.asarray(render.layer_owner_face_index, dtype=np.int64)
        valid = contributing & (layer_owner >= 0)
        generated_contribution = np.zeros_like(valid)
        if np.any(valid):
            generated_contribution[valid] = face_generated[layer_owner[valid]]
        identity_contribution = valid & ~generated_contribution
        extra3 = extra[:, :, None]
        extra_any_generated = extra & np.any(generated_contribution, axis=2)
        extra_any_identity = extra & np.any(identity_contribution, axis=2)

        representative_owner = np.asarray(render.owner_face_index, dtype=np.int64)
        representative_valid = extra & (representative_owner >= 0)
        rep_generated = np.zeros_like(extra)
        if np.any(representative_valid):
            rep_generated[representative_valid] = face_generated[
                representative_owner[representative_valid]
            ]

        provenance_values, provenance_counts = np.unique(
            np.asarray(render.provenance_code, dtype=np.uint8)[extra],
            return_counts=True,
        )

        ranked = np.argsort(face_pixel_count)[::-1]
        top_faces = []
        for face_index in ranked[:100]:
            count = int(face_pixel_count[int(face_index)])
            if count <= 0:
                break
            top_faces.append(
                {
                    "face_index": int(face_index),
                    "extra_pixel_count": count,
                    "extra_contribution_occurrence_count": int(
                        face_occurrence_count[int(face_index)]
                    ),
                    "generated_vertex_count": int(
                        generated_vertex_count[int(face_index)]
                    ),
                    "has_generated_vertex": bool(face_generated[int(face_index)]),
                    "support_surface_count": int(
                        support_surface_count[int(face_index)]
                    ),
                    "candidate_vertex_ids": [
                        str(candidate.faces[int(face_index)][corner])
                        for corner in range(3)
                    ],
                }
            )

        report["views"].append(
            {
                "view_index": int(view),
                "source_foreground_pixel_count": source_count,
                "rendered_final_alpha_pixel_count": final_count,
                "overlap_pixel_count": overlap_count,
                "extra_alpha_pixel_count": extra_count,
                "missing_source_pixel_count": missing_count,
                "rendered_to_source_area_ratio": (
                    None if source_count == 0 else float(final_count / source_count)
                ),
                "source_alpha_recall": (
                    None if source_count == 0 else float(overlap_count / source_count)
                ),
                "source_alpha_precision": (
                    None if final_count == 0 else float(overlap_count / final_count)
                ),
                "extra_fraction_of_rendered_alpha": (
                    None if final_count == 0 else float(extra_count / final_count)
                ),
                "geometry_visible_outside_source_pixel_count": int(
                    np.count_nonzero(geometry_extra)
                ),
                "extra_pixels_with_generated_face_contribution": int(
                    np.count_nonzero(extra_any_generated)
                ),
                "extra_pixels_with_identity_face_contribution": int(
                    np.count_nonzero(extra_any_identity)
                ),
                "extra_pixels_representative_owner_generated": int(
                    np.count_nonzero(rep_generated)
                ),
                "extra_provenance_code_histogram": {
                    str(int(code)): int(count)
                    for code, count in zip(
                        provenance_values.tolist(), provenance_counts.tolist()
                    )
                },
                "top_extra_contributing_faces": top_faces,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_CORRECTED_REST_VISUAL_OWNERSHIP_AUDIT="
        + json.dumps(
            {
                "candidate": report["candidate_lineage_hash"],
                "views": [
                    {
                        "view": row["view_index"],
                        "source": row["source_foreground_pixel_count"],
                        "rendered": row["rendered_final_alpha_pixel_count"],
                        "ratio": row["rendered_to_source_area_ratio"],
                        "precision": row["source_alpha_precision"],
                        "extra": row["extra_alpha_pixel_count"],
                        "extra_generated": row[
                            "extra_pixels_with_generated_face_contribution"
                        ],
                        "extra_identity": row[
                            "extra_pixels_with_identity_face_contribution"
                        ],
                    }
                    for row in report["views"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
