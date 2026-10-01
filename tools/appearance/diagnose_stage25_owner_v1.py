from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE,
    caa_compile_artifact_from_dict,
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.appearance_quality_v2 import (
    rgba_l1_premultiplied,
)
from compiler.realsas_compiler_core.geometry_substrate_v2 import (
    geometry_substrate_evidence_from_dict,
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
    _load_compile_arrays,
    _load_source_inputs,
    _load_texture_pages,
)


def _context(authority_root: Path, run_id: str) -> dict:
    run_root = (authority_root / "runs" / run_id).resolve()
    return {
        "repo_root": Path(".").resolve(),
        "authority_root": authority_root.resolve(),
        "run_root": run_root,
        "run_id": run_id,
        "run_manifest_path": run_root / "run_manifest.json",
        "run_manifest": json.loads(
            (run_root / "run_manifest.json").read_text(encoding="utf-8")
        ),
        "ledger": json.loads(
            (run_root / "ACTIVE_RUN_V2.json").read_text(encoding="utf-8")
        ),
        "stage": {"id": "CAA_STAGE25_OWNER_DIAGNOSTIC"},
    }


def _top_faces(face_index: np.ndarray, mask: np.ndarray, limit: int = 20):
    ids = face_index[mask]
    ids = ids[ids >= 0]
    counts = Counter(map(int, ids.tolist()))
    return [
        {"face_index": int(face), "pixel_count": int(count)}
        for face, count in counts.most_common(limit)
    ]


def diagnose(
    *,
    authority_root: Path,
    run_id: str,
    high_error_cut: float = 0.10,
) -> dict:
    ctx = _context(authority_root, run_id)
    geometry = geometry_substrate_evidence_from_dict(
        stage_output_payload(
            ctx,
            "13_GEOMETRY_SUBSTRATE_QUALIFIED",
            "RealSaS.GeometrySubstrateQualificationIR.v2",
        )
    )
    geometry_by_view = {
        int(row.view_index): row for row in geometry.views
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
    compile_artifact = caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,
            "21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    compile_arrays = _load_compile_arrays(
        compile_artifact,
        required_names={"provenance"},
    )
    compile_provenance = np.asarray(
        compile_arrays["provenance"], dtype=np.uint8
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    source_rgba, source_masks = _load_source_inputs(ctx, observation)
    face_uv = load_face_uv(asset)
    face_page_index = load_face_page_index(asset)
    provenance_all = load_provenance_atlas(asset)
    camera_by_view = {int(row.view_index): row for row in cameras.cameras}
    texture_by_view = {
        int(row.direction_index): row for row in asset.textures
    }

    view_rows = []
    aggregate = defaultdict(int)
    provenance_names = {
        int(code): name for name, code in CAA_PROVENANCE.items()
    }

    for view in range(8):
        texture = _load_texture_pages(texture_by_view[view])
        render = render_caa_reference(
            mesh=candidate,
            camera=camera_by_view[view],
            face_uv=face_uv,
            texture_rgba_u8=texture,
            provenance_atlas=provenance_all[view],
            face_page_index=face_page_index,
        )
        source = np.asarray(source_rgba[view], dtype=np.uint8)
        source_fg = np.asarray(source_masks[view], dtype=bool)
        visible = np.asarray(render.geometry_visible, dtype=bool)
        final_alpha = np.asarray(render.final_alpha, dtype=bool)
        prov = np.asarray(render.provenance_code, dtype=np.uint8)
        owner = np.asarray(render.owner_face_index, dtype=np.int32)

        error = np.zeros(source_fg.shape, dtype=np.float64)
        if np.any(source_fg):
            error[source_fg] = rgba_l1_premultiplied(
                render.straight_rgba_u8[source_fg],
                source[source_fg],
            )
        high_error = source_fg & (error > float(high_error_cut))

        structure = np.ones((3, 3), dtype=bool)
        interior = binary_erosion(
            source_fg,
            structure=structure,
            iterations=1,
            border_value=0,
        )
        interior_2px = binary_erosion(
            source_fg,
            structure=structure,
            iterations=2,
            border_value=0,
        )
        interior_3px = binary_erosion(
            source_fg,
            structure=structure,
            iterations=3,
            border_value=0,
        )
        boundary = source_fg & ~interior
        boundary_2px = source_fg & ~interior_2px
        boundary_3px = source_fg & ~interior_3px

        source_geometry_miss = source_fg & ~visible
        false_positive_alpha = final_alpha & ~source_fg
        source_alpha_miss = source_fg & ~final_alpha
        direct = prov == int(CAA_PROVENANCE["DIRECT_SOURCE"])
        completed = (
            (prov == int(CAA_PROVENANCE["OTHER_VIEW_SOURCE"]))
            | (prov == int(CAA_PROVENANCE["COMPILED_LOCAL_HARMONIC"]))
            | (
                prov
                == int(CAA_PROVENANCE["CANONICAL_GLOBAL_COMPLETION"])
            )
        )
        unsupported = prov == int(CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"])
        padding = prov == 255

        provenance_rows = []
        for code in sorted(set(map(int, np.unique(prov).tolist()))):
            take = source_fg & (prov == code)
            if not np.any(take):
                continue
            provenance_rows.append(
                {
                    "code": code,
                    "name": provenance_names.get(code, "PADDING_OR_UNKNOWN"),
                    "source_foreground_pixel_count": int(np.count_nonzero(take)),
                    "source_foreground_fraction": float(
                        np.count_nonzero(take)
                    )
                    / float(max(1, np.count_nonzero(source_fg))),
                    "mean_rgba_l1": float(np.mean(error[take])),
                    "high_error_pixel_count": int(
                        np.count_nonzero(high_error & take)
                    ),
                }
            )

        compile_codes = compile_provenance[view]
        atlas_codes = np.asarray(provenance_all[view], dtype=np.uint8)
        atlas_surface = atlas_codes != 255
        visible_count = int(np.count_nonzero(visible))
        compile_count = int(len(compile_codes))
        atlas_surface_count = int(np.count_nonzero(atlas_surface))

        lineage_counts = {}
        for name, code in CAA_PROVENANCE.items():
            code = int(code)
            lineage_counts[name] = {
                "stage21_sample_count": int(
                    np.count_nonzero(compile_codes == code)
                ),
                "stage21_fraction": float(
                    np.count_nonzero(compile_codes == code)
                ) / float(max(1, compile_count)),
                "stage23_surface_texel_count": int(
                    np.count_nonzero(atlas_surface & (atlas_codes == code))
                ),
                "stage23_surface_texel_fraction": float(
                    np.count_nonzero(atlas_surface & (atlas_codes == code))
                ) / float(max(1, atlas_surface_count)),
                "stage25_visible_pixel_count": int(
                    np.count_nonzero(visible & (prov == code))
                ),
                "stage25_visible_pixel_fraction": float(
                    np.count_nonzero(visible & (prov == code))
                ) / float(max(1, visible_count)),
            }

        g = geometry_by_view[view]
        row = {
            "view_index": int(view),
            "stage13_geometry": {
                "silhouette_recall": float(g.silhouette_recall),
                "silhouette_precision": float(g.silhouette_precision),
                "largest_coherent_hole_fraction": float(
                    g.largest_coherent_hole_fraction
                ),
                "interior_uncovered_fraction": float(
                    g.interior_uncovered_fraction
                ),
                "component_recall": float(g.component_recall),
                "silhouette_edge_p95_px": float(
                    g.silhouette_edge_p95_px
                ),
                "passed": bool(g.passed),
            },
            "provenance_lineage_counts": lineage_counts,
            "source_foreground_pixel_count": int(np.count_nonzero(source_fg)),
            "geometry_visible_pixel_count": int(np.count_nonzero(visible)),
            "final_alpha_pixel_count": int(np.count_nonzero(final_alpha)),
            "source_geometry_miss_pixel_count": int(
                np.count_nonzero(source_geometry_miss)
            ),
            "source_geometry_miss_fraction": float(
                np.count_nonzero(source_geometry_miss)
            )
            / float(max(1, np.count_nonzero(source_fg))),
            "source_alpha_miss_pixel_count": int(
                np.count_nonzero(source_alpha_miss)
            ),
            "false_positive_alpha_pixel_count": int(
                np.count_nonzero(false_positive_alpha)
            ),
            "false_positive_alpha_fraction_of_render": float(
                np.count_nonzero(false_positive_alpha)
            )
            / float(max(1, np.count_nonzero(final_alpha))),
            "high_error_cut_rgba_l1": float(high_error_cut),
            "high_error_pixel_count": int(np.count_nonzero(high_error)),
            "high_error_fraction_of_source_foreground": float(
                np.count_nonzero(high_error)
            )
            / float(max(1, np.count_nonzero(source_fg))),
            "boundary_high_error_pixel_count": int(
                np.count_nonzero(high_error & boundary)
            ),
            "boundary_2px_high_error_pixel_count": int(
                np.count_nonzero(high_error & boundary_2px)
            ),
            "boundary_3px_high_error_pixel_count": int(
                np.count_nonzero(high_error & boundary_3px)
            ),
            "interior_high_error_pixel_count": int(
                np.count_nonzero(high_error & interior)
            ),
            "deep_interior_3px_high_error_pixel_count": int(
                np.count_nonzero(high_error & interior_3px)
            ),
            "boundary_high_error_fraction_of_high_error": float(
                np.count_nonzero(high_error & boundary)
            )
            / float(max(1, np.count_nonzero(high_error))),
            "boundary_2px_high_error_fraction_of_high_error": float(
                np.count_nonzero(high_error & boundary_2px)
            )
            / float(max(1, np.count_nonzero(high_error))),
            "boundary_3px_high_error_fraction_of_high_error": float(
                np.count_nonzero(high_error & boundary_3px)
            )
            / float(max(1, np.count_nonzero(high_error))),
            "direct_source_foreground_pixel_count": int(
                np.count_nonzero(source_fg & direct)
            ),
            "direct_source_high_error_pixel_count": int(
                np.count_nonzero(high_error & direct)
            ),
            "completed_source_foreground_pixel_count": int(
                np.count_nonzero(source_fg & completed)
            ),
            "completed_source_high_error_pixel_count": int(
                np.count_nonzero(high_error & completed)
            ),
            "unsupported_visible_pixel_count": int(
                np.count_nonzero(visible & unsupported)
            ),
            "padding_visible_pixel_count": int(
                np.count_nonzero(visible & padding)
            ),
            "provenance_on_source_foreground": provenance_rows,
            "top_high_error_owner_faces": _top_faces(owner, high_error),
            "top_false_positive_owner_faces": _top_faces(
                owner, false_positive_alpha
            ),
            "top_geometry_miss_nearest_owner_faces": [],
        }
        view_rows.append(row)

        for key in (
            "source_geometry_miss_pixel_count",
            "source_alpha_miss_pixel_count",
            "false_positive_alpha_pixel_count",
            "high_error_pixel_count",
            "boundary_high_error_pixel_count",
            "boundary_2px_high_error_pixel_count",
            "boundary_3px_high_error_pixel_count",
            "interior_high_error_pixel_count",
            "deep_interior_3px_high_error_pixel_count",
            "direct_source_high_error_pixel_count",
            "completed_source_high_error_pixel_count",
            "unsupported_visible_pixel_count",
            "padding_visible_pixel_count",
        ):
            aggregate[key] += int(row[key])

    high = int(aggregate["high_error_pixel_count"])
    aggregate["boundary_high_error_fraction_of_high_error"] = (
        float(aggregate["boundary_high_error_pixel_count"])
        / float(max(1, high))
    )
    aggregate["interior_high_error_fraction_of_high_error"] = (
        float(aggregate["interior_high_error_pixel_count"])
        / float(max(1, high))
    )
    aggregate["boundary_2px_high_error_fraction_of_high_error"] = (
        float(aggregate["boundary_2px_high_error_pixel_count"])
        / float(max(1, high))
    )
    aggregate["boundary_3px_high_error_fraction_of_high_error"] = (
        float(aggregate["boundary_3px_high_error_pixel_count"])
        / float(max(1, high))
    )
    aggregate["deep_interior_3px_high_error_fraction_of_high_error"] = (
        float(aggregate["deep_interior_3px_high_error_pixel_count"])
        / float(max(1, high))
    )
    return {
        "schema": "RealSaS.CAAStage25OwnerDiagnostic.v1",
        "status": "MEASURED",
        "run_id": run_id,
        "high_error_cut_rgba_l1": float(high_error_cut),
        "stage13_policy": dict(geometry.policy),
        "stage13_qualification_report": dict(
            geometry.qualification_report
        ),
        "stage21_completion_rows": list(
            dict(compile_artifact.metadata or {}).get(
                "completion_rows", []
            )
        ),
        "stage21_global_completion": {
            "sample_count": int(
                dict(compile_artifact.metadata or {}).get(
                    "canonical_global_completion_sample_count", 0
                )
            ),
            "fraction": float(
                dict(compile_artifact.metadata or {}).get(
                    "canonical_global_completion_fraction", 0.0
                )
            ),
            "globally_unseen_dense_sample_count": int(
                dict(compile_artifact.metadata or {}).get(
                    "globally_unseen_dense_sample_count", 0
                )
            ),
            "metadata": dict(
                dict(compile_artifact.metadata or {}).get(
                    "canonical_global_completion_metadata", {}
                )
            ),
        },
        "views": view_rows,
        "aggregate": dict(aggregate),
        "interpretation_contract": {
            "geometry_miss_owner": (
                "SOURCE_FOREGROUND_PIXEL_WITHOUT_GEOMETRY_VISIBILITY"
            ),
            "direct_render_mismatch_owner": (
                "DIRECT_SOURCE_PROVENANCE_WITH_RGBA_ERROR_ABOVE_CUT"
            ),
            "completion_intrusion_owner": (
                "SOURCE_FOREGROUND_PIXEL_RENDERED_FROM_NON_DIRECT_APPEARANCE"
            ),
            "boundary_vs_interior": (
                "3X3_SOURCE_MASK_EROSION__DIAGNOSTIC_ONLY"
            ),
            "thresholds_changed": False,
            "product_authority_minted": False,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--high-error-cut", type=float, default=0.10)
    a = p.parse_args()
    report = diagnose(
        authority_root=Path(a.authority_root),
        run_id=a.run_id,
        high_error_cut=float(a.high_error_cut),
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "CAA_STAGE25_OWNER_DIAGNOSTIC",
        json.dumps(report["aggregate"], sort_keys=True),
    )


if __name__ == "__main__":
    main()
