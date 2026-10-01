from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    caa_compile_artifact_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_compile_arrays,
)
from compiler.realsas_compiler_services.orchestrator.adapters.mesh_v2 import (
    _load_surface,
    _load_partition_and_carrier,
    _load_candidate_and_policy,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def _surface_id(vertex):
    b = vertex.support_binding
    if b.mode == "IDENTITY_SURFACE_NODE" and len(b.coefficients) == 1:
        sid, w = b.coefficients[0]
        if abs(float(w) - 1.0) <= 1e-9:
            return str(sid)
    return None


def run(*, authority_root: Path, run_id: str, out_path: Path):
    ctx = _ctx(authority_root, run_id)
    surface = _load_surface(ctx)
    partition, carrier = _load_partition_and_carrier(ctx)
    candidate, _policy = _load_candidate_and_policy(ctx)
    artifact = caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,
            "21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    arrays = _load_compile_arrays(
        artifact,
        required_names={
            "sample_face_index",
            "direct_valid",
            "direct_foreground_donor_valid",
            "direct_rgba",
            "rgba",
            "provenance",
            "source_view",
        },
    )

    sample_face = np.asarray(arrays["sample_face_index"], dtype=np.int64)
    direct_valid = np.asarray(arrays["direct_valid"], dtype=bool)
    donor_valid = np.asarray(arrays["direct_foreground_donor_valid"], dtype=bool)
    direct_rgba = np.asarray(arrays["direct_rgba"], dtype=np.uint8)
    rgba = np.asarray(arrays["rgba"], dtype=np.uint8)
    provenance = np.asarray(arrays["provenance"], dtype=np.uint8)

    if direct_valid.shape[0] != 8 or rgba.shape[0] != 8:
        raise RuntimeError("EXPECTED_EIGHT_DIRECTIONS")
    sample_count = sample_face.shape[0]
    if direct_valid.shape[1] != sample_count or rgba.shape[1] != sample_count:
        raise RuntimeError("SAMPLE_CARDINALITY_DRIFT")

    any_direct = np.any(direct_valid, axis=0)
    any_fg_donor = np.any(donor_valid, axis=0)
    direct_alpha_max = np.max(direct_rgba[:, :, 3], axis=0)
    any_direct_foreground = np.any(
        direct_valid & (direct_rgba[:, :, 3] > 0),
        axis=0,
    )
    caa_alpha_max = np.max(rgba[:, :, 3], axis=0)
    caa_alpha_min = np.min(rgba[:, :, 3], axis=0)
    caa_opaque_any = caa_alpha_max > 0
    globally_unseen = ~any_direct
    background_observed = any_direct & ~any_direct_foreground
    donorless_opaque = (~any_fg_donor) & caa_opaque_any
    ownership_leak_opaque = (~any_direct_foreground) & caa_opaque_any
    globally_unseen_opaque = globally_unseen & caa_opaque_any
    background_only_opaque = background_observed & caa_opaque_any

    vertices = {str(v.candidate_vertex_id): v for v in candidate.vertices}
    rel_by_pair = {}
    rel_kind_hist = Counter()
    for rel in surface.local_relations:
        a = str(rel.a_surface_id)
        b = str(rel.b_surface_id)
        key = tuple(sorted((a, b)))
        rel_by_pair[key] = rel
        rel_kind_hist[str(rel.relation_kind)] += 1

    component_by_surface = {
        str(sid): str(component.component_id)
        for component in partition.components
        for sid in component.surface_ids
    }

    face_count = len(candidate.faces)
    face_sample_count = np.bincount(sample_face, minlength=face_count).astype(np.int64)
    face_suspicious_count = np.bincount(
        sample_face,
        weights=donorless_opaque.astype(np.int64),
        minlength=face_count,
    ).astype(np.int64)
    face_unseen_count = np.bincount(
        sample_face,
        weights=globally_unseen.astype(np.int64),
        minlength=face_count,
    ).astype(np.int64)
    face_fg_count = np.bincount(
        sample_face,
        weights=any_fg_donor.astype(np.int64),
        minlength=face_count,
    ).astype(np.int64)
    face_observed_bg_count = np.bincount(
        sample_face,
        weights=background_observed.astype(np.int64),
        minlength=face_count,
    ).astype(np.int64)
    face_caa_opaque_any_count = np.bincount(
        sample_face,
        weights=(caa_alpha_max > 0).astype(np.int64),
        minlength=face_count,
    ).astype(np.int64)
    face_caa_opaque_all_count = np.bincount(
        sample_face,
        weights=(caa_alpha_min > 0).astype(np.int64),
        minlength=face_count,
    ).astype(np.int64)

    face_rows = []
    signature_agg = defaultdict(lambda: {
        "face_count": 0,
        "sample_count": 0,
        "suspicious_opaque_sample_count": 0,
        "globally_unseen_sample_count": 0,
        "foreground_evidence_sample_count": 0,
    })
    component_agg = defaultdict(lambda: {
        "face_count": 0,
        "sample_count": 0,
        "suspicious_opaque_sample_count": 0,
        "globally_unseen_sample_count": 0,
        "foreground_evidence_sample_count": 0,
    })

    for fi, face in enumerate(candidate.faces):
        ids = tuple(map(str, face))
        vids = [vertices[x] for x in ids]
        sids = tuple(_surface_id(v) for v in vids)
        component_ids = {str(v.component_id) for v in vids}
        component_id = next(iter(component_ids)) if len(component_ids) == 1 else "CROSS_COMPONENT"

        relation_kinds = []
        relation_scores = []
        relation_metadata = []
        if all(sid is not None for sid in sids):
            for a, b in ((sids[0], sids[1]), (sids[1], sids[2]), (sids[2], sids[0])):
                rel = rel_by_pair.get(tuple(sorted((str(a), str(b)))))
                if rel is None:
                    relation_kinds.append("MISSING_RELATION")
                    relation_scores.append(None)
                    relation_metadata.append({})
                else:
                    relation_kinds.append(str(rel.relation_kind))
                    relation_scores.append(float(rel.score))
                    relation_metadata.append(dict(getattr(rel, "metadata", {}) or {}))
        else:
            relation_kinds = ["NON_IDENTITY_SUPPORT"] * 3
            relation_scores = [None] * 3
            relation_metadata = [{}, {}, {}]

        n = int(face_sample_count[fi])
        if n == 0:
            raise RuntimeError(f"FACE_WITHOUT_SAMPLES:{fi}")
        suspicious = int(face_suspicious_count[fi])
        unseen = int(face_unseen_count[fi])
        fg = int(face_fg_count[fi])
        observed_bg = int(face_observed_bg_count[fi])
        caa_opaque_any = int(face_caa_opaque_any_count[fi])
        caa_opaque_all = int(face_caa_opaque_all_count[fi])

        signature = "|".join(sorted(relation_kinds))
        row = {
            "face_index": int(fi),
            "candidate_vertex_ids": ids,
            "source_surface_ids": sids,
            "component_id": component_id,
            "relation_kinds": relation_kinds,
            "relation_scores": relation_scores,
            "relation_metadata": relation_metadata,
            "sample_count": n,
            "foreground_evidence_sample_count": fg,
            "foreground_evidence_fraction": float(fg / n),
            "globally_unseen_sample_count": unseen,
            "globally_unseen_fraction": float(unseen / n),
            "observed_background_sample_count": observed_bg,
            "suspicious_opaque_sample_count": suspicious,
            "suspicious_opaque_fraction": float(suspicious / n),
            "caa_opaque_any_direction_sample_count": caa_opaque_any,
            "caa_opaque_all_directions_sample_count": caa_opaque_all,
            "relation_signature": signature,
        }
        face_rows.append(row)

        for agg in (signature_agg[signature], component_agg[component_id]):
            agg["face_count"] += 1
            agg["sample_count"] += n
            agg["suspicious_opaque_sample_count"] += suspicious
            agg["globally_unseen_sample_count"] += unseen
            agg["foreground_evidence_sample_count"] += fg

    def finalize(mapping):
        rows = []
        for key, value in mapping.items():
            total = max(1, int(value["sample_count"]))
            rows.append({
                "key": key,
                **{k: int(v) for k, v in value.items()},
                "suspicious_opaque_fraction": float(
                    value["suspicious_opaque_sample_count"] / total
                ),
                "globally_unseen_fraction": float(
                    value["globally_unseen_sample_count"] / total
                ),
                "foreground_evidence_fraction": float(
                    value["foreground_evidence_sample_count"] / total
                ),
            })
        return sorted(
            rows,
            key=lambda row: (
                -row["suspicious_opaque_fraction"],
                -row["suspicious_opaque_sample_count"],
                row["key"],
            ),
        )

    suspicious_faces = sorted(
        face_rows,
        key=lambda row: (
            -row["suspicious_opaque_fraction"],
            -row["suspicious_opaque_sample_count"],
            row["face_index"],
        ),
    )

    producer = str(getattr(candidate, "producer_id", ""))
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    result = {
        "schema": "RealSaS.KnightRelationAppearanceOwnershipDiagnostic.v1",
        "run_id": run_id,
        "candidate_producer_id": producer,
        "candidate_metadata": metadata,
        "candidate_face_count": len(candidate.faces),
        "candidate_vertex_count": len(candidate.vertices),
        "all_candidate_vertices_identity_surface": bool(
            all(_surface_id(v) is not None for v in candidate.vertices)
        ),
        "relation_kind_histogram": dict(sorted(rel_kind_hist.items())),
        "sample_summary": {
            "sample_count": int(sample_count),
            "any_direct_sample_count": int(np.count_nonzero(any_direct)),
            "direct_foreground_observation_sample_count": int(np.count_nonzero(any_direct_foreground)),
            "foreground_donor_evidence_sample_count": int(np.count_nonzero(any_fg_donor)),
            "globally_unseen_sample_count": int(np.count_nonzero(globally_unseen)),
            "observed_background_sample_count": int(np.count_nonzero(background_observed)),
            "donorless_opaque_sample_count": int(np.count_nonzero(donorless_opaque)),
            "donorless_opaque_fraction": float(np.mean(donorless_opaque)),
            "ownership_leak_opaque_sample_count": int(np.count_nonzero(ownership_leak_opaque)),
            "ownership_leak_opaque_fraction": float(np.mean(ownership_leak_opaque)),
            "globally_unseen_opaque_sample_count": int(np.count_nonzero(globally_unseen_opaque)),
            "globally_unseen_opaque_fraction_of_all_samples": float(np.mean(globally_unseen_opaque)),
            "background_only_opaque_sample_count": int(np.count_nonzero(background_only_opaque)),
            "background_only_opaque_fraction_of_all_samples": float(np.mean(background_only_opaque)),
        },
        "relation_signature_aggregate": finalize(signature_agg),
        "component_aggregate": finalize(component_agg),
        "top_suspicious_faces": suspicious_faces[:200],
        "face_rows_omitted_count": max(0, len(face_rows) - 200),
        "diagnostic_definition": {
            "donorless_opaque": "NO_QUALIFIED_FOREGROUND_DONOR_IN_ANY_VIEW__BUT_CAA_ALPHA_POSITIVE_IN_AT_LEAST_ONE_DIRECTION",
            "ownership_leak_opaque": "NO_DIRECT_FOREGROUND_OBSERVATION_IN_ANY_VIEW__BUT_CAA_ALPHA_POSITIVE_IN_AT_LEAST_ONE_DIRECTION",
            "globally_unseen_opaque": "NO_DIRECT_SOURCE_OBSERVATION_FOREGROUND_OR_BACKGROUND_IN_ANY_VIEW__BUT_CAA_ALPHA_POSITIVE",
            "background_only_opaque": "DIRECTLY_OBSERVED_ONLY_AS_TRANSPARENT_BACKGROUND__BUT_CAA_ALPHA_POSITIVE_IN_AT_LEAST_ONE_DIRECTION",
            "note": "Ownership-leak and globally-unseen-opaque are the strict visual-ownership diagnostics. Donorless-opaque is broader because silhouette-boundary or grazing direct foreground may be immutable source authority but unsafe as a cross-view donor.",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_RELATION_APPEARANCE_OWNERSHIP_DIAGNOSTIC_PASS", json.dumps({
        "producer": producer,
        "face_count": len(candidate.faces),
        "donorless_opaque_sample_count": result["sample_summary"]["donorless_opaque_sample_count"],
        "ownership_leak_opaque_sample_count": result["sample_summary"]["ownership_leak_opaque_sample_count"],
        "globally_unseen_opaque_sample_count": result["sample_summary"]["globally_unseen_opaque_sample_count"],
    }, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-path", required=True)
    a = p.parse_args()
    run(
        authority_root=Path(a.authority_root).expanduser().resolve(),
        run_id=a.run_id,
        out_path=Path(a.out_path),
    )


if __name__ == "__main__":
    main()
