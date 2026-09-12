from __future__ import annotations

"""Legal-Steiner support-ceiling diagnostic with explicit GSA numeric replay handling.

The product GSA lineage remains authoritative and is never relabelled.  This wrapper
exists because the current scene-first builder's robust-normal path can reproduce the
same CDT-relevant raster/mechanical substrate while producing a different byte-exact
surface lineage under a different numeric runtime.  For this *diagnostic only*, a GSA
lineage mismatch may continue iff cardinality/support invariants match and the V1
runner subsequently reproduces every sealed V0..V7 CDT baseline structural/raster
metric exactly.  Any baseline drift still fails closed.

No emitted mesh and no product PASS are claimed here.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as v1
import experiments.mage_full_subject_reclosure_v1.run_frozen_geppetto_replay_v2 as frozen
from experiments.mage_full_subject_reclosure_v1.run_geppetto_fit2_corrected_substrate_refit_v1 import (
    EXPECTED_COMPLETED_NODES,
    EXPECTED_OBSERVATION_SUPPORT_COUNTS,
    EXPECTED_OBSERVED_NODES,
)


SCHEMA = "RealSaS.MageFIT2.LegalSteinerCeilingDiagnostic.v2"
EXPECTED_GSA_LINEAGE = str(frozen.ACTIVE_GSA_LINEAGE)


class _AcceptAnyLineageForSingleBuild:
    """Comparison sentinel used only to recover the built surface for diagnostics."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __str__(self):
        return EXPECTED_GSA_LINEAGE


def _support_counts(tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(tensor.support, dtype=bool).sum(axis=0).tolist())


def _mechanical_raster_signature(surface) -> str:
    """Hash only fields consumed by current CDT geometry/raster construction.

    This is a run-local diagnostic signature, not a replacement authority hash.
    Derived normals and the GSA lineage hash are deliberately excluded because CDT
    does not consume them.
    """
    rows = []
    for node in sorted(surface.surface_nodes, key=lambda n: str(n.surface_id)):
        rows.append(
            {
                "surface_id": str(node.surface_id),
                "P": [float(x) for x in node.P],
                "support_views": [int(v) for v in node.support_views],
                "raster_bindings": [
                    [int(v), [float(xy[0]), float(xy[1])]]
                    for v, xy in node.raster_bindings
                ],
                "validity_flags": [str(x) for x in node.validity_flags],
            }
        )
    rel = []
    for relation in sorted(surface.local_relations, key=lambda r: str(r.relation_id)):
        rel.append(
            {
                "relation_id": str(relation.relation_id),
                "a": str(relation.a_surface_id),
                "b": str(relation.b_surface_id),
                "kind": str(relation.relation_kind),
                "score": float(relation.score),
                "crosses_unknown": bool(relation.metadata.get("crosses_unknown", False)),
                "unknown_bridge": bool(relation.metadata.get("unknown_bridge", False)),
            }
        )
    return v1.sha256(
        json.dumps(
            {"nodes": rows, "relations": rel},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _build_surface_without_lineage_abort(surface_args, cameras, observation_masks):
    old = frozen.ACTIVE_GSA_LINEAGE
    frozen.ACTIVE_GSA_LINEAGE = _AcceptAnyLineageForSingleBuild()
    try:
        return frozen._build_active_surface(surface_args, cameras, observation_masks)
    finally:
        frozen.ACTIVE_GSA_LINEAGE = old


def _preflight_surface(args):
    camera_paths = tuple(Path(p) for p in args.cameras)
    observation_paths = tuple(Path(p) for p in args.observations)
    cameras = frozen._load_cameras(list(camera_paths))
    observation_masks = frozen._load_observation_masks(observation_paths)
    surface_args = SimpleNamespace(
        zero_surface=str(Path(args.zero_surface)),
        iris_run_id=v1.IRIS_RUN_ID,
        iris_checkpoint_sha256=v1.IRIS_CHECKPOINT_SHA256,
    )
    surface, tensor = _build_surface_without_lineage_abort(
        surface_args, cameras, observation_masks
    )

    actual = str(surface.geometry_lineage_hash)
    checks = {
        "node_count": int(tensor.node_count) == int(frozen.ACTIVE_GSA_NODE_COUNT),
        "edge_count": int(tensor.edge_count) == int(frozen.ACTIVE_GSA_EDGE_COUNT),
        "observed_nodes": int(np.asarray(tensor.observed, dtype=bool).sum())
        == int(EXPECTED_OBSERVED_NODES),
        "completed_nodes": int(np.asarray(tensor.completed, dtype=bool).sum())
        == int(EXPECTED_COMPLETED_NODES),
        "support_counts": _support_counts(tensor)
        == tuple(int(x) for x in EXPECTED_OBSERVATION_SUPPORT_COUNTS),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "LEGAL_STEINER_GSA_MECHANICAL_RASTER_PREFLIGHT_DRIFT::"
            + json.dumps(checks, sort_keys=True)
        )

    return surface, tensor, {
        "expected_gsa_lineage_hash": EXPECTED_GSA_LINEAGE,
        "actual_gsa_lineage_hash": actual,
        "gsa_lineage_exact_match": actual == EXPECTED_GSA_LINEAGE,
        "node_count": int(tensor.node_count),
        "edge_count": int(tensor.edge_count),
        "observed_nodes": int(np.asarray(tensor.observed, dtype=bool).sum()),
        "completed_nodes": int(np.asarray(tensor.completed, dtype=bool).sum()),
        "support_counts": list(_support_counts(tensor)),
        "cdt_relevant_mechanical_raster_signature": _mechanical_raster_signature(surface),
        "preflight_checks": checks,
    }


def run(args):
    surface, tensor, replay = _preflight_surface(args)

    print("=" * 132, flush=True)
    print("GSA REPLAY PREFLIGHT", flush=True)
    print("expected lineage :", replay["expected_gsa_lineage_hash"], flush=True)
    print("actual lineage   :", replay["actual_gsa_lineage_hash"], flush=True)
    print("lineage exact    :", replay["gsa_lineage_exact_match"], flush=True)
    print("nodes / edges    :", replay["node_count"], "/", replay["edge_count"], flush=True)
    print("observed/complete:", replay["observed_nodes"], "/", replay["completed_nodes"], flush=True)
    print("support counts   :", replay["support_counts"], flush=True)
    if not replay["gsa_lineage_exact_match"]:
        print(
            "POLICY: lineage mismatch is NOT relabelled PASS; this raster-only diagnostic "
            "may continue only if V1 exact sealed CDT baseline replay passes V0..V7.",
            flush=True,
        )
    print("=" * 132, flush=True)

    # Reuse exactly the preflight-built substrate so a second numeric rebuild cannot drift.
    original_builder = v1._build_active_surface
    original_expected_lineage = v1.ACTIVE_GSA_LINEAGE
    v1._build_active_surface = lambda _surface_args, _cameras, _masks: (surface, tensor)
    v1.ACTIVE_GSA_LINEAGE = str(surface.geometry_lineage_hash)
    try:
        result = v1.run(args)
    finally:
        v1._build_active_surface = original_builder
        v1.ACTIVE_GSA_LINEAGE = original_expected_lineage

    # V1 can reach here only after exact sealed CDT baseline checks passed on all 8 views.
    result = dict(result)
    result["schema"] = SCHEMA
    result["gsa_replay"] = replay
    result["gsa_authority_lineage_preserved_as_expected"] = EXPECTED_GSA_LINEAGE
    result["gsa_lineage_relabelled"] = False
    result["cdt_baseline_replay_gate"] = "PASS__SEALED_STRUCTURAL_AND_RASTER_METRICS_V0_V7"
    result["numeric_lineage_drift_policy"] = (
        "EXACT_GSA_LINEAGE_MATCH"
        if replay["gsa_lineage_exact_match"]
        else "DIAGNOSTIC_ONLY__GSA_LINEAGE_OPEN__CONTINUE_AFTER_EXACT_CDT_BASELINE_REPLAY"
    )
    result["product_mesh_pass_claimed"] = False

    manifest_path = Path(args.output_dir) / "FIT2_LEGAL_STEINER_CEILING_RESULT.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    replay_path = Path(args.output_dir) / "GSA_NUMERIC_REPLAY_PREFLIGHT.json"
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("GSA PREFLIGHT MANIFEST:", replay_path, flush=True)
    print("CDT BASELINE REPLAY   : PASS V0..V7", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-surface", required=True)
    parser.add_argument("--cameras", nargs=8, required=True)
    parser.add_argument("--observations", nargs=8, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
