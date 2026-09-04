from __future__ import annotations

from dataclasses import replace

import numpy as np

from compiler.realsas_compiler_core.types import ObservationEvidenceIR, RiggingSurfaceIR
from experiments.geppetto_arachne_r6_20260901.oracle_substrate_synthetic_v1 import (
    PERSISTENCE_TOLERANCE,
    ShellSample,
    Witness,
    build_u1_evidence,
)
from experiments.iris_reprojection_v2_20260831.persistence_adapter_v2 import (
    attach_dtb_nd1_from_evidence,
    compile_surface_v2,
)


STRESS_SPECS = {
    "CORE_75": {
        "x_abs_max": 0.35,
        "y_min": -0.20,
        "y_max": 0.55,
    },
    "CORE_60": {
        "x_abs_max": 0.45,
        "y_min": -0.35,
        "y_max": 0.55,
    },
}


def _normalized_points(samples: tuple[ShellSample, ...]) -> np.ndarray:
    p = np.asarray([s.P for s in samples], np.float64)
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half = np.maximum(half, 1e-12)
    return (p - mid[None]) / half[None]


def hidden_sample_indices(samples: tuple[ShellSample, ...], stress_name: str) -> set[int]:
    if stress_name not in STRESS_SPECS:
        raise KeyError(stress_name)
    spec = STRESS_SPECS[stress_name]
    n = _normalized_points(samples)
    hidden = (
        (np.abs(n[:, 0]) < float(spec["x_abs_max"]))
        & (n[:, 1] > float(spec["y_min"]))
        & (n[:, 1] < float(spec["y_max"]))
    )
    return {int(samples[i].sample_index) for i in np.flatnonzero(hidden)}


def _group_sample_index(group_id: str) -> int:
    if not group_id.startswith("G"):
        raise ValueError(f"unexpected hypothesis group id: {group_id}")
    return int(group_id[1:])


def _sigma_sample_index(key: str) -> int:
    # Current oracle metadata uses S###:M0.
    if not key.startswith("S") or ":M" not in key:
        raise ValueError(f"unexpected mode_sigma key: {key}")
    return int(key[1:].split(":M", 1)[0])


def filter_evidence_before_gsa(
    evidence: ObservationEvidenceIR,
    hidden_indices: set[int],
    *,
    stress_name: str,
) -> ObservationEvidenceIR:
    groups0 = dict(evidence.metadata.get("hypothesis_groups", {}))
    anchors0 = dict(evidence.metadata.get("hypothesis_anchor_raster", {}))
    sigma0 = dict(evidence.metadata.get("mode_sigma", {}))

    groups = {
        gid: tuple(obs_ids)
        for gid, obs_ids in groups0.items()
        if _group_sample_index(gid) not in hidden_indices
    }
    keep_ids = {obs_id for obs_ids in groups.values() for obs_id in obs_ids}
    samples = tuple(s for s in evidence.samples if s.observation_id in keep_ids)
    anchors = {
        gid: anchors0[gid]
        for gid in groups
        if gid in anchors0
    }
    sigma = {
        key: value
        for key, value in sigma0.items()
        if _sigma_sample_index(key) not in hidden_indices
    }

    metadata = dict(evidence.metadata)
    metadata.update(
        {
            "hypothesis_groups": groups,
            "hypothesis_anchor_raster": anchors,
            "mode_sigma": sigma,
            "coverage_stress": stress_name,
            "coverage_stress_applied_before_gsa": True,
            "coverage_stress_hidden_sample_indices": sorted(hidden_indices),
            "hidden_surface_completion": False,
            "source_mesh_consumer_input": False,
        }
    )
    return ObservationEvidenceIR(
        samples,
        camera_model=evidence.camera_model,
        observation_frame=evidence.observation_frame,
        evidence_version=evidence.evidence_version,
        metadata=metadata,
    )


def build_stressed_u1_surface(
    witness: Witness,
    samples: tuple[ShellSample, ...],
    cameras,
    visibility: tuple[tuple[int, ...], ...],
    *,
    stress_name: str,
) -> tuple[RiggingSurfaceIR, ObservationEvidenceIR, dict]:
    base = build_u1_evidence(witness, samples, cameras, visibility)
    hidden = hidden_sample_indices(samples, stress_name)
    evidence = filter_evidence_before_gsa(base, hidden, stress_name=stress_name)

    surface = compile_surface_v2(evidence, max_common_frame_error=PERSISTENCE_TOLERANCE)
    surface = attach_dtb_nd1_from_evidence(evidence, surface)

    support_counts = [len(n.support_views) for n in surface.surface_nodes]
    normal_count = sum(n.derived_normal is not None for n in surface.surface_nodes)
    natural_visible = sum(bool(v) for v in visibility)
    telemetry = {
        "stress_name": stress_name,
        "u0_full_sample_count": len(samples),
        "naturally_observable_pre_stress_count": natural_visible,
        "stress_hidden_sample_count": len(hidden),
        "gsa_output_surface_node_count": len(surface.surface_nodes),
        "retained_fraction_vs_u0": float(len(surface.surface_nodes) / max(len(samples), 1)),
        "support_view_min": min(support_counts, default=0),
        "support_view_mean": float(np.mean(support_counts)) if support_counts else 0.0,
        "support_view_max": max(support_counts, default=0),
        "dtb_nd1_normal_count": normal_count,
        "dtb_nd1_normal_fraction": float(normal_count / max(len(surface.surface_nodes), 1)),
        "local_relation_count": len(surface.local_relations),
        "hidden_surface_completion": bool(evidence.metadata.get("hidden_surface_completion")),
        "source_mesh_consumer_input": bool(evidence.metadata.get("source_mesh_consumer_input")),
        "gsa_builder_id": surface.builder_id,
        "geometry_lineage_hash": surface.geometry_lineage_hash,
    }
    return surface, evidence, telemetry
