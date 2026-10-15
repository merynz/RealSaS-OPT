from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.fit8_f_qualified_fa_bundle_v2 import canonical_source_carrier_sha256_v1
from tools.corpus.fit8_static_self_intersection_qualification_v1 import StaticSelfIntersectionPolicyV1
from tools.corpus.fit8_static_self_intersection_qualification_v2 import qualify_static_self_intersection_v2
from tools.corpus.fit8_surface_topology_invariants_v1 import compute_surface_topology_invariants_v1

SCHEMA_VERSION = "RealSaS.FGeometryCarrierQualification.v1"


def _canonical_json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _write(path: Path, payload: dict) -> str:
    blob = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def qualify_f_geometry_carrier_v1(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    self_intersection_policy: StaticSelfIntersectionPolicyV1,
    output_topology: Path,
    output_self_intersection: Path,
    output_manifest: Path,
    provenance: str,
) -> dict:
    if not provenance:
        raise ValueError("F_GEOMETRY_CARRIER_QUALIFICATION_PROVENANCE_MISSING")
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    carrier_sha = canonical_source_carrier_sha256_v1(xyz, tri)

    topology = compute_surface_topology_invariants_v1(tri.tolist())
    self_intersection = qualify_static_self_intersection_v2(
        vertices=xyz,
        faces=tri,
        policy=self_intersection_policy,
    )

    topology_payload = asdict(topology)
    topology_payload["source_carrier_sha256"] = carrier_sha
    topology_payload["provenance"] = f"{provenance};TOPOLOGY_INVARIANTS"
    topology_sha = _write(output_topology, topology_payload)

    self_payload = asdict(self_intersection)
    self_payload["source_carrier_sha256"] = carrier_sha
    self_sha = _write(output_self_intersection, self_payload)

    manifest = {
        "schema": SCHEMA_VERSION,
        "source_carrier_sha256": carrier_sha,
        "topology_qualification_sha256": topology_sha,
        "self_intersection_qualification_sha256": self_sha,
        "component_count": int(topology.component_count),
        "boundary_edge_count": int(topology.total_boundary_edges),
        "nonmanifold_edge_count": int(topology.total_nonmanifold_edges),
        "self_intersection_qualified": bool(self_intersection.qualified),
        "self_intersection_free": bool(self_intersection.self_intersection_free),
        "self_intersection_stable_across_tolerance_sweep": bool(self_intersection.stable_across_tolerance_sweep),
        "provenance": provenance,
    }
    manifest_sha = _write(output_manifest, manifest)
    return {**manifest, "manifest_sha256": manifest_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-npz", required=True, type=Path)
    parser.add_argument("--output-topology", required=True, type=Path)
    parser.add_argument("--output-self-intersection", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    with np.load(args.carrier_npz, allow_pickle=False) as archive:
        if not {"vertices", "faces"} <= set(archive.files):
            raise ValueError("F_GEOMETRY_CARRIER_NPZ_ARRAY_MISSING")
        vertices = np.asarray(archive["vertices"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
    policy = StaticSelfIntersectionPolicyV1(
        provenance=f"{args.provenance};STATIC_SELF_INTERSECTION_POLICY_V1",
    )
    result = qualify_f_geometry_carrier_v1(
        vertices=vertices,
        faces=faces,
        self_intersection_policy=policy,
        output_topology=args.output_topology,
        output_self_intersection=args.output_self_intersection,
        output_manifest=args.output_manifest,
        provenance=args.provenance,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
