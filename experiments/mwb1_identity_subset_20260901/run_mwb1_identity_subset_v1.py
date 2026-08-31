from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CONSUMER_DIR = HERE.parent / "consumer_interlock_20260829"
if str(CONSUMER_DIR) not in sys.path:
    sys.path.insert(0, str(CONSUMER_DIR))

from run_exact_compiler_consumer_interlock_v0 import load_fixture, reconstruct, a0  # noqa: E402
from realsas_compiler_core.hashing import content_sha256  # noqa: E402
from realsas_compiler_core.mesh_binding import (  # noqa: E402
    bind_identity_mesh_skin,
    mesh_candidate_lineage_hash,
    mesh_lineage_hash,
    mesh_skin_lineage_hash,
    qualify_identity_subset_mesh,
    validate_qualified_mesh_skin,
)
from realsas_compiler_core.product import assemble_product_v2, bind_proof, require_current_proof  # noqa: E402
from realsas_compiler_core.rig import qualify_skeleton  # noqa: E402
from realsas_compiler_core.skin import qualify_skin  # noqa: E402
from realsas_compiler_core.types import (  # noqa: E402
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    SurfaceSupportBinding,
    QualificationError,
)

EXPECTED_FIXTURE_TRANSPORT_SHA = "81634b7db6dbae3f7cc30d7f6942ace9be841416dd1db833185884d3e10584c5"
OUTPUT = HERE / "MWB1_IDENTITY_SUBSET_BASELINE_RESULT_V1.json"


def _identity_binding(surface_id: str) -> SurfaceSupportBinding:
    return SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((surface_id, 1.0),))


def select_local_triangle(surface):
    nodes = sorted(surface.surface_nodes, key=lambda n: n.surface_id)
    points = np.asarray([n.P for n in nodes], dtype=np.float64)
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if not np.isfinite(diag) or diag <= 0.0:
        raise RuntimeError("invalid surface bbox diagonal")
    threshold = 1e-12 * diag * diag

    for anchor in nodes:
        a = np.asarray(anchor.P, dtype=np.float64)
        candidates = []
        anchor_views = set(anchor.support_views)
        for node in nodes:
            if node.surface_id == anchor.surface_id:
                continue
            if not anchor_views.intersection(node.support_views):
                continue
            d = float(np.linalg.norm(np.asarray(node.P, dtype=np.float64) - a))
            candidates.append((d, node.surface_id, node))
        candidates.sort(key=lambda x: (x[0], x[1]))
        near = [x[2] for x in candidates[:64]]
        for i, b in enumerate(near):
            for c in near[i + 1 :]:
                common = set(anchor.support_views).intersection(b.support_views, c.support_views)
                if not common:
                    continue
                ab = np.asarray(b.P, dtype=np.float64) - a
                ac = np.asarray(c.P, dtype=np.float64) - a
                area2 = float(np.linalg.norm(np.cross(ab, ac)))
                if area2 > threshold:
                    return (anchor, b, c), min(common), area2, diag
    raise RuntimeError("MWB1 deterministic local identity triangle not found")


def make_candidate(surface, selected, shared_view: int, fixture_sha: str, asset_id: str):
    vertices = tuple(
        MeshVertexCandidate(
            candidate_vertex_id=f"CV:{i:04d}",
            P=tuple(float(x) for x in node.P),
            support_binding=_identity_binding(node.surface_id),
            metadata={"source_surface_id": node.surface_id},
        )
        for i, node in enumerate(selected)
    )
    ids = tuple(v.candidate_vertex_id for v in vertices)
    camera_binding_hash = content_sha256(
        {
            "fixture_transport_sha256": fixture_sha,
            "shared_support_view": int(shared_view),
            "authority_class": "CAMERA_NOT_CONSUMED_BY_MWB1_IDENTITY_BASELINE",
        }
    )
    candidate = MeshDiscretizationCandidateIR(
        vertices=vertices,
        faces=(ids,),
        edges=((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])),
        surface_binding_hash=surface.geometry_lineage_hash,
        view_index=int(shared_view),
        camera_binding_hash=camera_binding_hash,
        candidate_lineage_hash="",
        boundary_constraints=({"kind": "IDENTITY_SUBSET_LOCAL_TRIANGLE", "asset_id": asset_id},),
        coverage_classification="SACRIFICIAL_IDENTITY_SUBSET_ONLY",
        solver_provenance={"solver": None, "numerical_solver_promoted": False},
        residual_report={},
        metadata={"camera_geometry_consumed": False, "fixture_transport_sha256": fixture_sha},
    )
    return replace(candidate, candidate_lineage_hash=mesh_candidate_lineage_hash(candidate))


def exact_copy_verified(mesh, mesh_skin, skin) -> bool:
    source_rows = {row.surface_id: row for row in skin.rows}
    bound_rows = {row.canonical_mesh_vertex_id: row for row in mesh_skin.rows}
    for vertex in mesh.vertices:
        surface_id, coefficient = vertex.support_binding.coefficients[0]
        if coefficient != 1.0:
            return False
        if bound_rows[vertex.canonical_mesh_vertex_id].influences != source_rows[surface_id].influences:
            return False
    return True


def mutate_topology(mesh):
    mutated = replace(mesh, faces=(tuple(reversed(mesh.faces[0])),), mesh_lineage_hash="")
    return replace(mutated, mesh_lineage_hash=mesh_lineage_hash(mutated))


def rebind_mesh_skin_to_mesh(mesh_skin, mesh):
    rebound = replace(mesh_skin, mesh_binding_hash=mesh.mesh_lineage_hash, mesh_skin_lineage_hash="")
    return replace(rebound, mesh_skin_lineage_hash=mesh_skin_lineage_hash(rebound))


def mutate_weights(mesh_skin):
    rows = list(mesh_skin.rows)
    target_index = None
    target_pair = None
    for i, row in enumerate(rows):
        positive = [k for k, (_, w) in enumerate(row.influences) if float(w) > 0.0]
        if len(positive) >= 2:
            target_index = i
            target_pair = (positive[0], positive[1])
            break
    if target_index is None or target_pair is None:
        raise RuntimeError("MWB1 no copied mesh row has at least two positive influences")
    row = rows[target_index]
    influences = list(row.influences)
    i0, i1 = target_pair
    w0 = float(influences[i0][1]); w1 = float(influences[i1][1])
    delta = min(1e-6, 0.25 * min(w0, w1))
    if delta <= 0.0:
        raise RuntimeError("MWB1 weight mutation delta is not positive")
    influences[i0] = (influences[i0][0], w0 - delta)
    influences[i1] = (influences[i1][0], w1 + delta)
    rows[target_index] = replace(row, influences=tuple(influences))
    mutated = replace(mesh_skin, rows=tuple(rows), mesh_skin_lineage_hash="")
    return replace(mutated, mesh_skin_lineage_hash=mesh_skin_lineage_hash(mutated)), row.canonical_mesh_vertex_id, delta


def stale_rejected(product, proof) -> bool:
    try:
        require_current_proof(product, proof, require_pass=True)
    except QualificationError:
        return True
    return False


def main():
    fixture, fixture_sha = load_fixture()
    if fixture_sha != EXPECTED_FIXTURE_TRANSPORT_SHA:
        raise RuntimeError("MWB1 fixture transport authority mismatch")
    surface, g0 = reconstruct(fixture)
    if len(surface.surface_nodes) != 512:
        raise RuntimeError(f"MWB1 expected 512 surface nodes, got {len(surface.surface_nodes)}")

    skeleton = qualify_skeleton(surface, g0, run_ilp_shadow=False)
    skin = qualify_skin(surface, skeleton, a0(surface, skeleton), max_influences=4)
    if len(skin.rows) != 512:
        raise RuntimeError("MWB1 expected 512 qualified skin rows")

    selected, shared_view, area2, bbox_diag = select_local_triangle(surface)
    candidate = make_candidate(surface, selected, shared_view, fixture_sha, fixture["asset_id"])
    mesh = qualify_identity_subset_mesh(surface, candidate)
    mesh_skin = bind_identity_mesh_skin(surface, skeleton, skin, mesh)

    if not exact_copy_verified(mesh, mesh_skin, skin):
        raise RuntimeError("MWB1 identity mesh-skin path is not an exact qualified-skin copy")

    product = assemble_product_v2(surface, skeleton, skin, mesh, mesh_skin, editable_metadata={"mw_gate": "MWB1", "sacrificial_identity_subset": True})
    proof = bind_proof(
        product,
        probe_plan={"name": "MWB1_TYPED_LINEAGE_INVARIANT_V1"},
        measurements={"identity_copy": True, "mesh_vertices": len(mesh.vertices), "mesh_faces": len(mesh.faces)},
        passed=True,
        proof_payload={"mw_gate": "MWB1", "not_deformation_quality_claim": True},
    )
    require_current_proof(product, proof, require_pass=True)

    topology_mesh = mutate_topology(mesh)
    topology_mesh_skin = rebind_mesh_skin_to_mesh(mesh_skin, topology_mesh)
    validate_qualified_mesh_skin(topology_mesh_skin, surface=surface, skeleton=skeleton, skin=skin, mesh=topology_mesh)
    topology_product = assemble_product_v2(surface, skeleton, skin, topology_mesh, topology_mesh_skin)

    weight_mesh_skin, mutated_vertex_id, delta = mutate_weights(mesh_skin)
    validate_qualified_mesh_skin(weight_mesh_skin, surface=surface, skeleton=skeleton, skin=skin, mesh=mesh)
    weight_product = assemble_product_v2(surface, skeleton, skin, mesh, weight_mesh_skin)

    topology_hash_changed = topology_mesh.mesh_lineage_hash != mesh.mesh_lineage_hash
    topology_product_changed = topology_product.product_state_hash != product.product_state_hash
    topology_stale_rejected = stale_rejected(topology_product, proof)
    weight_hash_changed = weight_mesh_skin.mesh_skin_lineage_hash != mesh_skin.mesh_skin_lineage_hash
    weight_product_changed = weight_product.product_state_hash != product.product_state_hash
    weight_stale_rejected = stale_rejected(weight_product, proof)

    passed = all(
        [
            fixture_sha == EXPECTED_FIXTURE_TRANSPORT_SHA,
            len(surface.surface_nodes) == 512,
            len(skeleton.joints) == len(g0.joints),
            len(skin.rows) == 512,
            len(mesh.vertices) == 3,
            len(mesh.faces) == 1,
            exact_copy_verified(mesh, mesh_skin, skin),
            topology_hash_changed,
            topology_product_changed,
            topology_stale_rejected,
            weight_hash_changed,
            weight_product_changed,
            weight_stale_rejected,
        ]
    )

    result = {
        "schema": "RealSaS.MWB1IdentitySubsetBaselineResult.v1",
        "status": "PASS_MWB1_IDENTITY_SUBSET_BASELINE" if passed else "FAIL_MWB1_IDENTITY_SUBSET_BASELINE",
        "asset_id": fixture["asset_id"],
        "fixture_transport_sha256": fixture_sha,
        "surface_node_count": len(surface.surface_nodes),
        "qualified_joint_count": len(skeleton.joints),
        "qualified_skin_row_count": len(skin.rows),
        "selected_surface_ids": [n.surface_id for n in selected],
        "shared_support_view": int(shared_view),
        "triangle_cross_magnitude": area2,
        "surface_bbox_diag": bbox_diag,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "mesh_skin_lineage_hash": mesh_skin.mesh_skin_lineage_hash,
        "product_state_hash": product.product_state_hash,
        "identity_weight_copy_exact": exact_copy_verified(mesh, mesh_skin, skin),
        "camera_geometry_consumed": False,
        "camera_binding_hash": candidate.camera_binding_hash,
        "topology_mutation": {
            "mesh_hash_changed": topology_hash_changed,
            "product_hash_changed": topology_product_changed,
            "original_proof_rejected_as_stale": topology_stale_rejected,
        },
        "weight_mutation": {
            "mesh_skin_hash_changed": weight_hash_changed,
            "product_hash_changed": weight_product_changed,
            "original_proof_rejected_as_stale": weight_stale_rejected,
            "mutated_mesh_vertex_id": mutated_vertex_id,
            "delta": delta,
        },
        "scientific_optimizer_steps": 0,
        "training_authorized": False,
        "claim_boundary": "typed identity-subset lineage/weight-copy baseline only; no product mesh/deformation claim",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
