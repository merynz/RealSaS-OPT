from __future__ import annotations

"""Preregistered emitted-mesh diagnostic for bounded legal Steiner recovery.

This is experiment authority only. It never relabels the frozen product GSA lineage.
A full frozen mesh-policy PASS may be reported under diagnostic scope when the exact
sealed CDT baseline is replayed V0..V7, but PRODUCT PASS remains false while the
byte-exact GSA lineage replay is open.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compiler.realsas_compiler_core.mesh._historical_v05 import triangulate_production_cdt
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    qualify_supported_mesh,
)
from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
    _connected_components,
    _contract_boundary_recovery_vertices,
    _convex_hull_ids,
    _dedupe_projected_ids,
    _match_kernel_vertex,
    _oriented_triangle,
    _point_on_original_hull_segment,
    _safe_surface_graph,
    _signed_area2,
    _visible_subsets_of_full_components,
    build_mwb2_observation_cdt_candidate,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from compiler.realsas_compiler_core.types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualificationError,
    SurfaceSupportBinding,
)

import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageFIT2.LegalSteinerUniformRecoveryDiagnostic.v1"
TREATMENTS = (2, 3, 4)


def _surface_raster(surface, view: int):
    nodes = {str(n.surface_id): n for n in surface.surface_nodes}
    raster = {}
    for sid, node in nodes.items():
        rows = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view)]
        if len(rows) > 1:
            raise QualificationError(f"STEINER_DUPLICATE_RASTER_BINDING:{view}:{sid}")
        if rows:
            raster[sid] = rows[0]
    return nodes, raster


def _supported_kernel_id_triangles(surface, *, view_index: int, min_face_area_grid: float = 1.0e-8):
    nodes, visible, safe_edges, rejected_unknown = _safe_surface_graph(surface, int(view_index))
    full_components = _connected_components(set(nodes), safe_edges)
    components = _visible_subsets_of_full_components(full_components, visible)

    triangles = []
    kernel_triangle_count = 0
    post_contraction_triangle_count = 0
    generated_contracted = 0
    component_count = 0

    for component_index, component in enumerate(components):
        deduped = _dedupe_projected_ids(component, visible)
        if len(deduped) < 3:
            continue
        hull_ids = _convex_hull_ids(deduped, visible)
        if len(hull_ids) < 3:
            continue
        hull_set = set(hull_ids)
        support_ids = tuple(sid for sid in deduped if sid not in hull_set)
        result = triangulate_production_cdt(
            [visible[sid] for sid in hull_ids],
            support_points=[visible[sid] for sid in support_ids],
            target_min_angle_deg=0.0,
            max_boundary_vertices=max(256, len(hull_ids) + 16),
            max_support_vertices=max(128, len(support_ids) + 16),
            max_constraint_recovery_iterations=96,
            max_quality_iterations=0,
            min_feature_spacing=1.0e-8,
        )
        if not bool(result.success):
            raise QualificationError(f"STEINER_CDT_KERNEL_FAILURE:{result.reason}")
        if int(getattr(result, "quality_insert_count", 0)) != 0 or int(
            getattr(result, "inserted_steiner_count", 0)
        ) != 0:
            raise QualificationError("STEINER_UNBOUND_QUALITY_VERTEX_PRESENT")

        kernel_triangle_count += len(result.triangles)
        component_count += 1
        kernel_ids = []
        generated_indices = []
        for vertex_index, point in enumerate(result.vertices):
            sid = _match_kernel_vertex(point, deduped, visible)
            if sid is None:
                if _point_on_original_hull_segment(point, hull_ids, visible) is None:
                    raise QualificationError("STEINER_UNSUPPORTED_KERNEL_VERTEX")
                generated_indices.append(int(vertex_index))
            kernel_ids.append(sid)

        if generated_indices:
            if len(generated_indices) != int(result.constraint_split_count):
                raise QualificationError("STEINER_BOUNDARY_SPLIT_COUNT_MISMATCH")
            kernel_triangles = _contract_boundary_recovery_vertices(
                result, tuple(generated_indices)
            )
            generated_contracted += len(generated_indices)
        else:
            kernel_triangles = tuple(result.triangles)
        post_contraction_triangle_count += len(kernel_triangles)

        matched = [sid for sid in kernel_ids if sid is not None]
        if len(set(matched)) != len(matched):
            raise QualificationError("STEINER_KERNEL_VERTEX_COLLAPSE")

        for ia, ib, ic in kernel_triangles:
            tri_raw = (
                kernel_ids[int(ia)],
                kernel_ids[int(ib)],
                kernel_ids[int(ic)],
            )
            if any(sid is None for sid in tri_raw):
                raise QualificationError("STEINER_UNSUPPORTED_VERTEX_SURVIVED_CONTRACTION")
            tri = tuple(map(str, tri_raw))
            if len(set(tri)) != 3:
                continue
            tri = _oriented_triangle(tri, visible)
            pa, pb, pc = (visible[sid] for sid in tri)
            if 0.5 * abs(_signed_area2(pa, pb, pc)) < float(min_face_area_grid):
                continue
            triangles.append((int(component_index), tri))

    unique = []
    seen = set()
    for component_index, tri in triangles:
        key = tuple(sorted(tri))
        if key in seen:
            continue
        seen.add(key)
        unique.append((component_index, tri))
    return {
        "nodes": nodes,
        "visible": visible,
        "triangles": tuple(unique),
        "component_count": int(component_count),
        "kernel_triangle_count": int(kernel_triangle_count),
        "post_contraction_triangle_count": int(post_contraction_triangle_count),
        "contracted_boundary_recovery_vertex_count": int(generated_contracted),
        "rejected_unknown_relation_count": int(rejected_unknown),
    }


def _support_key(ids: tuple[str, str, str], nums: tuple[int, int, int], n: int):
    acc = {}
    for sid, num in zip(ids, nums):
        if int(num) <= 0:
            continue
        acc[str(sid)] = acc.get(str(sid), 0) + int(num)
    total = sum(acc.values())
    if total != int(n):
        raise RuntimeError(f"STEINER_RATIONAL_SIMPLEX_DRIFT:{total}:{n}")
    return tuple(sorted((sid, int(num)) for sid, num in acc.items()))


def _support_binding(key, n: int, *, view: int):
    coeff = tuple((sid, float(num) / float(n)) for sid, num in key)
    if len(coeff) == 1 and key[0][1] == int(n):
        return SurfaceSupportBinding(
            "IDENTITY_SURFACE_NODE",
            ((key[0][0], 1.0),),
            metadata={
                "observed_view": int(view),
                "legal_steiner_uniform_subdivision": False,
            },
        )
    return SurfaceSupportBinding(
        "LOCAL_CONVEX_INTERPOLATION",
        coeff,
        metadata={
            "observed_view": int(view),
            "legal_steiner_uniform_subdivision": True,
            "support_denominator": int(n),
        },
    )


def _derive_point(nodes, raster, binding):
    px = py = pz = rx = ry = 0.0
    for sid, coefficient in binding.coefficients:
        node = nodes[str(sid)]
        if str(sid) not in raster:
            raise QualificationError(f"STEINER_SUPPORT_NOT_VISIBLE:{sid}")
        c = float(coefficient)
        px += c * float(node.P[0])
        py += c * float(node.P[1])
        pz += c * float(node.P[2])
        rx += c * float(raster[str(sid)][0])
        ry += c * float(raster[str(sid)][1])
    return (px, py, pz), (rx, ry)


def _subtriangle_keys(parent: tuple[str, str, str], n: int):
    def q(i, j):
        return _support_key(parent, (n - i - j, i, j), n)

    out = []
    for i in range(n):
        for j in range(n - i):
            out.append((q(i, j), q(i + 1, j), q(i, j + 1)))
            if i + j <= n - 2:
                out.append((q(i + 1, j), q(i + 1, j + 1), q(i, j + 1)))
    if len(out) != n * n:
        raise RuntimeError(f"STEINER_SUBDIVISION_CARDINALITY_DRIFT:{n}:{len(out)}")
    return tuple(out)


def _make_candidate(surface, *, view: int, camera_hash: str, domain: ObservationRasterDomain, n: int):
    kernel = _supported_kernel_id_triangles(surface, view_index=int(view))
    nodes, raster = _surface_raster(surface, int(view))

    vertex_payload = {}
    face_support_keys = []
    rejected = 0
    candidate_subtriangle_count = 0

    for _component_index, parent in kernel["triangles"]:
        for tri_keys in _subtriangle_keys(parent, int(n)):
            candidate_subtriangle_count += 1
            points = []
            for key in tri_keys:
                if key not in vertex_payload:
                    binding = _support_binding(key, int(n), view=int(view))
                    P, xy = _derive_point(nodes, raster, binding)
                    vertex_payload[key] = (binding, P, xy)
                points.append(vertex_payload[key][2])
            tri_xy = tuple(points)
            if 0.5 * abs(_signed_area2(*tri_xy)) <= 1.0e-10:
                continue
            if not domain.triangle_inside(tri_xy):
                rejected += 1
                continue
            face_support_keys.append(tuple(tri_keys))

    if not face_support_keys:
        raise QualificationError("STEINER_RECOVERY_NO_ADMITTED_FACE")

    used_keys = sorted(
        {key for face in face_support_keys for key in face},
        key=lambda key: tuple((sid, int(num)) for sid, num in key),
    )
    id_map = {key: f"MWB2ST:{view}:N{n}:{i:06d}" for i, key in enumerate(used_keys)}

    vertices = []
    for key in used_keys:
        binding, P, xy = vertex_payload[key]
        vertices.append(
            MeshVertexCandidate(
                id_map[key],
                tuple(map(float, P)),
                binding,
                metadata={
                    "raster_xy": tuple(map(float, xy)),
                    "generated_geometry": binding.mode == "LOCAL_CONVEX_INTERPOLATION",
                    "steiner_role": (
                        "UNIFORM_BARYCENTRIC_RECOVERY"
                        if binding.mode == "LOCAL_CONVEX_INTERPOLATION"
                        else "ORIGINAL_SURFACE_CARRIER"
                    ),
                    "subdivision_factor": int(n),
                    "source_mesh_used": False,
                },
            )
        )

    xy_by_id = {id_map[key]: vertex_payload[key][2] for key in used_keys}
    faces = []
    seen_faces = set()
    for face in face_support_keys:
        ids = tuple(id_map[key] for key in face)
        area2 = _signed_area2(
            xy_by_id[ids[0]], xy_by_id[ids[1]], xy_by_id[ids[2]]
        )
        if abs(area2) <= 1.0e-10:
            continue
        if area2 < 0.0:
            ids = (ids[0], ids[2], ids[1])
        key = tuple(sorted(ids))
        if key in seen_faces:
            continue
        seen_faces.add(key)
        faces.append(ids)

    edge_set = set()
    for a, b, c in faces:
        edge_set.add(tuple(sorted((a, b))))
        edge_set.add(tuple(sorted((b, c))))
        edge_set.add(tuple(sorted((c, a))))
    edges = tuple(sorted(edge_set))

    triangles = tuple(
        (
            tuple(map(float, xy_by_id[a])),
            tuple(map(float, xy_by_id[b])),
            tuple(map(float, xy_by_id[c])),
        )
        for a, b, c in faces
    )
    coverage = domain.coverage(triangles)
    local_count = sum(v.support_binding.mode == "LOCAL_CONVEX_INTERPOLATION" for v in vertices)

    boundary = (
        {
            "kind": "EXACT_OBSERVATION_ALPHA_DOMAIN",
            "view_index": int(view),
            "mask_sha256": domain.mask_sha256,
            "source_alpha_sha256": domain.source_alpha_sha256,
            "width": int(domain.width),
            "height": int(domain.height),
        },
    )
    residual = {
        **coverage,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "edge_count": len(edges),
        "local_convex_interpolation_vertex_count": int(local_count),
        "subdivision_factor": int(n),
        "pre_alpha_kernel_face_count": len(kernel["triangles"]),
        "candidate_subtriangle_count": int(candidate_subtriangle_count),
        "alpha_rejected_subtriangle_count": int(rejected),
        "kernel_component_count": int(kernel["component_count"]),
        "kernel_triangle_count": int(kernel["kernel_triangle_count"]),
        "post_contraction_triangle_count": int(kernel["post_contraction_triangle_count"]),
        "contracted_boundary_recovery_vertex_count": int(
            kernel["contracted_boundary_recovery_vertex_count"]
        ),
    }

    provisional = MeshDiscretizationCandidateIR(
        vertices=tuple(vertices),
        faces=tuple(faces),
        edges=edges,
        surface_binding_hash=str(surface.geometry_lineage_hash),
        view_index=int(view),
        camera_binding_hash=str(camera_hash),
        candidate_lineage_hash="",
        boundary_constraints=boundary,
        coverage_classification="OBSERVATION_DOMAIN_CDT_LEGAL_STEINER_UNIFORM",
        solver_provenance={
            "solver": "CURRENT_CDT_PLUS_PREREGISTERED_UNIFORM_BARYCENTRIC_RECOVERY_V1",
            "subdivision_factor": int(n),
            "treatment_family": list(TREATMENTS),
            "selection_rule": "FIRST_GLOBAL_N_FULL_FROZEN_POLICY_PASS",
            "generated_vertex_policy": "LOCAL_CONVEX_INTERPOLATION_ONLY",
            "cross_component_bridge_used": False,
            "unknown_relation_bridge_used": False,
            "source_mesh_used": False,
            "teacher_topology_used": False,
        },
        residual_report=residual,
        metadata={
            "producer": SCHEMA,
            "source_mesh_used": False,
            "teacher_topology_used": False,
            "observed_view": int(view),
            "subdivision_factor": int(n),
            "surface_lineage_hash": str(surface.geometry_lineage_hash),
            "observation_mask_sha256": domain.mask_sha256,
            "source_alpha_sha256": domain.source_alpha_sha256,
            "target_view_winding": "CCW",
        },
    )
    return replace(
        provisional,
        candidate_lineage_hash=mesh_candidate_lineage_hash(provisional),
    )


def _checker(width: int, height: int, cell: int = 32):
    yy, xx = np.indices((height, width))
    blocks = ((xx // cell) + (yy // cell)) % 2
    rgb = np.where(blocks[..., None] == 0, 54, 72).astype(np.uint8)
    rgb = np.repeat(rgb, 3, axis=2)
    return Image.fromarray(rgb, "RGB").convert("RGBA")


def _mesh_only(source: Image.Image, domain: ObservationRasterDomain, mesh):
    xy = {}
    for v in mesh.vertices:
        xy[str(v.canonical_mesh_vertex_id)] = tuple(map(float, v.metadata["raster_xy"]))
    predicted = np.zeros((domain.height, domain.width), dtype=bool)
    from skimage.draw import polygon
    for face in mesh.faces:
        pts = [xy[str(x)] for x in face]
        rr, cc = polygon(
            [p[1] for p in pts], [p[0] for p in pts], shape=predicted.shape
        )
        predicted[rr, cc] = True
    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    rgba[..., 3] = np.where(predicted, rgba[..., 3], 0).astype(np.uint8)
    return Image.alpha_composite(_checker(source.width, source.height), Image.fromarray(rgba, "RGBA")).convert("RGB")


def _label(image: Image.Image, text: str):
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 44), fill=(0, 0, 0))
    draw.text((10, 11), text, fill=(255, 255, 255))
    return out


def run(args):
    surface, tensor, gsa_replay = ceiling_v2._preflight_surface(args)

    sealed_path = Path(__file__).with_name("MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    expected_by_view = {int(row["view"]): row for row in sealed["views"]}

    camera_paths = tuple(Path(p) for p in args.cameras)
    obs_paths = tuple(Path(p) for p in args.observations)
    ceiling_v1._load_cameras(list(camera_paths))
    ceiling_v1._load_observation_masks(obs_paths)

    baseline_rows = []
    domains = []
    sources = []

    for view, obs_path in enumerate(obs_paths):
        with Image.open(obs_path) as im:
            source = im.convert("RGBA")
        authority = np.asarray(source, dtype=np.uint8)[..., 3] >= 8
        domain = ObservationRasterDomain.from_rows(
            authority.tolist(),
            view_index=view,
            source_alpha_sha256=ceiling_v1.OBSERVATION_SHA256[view],
        )
        baseline = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=view,
            camera_binding_hash=ceiling_v1.CAMERA_SHA256[view],
            observation_domain=domain,
        )
        checks = ceiling_v1._exact_baseline_checks(baseline, expected_by_view[view])
        baseline_rows.append(
            {
                "view": int(view),
                "checks": checks,
                "source_alpha_recall": float(baseline.residual_report["source_alpha_recall"]),
                "precision_inside_alpha": float(baseline.residual_report["precision_inside_alpha"]),
                "face_count": len(baseline.faces),
            }
        )
        domains.append(domain)
        sources.append(source)

    print("=" * 144, flush=True)
    print("BASELINE REPLAY: PASS V0..V7 exact sealed structural+raster metrics", flush=True)
    print("GSA exact lineage:", bool(gsa_replay["gsa_lineage_exact_match"]), flush=True)
    print("=" * 144, flush=True)

    treatment_rows = []
    selected_n = None
    selected_meshes = None

    for n in TREATMENTS:
        print("=" * 144, flush=True)
        print(f"TREATMENT n={n}", flush=True)
        print("=" * 144, flush=True)
        per_view = []
        meshes = []
        for view in range(8):
            candidate = _make_candidate(
                surface,
                view=view,
                camera_hash=ceiling_v1.CAMERA_SHA256[view],
                domain=domains[view],
                n=int(n),
            )
            mesh = qualify_supported_mesh(surface, candidate)
            quality = mesh_raster_quality_report(mesh, surface=surface, view_index=view)
            full = evaluate_mesh_quality(
                coverage=dict(candidate.residual_report),
                raster_report=quality,
                policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
            )
            row = {
                "view": int(view),
                "passed": bool(full["passed"]),
                "failure_invariants": list(full["failure_invariants"]),
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "edge_count": len(mesh.edges),
                "local_convex_interpolation_vertex_count": int(
                    mesh.qualification_report["local_convex_interpolation_vertex_count"]
                ),
                "source_alpha_recall": float(full["source_alpha_recall"]),
                "precision_inside_alpha": float(full["precision_inside_alpha"]),
                "alpha_iou": float(full["alpha_iou"]),
                "largest_uncovered_component_fraction": float(
                    full["largest_uncovered_component_fraction"]
                ),
                "min_raster_triangle_angle_deg": float(full["min_raster_triangle_angle_deg"]),
                "max_raster_triangle_aspect_ratio": float(
                    full["max_raster_triangle_aspect_ratio"]
                ),
                "degenerate_faces": int(full["degenerate_faces"]),
                "duplicate_faces": int(full["duplicate_faces"]),
                "nonmanifold_edges": int(full["nonmanifold_edges"]),
                "candidate_lineage_hash": candidate.candidate_lineage_hash,
                "mesh_lineage_hash": mesh.mesh_lineage_hash,
            }
            per_view.append(row)
            meshes.append(mesh)
            print(
                f"V{view} {'PASS' if row['passed'] else 'FAIL'} "
                f"verts={row['vertex_count']:6d} faces={row['face_count']:7d} "
                f"local={row['local_convex_interpolation_vertex_count']:6d} "
                f"recall={row['source_alpha_recall']*100:7.3f}% "
                f"precision={row['precision_inside_alpha']*100:7.3f}% "
                f"IoU={row['alpha_iou']*100:7.3f}% "
                f"hole={row['largest_uncovered_component_fraction']*100:6.3f}% "
                f"angle={row['min_raster_triangle_angle_deg']:.3f} "
                f"aspect={row['max_raster_triangle_aspect_ratio']:.2f} "
                f"fail={','.join(row['failure_invariants']) or '-'}",
                flush=True,
            )

        global_pass = all(row["passed"] for row in per_view)
        treatment_rows.append(
            {"subdivision_factor": int(n), "global_pass": bool(global_pass), "views": per_view}
        )
        if global_pass:
            selected_n = int(n)
            selected_meshes = tuple(meshes)
            print(f"SELECTED n={n}: FIRST GLOBAL FULL FROZEN POLICY PASS", flush=True)
            break
        print(f"n={n}: GLOBAL FAIL; continue per preregistration", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    selected_images = []
    if selected_n is not None:
        for view, mesh in enumerate(selected_meshes):
            image = _mesh_only(sources[view], domains[view], mesh)
            row = next(
                row
                for treatment in treatment_rows
                if treatment["subdivision_factor"] == selected_n
                for row in treatment["views"]
                if row["view"] == view
            )
            image = _label(
                image,
                f"V{view} n={selected_n} REAL EMITTED MESH | recall={row['source_alpha_recall']*100:.2f}% | faces={row['face_count']}",
            )
            path = out / f"V{view}_LEGAL_STEINER_N{selected_n}_MESH_ONLY.png"
            image.save(path)
            selected_images.append(path.name)

        contact = Image.new("RGB", (2048, 1024))
        for view, name in enumerate(selected_images):
            with Image.open(out / name) as im:
                tile = im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            contact.paste(tile, ((view % 4) * 512, (view // 4) * 512))
        contact_name = f"FIT2_LEGAL_STEINER_N{selected_n}_8VIEW_MESH_ONLY.png"
        contact.save(out / contact_name)
    else:
        contact_name = ""

    decision = (
        f"PASS__FIRST_GLOBAL_FULL_FROZEN_MESH_POLICY__N{selected_n}"
        if selected_n is not None
        else "FAIL__NO_PREREGISTERED_UNIFORM_RECOVERY_LEVEL_PASSES_ALL_VIEWS"
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS__EXPERIMENT_COMPLETED",
        "decision": decision,
        "preregistered_treatments": list(TREATMENTS),
        "selection_rule": "FIRST_GLOBAL_N_FULL_FROZEN_POLICY_PASS",
        "selected_subdivision_factor": selected_n,
        "baseline_replay_gate": "PASS__SEALED_STRUCTURAL_AND_RASTER_METRICS_V0_V7",
        "baseline_views": baseline_rows,
        "gsa_replay": gsa_replay,
        "gsa_authority_lineage_preserved_as_expected": ceiling_v2.EXPECTED_GSA_LINEAGE,
        "gsa_lineage_relabelled": False,
        "scope": (
            "FULL_FROZEN_MESH_POLICY_PASS_DIAGNOSTIC__NOT_PRODUCT_PASS"
            if selected_n is not None
            else "DIAGNOSTIC_FAIL__NOT_PRODUCT_PASS"
        ),
        "product_mesh_pass_claimed": False,
        "cross_component_bridge_used": False,
        "unknown_relation_bridge_used": False,
        "source_mesh_used": False,
        "teacher_topology_used": False,
        "treatments": treatment_rows,
        "selected_contact_sheet_png": contact_name,
        "selected_mesh_only_pngs": selected_images,
    }
    path = out / "FIT2_LEGAL_STEINER_UNIFORM_RECOVERY_RESULT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 144, flush=True)
    print("DECISION:", decision, flush=True)
    print("GSA PRODUCT LINEAGE RELABELLED: FALSE", flush=True)
    print("PRODUCT_MESH_PASS: NOT CLAIMED", flush=True)
    print("RESULT:", path, flush=True)
    if contact_name:
        print("CONTACT:", out / contact_name, flush=True)
    print("=" * 144, flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
