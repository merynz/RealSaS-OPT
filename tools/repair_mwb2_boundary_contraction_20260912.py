from __future__ import annotations

from pathlib import Path

MWB2 = Path("compiler/realsas_compiler_core/mesh/mwb2_cdt.py")
TEST = Path("tests/compiler/test_mwb2_observation_cdt_v1.py")

HELPERS = r'''

def _point_on_original_hull_segment(
    point,
    hull_ids: tuple[str, ...],
    visible: dict[str, tuple[float, float]],
    *,
    eps: float = _CDT_MATCH_EPS,
) -> tuple[str, str, float] | None:
    px, py = float(point[0]), float(point[1])
    matches: list[tuple[float, str, str, float]] = []
    for i, a_sid in enumerate(hull_ids):
        b_sid = hull_ids[(i + 1) % len(hull_ids)]
        ax, ay = visible[a_sid]
        bx, by = visible[b_sid]
        dx, dy = float(bx) - float(ax), float(by) - float(ay)
        denom = dx * dx + dy * dy
        if denom <= eps * eps:
            continue
        t = ((px - float(ax)) * dx + (py - float(ay)) * dy) / denom
        if t <= eps or t >= 1.0 - eps:
            continue
        qx, qy = float(ax) + t * dx, float(ay) + t * dy
        dist = math.hypot(px - qx, py - qy)
        if dist <= eps:
            matches.append((dist, a_sid, b_sid, float(t)))
    if not matches:
        return None
    matches.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    best = matches[0]
    if len(matches) > 1 and abs(matches[1][0] - best[0]) <= 1.0e-12:
        raise QualificationError("MWB2_CDT_AMBIGUOUS_BOUNDARY_RECOVERY_VERTEX")
    return best[1], best[2], best[3]


def _polygon_signed_area2(indices: list[int], points) -> float:
    return float(
        sum(
            points[indices[i]][0] * points[indices[(i + 1) % len(indices)]][1]
            - points[indices[(i + 1) % len(indices)]][0] * points[indices[i]][1]
            for i in range(len(indices))
        )
    )


def _point_in_triangle_xy(point, a, b, c, *, eps: float = 1.0e-10) -> bool:
    s0 = _signed_area2(a, b, point)
    s1 = _signed_area2(b, c, point)
    s2 = _signed_area2(c, a, point)
    has_neg = s0 < -eps or s1 < -eps or s2 < -eps
    has_pos = s0 > eps or s1 > eps or s2 > eps
    return not (has_neg and has_pos)


def _ear_clip_cavity(path: list[int], points) -> list[tuple[int, int, int]]:
    poly = list(path)
    if len(poly) < 3 or len(set(poly)) != len(poly):
        raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_INVALID_CAVITY")
    if _polygon_signed_area2(poly, points) < 0.0:
        poly.reverse()
    out: list[tuple[int, int, int]] = []
    while len(poly) > 3:
        ears = []
        for i, cur in enumerate(poly):
            a, b, c = poly[i - 1], cur, poly[(i + 1) % len(poly)]
            if _signed_area2(points[a], points[b], points[c]) <= 1.0e-10:
                continue
            tri = (a, b, c)
            if any(
                j not in tri
                and _point_in_triangle_xy(points[j], points[a], points[b], points[c])
                for j in poly
            ):
                continue
            ears.append((tuple(sorted(tri)), i, tri))
        if not ears:
            raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_EAR_CLIP_FAIL")
        _, index, tri = min(ears)
        out.append(tri)
        poly.pop(index)
    out.append(tuple(poly))
    return out


def _contract_boundary_recovery_vertices(result, generated_indices: tuple[int, ...]):
    """Remove solver-only boundary split vertices and fill their planar cavities."""
    triangles = [tuple(map(int, tri)) for tri in result.triangles]
    constraints = {tuple(sorted(map(int, edge))) for edge in result.constraint_edges}
    points = result.vertices
    for g in sorted(map(int, generated_indices)):
        incident = [tri for tri in triangles if g in tri]
        if not incident:
            raise QualificationError("MWB2_CDT_BOUNDARY_RECOVERY_VERTEX_UNUSED")
        constraint_incident = [edge for edge in constraints if g in edge]
        if len(constraint_incident) != 2:
            raise QualificationError("MWB2_CDT_GENERATED_VERTEX_NOT_BOUNDARY_RECOVERY")
        endpoints = [edge[0] if edge[1] == g else edge[1] for edge in constraint_incident]
        adjacency: dict[int, set[int]] = {}
        for tri in incident:
            opposite = [idx for idx in tri if idx != g]
            if len(opposite) != 2:
                raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_INVALID_STAR")
            a, b = opposite
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        start, end = endpoints
        path = [start]
        previous = None
        current = start
        while current != end:
            candidates = [idx for idx in sorted(adjacency.get(current, ())) if idx != previous]
            candidates = [idx for idx in candidates if idx not in path or idx == end]
            if len(candidates) != 1:
                raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_NONMANIFOLD_STAR")
            nxt = candidates[0]
            previous, current = current, nxt
            path.append(current)
            if len(path) > len(adjacency) + 1:
                raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_PATH_LOOP")
        replacement = _ear_clip_cavity(path, points)
        triangles = [tri for tri in triangles if g not in tri] + replacement
        for edge in constraint_incident:
            constraints.remove(tuple(sorted(edge)))
        constraints.add(tuple(sorted((start, end))))
    if any(any(g in tri for g in generated_indices) for tri in triangles):
        raise QualificationError("MWB2_CDT_BOUNDARY_RECOVERY_CONTRACTION_INCOMPLETE")
    return tuple(triangles)
'''

OLD_VERTEX_BLOCK = '''        kernel_ids: list[str] = []
        for point in result.vertices:
            sid = _match_kernel_vertex(point, deduped, visible)
            if sid is None:
                raise QualificationError("MWB2_CDT_UNSUPPORTED_GENERATED_VERTEX")
            kernel_ids.append(sid)
        if len(set(kernel_ids)) != len(kernel_ids):
            raise QualificationError("MWB2_CDT_VERTEX_COLLAPSE_AFTER_BINDING")

        for ia, ib, ic in result.triangles:
            tri = (kernel_ids[int(ia)], kernel_ids[int(ib)], kernel_ids[int(ic)])
'''

NEW_VERTEX_BLOCK = '''        kernel_ids: list[str | None] = []
        generated_indices: list[int] = []
        for vertex_index, point in enumerate(result.vertices):
            sid = _match_kernel_vertex(point, deduped, visible)
            if sid is None:
                if _point_on_original_hull_segment(point, hull_ids, visible) is None:
                    raise QualificationError("MWB2_CDT_UNSUPPORTED_GENERATED_VERTEX")
                generated_indices.append(int(vertex_index))
            kernel_ids.append(sid)
        if generated_indices:
            if int(getattr(result, "quality_insert_count", 0)) != 0 or int(
                getattr(result, "inserted_steiner_count", 0)
            ) != 0:
                raise QualificationError("MWB2_CDT_UNBOUND_QUALITY_STEINER_FORBIDDEN")
            if len(generated_indices) != int(result.constraint_split_count):
                raise QualificationError("MWB2_CDT_CONSTRAINT_SPLIT_BINDING_COUNT_MISMATCH")
            kernel_triangles = _contract_boundary_recovery_vertices(result, tuple(generated_indices))
            contracted_boundary_recovery_vertex_count += len(generated_indices)
        else:
            kernel_triangles = tuple(result.triangles)
        matched_ids = [sid for sid in kernel_ids if sid is not None]
        if len(set(matched_ids)) != len(matched_ids):
            raise QualificationError("MWB2_CDT_VERTEX_COLLAPSE_AFTER_BINDING")
        post_contraction_triangle_count += len(kernel_triangles)

        for ia, ib, ic in kernel_triangles:
            tri_raw = (kernel_ids[int(ia)], kernel_ids[int(ib)], kernel_ids[int(ic)])
            if any(sid is None for sid in tri_raw):
                raise QualificationError("MWB2_CDT_BOUNDARY_RECOVERY_CONTRACTION_INCOMPLETE")
            tri = (str(tri_raw[0]), str(tri_raw[1]), str(tri_raw[2]))
'''

TEST_APPEND = r'''


def test_boundary_recovery_midpoint_is_solver_internal_only():
    surface = _chain_square_surface()
    domain = ObservationRasterDomain.from_rows(
        _rect_mask(10, 10, [(1, 1, 8, 8)]), view_index=0
    )
    fake = SimpleNamespace(
        success=True,
        reason="ok",
        vertices=[(1.0, 1.0), (8.0, 1.0), (8.0, 8.0), (1.0, 8.0), (4.5, 1.0)],
        triangles=[(0, 4, 3), (4, 2, 3), (4, 1, 2)],
        constraint_edges=[(0, 4), (4, 1), (1, 2), (2, 3), (3, 0)],
        constraint_split_count=1,
        quality_insert_count=0,
        inserted_steiner_count=0,
    )
    with patch(
        "compiler.realsas_compiler_core.mesh.mwb2_cdt.triangulate_production_cdt",
        return_value=fake,
    ):
        candidate = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=0,
            camera_binding_hash="camera",
            observation_domain=domain,
        )
    assert candidate.residual_report["contracted_boundary_recovery_vertex_count"] == 1
    assert candidate.residual_report["post_contraction_triangle_count"] == 2
    assert candidate.residual_report["source_alpha_recall"] == 1.0
    assert candidate.residual_report["precision_inside_alpha"] == 1.0
    assert all(v.support_binding.mode == "IDENTITY_SURFACE_NODE" for v in candidate.vertices)
    assert all(v.metadata["generated_geometry"] is False for v in candidate.vertices)
    qualified = qualify_mwb2_observation_cdt_mesh(surface, candidate)
    assert qualified.qualification_report["cdt_behavioral_gate_pass"] is True
'''


def patch_mwb2() -> None:
    text = MWB2.read_text(encoding="utf-8")
    if "def _point_on_original_hull_segment(" not in text:
        anchor = "    return best_sid if best_dist <= eps else None\n\n\ndef build_mwb2_observation_cdt_candidate("
        if anchor not in text:
            raise RuntimeError("mwb2 helper anchor drift")
        text = text.replace(
            anchor,
            "    return best_sid if best_dist <= eps else None" + HELPERS
            + "\n\ndef build_mwb2_observation_cdt_candidate(",
            1,
        )
    if "contracted_boundary_recovery_vertex_count = 0" not in text:
        anchor = "    kernel_triangle_count = 0\n    kernel_constraint_split_count = 0\n"
        if anchor not in text:
            raise RuntimeError("mwb2 counter anchor drift")
        text = text.replace(
            anchor,
            anchor
            + "    contracted_boundary_recovery_vertex_count = 0\n"
            + "    post_contraction_triangle_count = 0\n",
            1,
        )
    if OLD_VERTEX_BLOCK in text:
        text = text.replace(OLD_VERTEX_BLOCK, NEW_VERTEX_BLOCK, 1)
    elif "kernel_ids: list[str | None]" not in text:
        raise RuntimeError("mwb2 generated-vertex block drift")
    if '"contracted_boundary_recovery_vertex_count"' not in text:
        anchor = '        "kernel_constraint_split_count": int(kernel_constraint_split_count),\n'
        if anchor not in text:
            raise RuntimeError("mwb2 residual anchor drift")
        text = text.replace(
            anchor,
            anchor
            + '        "contracted_boundary_recovery_vertex_count": int(contracted_boundary_recovery_vertex_count),\n'
            + '        "post_contraction_triangle_count": int(post_contraction_triangle_count),\n',
            1,
        )
    text = text.replace(
        "HISTORICAL_V05_CDT_CURRENT_TYPED_ADAPTER_V2",
        "HISTORICAL_V05_CDT_CURRENT_TYPED_ADAPTER_V3",
    )
    text = text.replace(
        "REJECT_UNLESS_EXACT_ADMITTED_SURFACE_CARRIER",
        "CONTRACT_BOUNDARY_RECOVERY_VERTICES__REJECT_OTHER_GENERATED_VERTICES",
    )
    text = text.replace(
        "RealSaS.MWB2.ObservationDomainCDT.v2",
        "RealSaS.MWB2.ObservationDomainCDT.v3",
    )
    MWB2.write_text(text, encoding="utf-8")


def patch_test() -> None:
    text = TEST.read_text(encoding="utf-8")
    if "test_boundary_recovery_midpoint_is_solver_internal_only" not in text:
        TEST.write_text(text + TEST_APPEND, encoding="utf-8")


if __name__ == "__main__":
    patch_mwb2()
    patch_test()
    print("MWB2 boundary-recovery contraction patch applied")
