#!/usr/bin/env python3
"""RealSaS post-corpus forensic auditor.

Read-only with respect to corpus evidence. It writes only compact reports under
<root>/reports/post_corpus_audit unless --out-dir is supplied.

Scope: Gates 0-3 apparatus/data-plane audit. This deliberately does NOT train,
rerender, mutate assets, generate consumer exports, or change split authority.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

BUILD_EXPECTED = "REALSAS_MASTER_1024_V4_3_LOCAL_FIRST_FULL_PRODUCTION_20260822"
VIEWS = 8
NATIVE_RES = 1024
REQUIRED_VIEW_FILES = (
    "cel_clean.png",
    "ink_cel.png",
    "cel_clean_512.png",
    "ink_cel_512.png",
    "raster_authority.npz",
    "camera.json",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"bad jsonl {path}:{ln}: {e}") from e
            if isinstance(x, dict):
                out.append(x)
    return out


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_split(asset_id: str, qa_only: bool = False) -> str:
    if qa_only:
        return "QA_ONLY"
    x = int(sha256_text(asset_id)[:8], 16) % 10000
    if x < 7600:
        return "FIT"
    if x < 8400:
        return "TUNE"
    if x < 9000:
        return "CAL"
    if x < 9600:
        return "DEV"
    return "EXTERNAL_HOLDOUT"


def pending_key(rec: dict[str, Any]) -> str:
    for k in ("candidate_id", "canonical_asset_id", "source_record_id", "file_identifier"):
        if rec.get(k):
            return f"{k}:{rec[k]}"
    return sha256_text(json.dumps(rec, sort_keys=True, default=str))


def latest_unresolved_pending(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:
        latest[pending_key(r)] = r
    terminal = {"PASS", "ADMIT", "RESOLVED", "DONE", "COMPLETE", "REJECT_TECHNICAL"}
    return [r for r in latest.values() if str(r.get("status", "")).upper() not in terminal]


def counter(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    c = collections.Counter(str(x.get(key, "<missing>")) for x in rows)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def cap_count(rows: Iterable[dict[str, Any]], cap: str) -> int:
    return sum(bool((x.get("capabilities") or {}).get(cap)) for x in rows)


def is_finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())


def mesh_components(n_vertices: int, faces: np.ndarray) -> tuple[int, list[int]]:
    parent = np.arange(n_vertices, dtype=np.int64)
    size = np.ones(n_vertices, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    used = np.zeros(n_vertices, dtype=bool)
    for tri in faces:
        a, b, c = map(int, tri)
        used[[a, b, c]] = True
        union(a, b); union(b, c); union(c, a)
    roots = collections.Counter(find(int(i)) for i in np.flatnonzero(used))
    vals = sorted(roots.values(), reverse=True)
    return len(vals), vals[:20]


def edge_multiplicity(faces: np.ndarray) -> dict[str, int]:
    e = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    e = np.sort(e.astype(np.int64), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {
        "unique_edges": int(len(cnt)),
        "boundary_edges": int(np.sum(cnt == 1)),
        "manifold_edges": int(np.sum(cnt == 2)),
        "nonmanifold_edges_gt2": int(np.sum(cnt > 2)),
        "max_edge_multiplicity": int(cnt.max()) if len(cnt) else 0,
    }


def camera_project(P: np.ndarray, cam: dict[str, Any], W: int, H: int) -> np.ndarray:
    right = np.asarray(cam["right"], dtype=np.float64)
    up = np.asarray(cam["up"], dtype=np.float64)
    he = float(cam["half_extent"])
    ndc_x = P @ right / he
    ndc_y = P @ up / he
    x = (ndc_x + 1.0) * W / 2.0 - 0.5
    y = (1.0 - ndc_y) * H / 2.0 - 0.5
    return np.stack([x, y], axis=-1)


def deep_view_audit(view_dir: Path, geom: dict[str, np.ndarray], vi: int) -> dict[str, Any]:
    out: dict[str, Any] = {"view": vi, "path": str(view_dir)}
    with Image.open(view_dir / "cel_clean.png") as im:
        rgba = np.asarray(im.convert("RGBA"))
        W, H = im.size
    out["native_size"] = [W, H]
    alpha_idx = np.flatnonzero(rgba[..., 3].reshape(-1) > 0).astype(np.int64)
    out["alpha_foreground_count"] = int(len(alpha_idx))

    cam = read_json(view_dir / "camera.json")
    out["camera"] = {
        "yaw_deg": cam.get("yaw_deg"),
        "image_origin": cam.get("image_origin"),
        "image_y_direction": cam.get("image_y_direction"),
        "ndc_y_direction": cam.get("ndc_y_direction"),
        "vertical_flip_count": cam.get("vertical_flip_count"),
        "semantic_facing": cam.get("semantic_facing"),
        "contract": cam.get("contract"),
    }
    out["camera_contract_ok"] = bool(
        cam.get("image_origin") == "TOP_LEFT"
        and int(cam.get("vertical_flip_count", -1)) == 1
        and int(cam.get("yaw_deg", -999)) == vi * 45
    )

    with np.load(view_dir / "raster_authority.npz", allow_pickle=False) as z:
        keys = set(z.files)
        req = {"pixel_linear_index", "triangle_id", "barycentric_uv", "resolution", "origin"}
        out["authority_keys_ok"] = req.issubset(keys)
        pix = np.asarray(z["pixel_linear_index"], dtype=np.int64)
        tid = np.asarray(z["triangle_id"], dtype=np.int64)
        uv = np.asarray(z["barycentric_uv"], dtype=np.float64)
        res = np.asarray(z["resolution"]).reshape(-1).tolist()
        origin_raw = np.asarray(z["origin"])
        try:
            origin = str(origin_raw.item())
        except Exception:
            origin = str(origin_raw)
    out["authority_count"] = int(len(pix))
    out["authority_resolution"] = res
    out["authority_origin"] = origin
    out["pixel_unique"] = bool(len(np.unique(pix)) == len(pix))
    out["pixel_in_bounds"] = bool(len(pix) == 0 or (pix.min() >= 0 and pix.max() < W * H))
    out["alpha_authority_exact"] = bool(len(alpha_idx) == len(pix) and np.array_equal(np.sort(alpha_idx), np.sort(pix)))

    faces = geom["faces"]
    vertices = geom["vertices"]
    out["triangle_id_valid"] = bool(len(tid) == 0 or (tid.min() >= 0 and tid.max() < len(faces)))
    out["barycentric_finite"] = is_finite(uv)
    w = np.stack([uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]], axis=1) if len(uv) else np.empty((0, 3))
    out["barycentric_sum_max_abs_err"] = float(np.max(np.abs(w.sum(axis=1) - 1.0))) if len(w) else 0.0
    out["barycentric_min_weight"] = float(w.min()) if len(w) else 0.0
    out["barycentric_outside_fraction_1e5"] = float(np.mean(w < -1e-5)) if len(w) else 0.0

    if len(tid) and out["triangle_id_valid"]:
        tri = vertices[faces[tid]]
        P = tri[:, 0] * w[:, 0:1] + tri[:, 1] * w[:, 1:2] + tri[:, 2] * w[:, 2:3]
        xy = camera_project(P.astype(np.float64), cam, W, H)
        obs = np.stack([pix % W, pix // W], axis=1).astype(np.float64)
        err = np.linalg.norm(xy - obs, axis=1)
        out["reprojection_px"] = {
            "p50": float(np.quantile(err, .50)),
            "p95": float(np.quantile(err, .95)),
            "p99": float(np.quantile(err, .99)),
            "p999": float(np.quantile(err, .999)),
            "max": float(err.max()),
            "gt_0_5": int(np.sum(err > .5)),
            "gt_1_0": int(np.sum(err > 1.0)),
        }
    return out


def deep_asset_audit(root: Path, sel: dict[str, Any]) -> dict[str, Any]:
    aid = sel["canonical_asset_id"]
    asset = root / "master" / "assets" / aid
    out: dict[str, Any] = {
        "canonical_asset_id": aid,
        "source_registry_id": sel.get("source_registry_id"),
        "split": sel.get("split"),
        "capabilities": sel.get("capabilities"),
        "decision": sel.get("decision"),
        "errors": [],
    }
    try:
        with np.load(asset / "primary_geometry.npz", allow_pickle=False) as z:
            geom = {k: np.asarray(z[k]) for k in z.files}
        V = geom["vertices"].astype(np.float64)
        F = geom["faces"].astype(np.int64)
        N = geom.get("vertex_normals")
        out["geometry"] = {
            "vertex_count": int(len(V)),
            "face_count": int(len(F)),
            "vertices_finite": is_finite(V),
            "faces_in_bounds": bool(len(F) == 0 or (F.min() >= 0 and F.max() < len(V))),
        }
        tri = V[F]
        area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        out["geometry"].update({
            "area2_min": float(area2.min()) if len(area2) else 0.0,
            "degenerate_fraction_le_1e12": float(np.mean(area2 <= 1e-12)) if len(area2) else 0.0,
        })
        if N is not None:
            N = np.asarray(N, dtype=np.float64)
            nn = np.linalg.norm(N, axis=1)
            out["geometry"].update({
                "normals_finite": is_finite(N),
                "normal_norm_p01": float(np.quantile(nn, .01)) if len(nn) else 0.0,
                "normal_norm_p50": float(np.quantile(nn, .50)) if len(nn) else 0.0,
                "normal_norm_p99": float(np.quantile(nn, .99)) if len(nn) else 0.0,
                "zero_normal_count": int(np.sum(nn <= 1e-12)),
            })
        out["geometry"].update(edge_multiplicity(F))
        cc, top = mesh_components(len(V), F)
        out["geometry"]["connected_components"] = cc
        out["geometry"]["component_vertex_counts_top20"] = top

        if "skin" in geom:
            skin = np.asarray(geom["skin"], dtype=np.float64)
            rs = skin.sum(axis=1)
            out["skin"] = {
                "shape": list(skin.shape),
                "finite": is_finite(skin),
                "negative_fraction": float(np.mean(skin < -1e-8)),
                "row_sum_p50": float(np.quantile(rs, .5)),
                "row_sum_p99_abs_err": float(np.quantile(np.abs(rs - 1.0), .99)),
                "zero_row_fraction": float(np.mean(rs <= 1e-12)),
            }
        out["views"] = [deep_view_audit(asset / "renders" / f"V{vi}", geom, vi) for vi in range(VIEWS)]
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
    return out


def shallow_asset_audit(root: Path, sel: dict[str, Any]) -> dict[str, Any]:
    aid = sel["canonical_asset_id"]
    asset = root / "master" / "assets" / aid
    issues = []
    marker = asset / "RENDER_COMPLETE.json"
    if not marker.exists():
        issues.append("missing_RENDER_COMPLETE")
    else:
        try:
            m = read_json(marker)
            if m.get("build_id") != BUILD_EXPECTED:
                issues.append("build_id_mismatch")
            if int(m.get("views", -1)) != VIEWS:
                issues.append("marker_views_mismatch")
            if int(m.get("native_resolution", -1)) != NATIVE_RES:
                issues.append("marker_resolution_mismatch")
        except Exception as e:
            issues.append(f"bad_marker:{type(e).__name__}")
    if not (asset / "primary_geometry.npz").is_file():
        issues.append("missing_primary_geometry")
    for vi in range(VIEWS):
        vd = asset / "renders" / f"V{vi}"
        for fn in REQUIRED_VIEW_FILES:
            p = vd / fn
            if not p.is_file() or p.stat().st_size <= 0:
                issues.append(f"V{vi}:missing:{fn}")
        cp = vd / "camera.json"
        if cp.is_file():
            try:
                c = read_json(cp)
                if c.get("image_origin") != "TOP_LEFT": issues.append(f"V{vi}:camera_origin")
                if int(c.get("vertical_flip_count", -1)) != 1: issues.append(f"V{vi}:flip_count")
                if int(c.get("yaw_deg", -999)) != vi * 45: issues.append(f"V{vi}:yaw")
                if c.get("semantic_facing") != "UNKNOWN": issues.append(f"V{vi}:semantic_facing_not_unknown")
            except Exception as e:
                issues.append(f"V{vi}:bad_camera:{type(e).__name__}")
    return {"canonical_asset_id": aid, "issues": issues}


def choose_deep_sample(selected: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(selected):
        return list(selected)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in selected:
        caps = r.get("capabilities") or {}
        c = "ARACHNE" if caps.get("arachne") else ("GEPPETTO" if caps.get("geppetto") else "IRIS")
        buckets[(str(r.get("source_registry_id")), str(r.get("split")), c)].append(r)
    picked: dict[str, dict[str, Any]] = {}
    for b in buckets.values():
        b.sort(key=lambda x: sha256_text(x["canonical_asset_id"]))
        picked[b[0]["canonical_asset_id"]] = b[0]
    rest = sorted(selected, key=lambda x: sha256_text("deep:" + x["canonical_asset_id"]))
    for r in rest:
        if len(picked) >= n: break
        picked.setdefault(r["canonical_asset_id"], r)
    return list(picked.values())[:n]


def audit_provenance(root: Path, selected: list[dict[str, Any]]) -> dict[str, Any]:
    missing_variant = []
    missing_admission = []
    missing_fields = collections.Counter()
    raw_policy = collections.Counter()
    source_sha = collections.defaultdict(list)
    raw_path_missing = []
    ext = collections.Counter()
    for s in selected:
        vdir = root / str(s["variant_dir"])
        if not vdir.is_dir():
            missing_variant.append(s["canonical_asset_id"]); continue
        ap = vdir / "admission.json"
        if not ap.is_file():
            missing_admission.append(s["canonical_asset_id"]); continue
        try:
            a = read_json(ap)
        except Exception:
            missing_admission.append(s["canonical_asset_id"]); continue
        sid = a.get("source_identity") or {}
        for k in ("source_sha256", "source_size_bytes", "source_record_id", "file_identifier", "revision"):
            if sid.get(k) in (None, ""):
                missing_fields[k] += 1
        sh = sid.get("source_sha256")
        if sh:
            source_sha[str(sh)].append(s["canonical_asset_id"])
        raw = sid.get("preserved_raw_source_path")
        pol = str((a.get("source_extra") or {}).get("raw_storage_policy") or "<unspecified>")
        raw_policy[pol] += 1
        fid = str(sid.get("file_identifier") or "")
        ext[Path(fid.split("?")[0]).suffix.lower() or "<none>"] += 1
        if raw and not (root / str(raw)).exists():
            raw_path_missing.append(s["canonical_asset_id"])
    dup_groups = {k: v for k, v in source_sha.items() if len(v) > 1}
    return {
        "missing_variant_dir_count": len(missing_variant),
        "missing_admission_count": len(missing_admission),
        "missing_source_identity_fields": dict(missing_fields),
        "raw_storage_policy": dict(raw_policy),
        "preserved_raw_path_missing_count": len(raw_path_missing),
        "duplicate_source_sha_group_count": len(dup_groups),
        "duplicate_source_sha_asset_count": sum(len(v) for v in dup_groups.values()),
        "duplicate_source_sha_examples": list(dup_groups.items())[:20],
        "source_extension_distribution": dict(ext.most_common()),
    }


def markdown_report(r: dict[str, Any]) -> str:
    g0 = r["gate0"]
    sh = r["shallow"]
    lines = [
        "# RealSaS Post-Corpus Audit — Gates 0–3", "",
        f"Build: `{r['build_id']}`", f"Root: `{r['root']}`", "",
        "## Decision", "",
        "**TRAINING REMAINS BLOCKED until the audit's fatal apparatus checks are clean.**", "",
        "## Gate 0 — census / split / provenance", "",
        f"- master variants: **{g0['master_variant_count']}**",
        f"- canonical selected/renderable: **{g0['selected_count']}**",
        f"- IRIS-capable: **{g0['capabilities']['iris']}**",
        f"- Geppetto-capable: **{g0['capabilities']['geppetto']}**",
        f"- Arachne-capable: **{g0['capabilities']['arachne']}**",
        f"- deterministic split mismatches: **{g0['split_mismatch_count']}**",
        f"- unresolved pending latest-state records: **{g0['latest_unresolved_pending_count']}**",
        f"- duplicate source-SHA groups: **{g0['provenance']['duplicate_source_sha_group_count']}**", "",
        "### selected source distribution", "```json", json.dumps(g0["selected_source_distribution"], indent=2), "```", "",
        "### selected split distribution", "```json", json.dumps(g0["selected_split_distribution"], indent=2), "```", "",
        "## Gate 1 — full shallow render/data-plane scan", "",
        f"- assets scanned: **{sh['asset_count']}**",
        f"- assets with any shallow issue: **{sh['assets_with_issues']}**",
        f"- total shallow issues: **{sh['issue_count']}**", "",
        "## Gates 1–2 — deep deterministic sample", "",
        f"Deep sample assets: **{len(r['deep'])}**", "",
        "Deep per-asset/per-view numeric details are in `POST_CORPUS_AUDIT_RESULT.json`.", "",
        "## Gate 3 — appearance authority warning", "",
        "The current `cel_clean`/`ink_cel` render is a geometry-isolation observation pass. It must not be silently promoted to the sole natural/product observation authority. Raw-source appearance preservation and external texture dependencies require a separate coverage audit before appearance-sensitive representation conclusions.", "",
        "## Scope boundary", "",
        "This runner does not execute Representation Authority (Gate 4), SOI-2 (Gate 5), IRIS architecture parity (Gate 6), training, rerendering, consumer export, or downstream model evaluation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3")
    ap.add_argument("--deep-sample", type=int, default=128, help="0 means all selected assets")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    outdir = Path(args.out_dir) if args.out_dir else root / "reports" / "post_corpus_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    master = read_jsonl(root / "ledgers" / "MASTER_VARIANTS.jsonl")
    processed = read_jsonl(root / "ledgers" / "PROCESSED_VARIANTS.jsonl")
    pending = read_jsonl(root / "ledgers" / "PENDING.jsonl")
    selected_raw = read_json(root / "metadata" / "CANONICAL_VARIANT_SELECTION.json")
    if isinstance(selected_raw, dict):
        if "selected" in selected_raw and isinstance(selected_raw["selected"], list):
            selected = selected_raw["selected"]
        else:
            selected = list(selected_raw.values())
    else:
        selected = selected_raw
    if not isinstance(selected, list):
        raise RuntimeError("unexpected CANONICAL_VARIANT_SELECTION schema")

    split_bad = []
    for s in selected:
        expected = deterministic_split(s["canonical_asset_id"], False)
        if s.get("split") != expected:
            split_bad.append({"asset": s["canonical_asset_id"], "stored": s.get("split"), "expected": expected})

    prov = audit_provenance(root, selected)
    unresolved = latest_unresolved_pending(pending)
    g0 = {
        "master_variant_count": len(master),
        "processed_variant_count": len(processed),
        "selected_count": len(selected),
        "master_unique_candidate_ids": len({x.get("candidate_id") for x in master}),
        "master_unique_canonical_asset_ids": len({x.get("canonical_asset_id") for x in master}),
        "master_decisions": counter(master, "decision"),
        "processed_decisions": counter(processed, "decision"),
        "selected_source_distribution": counter(selected, "source_registry_id"),
        "selected_split_distribution": counter(selected, "split"),
        "capabilities": {c: cap_count(selected, c) for c in ("iris", "geppetto", "arachne")},
        "split_mismatch_count": len(split_bad),
        "split_mismatch_examples": split_bad[:20],
        "latest_unresolved_pending_count": len(unresolved),
        "latest_unresolved_pending_status": counter(unresolved, "status"),
        "latest_unresolved_pending_source": counter(unresolved, "source"),
        "provenance": prov,
    }

    shallow_rows = []
    for i, s in enumerate(selected, 1):
        shallow_rows.append(shallow_asset_audit(root, s))
        if i % 100 == 0 or i == len(selected):
            print(f"[shallow] {i}/{len(selected)}", flush=True)
    issue_rows = [x for x in shallow_rows if x["issues"]]
    issue_counter = collections.Counter(y for x in issue_rows for y in x["issues"])
    shallow = {
        "asset_count": len(shallow_rows),
        "assets_with_issues": len(issue_rows),
        "issue_count": sum(len(x["issues"]) for x in issue_rows),
        "issue_types": dict(issue_counter.most_common()),
        "issue_examples": issue_rows[:100],
    }

    deep_sel = choose_deep_sample(selected, args.deep_sample)
    deep = []
    for i, s in enumerate(deep_sel, 1):
        print(f"[deep] {i}/{len(deep_sel)} {s['canonical_asset_id']}", flush=True)
        deep.append(deep_asset_audit(root, s))

    result = {
        "schema_version": "realsas.post_corpus_audit.v1",
        "build_id": BUILD_EXPECTED,
        "root": str(root),
        "gate0": g0,
        "shallow": shallow,
        "deep_sample_policy": {"requested": args.deep_sample, "actual": len(deep_sel), "stratified": True},
        "deep": deep,
        "training_authorized_by_this_audit": False,
        "notes": [
            "Current cel/ink renders are geometry-isolation controls, not sole appearance authority.",
            "Deep max reprojection outliers must be interpreted with projected-triangle conditioning; max-only failure is forbidden.",
            "This audit never rerenders or mutates corpus evidence.",
        ],
    }
    (outdir / "POST_CORPUS_AUDIT_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (outdir / "POST_CORPUS_AUDIT_REPORT.md").write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(outdir),
        "selected": len(selected),
        "shallow_assets_with_issues": shallow["assets_with_issues"],
        "deep_assets": len(deep),
        "split_mismatches": g0["split_mismatch_count"],
        "duplicate_source_sha_groups": prov["duplicate_source_sha_group_count"],
        "latest_unresolved_pending": len(unresolved),
    }, indent=2))


if __name__ == "__main__":
    main()
