from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

OPEN_SPLITS = {"FIT", "TUNE"}
SAME_LOCUS_TOL = 0.003
P_SIGMAS = (0.0, 0.0005, 0.0010, 0.0025, 0.0050, 0.0100)
N_DEGREES = (0.0, 5.0, 10.0, 20.0, 40.0)
PAIR_CATEGORIES = {"adjacent": 1, "skip_one": 2, "opposite": 4}
CONFIRMATORY_ASSETS = 256
CONFIRMATORY_PER_CATEGORY = 16
PN_NORMAL_WEIGHT = 0.05
HARD_TAIL_ERROR = 0.01


def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strongest_capability(record: dict) -> str:
    caps = record.get("capabilities")
    if not isinstance(caps, dict):
        raise RuntimeError("selection record missing capabilities mapping")

    def enabled(name: str) -> bool:
        return bool(caps.get(name)) or bool(caps.get(name.lower())) or bool(caps.get(name.upper()))

    if enabled("arachne"):
        return "ARACHNE"
    if enabled("geppetto"):
        return "GEPPETTO"
    if enabled("iris"):
        return "IRIS"
    raise RuntimeError("selection record has no IRIS/GEPPETTO/ARACHNE capability")


def component_count(vertices: np.ndarray, faces: np.ndarray) -> int:
    parent = np.arange(len(vertices), dtype=np.int64)
    used = np.zeros(len(vertices), dtype=bool)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for tri in np.asarray(faces, np.int64):
        a, b, c = map(int, tri)
        used[[a, b, c]] = True
        union(a, b)
        union(b, c)
        union(c, a)
    return len({find(int(i)) for i in np.flatnonzero(used)})


def component_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 16:
        return "5-16"
    return "17+"


def support_bucket(n: int) -> str:
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "6-8"


def pair_distance(a: int, b: int) -> int:
    d = abs(int(a) - int(b)) % 8
    return min(d, 8 - d)


def philox_rng(asset_id: str, arm_id: str, view: int) -> np.random.Generator:
    raw = hashlib.sha256(f"repr-v1-noise|{asset_id}|{arm_id}|V{view}".encode()).digest()[:16]
    seed = int.from_bytes(raw, "little", signed=False)
    return np.random.Generator(np.random.Philox(seed))


def perturb_normals(n: np.ndarray, degrees: float, rng: np.random.Generator) -> np.ndarray:
    n = np.asarray(n, np.float32)
    if degrees == 0.0:
        return n.copy()
    r = rng.normal(size=n.shape).astype(np.float32)
    tangent = r - (r * n).sum(-1, keepdims=True) * n
    tn = np.linalg.norm(tangent, axis=-1, keepdims=True)
    bad = tn[:, 0] < 1e-8
    if np.any(bad):
        nb = n[bad]
        axis = np.zeros_like(nb)
        choose_x = np.abs(nb[:, 0]) < 0.9
        axis[choose_x, 0] = 1.0
        axis[~choose_x, 1] = 1.0
        tangent[bad] = np.cross(nb, axis)
        tn = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = tangent / np.maximum(tn, 1e-8)
    theta = math.radians(float(degrees))
    out = math.cos(theta) * n + math.sin(theta) * tangent
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)


def observed_fields(asset_id: str, arm: dict, p_exact: np.ndarray, n_exact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tracks, views, _ = n_exact.shape
    pobs = np.broadcast_to(p_exact[:, None, :], (tracks, views, 3)).copy().astype(np.float32)
    nobs = n_exact.copy().astype(np.float32)
    sigma = float(arm["p_sigma"])
    degrees = float(arm["n_deg"])
    for view in range(views):
        rng = philox_rng(asset_id, arm["id"], view)
        if sigma > 0:
            pobs[:, view] += rng.normal(0.0, sigma, size=(tracks, 3)).astype(np.float32)
        if degrees > 0:
            nobs[:, view] = perturb_normals(nobs[:, view], degrees, rng)
    return pobs, nobs


def arm_definitions() -> list[dict]:
    arms = [
        {"id": "R0_P_EXACT", "family": "R0", "use_n": False, "p_sigma": 0.0, "n_deg": 0.0},
        {"id": "R1_PN_EXACT", "family": "R1", "use_n": True, "p_sigma": 0.0, "n_deg": 0.0},
    ]
    for sigma in P_SIGMAS:
        arms.append({
            "id": f"R2_P_SIGMA_{sigma:.4f}",
            "family": "R2",
            "use_n": False,
            "p_sigma": sigma,
            "n_deg": 0.0,
        })
    for sigma in P_SIGMAS:
        for degrees in N_DEGREES:
            arms.append({
                "id": f"R3_PN_SIGMA_{sigma:.4f}_N_{int(degrees):02d}",
                "family": "R3",
                "use_n": True,
                "p_sigma": sigma,
                "n_deg": degrees,
            })
    return arms


def rank_candidates(query_p, query_n, cand_p, cand_n, use_n: bool) -> np.ndarray:
    dp = np.linalg.norm(cand_p - query_p[None], axis=-1)
    if use_n:
        cosine = np.clip((cand_n * query_n[None]).sum(-1), -1.0, 1.0)
        score = dp + PN_NORMAL_WEIGHT * (1.0 - cosine)
    else:
        score = dp
    return np.argsort(score, kind="stable")


def choose_third_view(asset_id: str, track: int, source: int, target: int, visible: np.ndarray) -> int | None:
    candidates = [v for v in range(8) if v not in (source, target) and bool(visible[track, v])]
    if not candidates:
        return None
    return min(candidates, key=lambda v: stable_key(f"{asset_id}|{track}|{source}|{target}|third|{v}"))


def query_panel(asset_id: str, visible: np.ndarray, per_category: int) -> tuple[list[dict], dict]:
    rows = []
    shortfall = {}
    legal_tracks = np.flatnonzero(visible.sum(axis=1) >= 2)
    for category, distance in PAIR_CATEGORIES.items():
        candidates = []
        for track in legal_tracks:
            views = np.flatnonzero(visible[track])
            for source in views:
                for target in views:
                    if source == target or pair_distance(int(source), int(target)) != distance:
                        continue
                    candidates.append((int(track), int(source), int(target)))
        candidates.sort(key=lambda x: stable_key(f"{asset_id}|{x[0]}|{x[1]}|{x[2]}|{category}"))
        chosen = candidates[:per_category]
        shortfall[category] = max(0, per_category - len(chosen))
        for track, source, target in chosen:
            rows.append({"track": track, "source": source, "target": target, "category": category})
    return rows, shortfall


def selection_record(selection: dict, asset_id: str) -> dict:
    if asset_id not in selection:
        raise RuntimeError(f"{asset_id} absent from CANONICAL_VARIANT_SELECTION")
    record = selection[asset_id]
    if not isinstance(record, dict):
        raise RuntimeError(f"bad selection record for {asset_id}")
    return record


def select_assets(cache_rows: list[dict], selection: dict, target: int) -> list[dict]:
    enriched = []
    for row in cache_rows:
        asset_id = row["asset_id"]
        split = str(row["split"]).upper()
        if split not in OPEN_SPLITS:
            raise RuntimeError(f"sealed/non-open cache record {asset_id}:{split}")
        sel = selection_record(selection, asset_id)
        if str(sel.get("split", "")).upper() != split:
            raise RuntimeError(f"split mismatch {asset_id}: cache={split} selection={sel.get('split')}")
        provider = str(sel.get("source_registry_id", ""))
        if not provider:
            raise RuntimeError(f"missing source_registry_id for {asset_id}")
        enriched.append({
            **row,
            "source_registry_id": provider,
            "capability_class": strongest_capability(sel),
        })
    if len(enriched) < target:
        raise RuntimeError(f"only {len(enriched)} legal OPEN cached assets; need {target}")
    strata = defaultdict(list)
    for row in enriched:
        strata[(row["source_registry_id"], row["split"], row["capability_class"])].append(row)
    picked = {}
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: stable_key("repr-v1:" + r["asset_id"]))
        picked[bucket[0]["asset_id"]] = bucket[0]
    if len(picked) > target:
        raise RuntimeError(f"{len(picked)} required strata exceed frozen panel size {target}")
    for row in sorted(enriched, key=lambda r: stable_key("repr-v1:" + r["asset_id"])):
        if len(picked) >= target:
            break
        picked.setdefault(row["asset_id"], row)
    return list(picked.values())


def verify_stage_cache_audit(audit_path: str | Path, cache: dict) -> dict:
    audit = json.load(open(audit_path, encoding="utf-8"))
    if audit.get("status") != "PASS" or int(audit.get("fatal_asset_count", -1)) != 0:
        raise RuntimeError(f"stage/cache audit is not PASS: status={audit.get('status')} fatal={audit.get('fatal_asset_count')}")
    if int(audit.get("asset_count", -1)) != len(cache["records"]):
        raise RuntimeError(f"stage/cache audit membership mismatch audit={audit.get('asset_count')} cache={len(cache['records'])}")
    return audit


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"queries": 0}
    top1 = np.asarray([r["top1"] for r in rows], np.float64)
    top4 = np.asarray([r["top4"] for r in rows], np.float64)
    top8 = np.asarray([r["top8"] for r in rows], np.float64)
    err = np.asarray([r["physical_error"] for r in rows], np.float64)
    reciprocal = np.asarray([r["reciprocal_success"] for r in rows if r["reciprocal_success"] is not None], np.float64)
    cycle = np.asarray([r["cycle_success"] for r in rows if r["cycle_success"] is not None], np.float64)
    ambiguity = np.asarray([r["ambiguity_set_size"] for r in rows], np.float64)
    return {
        "queries": int(len(rows)),
        "top1": float(top1.mean()),
        "top4": float(top4.mean()),
        "top8": float(top8.mean()),
        "physical_error": {
            "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)),
            "p95": float(np.percentile(err, 95)),
            "max": float(err.max()),
        },
        "reciprocal_success": float(reciprocal.mean()) if len(reciprocal) else None,
        "cycle_success": float(cycle.mean()) if len(cycle) else None,
        "ambiguous_fraction": float(np.mean(ambiguity > 1)),
        "ambiguity_set_size_p95": float(np.percentile(ambiguity, 95)),
    }


def grouped_summary(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize_rows(group) for name, group in sorted(groups.items())}


def family_tail(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["asset_id"]].append(row)
    values = []
    for asset_id, group in groups.items():
        values.append({
            "asset_id": asset_id,
            "top1": float(np.mean([x["top1"] for x in group])),
            "top8": float(np.mean([x["top8"] for x in group])),
            "physical_error_p95": float(np.percentile([x["physical_error"] for x in group], 95)),
        })
    if not values:
        return {"assets": 0}
    return {
        "assets": len(values),
        "top1_p10": float(np.percentile([x["top1"] for x in values], 10)),
        "top1_median": float(np.median([x["top1"] for x in values])),
        "top8_p10": float(np.percentile([x["top8"] for x in values], 10)),
        "physical_error_p95_across_asset_p95": float(np.percentile([x["physical_error_p95"] for x in values], 95)),
    }


def load_asset_component_bucket(cache_record: dict) -> tuple[int, str]:
    asset_dir = Path(cache_record["asset_dir"])
    with np.load(asset_dir / "primary_geometry.npz", allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces"}:
            raise RuntimeError(f"IRIS firewall drift {cache_record['asset_id']}: {z.files}")
        vertices = np.asarray(z["vertices"], np.float32)
        faces = np.asarray(z["faces"], np.int64)
    count = component_count(vertices, faces)
    return count, component_bucket(count)


def evaluate_asset(cache_record: dict, per_category: int, arms: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    asset_id = cache_record["asset_id"]
    truth_path = Path(cache_record["truth_path"])
    if not truth_path.is_file():
        raise FileNotFoundError(truth_path)
    if cache_record.get("truth_sha256") and sha256_file(truth_path) != cache_record["truth_sha256"]:
        raise RuntimeError(f"truth sha mismatch {asset_id}")
    with np.load(truth_path, allow_pickle=False) as z:
        required = {"track_p", "track_n_view", "track_visible", "track_support"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{asset_id} missing truth fields {sorted(missing)}")
        p_exact = np.asarray(z["track_p"], np.float32)
        n_exact = np.asarray(z["track_n_view"], np.float32)
        visible = np.asarray(z["track_visible"], bool)
        support = np.asarray(z["track_support"], np.int64)
    if p_exact.ndim != 2 or p_exact.shape[1] != 3:
        raise RuntimeError(f"bad track_p shape {asset_id}: {p_exact.shape}")
    if n_exact.shape != (len(p_exact), 8, 3) or visible.shape != (len(p_exact), 8):
        raise RuntimeError(f"bad track observation shapes {asset_id}")

    queries, shortfall = query_panel(asset_id, visible, per_category)
    components, comp_bucket = load_asset_component_bucket(cache_record)
    all_rows = []
    hard_tail = []
    for arm in arms:
        p_obs, n_obs = observed_fields(asset_id, arm, p_exact, n_exact)
        exact_arm = arm["id"] in {"R0_P_EXACT", "R1_PN_EXACT"}
        for query in queries:
            track = query["track"]
            source = query["source"]
            target = query["target"]
            target_ids = np.flatnonzero(visible[:, target])
            order = rank_candidates(
                p_obs[track, source],
                n_obs[track, source],
                p_obs[target_ids, target],
                n_obs[target_ids, target],
                bool(arm["use_n"]),
            )
            ranked = target_ids[order]
            exact_distance_ranked = np.linalg.norm(p_exact[ranked] - p_exact[track][None], axis=-1)
            legal_ranked = exact_distance_ranked <= SAME_LOCUS_TOL
            top1_id = int(ranked[0])
            top1 = bool(legal_ranked[0])
            top4 = bool(np.any(legal_ranked[:4]))
            top8 = bool(np.any(legal_ranked[:8]))
            physical_error = float(exact_distance_ranked[0])
            ambiguity_size = int(np.sum(np.linalg.norm(p_exact[target_ids] - p_exact[track][None], axis=-1) <= SAME_LOCUS_TOL))

            source_ids = np.flatnonzero(visible[:, source])
            reverse_order = rank_candidates(
                p_obs[top1_id, target],
                n_obs[top1_id, target],
                p_obs[source_ids, source],
                n_obs[source_ids, source],
                bool(arm["use_n"]),
            )
            reverse_id = int(source_ids[reverse_order[0]])
            reciprocal_success = bool(np.linalg.norm(p_exact[reverse_id] - p_exact[track]) <= SAME_LOCUS_TOL)

            third = choose_third_view(asset_id, track, source, target, visible)
            cycle_success = None
            if third is not None:
                third_ids = np.flatnonzero(visible[:, third])
                target_to_third = rank_candidates(
                    p_obs[top1_id, target],
                    n_obs[top1_id, target],
                    p_obs[third_ids, third],
                    n_obs[third_ids, third],
                    bool(arm["use_n"]),
                )
                third_id = int(third_ids[target_to_third[0]])
                third_to_source = rank_candidates(
                    p_obs[third_id, third],
                    n_obs[third_id, third],
                    p_obs[source_ids, source],
                    n_obs[source_ids, source],
                    bool(arm["use_n"]),
                )
                cycle_id = int(source_ids[third_to_source[0]])
                cycle_success = bool(np.linalg.norm(p_exact[cycle_id] - p_exact[track]) <= SAME_LOCUS_TOL)

            row = {
                "asset_id": asset_id,
                "provider": cache_record["source_registry_id"],
                "split": cache_record["split"],
                "capability": cache_record["capability_class"],
                "component_bucket": comp_bucket,
                "support_bucket": support_bucket(int(support[track])),
                "pair_category": query["category"],
                "arm": arm["id"],
                "track": int(track),
                "source": int(source),
                "target": int(target),
                "third": int(third) if third is not None else None,
                "top1": top1,
                "top4": top4,
                "top8": top8,
                "physical_error": physical_error,
                "reciprocal_success": reciprocal_success,
                "cycle_success": cycle_success,
                "ambiguity_set_size": ambiguity_size,
            }
            all_rows.append(row)
            if (not top8) or physical_error > HARD_TAIL_ERROR or (exact_arm and (not reciprocal_success or cycle_success is False)):
                hard_tail.append(row.copy())

    asset_info = {
        "asset_id": asset_id,
        "provider": cache_record["source_registry_id"],
        "split": cache_record["split"],
        "capability": cache_record["capability_class"],
        "connected_components": components,
        "component_bucket": comp_bucket,
        "tracks": int(len(p_exact)),
        "query_count": len(queries),
        "query_shortfall": shortfall,
    }
    return asset_info, all_rows, hard_tail


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimizer=0 R0-R3 representation authority study")
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--selection-json", required=True, help="master metadata/CANONICAL_VARIANT_SELECTION.json")
    ap.add_argument("--stage-cache-audit", required=True, help="PASS report from audit_staged_cache_v2.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel-out")
    ap.add_argument("--hard-tail-out")
    ap.add_argument("--dev-max-assets", type=int, default=0)
    ap.add_argument("--dev-per-category", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=8)
    args = ap.parse_args()

    cache = json.load(open(args.cache_manifest, encoding="utf-8"))
    if cache.get("schema") != "RealSaS.IRISSinglePoseV2.CacheManifest.v2":
        raise RuntimeError(f"bad cache schema: {cache.get('schema')}")
    selection = json.load(open(args.selection_json, encoding="utf-8"))
    if not isinstance(selection, dict):
        raise RuntimeError("CANONICAL_VARIANT_SELECTION must be an asset-id mapping")
    audit = verify_stage_cache_audit(args.stage_cache_audit, cache)

    development = bool(args.dev_max_assets or args.dev_per_category)
    asset_target = int(args.dev_max_assets) if args.dev_max_assets else CONFIRMATORY_ASSETS
    per_category = int(args.dev_per_category) if args.dev_per_category else CONFIRMATORY_PER_CATEGORY
    if asset_target <= 0 or per_category <= 0:
        raise ValueError("asset/query development overrides must be positive when used")

    try:
        chosen = select_assets(list(cache["records"]), selection, asset_target)
    except Exception as exc:
        fail = {
            "schema": "RealSaS.RepresentationAuthority.R0R3.v1",
            "optimizer_steps": 0,
            "status": "APPARATUS_TARGET_REOPEN_REQUIRED",
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(args.out, fail)
        raise

    panel = {
        "schema": "RealSaS.RepresentationAuthority.Panel.v1",
        "optimizer_steps": 0,
        "mode": "DEVELOPMENT_ONLY" if development else "CONFIRMATORY",
        "asset_target": asset_target,
        "per_category": per_category,
        "selected_asset_ids": [r["asset_id"] for r in chosen],
        "selected_strata": dict(Counter(
            f"{r['source_registry_id']}|{r['split']}|{r['capability_class']}" for r in chosen
        )),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "selection_json_sha256": sha256_file(args.selection_json),
        "stage_cache_audit_sha256": sha256_file(args.stage_cache_audit),
    }
    panel_path = Path(args.panel_out) if args.panel_out else Path(args.out).with_name("REPRESENTATION_AUTHORITY_PANEL_V1.json")
    atomic_json(panel_path, panel)
    panel_sha = sha256_file(panel_path)

    arms = arm_definitions()
    all_rows = []
    hard_tail = []
    asset_infos = []
    for index, record in enumerate(chosen, 1):
        info, rows, tail = evaluate_asset(record, per_category, arms)
        asset_infos.append(info)
        all_rows.extend(rows)
        hard_tail.extend(tail)
        if index % args.progress_every == 0 or index == len(chosen):
            print(f"[repr-r0-r3] {index}/{len(chosen)} rows={len(all_rows)} hard_tail={len(hard_tail)}", flush=True)

    by_arm = {}
    for arm in arms:
        rows = [r for r in all_rows if r["arm"] == arm["id"]]
        by_arm[arm["id"]] = {
            "definition": arm,
            "pooled": summarize_rows(rows),
            "family_tail": family_tail(rows),
            "by_provider": grouped_summary(rows, "provider"),
            "by_split": grouped_summary(rows, "split"),
            "by_capability": grouped_summary(rows, "capability"),
            "by_component_bucket": grouped_summary(rows, "component_bucket"),
            "by_support_bucket": grouped_summary(rows, "support_bucket"),
            "by_pair_category": grouped_summary(rows, "pair_category"),
        }

    hard_path = Path(args.hard_tail_out) if args.hard_tail_out else Path(args.out).with_name("REPRESENTATION_AUTHORITY_HARD_TAIL_V1.jsonl")
    atomic_jsonl(hard_path, hard_tail)
    status = "DEVELOPMENT_ONLY" if development else "R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED"
    result = {
        "schema": "RealSaS.RepresentationAuthority.R0R3.v1",
        "optimizer_steps": 0,
        "status": status,
        "mode": "DEVELOPMENT_ONLY" if development else "CONFIRMATORY",
        "panel_path": str(panel_path),
        "panel_sha256": panel_sha,
        "hard_tail_path": str(hard_path),
        "hard_tail_sha256": sha256_file(hard_path),
        "asset_count": len(chosen),
        "query_slots_per_asset_max": per_category * 3,
        "arm_count": len(arms),
        "selected_assets": asset_infos,
        "arms": by_arm,
        "exact_summary": {
            "R0_P_EXACT": by_arm["R0_P_EXACT"],
            "R1_PN_EXACT": by_arm["R1_PN_EXACT"],
        },
        "stage_cache_audit": {
            "status": audit.get("status"),
            "asset_count": audit.get("asset_count"),
            "fatal_asset_count": audit.get("fatal_asset_count"),
        },
        "decision_discipline": "Measurement only. No threshold-derived P/R/SOI2 decision is emitted by this runner; canonical interpretation is separate and no learner training is authorized.",
    }
    atomic_json(args.out, result)
    print(json.dumps({
        "status": status,
        "asset_count": len(chosen),
        "arm_count": len(arms),
        "R0": by_arm["R0_P_EXACT"]["pooled"],
        "R1": by_arm["R1_PN_EXACT"]["pooled"],
        "hard_tail_rows": len(hard_tail),
    }, indent=2))


if __name__ == "__main__":
    main()
