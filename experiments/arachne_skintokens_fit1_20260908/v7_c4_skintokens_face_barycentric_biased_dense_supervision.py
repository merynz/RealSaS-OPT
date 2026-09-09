from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = "RealSaS.ArachneMageA0V7C4SkinTokensFaceBarycentricBiasedDenseSupervision.v1"
PREREG_COMMIT = "372cb2dd510183800460b496a7ecfbeb1204369f"
PARENT_C3_SOURCE_COMMIT = "f50d58a1067a914a99e200df667b35f522aabd5b"
PARENT_C3_RESULT_SHA256 = "5fe9c350e2f09a1fd858d86d39c965c9c429321c76851994379a0466d925eaa0"
SKINTOKENS_UPSTREAM_COMMIT = "273b691d35989d71cd17ff2895fdc735097b92d1"
SKINTOKENS_SAMPLER_BLOB = "ef7bc0d8fccd2bd4e0b2bd13887c1c96b2caa6dc"
SKINTOKENS_UTILS_BLOB = "80a89f9fa5c3a3e2a6ef0646071ff638e640f3d1"

EXPECTED_NORMALIZED_SOURCE_SHA256 = "528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f"
EXPECTED_CACHE_SHA256 = "db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd"
EXPECTED_SKELETON_LINEAGE = "738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306"
EXPECTED_TEACHER_WEIGHTS_SHA256 = "7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d"
EXPECTED_C3_CONTROL_SCHEDULE_SHA256 = "e7a257f20a447e5b34ff850658851e937691fc60aed4c6286d6a8043b071cef2"
EXPECTED_COMMON_GLOBAL_PREFIX_SHA256 = "0875c257bb53f06439bb3de9eb28d4bc642cfb8807cff1c703a3c3932d74268e"
EXPECTED_FACE_BARY_SCHEDULE_SHA256 = "d6c9b895233414caffb0449902bcd3dda929bc6296ce274721a0aae57566c842"

GENERATION_TO_SOURCE_CONTROL = np.asarray(
    [1, 19, 20, 21, 22, 2, 3, 9, 10, 11, 12, 13, 14, 4, 5, 6, 7, 8, 15, 16, 17, 18],
    dtype=np.int64,
)

REFERENCE_ARM = "C4_REFERENCE_IMPORTANCE_ACTIVE_ONLY"
BIASED_ARM = "C4_BIASED_ACTIVE_ONLY"
FACE_BARY_ARM = "C4_SKINTOKENS_FACE_BARY_BIASED"


@dataclass(frozen=True)
class C4Contract:
    seed: int = 20260909
    optimizer_steps: int = 384
    query_count: int = 384
    dense_fraction: float = 0.5
    active_epsilon: float = 1e-8
    upstream_max_distance: float = 0.1
    upstream_rate_distance: float = 0.1
    num_vertex_samples: int = 0
    prefix_min: int = 1
    prefix_max: int = 4


CONTRACT = C4Contract()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def raw_array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order="C")).hexdigest()


def _subseed(seed: int, step: int, joint: int, stream: int) -> int:
    payload = f"{int(seed)}:{int(step)}:{int(joint)}:{int(stream)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def exact_positive_pool(teacher_joint: np.ndarray, supervised: np.ndarray, active_epsilon: float = 1e-8) -> np.ndarray:
    t = np.asarray(teacher_joint, np.float64)
    s = np.asarray(supervised, bool)
    if t.ndim != 1 or s.shape != t.shape:
        raise ValueError("C4 positive-pool shape drift")
    pool = np.flatnonzero(s & (t > float(active_epsilon))).astype(np.int64)
    if pool.size == 0:
        raise RuntimeError("C4 empty active GSA pool")
    return pool


def uniform_target_probabilities(supervised: np.ndarray) -> np.ndarray:
    s = np.asarray(supervised, bool)
    n = int(s.sum())
    if n <= 0:
        raise ValueError("C4 no supervised rows")
    u = np.zeros(len(s), np.float64)
    u[s] = 1.0 / n
    return u


def proposal_probabilities_from_pool(supervised: np.ndarray, dense_pool_indices: np.ndarray, dense_fraction: float = 0.5) -> np.ndarray:
    s = np.asarray(supervised, bool)
    p = np.asarray(dense_pool_indices, np.int64)
    if p.ndim != 1 or p.size == 0 or np.any(~s[p]):
        raise ValueError("C4 invalid dense pool")
    q = np.zeros(len(s), np.float64)
    q[s] += (1.0 - float(dense_fraction)) / float(s.sum())
    q[p] += float(dense_fraction) / float(p.size)
    if abs(float(q.sum()) - 1.0) > 1e-12:
        raise RuntimeError("C4 proposal mass drift")
    return q


def importance_weights_from_pool(supervised: np.ndarray, dense_pool_indices: np.ndarray, dense_fraction: float = 0.5) -> np.ndarray:
    q = proposal_probabilities_from_pool(supervised, dense_pool_indices, dense_fraction)
    u = uniform_target_probabilities(supervised)
    w = np.zeros_like(q)
    m = q > 0
    w[m] = u[m] / q[m]
    if not np.isfinite(w).all() or np.any(w < 0):
        raise RuntimeError("C4 invalid importance weights")
    if float(np.max(np.abs(q * w - u))) > 1e-12:
        raise RuntimeError("C4 q*w != u")
    return w


def source_columns_from_lineage(joint_ids: list[str] | tuple[str, ...], skeleton_json: dict[str, Any]) -> np.ndarray:
    if skeleton_json.get("skeleton_lineage_hash") != EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("C4 skeleton lineage drift")
    rows = skeleton_json.get("joints", [])
    if len(rows) != 22:
        raise RuntimeError("C4 skeleton joint count drift")
    by_joint = {str(r["canonical_joint_id"]): str(r["source_proposal_id"]) for r in rows}
    if set(by_joint) != set(map(str, joint_ids)):
        raise RuntimeError("C4 skeleton/cache joint-id drift")
    generations = []
    for jid in map(str, joint_ids):
        prop = by_joint[jid]
        if not prop.startswith("P:GRS:"):
            raise RuntimeError(f"C4 proposal-id format drift::{prop}")
        gi = int(prop.rsplit(":", 1)[1])
        generations.append(gi)
    if sorted(generations) != list(range(22)):
        raise RuntimeError("C4 proposal generation-index completeness drift")
    source_cols = GENERATION_TO_SOURCE_CONTROL[np.asarray(generations, np.int64)]
    if len(set(source_cols.tolist())) != 22:
        raise RuntimeError("C4 source-control duplicate")
    return source_cols.astype(np.int64)


def load_source_authority(normalized_path: Path, joint_ids: list[str] | tuple[str, ...], skeleton_json: dict[str, Any], active_epsilon: float = 1e-8) -> dict[str, Any]:
    normalized_path = Path(normalized_path)
    if sha_file(normalized_path) != EXPECTED_NORMALIZED_SOURCE_SHA256:
        raise RuntimeError("C4 normalized source SHA drift")
    with np.load(normalized_path, allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
        skin41 = np.asarray(z["skin"], np.float64)
    if vertices.shape != (5321, 3) or faces.shape != (5763, 3) or skin41.shape != (5321, 41):
        raise RuntimeError(f"C4 normalized source shape drift::{vertices.shape}:{faces.shape}:{skin41.shape}")
    if not np.isfinite(vertices).all() or not np.isfinite(skin41).all() or np.any(skin41 < 0):
        raise RuntimeError("C4 normalized source numeric drift")

    source_cols = source_columns_from_lineage(joint_ids, skeleton_json)
    excluded = np.asarray(sorted(set(range(41)) - set(source_cols.tolist())), np.int64)
    excluded_mass = float(np.abs(skin41[:, excluded]).sum())
    if excluded_mass > 1e-12:
        raise RuntimeError(f"C4 non-bridge skin mass::{excluded_mass}")

    skin22 = skin41[:, source_cols]
    selected_mass = skin22.sum(axis=1)
    positive_vertex = selected_mass > float(active_epsilon)
    zero_vertex_count = int((~positive_vertex).sum())
    if zero_vertex_count != 42:
        raise RuntimeError(f"C4 zero selected-skin vertex count drift::{zero_vertex_count}")
    if np.max(np.abs(selected_mass[positive_vertex] - 1.0)) > 2e-6:
        raise RuntimeError("C4 selected source skin simplex drift")
    skin22n = np.zeros_like(skin22)
    skin22n[positive_vertex] = skin22[positive_vertex] / selected_mass[positive_vertex, None]

    eligible_mask = np.all(positive_vertex[faces], axis=1)
    eligible_face_ids = np.flatnonzero(eligible_mask).astype(np.int64)
    if int(eligible_mask.sum()) != 5683 or int((~eligible_mask).sum()) != 80:
        raise RuntimeError(f"C4 eligible-face count drift::{eligible_mask.sum()}:{(~eligible_mask).sum()}")

    e0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    cross = np.cross(e0, e1)
    face_weight = np.linalg.norm(cross, axis=1)
    if np.any(face_weight <= 0) or not np.isfinite(face_weight).all():
        raise RuntimeError("C4 degenerate source face")
    face_normals = cross / face_weight[:, None]

    return {
        "vertices": vertices,
        "faces": faces,
        "skin22": skin22n,
        "source_cols": source_cols,
        "eligible_mask": eligible_mask,
        "eligible_face_ids": eligible_face_ids,
        "face_normals": face_normals,
        "face_weight": face_weight,
        "zero_vertex_count": zero_vertex_count,
        "excluded_face_count": int((~eligible_mask).sum()),
        "excluded_source_control_mass_l1": excluded_mass,
    }


def skintokens_face_mask_for_joint(source: dict[str, Any], joint: int, max_distance: float = 0.1, rate_distance: float = 0.1) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(source["vertices"], np.float64)
    faces = np.asarray(source["faces"], np.int64)
    skin = np.asarray(source["skin22"], np.float64)[:, int(joint)]
    eligible_mask = np.asarray(source["eligible_mask"], bool)
    eligible_faces = faces[eligible_mask]
    eligible_ids = np.flatnonzero(eligible_mask).astype(np.int64)

    face_has_skin = np.any(skin[eligible_faces] > 0, axis=-1)
    if int(face_has_skin.sum()) == 0:
        mask_local = np.ones(len(eligible_faces), bool)
        radius = None
        active_vertex_count = 0
        active_face_count = 0
    elif float(max_distance) < 1e-5:
        mask_local = face_has_skin.copy()
        radius = 0.0
        active_vertex_count = int(np.unique(eligible_faces[face_has_skin].reshape(-1)).size)
        active_face_count = int(face_has_skin.sum())
    else:
        p = np.unique(eligible_faces[face_has_skin].reshape(-1))
        tree = cKDTree(vertices[p])
        dis, _ = tree.query(vertices, k=1)
        span = float(np.sqrt(((np.max(vertices[p], axis=0) - np.min(vertices[p], axis=0)) ** 2).sum()))
        radius = float(min(float(max_distance), span * float(rate_distance)))
        mask_face_near = np.any(dis[eligible_faces] < radius, axis=-1)
        mask_local = face_has_skin | mask_face_near
        active_vertex_count = int(p.size)
        active_face_count = int(face_has_skin.sum())

    face_ids = eligible_ids[mask_local]
    if face_ids.size == 0:
        raise RuntimeError(f"C4 empty face pool joint {joint}")
    return face_ids.astype(np.int64), {
        "joint": int(joint),
        "active_face_count": int(active_face_count),
        "pool_face_count": int(face_ids.size),
        "added_near_face_count": int(face_ids.size - active_face_count),
        "active_vertex_count": int(active_vertex_count),
        "radius": None if radius is None else float(radius),
    }


def _sample_face_barycentric(source: dict[str, Any], face_pool_ids: np.ndarray, count: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
    face_pool_ids = np.asarray(face_pool_ids, np.int64)
    if count <= 0 or face_pool_ids.size <= 0:
        raise ValueError("C4 invalid face sample request")
    face_weight = np.asarray(source["face_weight"], np.float64)[face_pool_ids]
    weight_cum = np.cumsum(face_weight, axis=0)
    if not np.isfinite(weight_cum[-1]) or weight_cum[-1] <= 0:
        raise RuntimeError("C4 face area mass drift")
    face_pick = rng.rand(int(count)) * weight_cum[-1]
    local_face_index = np.searchsorted(weight_cum, face_pick)
    face_ids = face_pool_ids[local_face_index].astype(np.int64)

    random_lengths = rng.rand(int(count), 2, 1)
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    faces = np.asarray(source["faces"], np.int64)
    vertices = np.asarray(source["vertices"], np.float64)
    tri = faces[face_ids]
    origins = vertices[tri[:, 0]]
    vectors = vertices[tri[:, 1:]] - origins[:, None, :]
    xyz = origins + (vectors * random_lengths).sum(axis=1)

    skin22 = np.asarray(source["skin22"], np.float64)
    s0 = skin22[tri[:, 0]]
    sv = skin22[tri[:, 1:]] - s0[:, None, :]
    skin_samples = s0 + (sv * random_lengths).sum(axis=1)
    target_mass = skin_samples.sum(axis=1)
    if np.max(np.abs(target_mass - 1.0)) > 3e-6 or np.any(skin_samples < -1e-8):
        raise RuntimeError("C4 dense barycentric skin simplex drift")

    face_normals = np.asarray(source["face_normals"], np.float64)[face_ids]
    r0 = random_lengths[:, 0, 0]
    r1 = random_lengths[:, 1, 0]
    bary = np.stack([1.0 - r0 - r1, r0, r1], axis=1)
    if np.min(bary) < -1e-12 or np.max(np.abs(bary.sum(axis=1) - 1.0)) > 1e-12:
        raise RuntimeError("C4 barycentric coordinate drift")

    return {
        "face_ids": face_ids,
        "bary": bary.astype(np.float64),
        "xyz": xyz.astype(np.float64),
        "face_normals": face_normals.astype(np.float64),
        "skin22": skin_samples.astype(np.float64),
    }


def build_c4_schedules(
    teacher_weights: np.ndarray,
    supervised: np.ndarray,
    source: dict[str, Any],
    *,
    steps: int = 384,
    query_count: int = 384,
    dense_fraction: float = 0.5,
    prefix_min: int = 1,
    prefix_max: int = 4,
    seed: int = 20260909,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tw = np.asarray(teacher_weights, np.float64)
    sup = np.asarray(supervised, bool)
    if tw.shape != (950, 22) or sup.shape != (950,):
        raise ValueError("C4 schedule target shape drift")
    if query_count <= 0 or steps <= 0 or not (0 < dense_fraction < 1):
        raise ValueError("C4 schedule cardinality drift")
    dense_count = int(round(int(query_count) * float(dense_fraction)))
    global_count = int(query_count) - dense_count
    if dense_count != 192 or global_count != 192:
        raise RuntimeError("C4 fixed 192/192 query split drift")

    sup_idx = np.flatnonzero(sup).astype(np.int64)
    active_pools = [exact_positive_pool(tw[:, j], sup) for j in range(22)]
    face_pools = []
    face_pool_summary = []
    for j in range(22):
        fp, meta = skintokens_face_mask_for_joint(
            source, j,
            max_distance=CONTRACT.upstream_max_distance,
            rate_distance=CONTRACT.upstream_rate_distance,
        )
        face_pools.append(fp)
        face_pool_summary.append(meta)

    reference: dict[int, list[np.ndarray]] = {}
    face_bary: dict[int, list[dict[str, np.ndarray]]] = {}
    prefixes: dict[int, list[int]] = {}
    permutations: dict[int, list[np.ndarray]] = {}
    global_rows_by_step: dict[int, list[np.ndarray]] = {}

    h_ref = hashlib.sha256()
    h_common = hashlib.sha256()
    h_face = hashlib.sha256()

    for step in range(1, int(steps) + 1):
        ref_rows = []
        face_rows = []
        pref_rows = []
        perm_rows = []
        global_rows = []
        for j in range(22):
            rg = np.random.RandomState(_subseed(seed, step, j, 1))
            global_idx = rg.choice(sup_idx, size=global_count, replace=False).astype(np.int64)
            rp = np.random.RandomState(_subseed(seed, step, j, 2))
            prefix = int(rp.randint(int(prefix_min), int(prefix_max) + 1))
            rd = np.random.RandomState(_subseed(seed, step, j, 3))
            dense_idx = rd.choice(active_pools[j], size=dense_count, replace=(active_pools[j].size < dense_count)).astype(np.int64)
            rs = np.random.RandomState(_subseed(seed, step, j, 4))
            permutation = rs.permutation(int(query_count)).astype(np.int64)

            combined_ref = np.concatenate([global_idx, dense_idx]).astype(np.int64)[permutation]

            rf = np.random.RandomState(_subseed(seed, step, j, 5))
            dense_face = _sample_face_barycentric(source, face_pools[j], dense_count, rf)

            ref_rows.append(combined_ref)
            face_rows.append({
                "global_idx": global_idx,
                "dense_face_ids": dense_face["face_ids"],
                "dense_bary": dense_face["bary"],
                "dense_xyz": dense_face["xyz"],
                "dense_face_normals": dense_face["face_normals"],
                "dense_skin22": dense_face["skin22"],
                "permutation": permutation,
            })
            pref_rows.append(prefix)
            perm_rows.append(permutation)
            global_rows.append(global_idx)

            h_ref.update(np.ascontiguousarray(combined_ref, dtype=np.int64).tobytes())
            h_ref.update(prefix.to_bytes(2, "little", signed=False))
            h_common.update(np.ascontiguousarray(global_idx, dtype=np.int64).tobytes())
            h_common.update(prefix.to_bytes(2, "little", signed=False))

            h_face.update(np.ascontiguousarray(global_idx, dtype=np.int64).tobytes())
            h_face.update(np.ascontiguousarray(dense_face["face_ids"], dtype=np.int64).tobytes())
            h_face.update(np.ascontiguousarray(dense_face["bary"], dtype=np.float64).tobytes())
            h_face.update(np.ascontiguousarray(permutation, dtype=np.int64).tobytes())
            h_face.update(prefix.to_bytes(2, "little", signed=False))

        reference[step] = ref_rows
        face_bary[step] = face_rows
        prefixes[step] = pref_rows
        permutations[step] = perm_rows
        global_rows_by_step[step] = global_rows

    meta = {
        "schema": SCHEMA + ".ScheduleMeta.v1",
        "reference_schedule_sha256": h_ref.hexdigest(),
        "common_global_prefix_sha256": h_common.hexdigest(),
        "face_bary_schedule_sha256": h_face.hexdigest(),
        "face_pool_summary": face_pool_summary,
        "query_count": int(query_count),
        "dense_count": dense_count,
        "global_count": global_count,
        "steps": int(steps),
        "seed": int(seed),
        "source_semantics": {
            "face_mask": "SkinTokens SamplerMix.sample_on_skin restricted to 5683 deformation-supported faces",
            "surface_sampling": "face area via norm(cross); upstream reflected random_lengths barycentric construction",
            "dense_normal": "sampled face normal",
            "num_vertex_samples": 0,
        },
    }
    if meta["reference_schedule_sha256"] != EXPECTED_C3_CONTROL_SCHEDULE_SHA256:
        raise RuntimeError(f"C4 reference schedule does not reproduce C3::{meta['reference_schedule_sha256']}")
    if meta["common_global_prefix_sha256"] != EXPECTED_COMMON_GLOBAL_PREFIX_SHA256:
        raise RuntimeError(f"C4 common global/prefix does not reproduce C3::{meta['common_global_prefix_sha256']}")
    if meta["face_bary_schedule_sha256"] != EXPECTED_FACE_BARY_SCHEDULE_SHA256:
        raise RuntimeError(f"C4 face-bary schedule drift::{meta['face_bary_schedule_sha256']}")
    return {
        "reference_rows": reference,
        "face_bary": face_bary,
        "prefixes": prefixes,
        "permutations": permutations,
        "global_rows": global_rows_by_step,
        "active_pools": active_pools,
        "face_pools": face_pools,
    }, meta


def run_synthetic_fixture() -> dict[str, Any]:
    vertices = np.asarray([
        [0.00,0.00,0.00],[0.10,0.00,0.00],[0.00,0.10,0.00],
        [0.12,0.00,0.00],[0.22,0.00,0.00],[0.12,0.10,0.00],
    ], np.float64)
    faces = np.asarray([[0,1,2],[3,4,5]], np.int64)
    skin22 = np.zeros((6,22), np.float64)
    skin22[:3, 0] = [1.0, 0.5, 0.25]
    skin22[:3, 1] = 1.0 - skin22[:3, 0]
    skin22[3:, 1] = 1.0
    cross = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]], vertices[faces[:,2]]-vertices[faces[:,0]])
    fw = np.linalg.norm(cross, axis=1)
    source = {
        "vertices": vertices,
        "faces": faces,
        "skin22": skin22,
        "eligible_mask": np.ones(2, bool),
        "eligible_face_ids": np.arange(2,dtype=np.int64),
        "face_normals": cross/fw[:,None],
        "face_weight": fw,
    }
    pool, meta = skintokens_face_mask_for_joint(source, 0, max_distance=0.1, rate_distance=1.0)
    if pool.tolist() != [0,1] or meta["active_face_count"] != 1 or meta["added_near_face_count"] != 1:
        raise RuntimeError(f"C4 fixture face expansion failed::{pool.tolist()}::{meta}")
    rng = np.random.RandomState(123)
    sample = _sample_face_barycentric(source, pool, 8, rng)
    if sample["xyz"].shape != (8,3) or sample["skin22"].shape != (8,22):
        raise RuntimeError("C4 fixture sample shape")
    out = {
        "status": "PASS",
        "face_pool": pool.tolist(),
        "pool_meta": meta,
        "sample_face_ids": sample["face_ids"].tolist(),
        "sample_bary_sha256": raw_array_sha(sample["bary"]),
        "sample_xyz_sha256": raw_array_sha(sample["xyz"]),
        "sample_skin_sha256": raw_array_sha(sample["skin22"]),
    }
    expected = {
        "sample_face_ids": [1,0,0,1,1,0,1,1],
        "sample_bary_sha256": "35d2790067bb3845d85b44d5e77b9dd81f8f1e1507056c855f190a22ff9dd130",
        "sample_xyz_sha256": "2c1b69686d465690ac606a54114949122a61c63620c4daa8cb6e76f8803a38b3",
        "sample_skin_sha256": "aa43033fe8a45a95a4e22a0b8d6de73dd12703dd6242a507f2fe138671679432",
    }
    for k,v in expected.items():
        if out[k] != v:
            raise RuntimeError(f"C4 fixture hash drift::{k}::{out[k]}::{v}")
    return out


if __name__ == "__main__":
    print(json.dumps(run_synthetic_fixture(), indent=2, sort_keys=True))
