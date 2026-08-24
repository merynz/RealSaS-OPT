from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

EXPECTED_REPRESENTATION_SEED_SHA256 = "f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af"
EXPECTED_PANEL_DIGEST = "366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961"
EXPECTED_MEMBERSHIP_SHA256 = "ef19120e2cc98fac50a0c94ba863fd3cc4cec4a8392b9c48684f96ffeba3ba50"
ROLES = ("FIT_TRAIN", "FIT_SELECT", "TUNE_FINAL")


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest_ids(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the learner mini seed from the CI104 representation panel.")
    ap.add_argument("--representation-seed", required=True)
    ap.add_argument("--membership", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if sha256_file(a.representation_seed) != EXPECTED_REPRESENTATION_SEED_SHA256:
        raise RuntimeError("representation seed SHA drift")
    if sha256_file(a.membership) != EXPECTED_MEMBERSHIP_SHA256:
        raise RuntimeError("mini membership SHA drift")

    rep = json.load(open(a.representation_seed, encoding="utf-8"))
    mem = json.load(open(a.membership, encoding="utf-8"))
    if rep.get("record_count") != 256 or rep.get("sealed_splits_opened") is not False:
        raise RuntimeError("CI104 representation seed contract drift")
    if mem.get("source_panel_asset_id_digest") != EXPECTED_PANEL_DIGEST:
        raise RuntimeError("membership panel digest drift")
    if mem.get("sealed_splits_opened") is not False:
        raise RuntimeError("membership opened sealed split")

    rep_map = {r["asset_id"]: r for r in rep["records"]}
    panel_ids = [r["asset_id"] for r in rep["records"]]
    if digest_ids(panel_ids) != EXPECTED_PANEL_DIGEST:
        raise RuntimeError("representation panel identity drift")

    role_sets = {role: list(mem["roles"][role]) for role in ROLES}
    if [len(role_sets[x]) for x in ROLES] != [128, 32, 26]:
        raise RuntimeError("frozen mini role counts drift")
    if len(set().union(*(set(v) for v in role_sets.values()))) != sum(map(len, role_sets.values())):
        raise RuntimeError("mini roles overlap")

    for aid in role_sets["FIT_TRAIN"] + role_sets["FIT_SELECT"]:
        if aid not in rep_map or rep_map[aid]["split"] != "FIT":
            raise RuntimeError(f"FIT role mismatch {aid}")
    for aid in role_sets["TUNE_FINAL"]:
        if aid not in rep_map or rep_map[aid]["split"] != "TUNE":
            raise RuntimeError(f"TUNE role mismatch {aid}")

    records = []
    for role in ROLES:
        for aid in role_sets[role]:
            src = dict(rep_map[aid])
            src["mini_role"] = role
            records.append(src)

    out = {
        "schema": "RealSaS.IRISSinglePoseV2.MiniExtractabilitySeed.v1",
        "status": "FROZEN_OPEN_ONLY_MINI_SEED",
        "source_representation_seed_sha256": EXPECTED_REPRESENTATION_SEED_SHA256,
        "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "panel_asset_id_digest": EXPECTED_PANEL_DIGEST,
        "record_count": len(records),
        "role_counts": {k: len(v) for k, v in role_sets.items()},
        "records": records,
        "sealed_splits_opened": False,
        "optimizer_steps": 0,
    }
    atomic_json(a.out, out)
    print(json.dumps({k: out[k] for k in ("status", "record_count", "role_counts", "sealed_splits_opened")}, indent=2))


if __name__ == "__main__":
    main()
