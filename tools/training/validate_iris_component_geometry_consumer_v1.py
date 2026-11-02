#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed semantic court between IRIS geometry recipe and Corpus400 truth.

The court proves that a recipe can only consume component-scoped geometry bytes via
explicit target semantics. It does not instantiate an optimizer and does not mint
training readiness. A final corpus readiness receipt must bind this court before any
future optimizer is created.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.iris.truth.component_geometry_training_targets_v1 import (
    load_component_geometry_training_targets,
)


SCHEMA = "RealSaS.IrisComponentGeometryConsumerSemanticCourt.v1"
STATUS = "PASS_IRIS_COMPONENT_GEOMETRY_CONSUMER_SEMANTICS__TRAINING_READINESS_SEPARATE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _recipe_loss_roles(recipe: dict) -> dict[str, dict[str, float]]:
    geometry = dict(recipe.get("geometry") or {})
    roles = geometry.get("loss_weights_by_role")
    if not isinstance(roles, dict) or not roles:
        raise ValueError("IRIS_GEOMETRY_CONSUMER_RECIPE_LOSS_ROLES")
    clean = {}
    for role, weights in roles.items():
        if not isinstance(role, str) or not role or not isinstance(weights, dict) or not weights:
            raise ValueError("IRIS_GEOMETRY_CONSUMER_RECIPE_ROLE_LAYOUT")
        row = {}
        for name, value in weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError("IRIS_GEOMETRY_CONSUMER_RECIPE_LOSS_NAME")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or float(value) < 0:
                raise ValueError("IRIS_GEOMETRY_CONSUMER_RECIPE_LOSS_WEIGHT:" + name)
            row[name] = float(value)
        clean[role] = row
    return clean


def validate_recipe_semantics(contract: dict, recipe: dict) -> dict:
    requirements = dict(contract.get("recipe_requirements") or {})
    geometry = dict(recipe.get("geometry") or {})
    primary = str(geometry.get("primary_authority") or "")
    for token in requirements.get("primary_authority_must_include") or ():
        if str(token) not in primary:
            raise ValueError("IRIS_GEOMETRY_CONSUMER_PRIMARY_AUTHORITY_MISSING:" + str(token))
    if geometry.get("global_sign_policy") != requirements.get("global_sign_policy"):
        raise ValueError("IRIS_GEOMETRY_CONSUMER_GLOBAL_SIGN_POLICY")
    fstar = str(geometry.get("fstar_role") or "")
    prefix = str(requirements.get("fstar_role_prefix") or "")
    if not prefix or not fstar.startswith(prefix):
        raise ValueError("IRIS_GEOMETRY_CONSUMER_FSTAR_NOT_AUXILIARY")

    roles = _recipe_loss_roles(recipe)
    global_sign_losses = set(requirements.get("geometry_losses_requiring_compact_global_sign") or ())
    local_losses = set(requirements.get("geometry_losses_requiring_local_oriented_surface_truth") or ())
    active_losses = set()
    for role, weights in roles.items():
        if weights.get("surface_distance", 0.0) <= 0:
            raise ValueError("IRIS_GEOMETRY_CONSUMER_SURFACE_DISTANCE_DISABLED:" + role)
        for name, weight in weights.items():
            if weight > 0:
                active_losses.add(name)
        if weights.get("fstar_auxiliary", 0.0) >= weights.get("surface_distance", 0.0):
            raise ValueError("IRIS_GEOMETRY_CONSUMER_FSTAR_DOMINATES_PRIMARY:" + role)
    if not global_sign_losses.issubset(active_losses):
        missing = sorted(global_sign_losses - active_losses)
        raise ValueError("IRIS_GEOMETRY_CONSUMER_GLOBAL_SIGN_LOSS_MISSING:" + ",".join(missing))
    if not local_losses.issubset(active_losses):
        missing = sorted(local_losses - active_losses)
        raise ValueError("IRIS_GEOMETRY_CONSUMER_LOCAL_LOSS_MISSING:" + ",".join(missing))
    return {
        "primary_authority": primary,
        "global_sign_policy": geometry["global_sign_policy"],
        "fstar_role": fstar,
        "loss_roles": roles,
        "active_losses": sorted(active_losses),
    }


def validate_consumer(contract_path: Path, recipe_path: Path, geometry_manifest_path: Path, geometry_court_path: Path) -> dict:
    contract = load_json(contract_path, "IRIS_GEOMETRY_CONSUMER_CONTRACT")
    recipe = load_json(recipe_path, "IRIS_GEOMETRY_CONSUMER_RECIPE")
    manifest = load_json(geometry_manifest_path, "IRIS_GEOMETRY_CONSUMER_MANIFEST")
    court = load_json(geometry_court_path, "IRIS_GEOMETRY_CONSUMER_COURT")

    if contract.get("schema") != "RealSaS.IrisComponentGeometryConsumerContract.v1":
        raise ValueError("IRIS_GEOMETRY_CONSUMER_CONTRACT_SCHEMA")
    if manifest.get("schema") != contract.get("accepted_geometry_manifest_schema"):
        raise ValueError("IRIS_GEOMETRY_CONSUMER_MANIFEST_SCHEMA")
    if court.get("schema") != contract.get("required_geometry_court_schema") or court.get("status") != contract.get("required_geometry_court_status"):
        raise ValueError("IRIS_GEOMETRY_CONSUMER_COURT_STATUS")
    if manifest.get("subject_id") != court.get("subject_id"):
        raise ValueError("IRIS_GEOMETRY_CONSUMER_SUBJECT_DRIFT")
    parent = dict(court.get("parent_manifest") or {})
    if parent.get("sha256") and parent["sha256"] != sha256(geometry_manifest_path.resolve()):
        raise ValueError("IRIS_GEOMETRY_CONSUMER_COURT_PARENT_DRIFT")
    if not bool((court.get("authority") or {}).get("independent_court_passed")):
        raise ValueError("IRIS_GEOMETRY_CONSUMER_INDEPENDENT_COURT_REQUIRED")

    recipe_semantics = validate_recipe_semantics(contract, recipe)
    outputs = dict(manifest.get("outputs") or {})
    distance_path = bound(outputs.get("distance_sign"), "IRIS_GEOMETRY_CONSUMER_DISTANCE_DRIFT")
    surface_path = bound(outputs.get("surface"), "IRIS_GEOMETRY_CONSUMER_SURFACE_DRIFT")
    target = load_component_geometry_training_targets(distance_path, surface_path)
    global_rows = target.global_sign_rows()
    epsilons = tuple((recipe.get("geometry") or {}).get("surface_bracket_epsilons") or ())
    if not epsilons:
        raise ValueError("IRIS_GEOMETRY_CONSUMER_BRACKET_EPSILONS")
    local = target.local_oriented_brackets(epsilons)

    query_count = int(len(target.query_xyz))
    signed_count = int(len(global_rows["query_index"]))
    if signed_count > query_count:
        raise ValueError("IRIS_GEOMETRY_CONSUMER_SIGN_CARDINALITY")
    if signed_count == 0 and any(
        weights.get(name, 0.0) > 0
        for weights in recipe_semantics["loss_roles"].values()
        for name in ("signed_tsdf_when_valid", "occupancy_when_valid")
    ):
        # Zero valid sign rows are scientifically acceptable for an individual
        # subject, but the consumer must make the zero-contribution semantics explicit.
        global_sign_mode = "ZERO_VALID_ROWS__ZERO_GLOBAL_SIGN_LOSS_CONTRIBUTION"
    else:
        global_sign_mode = "COMPACT_VALID_ROWS_ONLY"

    bindings = {
        "surface_distance": {
            "query_count": query_count,
            "valid_count": int(np.count_nonzero(target.unsigned_valid)),
            "source": "unsigned_distance",
        },
        "global_sign": {
            "mode": global_sign_mode,
            "valid_query_count": signed_count,
            "query_index_is_compact": True,
            "unknown_dense_fill_performed": False,
        },
        "oriented_normal": {
            "surface_count": int(len(target.surface_xyz)),
            "valid_count": int(np.count_nonzero(target.surface_normal_valid)),
        },
        "local_oriented_sdf": {
            "query_count": int(len(local["query_xyz"])),
            "semantics": str(local["semantics"][0]),
            "global_occupancy_labels_used": False,
            "epsilons": list(map(float, epsilons)),
        },
        "fstar": {
            "role": recipe_semantics["fstar_role"],
            "primary_replacement_allowed": False,
        },
    }

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": manifest["subject_id"],
        "inputs": {
            "consumer_contract": {"path": str(contract_path.resolve()), "sha256": sha256(contract_path.resolve())},
            "recipe": {"path": str(recipe_path.resolve()), "sha256": sha256(recipe_path.resolve())},
            "geometry_manifest": {"path": str(geometry_manifest_path.resolve()), "sha256": sha256(geometry_manifest_path.resolve())},
            "geometry_court": {"path": str(geometry_court_path.resolve()), "sha256": sha256(geometry_court_path.resolve())},
        },
        "recipe_semantics": recipe_semantics,
        "target_bindings": bindings,
        "firewall": {
            "dense_unknown_signed_target_created": False,
            "dense_unknown_occupancy_target_created": False,
            "local_oriented_sign_used_as_global_occupancy": False,
            "fstar_used_as_primary_geometry_authority": False,
            "optimizer_created": False,
        },
        "authority": {
            "consumer_semantics_qualified": True,
            "corpus_admission_minted": False,
            "training_readiness_minted": False,
            "training_authorized": False,
        },
    }
    payload["receipt_identity_sha256"] = digest(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumer-contract", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-court", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_consumer(args.consumer_contract, args.recipe, args.geometry_manifest, args.geometry_court)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "subject_id": payload["subject_id"], "receipt_identity_sha256": payload["receipt_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
