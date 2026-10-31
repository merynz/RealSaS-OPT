#!/usr/bin/env python3
from __future__ import annotations

"""Build a read-only 13-family evidence inventory for the historical Truth139 set.

This consumes the already hash-bound pretrain readiness and Full139 response receipt.
It catalogues exact existing artifacts under the new shared truth-family vocabulary,
but deliberately leaves every new authority row PENDING. Legacy PASS strings are
provenance, not automatic promotion into the CorpusTruthAuthority contract.

An optional structural-audit root can be attached after the new 139 structural
measurement has executed. Its presence enriches geometry/topology/sign/normal
inventory, but still cannot mint family authority or final admission.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corpus.truth_family_inventory_v1 import (
    SCHEMA as SUBJECT_SCHEMA,
    bound_ref,
    digest,
    evidence_descriptor,
    inventory_subject,
    load_json,
    sha256,
    validate_inventory,
    verified_ref,
)


SCHEMA = "RealSaS.Truth139TruthFamilyInventory.v1"
STATUS = "PASS_TRUTH139_TRUTH_FAMILY_EVIDENCE_INVENTORIED__NO_AUTHORITY_MINTED"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    with staged.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    staged.replace(path)


def _json_descriptor(label: str, path: Path, expected_sha: str | None = None, *, role: str = "LEGACY_OR_EXISTING_EVIDENCE") -> dict:
    ref = verified_ref(path, expected_sha)
    payload = load_json(path, "TRUTH139_INVENTORY_JSON:" + label)
    return evidence_descriptor(label, ref, payload, role=role)


def _bound_descriptor(label: str, row: dict | None, code: str, *, role: str = "LEGACY_OR_EXISTING_EVIDENCE") -> tuple[dict, dict]:
    ref, payload = bound_ref(row, code)
    return evidence_descriptor(label, ref, payload, role=role), payload


def _add_if_file(bucket: dict[str, list[dict]], family: str, label: str, path: Path, *, role: str = "LEGACY_OR_EXISTING_EVIDENCE") -> dict | None:
    if not path.is_file():
        return None
    descriptor = _json_descriptor(label, path, role=role)
    bucket[family].append(descriptor)
    return load_json(path, "TRUTH139_INVENTORY_OPTIONAL:" + label)


def _source_sha(old_subject: dict, chain: dict) -> str:
    candidates = [
        old_subject.get("source_sha256"),
        ((chain.get("inputs") or {}).get("source") or {}).get("sha256"),
        ((chain.get("prepared_packet_lineage") or {}).get("source") or {}).get("sha256"),
    ]
    for value in candidates:
        value = str(value or "")
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
            return value
    raise ValueError("TRUTH139_INVENTORY_SOURCE_SHA_MISSING")


def _chain_path(binding: dict, subject_id: str) -> tuple[Path, dict, dict]:
    path = Path(str(binding.get("chain_path") or "")).resolve()
    expected = str(binding.get("chain_sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError("TRUTH139_INVENTORY_RESPONSE_CHAIN_DRIFT:" + subject_id)
    payload = load_json(path, "TRUTH139_INVENTORY_RESPONSE_CHAIN")
    return path, {"path": str(path), "sha256": expected}, payload


def _packet_root(binding: dict, chain: dict, subject_id: str) -> Path:
    candidates = [
        (binding.get("prepared_packet_lineage") or {}).get("packet_root"),
        (chain.get("inputs") or {}).get("packet_root"),
        (chain.get("prepared_packet") or {}).get("path"),
    ]
    for value in candidates:
        if value:
            path = Path(str(value)).resolve()
            if path.is_dir():
                return path
    raise FileNotFoundError("TRUTH139_INVENTORY_PACKET_ROOT:" + subject_id)


def _structural_detail(structural_root: Path | None, subject_id: str) -> Path | None:
    if structural_root is None:
        return None
    candidates = (
        structural_root / "subjects" / subject_id / "STRUCTURAL_GEOMETRY_AUDIT_V1.json",
        structural_root / subject_id / "STRUCTURAL_GEOMETRY_AUDIT_V1.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def collect_subject_evidence(
    *,
    subject_id: str,
    old_subject: dict,
    response_binding: dict,
    structural_root: Path | None,
) -> dict:
    chain_path, chain_ref, chain = _chain_path(response_binding, subject_id)
    packet = _packet_root(response_binding, chain, subject_id)
    source_sha = _source_sha(old_subject, chain)
    bucket: dict[str, list[dict]] = {
        family: []
        for family in (
            "SOURCE_IMPORT",
            "CAMERA_OBSERVATION",
            "SURFACE_GEOMETRY",
            "SURFACE_TOPOLOGY",
            "SIGNED_DISTANCE_OCCUPANCY",
            "ORIENTED_NORMALS",
            "APPEARANCE_CAA",
            "VISIBILITY_OCCLUSION",
            "RAY_COMPOSITING",
            "CAUSAL_RESPONSE",
            "RIG_HIERARCHY",
            "SKIN_WEIGHTS",
            "ANIMATION_KINEMATICS",
        )
    }

    source_scene = _add_if_file(
        bucket,
        "SOURCE_IMPORT",
        "SOURCE_SCENE_SNAPSHOT",
        packet / "SOURCE_SCENE_SNAPSHOT.json",
        role="SOURCE_IMPORT_EVIDENCE",
    )
    rest_candidate = _add_if_file(
        bucket,
        "SOURCE_IMPORT",
        "SOURCE_REST_RENDER_CANDIDATE",
        packet / "SOURCE_REST_RENDER_CANDIDATE.json",
        role="SOURCE_IMPORT_EVIDENCE",
    )
    camera = _add_if_file(
        bucket,
        "CAMERA_OBSERVATION",
        "CAMERA_BUNDLE",
        packet / "camera_bundle.json",
        role="OBSERVATION_EVIDENCE",
    )
    observation = _add_if_file(
        bucket,
        "CAMERA_OBSERVATION",
        "IRIS_OBSERVATION_ADMISSION",
        packet / "IRIS_OBSERVATION_ADMISSION.json",
        role="LEGACY_OBSERVATION_ADMISSION_EVIDENCE",
    )

    # Exact legacy pretrain artifacts are useful evidence, but do not become new authority.
    if old_subject.get("geometry"):
        descriptor, geometry = _bound_descriptor(
            "LEGACY_V32_GEOMETRY",
            old_subject["geometry"],
            "TRUTH139_INVENTORY_GEOMETRY_DRIFT:" + subject_id,
        )
        for family in (
            "SURFACE_GEOMETRY",
            "SURFACE_TOPOLOGY",
            "SIGNED_DISTANCE_OCCUPANCY",
            "ORIENTED_NORMALS",
        ):
            bucket[family].append(dict(descriptor))
    else:
        geometry = None

    for key, label, families in (
        ("rgb", "LEGACY_V32_RGB", ("APPEARANCE_CAA",)),
        ("rgb_court", "LEGACY_V32_RGB_COURT", ("APPEARANCE_CAA",)),
        ("stage48", "LEGACY_STAGE48_RENDERER_WITNESS", ("VISIBILITY_OCCLUSION", "RAY_COMPOSITING")),
        ("tessa_teacher", "LEGACY_TESSA_TEACHER", ("SURFACE_TOPOLOGY", "CAUSAL_RESPONSE")),
    ):
        if old_subject.get(key):
            descriptor, _ = _bound_descriptor(
                label,
                old_subject[key],
                "TRUTH139_INVENTORY_" + key.upper() + "_DRIFT:" + subject_id,
            )
            for family in families:
                bucket[family].append(dict(descriptor))

    outputs = dict(chain.get("outputs") or {})
    for key, label in (
        ("response_manifest", "IRIS_RESPONSE_EVIDENCE_MANIFEST"),
        ("response_court", "IRIS_RESPONSE_EVIDENCE_COURT"),
        ("teacher_scene", "TEACHER_SCENE"),
    ):
        if outputs.get(key):
            descriptor, payload = _bound_descriptor(
                label,
                outputs[key],
                "TRUTH139_INVENTORY_CHAIN_OUTPUT_DRIFT:" + subject_id + ":" + key,
                role="RESPONSE_OR_TEACHER_EVIDENCE",
            )
            if key.startswith("response_"):
                bucket["CAUSAL_RESPONSE"].append(descriptor)
            else:
                # TeacherScene carries privileged rig/skin/animation response provenance.
                for family in ("RIG_HIERARCHY", "SKIN_WEIGHTS", "ANIMATION_KINEMATICS", "CAUSAL_RESPONSE"):
                    bucket[family].append(dict(descriptor))

    # The rest candidate is privileged evidence for rig/skin/animation availability only.
    # Its own status fields frequently say PENDING; inventory preserves that state.
    if rest_candidate is not None:
        ref = verified_ref(packet / "SOURCE_REST_RENDER_CANDIDATE.json")
        descriptor = evidence_descriptor(
            "SOURCE_REST_RENDER_CANDIDATE_PRIVILEGED",
            ref,
            rest_candidate,
            role="PRIVILEGED_SOURCE_RIG_SKIN_ANIMATION_EVIDENCE",
        )
        for family in ("RIG_HIERARCHY", "SKIN_WEIGHTS", "ANIMATION_KINEMATICS"):
            bucket[family].append(dict(descriptor))

    detail = _structural_detail(structural_root, subject_id)
    if detail is not None:
        descriptor = _json_descriptor(
            "CORPUS400_STRUCTURAL_GEOMETRY_AUDIT",
            detail,
            role="CURRENT_STRUCTURAL_MEASUREMENT_EVIDENCE",
        )
        for family in (
            "SURFACE_GEOMETRY",
            "SURFACE_TOPOLOGY",
            "SIGNED_DISTANCE_OCCUPANCY",
            "ORIENTED_NORMALS",
        ):
            bucket[family].append(dict(descriptor))

    inventory = inventory_subject(
        subject_id=subject_id,
        source_sha256=source_sha,
        family_evidence=bucket,
    )
    inventory["lineage"] = {
        "response_chain": chain_ref,
        "packet_root": str(packet),
    }
    inventory["source_observations"] = {
        "source_scene_present": source_scene is not None,
        "rest_candidate_present": rest_candidate is not None,
        "camera_bundle_present": camera is not None,
        "observation_admission_present": observation is not None,
        "legacy_geometry_present": geometry is not None,
        "current_structural_audit_present": detail is not None,
    }
    inventory["inventory_identity_sha256"] = digest(
        {key: value for key, value in inventory.items() if key != "inventory_identity_sha256"}
    )
    validate_inventory(inventory)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-readiness", type=Path, required=True)
    parser.add_argument("--response-receipt", type=Path, required=True)
    parser.add_argument("--structural-audit-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    prior = load_json(args.prior_readiness, "TRUTH139_INVENTORY_PRIOR_READINESS")
    response = load_json(args.response_receipt, "TRUTH139_INVENTORY_RESPONSE_RECEIPT")
    if prior.get("status") != "PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY" or int(prior.get("subject_count", -1)) != 139:
        raise ValueError("TRUTH139_INVENTORY_PRIOR_NOT_READY")
    if response.get("status") != "PASS_FULL139_IRIS_RESPONSE_EVIDENCE_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE" or int(response.get("completed_subject_count", -1)) != 139:
        raise ValueError("TRUTH139_INVENTORY_RESPONSE_NOT_139")

    subjects = dict(prior.get("subjects") or {})
    bindings = dict(response.get("subject_output_bindings") or {})
    if len(subjects) != 139 or set(subjects) != set(bindings):
        raise ValueError("TRUTH139_INVENTORY_SUBJECT_SET_DRIFT")

    structural_root = args.structural_audit_root.resolve() if args.structural_audit_root else None
    output_root = args.output_root.resolve()
    rows = []
    failures = {}
    family_present = Counter()

    for index, subject_id in enumerate(sorted(subjects), 1):
        print(f"[TRUTH-FAMILY-INVENTORY {index:03d}/139] {subject_id}", flush=True)
        try:
            inventory = collect_subject_evidence(
                subject_id=subject_id,
                old_subject=dict(subjects[subject_id]),
                response_binding=dict(bindings[subject_id]),
                structural_root=structural_root,
            )
            subject_path = output_root / "subjects" / subject_id / "TRUTH_FAMILY_INVENTORY_V1.json"
            atomic_json(subject_path, inventory)
            present = [
                row["family"] for row in inventory["families"]
                if row["evidence_status"] == "PRESENT_UNQUALIFIED"
            ]
            family_present.update(present)
            rows.append(
                {
                    "subject_id": subject_id,
                    "source_sha256": inventory["source_sha256"],
                    "evidence_present_family_count": inventory["evidence_present_family_count"],
                    "present_families": present,
                    "authority_state": "PENDING",
                    "final_disposition": "PENDING_AUTHORITY_AUDIT",
                    "inventory": verified_ref(subject_path),
                }
            )
        except Exception as exc:
            failures[subject_id] = repr(exc)
            print(f"[TRUTH-FAMILY-INVENTORY-FAIL] {subject_id} {exc!r}", flush=True)
            if not args.continue_on_failure:
                break

    table = output_root / "TRUTH139_TRUTH_FAMILY_INVENTORY_TABLE_V1.jsonl"
    atomic_jsonl(table, rows)
    summary = {
        "schema": SCHEMA,
        "status": STATUS if len(rows) == 139 and not failures else "TRUTH139_TRUTH_FAMILY_INVENTORY_INCOMPLETE",
        "subject_count_expected": 139,
        "subject_count_completed": len(rows),
        "failed_subject_count": len(failures),
        "failures": failures,
        "family_evidence_present_subject_count": {
            family: int(family_present.get(family, 0))
            for family in (
                "SOURCE_IMPORT", "CAMERA_OBSERVATION", "SURFACE_GEOMETRY",
                "SURFACE_TOPOLOGY", "SIGNED_DISTANCE_OCCUPANCY", "ORIENTED_NORMALS",
                "APPEARANCE_CAA", "VISIBILITY_OCCLUSION", "RAY_COMPOSITING",
                "CAUSAL_RESPONSE", "RIG_HIERARCHY", "SKIN_WEIGHTS",
                "ANIMATION_KINEMATICS",
            )
        },
        "authority": {
            "training_authorized": False,
            "new_family_authority_minted": False,
            "final_disposition_minted": False,
            "historical_split_reused_as_final": False,
            "purpose": "READ_ONLY_EVIDENCE_INVENTORY",
        },
        "prior_readiness": verified_ref(args.prior_readiness.resolve()),
        "response_receipt": verified_ref(args.response_receipt.resolve()),
        "structural_audit_root": str(structural_root) if structural_root else None,
        "table": verified_ref(table),
        "completed_at_utc": utc(),
    }
    summary["inventory_identity_sha256"] = digest(summary)
    atomic_json(output_root / "TRUTH139_TRUTH_FAMILY_INVENTORY_V1.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failures or len(rows) != 139:
        raise RuntimeError(f"TRUTH139_TRUTH_FAMILY_INVENTORY_INCOMPLETE:{len(rows)}/139 failures={len(failures)}")


if __name__ == "__main__":
    main()
