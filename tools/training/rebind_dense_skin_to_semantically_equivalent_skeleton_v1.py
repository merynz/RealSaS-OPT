from __future__ import annotations

"""Losslessly rebind dense skin columns across equivalent compiler skeleton IDs.

This utility exists for compiler-identity migrations only. It is not a skin
solver and cannot alter numeric weights, surface order, joint semantics, or
hierarchy. Equivalence is proven through the model proposal identity carried by
QualifiedJoint.source_proposal_id.

The output NPZ preserves the exact F64 weight matrix and surface-id order and
replaces only canonical_joint_ids. Compiler.qualify_skin must still mint the
new qualified skin authority.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _joint_by_source(skeleton):
    rows = {}
    for joint in skeleton.joints:
        source = str(joint.source_proposal_id)
        if not source:
            raise QualificationError(
                "SKIN_REBIND_SKELETON_SOURCE_PROPOSAL_ID_MISSING"
            )
        if source in rows:
            raise QualificationError(
                f"SKIN_REBIND_SKELETON_SOURCE_PROPOSAL_ID_DUPLICATE:{source}"
            )
        rows[source] = joint
    return rows


def _semantic_tree(skeleton) -> tuple[dict, ...]:
    by_source = _joint_by_source(skeleton)
    source_by_canonical = {
        str(joint.canonical_joint_id): source
        for source, joint in by_source.items()
    }
    rows = []
    for source in sorted(by_source):
        joint = by_source[source]
        parent_source = None
        if joint.parent_canonical_id is not None:
            parent = str(joint.parent_canonical_id)
            if parent not in source_by_canonical:
                raise QualificationError(
                    f"SKIN_REBIND_PARENT_CANONICAL_ID_UNKNOWN:{parent}"
                )
            parent_source = source_by_canonical[parent]
        rows.append(
            {
                "source_proposal_id": source,
                "parent_source_proposal_id": parent_source,
                "position": [float(value) for value in joint.position],
                # Support is set-valued authority. Sort for semantic identity.
                "support_surface_ids": sorted(
                    map(str, joint.support_surface_ids)
                ),
            }
        )
    return tuple(rows)


def _semantic_tree_hash(skeleton) -> str:
    # Must match the canonical Geppetto determinism probe contract exactly:
    # hash the normalized semantic-tree rows and nothing else.
    return content_sha256(list(_semantic_tree(skeleton)))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(array),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write stable NPZ bytes with fixed ZIP metadata and sorted array names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def rebind_dense_skin(
    *,
    input_npz: Path,
    old_skeleton_json: Path,
    new_skeleton_json: Path,
    output_npz: Path,
    receipt_json: Path,
    expected_old_skeleton_lineage: str | None = None,
    expected_new_skeleton_lineage: str | None = None,
) -> dict:
    old_skeleton = qualified_skeleton_from_dict(_load_json(old_skeleton_json))
    new_skeleton = qualified_skeleton_from_dict(_load_json(new_skeleton_json))

    if (
        expected_old_skeleton_lineage
        and old_skeleton.skeleton_lineage_hash
        != expected_old_skeleton_lineage
    ):
        raise QualificationError(
            "SKIN_REBIND_OLD_SKELETON_LINEAGE_DRIFT"
        )
    if (
        expected_new_skeleton_lineage
        and new_skeleton.skeleton_lineage_hash
        != expected_new_skeleton_lineage
    ):
        raise QualificationError(
            "SKIN_REBIND_NEW_SKELETON_LINEAGE_DRIFT"
        )

    old_tree = _semantic_tree(old_skeleton)
    new_tree = _semantic_tree(new_skeleton)
    if old_tree != new_tree:
        raise QualificationError(
            "SKIN_REBIND_SKELETON_SEMANTICS_NOT_EQUIVALENT"
        )
    semantic_hash = _semantic_tree_hash(old_skeleton)
    if semantic_hash != _semantic_tree_hash(new_skeleton):
        raise QualificationError(
            "SKIN_REBIND_SEMANTIC_TREE_HASH_DRIFT"
        )

    old_by_source = _joint_by_source(old_skeleton)
    new_by_source = _joint_by_source(new_skeleton)
    if set(old_by_source) != set(new_by_source):
        raise QualificationError(
            "SKIN_REBIND_SOURCE_PROPOSAL_ID_SET_DRIFT"
        )

    old_source_by_joint = {
        str(joint.canonical_joint_id): source
        for source, joint in old_by_source.items()
    }
    new_joint_by_source = {
        source: str(joint.canonical_joint_id)
        for source, joint in new_by_source.items()
    }
    if len(old_source_by_joint) != len(old_by_source):
        raise QualificationError(
            "SKIN_REBIND_OLD_CANONICAL_ID_DUPLICATE"
        )
    if len(set(new_joint_by_source.values())) != len(new_joint_by_source):
        raise QualificationError(
            "SKIN_REBIND_NEW_CANONICAL_ID_DUPLICATE"
        )

    with np.load(input_npz, allow_pickle=False) as data:
        if set(data.files) != {
            "weights",
            "surface_ids",
            "canonical_joint_ids",
        }:
            raise QualificationError(
                "SKIN_REBIND_INPUT_NPZ_CONTRACT_DRIFT"
            )
        weights = np.asarray(data["weights"], dtype=np.float64).copy()
        surface_ids = np.asarray(data["surface_ids"]).copy()
        old_joint_ids = np.asarray(data["canonical_joint_ids"]).copy()

    if weights.ndim != 2:
        raise QualificationError("SKIN_REBIND_WEIGHT_RANK_INVALID")
    if weights.shape != (len(surface_ids), len(old_joint_ids)):
        raise QualificationError("SKIN_REBIND_WEIGHT_SHAPE_INVALID")
    if not np.isfinite(weights).all():
        raise QualificationError("SKIN_REBIND_WEIGHT_NONFINITE")

    old_joint_id_strings = tuple(map(str, old_joint_ids.tolist()))
    if set(old_joint_id_strings) != set(old_source_by_joint):
        raise QualificationError(
            "SKIN_REBIND_INPUT_JOINT_ID_SET_DRIFT"
        )
    if len(set(old_joint_id_strings)) != len(old_joint_id_strings):
        raise QualificationError(
            "SKIN_REBIND_INPUT_JOINT_ID_DUPLICATE"
        )

    new_joint_id_strings = tuple(
        new_joint_by_source[old_source_by_joint[old_id]]
        for old_id in old_joint_id_strings
    )
    if len(set(new_joint_id_strings)) != len(new_joint_id_strings):
        raise QualificationError(
            "SKIN_REBIND_OUTPUT_JOINT_ID_DUPLICATE"
        )

    # Preserve column positions exactly: column j keeps the identical learned
    # field, only the compiler-owned identity string changes.
    max_chars = max(len(value) for value in new_joint_id_strings)
    new_joint_ids = np.asarray(
        new_joint_id_strings,
        dtype=f"<U{max_chars}",
    )

    weight_bytes_before = weights.tobytes(order="C")
    surface_bytes_before = surface_ids.tobytes(order="C")
    arrays = {
        "weights": weights,
        "surface_ids": surface_ids,
        "canonical_joint_ids": new_joint_ids,
    }
    _write_deterministic_npz(output_npz, arrays)

    with np.load(output_npz, allow_pickle=False) as check:
        rebound_weights = np.asarray(check["weights"], dtype=np.float64)
        rebound_surface_ids = np.asarray(check["surface_ids"])
        rebound_joint_ids = tuple(
            map(str, check["canonical_joint_ids"].tolist())
        )
    if not np.array_equal(weights, rebound_weights):
        raise QualificationError(
            "SKIN_REBIND_NUMERIC_WEIGHT_MUTATION"
        )
    if not np.array_equal(surface_ids, rebound_surface_ids):
        raise QualificationError(
            "SKIN_REBIND_SURFACE_ORDER_MUTATION"
        )
    if rebound_joint_ids != new_joint_id_strings:
        raise QualificationError(
            "SKIN_REBIND_JOINT_ID_SERIALIZATION_DRIFT"
        )

    mapping_rows = [
        {
            "column_index": int(index),
            "source_proposal_id": old_source_by_joint[old_id],
            "old_canonical_joint_id": old_id,
            "new_canonical_joint_id": new_joint_id_strings[index],
        }
        for index, old_id in enumerate(old_joint_id_strings)
    ]
    receipt = {
        "schema": "RealSaS.SemanticSkeletonSkinIdRebindReceipt.v1",
        "status": "PASS",
        "old_skeleton_lineage_hash": old_skeleton.skeleton_lineage_hash,
        "new_skeleton_lineage_hash": new_skeleton.skeleton_lineage_hash,
        "semantic_tree_hash": semantic_hash,
        "joint_count": len(mapping_rows),
        "surface_row_count": len(surface_ids),
        "weight_shape": list(map(int, weights.shape)),
        "input_npz_sha256": _sha256_file(input_npz),
        "output_npz_sha256": _sha256_file(output_npz),
        "weights_f64_raw_sha256_before": _sha256_bytes(
            weight_bytes_before
        ),
        "weights_f64_raw_sha256_after": _sha256_bytes(
            rebound_weights.tobytes(order="C")
        ),
        "surface_ids_raw_sha256_before": _sha256_bytes(
            surface_bytes_before
        ),
        "surface_ids_raw_sha256_after": _sha256_bytes(
            rebound_surface_ids.tobytes(order="C")
        ),
        "numeric_weight_array_unchanged": True,
        "surface_order_unchanged": True,
        "column_order_unchanged": True,
        "only_canonical_joint_identity_changed": True,
        "mapping_hash": content_sha256(mapping_rows),
        "mapping": mapping_rows,
        "compiler_requalification_required": True,
        "teacher_input_used": False,
        "model_retraining_used": False,
        "product_authority_minted": False,
    }
    if (
        receipt["weights_f64_raw_sha256_before"]
        != receipt["weights_f64_raw_sha256_after"]
    ):
        raise QualificationError(
            "SKIN_REBIND_WEIGHT_RAW_HASH_DRIFT"
        )
    if (
        receipt["surface_ids_raw_sha256_before"]
        != receipt["surface_ids_raw_sha256_after"]
    ):
        raise QualificationError(
            "SKIN_REBIND_SURFACE_RAW_HASH_DRIFT"
        )
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--old-skeleton-json", required=True)
    parser.add_argument("--new-skeleton-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--receipt-json", required=True)
    parser.add_argument("--expected-old-skeleton-lineage")
    parser.add_argument("--expected-new-skeleton-lineage")
    args = parser.parse_args()

    receipt = rebind_dense_skin(
        input_npz=Path(args.input_npz),
        old_skeleton_json=Path(args.old_skeleton_json),
        new_skeleton_json=Path(args.new_skeleton_json),
        output_npz=Path(args.output_npz),
        receipt_json=Path(args.receipt_json),
        expected_old_skeleton_lineage=args.expected_old_skeleton_lineage,
        expected_new_skeleton_lineage=args.expected_new_skeleton_lineage,
    )
    print(
        "DENSE_SKIN_SEMANTIC_ID_REBIND_PASS",
        json.dumps(
            {
                "semantic_tree_hash": receipt["semantic_tree_hash"],
                "joint_count": receipt["joint_count"],
                "surface_row_count": receipt["surface_row_count"],
                "weights_f64_raw_sha256": receipt[
                    "weights_f64_raw_sha256_after"
                ],
                "mapping_hash": receipt["mapping_hash"],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
