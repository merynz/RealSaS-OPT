from __future__ import annotations

"""Materialize a SkinProposalIR from dense canonical weights.

This is an execution-bridge utility, not a qualifier. It never mints skin
authority: Stage32 / Compiler.qualify_skin remains the sole qualification
owner. The NPZ row/column ordering is preserved exactly so an external fit can
be replayed through the canonical Compiler without teacher or category inputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    SkinInfluenceProposal,
    SkinProposalIR,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(
    *,
    weights_npz: Path,
    surface_json: Path,
    skeleton_json: Path,
    output_json: Path,
    model_provenance: str,
    metadata: dict,
) -> SkinProposalIR:
    surface = rigging_surface_from_dict(_load_json(surface_json))
    skeleton = qualified_skeleton_from_dict(_load_json(skeleton_json))

    with np.load(weights_npz, allow_pickle=False) as data:
        required = {"weights", "surface_ids", "canonical_joint_ids"}
        if not required.issubset(data.files):
            raise QualificationError("DENSE_SKIN_PROPOSAL_NPZ_ARRAYS_MISSING")
        weights = np.asarray(data["weights"], dtype=np.float64)
        surface_ids = tuple(map(str, data["surface_ids"].tolist()))
        joint_ids = tuple(map(str, data["canonical_joint_ids"].tolist()))

    if weights.shape != (len(surface_ids), len(joint_ids)):
        raise QualificationError("DENSE_SKIN_PROPOSAL_WEIGHT_SHAPE_INVALID")
    if not np.isfinite(weights).all():
        raise QualificationError("DENSE_SKIN_PROPOSAL_NONFINITE")
    if len(set(surface_ids)) != len(surface_ids) or len(set(joint_ids)) != len(joint_ids):
        raise QualificationError("DENSE_SKIN_PROPOSAL_ID_DUPLICATE")

    canonical_surface_ids = {str(row.surface_id) for row in surface.surface_nodes}
    canonical_joint_ids = {str(row.canonical_joint_id) for row in skeleton.joints}
    if set(surface_ids) != canonical_surface_ids:
        raise QualificationError("DENSE_SKIN_PROPOSAL_SURFACE_ID_SET_DRIFT")
    if set(joint_ids) != canonical_joint_ids:
        raise QualificationError("DENSE_SKIN_PROPOSAL_JOINT_ID_SET_DRIFT")

    influences = tuple(
        SkinInfluenceProposal(surface_id, joint_id, float(weights[si, ji]))
        for si, surface_id in enumerate(surface_ids)
        for ji, joint_id in enumerate(joint_ids)
    )
    proposal = SkinProposalIR(
        influences,
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        model_provenance=str(model_provenance),
        metadata=dict(metadata),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(proposal.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return proposal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-npz", required=True)
    ap.add_argument("--surface-json", required=True)
    ap.add_argument("--skeleton-json", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--model-provenance", required=True)
    ap.add_argument("--metadata-json", required=True)
    args = ap.parse_args()
    metadata = json.loads(args.metadata_json)
    proposal = materialize(
        weights_npz=Path(args.weights_npz),
        surface_json=Path(args.surface_json),
        skeleton_json=Path(args.skeleton_json),
        output_json=Path(args.output_json),
        model_provenance=args.model_provenance,
        metadata=metadata,
    )
    print(
        "DENSE_SKIN_PROPOSAL_MATERIALIZED",
        json.dumps(
            {
                "surface_binding_hash": proposal.surface_binding_hash,
                "skeleton_binding_hash": proposal.skeleton_binding_hash,
                "influence_count": len(proposal.influences),
                "model_provenance": proposal.model_provenance,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
