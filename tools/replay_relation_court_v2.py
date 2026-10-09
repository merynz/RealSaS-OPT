"""Strict exact-source-cut ablation for the fast V7 relation court.

This does not alter product/research core contracts. It asks a narrow causal
question: if broad 12 px contact candidates are restricted to exact source-chart
cuts that still pass the canonical/skin qualification in visual_contact_v1, does
contact preservation become coherent?
"""

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visual_contact_v1 import (
    QualifiedVisualContactSet,
    qualify_visual_contacts as _broad_qualify,
)
from tools import replay_relation_court_v1 as court


def _strict_qualify(**kwargs):
    value = _broad_qualify(**kwargs)
    mask = np.asarray(value.rest_distances_px) <= 1.0e-12
    pairs = np.asarray(value.pairs)[mask].copy()
    codes = np.asarray(value.relation_codes)[mask].copy()
    domains = np.asarray(value.domain_pairs)[mask].copy()
    distances = np.asarray(value.rest_distances_px)[mask].copy()
    scores = np.asarray(value.evidence_scores)[mask].copy()
    digest = content_sha256({
        "schema": "RealSaS.StrictSourceCutContactAblation.v1",
        "parent_contact_hash": value.contact_hash,
        "pairs": pairs.astype(int).tolist(),
        "relation_codes": codes.astype(int).tolist(),
        "domain_pairs": domains.astype(int).tolist(),
    })
    return QualifiedVisualContactSet(
        view_index=value.view_index,
        pairs=pairs,
        relation_codes=codes,
        domain_pairs=domains,
        rest_distances_px=distances,
        evidence_scores=scores,
        contact_hash=digest,
    )


court.qualify_visual_contacts = _strict_qualify

if __name__ == "__main__":
    court.main()
