from __future__ import annotations

"""MIRA V5.5 carrier-native proposal boundary.

The learned decoder may use GSA evidence features internally, but its semantic
output domain is the exact Stage19 mechanical carrier M. This module seals the
proposal contract consumed by Compiler Stage31/32; no GSA skin authority or
post-hoc semantic skin transfer is permitted.
"""

import numpy as np
from compiler.realsas_compiler_core.carrier_skin_v1 import CarrierSkinInfluenceProposal,CarrierSkinProposalIR
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError

MIRA_V55_ARCHITECTURE_ID="RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"


def proposal_from_dense_wm_v55(*,weights,carrier,skeleton,model_provenance:str=MIRA_V55_ARCHITECTURE_ID):
    W=np.asarray(weights,dtype=np.float64)
    joint_ids=tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    vertex_ids=tuple(map(str,carrier.ordered_vertex_ids))
    if W.shape!=(len(vertex_ids),len(joint_ids)) or not np.isfinite(W).all() or float(W.min())<0.0:
        raise QualificationError("MIRA_V55_DENSE_WM_INVALID")
    if float(np.max(np.abs(W.sum(axis=1)-1.0)))>1e-6:
        raise QualificationError("MIRA_V55_DENSE_WM_SIMPLEX_INVALID")
    influences=[]
    for vi,vid in enumerate(vertex_ids):
        for ji,jid in enumerate(joint_ids):
            w=float(W[vi,ji])
            if w>0.0: influences.append(CarrierSkinInfluenceProposal(vid,jid,w))
    provenance=content_sha256({"architecture":MIRA_V55_ARCHITECTURE_ID,"model_provenance":str(model_provenance),"carrier_evidence_hash":carrier.carrier_evidence_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash})
    return CarrierSkinProposalIR(
        influences=tuple(influences),carrier_evidence_hash=carrier.carrier_evidence_hash,carrier_topology_hash=carrier.topology_hash,carrier_geometry_hash=carrier.geometry_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,model_provenance=provenance,
        metadata={
            "architecture":MIRA_V55_ARCHITECTURE_ID,"output_domain":"MECHANICAL_CARRIER_M","direct_simplex_readout":True,
            "gsa_role":"EVIDENCE_SUBSTRATE_ONLY","gsa_skin_field_authority_minted":False,"semantic_skin_transfer_performed":False,
            "source_skin_runtime_authority":False,"mesh_mutation_invalidates_prediction":True,
        },
    )

__all__=["MIRA_V55_ARCHITECTURE_ID","proposal_from_dense_wm_v55"]
