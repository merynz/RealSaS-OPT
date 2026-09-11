from __future__ import annotations
import numpy as np
from compiler.realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR


def make_skin_proposal_v4(pred, conditioning, model_provenance: str):
    p=np.asarray(pred,np.float64); sids=conditioning.surface_ids[0]; jids=conditioning.joint_ids[0]
    if p.shape!=(len(sids),len(jids)): raise ValueError("skin proposal shape drift")
    inf=tuple(SkinInfluenceProposal(str(sid),str(jid),float(p[si,ji])) for si,sid in enumerate(sids) for ji,jid in enumerate(jids))
    return SkinProposalIR(inf,conditioning.source_surface_hashes[0],conditioning.source_skeleton_hashes[0],model_provenance=model_provenance,metadata={"candidate_architecture":"RealSaS.Arachne.A1.RichQualifiedBidirectional.v4","field_tokens":4,"dense_proposal":True,"compiler_owns_qualification":True,"teacher_input_used":False,"ordered_latent_alignment_used":False,"bidirectional_surface_joint_field_reasoning":True})


__all__=["make_skin_proposal_v4"]
