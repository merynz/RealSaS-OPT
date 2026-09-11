from __future__ import annotations
"""Pre-optimizer FIX1: bind V4-native SkinProposal metadata into the sealed V4 runner."""
import experiments.arachne_a1_v4_fit1.run_arachne_a1_v4_fit1 as core
from models.arachne.v4.proposal_v4 import make_skin_proposal_v4

core.make_skin_proposal = make_skin_proposal_v4

if __name__ == "__main__":
    core.main()
