from __future__ import annotations
from typing import Dict, Any
import numpy as np

G1_KEYS = ('family_id','P_A','N_A','XY_A','V_A','direct_obs_A')

def observation_target_to_g1(target: Dict[str, Any]) -> Dict[str, Any]:
    """Non-destructive projection of the existing observation target onto G1."""
    out = {k: target[k] for k in G1_KEYS if k in target}
    out['schema'] = np.array('RealSaS.SurfaceEvidence.G1.Target.v1')
    return out
