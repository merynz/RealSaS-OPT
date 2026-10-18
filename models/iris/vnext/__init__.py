"""IRIS vNext: single-scene evidence, no named semantic taxonomy or visual mesh split."""
from .contracts_v1 import IRISVNextConfigV1
from .model_v1 import IRISVNextV1
from .teacher_truth_v1 import MechanicalDemandTruthPolicyV1, MechanicalDemandTruthV1, build_mechanical_demand_truth_v1
__all__=["IRISVNextConfigV1","IRISVNextV1","MechanicalDemandTruthPolicyV1","MechanicalDemandTruthV1","build_mechanical_demand_truth_v1"]
