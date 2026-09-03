from __future__ import annotations

import json

import numpy as np
import torch

from experiments.geppetto_arachne_r6_20260901.oracle_substrate_synthetic_v1 import build_u0_surface, build_u1_surface, substrate_telemetry
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import WITNESSES
from experiments.geppetto_arachne_r6_20260901.test_r6_oracle_substrate_geppetto_one_family_v1 import _fit_arm, REQUIRED_STABLE


def test_r6_geppetto_heterogeneous_u1_observation_oracle_panel() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    panel = []
    for witness in WITNESSES:
        torch.manual_seed(witness.seed)
        np.random.seed(witness.seed)
        u0, samples, cameras, visibility = build_u0_surface(witness)
        u1, evidence = build_u1_surface(witness, samples, cameras, visibility)
        telemetry = substrate_telemetry(u0, u1, visibility)

        assert len(u1.surface_nodes) > 0
        assert evidence.metadata["hidden_surface_completion"] is False
        assert evidence.metadata["source_mesh_consumer_input"] is False
        assert all("reference_full_surface" not in n.metadata for n in u1.surface_nodes)
        assert all(
            n.derived_normal is None
            or n.metadata.get("derived_normal_operator_id") == "RealSaS.DTB-ND1.RobustLocalPlane.v1"
            for n in u1.surface_nodes
        )

        result = _fit_arm("U1_OBSERVATION_ORACLE_SUBSTRATE", u1, witness, witness.seed)
        record = {"witness": witness.name, "telemetry": telemetry, "result": result}
        panel.append(record)
        print("R6_G_SYN_HET_U1_RESULT=" + json.dumps(record, sort_keys=True))
        assert result["pass_step"] is not None, {"panel": panel}
        assert result["stable_passes"] >= REQUIRED_STABLE, {"panel": panel}

    print("R6_G_SYN_HET_U1_PANEL=" + json.dumps(panel, sort_keys=True))
    assert len(panel) == 3
