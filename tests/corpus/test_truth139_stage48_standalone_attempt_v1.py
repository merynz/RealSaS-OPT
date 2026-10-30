from __future__ import annotations

import json
from pathlib import Path

from tools.corpus.run_truth139_stage48_witness_subject_v1 import (
    STAGE_ID,
    canonical_policy,
    make_attempt_manifest,
)


def test_stage48_canonical_policy_is_preregistered_32_16_6_256() -> None:
    _request, cfg = canonical_policy()
    assert cfg["mode"] == "OBSERVATION_REST_PRESENTATION_RESPONSE_TRUTH_V1"
    assert cfg["measured_rays_per_subject"] == 32
    assert cfg["minimum_pass_rays_per_subject"] == 16
    assert cfg["maximum_hits_per_ray"] == 6
    assert cfg["cycles_samples"] == 256


def test_stage48_standalone_attempt_layout_matches_adapter_parent_rule(tmp_path: Path) -> None:
    response_chain = tmp_path / "response_chain.json"
    response_chain.write_text("{}\n")
    attempt_root = tmp_path / "attempt"
    artifact_root = attempt_root / "artifacts" / STAGE_ID
    artifact_root.mkdir(parents=True)
    packed = artifact_root / "PACKED_A.json"

    _request, cfg = canonical_policy()
    make_attempt_manifest(attempt_root, "truth139_test", response_chain, cfg)

    # blender_presentation_ray_measure_v1.py resolves output.parents[2] as run_root.
    assert packed.parents[2] == attempt_root
    manifest_path = packed.parents[2] / "run_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["subject_id"] == "truth139_test"
    assert manifest["stage_id"] == STAGE_ID
    assert manifest["presentation_truth"] == cfg
    assert manifest["training_allowed"] is False
    assert manifest["product_pass_authority"] is False
