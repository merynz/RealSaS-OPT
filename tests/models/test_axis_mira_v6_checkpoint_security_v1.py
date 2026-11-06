from __future__ import annotations

"""V6 producer-runtime package security, not trained model admission."""

from dataclasses import replace
import os
import pickle

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6, AXISRigProgramModelV6,
)
from models.axis.runtime_checkpoint_v2 import (
    build_axis_runtime_checkpoint_v2, load_axis_runtime_model_v2,
)
from models.axis.scene_conditioning_v6 import AXIS_V6_SURFACE_SCENE_DIM
from models.mira.runtime_checkpoint_v2 import (
    build_mira_runtime_checkpoint_v2, load_mira_runtime_model_v2,
)
from models.mira.skin_fsq_codec_v6 import (
    MIRASkinFSQCodecConfig, MIRASkinFSQCodecV6, MIRA_V6_CODEC_ID,
)


class _UnsafeOSAction:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        # The marker would only be written if checkpoint loading unpickled an
        # untrusted executable global. torch.save must only record the reducer.
        return (os.system, ("touch " + str(self.marker),))


@pytest.mark.parametrize("family", ("axis", "mira"))
def test_v6_runtime_checkpoint_loader_rejects_executable_pickle_payload(family, tmp_path):
    marker = tmp_path / ("executed_" + family)
    package = tmp_path / (family + ".pt")
    torch.save({"unsafe_payload": _UnsafeOSAction(marker)}, package)
    load = load_axis_runtime_model_v2 if family == "axis" else load_mira_runtime_model_v2
    with pytest.raises(pickle.UnpicklingError):
        load(package, device=torch.device("cpu"))
    assert not marker.exists(), "unsafe pickle executed during runtime loading"


def test_axis_runtime_rejects_resealed_wrong_codec_binding(tmp_path):
    cfg = AXISRigProgramConfigV6(
        scene_dim=AXIS_V6_SURFACE_SCENE_DIM,
        hidden_dim=32, layers=1, heads=4, ff_mult=2,
        max_sequence_length=64, max_pointer_entities=16, skin_vocab_size=27,
    )
    model = AXISRigProgramModelV6(cfg)
    package = build_axis_runtime_checkpoint_v2(
        model,
        skin_codec_architecture_id=MIRA_V6_CODEC_ID,
        skin_codec_checkpoint_sha256="a" * 64,
        skin_codebook_size=27,
        tokens_per_joint=2,
    )
    package["skin_codec_binding"]["codebook_size"] = 28
    target = tmp_path / "invalid_axis.pt"
    torch.save(package, target)
    with pytest.raises(RuntimeError, match="AXIS_V6_RUNTIME_SKIN_VOCAB_DRIFT"):
        load_axis_runtime_model_v2(target, device=torch.device("cpu"))


def test_mira_runtime_rejects_tampered_codec_contract(tmp_path):
    cfg = MIRASkinFSQCodecConfig(
        vertex_feature_dim=6, joint_feature_dim=5,
        hidden_dim=32, tokens_per_joint=2, levels=(3, 3, 3),
    )
    model = MIRASkinFSQCodecV6(cfg)
    package = build_mira_runtime_checkpoint_v2(model)
    package["codec_contract"]["compiler_code_repair_allowed"] = True
    target = tmp_path / "invalid_mira.pt"
    torch.save(package, target)
    with pytest.raises(RuntimeError, match="MIRA_V6_RUNTIME_COMPILER_CODE_AUTHORITY_DRIFT"):
        load_mira_runtime_model_v2(target, device=torch.device("cpu"))
