from __future__ import annotations

"""Official deterministic FP32 execution policy for focused IRIS V9 training.

`cohort_v9.py` owns the reviewed corpus, bootstrap, split, metrics, and frozen
parameter contracts.  This module is the only V9 optimizer execution authority.
It deliberately replaces the provisional mixed-precision execution policy with
FP32, TF32-off, deterministic CUDA, and non-fused AdamW before any optimizer
step can occur.
"""

from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch

from .cohort_v1 import VNextCheckpointStoreV1
from .cohort_v9 import (
    IRISVNextFocusedRelationTrainerV9,
    V9_SPLIT_SEED,
    _value_hash,
)


V9_NUMERIC_POLICY = "FP32_TF32_OFF_DETERMINISTIC_CUDA_NONFUSED_ADAMW_V1"


def configure_v9_determinism() -> None:
    random.seed(V9_SPLIT_SEED)
    np.random.seed(V9_SPLIT_SEED)
    torch.manual_seed(V9_SPLIT_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("VNEXT_V9_CUDA_REQUIRED")
    torch.cuda.manual_seed_all(V9_SPLIT_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _trainable_relation_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    expected = {
        id(p)
        for module in (model.relations.view, model.relations.view_pair)
        for p in module.parameters()
    }
    if {id(p) for p in params} != expected:
        raise RuntimeError("VNEXT_V9_FP32_TRAINABLE_PARAMETER_SCOPE_DRIFT")
    return params


class IRISVNextFocusedRelationTrainerV9FP32(IRISVNextFocusedRelationTrainerV9):
    """Fail-closed official V9 trainer.

    The base class is used only to reuse its already-reviewed authority checks.
    Its temporary step-zero store and mixed-precision cache are discarded before
    the official binding/store are minted.  No optimizer step is executed by the
    base constructor.
    """

    numeric_policy = V9_NUMERIC_POLICY

    def __init__(self, *args: Any, checkpoint_root: Path, **kwargs: Any) -> None:
        final_root = Path(checkpoint_root).resolve()
        bootstrap_root = final_root.parent / (final_root.name + ".provisional_bootstrap")
        if bootstrap_root.exists():
            shutil.rmtree(bootstrap_root)
        try:
            super().__init__(*args, checkpoint_root=bootstrap_root, **kwargs)
            if self.step != 0:
                raise RuntimeError("VNEXT_V9_FP32_PROVISIONAL_BOOTSTRAP_STEP_DRIFT")
        finally:
            if bootstrap_root.exists():
                shutil.rmtree(bootstrap_root)

        configure_v9_determinism()

        # Rebuild the optimizer before any official checkpoint is minted.  Fused
        # kernels and foreach variants are intentionally disabled for replay
        # comparability across the local witness GPU and the A100 training lane.
        trainable = _trainable_relation_parameters(self.model)
        self.optimizer = torch.optim.AdamW(
            [{
                "params": trainable,
                "lr": self.policy.relation_learning_rate,
                "name": "relation_view_branch",
            }],
            weight_decay=self.policy.weight_decay,
            betas=self.policy.betas,
            fused=False,
            foreach=False,
        )

        # The provisional BF16 scene cache is not authoritative.  Recompute H
        # from the frozen V8 carrier in strict FP32 and detach it permanently.
        self.model.eval()
        with torch.no_grad():
            self.scene_planes = self.model.encode_scene(
                self.corpus.source["image_tokens"].float(),
                self.corpus.source["token_xy"].float(),
                self.corpus.source["camera_features"].float(),
            ).detach()
        if self.scene_planes.dtype != torch.float32:
            raise RuntimeError("VNEXT_V9_FP32_SCENE_CACHE_DTYPE_DRIFT")

        self.binding = dict(self.binding)
        self.binding["numeric_execution"] = {
            "policy": V9_NUMERIC_POLICY,
            "model_dtype": "float32",
            "scene_cache_dtype": "float32",
            "autocast": False,
            "tf32": False,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "optimizer": "AdamW",
            "optimizer_fused": False,
            "optimizer_foreach": False,
        }
        self.binding_hash = _value_hash(self.binding)
        self.store = VNextCheckpointStoreV1(final_root, self.binding_hash)
        self.step = 0
        previous = self.store.load()
        if previous is not None:
            self._restore(previous)
        else:
            self.save()
        self._assert_numeric_policy()

    def _assert_numeric_policy(self) -> None:
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("VNEXT_V9_FP32_TF32_REENABLED")
        if torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
            raise RuntimeError("VNEXT_V9_FP32_CUDNN_POLICY_DRIFT")
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("VNEXT_V9_FP32_DETERMINISM_DISABLED")
        if self.scene_planes.dtype != torch.float32:
            raise RuntimeError("VNEXT_V9_FP32_SCENE_CACHE_DTYPE_DRIFT")
        group = self.optimizer.param_groups[0]
        if group.get("fused") not in (False, None) or group.get("foreach") not in (False, None):
            raise RuntimeError("VNEXT_V9_FP32_OPTIMIZER_KERNEL_POLICY_DRIFT")

    def _predict(
        self,
        point_a: torch.Tensor,
        point_b: torch.Tensor,
        view_index: int,
    ) -> dict[str, torch.Tensor]:
        self._assert_numeric_policy()
        with torch.no_grad():
            feature_a = self.model.scene_core.query(
                self.scene_planes, point_a.float()
            )["H"].detach()
            feature_b = self.model.scene_core.query(
                self.scene_planes, point_b.float()
            )["H"].detach()
        if feature_a.dtype != torch.float32 or feature_b.dtype != torch.float32:
            raise RuntimeError("VNEXT_V9_FP32_FEATURE_DTYPE_DRIFT")
        cameras = self.corpus.source["camera_features"][:, int(view_index), None, :].expand(
            -1, feature_a.shape[1], -1
        ).float()
        return self.model.relations.relation_probabilities(feature_a, feature_b, cameras)

    def one_step(self) -> dict[str, Any]:
        self._assert_numeric_policy()
        result = super().one_step()
        self._assert_numeric_policy()
        result["numeric_policy"] = V9_NUMERIC_POLICY
        return result

    def evaluate_heldout(self) -> dict[str, Any]:
        self._assert_numeric_policy()
        result = super().evaluate_heldout()
        result["numeric_policy"] = V9_NUMERIC_POLICY
        return result
