from __future__ import annotations

"""Resume-safe authority wrapper for the repaired Fit8 trainer.

V5 preserves V4 model/training semantics exactly.  Its only change is bootstrap
isolation: superclass constructors build against a temporary empty checkpoint
store, then the final V7 binding is minted before the real checkpoint root is
opened.  Therefore later Colab sessions can resume V7 checkpoints without an
intermediate V5/V6 binding-drift failure.
"""

from dataclasses import asdict
from pathlib import Path
import tempfile

from .cohort_v1 import VNextCheckpointStoreV1, _sha, _vh
from .cohort_v4 import IRISVNextFit8TrainerV4, vnext_fit8_policy_v3, _source_identity_v4
from .contracts_v1 import IRISVNextConfigV1


def _source_identity_v5(root: Path):
    value = dict(_source_identity_v4(root))
    value["models/iris/vnext/cohort_v5.py"] = _sha(root / "models/iris/vnext/cohort_v5.py")
    return value


class IRISVNextFit8TrainerV5(IRISVNextFit8TrainerV4):
    """V4 training semantics with final-binding-first persistent restore."""

    architecture_id = "RealSaS.IRISVNextFit8Trainer.v5"

    def __init__(
        self,
        *,
        data_root,
        manifest,
        checkpoint_root,
        device,
        model_config=IRISVNextConfigV1(),
        policy=None,
        repo_root=None,
        flexicubes_root=None,
    ):
        policy = vnext_fit8_policy_v3() if policy is None else policy
        policy.validate()
        target_root = Path(checkpoint_root)

        # Never let an ancestor binding inspect the final persistent directory.
        with tempfile.TemporaryDirectory(prefix="iris_vnext_v5_bootstrap_") as bootstrap:
            super().__init__(
                data_root=data_root,
                manifest=manifest,
                checkpoint_root=bootstrap,
                device=device,
                model_config=model_config,
                policy=policy,
                repo_root=repo_root,
                flexicubes_root=flexicubes_root,
            )

        self.binding = {
            **self.binding,
            "schema": "RealSaS.IRISVNextFit8Binding.v7",
            "source": _source_identity_v5(self.repo_root),
            "policy": asdict(policy),
            "resume_semantics": "FINAL_BINDING_FIRST__ANCESTOR_STORES_BOOTSTRAP_ISOLATED",
        }
        self.binding_hash = _vh(self.binding)
        self.store = VNextCheckpointStoreV1(target_root, self.binding_hash)

        previous = self.store.load()
        if previous:
            self._restore(previous)
        else:
            self.step = 0
            self.stage_index = 0
            self.stage_step = 0
            self._apply_stage_lrs()
