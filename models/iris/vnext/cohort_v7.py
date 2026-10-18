from __future__ import annotations

"""IRIS vNext Fit8 trainer v7: resume V6 T1-end with canonical witness projection.

V7 changes no curriculum weights, stage lengths, optimizer policy, model tensor
shapes, or checkpoint parameter names.  It updates only the structure-coordinate
contract through IRISVNextStructureHeadV2 and provides an explicit migration from
the protected V6 step-8800 T1-end checkpoint.
"""

from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from .cohort_v1 import VNextCheckpointStoreV1, _sha, _vh
from .cohort_v6 import IRISVNextFit8TrainerV6, _source_identity_v6, vnext_fit8_policy_v4
from .contracts_v1 import IRISVNextConfigV1


V6_T1_END_STEP = 8800
V6_T1_END_SHA256 = "7610c349902ba36e2a05eccad3740d220e978cde865ad91caa124f6858ad5f94"
V6_BINDING_SHA256 = "0a17710c6b5cda7bcd635d99a38f9556a4243d2a729baf5738898a77033cece5"


def _source_identity_v7(root: Path):
    value = dict(_source_identity_v6(root))
    value["models/iris/vnext/structure_v2.py"] = _sha(root / "models/iris/vnext/structure_v2.py")
    value["models/iris/vnext/cohort_v7.py"] = _sha(root / "models/iris/vnext/cohort_v7.py")
    return value


class IRISVNextFit8TrainerV7(IRISVNextFit8TrainerV6):
    """V6 cumulative curriculum with repaired TESSA witness-domain semantics."""

    architecture_id = "RealSaS.IRISVNextFit8Trainer.v7"

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
        policy = vnext_fit8_policy_v4() if policy is None else policy
        policy.validate()
        target_root = Path(checkpoint_root)

        # Ancestor bindings must never inspect the final V9 directory.
        with tempfile.TemporaryDirectory(prefix="iris_vnext_v7_bootstrap_") as bootstrap:
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
            "schema": "RealSaS.IRISVNextFit8Binding.v9",
            "source": _source_identity_v7(self.repo_root),
            "policy": asdict(policy),
            "model_architecture_id": self.model.architecture_id,
            "structure_architecture_id": self.model.structure.architecture_id,
            "topology_witness_coordinate_semantics": {
                "extractor_coordinate_domain": "IRIS_BASE_MINUS1_TO1_PLUS_BOUNDED_EXTRACTION_DELTA",
                "authorized_iris_overshoot": float(model_config.maximum_extraction_delta),
                "tessa_memory_domain": "MINUS0.5_TO0.5",
                "repair": "VERIFY_AUTHORIZED_OVERSHOOT_THEN_PROJECT_TO_TESSA_CANONICAL",
                "extraction_geometry_or_loss_modified": False,
            },
            "migration_source": {
                "binding_sha256": V6_BINDING_SHA256,
                "checkpoint_step": V6_T1_END_STEP,
                "checkpoint_sha256": V6_T1_END_SHA256,
            },
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

    def migrate_v6_t1_end(self, payload: dict, *, source_checkpoint_sha256: str):
        if self.step != 0 or self.stage_index != 0 or self.stage_step != 0:
            raise RuntimeError("VNEXT_V7_MIGRATION_TARGET_NOT_FRESH")
        if source_checkpoint_sha256 != V6_T1_END_SHA256:
            raise RuntimeError("VNEXT_V7_MIGRATION_SOURCE_SHA_DRIFT")
        if int(payload.get("step", -1)) != V6_T1_END_STEP:
            raise RuntimeError("VNEXT_V7_MIGRATION_SOURCE_STEP_DRIFT")
        if int(payload.get("stage_index", -1)) != 5 or int(payload.get("stage_step", -1)) != 2400:
            raise RuntimeError("VNEXT_V7_MIGRATION_SOURCE_STAGE_DRIFT")
        if payload.get("binding_sha256") != V6_BINDING_SHA256:
            raise RuntimeError("VNEXT_V7_MIGRATION_SOURCE_BINDING_DRIFT")
        if payload.get("schema") != "RealSaS.IRISVNextFit8Checkpoint.v4":
            raise RuntimeError("VNEXT_V7_MIGRATION_SOURCE_SCHEMA_DRIFT")

        # Structure v2 has the same parameter names/shapes as v1; only its
        # forward coordinate contract changed.  Therefore state transfer is
        # exact at the tensor/optimizer/RNG level.
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

        self.step = V6_T1_END_STEP
        self.stage_index = 5
        self.stage_step = 2400
        self._apply_stage_lrs()
        self.optimizer.zero_grad(set_to_none=True)

        receipt = {
            "schema": "RealSaS.IRISVNextFit8V6ToV7MigrationReceipt.v1",
            "status": "PASS",
            "migration": "V6_T1_END_STEP8800__COORDINATE_CONTRACT_REPAIR_ONLY",
            "source_checkpoint_step": V6_T1_END_STEP,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_binding_sha256": payload.get("binding_sha256"),
            "target_binding_sha256": self.binding_hash,
            "target_binding_schema": self.binding["schema"],
            "model_parameter_state_exactly_loaded": True,
            "optimizer_state_exactly_loaded": True,
            "rng_state_exactly_loaded": True,
            "curriculum_changed": False,
            "extraction_geometry_or_loss_changed": False,
            "repair": self.binding["topology_witness_coordinate_semantics"],
        }
        pointer = self.save()
        receipt["migrated_checkpoint_pointer"] = pointer
        return receipt
